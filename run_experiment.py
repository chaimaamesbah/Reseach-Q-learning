"""
run_experiment.py — Lance et agrège des expériences Q-learning
sur plusieurs instances, méthodes et seeds.

Usage:
    python run_experiment.py [options]

Navigation:
    python run_experiment.py --domain navigation --instances 1 \
        --methods q-none q-cp-etr --episodes 500 --seeds 5

Options:
    --episodes N        Training episodes per run          (default: 2000)
    --seeds N           Number of seeds to run
    --seed-start N      First seed to run                 (default: 1)
    --no-plots          Skip generating plots and summary table
    --instances ID...   Instance IDs to benchmark
    --methods M...      Methods to benchmark              (default: all six)
    --output-dir DIR    Directory for all outputs         (default: ./experiment_results)
    --workers N         Parallel workers (default: 1). Java methods utilisent un port dédié par worker.
    --base-port P       Premier port TCP pour les serveurs Java (default: 12345)
    --force             Re-run even if cache entry exists
    --plots-only        Skip running, only regenerate plots/table from cached data

Methods:
    q-none      Q-learning, no reward shaping
    q-classic   Q-learning, classic potential shaping
    q-cp-ms     Q-learning, CP-MS shaping  (Java mode: MS)
    q-cp-etr    Q-learning, CP-ETR shaping (Java mode: ETR)
    cp-greedy   CP-MS greedy agent (no Q-learning)

Cache:
    After each successful run, the script writes a compact JSON to a cache path
    that includes the domain, reward mode, shaping parameters and Navigation
    horizons, so incompatible variants never share data. It contains the
    evaluation logs and the parameters used for the run.
    The results/ folder written by main.py is never read for skip logic.
"""

import argparse
import csv
import hashlib
import json
import queue
import shutil
import time
import subprocess
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import config

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DEFAULT_INSTANCES = ["4s", "4medium", "4hard", "8s", "8medium", "8hard"]
DEFAULT_METHODS = ["q-none", "q-classic", "q-cp-etr", "cp-greedy", "optimal"]
DEFAULT_SEEDS = 20
DEFAULT_EPISODES = 5_000
DEFAULT_OUTPUT = ROOT / "experiment_results"

# Maps our method name -> (--agent, --shaping) arguments for main.py
METHOD_ARGS: dict[str, tuple[str, str | None]] = {
    "q-none": ("q", "none"),
    "q-classic": ("q", "classic"),
    "q-cp-ms": ("q", "cp-ms"),
    "q-cp-etr": ("q", "cp-etr"),
    "cp-greedy": ("cp_greedy", None),
    "optimal": ("optimal", None),
}

# Methods that include a Q-learning training phase
Q_METHODS = {"q-none", "q-classic", "q-cp-ms", "q-cp-etr"}

# Methods that require a Java CP server (must run sequentially for now)
JAVA_METHODS = {"q-cp-ms", "q-cp-etr", "cp-greedy"}

# Methods displayed as horizontal lines (no training curve)
HLINE_METHODS = {"cp-greedy", "optimal"}

# Human-readable labels for plots and table
METHOD_LABELS = {
    "q-none": "Q-learning",
    "q-classic": "Q-learning + Manhattan",
    "q-cp-ms": "Q-CP-MS",
    "q-cp-etr": "Q-learning + CP",
    "cp-greedy": "CP-MS Greedy",
    "optimal": "Optimal",
}

# Consistent colour palette
METHOD_COLORS = {
    "q-none": "#7B1FA2",
    "q-classic": "#2196F3",
    "q-cp-ms": "#2ecc71",
    "q-cp-etr": "#CE93D8",
    "cp-greedy": "#f39c12",
    "optimal": "#1abc9c",
}

# Extra colours for budget variants
_EXTRA_COLORS = [
    "#1abc9c", "#d35400", "#8e44ad", "#16a085", "#c0392b",
    "#2980b9", "#27ae60", "#7f8c8d", "#2c3e50",
]

VALID_STRATEGIES = {"fail", "full-budget"}


def _parse_method(raw: str) -> tuple[str, int, str | None]:
    """
    Parse a method string like "q-cp-etr:b3:fail" or plain "q-cp-etr".
    Returns (base, budget, strategy).
    Raises ValueError on bad input.
    """
    parts = raw.split(":")
    base = parts[0]
    if base not in METHOD_ARGS:
        raise ValueError(
            f"Unknown base method '{base}'. Valid: {sorted(METHOD_ARGS)}"
        )
    if base == "optimal":
        return base, 0, None
    if len(parts) == 1:
        return base, 0, None
    if len(parts) == 3:
        b_part, s_part = parts[1], parts[2]
        if not b_part.startswith("b"):
            raise ValueError(f"Budget part must start with 'b' (e.g. 'b3'), got '{b_part}'")
        try:
            budget = int(b_part[1:])
        except ValueError:
            raise ValueError(f"Cannot parse budget from '{b_part}'")
        if s_part not in VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy '{s_part}'. Valid: {sorted(VALID_STRATEGIES)}")
        return base, budget, s_part
    raise ValueError(f"Method must be 'base' or 'base:bN:strategy', got '{raw}'")


def _method_label(raw: str) -> str:
    base, budget, strategy = _parse_method(raw)
    lbl = METHOD_LABELS.get(base, base)
    if budget:
        lbl += f" b{budget}/{strategy}"
    return lbl


def _method_color(raw: str, extra_iter) -> str:
    base, _, _ = _parse_method(raw)
    if base in METHOD_COLORS:
        return METHOD_COLORS[base]
    return next(extra_iter, "#95a5a6")


# ---------------------------------------------------------------------------
# Cache helpers  (experiment_results/cache/ — never touches results/)
# ---------------------------------------------------------------------------

def _domain_horizons(inst: str, domain: str) -> tuple[int | None, int | None]:
    """Return Python/CP horizons from the metadata of a supported domain."""
    metadata_name = {
        "navigation": ("navigation", "navigation_mdp_ippc2011.json"),
        "crossingtraffic": ("crossingtraffic", "crossingtraffic_mdp_ippc2011.json"),
    }.get(domain)
    if metadata_name is None:
        return None, None
    metadata_path = ROOT / metadata_name[0] / metadata_name[1]
    try:
        with metadata_path.open(encoding="utf-8") as f:
            data = json.load(f)
        instance = data[str(inst)]
        return int(instance["horizon"]), int(instance["cp_nbSteps"])
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None, None


def _navigation_horizons(inst: str) -> tuple[int | None, int | None]:
    """Backward-compatible Navigation horizon helper."""
    return _domain_horizons(inst, "navigation")


def _cache_parameters(inst: str, domain: str, alpha: float,
                      bp_iter: int, python_horizon: int | None = None,
                      cp_nb_steps: int | None = None) -> tuple:
    """Normalize every experiment parameter that affects a cached result."""
    if domain in ("navigation", "crossingtraffic") and (
        python_horizon is None or cp_nb_steps is None
    ):
        json_horizon, json_cp_steps = _domain_horizons(inst, domain)
        python_horizon = json_horizon if python_horizon is None else python_horizon
        cp_nb_steps = json_cp_steps if cp_nb_steps is None else cp_nb_steps
    return (
        float(alpha),
        int(bp_iter),
        None if python_horizon is None else int(python_horizon),
        None if cp_nb_steps is None else int(cp_nb_steps),
    )


def _cache_token(value) -> str:
    """Create a filesystem-safe, stable token for a cache parameter."""
    if value is None:
        return "na"
    if isinstance(value, float):
        token = format(value, ".12g")
    else:
        token = str(value)
    return token.replace("-", "m").replace("+", "p").replace(".", "p")


def _model_fingerprint(inst: str, domain: str) -> str | None:
    """Return a stable fingerprint for the CrossingTraffic model metadata."""
    if domain != "crossingtraffic":
        return None
    metadata_path = ROOT / "crossingtraffic" / "crossingtraffic_mdp_ippc2011.json"
    try:
        with metadata_path.open(encoding="utf-8") as f:
            instance = json.load(f)[str(inst)]
        payload = json.dumps(instance, sort_keys=True, separators=(",", ":"))
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

def _cache_path(output_dir: Path, inst: str, meth: str, seed: int, episodes: int,
                reward_mode: str = "original",
                domain: str = "frozenlake",
                alpha: float = config.CP_ETR_SHAPING_COEFF,
                bp_iter: int = 1,
                python_horizon: int | None = None,
                cp_nb_steps: int | None = None) -> Path:
    alpha, bp_iter, python_horizon, cp_nb_steps = _cache_parameters(
        inst, domain, alpha, bp_iter, python_horizon, cp_nb_steps
    )
    model_fingerprint = _model_fingerprint(inst, domain)
    safe_meth = meth.replace(":", "-")
    model_suffix = (
        f"_cfg{_cache_token(model_fingerprint)}"
        if domain == "crossingtraffic" else ""
    )
    return output_dir / "cache" / (
        f"{domain}_{inst}_{safe_meth}_reward-{reward_mode}_"
        f"a{_cache_token(alpha)}_bp{bp_iter}_h{_cache_token(python_horizon)}_"
        f"cp{_cache_token(cp_nb_steps)}{model_suffix}_"
        f"seed{seed}_{episodes}eps.json"
    )


def _load_cache(output_dir: Path, inst: str, meth: str, seed: int,
                episodes: int, reward_mode: str = "original",
                domain: str = "frozenlake",
                alpha: float = config.CP_ETR_SHAPING_COEFF,
                bp_iter: int = 1,
                python_horizon: int | None = None,
                cp_nb_steps: int | None = None) -> dict | None:
    alpha, bp_iter, python_horizon, cp_nb_steps = _cache_parameters(
        inst, domain, alpha, bp_iter, python_horizon, cp_nb_steps
    )
    model_fingerprint = _model_fingerprint(inst, domain)
    p = _cache_path(
        output_dir, inst, meth, seed, episodes, reward_mode, domain,
        alpha, bp_iter, python_horizon, cp_nb_steps,
    )
    if not p.exists():
        return None
    try:
        with open(p) as f:
            entry = json.load(f)
            # Old entries without an explicit mode are intentionally ignored;
            # they cannot be proven to match the current reward semantics.
            expected = {
                "reward_mode": reward_mode,
                "domain": domain,
                "alpha": alpha,
                "bp_iter": bp_iter,
                "python_horizon": python_horizon,
                "cp_nb_steps": cp_nb_steps,
            }
            if domain == "crossingtraffic":
                expected["model_fingerprint"] = model_fingerprint
            if any(entry.get(key) != value for key, value in expected.items()):
                return None
            return entry
    except Exception as e:
        print(f"  [WARN] Corrupt cache entry {p.name}: {e} — will re-run")
        return None


def _save_cache(output_dir: Path, inst: str, meth: str, seed: int,
                episodes: int, entry: dict, reward_mode: str = "original",
                domain: str = "frozenlake",
                alpha: float = config.CP_ETR_SHAPING_COEFF,
                bp_iter: int = 1,
                python_horizon: int | None = None,
                cp_nb_steps: int | None = None) -> None:
    alpha, bp_iter, python_horizon, cp_nb_steps = _cache_parameters(
        inst, domain, alpha, bp_iter, python_horizon, cp_nb_steps
    )
    model_fingerprint = _model_fingerprint(inst, domain)
    p = _cache_path(
        output_dir, inst, meth, seed, episodes, reward_mode, domain,
        alpha, bp_iter, python_horizon, cp_nb_steps,
    )
    entry = dict(entry)
    entry.update({
        "domain": domain,
        "reward_mode": reward_mode,
        "alpha": alpha,
        "bp_iter": bp_iter,
        "python_horizon": python_horizon,
        "cp_nb_steps": cp_nb_steps,
    })
    if domain == "crossingtraffic":
        entry["model_fingerprint"] = model_fingerprint
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(entry, f, indent=2)


# ---------------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------------

def _extract_entry(log: dict, domain: str = "frozenlake",
                   reward_mode: str = "original",
                   alpha: float = config.CP_ETR_SHAPING_COEFF,
                   bp_iter: int = 1,
                   python_horizon: int | None = None,
                   cp_nb_steps: int | None = None) -> dict | None:
    """Pull the fields we care about out of a main.py result JSON."""
    eval_log = log.get("evaluation_log", [])
    if not eval_log:
        return None
    last = eval_log[-1]
    alpha, bp_iter, python_horizon, cp_nb_steps = _cache_parameters(
        log.get("instance_id", ""), domain, alpha, bp_iter,
        python_horizon, cp_nb_steps,
    )
    return {
        "eval_log": eval_log,
        "episode_log": log.get("episode_log", []),
        "final_sr": last.get("eval_success_rate", float("nan")),
        "final_dr": last.get("eval_avg_discounted_return", float("nan")),
        "domain": domain,
        "reward_mode": reward_mode,
        "alpha": alpha,
        "bp_iter": bp_iter,
        "python_horizon": python_horizon,
        "cp_nb_steps": cp_nb_steps,
    }


def _budget_stage_changes(meth: str, episodes: int) -> list[int]:
    """
    Returns the list of training episodes at which the curriculum budget increases,
    for a method with strategy=fail. Empty list if not applicable.

    FailStrategy: stage_size = episodes // (max_budget + 1)
    Budget increases at episode k*stage_size for k = 1..max_budget.
    """
    _, budget, strategy = _parse_method(meth)
    if budget == 0 or strategy != "fail":
        return []
    stage_size = episodes // (budget + 1)
    if stage_size == 0:
        return []
    return [k * stage_size for k in range(1, budget + 1)]


# ---------------------------------------------------------------------------
# Subprocess command builder
# ---------------------------------------------------------------------------

def _build_cmd(meth: str, inst: str, episodes: int, seed: int, tmp_dir: str,
               domain: str = "frozenlake", alpha: float = config.CP_ETR_SHAPING_COEFF,
               bp_iter: int = 1, port: int = 12345,
               no_compile: bool = False,
               reward_mode: str = "original") -> list[str]:
    base, budget, strategy = _parse_method(meth)
    agent, shaping = METHOD_ARGS[base]
    cmd = [
        sys.executable, str(ROOT / "main.py"),
        "--domain", domain,
        "--instance", inst,
        "--agent", agent,
        "--seed", str(seed),
        "--results-dir", tmp_dir,
    ]
    # optimal ne prend pas --episodes (il utilise config.EVAL_EPISODES)
    if base != "optimal":
        cmd += ["--episodes", str(episodes)]
    if shaping is not None:
        cmd += ["--shaping", shaping]
    if domain in ("navigation", "crossingtraffic") and shaping == "cp-etr":
        cmd += ["--alpha", str(alpha), "--bp-iter", str(bp_iter)]
    if domain == "navigation":
        cmd += ["--reward-mode", reward_mode]
    if budget:
        cmd += ["--budget", str(budget)]
    if strategy:
        cmd += ["--noslip-strategy", strategy]
    if base in JAVA_METHODS:
        cmd += ["--port", str(port)]
        if no_compile:
            cmd += ["--no-compile"]
    return cmd


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

def _run_one(inst: str, meth: str, seed: int, episodes: int,
             output_dir: Path, tag: str, print_lock: threading.Lock,
             domain: str = "frozenlake", alpha: float = config.CP_ETR_SHAPING_COEFF,
             bp_iter: int = 1,
             port: int = 12345, no_compile: bool = False,
             reward_mode: str = "original",
             progress_bar: "tqdm | None" = None) -> tuple[tuple, dict | None]:
    """Execute a single (inst, meth, seed) run and return ((inst, meth, seed), entry)."""
    tmp_dir = tempfile.mkdtemp(prefix="fl_run_")

    def _done(msg: str, entry):
        with print_lock:
            tqdm.write(msg)
            if progress_bar is not None:
                progress_bar.update(1)
        return (inst, meth, seed), entry

    try:
        _, _, python_horizon, cp_nb_steps = _cache_parameters(
            inst, domain, alpha, bp_iter
        )
        cmd = _build_cmd(
            meth, inst, episodes, seed, tmp_dir, domain=domain,
            alpha=alpha, bp_iter=bp_iter, port=port, no_compile=no_compile,
            reward_mode=reward_mode,
        )
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=None,
        )
        if result.returncode != 0:
            lines = (result.stdout + result.stderr).strip().splitlines()[-20:]
            msg = f"  [ERROR] {tag} — exit code {result.returncode}\n" + "\n".join(f"    {l}" for l in lines)
            return _done(msg, None)

        result_files = list(Path(tmp_dir).rglob("result.json")) or list(Path(tmp_dir).rglob("*_log.json"))
        if not result_files:
            return _done(f"  [WARN]  {tag} — run OK but no result log found", None)

        with open(result_files[0]) as f:
            raw = json.load(f)

        entry = _extract_entry(
            raw, domain=domain, reward_mode=reward_mode,
            alpha=alpha, bp_iter=bp_iter,
            python_horizon=python_horizon, cp_nb_steps=cp_nb_steps,
        )
        if entry is None:
            return _done(f"  [WARN]  {tag} — result log has no evaluation_log", None)

        _save_cache(
            output_dir, inst, meth, seed, episodes, entry,
            reward_mode=reward_mode, domain=domain,
            alpha=alpha, bp_iter=bp_iter,
            python_horizon=python_horizon, cp_nb_steps=cp_nb_steps,
        )
        return _done(f"  [OK]    {tag}", entry)

    except Exception as exc:
        with print_lock:
            tqdm.write(f"  [ERROR] {tag} — {exc}")
            traceback.print_exc()
            if progress_bar is not None:
                progress_bar.update(1)
        return (inst, meth, seed), None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _precompile_java(print_lock: threading.Lock) -> bool:
    """Run 'mvn compile' + generate classpath file once. Returns True on success."""
    cp_dir = ROOT / "MiniCPBP"
    if not cp_dir.is_dir():
        cp_dir = ROOT / "java"
    pom = cp_dir / "pom.xml"
    if not pom.is_file():
        with print_lock:
            print("[ERROR] pom.xml not found — cannot pre-compile Java.")
        return False
    mvn_exec = "mvn.cmd" if sys.platform == "win32" else "mvn"
    cp_file = cp_dir / "target" / "java_classpath.txt"

    # Step 1: compile
    cmd_compile = [mvn_exec, "-f", str(pom), "compile", "-Dexec.cleanupDaemonThreads=false"]
    with print_lock:
        print(f"[PRE-COMPILE] {' '.join(cmd_compile)}")
    result = subprocess.run(cmd_compile, cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        with print_lock:
            print("[ERROR] Maven pre-compile failed:")
            for line in (result.stdout + result.stderr).strip().splitlines()[-30:]:
                print(f"  {line}")
        return False

    # Step 2: generate classpath file for direct java invocation
    cmd_cp = [mvn_exec, "-f", str(pom),
              "dependency:build-classpath",
              f"-Dmdep.outputFile={cp_file}",
              "-q"]
    result2 = subprocess.run(cmd_cp, cwd=str(cp_dir), capture_output=True, text=True, timeout=120)
    if result2.returncode != 0 or not cp_file.exists():
        with print_lock:
            print("[ERROR] Maven dependency:build-classpath failed:")
            for line in (result2.stdout + result2.stderr).strip().splitlines()[-10:]:
                print(f"  {line}")
        return False

    with print_lock:
        print("[PRE-COMPILE] Done.\n")
    return True


def run_all(instances, methods, seeds, episodes, output_dir, force,
            domain: str = "frozenlake", alpha: float = config.CP_ETR_SHAPING_COEFF,
            bp_iter: int = 1, workers: int = 1,
            base_port: int = 12345,
            reward_mode: str = "original") -> dict:
    """
    Launch main.py for each (instance, method, seed) not already cached.

    workers > 1 enables parallel execution for all methods:
    - Non-Java (q-none, q-classic, optimal) : partagent les workers
    - Java (q-cp-ms, q-cp-etr, cp-greedy)  : chaque worker reçoit un port
      dédié du pool [base_port, base_port + workers - 1]
    """
    runs = [(inst, meth, seed)
            for inst in instances
            for meth in methods
            for seed in seeds]

    total = len(runs)
    w = len(str(total))
    data: dict[tuple, dict | None] = {}
    print_lock = threading.Lock()

    # Pool de ports pour les workers Java
    port_pool: queue.Queue[int] = queue.Queue()
    for i in range(workers):
        port_pool.put(base_port + i)

    # Pre-compile Java once if workers > 1 and Java methods are present
    needs_java = any(_parse_method(m)[0] in JAVA_METHODS for m in methods)
    no_compile = False
    # Navigation always uses the precompiled shaded JAR directly; do not
    # trigger Maven during a Narval experiment. FrozenLake keeps its existing
    # pre-compilation path for parallel Java methods.
    if workers > 1 and needs_java and domain not in ("navigation", "crossingtraffic"):
        if not _precompile_java(print_lock):
            print("[WARN] Pre-compile failed; workers will each compile (may conflict).")
        else:
            no_compile = True

    start_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"  Experiment  |  {total} runs  |  episodes={episodes}  |  workers={workers}")
    print(f"{'=' * 60}\n")

    # Separate skippable runs and pending runs
    skipped = 0
    pending = []
    for idx, (inst, meth, seed) in enumerate(runs, 1):
        tag = f"[{idx:{w}}/{total}] {inst:<10} {meth:<16} seed={seed}"
        cached = _load_cache(
            output_dir, inst, meth, seed, episodes,
            reward_mode=reward_mode, domain=domain,
            alpha=alpha, bp_iter=bp_iter,
        )
        if cached is not None and not force:
            tqdm.write(f"{tag}  --> SKIP  (cached)")
            data[(inst, meth, seed)] = cached
            skipped += 1
        else:
            pending.append((inst, meth, seed, tag))

    pbar = tqdm(total=total, initial=skipped, desc="Progress",
                unit="run", dynamic_ncols=True, leave=True)

    def _worker_with_port(inst, meth, seed, tag):
        """Wrapper qui gère l'acquisition/libération du port pour les runs Java."""
        base = _parse_method(meth)[0]
        if base in JAVA_METHODS:
            port = port_pool.get()
            try:
                return _run_one(
                    inst, meth, seed, episodes, output_dir, tag, print_lock,
                    domain=domain, alpha=alpha, bp_iter=bp_iter,
                    port=port, no_compile=no_compile, reward_mode=reward_mode,
                    progress_bar=pbar,
                )
            finally:
                port_pool.put(port)
        else:
            return _run_one(
                inst, meth, seed, episodes, output_dir, tag, print_lock,
                domain=domain, alpha=alpha, bp_iter=bp_iter,
                reward_mode=reward_mode, progress_bar=pbar,
            )

    if pending and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_worker_with_port, inst, meth, seed, tag): (inst, meth, seed)
                for inst, meth, seed, tag in pending
            }
            for future in as_completed(futures):
                try:
                    key, entry = future.result()
                    data[key] = entry
                except Exception as exc:
                    key = futures[future]
                    with print_lock:
                        tqdm.write(f"  [ERROR] {key}: {exc}")
                        pbar.update(1)
                    data[key] = None
    else:
        for inst, meth, seed, tag in pending:
            tqdm.write(f"{tag}  --> running ...", )
            _, entry = _worker_with_port(inst, meth, seed, tag)
            data[(inst, meth, seed)] = entry

    pbar.close()

    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    valid_entries = sum(1 for v in data.values() if v is not None)
    succeeded = valid_entries - skipped
    failed = total - valid_entries
    print(f"\nDone: {succeeded} succeeded, {skipped} skipped, {failed} failed  "
          f"| Durée totale : {minutes}m{seconds:02d}s\n")
    return data


# ---------------------------------------------------------------------------
# Load data from cache only (--plots-only)
# ---------------------------------------------------------------------------

def load_from_cache(instances, methods, seeds, episodes, output_dir,
                    reward_mode: str = "original",
                    domain: str = "frozenlake",
                    alpha: float = config.CP_ETR_SHAPING_COEFF,
                    bp_iter: int = 1) -> dict:
    data: dict[tuple, dict | None] = {}
    for inst in instances:
        for meth in methods:
            for seed in seeds:
                data[(inst, meth, seed)] = _load_cache(
                    output_dir, inst, meth, seed, episodes,
                    reward_mode=reward_mode, domain=domain,
                    alpha=alpha, bp_iter=bp_iter,
                )
    found = sum(1 for v in data.values() if v is not None)
    print(f"Found {found} cache entries.\n")
    return data


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def make_summary_table(data: dict, instances, methods, seeds, output_dir: Path,
                       domain: str = "frozenlake",
                       reward_mode: str = "original") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[tuple, dict | None] = {}
    for inst in instances:
        for meth in methods:
            srs = [data[(inst, meth, s)]["final_sr"]
                   for s in seeds if data.get((inst, meth, s)) is not None]
            drs = [data[(inst, meth, s)]["final_dr"]
                   for s in seeds if data.get((inst, meth, s)) is not None]
            if srs:
                summary[(inst, meth)] = {
                    "sr_mean": float(np.mean(srs)),
                    "sr_std": float(np.std(srs)),
                    "dr_mean": float(np.mean(drs)),
                    "dr_std": float(np.std(drs)),
                    "n": len(srs),
                }
            else:
                summary[(inst, meth)] = None

    table_suffix = "" if domain == "frozenlake" and reward_mode == "original" else (
        f"_{domain}_reward-{reward_mode}"
    )
    csv_path = output_dir / f"summary_table{table_suffix}.csv"
    # Columns grouped by method: instance, m1_SR_mean, m1_SR_std, m1_DR_mean, m1_DR_std, m1_n, m2_...
    header = ["instance"]
    for m in methods:
        lbl = _method_label(m)
        header += [f"{lbl}_SR_mean", f"{lbl}_SR_std", f"{lbl}_DR_mean", f"{lbl}_DR_std", f"{lbl}_n"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for inst in instances:
            row = [inst]
            for m in methods:
                v = summary.get((inst, m))
                if v:
                    row += [f"{v['sr_mean']:.4f}", f"{v['sr_std']:.4f}",
                            f"{v['dr_mean']:.4f}", f"{v['dr_std']:.4f}", str(v['n'])]
                else:
                    row += ["", "", "", "", ""]
            writer.writerow(row)
    print(f"Summary CSV saved to {csv_path}")

    col_w = 18
    labels = [_method_label(m) for m in methods]
    sep = "-" * (12 + col_w * len(methods) * 2)

    def _cell(v, mk, sk):
        return f"{v[mk]:.3f}±{v[sk]:.3f}" if v else "–"

    print("\n" + sep)
    header = f"{'Instance':<12}"
    for lbl in labels:
        header += f"{lbl + ' SR':>{col_w}}"
    for lbl in labels:
        header += f"{lbl + ' DR':>{col_w}}"
    print(header)
    print(sep)
    for inst in instances:
        row = f"{inst:<12}"
        for m in methods:
            row += f"{_cell(summary.get((inst, m)), 'sr_mean', 'sr_std'):>{col_w}}"
        for m in methods:
            row += f"{_cell(summary.get((inst, m)), 'dr_mean', 'dr_std'):>{col_w}}"
        print(row)
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def make_learning_curves(data: dict, instances, methods, seeds, episodes: int,
                         output_dir: Path, domain: str = "frozenlake",
                         reward_mode: str = "original") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid")

    for inst in instances:
        fig, ax = plt.subplots(figsize=(9, 5))
        has_data = False

        # Collect curriculum stage-change x-positions across all methods for this instance
        # Each entry: (x_episode, color, label_suffix)
        vlines: list[tuple[int, str, str]] = []

        extra_iter = iter(_EXTRA_COLORS)
        for meth in methods:
            color = _method_color(meth, extra_iter)
            label = _method_label(meth)
            base, _, _ = _parse_method(meth)

            if base in HLINE_METHODS:
                srs = [data[(inst, meth, s)]["final_sr"]
                       for s in seeds if data.get((inst, meth, s)) is not None]
                if srs:
                    mean_sr = float(np.mean(srs))
                    ax.axhline(mean_sr, linestyle="--", color=color, linewidth=1.8,
                               label=f"{label} (SR={mean_sr:.3f})")
                    has_data = True
                continue

            if base not in Q_METHODS:
                continue

            seed_curves: dict[int, tuple] = {}
            for seed in seeds:
                entry = data.get((inst, meth, seed))
                if entry is None:
                    continue
                eps = [e["training_episode"] for e in entry["eval_log"]]
                srs = [e.get("eval_success_rate", float("nan"))
                       for e in entry["eval_log"]]
                seed_curves[seed] = (eps, srs)

            if not seed_curves:
                continue
            all_ep_sets = [set(v[0]) for v in seed_curves.values()]
            common_eps = sorted(set.intersection(*all_ep_sets)) if all_ep_sets else []
            if not common_eps:
                common_eps = sorted(set.union(*all_ep_sets))

            matrix = []
            for eps, srs in seed_curves.values():
                lookup = dict(zip(eps, srs))
                matrix.append([lookup.get(e, float("nan")) for e in common_eps])
            mat = np.array(matrix, dtype=float)
            mean_c = np.nanmean(mat, axis=0)

            ax.plot(common_eps, mean_c, color=color, linewidth=1.8, label=label)
            has_data = True

            # Collect curriculum stage-change positions for this method
            stage_changes = _budget_stage_changes(meth, episodes)
            for k, ep in enumerate(stage_changes):
                vlines.append((ep, color, f"{label} b{k + 1}→{k + 2}"))

        # Draw curriculum vertical lines (after all curves so they appear on top)
        drawn_eps: set[int] = set()
        for ep, color, lbl in vlines:
            if ep in drawn_eps:
                # Same episode already drawn by another method — just add a grey shared line
                ax.axvline(ep, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
            else:
                ax.axvline(ep, color=color, linestyle=":", linewidth=1.2, alpha=0.7,
                           label=f"curriculum ↑ {lbl}")
                drawn_eps.add(ep)

        if has_data:
            ax.set_title(
                f"Learning curves — Navigation ({reward_mode}) — Instance {inst}"
                if domain == "navigation"
                else (
                    f"Learning curves — CrossingTraffic — Instance {inst}"
                    if domain == "crossingtraffic"
                    else f"Learning curves — {inst}"
                ),
                fontsize=13, fontweight="bold",
            )
            ax.set_xlabel("Training episode")
            ax.set_ylabel("Eval success rate")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=9)
            plt.tight_layout()
            out = (
                output_dir / f"navigation_success_rate_{inst}_reward-{reward_mode}.png"
                if domain == "navigation"
                else output_dir / f"{domain}_learning_curve_{inst}.png"
            )
            plt.savefig(out, dpi=150)
            print(f"Learning curve saved: {out}")
        else:
            print(f"[WARN] No data for instance {inst}, skipping plot.")
        plt.close(fig)

        if domain in ("navigation", "crossingtraffic"):
            # Raw environment-return curves use the same episode logs as the
            # success-rate curves.  Navigation keeps its PPO-compatible
            # timestep axis; CrossingTraffic uses training episodes directly.
            return_fig, return_ax = plt.subplots(figsize=(9, 5))
            return_has_data = False
            extra_iter = iter(_EXTRA_COLORS)
            for meth in methods:
                if _parse_method(meth)[0] not in Q_METHODS:
                    continue
                seed_curves = []
                for seed in seeds:
                    entry = data.get((inst, meth, seed))
                    logs = entry.get("episode_log", []) if entry else []
                    if not logs:
                        continue
                    eps = np.asarray([e["episode"] for e in logs], dtype=float)
                    rewards = np.asarray(
                        [e.get("env_reward", np.nan) for e in logs],
                        dtype=float,
                    )
                    if len(rewards) >= 50:
                        kernel = np.ones(50, dtype=float) / 50
                        padded = np.pad(rewards, (49, 0), mode="edge")
                        rewards = np.convolve(padded, kernel, mode="valid")
                    seed_curves.append((eps * (40 if domain == "navigation" else 1), rewards))
                if not seed_curves:
                    continue
                common_steps = sorted(set.intersection(
                    *(set(curve[0]) for curve in seed_curves)
                ))
                if not common_steps:
                    continue
                matrix = []
                for steps, rewards in seed_curves:
                    lookup = dict(zip(steps, rewards))
                    matrix.append([lookup.get(step, np.nan) for step in common_steps])
                mean_rewards = np.nanmean(np.asarray(matrix, dtype=float), axis=0)
                return_ax.plot(
                    common_steps, mean_rewards, linewidth=1.8,
                    color=_method_color(meth, extra_iter),
                    label=_method_label(meth),
                )
                return_has_data = True
            if return_has_data:
                return_ax.set_title(
                    (
                        f"Learning curves — Navigation ({reward_mode}) — Instance {inst}"
                        if domain == "navigation"
                        else f"Learning curves — CrossingTraffic — Instance {inst}"
                    ),
                    fontsize=13, fontweight="bold",
                )
                return_ax.set_xlabel(
                    "Timestep" if domain == "navigation" else "Training episode"
                )
                return_ax.set_ylabel("Episode return (raw)")
                return_ax.legend(fontsize=9)
                return_ax.grid(alpha=0.3)
                plt.tight_layout()
                return_out = (
                    output_dir / f"navigation_episode_return_{inst}_reward-{reward_mode}.png"
                    if domain == "navigation"
                    else output_dir / f"crossingtraffic_episode_return_{inst}.png"
                )
                plt.savefig(return_out, dpi=150)
                print(f"{domain.capitalize()} return curve saved: {return_out}")
            else:
                print(f"[WARN] No {domain} return data for instance {inst}.")
            plt.close(return_fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help=f"Training episodes per run (default: {DEFAULT_EPISODES})")
    parser.add_argument("--domain", choices=["frozenlake", "navigation", "crossingtraffic"],
                        default="frozenlake",
                        help="Environment domain (default: frozenlake)")
    parser.add_argument("--reward-mode", choices=["original", "frozenlake"],
                        default="original",
                        help="Navigation reward mode (default: original)")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help=f"Number of seeds to run (default: {DEFAULT_SEEDS})")
    parser.add_argument("--seed-start", type=int, default=1,
                        help="Première seed à exécuter (default: 1)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Ne pas générer les graphiques ni le tableau récapitulatif")
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES,
                        metavar="ID", help="Instance IDs to benchmark")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        metavar="METHOD",
                        help=(
                            "Methods to benchmark. Base: q-none q-classic q-cp-ms q-cp-etr cp-greedy. "
                            "With budget: base:bN:strategy  e.g. q-cp-etr:b3:fail"
                        ))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Root output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1 = sequential). "
                             "Java methods utilisent un port dédié par worker.")
    parser.add_argument("--base-port", type=int, default=12345,
                        help="Premier port TCP pour les serveurs Java (default: 12345). "
                             "Worker i utilise base_port + i.")
    parser.add_argument("--alpha", type=float, default=config.CP_ETR_SHAPING_COEFF,
                        help="CP-ETR shaping coefficient")
    parser.add_argument("--bp-iter", type=int, default=1,
                        help="CP server belief-propagation iterations")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if a cache entry already exists")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip running; only regenerate plots/table from cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds < 1 or args.seed_start < 1:
        print("[ERROR] --seeds et --seed-start doivent être >= 1")
        sys.exit(1)

    if args.no_plots and args.plots_only:
        print("[ERROR] --no-plots et --plots-only ne peuvent pas être utilisés ensemble")
        sys.exit(1)

    if args.domain in ("navigation", "crossingtraffic"):
        if args.instances == DEFAULT_INSTANCES:
            args.instances = ["1"] if args.domain == "navigation" else ["2", "4"]
        if args.methods == DEFAULT_METHODS:
            args.methods = ["q-none", "q-cp-etr"]
        if args.domain == "crossingtraffic" and args.reward_mode != "original":
            print("[ERROR] --reward-mode est réservé au domaine Navigation.")
            sys.exit(1)
        unsupported = set(args.methods) - {"q-none", "q-cp-etr"}
        if unsupported:
            print(
                f"[ERROR] {args.domain} supporte seulement q-none et q-cp-etr; "
                f"methodes invalides: {sorted(unsupported)}"
            )
            sys.exit(1)
        if args.alpha < 0 or not np.isfinite(args.alpha):
            print("[ERROR] --alpha doit etre un nombre fini positif ou nul.")
            sys.exit(1)
        if args.bp_iter < 1:
            print("[ERROR] --bp-iter doit etre >= 1.")
            sys.exit(1)
    elif args.reward_mode != "original":
        print("[ERROR] --reward-mode frozenlake est réservé au domaine Navigation.")
        sys.exit(1)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    # Validate method strings early
    try:
        for m in args.methods:
            _parse_method(m)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not args.plots_only:
        data = run_all(
            instances=args.instances,
            methods=args.methods,
            seeds=seeds,
            episodes=args.episodes,
            output_dir=args.output_dir,
            force=args.force,
            domain=args.domain,
            alpha=args.alpha,
            bp_iter=args.bp_iter,
            workers=args.workers,
            base_port=args.base_port,
            reward_mode=args.reward_mode,
        )
    else:
        print("--plots-only: loading from cache …")
        data = load_from_cache(args.instances, args.methods, seeds,
                               args.episodes, args.output_dir,
                               reward_mode=args.reward_mode,
                               domain=args.domain,
                               alpha=args.alpha,
                               bp_iter=args.bp_iter)

    if not args.no_plots:
        make_summary_table(data, args.instances, args.methods, seeds,
                           output_dir=args.output_dir, domain=args.domain,
                           reward_mode=args.reward_mode)
        make_learning_curves(data, args.instances, args.methods, seeds,
                             episodes=args.episodes, output_dir=args.output_dir,
                             domain=args.domain, reward_mode=args.reward_mode)

    if not args.plots_only:
        failed_runs = [
            (inst, method, seed)
            for inst in args.instances
            for method in args.methods
            for seed in seeds
            if data.get((inst, method, seed)) is None
        ]

        if failed_runs:
            print(
                f"[ERROR] {len(failed_runs)} expérience(s) échouée(s).",
                file=sys.stderr,
            )
            for run in failed_runs:
                print(f"  {run}", file=sys.stderr)
            sys.exit(1)
    print("All done.")


if __name__ == "__main__":
    main()

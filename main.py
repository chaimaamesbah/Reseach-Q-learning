# FL/main.py
import argparse
import datetime
import json
import logging
import math
import shutil
import os
import socket
import subprocess
import time
import random
import numpy as np
from tqdm import tqdm

import config
import environment
import utils
import q_learning_standard
import q_learning_cp
import heuristic_agents


def _setup_logging(verbose: int, run_dir: str) -> logging.Logger:
    """
    verbose=0 : terminal uniquement (pas de fichier log)
    verbose>=1 : terminal + run_dir/run.log
    """
    logger = logging.getLogger("fl")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if verbose >= 1:
        fh = logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def vprint(msg: str, verbose: int, min_level: int = 1, logger: logging.Logger = None):
    """Affiche msg si verbose >= min_level ; écrit toujours dans le logger si fourni."""
    if verbose >= min_level:
        tqdm.write(msg)
    if logger:
        logger.info(msg)


def _start_navigation_cp_server(cp_dir: str, port: int, bp_iter: int, run_dir: str):

    metadata_source = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "navigation",
        "navigation_mdp_ippc2011.json",
    )
    metadata_target = os.path.join(cp_dir, "navigation_mdp_ippc2011.json")
    if not os.path.isfile(metadata_source):
        raise FileNotFoundError(f"Metadonnees Navigation manquantes : {metadata_source}")
    if not os.path.isfile(metadata_target):
        raise FileNotFoundError(f"Metadonnees Navigation manquantes : {metadata_target}")

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            raise RuntimeError(f"Le port CP {port} est deja occupe.")
    except ConnectionRefusedError:
        pass

    stdout_path = os.path.join(run_dir, "java_cp_stdout.log")
    stderr_path = os.path.join(run_dir, "java_cp_stderr.log")

    jar_path = os.path.join(
        cp_dir, "target", "minicpbp-1.0.jar"
    )

    if not os.path.isfile(jar_path):
        raise FileNotFoundError(
            f"JAR MiniCPBP introuvable : {jar_path}"
        )

    cmd = [
        "java",
        f"-Dbp.iterations={bp_iter}",
        "-cp",
        jar_path,
        "minicpbp.examples.NavigationService",
        str(port),
    ]

    stdout = open(stdout_path, "w", encoding="utf-8")
    stderr = open(stderr_path, "w", encoding="utf-8")
    process = subprocess.Popen(cmd, cwd=cp_dir, stdout=stdout, stderr=stderr, text=True)

    for _ in range(120):
        if process.poll() is not None:
            stdout.close()
            stderr.close()
            raise RuntimeError(
                f"Serveur Navigation arrete avec code {process.returncode}; "
                f"voir {stderr_path}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.4):
                return process, stdout, stderr
        except OSError:
            time.sleep(0.5)

    _cleanup_cp(process, None)
    stdout.close()
    stderr.close()
    raise TimeoutError(f"Serveur Navigation indisponible; voir {stderr_path}")


def _run_navigation(args, current_seed: int, run_dir: str, log_file: str, logger: logging.Logger):
    """Run standard or CP-ETR Q-learning on the Navigation RDDL environment."""
    from navigation.navigation_environment import NavigationEnv

    if args.agent != "q":
        raise ValueError("Navigation supporte uniquement --agent q.")
    if args.shaping not in ("none", "cp-etr"):
        raise ValueError("Navigation supporte uniquement --shaping none ou cp-etr.")
    if args.budget != 0:
        raise ValueError("--budget n'est pas supporte pour Navigation.")
    if args.episodes <= 0:
        raise ValueError("--episodes doit etre strictement positif.")
    if args.bp_iter <= 0:
        raise ValueError("--bp-iter doit etre strictement positif.")
    if not math.isfinite(args.alpha):
        raise ValueError("--alpha doit etre fini.")

    java_process = None
    cp_client = None
    log_streams = []
    try:
        if args.shaping == "cp-etr":
            from navigation.navigation_cp_etr import NavigationCPClient, NavigationCPEnv

            project_dir = os.path.dirname(os.path.abspath(__file__))
            cp_dir = os.path.join(project_dir, "MiniCPBP")
            if not os.path.isfile(os.path.join(cp_dir, "pom.xml")):
                raise FileNotFoundError(f"pom.xml not found in {cp_dir}")

            cp_port = args.port if args.port is not None else 12350
            if not 1 <= cp_port <= 65535:
                raise ValueError("--port doit etre compris entre 1 et 65535.")
            java_process, *log_streams = _start_navigation_cp_server(
                cp_dir, cp_port, args.bp_iter, run_dir
            )
            cp_client = NavigationCPClient(port=cp_port)
            if not cp_client.connect(str(args.instance)):
                raise RuntimeError("Connexion au serveur Navigation CP impossible.")
            env = NavigationCPEnv(
                instance_id=str(args.instance),
                client=cp_client,
                alpha=args.alpha,
                gamma=config.DISCOUNT_FACTOR,
            )
        else:
            env = NavigationEnv(instance_id=str(args.instance))

        env.reset(seed=current_seed)
        env.action_space.seed(current_seed)
        metadata = {
            "instance_id": f"navigation_{args.instance}",
            "max_steps": env.horizon,
            "size": None,
            "holes": [],
            "goal": env.goal_state_idx,
        }
        vprint(
            f"Instance Navigation '{args.instance}' | Agent: q | "
            f"Shaping: {args.shaping} | Seed: {current_seed}",
            args.verbose,
            min_level=1,
            logger=logger,
        )
        train_fn = lambda e, episodes, max_steps: q_learning_standard.train_q_learning(
            e, episodes, max_steps,
            metadata["size"], metadata["holes"], metadata["goal"],
            args.shaping, metadata["instance_id"],
            verbose=args.verbose, run_dir=run_dir, logger=logger,
        )
        _run_q_learning_session(
            env, train_fn, args, metadata["max_steps"], log_file,
            java_process=java_process, cp_client=cp_client,
            size=None, holes=[], goal=metadata["goal"],
            verbose=args.verbose, run_dir=run_dir, logger=logger,
        )
    finally:
        if cp_client is not None:
            cp_client.close()
        if java_process is not None:
            _cleanup_cp(java_process, None)
        for stream in log_streams:
            stream.close()
        if "env" in locals():
            env.close()


def main():
    parser = argparse.ArgumentParser(description="Train/Evaluate FrozenLake or Navigation agents.")
    parser.add_argument('--domain', choices=['frozenlake', 'navigation'], default='frozenlake')
    parser.add_argument('--instance', type=str, required=True, help="Instance ID for the selected domain")
    parser.add_argument('--agent', type=str, choices=['q', 'optimal', 'cp_greedy'], default='q', help="Agent type")
    parser.add_argument('--shaping', type=str, choices=['none', 'classic', 'cp-ms', 'cp-etr'],
                        default='none', help="Reward shaping for Q-learning")
    parser.add_argument('--budget', type=int, default=0,
                        help="Max no-slip actions per episode (0 = no budget constraint)")
    parser.add_argument('--noslip-strategy', type=str,
                        choices=['fail', 'full-budget'],
                        default='fail',
                        help=(
                            "No-slip action strategy when budget > 0 (default: fail).\n"
                            "  fail        : curriculum croissant, terminaison si budget épuisé, Q init -0.1\n"
                            "  full-budget : budget max dès le début, terminaison si dépassé, Q init 0.0"
                        ))
    parser.add_argument('--episodes', type=int, default=500, help="Training episodes for Q-learning")
    parser.add_argument('--seed', type=int, default=None, help="Random seed to override config.seed_value")
    parser.add_argument('--port', type=int, default=None,
                        help="TCP port for the Java CP server (FrozenLake: 12345, Navigation: 12350)")
    parser.add_argument('--no-compile', action='store_true',
                        help="Skip 'mvn compile' step (use when Java is already compiled)")
    parser.add_argument('--cp-nbsteps-override', type=int, default=-1,
                        help="Override cp_nbSteps for the Java CP server (-1 = use instances.json value)")
    parser.add_argument('--alpha', type=float, default=40.0,
                        help="Navigation CP-ETR shaping coefficient")
    parser.add_argument('--bp-iter', type=int, default=1,
                        help="Navigation CP belief-propagation iterations")
    parser.add_argument('--results-dir', type=str, default=None,
                        help="Dossier racine où écrire les résultats (default: results/)")
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0,
                        help=(
                            "Verbosity level (default: 0).\n"
                            "  0 : JSON résultat + résumé final (terminal uniquement)\n"
                            "  1 : niveau 0 + barre tqdm entraînement + fichier run.log\n"
                            "  2 : niveau 1 + Q-tables CSV à chaque évaluation"
                        ))
    args = parser.parse_args()

    if args.seed is not None:
        current_seed = args.seed
    else:
        current_seed = config.seed_value
    random.seed(current_seed)
    np.random.seed(current_seed)
    args.seed = current_seed

    if args.domain == "navigation":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"q_{args.shaping}_navigation_{args.instance}_{args.episodes}eps_seed{current_seed}_{timestamp}"
        results_base = args.results_dir if args.results_dir is not None else "results"
        run_dir = os.path.join(results_base, run_name)
        os.makedirs(run_dir, exist_ok=True)
        log_file = os.path.join(run_dir, "result.json")
        logger = _setup_logging(args.verbose, run_dir)
        try:
            _run_navigation(args, current_seed, run_dir, log_file, logger)
        except (FileNotFoundError, RuntimeError, ValueError, TimeoutError) as err:
            tqdm.write(f"ERROR Navigation: {err}")
            raise
        return

    if args.port is None:
        args.port = 12345

    instances_path = "instances.json"
    try:
        with open(instances_path, 'r') as f:
            instances = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as err:
        tqdm.write(f"ERROR loading {instances_path}: {err}")
        return
    instance_id = args.instance
    if instance_id not in instances:
        tqdm.write(f"ERROR: Instance '{instance_id}' not found.")
        return
    instance = instances[instance_id]
    try:
        size = instance['size']
        holes = instance['holes']
        goal = instance['goal']
        slippery = instance['slippery']
        max_steps_config = instance['max_steps']
        map_name = instance.get('map_name', f"{size}x{size}")
        desc = instance.get('desc', None)
        optimal_policy_data = instance.get('op', None)
        description = instance.get('description', '')
    except KeyError as e:
        tqdm.write(f"ERROR: Missing key {e} in instance '{instance_id}'.")
        return
    if args.agent == 'optimal' and not optimal_policy_data:
        tqdm.write(f"ERROR: Optimal policy data missing for '{instance_id}'.")
        return

    # --- Dossier de la run ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_log_name = args.agent
    if args.agent == 'q':
        agent_log_name = f"q_{args.shaping}"
        if args.budget > 0:
            agent_log_name += f"_b{args.budget}_{args.noslip_strategy}"
    elif args.agent == 'cp_greedy' and args.budget > 0:
        agent_log_name = f"cp_greedy_b{args.budget}"
    n_eps = args.episodes if args.agent == 'q' else config.EVAL_EPISODES
    run_name = f"{agent_log_name}_{instance_id}_{n_eps}eps_seed{current_seed}_{timestamp}"

    results_base = args.results_dir if args.results_dir is not None else "results"
    run_dir = os.path.join(results_base, run_name)
    os.makedirs(run_dir, exist_ok=True)
    if args.verbose >= 2:
        os.makedirs(os.path.join(run_dir, "qtables"), exist_ok=True)

    log_file = os.path.join(run_dir, "result.json")
    java_stdout_log = os.path.join(run_dir, "java_stdout.log")
    java_stderr_log = os.path.join(run_dir, "java_stderr.log")

    logger = _setup_logging(args.verbose, run_dir)

    vprint(f"Instance: '{instance_id}' ({description}) | Agent: {args.agent} | Seed: {current_seed}",
           args.verbose, min_level=1, logger=logger)
    if args.agent == 'q':
        vprint(f"  Shaping: {args.shaping} | Episodes: {args.episodes} | Verbose: {args.verbose}",
               args.verbose, min_level=1, logger=logger)

    env = environment.create_environment(map_name=map_name, is_slippery=slippery, render_mode=None,
                                         desired_max_steps=max_steps_config, desc=desc, budget=args.budget)
    if env is None:
        tqdm.write("ERROR: Failed to create environment.")
        return
    try:
        env.reset(seed=current_seed)
        env.action_space.seed(current_seed)
    except Exception as e:
        vprint(f"Warning: Could not fully seed environment: {e}", args.verbose, min_level=1, logger=logger)

    java_process = None
    cp_client = None
    final_evaluations = None

    if args.agent == 'optimal':
        final_evaluations = heuristic_agents.run_optimal_policy(env, optimal_policy_data, size, max_steps_config)
        utils.save_results_log({'episode_log': [], 'evaluation_log': final_evaluations}, log_file)

    elif args.agent == 'cp_greedy' or (args.agent == 'q' and args.shaping in ['cp-ms', 'cp-etr']):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        cp_dir = os.path.join(project_dir, 'MiniCPBP')
        if not os.path.isdir(cp_dir):
            cp_dir = os.path.join(project_dir, 'java')
        if not os.path.isdir(cp_dir):
            tqdm.write("ERROR: CP directory not found.")
            env.close()
            return
        pom = os.path.join(cp_dir, 'pom.xml')
        if not os.path.isfile(pom):
            tqdm.write(f"ERROR: pom.xml not found in {cp_dir}.")
            env.close()
            return

        if args.agent == 'cp_greedy' or args.shaping == 'cp-ms':
            mode_str = "MS"
        elif args.shaping == 'cp-etr':
            mode_str = "ETR"
        else:
            tqdm.write(f"ERROR: Cannot determine Java mode for agent='{args.agent}', shaping='{args.shaping}'.")
            env.close()
            return
        java_args_list = [mode_str, str(args.budget), str(args.port), str(args.cp_nbsteps_override)]

        if args.no_compile:
            cp_file = os.path.join(cp_dir, 'target', 'java_classpath.txt')
            if not os.path.isfile(cp_file):
                tqdm.write(f"ERROR: Classpath file not found: {cp_file}. Run pre-compile first.")
                env.close()
                return
            with open(cp_file) as f:
                dep_cp = f.read().strip()
            classes_dir = os.path.join(cp_dir, 'target', 'classes')
            full_cp = classes_dir + os.pathsep + dep_cp
            cmd = ['java', '-cp', full_cp, 'minicpbp.examples.FrozenLakeCPService'] + java_args_list
        else:
            mvn_exec = 'mvn.cmd' if os.name == 'nt' else 'mvn'
            main_class_arg = '-Dexec.mainClass=minicpbp.examples.FrozenLakeCPService'
            exec_args = ' '.join(java_args_list)
            cmd = [mvn_exec, '-f', pom, 'compile', 'exec:java', '-Dexec.cleanupDaemonThreads=false',
                   main_class_arg, f'-Dexec.args={exec_args}']

        vprint(f"Starting Java server (mode={mode_str})...", args.verbose, min_level=1, logger=logger)

        try:
            with open(java_stdout_log, 'w') as out, open(java_stderr_log, 'w') as err:
                java_process = subprocess.Popen(cmd, stdout=out, stderr=err, text=True)
            vprint("Waiting for Java server (10s)...", args.verbose, min_level=1, logger=logger)
            time.sleep(10)
            if java_process.poll() is not None:
                tqdm.write(f"ERROR: Java server terminated (exit {java_process.poll()}).")
                _print_logs(java_stderr_log, java_stdout_log)
                env.close()
                return
            vprint("Java server started.", args.verbose, min_level=1, logger=logger)
            cp_client = q_learning_cp.CPRewardClient(port=args.port)
            if not cp_client.connect():
                tqdm.write("ERROR: Failed to connect to CP server.")
                _print_logs(java_stderr_log, java_stdout_log)
                _cleanup_cp(java_process, cp_client)
                env.close()
                return
            init_resp = cp_client.send_receive(f"INIT {instance_id}")
            if not init_resp.startswith("OK INIT"):
                _cleanup_cp(java_process, cp_client)
                env.close()
                raise RuntimeError(f"CP Server INIT failed: {init_resp}")
            vprint("CP Server INIT successful.", args.verbose, min_level=1, logger=logger)
        except Exception as e:
            tqdm.write(f"ERROR starting Java/CP server: {e}")
            _print_logs(java_stderr_log, java_stdout_log)
            _cleanup_cp(java_process, cp_client)
            env.close()
            return

        if args.agent == 'cp_greedy':
            final_evaluations = heuristic_agents.run_cp_ms_greedy_agent(env, cp_client, config.EVAL_EPISODES,
                                                                        max_steps_config, size, env.action_space.n,
                                                                        instance_id, budget=args.budget)
            utils.save_results_log({'episode_log': [], 'evaluation_log': final_evaluations}, log_file)
            _cleanup_cp(java_process, cp_client)

        elif args.agent == 'q':
            noslip_strat = args.noslip_strategy
            train_fn = lambda env, episode, maxSteps: q_learning_cp.train_q_learning_with_cp_shaping(
                env, cp_client, episode, maxSteps, args.shaping, size, holes, goal, instance_id,
                noslip_strategy_name=noslip_strat,
                verbose=args.verbose, run_dir=run_dir, logger=logger)
            _run_q_learning_session(env, train_fn, args, max_steps_config, log_file,
                                    java_process, cp_client, java_stdout_log, java_stderr_log,
                                    size=size, holes=holes, goal=goal,
                                    verbose=args.verbose, run_dir=run_dir, logger=logger)

    elif args.agent == 'q':
        current_shaping_name = args.shaping
        if args.shaping == 'none':
            train_fn = lambda e, ep, ms: q_learning_standard.train_q_learning(
                e, ep, ms, size, holes, goal, current_shaping_name, instance_id,
                verbose=args.verbose, run_dir=run_dir, logger=logger)
        elif args.shaping == 'classic':
            train_fn = lambda e, ep, ms: q_learning_standard.train_q_learning_with_classic_shaping(
                e, ep, ms, size, holes, goal, current_shaping_name, instance_id,
                verbose=args.verbose, run_dir=run_dir, logger=logger)
        else:
            tqdm.write(f"ERROR: Invalid shaping '{args.shaping}' for non-CP Q-agent.")
            env.close()
            return
        _run_q_learning_session(env, train_fn, args, max_steps_config, log_file,
                                size=size, holes=holes, goal=goal,
                                verbose=args.verbose, run_dir=run_dir, logger=logger)
    else:
        tqdm.write(f"ERROR: Unknown agent type '{args.agent}'")

    if env:
        env.close()


def _run_q_learning_session(env, train_fn, args, max_steps_config, log_file,
                            java_process=None, cp_client=None,
                            java_stdout_log=None, java_stderr_log=None,
                            size=None, holes=None, goal=None,
                            verbose=0, run_dir=None, logger=None):
    q_table, episodes, evaluations, aug_mapper = None, [], [], None
    start_time = time.time()
    try:
        vprint(f"Starting Q-learning: {args.episodes} episodes | shaping={args.shaping}",
               verbose, min_level=1, logger=logger)
        result = train_fn(env, args.episodes, max_steps_config)
        # CP training returns (q_table, episodes, evaluations, aug_mapper); standard returns (q_table, episodes, evaluations)
        if len(result) == 4:
            q_table, episodes, evaluations, aug_mapper = result
        else:
            q_table, episodes, evaluations = result
    except Exception as e:
        tqdm.write(f"ERROR during training: {e}")
        if args.domain == "navigation":
            raise
        import traceback
        traceback.print_exc()
    finally:
        duration = time.time() - start_time
        if java_process:
            _cleanup_cp(java_process, cp_client)

    if not episodes and not evaluations:
        tqdm.write("No results from Q-learning to save.")
        return

    try:
        utils.save_results_log({'episode_log': episodes, 'evaluation_log': evaluations}, log_file)
    except Exception as e:
        tqdm.write(f"Error saving results log: {e}")

    plots_dir = os.path.join(run_dir, "plots")
    try:
        utils.plot_results([log_file], plots_dir)
        if q_table is not None and size is not None:
            policy_label = args.shaping
            if args.budget > 0:
                policy_label += f"_b{args.budget}_{args.noslip_strategy}"
            utils.visualize_policy(
                q_table, size, holes, goal,
                title=f"Politique finale — {policy_label} ({args.instance})",
                save_path=os.path.join(plots_dir, "policy.png"),
                aug_mapper=aug_mapper,
            )
    except Exception as e:
        tqdm.write(f"Error generating plots: {e}")

    final_eval = evaluations[-1] if evaluations else {}
    sr = final_eval.get('eval_success_rate', float('nan'))
    dr = final_eval.get('eval_avg_discounted_return', float('nan'))
    tqdm.write(f"\n=== Run complete | {args.instance} | {args.shaping} | seed={args.seed} ===")
    tqdm.write(f"  Duration :         {duration:.1f}s")
    tqdm.write(f"  Final SR :         {sr:.2%}")
    tqdm.write(f"  Final Disc.Return: {dr:.4f}")
    tqdm.write(f"  Result JSON :      {log_file}")
    tqdm.write(f"  Plots :            {plots_dir}/")
    if logger:
        logger.info(f"Run complete | SR={sr:.2%} | DR={dr:.4f} | duration={duration:.1f}s")


def _print_logs(stderr_path: str, stdout_path: str) -> None:
    tqdm.write("\n--- Java Server Logs ---")
    for path in (stderr_path, stdout_path):
        if path and os.path.exists(path):
            tqdm.write(f"--- {os.path.basename(path)} ---")
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    tqdm.write(content if content else "[File is empty]")
            except Exception as e:
                tqdm.write(f"Error reading log file {path}: {e}")
    tqdm.write("--- End Java Server Logs ---\n")


def _cleanup_cp(java_proc, client) -> None:
    if client and getattr(client, 'is_connected', False):
        client.close()
    if java_proc and java_proc.poll() is None:
        java_proc.terminate()
        try:
            java_proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            java_proc.kill()
            java_proc.communicate()
        except Exception:
            pass


if __name__ == '__main__':
    main()

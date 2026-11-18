#!/bin/bash
#SBATCH --job-name=ct-qlearning
#SBATCH --array=0-39
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/crossingtraffic_%A_%a.out
#SBATCH --error=slurm_logs/crossingtraffic_%A_%a.err

set -euo pipefail

# The Java shaded JAR must be built once before submitting this array.  This
# script deliberately never invokes Maven on Narval.
PROJECT="${HOME}/Research-Q-learning-navigation"
ENV_DIR="${HOME}/ENV"
source "${ENV_DIR}/bin/activate"
cd "${PROJECT}"
mkdir -p slurm_logs

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 20 )); then
    INSTANCE="2"
else
    INSTANCE="4"
fi
LOCAL_ID=$(( TASK_ID % 20 ))
if (( LOCAL_ID < 10 )); then
    METHOD="q-none"
else
    METHOD="q-cp-etr"
fi
SEED=$(( LOCAL_ID % 10 + 1 ))
PORT=$(( 20000 + TASK_ID ))

exec python run_experiment.py \
    --domain crossingtraffic \
    --instances "${INSTANCE}" \
    --methods "${METHOD}" \
    --episodes 10000 \
    --seeds 1 \
    --seed-start "${SEED}" \
    --alpha 1 \
    --bp-iter 1 \
    --workers 1 \
    --base-port "${PORT}" \
    --no-plots \
    --output-dir crossingtraffic_10k

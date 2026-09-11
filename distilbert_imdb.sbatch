#!/usr/bin/env bash
#SBATCH --job-name=distilbert-imdb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="${DISTILBERT_PROJECT_DIR:-$SCRIPT_DIR}"
readonly PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_SEED="${1:-42}"
RUN_EPOCHS="${2:-2}"
NUM_WORKERS="${3:-4}"

if [[ ! "$RUN_SEED" =~ ^[0-9]+$ ]]; then
    echo "Error: seed must be a non-negative integer." >&2
    exit 2
fi

if [[ ! "$RUN_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: epochs must be a positive integer." >&2
    exit 2
fi

if [[ ! "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "Error: num-workers must be a non-negative integer." >&2
    exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/train_distilbert.py" ]]; then
    echo "Error: trainer not found: $PROJECT_DIR/train_distilbert.py" >&2
    exit 1
fi

readonly RUN_DIR="$PROJECT_DIR/runs/seed_${RUN_SEED}"
if [[ -s "$RUN_DIR/metrics.json" ]]; then
    echo "Error: completed results already exist: $RUN_DIR/metrics.json" >&2
    echo "Choose a different seed to avoid overwriting the run." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"

mkdir -p \
    "$PROJECT_DIR/logs" \
    "$PROJECT_DIR/runs" \
    "$HF_DATASETS_CACHE" \
    "$HF_HUB_CACHE"

cd "$PROJECT_DIR"

echo "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "Seed: $RUN_SEED"
echo "Maximum epochs: $RUN_EPOCHS"
echo "DataLoader workers: $NUM_WORKERS"
echo "Python: $PYTHON_BIN"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" train_distilbert.py \
    --seed "$RUN_SEED" \
    --epochs "$RUN_EPOCHS" \
    --train-batch-size 32 \
    --eval-batch-size 64 \
    --max-length 256 \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --num-workers "$NUM_WORKERS" \
    --patience 2

echo "Training completed successfully."
echo "Results: $RUN_DIR"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# ============================================================
# Training run: P-LoRA sweep (or single trial) on Qwen3-VL-4B,
# same skeleton as run_all.sh (venv -> deps -> dvc pull ->
# preflight -> the expensive step -> dvc/git push) but pointed
# at src/training/train.py instead of the benchmark worker.
#
# Assumes you've already run, on this pod, before launching
# this script:
#   wandb login
#   git config / gh auth (or SSH key already on the pod)
#   dvc remote already configured with working credentials
# This script does not touch any of that -- it just consumes
# --wandb-project if you set WANDB_PROJECT below.
# ============================================================


# ============================================================
# Configuration -- edit these, don't pass CLI args, to keep
# this simple and match how run_all.sh works.
# ============================================================

TRAIN_IMAGES="$PROJECT_ROOT/data/invoices/processed/splits/train/images"
TRAIN_LABELS="$PROJECT_ROOT/data/invoices/processed/splits/train/labels"
VAL_IMAGES="$PROJECT_ROOT/data/invoices/processed/splits/val/images"
VAL_LABELS="$PROJECT_ROOT/data/invoices/processed/splits/val/labels"

MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
MODE="plora"                 # "plora" or "single"
TRIAL="r32_lr2e-4"           # only used when MODE=single

TOTAL_STEPS=2000
EVAL_EVERY=100
SAVE_MERGED=true             # also produce a merged checkpoint for vLLM serving

# Leave WANDB_PROJECT empty ("") to skip wandb entirely.
WANDB_PROJECT="fatura-plora"
WANDB_RUN_NAME=""            # optional; empty lets wandb auto-name the run

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$PROJECT_ROOT/runs/${MODE}_${TIMESTAMP}"
LOG_FILE_DIR="$PROJECT_ROOT/runs"
mkdir -p "$LOG_FILE_DIR"
LOG_FILE="$LOG_FILE_DIR/run_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1


echo "========================================"
echo " Invoice LoRA Training Run"
echo "========================================"
echo "Project root : $PROJECT_ROOT"
echo "Mode         : $MODE"
echo "Model        : $MODEL_ID"
echo "Output       : $OUT_ROOT"
echo "W&B project  : ${WANDB_PROJECT:-<disabled>}"
echo "Started      : $(date)"
echo "========================================"


# ============================================================
# 0. Python environment
# ============================================================

echo ""
echo "[0/7] Setting up Python environment..."

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip


# ============================================================
# 1. Install dependencies
# ============================================================

echo ""
echo "[1/7] Installing dependencies..."

python -m pip install -r requirements.txt


# ============================================================
# 2. Pull dataset
# ============================================================

echo ""
echo "[2/7] Pulling dataset with DVC..."

dvc pull


# ============================================================
# 3. Preflight checks -- must pass before we touch the GPU.
# ============================================================

echo ""
echo "[3/7] Preflight checks..."

for d in "$TRAIN_IMAGES" "$TRAIN_LABELS" "$VAL_IMAGES" "$VAL_LABELS"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: expected directory not found: $d"
        echo "Did 'dvc pull' actually fetch the split, or does it need building first?"
        exit 1
    fi
done

TRAIN_COUNT=$(find "$TRAIN_IMAGES" -maxdepth 1 -type f | wc -l)
VAL_COUNT=$(find "$VAL_IMAGES" -maxdepth 1 -type f | wc -l)
echo "Train images: $TRAIN_COUNT | Val images: $VAL_COUNT"

if [ "$TRAIN_COUNT" -eq 0 ] || [ "$VAL_COUNT" -eq 0 ]; then
    echo "ERROR: train or val image directory is empty."
    exit 1
fi

echo "Checking model repo is reachable on Hugging Face Hub..."
python - <<PY
from huggingface_hub import HfApi
api = HfApi()
model_id = "$MODEL_ID"
try:
    api.model_info(model_id)
    print(f"  OK   {model_id}")
except Exception as e:
    print(f"  FAIL {model_id}: {e}")
    raise SystemExit(1)
PY

if [ -n "$WANDB_PROJECT" ]; then
    python -c "import wandb" 2>/dev/null || {
        echo "ERROR: WANDB_PROJECT is set but wandb isn't installed."
        echo "Add wandb to requirements.txt or unset WANDB_PROJECT."
        exit 1
    }
fi

echo "Preflight checks passed."


# ============================================================
# 4. Run training (the expensive part)
# ============================================================

echo ""
echo "[4/7] Running training ($MODE)..."

mkdir -p "$OUT_ROOT"

WANDB_ARGS=()
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARGS+=(--wandb-project "$WANDB_PROJECT")
    if [ -n "$WANDB_RUN_NAME" ]; then
        WANDB_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
    fi
fi

SAVE_MERGED_ARGS=()
if [ "$SAVE_MERGED" = true ]; then
    SAVE_MERGED_ARGS+=(--save-merged)
fi

TRAIN_ARGS=(
    --model-id "$MODEL_ID"
    --train-images "$TRAIN_IMAGES"
    --train-labels "$TRAIN_LABELS"
    --val-images "$VAL_IMAGES"
    --val-labels "$VAL_LABELS"
    --output-dir "$OUT_ROOT"
    --mode "$MODE"
    --total-steps "$TOTAL_STEPS"
    --eval-every "$EVAL_EVERY"
)
if [ "$MODE" = "single" ]; then
    TRAIN_ARGS+=(--trial "$TRIAL")
fi

python src/training/train.py "${TRAIN_ARGS[@]}" "${WANDB_ARGS[@]}" "${SAVE_MERGED_ARGS[@]}"

if [ ! -d "$OUT_ROOT/best_adapter" ]; then
    echo "ERROR: training finished but no best_adapter found in $OUT_ROOT -- not pushing."
    exit 1
fi

echo "Training finished. Best adapter saved to $OUT_ROOT/best_adapter"


# ============================================================
# 5. Track results with DVC
# ============================================================

echo ""
echo "[5/7] Tracking trained model with DVC..."

dvc add "$OUT_ROOT"

echo "DVC tracking complete."


# ============================================================
# 6. Commit Git metadata and push DVC data
# ============================================================

echo ""
echo "[6/7] Committing metadata and pushing results..."

git add .
git status

git commit -m "Add training run ${MODE} ${TIMESTAMP} -> $OUT_ROOT" \
    || echo "Nothing new to commit."

git push

echo "Git push completed."

echo ""
echo "Pushing model to DVC remote..."

dvc push

echo "DVC push completed."


# ============================================================
# 7. Final verification
# ============================================================

echo ""
echo "========================================"
echo " RUN FINISHED"
echo "========================================"
echo "Results         : $OUT_ROOT"
echo "Best adapter    : $OUT_ROOT/best_adapter"
if [ "$SAVE_MERGED" = true ]; then
    echo "Merged model    : $OUT_ROOT/merged"
fi
echo "Summary JSON    : $OUT_ROOT/sweep_summary.json"
echo ""
echo "DVC status:"
dvc status
echo ""
echo "Git status:"
git status
echo ""
echo "Finished: $(date)"
echo "========================================"
echo " MODEL TRAINED AND PERSISTED (DVC + Git)."
echo " You can now safely shut down the Pod."
echo "========================================"
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# ============================================================
# Full training run: P-LoRA sweep (or single trial) on Qwen3-VL-4B.
#
# Based on smoke_test.sh (which is known to work end-to-end), plus:
#   - trained model is tracked with DVC and pushed
#   - merged checkpoint is saved for vLLM serving
#   - TOTAL_STEPS is derived from the train set size (EPOCHS epochs)
#   - W&B logging on; full 1k-validation eval done with vLLM on the
#     merged checkpoint (fast) instead of HF generate (slow)
#
# Assumes the same one-time setup as before:
#   git auth / dvc remote already configured
# ============================================================


# ============================================================
# Configuration
# ============================================================

TRAIN_IMAGES="$PROJECT_ROOT/data/invoices/processed/splits/train/images"
TRAIN_LABELS="$PROJECT_ROOT/data/invoices/processed/splits/train/labels"
VAL_IMAGES="$PROJECT_ROOT/data/invoices/processed/splits/val/images"
VAL_LABELS="$PROJECT_ROOT/data/invoices/processed/splits/val/labels"

MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
MODE="${MODE:-plora}"        # "plora" or "single"
TRIAL="r32_lr2e-4"           # only used when MODE=single

# Batch shape (must match what the smoke test ran with: train.py defaults).
MICRO_BATCH=4
GRAD_ACCUM=4
EPOCHS=1                     # 1 epoch is plenty: smoke test hit 98.6% after 50 steps

EVAL_EVERY=50                # only controls how often train loss is printed
TRAIN_TIME_EVAL=20           # tiny HF-generate sanity check; the real eval is vLLM below
SAVE_MERGED=true             # also produce a merged checkpoint for vLLM serving

# Requires the patched train.py / multi_lora_trainer.py (wandb flags + logging),
# `wandb` in requirements.txt, and `wandb login` (or WANDB_API_KEY) on the pod.
# Set WANDB_PROJECT="" to disable.
WANDB_PROJECT="${WANDB_PROJECT-fatura-plora}"   # WANDB_PROJECT="" disables
VLLM_EVAL_MAX="${VLLM_EVAL_MAX:-0}"             # 0 = all validation images
WANDB_RUN_NAME=""            # empty -> "<mode>_<timestamp>"

# Reduces fragmentation; the smoke test showed a CUDA OOM-retry warning
# (4GB alloc with ~3.8GB free) around the first pruning rung.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${MODE}_${TIMESTAMP}}"
OUT_ROOT="$PROJECT_ROOT/runs/${MODE}_${TIMESTAMP}"
LOG_FILE_DIR="$PROJECT_ROOT/runs/train_logs"
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

# Only the processed data -- a bare `dvc pull` would also download every
# previous training run tracked in the repo.
dvc pull data/invoices/processed


# ============================================================
# 3. Preflight checks -- must pass before we touch the GPU.
# ============================================================

echo ""
echo "[3/7] Preflight checks..."

for d in "$TRAIN_IMAGES" "$TRAIN_LABELS" "$VAL_IMAGES" "$VAL_LABELS"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: expected directory not found: $d"
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

# One optimizer step consumes MICRO_BATCH * GRAD_ACCUM documents.
EFFECTIVE_BATCH=$((MICRO_BATCH * GRAD_ACCUM))
STEPS_PER_EPOCH=$((TRAIN_COUNT / EFFECTIVE_BATCH))
TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
echo "Effective batch: $EFFECTIVE_BATCH docs/step"
echo "Steps per epoch: $STEPS_PER_EPOCH  ->  TOTAL_STEPS=$TOTAL_STEPS ($EPOCHS epoch(s))"

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
        echo "Add wandb to requirements.txt or set WANDB_PROJECT=\"\"."
        exit 1
    }
    if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
        echo "ERROR: not logged in to W&B. Run 'wandb login' or export WANDB_API_KEY."
        exit 1
    fi
    echo "W&B OK: project=$WANDB_PROJECT run=$WANDB_RUN_NAME"
fi

echo "GPU check:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo "Preflight checks passed."


# ============================================================
# 4. Run training (the expensive part)
# ============================================================

echo ""
echo "[4/7] Running training ($MODE, $TOTAL_STEPS steps)..."

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
    --micro-batch-size "$MICRO_BATCH"
    --grad-accum-steps "$GRAD_ACCUM"
    --total-steps "$TOTAL_STEPS"
    --eval-every "$EVAL_EVERY"
    --final-eval-examples "$TRAIN_TIME_EVAL"
)
if [ "$MODE" = "single" ]; then
    TRAIN_ARGS+=(--trial "$TRIAL")
fi

# Run as a module (same as the smoke test) so `from src.training...` imports work.
python -m src.training.train \
    "${TRAIN_ARGS[@]}" \
    ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} \
    ${SAVE_MERGED_ARGS[@]+"${SAVE_MERGED_ARGS[@]}"}

if [ ! -d "$OUT_ROOT/best_adapter" ]; then
    echo "ERROR: training finished but no best_adapter found in $OUT_ROOT -- not pushing."
    exit 1
fi

if [ "$SAVE_MERGED" = true ] && [ ! -d "$OUT_ROOT/merged" ]; then
    # best_adapter exists, so still persist it rather than losing hours of training.
    echo "WARNING: SAVE_MERGED=true but $OUT_ROOT/merged is missing."
    echo "         Continuing: best_adapter will be pushed; merge it manually later."
fi

echo "Training finished. Best adapter saved to $OUT_ROOT/best_adapter"


# ============================================================
# 4b. Full-validation eval with vLLM (separate process -> GPU is free).
#     Non-fatal: a failure here must never block persisting the model.
# ============================================================

echo ""
echo "[4b/7] Full validation eval with vLLM ($VAL_COUNT examples)..."

run_vllm_eval() {   # $1 = model path/id, $2 = tag
    env -u PYTORCH_CUDA_ALLOC_CONF python -m src.training.vllm_eval \
        --model "$1" --tag "$2" \
        --val-images "$VAL_IMAGES" --val-labels "$VAL_LABELS" \
        --out-dir "$OUT_ROOT" --max-examples "$VLLM_EVAL_MAX" \
        ${WANDB_PROJECT:+--wandb-project "$WANDB_PROJECT"} \
        || echo "WARNING: vLLM eval ($2) failed -- continuing."
}

if [ -d "$OUT_ROOT/merged" ]; then
    run_vllm_eval "$OUT_ROOT/merged" finetuned
    run_vllm_eval "$MODEL_ID" base        # full-set baseline, ~minutes
else
    echo "Skipping: no merged checkpoint to evaluate."
fi


# ============================================================
# 5. Track results with DVC
# ============================================================

echo ""
echo "[5/7] Tracking trained model with DVC..."

dvc add "$OUT_ROOT"

echo "DVC tracking complete."


# ============================================================
# 6. Push model to DVC remote FIRST, then commit + push metadata
#    (so git never points at data that isn't on the remote)
# ============================================================

echo ""
echo "[6/7] Pushing results..."

echo "Pushing model to DVC remote..."
dvc push
echo "DVC push completed."

# Stage only what this run produced -- not `git add .`
git add "${OUT_ROOT}.dvc" "$PROJECT_ROOT/runs/.gitignore"
git status

git commit -m "Add training run ${MODE} ${TIMESTAMP} (${TOTAL_STEPS} steps, ${EPOCHS} epoch)" \
    || echo "Nothing new to commit."

git push
echo "Git push completed."


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
echo "DVC status vs remote (should report up to date):"
dvc status -c
echo ""
echo "Git status:"
git status
echo ""
echo "Finished: $(date)"
echo "========================================"
echo " MODEL TRAINED AND PERSISTED (DVC + Git)."
echo " You can now safely shut down the Pod."
echo "========================================"

# Best-effort: commit the run log too (non-fatal if it fails).
git add "$LOG_FILE" 2>/dev/null \
    && git commit -m "Training log ${TIMESTAMP}" >/dev/null 2>&1 \
    && git push >/dev/null 2>&1 \
    || echo "(log commit skipped)"
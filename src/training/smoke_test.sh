#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# ============================================================
# Smoke test: run training for a handful of steps to verify
# the pipeline works end-to-end and to measure real steps/sec
# on this GPU, BEFORE committing to a full paid run.
#
# Differs from run_train.sh:
#   - very low TOTAL_STEPS (default 30)
#   - NO dvc add / dvc push of the model -- this run's output
#     is throwaway, not meant to be tracked
#   - only the run log gets committed and pushed to GitHub,
#     so you have a timing record without any model artifact
#   - prints an extrapolated full-run time/cost estimate at
#     the end
#
# Assumes the same one-time setup as run_train.sh:
#   wandb login / git auth / dvc remote already configured
# (wandb is optional here too -- leave WANDB_PROJECT empty to
# skip it, useful if you don't want smoke-test noise in your
# real project's run history)
# ============================================================


# ============================================================
# Configuration
# ============================================================

TRAIN_IMAGES="$PROJECT_ROOT/data/invoices/processed/splits/train/images"
TRAIN_LABELS="$PROJECT_ROOT/data/invoices/processed/splits/train/labels"
VAL_IMAGES="$PROJECT_ROOT/data/invoices/processed/splits/val/images"
VAL_LABELS="$PROJECT_ROOT/data/invoices/processed/splits/val/labels"

MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
MODE="plora"                # smoke-test single-adapter by default -- fastest
                              # signal on whether the pipeline works at all;
                              # switch to "plora" if you specifically want to
                              # time the multi-adapter overhead
TRIAL="r32_lr2e-4"

SMOKE_STEPS=50                # keep this small -- this is a timing probe, not a run
EVAL_EVERY=10

# The real run you're planning to extrapolate towards:
TARGET_TOTAL_STEPS=1500

# Leave empty ("") to skip wandb for this smoke test.
WANDB_PROJECT=""
WANDB_RUN_NAME=""

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$PROJECT_ROOT/runs/smoketest_${TIMESTAMP}"
LOG_FILE_DIR="$PROJECT_ROOT/runs/smoketest_logs"
mkdir -p "$LOG_FILE_DIR"
LOG_FILE="$LOG_FILE_DIR/smoketest_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1


echo "========================================"
echo " Smoke Test - Invoice LoRA Training"
echo "========================================"
echo "Project root  : $PROJECT_ROOT"
echo "Mode          : $MODE"
echo "Model         : $MODEL_ID"
echo "Smoke steps   : $SMOKE_STEPS"
echo "Output (temp) : $OUT_ROOT"
echo "W&B project   : ${WANDB_PROJECT:-<disabled>}"
echo "Started       : $(date)"
echo "========================================"
echo ""
echo "NOTE: this run's model output is NOT tracked with DVC and will"
echo "not be pushed anywhere. Only this log file gets committed to git."
echo "========================================"


# ============================================================
# 0. Python environment
# ============================================================

echo ""
echo "[0/6] Setting up Python environment..."

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
echo "[1/6] Installing dependencies..."

python -m pip install -r requirements.txt

# ============================================================
# 2. Pull dataset
# ============================================================

echo ""
echo "[2/6] Pulling dataset with DVC..."

dvc pull data/invoices/processed


# ============================================================
# 3. Preflight checks
# ============================================================

echo ""
echo "[3/6] Preflight checks..."

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

echo "GPU check:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo "Preflight checks passed."


# ============================================================
# 4. Run the smoke test (timed)
# ============================================================

echo ""
echo "[4/6] Running smoke test ($SMOKE_STEPS steps, mode=$MODE)..."

mkdir -p "$OUT_ROOT"

WANDB_ARGS=()
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARGS+=(--wandb-project "$WANDB_PROJECT")
    if [ -n "$WANDB_RUN_NAME" ]; then
        WANDB_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
    fi
fi

TRAIN_ARGS=(
    --model-id "$MODEL_ID"
    --train-images "$TRAIN_IMAGES"
    --train-labels "$TRAIN_LABELS"
    --val-images "$VAL_IMAGES"
    --val-labels "$VAL_LABELS"
    --output-dir "$OUT_ROOT"
    --mode "$MODE"
    --total-steps "$SMOKE_STEPS"
    --eval-every "$EVAL_EVERY"
)
if [ "$MODE" = "single" ]; then
    TRAIN_ARGS+=(--trial "$TRIAL")
fi

START_EPOCH=$(date +%s)

python src/training/train.py "${TRAIN_ARGS[@]}" "${WANDB_ARGS[@]}"

END_EPOCH=$(date +%s)
ELAPSED_SEC=$((END_EPOCH - START_EPOCH))

if [ ! -d "$OUT_ROOT/best_adapter" ]; then
    echo "ERROR: smoke test finished but no best_adapter found -- something"
    echo "is wrong with the pipeline. Do NOT trust this as a timing result"
    echo "either, since it may have failed partway through."
    exit 1
fi

echo "Smoke test training call finished."


# ============================================================
# 5. Timing / cost extrapolation
# ============================================================

echo ""
echo "[5/6] Timing results..."
echo "========================================"
echo " Elapsed for $SMOKE_STEPS steps : ${ELAPSED_SEC}s"

SEC_PER_STEP=$(awk -v e="$ELAPSED_SEC" -v s="$SMOKE_STEPS" 'BEGIN { printf "%.3f", e/s }')
echo " Seconds/step (incl. one-time model load overhead) : $SEC_PER_STEP"

echo ""
echo " NOTE: this includes ONE-TIME fixed overhead (model download/load,"
echo " dataset init, CUDA kernel autotuning on step 1) amortized over only"
echo " $SMOKE_STEPS steps. On a longer run that overhead is amortized over"
echo " many more steps, so the extrapolation below is a conservative"
echo " (slightly pessimistic) upper bound on real per-step time."

EST_TOTAL_SEC=$(awk -v spt="$SEC_PER_STEP" -v t="$TARGET_TOTAL_STEPS" 'BEGIN { printf "%.0f", spt*t }')
EST_TOTAL_MIN=$(awk -v s="$EST_TOTAL_SEC" 'BEGIN { printf "%.1f", s/60 }')
EST_TOTAL_HR=$(awk -v s="$EST_TOTAL_SEC" 'BEGIN { printf "%.2f", s/3600 }')

echo ""
echo " Extrapolated estimate for TARGET_TOTAL_STEPS=$TARGET_TOTAL_STEPS ($MODE mode):"
echo "   ~${EST_TOTAL_SEC}s  (~${EST_TOTAL_MIN} min / ~${EST_TOTAL_HR} hr)"
echo ""
echo " If mode=plora and your real run uses more concurrently-live trials"
echo " than this smoke test did, multiply this estimate accordingly for"
echo " the early (pre-pruning) phase of the run."
echo "========================================"


# ============================================================
# 6. Commit and push ONLY the log file to GitHub
#    (no dvc add, no dvc push -- this run's model output is
#    discarded, not tracked)
# ============================================================

echo ""
echo "[6/6] Committing smoke-test log to GitHub..."

git add "$LOG_FILE"
git status

git commit -m "Smoke test log ${TIMESTAMP} (${SMOKE_STEPS} steps, ${SEC_PER_STEP}s/step, mode=${MODE})" \
    || echo "Nothing new to commit."

git push

echo "Git push completed."

echo ""
echo "Cleaning up throwaway smoke-test output dir..."
rm -rf "$OUT_ROOT"
echo "Removed $OUT_ROOT (was never tracked by DVC)."


# ============================================================
# Final summary
# ============================================================

echo ""
echo "========================================"
echo " SMOKE TEST COMPLETE"
echo "========================================"
echo "Seconds/step        : $SEC_PER_STEP"
echo "Estimated full run  : ~${EST_TOTAL_HR} hr for $TARGET_TOTAL_STEPS steps"
echo "Log pushed to GitHub: $LOG_FILE"
echo "Model output        : discarded (not tracked)"
echo "Finished            : $(date)"
echo "========================================"
echo " Review the estimate above before launching run_train.sh."
echo "========================================"
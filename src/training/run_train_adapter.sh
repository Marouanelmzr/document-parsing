#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# ============================================================
# Retrain ONLY the winning adapter (r64_lr1e-4) from the P-LoRA sweep.
#   - single-adapter mode (no sweep, no pruning)
#   - merged checkpoint is scratch-only (vLLM eval), never tracked/pushed
#   - only best_adapter + sweep_summary.json go to DVC
# ============================================================

TRAIN_IMAGES="$PROJECT_ROOT/data/invoices/processed/splits/train/images"
TRAIN_LABELS="$PROJECT_ROOT/data/invoices/processed/splits/train/labels"
VAL_IMAGES="$PROJECT_ROOT/data/invoices/processed/splits/val/images"
VAL_LABELS="$PROJECT_ROOT/data/invoices/processed/splits/val/labels"

MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
MODE="${MODE:-single}"
TRIAL="${TRIAL:-r64_lr1e-4}"          # winner of the previous sweep

MICRO_BATCH=4
GRAD_ACCUM=4
EPOCHS=1

EVAL_EVERY=50
TRAIN_TIME_EVAL=20

RUN_BASE_EVAL="${RUN_BASE_EVAL:-false}"   # base acc already known: 0.8412
WANDB_PROJECT="${WANDB_PROJECT-fatura-plora}"
VLLM_EVAL_MAX="${VLLM_EVAL_MAX:-0}"
WANDB_RUN_NAME=""

# Optional fallback: also upload the adapter to the HF Hub (needs HF_TOKEN)
HF_ADAPTER_REPO="${HF_ADAPTER_REPO:-}"    # e.g. "yourname/fatura-qwen3vl-r64-adapter"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${MODE}_${TRIAL}_${TIMESTAMP}}"
OUT_ROOT="$PROJECT_ROOT/runs/${MODE}_${TIMESTAMP}"
MERGED_SCRATCH="/tmp/merged_${TIMESTAMP}"     # NOT inside OUT_ROOT -> never tracked
LOG_FILE_DIR="$PROJECT_ROOT/runs/train_logs"
mkdir -p "$LOG_FILE_DIR"
LOG_FILE="$LOG_FILE_DIR/run_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo " Invoice LoRA Training Run (single adapter)"
echo "========================================"
echo "Project root : $PROJECT_ROOT"
echo "Mode / trial : $MODE / $TRIAL"
echo "Model        : $MODEL_ID"
echo "Output       : $OUT_ROOT"
echo "W&B project  : ${WANDB_PROJECT:-<disabled>}"
echo "Started      : $(date)"
echo "========================================"

# ---- 0. env ----
echo ""; echo "[0/7] Setting up Python environment..."
if [ ! -d ".venv" ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip

# ---- 1. deps ----
echo ""; echo "[1/7] Installing dependencies..."
python -m pip install -r requirements.txt

# ---- 2. data ----
echo ""; echo "[2/7] Pulling dataset with DVC..."
dvc pull data/invoices/processed

# ---- 3. preflight ----
echo ""; echo "[3/7] Preflight checks..."
for d in "$TRAIN_IMAGES" "$TRAIN_LABELS" "$VAL_IMAGES" "$VAL_LABELS"; do
    [ -d "$d" ] || { echo "ERROR: expected directory not found: $d"; exit 1; }
done

TRAIN_COUNT=$(find "$TRAIN_IMAGES" -maxdepth 1 -type f | wc -l)
VAL_COUNT=$(find "$VAL_IMAGES" -maxdepth 1 -type f | wc -l)
echo "Train images: $TRAIN_COUNT | Val images: $VAL_COUNT"
if [ "$TRAIN_COUNT" -eq 0 ] || [ "$VAL_COUNT" -eq 0 ]; then
    echo "ERROR: train or val image directory is empty."; exit 1
fi

EFFECTIVE_BATCH=$((MICRO_BATCH * GRAD_ACCUM))
STEPS_PER_EPOCH=$((TRAIN_COUNT / EFFECTIVE_BATCH))
TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
echo "Effective batch: $EFFECTIVE_BATCH docs/step -> TOTAL_STEPS=$TOTAL_STEPS"

python - <<PY
from huggingface_hub import HfApi
try:
    HfApi().model_info("$MODEL_ID"); print("  OK   $MODEL_ID")
except Exception as e:
    print(f"  FAIL $MODEL_ID: {e}"); raise SystemExit(1)
PY

if [ -n "$WANDB_PROJECT" ]; then
    python -c "import wandb" 2>/dev/null || { echo "ERROR: wandb not installed."; exit 1; }
    if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
        echo "ERROR: not logged in to W&B."; exit 1
    fi
fi

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "Preflight checks passed."

# ---- 4. train ----
echo ""; echo "[4/7] Running training ($MODE / $TRIAL, $TOTAL_STEPS steps)..."
mkdir -p "$OUT_ROOT"

WANDB_ARGS=()
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME")
fi

TRAIN_ARGS=(
    --model-id "$MODEL_ID"
    --train-images "$TRAIN_IMAGES" --train-labels "$TRAIN_LABELS"
    --val-images "$VAL_IMAGES" --val-labels "$VAL_LABELS"
    --output-dir "$OUT_ROOT"
    --mode "$MODE"
    --micro-batch-size "$MICRO_BATCH"
    --grad-accum-steps "$GRAD_ACCUM"
    --total-steps "$TOTAL_STEPS"
    --eval-every "$EVAL_EVERY"
    --final-eval-examples "$TRAIN_TIME_EVAL"
    --save-merged                      # scratch copy for fast vLLM eval only
)
[ "$MODE" = "single" ] && TRAIN_ARGS+=(--trial "$TRIAL")

python -m src.training.train "${TRAIN_ARGS[@]}" ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}

if [ ! -f "$OUT_ROOT/best_adapter/adapter_model.safetensors" ]; then
    echo "ERROR: no adapter_model.safetensors in $OUT_ROOT/best_adapter -- not pushing."
    exit 1
fi
echo "Adapter size: $(du -sh "$OUT_ROOT/best_adapter" | cut -f1)"

# Move the merged checkpoint OUT of the tracked run folder.
if [ -d "$OUT_ROOT/merged" ]; then
    mv "$OUT_ROOT/merged" "$MERGED_SCRATCH"
fi

# ---- 4b. vLLM eval on scratch merged model ----
echo ""; echo "[4b/7] Full validation eval with vLLM ($VAL_COUNT examples)..."

run_vllm_eval() {   # $1 = model path/id, $2 = tag
    env -u PYTORCH_CUDA_ALLOC_CONF python -m src.training.vllm_eval \
        --model "$1" --tag "$2" \
        --val-images "$VAL_IMAGES" --val-labels "$VAL_LABELS" \
        --out-dir "$OUT_ROOT" --max-examples "$VLLM_EVAL_MAX" \
        ${WANDB_PROJECT:+--wandb-project "$WANDB_PROJECT"} \
        || echo "WARNING: vLLM eval ($2) failed -- continuing."
}

if [ -d "$MERGED_SCRATCH" ]; then
    run_vllm_eval "$MERGED_SCRATCH" finetuned
    [ "$RUN_BASE_EVAL" = true ] && run_vllm_eval "$MODEL_ID" base
    rm -rf "$MERGED_SCRATCH"          # never persisted
else
    echo "Skipping: no merged checkpoint to evaluate."
fi

# Safety: make sure nothing big is left in the tracked folder
rm -rf "$OUT_ROOT"/_tmp_* "$OUT_ROOT/merged"
echo "Tracked run size: $(du -sh "$OUT_ROOT" | cut -f1)"

# ---- 5. DVC ----
echo ""; echo "[5/7] Tracking adapter with DVC..."
dvc add "$OUT_ROOT"

# ---- 6. push ----
echo ""; echo "[6/7] Pushing results..."
# Targeted push: a bare `dvc push` would retry the old 8GB merged run too.
dvc push "${OUT_ROOT}.dvc"
echo "DVC push completed."

git add "${OUT_ROOT}.dvc" "$PROJECT_ROOT/runs/.gitignore"
git status
git commit -m "Add adapter run ${MODE} ${TRIAL} ${TIMESTAMP} (${TOTAL_STEPS} steps)" \
    || echo "Nothing new to commit."
git push
echo "Git push completed."

# Optional fallback: HF Hub (adapter only)
if [ -n "$HF_ADAPTER_REPO" ]; then
    echo "Uploading adapter to HF Hub: $HF_ADAPTER_REPO"
    python - <<PY || echo "WARNING: HF upload failed -- continuing."
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("$HF_ADAPTER_REPO", private=True, exist_ok=True)
api.upload_folder(folder_path="$OUT_ROOT/best_adapter", repo_id="$HF_ADAPTER_REPO")
print("HF upload OK")
PY
fi

# ---- 7. verify ----
echo ""
echo "========================================"
echo " RUN FINISHED"
echo "========================================"
echo "Adapter  : $OUT_ROOT/best_adapter"
echo "Summary  : $OUT_ROOT/sweep_summary.json"
echo ""
echo "DVC status vs remote (should report up to date):"
dvc status -c
git status
echo "Finished: $(date)"
echo " ADAPTER TRAINED AND PERSISTED. Safe to shut down the Pod."

git add "$LOG_FILE" 2>/dev/null \
    && git commit -m "Training log ${TIMESTAMP}" >/dev/null 2>&1 \
    && git push >/dev/null 2>&1 \
    || echo "(log commit skipped)"
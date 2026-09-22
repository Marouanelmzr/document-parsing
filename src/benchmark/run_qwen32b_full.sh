#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONVERTER_DIR="$PROJECT_ROOT/scripts"

cd "$PROJECT_ROOT"

# ============================================================
# Production run: qwen32b only, on the FULL 10k-document raw
# corpus (not the 400-doc benchmark sample). Structurally this
# is run_all.sh with:
#   - steps 0-4: same logic, different files/folders (full raw
#     images instead of fatura_sample_400, a freshly-built
#     manifest instead of the pre-existing 400-doc one)
#   - step 5: one model instead of three
#   - step 6: reconcile.py instead of evaluate.py -- this run
#     is producing a corrected dataset, not benchmarking, so we
#     merge ground truth into the model's output instead of
#     scoring the model against it
#   - steps 7-9: unchanged
# ============================================================


# ============================================================
# Configuration
# ============================================================

IMAGES_DIR="$PROJECT_ROOT/data/invoices/raw/images"
ANNOTATIONS_DIR="$PROJECT_ROOT/data/invoices/raw/Annotations/Original_Format"
MANIFEST="$PROJECT_ROOT/data/invoices/processed/fatura_full_manifest.csv"
EXPECTED_DOCS=10000

OUT_ROOT="$PROJECT_ROOT/benchmark_results_full"

CONVERTER_OUTPUT="$PROJECT_ROOT/data/invoices/processed/fatura_full_output.json"
GT_EVAL_DIR="$OUT_ROOT/ground_truth"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_ROOT/run_${TIMESTAMP}.log"

MODEL_KEYS=(qwen32b)


# ============================================================
# Create result directory
# ============================================================

mkdir -p "$OUT_ROOT"

exec > >(tee -a "$LOG_FILE") 2>&1


echo "========================================"
echo " Document Parsing - Full Corpus Production Run"
echo "========================================"
echo "Project root : $PROJECT_ROOT"
echo "Images       : $IMAGES_DIR"
echo "Annotations  : $ANNOTATIONS_DIR"
echo "Model        : ${MODEL_KEYS[*]}"
echo "Results      : $OUT_ROOT"
echo "Started      : $(date)"
echo "========================================"
echo ""
echo "NOTE: this is 10,000 documents vs the 400-doc benchmark sample --"
echo "expect roughly 25x the wall-clock time of the benchmark run. Make"
echo "sure this is running inside tmux (or equivalent) before walking away."
echo "========================================"


# ============================================================
# 0. Python environment
# ============================================================

echo ""
echo "[0/9] Setting up Python environment..."

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
echo "[1/9] Installing dependencies..."

python -m pip install -r requirements.txt


# ============================================================
# 2. Pull dataset
# ============================================================

echo ""
echo "[2/9] Pulling dataset with DVC..."

dvc pull


# ============================================================
# 3. Preflight checks -- everything here must pass BEFORE we
#    touch the GPU, since that's when billing starts to matter.
# ============================================================

echo ""
echo "[3/9] Preflight checks..."

# 3a. Dataset presence
if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: raw images directory not found:"
    echo "  $IMAGES_DIR"
    exit 1
fi

if [ ! -d "$ANNOTATIONS_DIR" ]; then
    echo "ERROR: annotations directory not found: $ANNOTATIONS_DIR"
    exit 1
fi

# 3b. Exact document count -- this is the check that stops an
#     accidentally-wrong subset from silently being billed for.
IMAGE_COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -type f \( \
    -iname "*.jpg" -o \
    -iname "*.jpeg" -o \
    -iname "*.png" \
\) | wc -l)

echo "Found $IMAGE_COUNT images in raw corpus (expected $EXPECTED_DOCS)"

if [ "$IMAGE_COUNT" -ne "$EXPECTED_DOCS" ]; then
    echo "ERROR: expected exactly $EXPECTED_DOCS images, found $IMAGE_COUNT."
    echo "Refusing to run inference -- update EXPECTED_DOCS in this script"
    echo "if the corpus size has genuinely changed, don't just bypass this."
    exit 1
fi

echo "Dataset validation successful ($IMAGE_COUNT documents)."

# 3c. Confirm the model repo resolves on the Hub before paying for it.
echo ""
echo "Checking model repo is reachable on Hugging Face Hub..."

python - <<PY
import sys
sys.path.insert(0, "src/benchmark")
from configs import CONFIGS
from huggingface_hub import HfApi

api = HfApi()
model_id = CONFIGS["qwen32b"].model_id
try:
    api.model_info(model_id)
    print(f"  OK   qwen32b      {model_id}")
except Exception as e:
    print(f"  FAIL qwen32b      {model_id}: {e}")
    print("\nERROR: fix model_id in src/benchmark/configs.py before running.")
    sys.exit(1)
PY

echo "Preflight checks passed."


# ============================================================
# 4. Build a full-corpus manifest and convert ground truth
# ============================================================

echo ""
echo "[4/9] Building manifest and converting FATURA annotations "
echo "      for the full $EXPECTED_DOCS-document corpus..."

python "$CONVERTER_DIR/build_full_manifest.py" \
    --images-dir "$IMAGES_DIR" \
    --output "$MANIFEST"

# fatura_converter.py accepts explicit paths (see scripts/fatura_converter.py);
# must still be run with cwd = scripts/ for its own internal imports.
( cd "$CONVERTER_DIR" && python fatura_converter.py \
    --manifest "$MANIFEST" \
    --annotations-dir "$ANNOTATIONS_DIR" \
    --output "$CONVERTER_OUTPUT" \
    --min-samples 5 )

if [ ! -f "$CONVERTER_OUTPUT" ]; then
    echo "ERROR: expected converter output not found:"
    echo "  $CONVERTER_OUTPUT"
    exit 1
fi

# evaluate.py's load_gt() (reused by reconcile.py) expects a DIRECTORY
# containing exactly one JSON file, not a file path.
rm -rf "$GT_EVAL_DIR"
mkdir -p "$GT_EVAL_DIR"
cp "$CONVERTER_OUTPUT" "$GT_EVAL_DIR/"

echo "Ground truth ready: $GT_EVAL_DIR"


# ============================================================
# 5. Run inference (the expensive part)
# ============================================================

echo ""
echo "[5/9] Running inference on $IMAGE_COUNT documents x ${#MODEL_KEYS[@]} model..."

FAILED_MODELS=()

for key in "${MODEL_KEYS[@]}"; do

    echo ""
    echo "========================================"
    echo " Running $key"
    echo "========================================"

    if python src/benchmark/worker.py \
        --model-key "$key" \
        --images-dir "$IMAGES_DIR" \
        --output-root "$OUT_ROOT"; then
        echo ""
        echo "Finished $key."
    else
        echo ""
        echo "WARNING: $key FAILED."
        FAILED_MODELS+=("$key")
    fi

done

if [ "${#FAILED_MODELS[@]}" -gt 0 ]; then
    echo ""
    echo "WARNING: the following models failed: ${FAILED_MODELS[*]}"
fi


# ============================================================
# 6. Reconcile qwen32b's output against ground truth
#    (replaces evaluate.py -- this run produces a corrected
#    dataset, it doesn't score a benchmark)
# ============================================================

echo ""
echo "[6/9] Reconciling qwen32b predictions with ground truth..."

if [[ " ${FAILED_MODELS[*]-} " == *" qwen32b "* ]]; then
    echo "WARNING: qwen32b failed in step 5 -- skipping reconciliation, "
    echo "there is nothing to reconcile."
else
    python src/benchmark/Reconcile.py \
        --predictions-dir "$OUT_ROOT/qwen32b" \
        --ground-truth-dir "$GT_EVAL_DIR" \
        --output-dir "$OUT_ROOT/qwen32b_final"
fi


# ============================================================
# 7. Track results with DVC
# ============================================================

echo ""
echo "[7/9] Tracking results with DVC..."

if [ -f "benchmark_results_full.dvc" ]; then
    echo "Updating existing benchmark_results_full.dvc..."
fi

dvc add benchmark_results_full

echo ""
echo "DVC tracking complete."


# ============================================================
# 8. Commit Git metadata and push DVC data
# ============================================================

echo ""
echo "[8/9] Committing metadata and pushing results..."

git add .
git status

git commit -m "Add full-corpus qwen32b results ${TIMESTAMP} (failed: ${FAILED_MODELS[*]:-none})" \
    || echo "Nothing new to commit."

git push

echo ""
echo "Git push completed."

echo ""
echo "Pushing artifacts to DVC remote..."

dvc push

echo ""
echo "DVC push completed."


# ============================================================
# 9. Final verification
# ============================================================

echo ""
echo "========================================"
echo " RUN FINISHED"
echo "========================================"

echo "Results:"
echo "$OUT_ROOT"

echo ""
echo "DVC status:"
dvc status

echo ""
echo "Git status:"
git status

echo ""
echo "Finished:"
date

echo "========================================"
if [ "${#FAILED_MODELS[@]}" -gt 0 ]; then
    echo " COMPLETED WITH FAILURES: ${FAILED_MODELS[*]}"
    echo " Investigate before shutting down -- there is no successful"
    echo " qwen32b output to fall back on in this run (single model)."
else
    echo " FULL CORPUS RUN COMPLETE -- QWEN32B SUCCEEDED"
    echo " RESULTS ARE PERSISTED."
    echo " Corrected final output: $OUT_ROOT/qwen32b_final/"
    echo " You can now safely shut down the Pod."
fi
echo "========================================"
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONVERTER_DIR="$PROJECT_ROOT/scripts"

cd "$PROJECT_ROOT"


# ============================================================
# Configuration
# ============================================================

# NOTE: point this at the 400-doc SAMPLE, not the full raw corpus.
SAMPLE_ROOT="$PROJECT_ROOT/data/invoices/raw/fatura_sample_400"
IMAGES_DIR="$SAMPLE_ROOT/images"
MANIFEST="$SAMPLE_ROOT/manifest.csv"
ANNOTATIONS_DIR="$PROJECT_ROOT/data/invoices/raw/Annotations/Original_Format"
EXPECTED_DOCS=400

OUT_ROOT="$PROJECT_ROOT/benchmark_results"

# scripts/fatura_converter.py hardcodes this exact output path (relative to
# its own location). If that script ever changes, update this line too.
CONVERTER_OUTPUT="$PROJECT_ROOT/data/invoices/processed/fatura_sample_400_output.json"
GT_EVAL_DIR="$OUT_ROOT/ground_truth"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_ROOT/run_${TIMESTAMP}.log"

MODEL_KEYS=(qwen32b mistral24b nuextract3)


# ============================================================
# Create result directory
# ============================================================

mkdir -p "$OUT_ROOT"

# Keep a complete copy of the terminal output.
exec > >(tee -a "$LOG_FILE") 2>&1


echo "========================================"
echo " Document Parsing - Full Benchmark Run"
echo "========================================"
echo "Project root : $PROJECT_ROOT"
echo "Images       : $IMAGES_DIR"
echo "Annotations  : $ANNOTATIONS_DIR"
echo "Results      : $OUT_ROOT"
echo "Started      : $(date)"
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
    echo "ERROR: sample image directory not found:"
    echo "  $IMAGES_DIR"
    echo "Did 'dvc pull' actually bring down data/invoices/raw/fatura_sample_400/?"
    echo "Run 'ls data/invoices/raw/' and update SAMPLE_ROOT in this script if the"
    echo "real folder name is different."
    exit 1
fi

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST"
    exit 1
fi

if [ ! -d "$ANNOTATIONS_DIR" ]; then
    echo "ERROR: annotations directory not found: $ANNOTATIONS_DIR"
    exit 1
fi

# 3b. Exact document count -- this is the check that stops an
#     accidental full-10000-image run from happening.
IMAGE_COUNT=$(find "$IMAGES_DIR" -type f \( \
    -iname "*.jpg" -o \
    -iname "*.jpeg" -o \
    -iname "*.png" \
\) | wc -l)

echo "Found $IMAGE_COUNT images in sample set (expected $EXPECTED_DOCS)"

if [ "$IMAGE_COUNT" -ne "$EXPECTED_DOCS" ]; then
    echo "ERROR: expected exactly $EXPECTED_DOCS images, found $IMAGE_COUNT."
    echo "Refusing to run inference -- this guards against silently billing"
    echo "for the full dataset instead of the intended $EXPECTED_DOCS-doc sample."
    exit 1
fi

MANIFEST_ROWS=$(($(wc -l < "$MANIFEST") - 1))
if [ "$MANIFEST_ROWS" -ne "$EXPECTED_DOCS" ]; then
    echo "ERROR: manifest.csv lists $MANIFEST_ROWS documents, expected $EXPECTED_DOCS."
    exit 1
fi

echo "Dataset validation successful ($IMAGE_COUNT documents)."

# 3c. Confirm all three model repos resolve on the Hub BEFORE
#     paying for any of them. Catches a bad/gated nuextract3 id
#     before qwen32b + mistral24b have already been run.
echo ""
echo "Checking model repos are reachable on Hugging Face Hub..."

python - <<PY
import sys
sys.path.insert(0, "src/benchmark")
from configs import CONFIGS
from huggingface_hub import HfApi

api = HfApi()
bad = []
for key in ("qwen32b", "mistral24b", "nuextract3"):
    model_id = CONFIGS[key].model_id
    try:
        api.model_info(model_id)
        print(f"  OK   {key:12s} {model_id}")
    except Exception as e:
        bad.append((key, model_id))
        print(f"  FAIL {key:12s} {model_id}: {e}")

if bad:
    print("\nERROR: fix the model_id(s) above in src/benchmark/configs.py "
          "before running -- do not proceed to inference.")
    sys.exit(1)
PY

# 3d. Flag the fp16-on-H100 leftover from configs.py so it's not
#     missed silently.
python - <<PY
import sys
sys.path.insert(0, "src/benchmark")
from configs import CONFIGS
dtype = CONFIGS["nuextract3"].dtype
if dtype != "bfloat16":
    print(f"\nWARNING: nuextract3 dtype is '{dtype}'. configs.py has a comment "
          f"saying to change this for H100 inference (the other two models use "
          f"bfloat16). Confirm this is intentional before continuing.\n")
PY

echo "Preflight checks passed."


# ============================================================
# 4. Convert FATURA ground truth (400-doc sample only)
# ============================================================

echo ""
echo "[4/9] Converting FATURA annotations for the 400-doc sample..."

# fatura_converter.py takes no CLI args and hardcodes paths relative to
# its own directory -- it must be run with cwd = scripts/.
( cd "$CONVERTER_DIR" && python fatura_converter.py )

if [ ! -f "$CONVERTER_OUTPUT" ]; then
    echo "ERROR: expected converter output not found:"
    echo "  $CONVERTER_OUTPUT"
    exit 1
fi

# evaluate.py expects --ground-truth-dir to be a DIRECTORY containing
# exactly one JSON file, not a file path. Stage a clean copy so it's
# never confused by other stray files that might exist in
# data/invoices/processed/.
rm -rf "$GT_EVAL_DIR"
mkdir -p "$GT_EVAL_DIR"
cp "$CONVERTER_OUTPUT" "$GT_EVAL_DIR/"

echo "Ground truth ready: $GT_EVAL_DIR"


# ============================================================
# 5. Run benchmark (the expensive part)
# ============================================================

echo ""
echo "[5/9] Running benchmark on $IMAGE_COUNT documents x ${#MODEL_KEYS[@]} models..."

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
        echo "WARNING: $key FAILED. Continuing with remaining models so"
        echo "whatever does succeed still gets evaluated and persisted."
        FAILED_MODELS+=("$key")
    fi

done

if [ "${#FAILED_MODELS[@]}" -gt 0 ]; then
    echo ""
    echo "WARNING: the following models failed: ${FAILED_MODELS[*]}"
fi


# ============================================================
# 6. Evaluate whatever completed
# ============================================================

echo ""
echo "[6/9] Evaluating benchmark results..."

if ! python src/benchmark/evaluate.py \
    --predictions-root "$OUT_ROOT" \
    --ground-truth-dir "$GT_EVAL_DIR" \
    --images-dir "$IMAGES_DIR"; then
    echo "WARNING: evaluation step failed. Raw predictions are still on disk"
    echo "under $OUT_ROOT -- do not shut down the pod until you've checked them."
fi


# ============================================================
# 7. Track benchmark results with DVC
# ============================================================

echo ""
echo "[7/9] Tracking benchmark results with DVC..."

if [ -f "benchmark_results.dvc" ]; then
    echo "Updating existing benchmark_results.dvc..."
fi

dvc add benchmark_results

echo ""
echo "DVC tracking complete."


# ============================================================
# 8. Commit Git metadata and push DVC data
# ============================================================

echo ""
echo "[8/9] Committing metadata and pushing results..."

git add .
git status

git commit -m "Add benchmark results ${TIMESTAMP} (failed models: ${FAILED_MODELS[*]:-none})" \
    || echo "Nothing new to commit."

git push

echo ""
echo "Git push completed."

echo ""
echo "Pushing benchmark artifacts to DVC remote..."

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
    echo " Results for the models that DID succeed are persisted."
    echo " Investigate the failure(s) above before re-running those models."
    echo " Safe to shut down once you've confirmed the git/dvc push above"
    echo " actually succeeded."
else
    echo " FULL RUN COMPLETE -- ALL 3 MODELS SUCCEEDED"
    echo " RESULTS ARE PERSISTED."
    echo " You can now safely shut down the Pod."
fi
echo "========================================"
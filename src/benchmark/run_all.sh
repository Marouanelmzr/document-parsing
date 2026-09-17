#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"


# ============================================================
# Configuration
# ============================================================

IMAGES_DIR="$PROJECT_ROOT/data/invoices/raw/fatura/images"
GT_RAW_DIR="$PROJECT_ROOT/data/invoices/raw/fatura/annotations"

OUT_ROOT="$PROJECT_ROOT/benchmark_results"
GT_PROCESSED="$OUT_ROOT/ground_truth/fatura_output.json"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_ROOT/run_${TIMESTAMP}.log"


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
echo "Raw GT       : $GT_RAW_DIR"
echo "Results      : $OUT_ROOT"
echo "Started      : $(date)"
echo "========================================"


# ============================================================
# 0. Python environment
# ============================================================

echo ""
echo "[0/8] Setting up Python environment..."

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
echo "[1/8] Installing dependencies..."

python -m pip install -r requirements.txt


# ============================================================
# 2. Pull dataset
# ============================================================

echo ""
echo "[2/8] Pulling dataset with DVC..."

dvc pull


# ============================================================
# 3. Validate dataset
# ============================================================

echo ""
echo "[3/8] Validating dataset..."

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Image directory does not exist:"
    echo "$IMAGES_DIR"
    exit 1
fi

if [ ! -d "$GT_RAW_DIR" ]; then
    echo "ERROR: Raw ground-truth directory does not exist:"
    echo "$GT_RAW_DIR"
    exit 1
fi

IMAGE_COUNT=$(find "$IMAGES_DIR" -type f \( \
    -iname "*.jpg" -o \
    -iname "*.jpeg" -o \
    -iname "*.png" \
\) | wc -l)

echo "Found $IMAGE_COUNT images"

if [ "$IMAGE_COUNT" -ne 10000 ]; then
    echo "ERROR: Expected exactly 10000 images, found $IMAGE_COUNT"
    exit 1
fi

echo "Dataset validation successful."


# ============================================================
# 4. Convert FATURA ground truth
# ============================================================

echo ""
echo "[4/8] Converting FATURA annotations..."

mkdir -p "$(dirname "$GT_PROCESSED")"

python src/data/fatura_converter.py \
    --input-dir "$GT_RAW_DIR" \
    --output "$GT_PROCESSED"

if [ ! -f "$GT_PROCESSED" ]; then
    echo "ERROR: FATURA converter did not create:"
    echo "$GT_PROCESSED"
    exit 1
fi

echo "Converted ground truth:"
echo "$GT_PROCESSED"


# ============================================================
# 5. Run benchmark
# ============================================================

echo ""
echo "[5/8] Running benchmark..."

for key in qwen32b mistral24b nuextract3; do

    echo ""
    echo "========================================"
    echo " Running $key"
    echo "========================================"

    python src/benchmark/worker.py \
        --model-key "$key" \
        --images-dir "$IMAGES_DIR" \
        --output-root "$OUT_ROOT"

    echo ""
    echo "Finished $key."

done


# ============================================================
# 6. Evaluate
# ============================================================

echo ""
echo "[6/8] Evaluating benchmark results..."

python src/benchmark/evaluate.py \
    --predictions-root "$OUT_ROOT" \
    --ground-truth-dir "$GT_PROCESSED" \
    --images-dir "$IMAGES_DIR"


# ============================================================
# 7. Track benchmark results with DVC
# ============================================================

echo ""
echo "[7/8] Tracking benchmark results with DVC..."

# Remove an old DVC file if this is a rerun and the result
# directory is being re-created.
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
echo "[8/8] Committing metadata and pushing results..."

git add .

git status

git commit -m "Add benchmark results ${TIMESTAMP}"

git push

echo ""
echo "Git push completed."

echo ""
echo "Pushing benchmark artifacts to DVC remote..."

dvc push

echo ""
echo "DVC push completed."


# ============================================================
# Final verification
# ============================================================

echo ""
echo "========================================"
echo " FULL RUN COMPLETE"
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
echo " RESULTS ARE PERSISTED"
echo " You can now safely shut down the Pod."
echo "========================================"
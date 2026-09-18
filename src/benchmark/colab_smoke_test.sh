#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Colab smoke test -- proves the pipeline mechanics work
# (sampling -> ground-truth conversion -> worker -> evaluate)
# on a small, cheap slice before you spend real money on H100.
#
# Differences from src/benchmark/run_all.sh:
#   - Builds its own small N-doc sample on the fly (the H100 script
#     assumes the 400-doc sample already exists in DVC; here it doesn't).
#   - Runs ONE lightweight model (qwen8b_t4) instead of the 3 real
#     24B-32B FP8 models -- those don't fit on Colab GPUs.
#   - No `dvc add` / `git commit` / `git push` / `dvc push`. This is a
#     throwaway test; nothing here should land in your production
#     benchmark_results history.
#   - No venv (Colab's runtime is already an isolated, disposable
#     container -- creating one just adds install time for nothing).
#
# Usage (from a Colab cell, after `%cd document-parsing`):
#   !bash scripts/colab_smoke_test.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"


# ============================================================
# Configuration
# ============================================================

N_SAMPLES=20
MODEL_KEYS=(qwen8b_t4)   # add more keys here ONLY if your Colab GPU can
                         # actually fit them (e.g. an A100 might just about
                         # handle one 24B FP8 model -- a T4 cannot).

RAW_IMAGES_DIR="$PROJECT_ROOT/data/invoices/raw/images"
ANNOTATIONS_DIR="$PROJECT_ROOT/data/invoices/raw/Annotations/Original_Format"

SAMPLE_ROOT="$PROJECT_ROOT/data/invoices/raw/fatura_sample_colab"
IMAGES_DIR="$SAMPLE_ROOT/images"
MANIFEST="$SAMPLE_ROOT/manifest.csv"

OUT_ROOT="$PROJECT_ROOT/benchmark_results_colab"
GT_EVAL_DIR="$OUT_ROOT/ground_truth"
GT_OUTPUT="$OUT_ROOT/fatura_sample_colab_output.json"


echo "========================================"
echo " Colab Smoke Test -- Pipeline Mechanics"
echo "========================================"
echo "Project root : $PROJECT_ROOT"
echo "Sample size  : $N_SAMPLES documents"
echo "Model(s)     : ${MODEL_KEYS[*]}"
echo "Results      : $OUT_ROOT"
echo "Started      : $(date)"
echo "========================================"


# ============================================================
# 1. Install dependencies
# ============================================================

echo ""
echo "[1/7] Installing dependencies..."

python -m pip install -q -r requirements.txt


# ============================================================
# 2. Pull dataset (read-only -- this script never pushes)
# ============================================================

echo ""
echo "[2/7] Pulling dataset with DVC..."

dvc pull

if [ ! -d "$RAW_IMAGES_DIR" ]; then
    echo "ERROR: raw images dir not found: $RAW_IMAGES_DIR"
    echo "Run 'ls data/invoices/raw/' and update RAW_IMAGES_DIR in this script"
    echo "if the real folder name is different."
    exit 1
fi

if [ ! -d "$ANNOTATIONS_DIR" ]; then
    echo "ERROR: annotations dir not found: $ANNOTATIONS_DIR"
    exit 1
fi


# ============================================================
# 3. Build a small N-doc sample on the fly
#    (the production pipeline expects pre-sampled data; on Colab
#    we generate it fresh each run instead of relying on the
#    400-doc set from DVC)
# ============================================================

echo ""
echo "[3/7] Building a fresh $N_SAMPLES-document sample..."

rm -rf "$SAMPLE_ROOT"

python src/data/sample_fatura_dataset.py \
    --images-dir "$RAW_IMAGES_DIR" \
    --output-dir "$SAMPLE_ROOT" \
    --n-samples "$N_SAMPLES" \
    --no-zip

IMAGE_COUNT=$(find "$IMAGES_DIR" -type f \( \
    -iname "*.jpg" -o \
    -iname "*.jpeg" -o \
    -iname "*.png" \
\) | wc -l)

if [ "$IMAGE_COUNT" -ne "$N_SAMPLES" ]; then
    echo "ERROR: expected $N_SAMPLES sampled images, found $IMAGE_COUNT."
    exit 1
fi

echo "Sample ready: $IMAGE_COUNT documents in $IMAGES_DIR"


# ============================================================
# 4. Preflight: confirm the test model resolves on the Hub
#    before the GPU starts loading anything
# ============================================================

echo ""
echo "[4/7] Checking model repo(s) are reachable on Hugging Face Hub..."

MODEL_KEYS_PY="$(printf '"%s",' "${MODEL_KEYS[@]}")"

python - <<PY
import sys
sys.path.insert(0, "src/benchmark")
from configs import CONFIGS
from huggingface_hub import HfApi

api = HfApi()
bad = []
for key in ($MODEL_KEYS_PY):
    model_id = CONFIGS[key].model_id
    try:
        api.model_info(model_id)
        print(f"  OK   {key:12s} {model_id}")
    except Exception as e:
        bad.append((key, model_id))
        print(f"  FAIL {key:12s} {model_id}: {e}")

if bad:
    print("\nERROR: fix the model_id(s) above in src/benchmark/configs.py.")
    sys.exit(1)
PY


# ============================================================
# 5. Convert ground truth for the sample
# ============================================================

echo ""
echo "[5/7] Converting FATURA annotations for the sample..."

python scripts/fatura_converter.py \
    --manifest "$MANIFEST" \
    --annotations-dir "$ANNOTATIONS_DIR" \
    --output "$GT_OUTPUT" \
    --min-samples 2

# With only $N_SAMPLES docs spread across FATURA's templates, most
# templates will fall below the default min_samples=5, so pattern
# induction is skipped for them and fields fall back to fuzzy/
# containment matching (see the printed warnings above). That's fine
# for a mechanics smoke test -- don't read the ground-truth *quality*
# here as representative of the real 400-doc run.

if [ ! -f "$GT_OUTPUT" ]; then
    echo "ERROR: expected converter output not found: $GT_OUTPUT"
    exit 1
fi

# evaluate.py expects --ground-truth-dir to be a directory containing
# exactly one JSON file.
rm -rf "$GT_EVAL_DIR"
mkdir -p "$GT_EVAL_DIR"
cp "$GT_OUTPUT" "$GT_EVAL_DIR/"

echo "Ground truth ready: $GT_EVAL_DIR"


# ============================================================
# 6. Run the benchmark
# ============================================================

echo ""
echo "[6/7] Running benchmark on $IMAGE_COUNT documents x ${#MODEL_KEYS[@]} model(s)..."

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
        echo "FAILED: $key"
        FAILED_MODELS+=("$key")
    fi

done


# ============================================================
# 7. Evaluate
# ============================================================

echo ""
echo "[7/7] Evaluating..."

python src/benchmark/evaluate.py \
    --predictions-root "$OUT_ROOT" \
    --ground-truth-dir "$GT_EVAL_DIR" \
    --images-dir "$IMAGES_DIR"


# ============================================================
# Final verdict
# ============================================================

echo ""
echo "========================================"
if [ "${#FAILED_MODELS[@]}" -gt 0 ] || [ ! -f "$OUT_ROOT/evaluation.json" ]; then
    echo " SMOKE TEST FAILED"
    echo " Failed model(s): ${FAILED_MODELS[*]:-none, but evaluation.json missing}"
    echo " Fix this before trusting the H100 run to work."
else
    echo " SMOKE TEST PASSED"
    echo " Pipeline mechanics (sample -> convert -> worker -> evaluate)"
    echo " ran end to end cleanly. run_all.sh follows the exact same code"
    echo " path (worker.py / evaluate.py / fatura_converter.py) with the"
    echo " real 3 models on the real 400-doc sample -- this is your best"
    echo " available signal it will run without erroring on H100."
    echo ""
    echo " Results: $OUT_ROOT/evaluation.json"
fi
echo "========================================"
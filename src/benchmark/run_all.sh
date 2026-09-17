#!/usr/bin/env bash
# Usage: ./run_all.sh /path/to/400_sampled_images /path/to/fatura_ground_truth [output_root]
set -euo pipefail

IMAGES_DIR=${1:?images dir required}
GT_DIR=${2:?ground-truth dir required}
OUT_ROOT=${3:-./bench_out}

#for key in qwen32b mistral24b nuextract3; do
for key in qwen8b_t4 nuextract3; do
  echo "=== [$(date +%H:%M:%S)] running $key ==="
  # each model is its own process -> GPU is fully released when it exits,
  # so the next model always starts with a clean 94GB card.
  python worker.py --model-key "$key" --images-dir "$IMAGES_DIR" --output-root "$OUT_ROOT"
done

echo "=== [$(date +%H:%M:%S)] all models done, evaluating against ground truth ==="
python evaluate.py --predictions-root "$OUT_ROOT" --ground-truth-dir "$GT_DIR" --images-dir "$IMAGES_DIR"
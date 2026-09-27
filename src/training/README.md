# LoRA / P-LoRA training for Qwen3-VL-4B invoice extraction

Fine-tunes `Qwen/Qwen3-VL-4B-Instruct` on the exact task your benchmark
already runs against it (`src/benchmark/adapters.py`'s `VISION_CHAT_PROMPT`
+ the `InvoiceExtraction` schema in `common.py`), on a single H100 80-94GB.

## Install

```bash
pip install -r requirements-train.txt
```

`requirements-train.txt` adds `torch`, `peft`, `accelerate`, `flash-attn`
(optional but recommended) on top of the repo's existing `requirements.txt`
(vLLM stays for inference/benchmarking; this uses plain `transformers` for
training, since vLLM's LoRA path is inference-only).

## Data layout expected

Whatever `scripts/split_dataset.py` produces:

```
<split>/images/TemplateN_InstanceM.jpg
<split>/labels/TemplateN_InstanceM.json   # {"fields": {...}, ...}
```

The `"fields"` object must match `InvoiceFields` in `src/benchmark/common.py`
-- i.e. either human-annotated ground truth, or (as your repo currently
does) distilled labels from a larger teacher model (`qwen32b`'s output).

## Run it

Quick single-adapter baseline (no sweep):

```bash
cd training
python train.py \
    --train-images ../data/invoices/processed/splits/train/images \
    --train-labels ../data/invoices/processed/splits/train/labels \
    --val-images   ../data/invoices/processed/splits/val/images \
    --val-labels   ../data/invoices/processed/splits/val/labels \
    --mode single --trial r32_lr2e-4 \
    --micro-batch-size 6 --grad-accum-steps 4 \
    --total-steps 1500 --output-dir runs/single_r32
```

P-LoRA sweep (several LoRA configs trained concurrently on one shared
backbone, ASHA-pruned, only the winner saved):

```bash
python train.py \
    --train-images ... --train-labels ... --val-images ... --val-labels ... \
    --mode plora \
    --micro-batch-size 6 --grad-accum-steps 4 \
    --total-steps 2000 --eval-every 100 --rungs 0.25 0.5 0.75 \
    --output-dir runs/plora_sweep --save-merged
```

Output: `runs/.../best_adapter/` (a normal PEFT adapter directory -- load it
directly with vLLM's `--enable-lora`/`LoRARequest`, no merge needed for
serving), plus `sweep_summary.json` with every trial's trajectory and the
selection rationale. `--save-merged` additionally writes a full merged
checkpoint if you'd rather serve without a LoRA flag at all.

## Design summary

**Dataloading** (`dataset.py`)
- `LengthGroupedSampler` buckets samples by a cheap length estimate (image
  header size + label file size, no decode/tokenize) so batches contain
  similarly-sized examples -- minimal padding waste, which is where naive
  VLM dataloaders bleed GPU time.
- Qwen's processor already concatenates `pixel_values` across a batch as
  flat patch sequences (`image_grid_thw` says how many patches each image
  contributed) -- so batching images is a concat, never a pad-to-max-size
  operation. `collate_fn` just does `torch.cat`.
- Labels are masked (`-100`) over the prompt+image tokens; loss is computed
  only on the assistant's JSON completion.
- `num_workers` + `persistent_workers` + `prefetch_factor=4` keep a queue of
  ready batches ahead of the GPU so it's never waiting on PIL/tokenization.

**LoRA** (`train.py`, `lora_trials.py`)
- Backbone loaded once in bf16, frozen, flash-attention-2, gradient
  checkpointing (non-reentrant) -- trades ~30% more compute for much lower
  activation memory, which is the right trade on a VRAM-bound single-GPU
  setup.
- LoRA targets the language-model projections by default (`q/k/v/o_proj`,
  `gate/up/down_proj`); flip `include_vision=True` on a `TrialConfig` if
  eval shows visual-grounding errors rather than field-mapping errors.
- `fused=True` AdamW, only over LoRA parameters (the frozen backbone needs
  no optimizer state at all).

**P-LoRA sweep** (`multi_lora_trainer.py`)
- Several `TrialConfig`s (rank/alpha/dropout/lr) become named PEFT adapters
  on the *same* base model instance -- one 8GB backbone shared, not N copies.
- Each optimizer step runs every live adapter on the *same* micro-batch(es)
  in round-robin, so comparisons across trials are apples-to-apples and the
  GPU is never idle waiting for a trial-specific dataloader.
- ASHA-style pruning at `--rungs` (fractions of `--total-steps`): cheap val
  loss ranks live trials, the bottom `1 - keep_fraction` are dropped
  (`model.delete_adapter`, freeing their optimizer state), survivors get the
  rest of the budget.
- Final selection uses real field-level JSON accuracy (`metrics.py`,
  matching `InvoiceFields`'s structure) via `.generate()` on the survivors
  only -- this expensive step never runs on trials already pruned by loss.
- Only the winning adapter is saved; `sweep_summary.json` keeps the full
  trajectory for audit/reproducibility.

**What this is not**: it doesn't batch multiple adapters into one fused
kernel call (that needs custom grouped-GEMM, à la vLLM/Punica/S-LoRA-style
multi-LoRA *inference* serving, not training). What you get here is: one
model load, shared data pipeline, and idle-free interleaving -- which is
most of the realistic win for a training-time hyperparameter sweep on a
single GPU.

## Tuning knobs if you want to push utilization further

- Raise `--micro-batch-size` until `nvidia-smi` shows you're near the VRAM
  ceiling (FATURA invoice pages are usually well under the 2048-image-token
  cap in `configs.py`, so there's often room past the default of 6).
- If `nvidia-smi dmon` shows GPU utilization dropping between steps (not
  pinned near 100%), the bottleneck is the CPU-side pipeline -- raise
  `--num-workers` before touching batch size.
- `torch.compile(model)` is not wired in here because Qwen3-VL's dynamic
  image-token counts make it prone to frequent recompiles unless you also
  bucket by exact token count (stricter than the length-bucketing done
  here); worth revisiting if profiling shows kernel-launch overhead
  dominating at your batch size.

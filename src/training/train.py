"""LoRA fine-tuning entrypoint for Qwen3-VL-4B-Instruct on the invoice
extraction task, tuned for a single H100 80-94GB.

Usage (single adapter, no HPO):
    python train.py \
        --train-images ../data/invoices/processed/splits/train/images \
        --train-labels ../data/invoices/processed/splits/train/labels \
        --val-images   ../data/invoices/processed/splits/val/images \
        --val-labels   ../data/invoices/processed/splits/val/labels \
        --mode single --trial r32_lr2e-4 \
        --output-dir runs/single_r32

Usage (P-LoRA sweep -- several hyperparameter configs trained concurrently
on one shared backbone, ASHA-pruned, best one kept):
    python train.py \
        --train-images ... --train-labels ... --val-images ... --val-labels ... \
        --mode plora --total-steps 3000 --eval-every 100 \
        --output-dir runs/plora_sweep

GPU-utilization defaults baked in below (override via flags if your images
run smaller/larger than FATURA's, or if you have headroom to push further):
  - bf16 everywhere, flash_attention_2, gradient checkpointing (non-reentrant)
  - fused AdamW (LoRA params only -- backbone is frozen, no optimizer state for it)
  - TF32 matmuls enabled
  - length-bucketed sampler (see dataset.py) to minimize padding waste
  - persistent_workers + prefetch so the CPU pipeline stays ahead of the GPU
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

from dataset import InvoiceVLDataset, LengthGroupedSampler, make_collate_fn
from lora_trials import default_trials
from multi_lora_trainer import MultiLoRATrainer, TrainConfig


def build_loaders(args, processor):
    train_ds = InvoiceVLDataset(args.train_images, args.train_labels, processor,
                                 max_length=args.max_length)
    val_ds = InvoiceVLDataset(args.val_images, args.val_labels, processor,
                               max_length=args.max_length)
    collate = make_collate_fn(processor.tokenizer.pad_token_id)

    train_sampler = LengthGroupedSampler(train_ds.lengths, batch_size=args.micro_batch_size,
                                          shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.micro_batch_size, sampler=train_sampler,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.micro_batch_size, shuffle=False,
        collate_fn=collate, num_workers=max(1, args.num_workers // 2), pin_memory=True,
    )
    return train_loader, val_loader


def load_model_and_processor(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    except (ImportError, ValueError):
        print("[warn] flash-attention-2 unavailable, falling back to sdpa")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa")

    model.config.use_cache = False  # required alongside gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model, processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--train-images", type=Path, required=True)
    ap.add_argument("--train-labels", type=Path, required=True)
    ap.add_argument("--val-images", type=Path, required=True)
    ap.add_argument("--val-labels", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)

    ap.add_argument("--mode", choices=["single", "plora"], default="plora")
    ap.add_argument("--trial", default="r32_lr2e-4",
                     help="trial name to use in --mode single (see lora_trials.py)")

    ap.add_argument("--micro-batch-size", type=int, default=4,
                     help="per-step batch size fed to the model. With max-length "
                          "4096 and a 4B model in bf16 + grad checkpointing, 4-8 "
                          "comfortably fits an H100-80GB; raise it and watch "
                          "nvidia-smi if your images run smaller than FATURA's.")
    ap.add_argument("--grad-accum-steps", type=int, default=4,
                     help="effective_batch = micro-batch-size * grad-accum-steps, "
                          "at no extra VRAM cost. --total-steps counts OPTIMIZER "
                          "steps (each spans this many micro-batches).")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=8)

    ap.add_argument("--total-steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--rungs", type=float, nargs="*", default=[0.25, 0.5, 0.75])
    ap.add_argument("--keep-fraction", type=float, default=0.5)
    ap.add_argument("--final-eval-examples", type=int, default=40)
    ap.add_argument("--save-merged", action="store_true",
                     help="also save a merged (base+adapter) checkpoint for vLLM serving without --lora")

    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, processor = load_model_and_processor(args.model_id)
    train_loader, val_loader = build_loaders(args, processor)

    trials = default_trials()
    if args.mode == "single":
        trials = [t for t in trials if t.name == args.trial]
        if not trials:
            raise SystemExit(f"unknown trial '{args.trial}', see lora_trials.py for names")

    cfg = TrainConfig(
        total_steps=args.total_steps,
        eval_every=args.eval_every,
        rungs=args.rungs if args.mode == "plora" else [],
        keep_fraction=args.keep_fraction,
        grad_accum_steps=args.grad_accum_steps,
        final_eval_examples=args.final_eval_examples,
        ckpt_dir=args.output_dir,
    )

    trainer = MultiLoRATrainer(model, processor, trials, train_loader, val_loader, cfg)
    best = trainer.run()
    trainer.save_best_adapter(best, args.output_dir / "best_adapter")

    if args.save_merged:
        trainer.merge_and_save(best, args.model_id, args.output_dir / "merged")


if __name__ == "__main__":
    main()

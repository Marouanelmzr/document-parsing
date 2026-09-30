"""Full-validation eval with vLLM (offline LLM API). Runs in its own process
so the GPU is free. Same prompts / lenient parsing / scorer as the HF eval in
multi_lora_trainer.py, so numbers are comparable.

    python -m src.training.vllm_eval --model runs/x/merged --tag finetuned \
        --val-images ... --val-labels ... --out-dir runs/x [--wandb-project P]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged checkpoint dir or HF model id")
    ap.add_argument("--tag", required=True, help="e.g. finetuned / base")
    ap.add_argument("--val-images", type=Path, required=True)
    ap.add_argument("--val-labels", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=4096, help="dataset max_length (as in training)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=1600)
    ap.add_argument("--gpu-mem", type=float, default=0.95)
    ap.add_argument("--chunk", type=int, default=250, help="examples held in RAM / submitted per generate call")
    ap.add_argument("--max-examples", type=int, default=0, help="0 = all")
    ap.add_argument("--wandb-project", default="")
    args = ap.parse_args()

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from src.training.dataset import InvoiceVLDataset
    from src.training.multi_lora_trainer import _parse_lenient, _safe_score

    processor = AutoProcessor.from_pretrained(args.model)
    ds = InvoiceVLDataset(args.val_images, args.val_labels, processor, max_length=args.max_length)
    n_total = len(ds) if args.max_examples <= 0 else min(args.max_examples, len(ds))
    print(f"[vllm-eval:{args.tag}] {n_total} examples, model={args.model}")

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=True,
        max_num_seqs=128,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = args.out_dir / f"vllm_preds_{args.tag}.jsonl"
    scores, floors, parse_fail = [], [], 0
    t0 = time.time()

    with open(preds_path, "w") as fout:
        for s in range(0, n_total, args.chunk):
            idxs = list(range(s, min(s + args.chunk, n_total)))
            inputs, golds = [], []
            for i in idxs:
                text, image, gold = ds.prompt_and_gold(i)
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                inputs.append({"prompt": text, "multi_modal_data": {"image": image}})
                golds.append(gold)

            outs = llm.generate(inputs, sp, use_tqdm=True)
            for i, gold, o in zip(idxs, golds, outs):
                raw = o.outputs[0].text
                pred = _parse_lenient(raw)
                parse_fail += pred is None
                sc = _safe_score(pred, gold)
                scores.append(sc)
                floors.append(_safe_score({}, gold))
                fout.write(json.dumps({"idx": i, "score": sc, "raw": raw}, ensure_ascii=False) + "\n")
            print(f"[vllm-eval:{args.tag}] {len(scores)}/{n_total} running acc={sum(scores)/len(scores):.4f}")

    n = max(1, len(scores))
    result = {
        "tag": args.tag, "model": str(args.model), "n": len(scores),
        "field_accuracy": sum(scores) / n,
        "empty_pred_floor": sum(floors) / n,
        "parse_fail": parse_fail,
        "seconds": time.time() - t0,
    }
    print(f"[vllm-eval:{args.tag}] {json.dumps(result)}")
    with open(args.out_dir / f"vllm_eval_{args.tag}.json", "w") as f:
        json.dump(result, f, indent=2)

    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, job_type="eval",
                         name=f"{args.out_dir.name}_eval_{args.tag}", config={"model": str(args.model)})
        for k, v in result.items():
            run.summary[k] = v
        run.finish()


if __name__ == "__main__":
    main()
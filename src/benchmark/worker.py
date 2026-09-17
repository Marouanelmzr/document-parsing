"""Runs exactly ONE model end to end, then exits -- deliberately a fresh
process per invocation (see run_all.sh) so vLLM's CUDA context is fully
torn down by the OS before the next model loads. Don't call this in a loop
inside one long-lived Python process; instantiate LLM() twice in the same
interpreter and you risk leftover GPU memory fragmenting/OOMing the next
model.
"""
import argparse
from pathlib import Path

from common import run_extraction
from configs import (
    CONFIGS,
    build_llm_kwargs,
    build_sampling_params,
    build_conversation_fn,
    build_chat_template_kwargs,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=list(CONFIGS))
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--output-root", default=Path("./bench_out"), type=Path)
    args = ap.parse_args()

    cfg = CONFIGS[args.model_key]
    output_dir = args.output_root / cfg.key

    run_extraction(
        images_dir=args.images_dir,
        output_dir=output_dir,
        llm_kwargs=build_llm_kwargs(cfg),
        sampling_params=build_sampling_params(cfg),
        build_conversation=build_conversation_fn(cfg),
        chat_template_kwargs=build_chat_template_kwargs(cfg),
        batch_size=cfg.batch_size,
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
    )


if __name__ == "__main__":
    main()
"""One ModelConfig per model. This is the only file you edit to tune
per-model resource knobs (they don't change model *output*, only
throughput/memory, since each model has the whole GPU to itself in turn).

Notes on the two model-specific quirks baked in below:
- Mistral-Small needs tokenizer_mode/config_format/load_format="mistral" in
  vLLM (its checkpoint isn't a plain HF safetensors-mode repo). Without
  these three flags LLM() fails to load it.
- NuExtract needs trust_remote_code=True and its own chat_template_kwargs
  (schema goes in the template, not the prompt text).
"""
import json
from dataclasses import dataclass, field
from typing import Literal

from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from common import InvoiceExtraction, PATCH_PIXELS
from adapters import (
    build_vision_chat_conversation,
    build_nuextract_conversation,
    NUEXTRACT_TEMPLATE,
    NUEXTRACT_INSTRUCTIONS,
)


@dataclass
class ModelConfig:
    key: str
    model_id: str
    family: Literal["vision_chat", "nuextract"]
    dtype: str = "bfloat16"          # compute dtype; FP8 weights are auto-detected from the checkpoint
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    batch_size: int = 400            # python-side chunking only, not the real vLLM inference batch
    min_pixels: int = 256 * PATCH_PIXELS
    max_pixels: int = 2048 * PATCH_PIXELS
    max_new_tokens: int = 1500
    temperature: float = 0.0         # kept identical across models -- deterministic, fair comparison
    guided_decoding_backend: str = "xgrammar"
    enable_thinking: bool = False    # only read for the nuextract family
    extra_llm_kwargs: dict = field(default_factory=dict)


CONFIGS: dict[str, ModelConfig] = {
    "qwen32b": ModelConfig(
        key="qwen32b",
        model_id="Qwen/Qwen3-VL-32B-Instruct-FP8",
        family="vision_chat",
        max_model_len=8192,
        gpu_memory_utilization=0.90,   # alone on the card -- give it almost all of it
        batch_size=400,                # 400 images fit fine as PIL objects in one go
    ),
    "mistral24b": ModelConfig(
        key="mistral24b",
        model_id="RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8",
        family="vision_chat",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        batch_size=400,
        extra_llm_kwargs={
            "tokenizer_mode": "mistral",
            "config_format": "mistral",
            "load_format": "mistral",
        },
    ),
    "nuextract3": ModelConfig(
        key="nuextract3",
        model_id="numind/NuExtract3",  # confirm exact repo id/size variant before running
        family="nuextract",
        dtype="float16", # remove for H100 inference
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        batch_size=100,
        extra_llm_kwargs={"trust_remote_code": True},
    ),
    "qwen8b_t4": ModelConfig(
        key="qwen8b_t4",
        model_id="cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit",
        family="vision_chat",
        dtype="float16",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        batch_size=100,
        max_new_tokens=1000,
        extra_llm_kwargs={},
    ),
}


def build_llm_kwargs(cfg: ModelConfig) -> dict:
    kwargs = dict(
        model=cfg.model_id,
        dtype=cfg.dtype,
        limit_mm_per_prompt={"image": 1},
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enforce_eager=True,
        structured_outputs_config={"backend": cfg.guided_decoding_backend},
        enable_prefix_caching=True,
        mm_processor_kwargs={"min_pixels": cfg.min_pixels, "max_pixels": cfg.max_pixels},
    )
    kwargs.update(cfg.extra_llm_kwargs)
    return kwargs


def build_sampling_params(cfg: ModelConfig) -> SamplingParams:
    return SamplingParams(
        temperature=cfg.temperature,
        max_tokens=cfg.max_new_tokens,
        structured_outputs=StructuredOutputsParams(json=InvoiceExtraction.model_json_schema()),
    )


def build_conversation_fn(cfg: ModelConfig):
    return build_vision_chat_conversation if cfg.family == "vision_chat" else build_nuextract_conversation


def build_chat_template_kwargs(cfg: ModelConfig):
    if cfg.family != "nuextract":
        return None
    return {
        "template": json.dumps(NUEXTRACT_TEMPLATE, separators=(",", ":"), ensure_ascii=False),
        "instructions": NUEXTRACT_INSTRUCTIONS,
        "enable_thinking": cfg.enable_thinking,
    }
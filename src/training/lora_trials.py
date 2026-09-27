"""Hyperparameter trial definitions for the P-LoRA sweep.

Each TrialConfig becomes one named PEFT adapter living on the *same* frozen
Qwen3-VL-4B backbone. Since the backbone (~8GB in bf16) is loaded once and
shared, running N trials costs N adapters' worth of LoRA params + optimizer
state (tens of MB each) instead of N full model copies -- this is what makes
it feasible to explore several ranks/LRs concurrently on one H100 instead of
one-at-a-time.
"""
from dataclasses import dataclass, field
from typing import List


# LM-only target modules keep the sweep cheap and is almost always enough
# for a structured-extraction task like this one (the vision tower doesn't
# need to change to get better at *reading* invoices it can already see
# fine -- the failure mode this fine-tune targets is field-assignment /
# formatting, which lives in the language model). Flip
# `include_vision=True` on a TrialConfig if you find the model is making
# visual-grounding mistakes (e.g. misreading digits) rather than
# field-mapping mistakes.
LM_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
VISION_TARGET_MODULES = ["qkv", "proj"]  # Qwen3-VL vision block attention/mlp proj names


@dataclass
class TrialConfig:
    name: str
    r: int
    lora_alpha: int
    lora_dropout: float
    lr: float
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    include_vision: bool = False
    target_modules: List[str] = field(default_factory=lambda: list(LM_TARGET_MODULES))

    def peft_target_modules(self) -> List[str]:
        return self.target_modules + (VISION_TARGET_MODULES if self.include_vision else [])


def default_trials() -> List[TrialConfig]:
    """A sensible starting sweep: rank and LR are the two knobs that matter
    most for LoRA on a 4B model; alpha is kept at 2*r (a common default that
    keeps the effective LoRA scale roughly rank-independent)."""
    return [
        TrialConfig(name="r16_lr2e-4", r=16, lora_alpha=32, lora_dropout=0.05, lr=2e-4),
        TrialConfig(name="r16_lr1e-4", r=16, lora_alpha=32, lora_dropout=0.05, lr=1e-4),
        TrialConfig(name="r32_lr2e-4", r=32, lora_alpha=64, lora_dropout=0.05, lr=2e-4),
        TrialConfig(name="r32_lr1e-4", r=32, lora_alpha=64, lora_dropout=0.05, lr=1e-4, weight_decay=0.01),
        TrialConfig(name="r64_lr1e-4", r=64, lora_alpha=128, lora_dropout=0.1, lr=1e-4, weight_decay=0.01),
    ]

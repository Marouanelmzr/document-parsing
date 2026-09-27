"""Field-level accuracy against the InvoiceExtraction schema (src/benchmark/common.py).

Used only at final adapter selection time (expensive: requires .generate()),
not during the cheap ASHA pruning rungs (which use val loss instead).
"""
from __future__ import annotations

import json
import math
from typing import Any


def _leaves(obj: Any, path: str = ""):
    """Yields (path, value) for every scalar leaf in a nested dict/list,
    matching InvoiceFields' shape (dicts, lists of dicts, scalars)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def _norm(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}" if not math.isnan(v) else ""
    return str(v).strip().lower()


def field_accuracy(pred: dict, gold: dict) -> float:
    """Fraction of gold leaf fields whose (normalized) value matches the
    prediction's value at the same path. A missing path in `pred` counts as
    a mismatch. Line items are compared positionally (matching how the
    task's prompt instructs the model to fill them), which is a slight
    simplification if row order legitimately differs -- acceptable for
    comparing LoRA trials against each other, not meant as the final
    production eval.
    """
    gold_leaves = dict(_leaves(gold))
    if not gold_leaves:
        return 1.0
    pred_leaves = dict(_leaves(pred))
    correct = sum(1 for path, gv in gold_leaves.items()
                  if _norm(pred_leaves.get(path)) == _norm(gv))
    return correct / len(gold_leaves)


def safe_parse(raw_text: str) -> dict:
    try:
        return json.loads(raw_text).get("fields", {})
    except (json.JSONDecodeError, AttributeError):
        return {}

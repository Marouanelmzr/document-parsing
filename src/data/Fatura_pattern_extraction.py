"""
FATURA per-template field pattern induction.

Given ~200 raw annotations sharing the same visual template, this induces,
per field, a regex that separates the template's fixed label/formatting
from the variable value slot(s) — without any hand-written per-template
rules and without an LLM call per template.

Core idea: within one template, a field's label text and punctuation are
identical across all instances; only the value(s) differ. So the fixed
vs. variable split can be recovered by aligning the tokenized samples and
keeping tokens that match, in order, across every sample as "anchors".
Gaps between anchors are the variable slots — this naturally supports
more than one variable region per field (e.g. a percentage AND an amount
in the same string), which a simple prefix/suffix split cannot.
"""

from __future__ import annotations

import re
import difflib
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Step 1: field extraction (native FATURA format -> {field: raw_text})
# Already established for the verification/validation-layer use case;
# reused here so calibration and validation run on the same field set.
# ---------------------------------------------------------------------------

EXCLUDE_KEYS = {"TABLE", "INVOICE_INFO", "OTHER", "LOGO"}


def extract_fatura_fields(raw_annotation: dict) -> dict:
    fields = {}
    for key, value in raw_annotation.items():
        if key in EXCLUDE_KEYS:
            continue
        if isinstance(value, dict) and value.get("text"):
            fields[key] = value["text"].strip()
    return fields


# ---------------------------------------------------------------------------
# Step 2: tokenize into meaningful chunks (numbers, words, punctuation runs,
# whitespace) rather than raw characters. This keeps anchors human-readable
# and avoids spurious single-character "matches".
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\d+\.\d+|\d+|[A-Za-z]+|[^\sA-Za-z0-9]+|\s+")


def tokenize(text: str) -> list:
    return _TOKEN_RE.findall(text)


# ---------------------------------------------------------------------------
# Step 3: consensus alignment across N samples of the SAME (template, field)
# ---------------------------------------------------------------------------

@dataclass
class InducedPattern:
    field: str
    regex: "re.Pattern"
    n_groups: int
    n_samples: int
    n_validated: int

    @property
    def coverage(self) -> float:
        return self.n_validated / self.n_samples if self.n_samples else 0.0

    def extract(self, raw_text: str):
        m = self.regex.fullmatch(raw_text)
        return m.groups() if m else None


def _refine_anchors(anchors: list, sample_tokens: list) -> list:
    """Keep only anchor tokens that still align, in order, with sample_tokens."""
    sm = difflib.SequenceMatcher(None, anchors, sample_tokens, autojunk=False)
    kept = []
    for block in sm.get_matching_blocks():
        kept.extend(anchors[block.a: block.a + block.size])
    return kept


def induce_pattern(field: str, texts: list, min_samples: int = 5) -> Optional[InducedPattern]:
    """
    texts: raw text values for one field, all from the SAME template
           (same on-page label/formatting; only the value(s) differ).

    Returns None if there's not enough calibration data, or no stable
    fixed structure exists (e.g. free-form multi-line fields like
    BUYER / SELLER_ADDRESS where line-wrapping tracks content length) —
    caller should fall back to fuzzy/containment matching for those.
    """
    texts = [t for t in texts if t]
    if len(texts) < min_samples:
        return None

    tokenized = [tokenize(t) for t in texts]

    # Seed with the shortest sample (cheapest to refine against), then
    # progressively intersect with every other sample's tokens. Refining
    # the seed sample against itself is skipped: it's always a no-op
    # (anchors already equal that sample's tokens).
    seed_index = min(range(len(tokenized)), key=lambda i: len(tokenized[i]))
    anchors = tokenized[seed_index]
    for i, tok in enumerate(tokenized):
        if i == seed_index:
            continue
        anchors = _refine_anchors(anchors, tok)
        if not anchors:
            break

    if not anchors:
        return None  # no stable fixed structure -> not template-able

    # Rebuild a regex by aligning one sample against the final anchor set:
    # matched blocks -> escaped literals, gaps -> capture groups.
    ref = tokenized[0]
    sm = difflib.SequenceMatcher(None, anchors, ref, autojunk=False)
    parts = []
    n_groups = 0
    prev_end = 0
    for block in sm.get_matching_blocks():
        if block.b > prev_end:
            parts.append(r"(.*?)")
            n_groups += 1
        literal = "".join(ref[block.b: block.b + block.size])
        if literal:
            parts.append(re.escape(literal))
        prev_end = block.b + block.size
    if prev_end < len(ref):
        parts.append(r"(.*)")
        n_groups += 1

    if n_groups == 0:
        return None  # field is constant across samples -> nothing to extract

    pattern = re.compile("".join(parts), re.DOTALL)
    n_validated = sum(1 for t in texts if pattern.fullmatch(t))

    return InducedPattern(field=field, regex=pattern, n_groups=n_groups,
                           n_samples=len(texts), n_validated=n_validated)


# ---------------------------------------------------------------------------
# Step 4: run this over every (template, field) pair from the 200-sample
# calibration set, keep only what validates above threshold.
# ---------------------------------------------------------------------------

def build_template_patterns(
    calibration_samples: dict,       # {template_id: [raw_annotation, ...]}
    min_samples: int = 5,
    min_coverage: float = 0.95,
) -> dict:
    """Returns {template_id: {field_name: InducedPattern}}.

    Fields that don't reach min_coverage are simply absent from the
    result: the caller falls back to containment/fuzzy matching against
    the raw text for those, exactly as already planned for the
    verification-layer use case.
    """
    result = {}
    for template_id, raw_docs in calibration_samples.items():
        field_texts = defaultdict(list)
        for raw in raw_docs:
            for field, text in extract_fatura_fields(raw).items():
                field_texts[field].append(text)

        template_result = {}
        for field, texts in field_texts.items():
            induced = induce_pattern(field, texts, min_samples=min_samples)
            if induced is not None and induced.coverage >= min_coverage:
                template_result[field] = induced
        result[template_id] = template_result
    return result


def coverage_report(patterns: dict) -> str:
    lines = []
    for template_id, fields in patterns.items():
        lines.append(f"{template_id}: {len(fields)} field(s) with a stable pattern")
        for field, p in fields.items():
            lines.append(f"    {field:<20} groups={p.n_groups}  "
                          f"coverage={p.n_validated}/{p.n_samples}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo: synthetic same-template samples (the three examples in the prompt
# are from three DIFFERENT templates, so they can't demonstrate
# within-template alignment — this builds realistic same-template
# variation instead, including the multi-slot DISCOUNT/TAX case).
# ---------------------------------------------------------------------------

def _demo():
    totals = [f"TOTAL : {v} EUR" for v in
              ["734.33", "1107.09", "963.76", "205.10", "58.02", "1290.44"]]

    discounts = [f"DISCOUNT({pct}%): (-)  {amt}" for pct, amt in
                 [("1.85", "13.42"), ("3.50", "33.60"), ("0.75", "4.11"),
                  ("2.20", "19.05"), ("5.00", "61.30"), ("1.00", "7.88")]]

    dates = [f"Date: {d}" for d in
             ["20-Mar-2008", "04-Aug-2016", "01-Apr-2021",
              "16-Oct-2016", "28-Jun-1997", "11-Nov-2011"]]

    calibration = {
        "template_017": {
            "TOTAL": totals,
            "DISCOUNT": discounts,
            "DATE": dates,
        }
    }

    result = {}
    for template_id, field_texts in calibration.items():
        result[template_id] = {}
        for field, texts in field_texts.items():
            induced = induce_pattern(field, texts, min_samples=5)
            if induced:
                result[template_id][field] = induced

    print(coverage_report(result))
    print()

    total_p = result["template_017"]["TOTAL"]
    print("TOTAL pattern  :", total_p.regex.pattern)
    print("  extract('TOTAL : 42.00 EUR') ->", total_p.extract("TOTAL : 42.00 EUR"))
    print()

    disc_p = result["template_017"]["DISCOUNT"]
    print("DISCOUNT pattern:", disc_p.regex.pattern)
    print("  extract('DISCOUNT(9.00%): (-)  100.00') ->",
          disc_p.extract("DISCOUNT(9.00%): (-)  100.00"))
    print("  -> correctly separates percentage and amount as two groups,")
    print("     even though both vary in the middle of the string.")


if __name__ == "__main__":
    _demo()
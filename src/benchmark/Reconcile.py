"""Reconciles one model's predictions against FATURA-derived ground
truth: for every field, if ground truth actually has a value, that value
wins (whether or not it agrees with the model); if ground truth has
nothing for that field, the model's own prediction is kept as-is.

This is NOT a scoring script (see evaluate.py for that) -- it writes a
corrected copy of each document's prediction rather than an accuracy
report. Reuses evaluate.py's load_gt() and equal() directly so both
scripts agree on what "ground truth directory" and "the same value"
mean.

Empty-container handling matters here: FATURA's ground-truth converter
always initializes line_items to [] (it can never populate it -- FATURA's
TABLE field has no structured line-item data) and only populates taxes
when tax info was actually extractable, so an empty list from ground
truth must be treated as "no information", not as "confirmed zero items" 
-- otherwise every document's real model-predicted line_items/taxes
would get silently wiped out to [].
"""
import argparse
import json
from pathlib import Path

from evaluate import load_gt, equal


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_empty(value):
    return value is None or value == "" or value == []


def merge_fields(pred, gt, path=""):
    """Recursively merge one InvoiceFields-shaped dict (pred) against the
    matching ground-truth dict (gt). Returns (merged_dict, corrections)
    where corrections lists (field_path, pred_value, gt_value) for every
    leaf/list where ground truth actually overrode a *different* model
    value (used only for the summary report -- the merge itself doesn't
    need this distinction, since gt wins either way whenever it has a
    value)."""
    merged = {}
    corrections = []

    for key, pred_value in pred.items():
        field_path = f"{path}.{key}" if path else key
        gt_value = gt.get(key) if isinstance(gt, dict) else None

        if isinstance(pred_value, dict):
            sub_gt = gt_value if isinstance(gt_value, dict) else {}
            sub_merged, sub_corrections = merge_fields(pred_value, sub_gt, field_path)
            merged[key] = sub_merged
            corrections.extend(sub_corrections)

        elif isinstance(pred_value, list):
            # taxes / line_items: wholesale replace only if ground truth's
            # list is actually non-empty. Never merge position-by-position
            # across lists that may differ in length.
            if not is_empty(gt_value):
                if pred_value != gt_value:
                    corrections.append((field_path, pred_value, gt_value))
                merged[key] = gt_value
            else:
                merged[key] = pred_value

        else:
            if not is_empty(gt_value):
                if not equal(pred_value, gt_value, field_path):
                    corrections.append((field_path, pred_value, gt_value))
                merged[key] = gt_value
            else:
                merged[key] = pred_value

    return merged, corrections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", type=Path, required=True,
                     help="e.g. benchmark_results/qwen32b -- one <stem>.json per doc")
    ap.add_argument("--ground-truth-dir", type=Path, required=True,
                     help="directory containing exactly one converted FATURA "
                          "ground-truth JSON (same convention as evaluate.py)")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    gt_all = load_gt(args.ground_truth_dir)
    pred_files = sorted(args.predictions_dir.glob("*.json"))

    print(f"[info] {len(pred_files)} model predictions, {len(gt_all)} ground-truth records")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_corrections = 0
    field_correction_counts = {}
    docs_missing_gt = []

    for pred_file in pred_files:
        stem = pred_file.stem
        pred_record = load_json(pred_file)
        pred_fields = pred_record.get("fields", pred_record)

        gt_record = gt_all.get(stem)

        if gt_record is None:
            docs_missing_gt.append(stem)
            merged_fields, corrections = pred_fields, []
        else:
            gt_fields = gt_record.get("fields", gt_record)
            merged_fields, corrections = merge_fields(pred_fields, gt_fields)

        total_docs += 1
        total_corrections += len(corrections)
        for field_path, _, _ in corrections:
            field_correction_counts[field_path] = field_correction_counts.get(field_path, 0) + 1

        output_record = {
            "fields": merged_fields,
            "raw_text": pred_record.get("raw_text"),
            "rag": pred_record.get("rag"),
        }

        with open(args.output_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(output_record, f, indent=2, ensure_ascii=False)

    print(f"\n[done] {total_docs} documents reconciled, "
          f"{total_corrections} field value(s) overridden by ground truth")

    if docs_missing_gt:
        preview = docs_missing_gt[:10]
        suffix = "..." if len(docs_missing_gt) > 10 else ""
        print(f"[warn] {len(docs_missing_gt)} document(s) had no matching ground truth "
              f"(model output kept unchanged): {preview}{suffix}")

    print("\nCorrections by field (documents where ground truth overrode the model):")
    for field_path, count in sorted(field_correction_counts.items(), key=lambda x: -x[1]):
        print(f"  {field_path:40s} {count:5d}/{total_docs} ({count / total_docs:.1%})")

    summary = {
        "documents": total_docs,
        "total_corrections": total_corrections,
        "documents_missing_ground_truth": docs_missing_gt,
        "corrections_by_field": field_correction_counts,
    }

    summary_path = args.output_dir.parent / f"_{args.output_dir.name}_reconciliation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved corrected predictions to: {args.output_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
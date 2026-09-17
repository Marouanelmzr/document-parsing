import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

IGNORED = {"document_type", "line_items", "language", "period"}
NUMERIC = {"total_net", "total_tax", "total_amount", "rate", "base", "amount"}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_gt(path):
    files = list(path.glob("*.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one GT JSON in {path}")

    data = load_json(files[0])
    return {Path(k).stem: v for k, v in data.items()}


def load_predictions(root):
    models = {}

    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue

        files = list(model_dir.glob("*.json"))
        predictions = {}

        for file in files:
            data = load_json(file)

            if any(
                isinstance(v, dict) and "fields" in v
                for v in data.values()
            ):
                predictions.update({
                    Path(k).stem: v
                    for k, v in data.items()
                })
            else:
                predictions[file.stem] = data

        if predictions:
            models[model_dir.name] = predictions

    return models


def flatten(obj, prefix=""):
    result = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            if not prefix and key in IGNORED:
                continue

            path = f"{prefix}.{key}" if prefix else key

            if key in IGNORED and not prefix:
                continue

            result.update(flatten(value, path))

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            path = f"{prefix}[{i}]"
            result.update(flatten(value, path))

    else:
        result[prefix] = obj

    return result


def normalize_string(value):
    if value is None:
        return None

    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_number(value):
    if value is None:
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    try:
        value = str(value).strip().replace(" ", "")
        value = re.sub(r"[€$£]", "", value)

        if "," in value and "." not in value:
            value = value.replace(",", ".")

        return float(value)
    except (ValueError, TypeError):
        return None


def equal(pred, gt, field):
    if gt is None:
        return pred is None

    if pred is None:
        return False

    name = field.split(".")[-1].split("[")[0]

    if name in NUMERIC:
        p = normalize_number(pred)
        g = normalize_number(gt)

        return p is not None and g is not None and p == g

    return normalize_string(pred) == normalize_string(gt)


def fields(record):
    return flatten(record.get("fields", record))


def evaluate(pred, gt):
    gt = fields(gt)
    pred = fields(pred)

    results = {}

    for field, gt_value in gt.items():
        pred_value = pred.get(field)

        results[field] = {
            "ground_truth": gt_value,
            "prediction": pred_value,
            "correct": equal(pred_value, gt_value, field),
        }

    correct = sum(x["correct"] for x in results.values())
    total = len(results)

    return {
        "fields": results,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0,
    }


def summarize(documents):
    field_stats = defaultdict(lambda: [0, 0])
    correct = total = 0
    doc_scores = []

    for result in documents.values():
        correct += result["correct"]
        total += result["total"]
        doc_scores.append(result["accuracy"])

        for field, value in result["fields"].items():
            field_stats[field][0] += 1
            field_stats[field][1] += value["correct"]

    return {
        "documents": len(documents),
        "fields": total,
        "correct": correct,
        "micro_accuracy": correct / total if total else 0,
        "macro_document_accuracy": (
            sum(doc_scores) / len(doc_scores)
            if doc_scores else 0
        ),
        "per_field": {
            field: {
                "evaluated": n,
                "correct": c,
                "accuracy": c / n if n else 0,
            }
            for field, (n, c) in sorted(field_stats.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path)
    args = parser.parse_args()

    gt = load_gt(args.ground_truth_dir)
    models = load_predictions(args.predictions_root)

    report = {
        "ignored_fields": sorted(IGNORED),
        "models": {},
    }

    for model, predictions in models.items():
        common = sorted(set(gt) & set(predictions))

        documents = {
            doc: evaluate(predictions[doc], gt[doc])
            for doc in common
        }

        summary = summarize(documents)

        report["models"][model] = {
            "summary": summary,
            "documents": documents,
        }

        print(
            f"{model:15s} "
            f"docs={summary['documents']:3d} "
            f"fields={summary['fields']:5d} "
            f"accuracy={summary['micro_accuracy']:.4f}"
        )

        print(
            f"{'':15s}"
            f"macro_doc={summary['macro_document_accuracy']:.4f}"
        )

        for field, stats in summary["per_field"].items():
            print(
                f"  {field:38s} "
                f"{stats['correct']:3d}/{stats['evaluated']:3d} "
                f"{stats['accuracy']:.4f}"
            )

    output = args.predictions_root / "evaluation.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
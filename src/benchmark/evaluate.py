import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

IGNORED = set()  # kept as a hook; currently everything is scored
NUMERIC = {
    "total_net", "total_tax", "total_amount", "rate", "base", "amount",
    "quantity", "unit_price", "total_price", "tax_amount", "tax_rate",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_gt(path):
    files = list(path.glob("*.json"))

    if len(files) == 1:
        data = load_json(files[0])
        return {Path(k).stem: v for k, v in data.items()}

    gt = {}
    for file in files:
        data = load_json(file)
        if any(isinstance(v, dict) and "fields" in v for v in data.values()):
            gt.update({Path(k).stem: v for k, v in data.items()})
        else:
            gt[file.stem] = data
    return gt


def load_predictions(root):
    models = {}

    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue

        files = list(model_dir.glob("*.json"))
        predictions = {}

        for file in files:
            data = load_json(file)
            if any(isinstance(v, dict) and "fields" in v for v in data.values()):
                predictions.update({Path(k).stem: v for k, v in data.items()})
            else:
                predictions[file.stem] = data

        if predictions:
            models[model_dir.name] = predictions

    return models


def flatten(obj, prefix=""):
    result = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in IGNORED:
                continue
            path = f"{prefix}.{key}" if prefix else key
            result.update(flatten(value, path))

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            result.update(flatten(value, f"{prefix}[{i}]"))

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


def evaluate_list_field(name, pred_list, gt_list):
    """Order-agnostic comparison for list-of-object fields (line_items, taxes, ...).
    Greedily matches each ground-truth item to the prediction item it agrees
    with most, then scores fields on the matched pair."""
    pred_list = pred_list or []
    gt_list = gt_list or []
    used = set()
    results = {}

    for i, gt_item in enumerate(gt_list):
        best_idx, best_score = None, -1
        for j, pred_item in enumerate(pred_list):
            if j in used:
                continue
            score = sum(equal(pred_item.get(k), v, k) for k, v in gt_item.items())
            if score > best_score:
                best_idx, best_score = j, score

        pred_item = pred_list[best_idx] if best_idx is not None else {}
        if best_idx is not None:
            used.add(best_idx)

        for field, gt_value in gt_item.items():
            key = f"{name}[{i}].{field}"
            results[key] = {
                "ground_truth": gt_value,
                "prediction": pred_item.get(field),
                "correct": equal(pred_item.get(field), gt_value, field),
            }

    results[f"{name}._count_match"] = {
        "ground_truth": len(gt_list),
        "prediction": len(pred_list),
        "correct": len(gt_list) == len(pred_list),
    }
    return results


def fields(record):
    data = record.get("fields", record)
    scalars = {k: v for k, v in data.items() if not isinstance(v, list)}
    list_fields = {k: v for k, v in data.items() if isinstance(v, list)}
    return flatten(scalars), list_fields


def evaluate(pred, gt):
    gt_flat, gt_lists = fields(gt)
    pred_flat, pred_lists = fields(pred)

    results = {}
    for field, gt_value in gt_flat.items():
        pred_value = pred_flat.get(field)
        results[field] = {
            "ground_truth": gt_value,
            "prediction": pred_value,
            "correct": equal(pred_value, gt_value, field),
        }

    for name, gt_list in gt_lists.items():
        results.update(evaluate_list_field(name, pred_lists.get(name), gt_list))

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
        "macro_document_accuracy": sum(doc_scores) / len(doc_scores) if doc_scores else 0,
        "per_field": {
            field: {"evaluated": n, "correct": c, "accuracy": c / n if n else 0}
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

    report = {"ignored_fields": sorted(IGNORED), "models": {}}

    for model, predictions in models.items():
        common = sorted(set(gt) & set(predictions))
        documents = {doc: evaluate(predictions[doc], gt[doc]) for doc in common}
        summary = summarize(documents)

        report["models"][model] = {"summary": summary, "documents": documents}

        print(f"{model:15s} docs={summary['documents']:3d} "
              f"fields={summary['fields']:5d} accuracy={summary['micro_accuracy']:.4f}")
        print(f"{'':15s}macro_doc={summary['macro_document_accuracy']:.4f}")

        for field, stats in summary["per_field"].items():
            print(f"  {field:38s} {stats['correct']:3d}/{stats['evaluated']:3d} {stats['accuracy']:.4f}")

    output = args.predictions_root / "evaluation.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
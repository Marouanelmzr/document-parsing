import sys
import csv
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json

from src.data.Fatura_pattern_extraction import (
    build_template_patterns,
    coverage_report,
)
from src.data.Fatura_schema_conversion import (
    convert_dataset,
    save_json,
)


MANIFEST_PATH = Path(
    "../data/invoices/raw/fatura_sample_400/manifest.csv"
)

ANNOTATIONS_DIR = Path(
    "../data/invoices/raw/Annotations/Original_Format"
)

OUTPUT_PATH = Path(
    "../data/invoices/processed/fatura_sample_400_output.json"
)


def load_manifest():
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No documents found in {MANIFEST_PATH}")

    return rows


def load_annotations(rows):
    annotations = {}

    for row in rows:
        stem = Path(row["filename"]).stem
        annotation_path = ANNOTATIONS_DIR / f"{stem}.json"

        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Annotation not found for {row['filename']}: "
                f"{annotation_path}"
            )

        annotations[stem] = json.loads(
            annotation_path.read_text(encoding="utf-8")
        )

    return annotations


def main():
    rows = load_manifest()

    print(f"Found {len(rows)} documents in manifest.")

    by_template = defaultdict(list)

    for row in rows:
        by_template[row["template_id"]].append(row)

    print(f"Found {len(by_template)} templates.")

    annotations = load_annotations(rows)

    print(f"Loaded {len(annotations)} annotations.")

    template_patterns = {}

    for template_id, template_rows in sorted(by_template.items()):
        doc_ids = [
            Path(row["filename"]).stem
            for row in template_rows
        ]

        template_samples = {
            template_id: [
                annotations[doc_id]
                for doc_id in doc_ids
            ]
        }

        patterns = build_template_patterns(
            template_samples,
            min_samples=5,
            min_coverage=0.95,
        )

        template_patterns.update(patterns)

        print(
            f"\n{template_id}: "
            f"{len(doc_ids)} documents used for pattern induction"
        )

        print(coverage_report(patterns))

    template_ids = {
        Path(row["filename"]).stem: row["template_id"]
        for row in rows
    }

    converted = convert_dataset(
        annotations,
        template_patterns=template_patterns,
        template_ids=template_ids,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    save_json(converted, OUTPUT_PATH)

    print("\n" + "=" * 70)
    print(f"Sampled documents: {len(rows)}")
    print(f"Converted documents: {len(converted)}")
    print(f"Templates: {len(by_template)}")
    print(f"Saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()

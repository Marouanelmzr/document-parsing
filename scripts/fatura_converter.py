import argparse
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


# Defaults match the production 400-doc sample so run_all.sh's existing
# no-args invocation (`cd scripts && python fatura_converter.py`) keeps
# working unchanged. Pass --manifest/--annotations-dir/--output to target
# a different sample (e.g. a small ad hoc set for a Colab smoke test).
DEFAULT_MANIFEST_PATH = Path(
    "../data/invoices/raw/fatura_sample_400/manifest.csv"
)

DEFAULT_ANNOTATIONS_DIR = Path(
    "../data/invoices/raw/Annotations/Original_Format"
)

DEFAULT_OUTPUT_PATH = Path(
    "../data/invoices/processed/fatura_sample_400_output.json"
)


def load_manifest(manifest_path: Path):
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No documents found in {manifest_path}")

    return rows


def load_annotations(rows, annotations_dir: Path):
    annotations = {}

    for row in rows:
        stem = Path(row["filename"]).stem
        annotation_path = annotations_dir / f"{stem}.json"

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH,
                     help="manifest.csv listing the sampled documents "
                          "(sample_id, template_id, filename)")
    ap.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR,
                     help="directory of raw FATURA annotation JSON files")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                     help="where to write the converted ground-truth JSON")
    ap.add_argument("--min-samples", type=int, default=5,
                     help="minimum docs per template before pattern induction "
                          "kicks in (see Fatura_pattern_extraction.induce_pattern). "
                          "Below this, fields fall back to fuzzy/containment "
                          "matching instead of an induced regex -- expected for "
                          "small ad hoc samples, e.g. a 20-doc Colab smoke test.")
    args = ap.parse_args()

    manifest_path = args.manifest
    annotations_dir = args.annotations_dir
    output_path = args.output
    min_samples = args.min_samples

    rows = load_manifest(manifest_path)

    print(f"Found {len(rows)} documents in manifest.")

    by_template = defaultdict(list)

    for row in rows:
        by_template[row["template_id"]].append(row)

    print(f"Found {len(by_template)} templates.")

    annotations = load_annotations(rows, annotations_dir)

    print(f"Loaded {len(annotations)} annotations.")

    template_patterns = {}

    for template_id, template_rows in sorted(by_template.items()):
        doc_ids = [
            Path(row["filename"]).stem
            for row in template_rows
        ]

        if len(doc_ids) < min_samples:
            print(
                f"\n{template_id}: only {len(doc_ids)} document(s) "
                f"(< min_samples={min_samples}) -- pattern induction will be "
                f"skipped for this template; fields fall back to fuzzy/"
                f"containment matching."
            )

        template_samples = {
            template_id: [
                annotations[doc_id]
                for doc_id in doc_ids
            ]
        }

        patterns = build_template_patterns(
            template_samples,
            min_samples=min_samples,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_json(converted, output_path)

    print("\n" + "=" * 70)
    print(f"Sampled documents: {len(rows)}")
    print(f"Converted documents: {len(converted)}")
    print(f"Templates: {len(by_template)}")
    print(f"Saved: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
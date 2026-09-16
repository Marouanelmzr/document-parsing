import sys
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


ANNOTATIONS_DIR = Path("data/invoices/raw/Annotations/Original_Format")
OUTPUT_PATH = Path("data/invoices/processed/fatura_test_output.json")

TEMPLATE_ID = "Template12"
N_CALIBRATION = 10
N_TEST = 10


def load_annotations():
    files = sorted(ANNOTATIONS_DIR.glob(f"{TEMPLATE_ID}_Instance*.json"))

    files = files[:N_CALIBRATION + N_TEST]

    return {
        f.stem: json.loads(f.read_text(encoding="utf-8"))
        for f in files
    }


def main():
    annotations = load_annotations()

    doc_ids = list(annotations)

    calibration_ids = doc_ids[:N_CALIBRATION]
    test_ids = doc_ids[N_CALIBRATION:]

    print(f"Found {len(doc_ids)} documents for {TEMPLATE_ID}")
    print(f"Calibration: {len(calibration_ids)}")
    print(f"Test: {len(test_ids)}")

    # 1. Induce patterns from calibration documents
    calibration_samples = {
        TEMPLATE_ID: [
            annotations[doc_id]
            for doc_id in calibration_ids
        ]
    }

    patterns = build_template_patterns(
        calibration_samples,
        min_samples=5,
        min_coverage=0.95,
    )

    print("\n" + coverage_report(patterns))

    # 2. Apply patterns to unseen documents
    test_annotations = {
        doc_id: annotations[doc_id]
        for doc_id in test_ids
    }

    template_ids = {
        doc_id: TEMPLATE_ID
        for doc_id in test_ids
    }

    converted = convert_dataset(
        test_annotations,
        template_patterns=patterns,
        template_ids=template_ids,
    )

    save_json(converted, OUTPUT_PATH)

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
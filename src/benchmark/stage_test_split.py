"""
Stages symlink-only directories containing just the test-split documents,
so worker.py / evaluate.py can be pointed at them without any changes to
their own logic (they just see a smaller directory).

Usage:
    python stage_test_split.py \
        --split-file data/invoices/splits/test.csv \
        --images-dir data/invoices/raw/images \
        --gt-dir benchmark_results_full/qwen32b_final \
        --out-images bench_out/test_images \
        --out-gt bench_out/test_gt
"""
import argparse
import csv
from pathlib import Path

CANDIDATE_ID_COLUMNS = ["doc_id", "id", "filename", "file", "image", "document_id"]


def read_doc_ids(split_file: Path, id_column: str | None) -> list[str]:
    if split_file.suffix.lower() != ".csv":
        # plain one-id-per-line manifest
        return [l.strip() for l in split_file.read_text().splitlines() if l.strip()]

    with open(split_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        col = id_column
        if col is None:
            col = next((c for c in CANDIDATE_ID_COLUMNS if c in fieldnames), None)
        if col is None:
            # no header match -- fall back to the first column
            col = fieldnames[0] if fieldnames else None
        if col is None:
            raise ValueError(f"Could not determine an id column in {split_file} "
                              f"(columns found: {fieldnames}). Pass --id-column explicitly.")

        print(f"Reading doc ids from column '{col}' (columns available: {fieldnames})")
        # strip any extension in case the manifest stores "Template1_Instance0.json"
        return [Path(row[col].strip()).stem for row in reader if row.get(col, "").strip()]


def stage(doc_ids, src_dir: Path, dst_dir: Path, suffixes):
    dst_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for doc_id in doc_ids:
        found = False
        for suf in suffixes:
            src = src_dir / f"{doc_id}{suf}"
            if src.exists():
                link = dst_dir / src.name
                if not link.exists():
                    link.symlink_to(src.resolve())
                found = True
                break
        if not found:
            missing.append(doc_id)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--out-images", required=True, type=Path)
    ap.add_argument("--out-gt", required=True, type=Path)
    ap.add_argument("--id-column", default=None,
                     help="Column in the split CSV holding the doc id/filename "
                          "(auto-detected from doc_id/id/filename/file/image/document_id if omitted)")
    args = ap.parse_args()

    doc_ids = read_doc_ids(args.split_file, args.id_column)
    print(f"{len(doc_ids)} doc ids in split file")

    # images and gt json share the same stem (Template1_Instance0.jpg / .json)
    missing_img = stage(doc_ids, args.images_dir, args.out_images, [".jpg", ".jpeg", ".png"])
    missing_gt = stage(doc_ids, args.gt_dir, args.out_gt, [".json"])

    print(f"images staged: {len(doc_ids) - len(missing_img)}/{len(doc_ids)}  "
          f"(missing: {missing_img[:5]}{'...' if len(missing_img) > 5 else ''})")
    print(f"gt staged:     {len(doc_ids) - len(missing_gt)}/{len(doc_ids)}  "
          f"(missing: {missing_gt[:5]}{'...' if len(missing_gt) > 5 else ''})")


if __name__ == "__main__":
    main()
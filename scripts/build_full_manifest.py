"""Builds a manifest.csv covering EVERY image in a raw FATURA images
directory -- no sampling, no copying. Needed for the full-corpus
production run, as opposed to sample_fatura_dataset.py (which builds a
stratified subset like fatura_sample_400).

Reuses the exact same "TemplateN_InstanceM.jpg" filename parsing as
sample_fatura_dataset.py, so template_id values line up with what
fatura_converter.py's pattern induction expects -- this is the same
grouping logic that already ran successfully during the Colab smoke test
("10000 images across 50 templates"), just without the sampling step.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
from sample_fatura_dataset import group_by_template


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if not args.images_dir.exists():
        sys.exit(f"[error] images dir not found: {args.images_dir}")

    groups = group_by_template(args.images_dir)
    if not groups:
        sys.exit(
            f"[error] no files matched 'TemplateN_InstanceM.ext' in "
            f"{args.images_dir}"
        )

    total = sum(len(v) for v in groups.values())
    print(f"[info] {total} images across {len(groups)} templates.")

    rows = []
    for tid, paths in sorted(groups.items()):
        for p in sorted(paths, key=lambda x: x.name):
            rows.append({
                "sample_id": f"{len(rows):05d}",
                "template_id": tid,
                "filename": p.name,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "template_id", "filename"])
        w.writeheader()
        w.writerows(rows)

    print(f"[info] wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
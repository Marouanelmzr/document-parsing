#!/usr/bin/env python3
"""Materialize physical train/val/test split folders from the split CSVs.

Reads data/invoices/processed/splits/{train,val,test}.csv and, for each row
(filename like "Template10_Instance112.json"), copies:
  - the matching image from data/invoices/raw/images/{stem}.{ext}
  - the matching label from benchmark_results_full/qwen32b_final
      (either a per-document file named {stem}.json, or -- if that's not
      found -- a single merged JSON in that folder keyed by filename)

into:
  data/invoices/processed/splits/{split}/images/{stem}.{ext}
  data/invoices/processed/splits/{split}/labels/{stem}.json

Usage:
    python scripts/physical_split.py
    python scripts/physical_split.py --mode symlink   # save disk instead of copying
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust if script lives elsewhere
SPLITS_DIR = PROJECT_ROOT / "data/invoices/processed/splits"
RAW_IMAGES_DIR = PROJECT_ROOT / "data/invoices/raw/images"
LABELS_SOURCE_DIR = PROJECT_ROOT / "benchmark_results_full/qwen32b_final"

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]
SPLITS = ["train", "val", "test"]


def find_image(stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = RAW_IMAGES_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_merged_labels() -> dict | None:
    """Fallback: if qwen32b_final isn't per-document files, look for exactly
    one JSON file in it and treat it as {filename_or_stem: label_dict}."""
    json_files = list(LABELS_SOURCE_DIR.glob("*.json"))
    if len(json_files) != 1:
        return None
    with open(json_files[0], encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    return data


def resolve_label(stem: str, filename: str, merged: dict | None) -> dict | None:
    """Returns the label dict for this document, or None if not found."""
    per_doc_path = LABELS_SOURCE_DIR / f"{stem}.json"
    if per_doc_path.exists():
        with open(per_doc_path, encoding="utf-8") as f:
            return json.load(f)

    if merged is not None:
        if filename in merged:
            return merged[filename]
        if stem in merged:
            return merged[stem]

    return None


def materialize_split(split: str, mode: str, merged: dict | None) -> None:
    csv_path = SPLITS_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise SystemExit(f"ERROR: missing split CSV: {csv_path}")

    out_images_dir = SPLITS_DIR / split / "images"
    out_labels_dir = SPLITS_DIR / split / "labels"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    missing_images: list[str] = []
    missing_labels: list[str] = []
    written = 0

    for row in rows:
        filename = row["filename"]              # e.g. "Template10_Instance112.json"
        stem = Path(filename).stem               # "Template10_Instance112"

        image_path = find_image(stem)
        if image_path is None:
            missing_images.append(filename)
            continue

        label_data = resolve_label(stem, filename, merged)
        if label_data is None:
            missing_labels.append(filename)
            continue

        out_image_path = out_images_dir / image_path.name
        out_label_path = out_labels_dir / f"{stem}.json"

        if mode == "symlink":
            if out_image_path.exists() or out_image_path.is_symlink():
                out_image_path.unlink()
            out_image_path.symlink_to(image_path.resolve())
        else:
            shutil.copy2(image_path, out_image_path)

        with open(out_label_path, "w", encoding="utf-8") as f:
            json.dump(label_data, f, ensure_ascii=False, indent=2)

        written += 1

    print(f"[{split}] {written}/{len(rows)} documents materialized "
          f"-> {out_images_dir.parent}")

    if missing_images:
        print(f"[{split}] ERROR: {len(missing_images)} missing images, e.g. "
              f"{missing_images[:5]}")
    if missing_labels:
        print(f"[{split}] ERROR: {len(missing_labels)} missing labels, e.g. "
              f"{missing_labels[:5]}")
    if missing_images or missing_labels:
        raise SystemExit(
            f"[{split}] FAILED: {len(missing_images)} missing images, "
            f"{len(missing_labels)} missing labels. Refusing to leave a "
            f"partial split on disk -- fix the source data and re-run."
        )

    if written != len(rows):
        raise SystemExit(f"[{split}] FAILED: wrote {written}, expected {len(rows)}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["copy", "symlink"], default="copy",
                     help="copy duplicates data 3x across splits; symlink saves "
                          "disk but breaks if the raw corpus / qwen32b_final "
                          "later moves or is deleted.")
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    args = ap.parse_args()

    if not RAW_IMAGES_DIR.exists():
        raise SystemExit(f"ERROR: raw images dir not found: {RAW_IMAGES_DIR}")
    if not LABELS_SOURCE_DIR.exists():
        raise SystemExit(f"ERROR: labels source dir not found: {LABELS_SOURCE_DIR}")

    per_doc_sample = next(LABELS_SOURCE_DIR.glob("*.json"), None)
    merged = None
    if per_doc_sample is not None and len(list(LABELS_SOURCE_DIR.glob("*.json"))) == 1:
        # Only one JSON in the whole dir -- treat as merged, not per-document.
        merged = load_merged_labels()
        print(f"[info] treating {LABELS_SOURCE_DIR} as a single merged JSON "
              f"({per_doc_sample.name}).")
    else:
        print(f"[info] treating {LABELS_SOURCE_DIR} as per-document JSON files.")

    for split in args.splits:
        materialize_split(split, args.mode, merged)

    print("\nAll requested splits materialized successfully.")


if __name__ == "__main__":
    main()
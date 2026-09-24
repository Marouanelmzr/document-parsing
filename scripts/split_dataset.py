"""
Usage:
    python split_dataset.py \
        --labels-dir ../../benchmark_results_full/qwen32b_final \
        --output-dir ../../data/invoices/processed/splits \
        --images-dir ../../data/invoices/raw/images \
        --copy-labels --copy-images
"""
import argparse
import csv
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

FILENAME_RE = re.compile(r"^(Template\d+)_Instance\d+\.json$", re.IGNORECASE)


def group_by_template(labels_dir: Path):
    groups = defaultdict(list)
    unmatched = []
    for p in sorted(labels_dir.iterdir()):
        if not p.is_file():
            continue
        m = FILENAME_RE.match(p.name)
        if m:
            groups[m.group(1)].append(p)
        else:
            unmatched.append(p.name)
    if unmatched:
        print(f"[warn] {len(unmatched)} files didn't match 'TemplateN_InstanceM.json' "
              f"and were skipped, e.g. {unmatched[:3]}")
    return groups


def stratified_split(groups: dict, train_frac: float, val_frac: float, seed: int):
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for tid, paths in sorted(groups.items()):
        pool = paths[:]
        rng.shuffle(pool)
        n = len(pool)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        splits["train"].extend((tid, p) for p in pool[:n_train])
        splits["val"].extend((tid, p) for p in pool[n_train:n_train + n_val])
        splits["test"].extend((tid, p) for p in pool[n_train + n_val:])

    for name in splits:
        rng.shuffle(splits[name])
    return splits


def materialize(entries, output_dir: Path, split_name: str, images_dir: Path,
                 copy_labels: bool, copy_images: bool):
    label_out = output_dir / split_name / "labels"
    image_out = output_dir / split_name / "images"
    if copy_labels:
        label_out.mkdir(parents=True, exist_ok=True)
    if copy_images:
        image_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (tid, label_path) in enumerate(sorted(entries, key=lambda x: x[1].name)):
        stem = label_path.stem
        rows.append({"sample_id": f"{i:05d}", "template_id": tid, "filename": label_path.name})

        if copy_labels:
            shutil.copy2(label_path, label_out / label_path.name)

        if copy_images:
            matches = sorted(
                p for p in images_dir.glob(f"{stem}.*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if not matches:
                sys.exit(f"[error] no image found for '{stem}' in {images_dir}")
            shutil.copy2(matches[0], image_out / matches[0].name)

    return rows


def write_manifest(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "template_id", "filename"])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path,
                     default=Path("../../benchmark_results_full/qwen32b_final"),
                     help="one <TemplateN_InstanceM>.json per doc (default: "
                          "benchmark_results_full/qwen32b_final)")
    ap.add_argument("--output-dir", type=Path,
                     default=Path("../../data/invoices/processed/splits"))
    ap.add_argument("--images-dir", type=Path,
                     default=Path("../../data/invoices/raw/images"),
                     help="raw FATURA images; only read when --copy-images is passed")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy-labels", action="store_true",
                     help="copy each split's JSON files into output-dir/<split>/labels/ "
                          "(default: manifests only, no files copied)")
    ap.add_argument("--copy-images", action="store_true",
                     help="also copy each split's matching images into "
                          "output-dir/<split>/images/ (requires --images-dir to exist)")
    args = ap.parse_args()

    if args.train_frac + args.val_frac >= 1.0:
        sys.exit(f"[error] train_frac + val_frac must be < 1.0 "
                  f"(got {args.train_frac + args.val_frac})")

    if not args.labels_dir.exists():
        sys.exit(f"[error] labels dir not found: {args.labels_dir}")

    if args.copy_images and not args.images_dir.exists():
        sys.exit(f"[error] --copy-images requires --images-dir to exist: {args.images_dir}")

    groups = group_by_template(args.labels_dir)
    if not groups:
        sys.exit("[error] no files matched 'TemplateN_InstanceM.json' in labels dir.")

    total = sum(len(v) for v in groups.values())
    print(f"[info] {total} documents across {len(groups)} templates.")

    splits = stratified_split(groups, args.train_frac, args.val_frac, args.seed)

    # Sanity checks: every document ends up in exactly one split, none
    # dropped, none duplicated.
    all_stems = {p.stem for paths in groups.values() for p in paths}
    split_stems = [p.stem for entries in splits.values() for _, p in entries]
    assert len(split_stems) == len(set(split_stems)), \
        "a document ended up in more than one split"
    assert set(split_stems) == all_stems, \
        "some documents were dropped during splitting"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print()
    for split_name, entries in splits.items():
        rows = materialize(entries, args.output_dir, split_name, args.images_dir,
                            args.copy_labels, args.copy_images)
        manifest_path = args.output_dir / f"{split_name}.csv"
        write_manifest(rows, manifest_path)
        n_templates = len({tid for tid, _ in entries})
        print(f"[info] {split_name:5s}: {len(rows):5d} docs "
              f"({100 * len(rows) / total:4.1f}%) across {n_templates} templates "
              f"-> {manifest_path}")

    print("\n[info] per-template breakdown:")
    print(f"  {'template':12s} {'total':>6s} {'train':>6s} {'val':>6s} {'test':>6s}")
    per_template = defaultdict(lambda: defaultdict(int))
    for split_name, entries in splits.items():
        for tid, _ in entries:
            per_template[tid][split_name] += 1
    for tid in sorted(groups):
        c = per_template[tid]
        print(f"  {tid:12s} {len(groups[tid]):6d} {c['train']:6d} {c['val']:6d} {c['test']:6d}")


if __name__ == "__main__":
    main()
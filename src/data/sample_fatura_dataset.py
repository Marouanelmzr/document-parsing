import argparse
import csv
import random
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

FILENAME_RE = re.compile(r"^(Template\d+)_Instance\d+\.(jpg)$", re.IGNORECASE)


def group_by_template(images_dir: Path):
    groups = defaultdict(list)
    unmatched = []
    for p in sorted(images_dir.iterdir()):
        m = FILENAME_RE.match(p.name)
        if m:
            groups[m.group(1)].append(p)
        elif p.is_file():
            unmatched.append(p.name)
    if unmatched:
        print(f"[warn] {len(unmatched)} files didn't match 'TemplateN_InstanceM.ext' and were skipped, "
              f"e.g. {unmatched[:3]}")
    return groups


def stratified_sample(groups: dict, n_samples: int, seed: int):
    rng = random.Random(seed)
    template_ids = list(groups.keys())
    n_templates = len(template_ids)
    base_quota, remainder = divmod(n_samples, n_templates)
    rng.shuffle(template_ids)

    selected, shortfall = [], 0
    for i, tid in enumerate(template_ids):
        quota = base_quota + (1 if i < remainder else 0)
        pool = groups[tid][:]
        rng.shuffle(pool)
        selected.extend((tid, p) for p in pool[:quota])
        shortfall += max(0, quota - len(pool))

    if shortfall:
        taken = {p for _, p in selected}
        leftover = [(tid, p) for tid, imgs in groups.items() for p in imgs if p not in taken]
        rng.shuffle(leftover)
        selected.extend(leftover[:shortfall])

    rng.shuffle(selected)
    return selected[:n_samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default=Path("../../data/invoices/raw/images"), type=Path)
    ap.add_argument("--output-dir", default=Path("../../data/invoices/raw/fatura_sample_200"), type=Path)
    ap.add_argument("--n-samples", default=200, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    if not args.images_dir.exists():
        sys.exit(f"[error] images dir not found: {args.images_dir}")

    groups = group_by_template(args.images_dir)
    if not groups:
        sys.exit("[error] no files matched 'TemplateN_InstanceM.ext' in images dir.")
    print(f"[info] {sum(len(v) for v in groups.values())} images across {len(groups)} templates.")

    if args.n_samples > sum(len(v) for v in groups.values()):
        sys.exit(f"[error] requested {args.n_samples} samples but only "
                  f"{sum(len(v) for v in groups.values())} images available.")

    selected = stratified_sample(groups, args.n_samples, args.seed)
    print(f"[info] selected {len(selected)} images across {len({t for t, _ in selected})} templates.")

    images_out = args.output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (tid, src) in enumerate(sorted(selected, key=lambda x: (x[0], x[1].name))):
        shutil.copy2(src, images_out / src.name)
        rows.append({"sample_id": f"{i:04d}", "template_id": tid, "filename": src.name})

    manifest_path = args.output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "template_id", "filename"])
        w.writeheader()
        w.writerows(rows)
    print(f"[info] wrote {manifest_path}")

    if not args.no_zip:
        zip_path = args.output_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in images_out.rglob("*"):
                zf.write(p, p.relative_to(args.output_dir))
            zf.write(manifest_path, manifest_path.relative_to(args.output_dir))
        print(f"[info] zipped to {zip_path} -- upload this to Colab.")


if __name__ == "__main__":
    main()
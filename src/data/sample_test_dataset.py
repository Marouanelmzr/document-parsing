from pathlib import Path
from collections import defaultdict
import csv
import random

TEST_CSV = Path("../../data/invoices/processed/splits/test.csv")
OUTPUT_CSV = Path("../../data/invoices/processed/splits/test_200.csv")

SEED = 42
N = 200

with open(TEST_CSV, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

by_template = defaultdict(list)

for row in rows:
    by_template[row["template_id"]].append(row)

rng = random.Random(SEED)

for template_rows in by_template.values():
    rng.shuffle(template_rows)

# 4 per template = 200 for 50 templates
subset = []

for template_id in sorted(by_template):
    subset.extend(by_template[template_id][:4])

rng.shuffle(subset)

assert len(subset) == 200
assert len({row["filename"] for row in subset}) == 200

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["sample_id", "template_id", "filename"]
    )
    writer.writeheader()

    for i, row in enumerate(subset):
        row["sample_id"] = f"{i:05d}"
        writer.writerow(row)

print(f"Created {OUTPUT_CSV}")
print(f"Samples: {len(subset)}")
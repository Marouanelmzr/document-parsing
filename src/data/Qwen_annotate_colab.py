import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Fixed schema shown to the model. This text never changes between invoices,
# so prompt length is constant across the whole run -- only the image varies.
TARGET_SCHEMA = {
    "fields": {
        "date": "...", "taxes": {"items": []},
        "locale": {"country": "...", "currency": "...", "language": "..."},
        "due_date": "...", "po_number": "...", "total_net": "...", "total_tax": "...",
        "line_items": [{
            "quantity": "...", "tax_rate": "...", "tax_amount": "...", "unit_price": "...",
            "description": "...", "total_price": "...", "product_code": "...", "unit_measure": "..."
        }],
        "customer_id": "...", "payment_date": "...", "total_amount": "...",
        "customer_name": "...", "document_type": "...", "supplier_name": "...",
        "invoice_number": "...", "supplier_email": "...",
        "billing_address": {"city": "...", "state": "...", "po_box": "...", "address": "...",
                             "country": "...", "postal_code": "...", "street_name": "...",
                             "street_number": "...", "address_complement": "..."},
        "customer_address": {"city": "...", "state": "...", "po_box": "...", "address": "...",
                              "country": "...", "postal_code": "...", "street_name": "...",
                              "street_number": "...", "address_complement": "..."},
        "shipping_address": {"city": "...", "state": "...", "po_box": "...", "address": "...",
                              "country": "...", "postal_code": "...", "street_name": "...",
                              "street_number": "...", "address_complement": "..."},
        "supplier_address": {"city": "...", "state": "...", "po_box": "...", "address": "...",
                              "country": "...", "postal_code": "...", "street_name": "...",
                              "street_number": "...", "address_complement": "..."},
        "supplier_website": "...", "reference_numbers": ["..."], "supplier_phone_number": "...",
        "supplier_payment_details": {"items": []},
        "customer_company_registration": {"items": []},
        "supplier_company_registration": {"items": []},
    },
}

PROMPT = f"""You are an information-extraction engine for invoice images.
Read the attached invoice image carefully and extract every field you can find.

Return ONLY a single valid JSON object with exactly this shape (fill values you find,
use null for any field you cannot find, use an empty list [] for list fields with no
items, and do not add extra keys):

{json.dumps(TARGET_SCHEMA, indent=2)}

Rules:
- Copy text values exactly as they appear on the invoice (dates, numbers, names, addresses).
- Split any combined "bill to" / buyer block into customer_name and customer_address parts.
- Fill line_items as a list, one entry per invoice line, even if some sub-fields are missing.
- Do not compute or validate totals; just transcribe whatever amounts are printed.
- Output raw JSON only, no markdown fences, no commentary.
"""

# Sanity check: this prompt template is identical for every image, so its
# token length never varies across the run.
FIXED_PROMPT_LEN_CHARS = len(PROMPT)


def load_model():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[info] loading {MODEL_ID} in 4-bit (NF4) for T4...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def build_inputs(processor, image: Image.Image):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    return inputs


def extract_json(raw_text: str):
    """Best-effort extraction of the JSON object from the model's raw output."""
    cleaned = re.sub(r"```json|```", "", raw_text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return None, cleaned
    candidate = cleaned[start:end + 1]
    try:
        return json.loads(candidate), cleaned
    except json.JSONDecodeError:
        return None, cleaned


def run(images_dir: Path, output_dir: Path, manifest_path: Path = None, max_new_tokens: int = 1500):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_text"
    raw_dir.mkdir(exist_ok=True)

    if manifest_path and manifest_path.exists():
        with open(manifest_path) as f:
            rows = list(csv.DictReader(f))
        image_files = [images_dir / r["filename"] for r in rows]
    else:
        image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    print(f"[info] {len(image_files)} images to annotate. Fixed prompt length: {FIXED_PROMPT_LEN_CHARS} chars.")

    model, processor = load_model()
    model.eval()

    failures = []
    t0 = time.time()
    for i, img_path in enumerate(image_files):
        if not img_path.exists():
            print(f"[warn] missing file, skipping: {img_path}")
            continue
        image = Image.open(img_path).convert("RGB")
        inputs = build_inputs(processor, image).to(model.device)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        raw_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        parsed, cleaned = extract_json(raw_text)
        stem = img_path.stem

        (raw_dir / f"{stem}.txt").write_text(cleaned)

        if parsed is None:
            failures.append(stem)
            print(f"[warn] ({i+1}/{len(image_files)}) failed to parse JSON for {stem}")
            continue

        record = {
            "fields": parsed.get("fields", parsed),
            "raw_text": cleaned,
            "rag": {"retrieved_document_id": stem},
        }
        with open(output_dir / f"{stem}.json", "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[info] {i+1}/{len(image_files)} done ({elapsed:.0f}s elapsed)")

    print(f"[done] {len(image_files) - len(failures)} succeeded, {len(failures)} failed to parse.")
    if failures:
        (output_dir / "_failed_parses.txt").write_text("\n".join(failures))
        print(f"[info] failed ids written to {output_dir / '_failed_parses.txt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--output-dir", default=Path("./qwen_annotations"), type=Path)
    ap.add_argument("--manifest", default=None, type=Path,
                     help="Optional manifest.csv from sample_fatura_dataset.py (uses filename column)")
    ap.add_argument("--max-new-tokens", default=1500, type=int)
    args = ap.parse_args()

    if not args.images_dir.exists():
        sys.exit(f"[error] images dir not found: {args.images_dir}")

    run(args.images_dir, args.output_dir, args.manifest, args.max_new_tokens)


if __name__ == "__main__":
    main()
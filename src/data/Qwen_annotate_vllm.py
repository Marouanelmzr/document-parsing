import argparse
import asyncio
import base64
import csv
import io
import json
import re
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image

# MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

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

CRITICAL RULES - read carefully, these prevent common extraction errors:

1. TWO PARTIES ONLY. There are exactly two parties on this document: SUPPLIER (the
   issuing company, sometimes labeled "From", "Seller", "Bill From") and CUSTOMER
   (the buyer, sometimes labeled "To", "Buyer", "Bill To", "Ship To"). For every
   contact field (address, phone, email, website), only use a value that is printed
   directly under that party's OWN label block. Never copy a phone number, website,
   email, or address from the customer's block into any supplier_* field, or vice
   versa. If you are not sure which block a value belongs to, leave the field null
   rather than guessing.

2. NO INFERRED OR COPIED VALUES. If a field has no explicit printed label on the
   document, output null for it. In particular: do not copy invoice_number into
   po_number (or any other field) just because no PO number is printed -- if there
   is no line/label that says "PO Number" or "P.O.", po_number must be null. Never
   fill a field by duplicating the value of a different field.

3. LINE ITEM COLUMNS ARE POSITIONAL, NOT INTERCHANGEABLE. Each line item row has
   distinct printed columns (for example: description, quantity, unit price, tax
   rate, tax amount, total price). Map each value strictly by its column position
   in the table header -- do not reuse the same printed number for two different
   fields. In particular, tax_amount and total_price are almost always different
   values; only set them equal if the table genuinely prints the same number in
   both columns.

4. Copy text values exactly as they appear on the invoice (dates, numbers, names,
   addresses). Split any combined "bill to" / buyer block into customer_name and
   customer_address parts. Fill line_items as a list, one entry per invoice line,
   even if some sub-fields are missing. Do not compute or validate totals; just
   transcribe whatever amounts are printed.

Output raw JSON only, no markdown fences, no commentary.
"""

# Sanity check: this prompt template is identical for every image, so its
# token length never varies across the run.
FIXED_PROMPT_LEN_CHARS = len(PROMPT)


def image_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a base64 data URL for the OpenAI-style image_url content type."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


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


async def annotate_one(client, sem, img_path: Path, raw_dir: Path, output_dir: Path,
                        max_new_tokens: int, served_model: str):
    async with sem:
        image = Image.open(img_path).convert("RGB")
        data_url = image_to_data_url(image)
        stem = img_path.stem

        try:
            response = await client.chat.completions.create(
                model=served_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": PROMPT},
                    ],
                }],
                max_tokens=max_new_tokens,
                temperature=0,  # matches do_sample=False in the transformers version
            )
            raw_text = response.choices[0].message.content
        except Exception as e:
            print(f"[warn] request failed for {stem}: {e}")
            return stem, False

        parsed, cleaned = extract_json(raw_text)
        (raw_dir / f"{stem}.txt").write_text(cleaned)

        if parsed is None:
            print(f"[warn] failed to parse JSON for {stem}")
            return stem, False

        record = {
            "fields": parsed.get("fields", parsed),
            "raw_text": cleaned,
            "rag": {"retrieved_document_id": stem},
        }
        with open(output_dir / f"{stem}.json", "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        return stem, True


async def run_async(images_dir: Path, output_dir: Path, manifest_path: Path, max_new_tokens: int,
                     server_url: str, api_key: str, served_model: str, concurrency: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_text"
    raw_dir.mkdir(exist_ok=True)

    if manifest_path and manifest_path.exists():
        with open(manifest_path) as f:
            rows = list(csv.DictReader(f))
        image_files = [images_dir / r["filename"] for r in rows]
    else:
        image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    image_files = [p for p in image_files if p.exists()]

    print(f"[info] {len(image_files)} images to annotate. Fixed prompt length: {FIXED_PROMPT_LEN_CHARS} chars.")
    print(f"[info] vLLM server at {server_url} (served model: {served_model}), concurrency={concurrency}")

    client = AsyncOpenAI(base_url=server_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    t0 = time.time()
    done_count = 0
    failures = []

    tasks = [
        annotate_one(client, sem, img_path, raw_dir, output_dir, max_new_tokens, served_model)
        for img_path in image_files
    ]

    for coro in asyncio.as_completed(tasks):
        stem, ok = await coro
        done_count += 1
        if not ok:
            failures.append(stem)
        if done_count % 10 == 0:
            elapsed = time.time() - t0
            print(f"[info] {done_count}/{len(image_files)} done ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    print(f"[done] {len(image_files) - len(failures)} succeeded, {len(failures)} failed, "
          f"in {elapsed:.1f}s ({len(image_files) / elapsed:.2f} img/s)")
    if failures:
        (output_dir / "_failed_parses.txt").write_text("\n".join(failures))
        print(f"[info] failed ids written to {output_dir / '_failed_parses.txt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--output-dir", default=Path("./qwen_annotations_vllm_batched"), type=Path)
    ap.add_argument("--manifest", default=None, type=Path,
                     help="Optional manifest.csv from sample_fatura_dataset.py (uses filename column)")
    ap.add_argument("--max-new-tokens", default=1500, type=int)
    ap.add_argument("--server-url", default="http://localhost:8000/v1",
                     help="Base URL of the running vLLM OpenAI-compatible server")
    ap.add_argument("--api-key", default="EMPTY", help="vLLM server doesn't check this by default")
    ap.add_argument("--served-model", default=MODEL_ID,
                     help="Model name as registered with `vllm serve` (usually same as MODEL_ID)")
    ap.add_argument("--concurrency", default=6, type=int,
                     help="Max number of in-flight requests sent to the server at once. "
                          "This is what lets vLLM's continuous batching actually kick in.")
    args = ap.parse_args()

    if not args.images_dir.exists():
        sys.exit(f"[error] images dir not found: {args.images_dir}")

    asyncio.run(run_async(args.images_dir, args.output_dir, args.manifest, args.max_new_tokens,
                           args.server_url, args.api_key, args.served_model, args.concurrency))


if __name__ == "__main__":
    main()
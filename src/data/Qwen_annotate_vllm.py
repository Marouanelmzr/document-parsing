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
        "supplier_name": "<string>",
        "supplier_phone_number": "<string>",
        "supplier_address": {
            "address": "<string>", "street_number": "<string>", "street_name": "<string>",
            "po_box": "<string>", "address_complement": "<string, e.g. floor/building/suite>",
            "city": "<string>", "postal_code": "<string>", "state": "<string>", "country": "<string>"
        },
        "customer_name": "<string>",
        "customer_address": {
            "address": "<string>", "street_number": "<string>", "street_name": "<string>",
            "po_box": "<string>", "address_complement": "<string, e.g. floor/building/suite>",
            "city": "<string>", "postal_code": "<string>", "state": "<string>", "country": "<string>"
        },
        "invoice_number": "<string>",
        "document_type": "<classification>",
        "date": "<date>",
        "due_date": "<date>",
        "period": "<date>",
        "locale": {"language": "<string, ISO 639-1>", "country": "<string, ISO 3166-1 alpha-2>", "currency": "<string, ISO 4217>"},
        "total_net": "<number, total before taxes>",
        "total_tax": "<number>",
        "total_amount": "<number, final total the customer owes>",
        "taxes": [{"rate": "<number, decimal e.g. 0.20>", "base": "<number, amount tax computed on>", "amount": "<number>"}],
        "line_items": [{
            "description": "<string>", "quantity": "<number>", "unit_price": "<number>",
            "total_price": "<number, printed line total>",
            "tax_amount": "<number>", "tax_rate": "<number, decimal>", "unit_measure": "<string>"
        }],
    },
}

PROMPT = f"""You are an information-extraction engine for invoice images. Read the attached invoice image and extract every field you can find.

Return ONLY a single valid JSON object matching this shape exactly (placeholders like "<number>"/"<date>" show the expected type; use null when a field isn't found, [] for empty lists, and add no extra keys):

{json.dumps(TARGET_SCHEMA, separators=(',', ':'))}

CRITICAL RULES:

1. TWO PARTIES ONLY: SUPPLIER (issuer; may be labeled "From"/"Seller"/"Bill From") vs CUSTOMER (buyer; "To"/"Buyer"/"Bill To"/"Ship To"). For address/phone fields, only use values printed under that party's own block. Never cross-copy between supplier_* and customer_*. If unsure which block a value belongs to, use null.

2. NO INFERRED OR COPIED VALUES: a field with no explicit printed label is null — never fill it by duplicating a different field's value.

3. LINE ITEM COLUMNS ARE POSITIONAL: map each value by its column position in the table header (e.g. description, quantity, unit price, tax rate, tax amount, total price) — never reuse one printed number for two fields. tax_amount and total_price are almost always different; only set them equal if the table genuinely prints the same number in both columns.

4. Transcribe values exactly as printed (dates, numbers, names, addresses); do not compute or validate totals. Split a combined "bill to"/buyer block into customer_name and customer_address. Fill line_items with one entry per row, even if some sub-fields are missing.

5. THREE DISTINCT DATES — do not confuse them:
   - date = invoice issue date. Keywords: "Date", "Date de facture", "Facturé le", "Invoice date", "Date of issue". Usually the earliest date, near the invoice number. Never the payment deadline.
   - due_date = payment deadline. Keywords (FR): "Échéance", "Date d'échéance", "À payer avant le", "Payable au", "Date limite de paiement", "Valable jusqu'au". (EN): "Due date", "Payment due", "Payable by", "Deadline". Usually ≥ the issue date — if two dates appear, the later one is usually due_date.
   - period = the timeframe the invoiced work covers (not issue date, not deadline). Keywords: "Période", "Prestations du", "Mois de", "Billing period". Often a month or date range; null if the invoice is a one-off with no stated period.

6. document_type must be exactly one of: invoice, tax_invoice.

Output raw JSON only — no markdown fences, no commentary.
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
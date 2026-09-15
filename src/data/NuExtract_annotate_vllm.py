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

MODEL_ID = "numind/NuExtract3"

# NuExtract's template format: leaf values are TYPES, not free-text placeholders.
# Supported: verbatim-string, string, integer, number, date-time (or "date"),
# currency, country, email, arrays (["type"]), and enums (["opt1","opt2",...]).
TARGET_TEMPLATE = {
    "supplier_name": "verbatim-string",
    "supplier_phone_number": "verbatim-string",
    "supplier_address": {
        "address": "verbatim-string", "street_number": "verbatim-string", "street_name": "verbatim-string",
        "po_box": "verbatim-string", "address_complement": "verbatim-string",
        "city": "verbatim-string", "postal_code": "verbatim-string", "state": "verbatim-string", "country": "country"
    },
    "customer_name": "verbatim-string",
    "customer_address": {
        "address": "verbatim-string", "street_number": "verbatim-string", "street_name": "verbatim-string",
        "po_box": "verbatim-string", "address_complement": "verbatim-string",
        "city": "verbatim-string", "postal_code": "verbatim-string", "state": "verbatim-string", "country": "country"
    },
    "invoice_number": "verbatim-string",
    "document_type": ["invoice", "tax_invoice"],
    "date": "date-time",
    "due_date": "date-time",
    "period": "verbatim-string",
    "locale": {"language": "verbatim-string", "country": "country", "currency": "currency"},
    "total_net": "number",
    "total_tax": "number",
    "total_amount": "number",
    "taxes": [{"rate": "number", "base": "number", "amount": "number"}],
    "line_items": [{
        "description": "verbatim-string", "quantity": "number", "unit_price": "number",
        "total_price": "number", "tax_amount": "number", "tax_rate": "number", "unit_measure": "verbatim-string"
    }],
}

# This is the same content as your old "CRITICAL RULES" block, just relocated:
# NuExtract takes rules via the separate `instructions` chat_template_kwarg
# rather than folded into the user-facing prompt text.
INSTRUCTIONS = """TWO PARTIES ONLY: SUPPLIER (issuer; may be labeled "From"/"Seller"/"Bill From") vs CUSTOMER \
(buyer; "To"/"Buyer"/"Bill To"/"Ship To"). For address/phone fields, only use values printed under that \
party's own block. Never cross-copy between supplier_* and customer_*. If unsure which block a value \
belongs to, leave it null.

NO INFERRED OR COPIED VALUES: a field with no explicit printed label is null -- never fill it by \
duplicating a different field's value (e.g. do not copy invoice_number into a missing PO field).

LINE ITEM COLUMNS ARE POSITIONAL: map each value by its column position in the table header (description, \
quantity, unit price, tax rate, tax amount, total price) -- never reuse one printed number for two fields. \
tax_amount and total_price are almost always different; only set them equal if the table genuinely prints \
the same number in both columns.

Transcribe values exactly as printed; do not compute or validate totals. Split a combined "bill to"/buyer \
block into customer_name and customer_address. Fill line_items with one entry per row, even if some \
sub-fields are missing.

THREE DISTINCT DATES:
- date = invoice issue date. Keywords: "Date", "Date de facture", "Facturé le", "Invoice date", "Date of \
issue". Usually the earliest date, near the invoice number. Never the payment deadline.
- due_date = payment deadline. Keywords (FR): "Échéance", "Date d'échéance", "À payer avant le", "Payable \
au", "Date limite de paiement", "Valable jusqu'au". (EN): "Due date", "Payment due", "Payable by", \
"Deadline". Usually on or after the issue date -- if two dates appear, the later one is usually due_date.
- period = the timeframe the invoiced work covers (not issue date, not deadline). Keywords: "Période", \
"Prestations du", "Mois de", "Billing period". Often a month or date range; null if the invoice is a \
one-off with no stated period.
"""

# Sanity check: template + instructions are identical for every image, so their
# combined length never varies across the run -- only the image varies.
FIXED_TEMPLATE_LEN_CHARS = len(json.dumps(TARGET_TEMPLATE)) + len(INSTRUCTIONS)


def image_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a base64 data URL for the OpenAI-style image_url content type."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_json(raw_text: str):
    """Best-effort extraction of the JSON object from the model's raw output.
    Also strips a <think>...</think> block if reasoning mode was left on."""
    cleaned = raw_text
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    cleaned = re.sub(r"```json|```", "", cleaned).strip()
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
                        max_new_tokens: int, served_model: str, temperature: float):
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
                    ],
                }],
                max_tokens=max_new_tokens,
                temperature=temperature,  # NuExtract recommends ~0.2 for non-thinking mode, not 0
                extra_body={
                    "chat_template_kwargs": {
                        "template": json.dumps(TARGET_TEMPLATE),
                        "instructions": INSTRUCTIONS,
                        "enable_thinking": False,  # fast/deterministic mode; flip to True for hard docs
                    }
                },
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
            "fields": parsed,
            "raw_text": cleaned,
            "rag": {"retrieved_document_id": stem},
        }
        with open(output_dir / f"{stem}.json", "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        return stem, True


async def run_async(images_dir: Path, output_dir: Path, manifest_path: Path, max_new_tokens: int,
                     server_url: str, api_key: str, served_model: str, concurrency: int, temperature: float):
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

    print(f"[info] {len(image_files)} images to annotate. Fixed template+instructions length: "
          f"{FIXED_TEMPLATE_LEN_CHARS} chars.")
    print(f"[info] vLLM server at {server_url} (served model: {served_model}), "
          f"concurrency={concurrency}, temperature={temperature}")

    client = AsyncOpenAI(base_url=server_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    t0 = time.time()
    done_count = 0
    failures = []

    tasks = [
        annotate_one(client, sem, img_path, raw_dir, output_dir, max_new_tokens, served_model, temperature)
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
    ap.add_argument("--output-dir", default=Path("./nuextract3_annotations"), type=Path)
    ap.add_argument("--manifest", default=None, type=Path,
                     help="Optional manifest.csv from sample_fatura_dataset.py (uses filename column)")
    ap.add_argument("--max-new-tokens", default=1500, type=int)
    ap.add_argument("--server-url", default="http://localhost:8000/v1",
                     help="Base URL of the running vLLM OpenAI-compatible server")
    ap.add_argument("--api-key", default="EMPTY", help="vLLM server doesn't check this by default")
    ap.add_argument("--served-model", default=MODEL_ID,
                     help="Model name as registered with `vllm serve` (usually same as MODEL_ID)")
    ap.add_argument("--concurrency", default=6, type=int,
                     help="Max number of in-flight requests sent to the server at once.")
    ap.add_argument("--temperature", default=0.2, type=float,
                     help="NuExtract's model card recommends ~0.2 for non-thinking mode rather than 0.")
    args = ap.parse_args()

    if not args.images_dir.exists():
        sys.exit(f"[error] images dir not found: {args.images_dir}")

    asyncio.run(run_async(args.images_dir, args.output_dir, args.manifest, args.max_new_tokens,
                           args.server_url, args.api_key, args.served_model, args.concurrency,
                           args.temperature))


if __name__ == "__main__":
    main()
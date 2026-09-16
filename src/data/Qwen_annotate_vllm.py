import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams

# MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
MODEL_ID = "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"

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


def build_conversation(img_path: Path):
    """Build one chat conversation for a single invoice image.

    Offline batching doesn't need a base64 data URL -- vLLM's chat() accepts
    a PIL.Image directly via the "image" content type, so the image is
    just opened and handed straight to the engine.
    """
    image = Image.open(img_path).convert("RGB")
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]


def process_output(stem: str, raw_text: str, raw_dir: Path, output_dir: Path) -> bool:
    parsed, cleaned = extract_json(raw_text)
    (raw_dir / f"{stem}.txt").write_text(cleaned)

    if parsed is None:
        print(f"[warn] failed to parse JSON for {stem}")
        return False

    record = {
        "fields": parsed.get("fields", parsed),
        "raw_text": cleaned,
        "rag": {"retrieved_document_id": stem},
    }
    with open(output_dir / f"{stem}.json", "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return True


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def run(images_dir: Path, output_dir: Path, manifest_path: Path, max_new_tokens: int,
        served_model: str, dtype: str, max_model_len: int, gpu_memory_utilization: float,
        batch_size: int):
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
    print(f"[info] loading {served_model} via vLLM offline LLM() -- no HTTP server involved")
    print(f"[info] processing in batches of {batch_size} to bound memory usage")

    # Same engine args as the `vllm serve` invocation, just passed straight to LLM().
    llm = LLM(
        model=served_model,
        dtype=dtype,
        limit_mm_per_prompt={"image": 1},
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)

    t0 = time.time()
    done_count = 0
    failures = []

    # Only one batch's worth of PIL images / conversations is held in memory at
    # a time; 
    # vLLM schedules and batches the requests internally during this offline
    # llm.chat() call. The outer batch_size only controls how many documents
    # are held in host memory at once.
    for batch_paths in chunked(image_files, batch_size):
        stems = [p.stem for p in batch_paths]
        conversations = [build_conversation(p) for p in batch_paths]

        outputs = llm.chat(conversations, sampling_params)

        for stem, output in zip(stems, outputs):
            raw_text = output.outputs[0].text
            ok = process_output(stem, raw_text, raw_dir, output_dir)
            if not ok:
                failures.append(stem)

        done_count += len(batch_paths)
        elapsed = time.time() - t0
        print(f"[info] {done_count}/{len(image_files)} done ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    n = len(image_files)
    print(f"[done] {n - len(failures)} succeeded, {len(failures)} failed, "
          f"in {elapsed:.1f}s ({n / elapsed:.2f} img/s)")
    if failures:
        (output_dir / "_failed_parses.txt").write_text("\n".join(failures))
        print(f"[info] failed ids written to {output_dir / '_failed_parses.txt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--output-dir", default=Path("./qwen_annotations_vllm_offline"), type=Path)
    ap.add_argument("--manifest", default=None, type=Path,
                     help="Optional manifest.csv from sample_fatura_dataset.py (uses filename column)")
    ap.add_argument("--max-new-tokens", default=1500, type=int)
    ap.add_argument("--served-model", default=MODEL_ID,
                     help="HF repo id or local path passed to vllm.LLM(model=...)")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-model-len", default=4096, type=int)
    ap.add_argument("--gpu-memory-utilization", default=0.95, type=float)
    ap.add_argument("--batch-size", default=200, type=int,
                     help="Number of images to load and submit to llm.chat() per offline batch")
    args = ap.parse_args()

    if not args.images_dir.exists():
        sys.exit(f"[error] images dir not found: {args.images_dir}")

    run(args.images_dir, args.output_dir, args.manifest, args.max_new_tokens,
        args.served_model, args.dtype, args.max_model_len, args.gpu_memory_utilization,
        args.batch_size)


if __name__ == "__main__":
    main()
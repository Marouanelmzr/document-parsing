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

# Qwen resizes to multiples of a 28x28 patch (14px ViT patch, 2x2 merge).
# 1 visual token == 28*28 == 784 pixels after resizing.
PATCH_PIXELS = 28 * 28


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


def build_conversation(img_path: Path, size_log: list):
    """Build one chat conversation for a single invoice image.

    Offline batching doesn't need a base64 data URL -- vLLM's chat() accepts
    a PIL.Image directly via the "image" content type. min_pixels/max_pixels
    are set globally on the LLM via mm_processor_kwargs (see run()), so we just
    log each image's native size here to monitor how much resizing is happening.
    """
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    size_log.append({"stem": img_path.stem, "width": w, "height": h, "pixels": w * h})
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


def summarize_sizes(size_log: list, min_pixels: int, max_pixels: int, output_dir: Path):
    """Report how each image's *native* pixel count compares to the configured
    min/max bounds, so under/over-sized scans in the dataset are visible before
    they silently degrade extraction quality."""
    if not size_log:
        return

    pixels = [r["pixels"] for r in size_log]
    below_min = [r for r in size_log if r["pixels"] < min_pixels]
    above_max = [r for r in size_log if r["pixels"] > max_pixels]

    summary = {
        "count": len(size_log),
        "min_pixels_seen": min(pixels),
        "max_pixels_seen": max(pixels),
        "mean_pixels_seen": sum(pixels) / len(pixels),
        "configured_min_pixels": min_pixels,
        "configured_max_pixels": max_pixels,
        "num_upscaled_below_min": len(below_min),
        "num_downscaled_above_max": len(above_max),
        "upscaled_examples": [r["stem"] for r in below_min[:10]],
        "downscaled_examples": [r["stem"] for r in above_max[:10]],
    }
    with open(output_dir / "_image_size_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[info] image size report: native pixels range "
          f"[{summary['min_pixels_seen']:,} .. {summary['max_pixels_seen']:,}], "
          f"mean {summary['mean_pixels_seen']:,.0f}")
    print(f"[info] {summary['num_upscaled_below_min']} images will be upscaled "
          f"(native < min_pixels={min_pixels:,})")
    print(f"[info] {summary['num_downscaled_above_max']} images will be downscaled "
          f"(native > max_pixels={max_pixels:,}) -- check these for potential text loss")
    print(f"[info] full report written to {output_dir / '_image_size_report.json'}")


def run(images_dir: Path, output_dir: Path, manifest_path: Path, max_new_tokens: int,
        served_model: str, dtype: str, max_model_len: int, gpu_memory_utilization: float,
        batch_size: int, min_pixels: int, max_pixels: int):
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

    max_image_tokens = max_pixels // PATCH_PIXELS
    print(f"[info] {len(image_files)} images to annotate. Fixed prompt length: {FIXED_PROMPT_LEN_CHARS} chars.")
    print(f"[info] loading {served_model} via vLLM offline LLM() -- no HTTP server involved")
    print(f"[info] processing in batches of {batch_size} to bound memory usage")
    print(f"[info] min_pixels={min_pixels:,} max_pixels={max_pixels:,} "
          f"(up to ~{max_image_tokens} image tokens; max_model_len={max_model_len})")

    # Same engine args as the `vllm serve` invocation, plus min_pixels/max_pixels
    # forwarded to Qwen's image processor via mm_processor_kwargs. This is what
    # controls/monitors the resolution every image gets resized to before encoding.
    llm = LLM(
        model=served_model,
        dtype=dtype,
        limit_mm_per_prompt={"image": 1},
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        mm_processor_kwargs={
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)

    t0 = time.time()
    done_count = 0
    failures = []
    size_log = []

    # Only one batch's worth of PIL images / conversations is held in memory at
    # a time; vLLM still continuously batches/schedules requests within each
    # llm.chat() call.
    for batch_paths in chunked(image_files, batch_size):
        stems = [p.stem for p in batch_paths]
        conversations = [build_conversation(p, size_log) for p in batch_paths]

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

    summarize_sizes(size_log, min_pixels, max_pixels, output_dir)


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
    ap.add_argument("--max-model-len", default=8192, type=int,
                     help="Raised from 4096: prompt (~1k tokens) + max-new-tokens (1500) + "
                          "up to max-pixels/784 image tokens must all fit in this budget.")
    ap.add_argument("--gpu-memory-utilization", default=0.95, type=float)
    ap.add_argument("--batch-size", default=200, type=int,
                     help="Number of images to load and submit to llm.chat() per offline batch")
    ap.add_argument("--min-pixels", default=256 * PATCH_PIXELS, type=int,
                     help="Floor on resized image pixel count (default 256 tokens' worth). "
                          "Small/blurry scans get upscaled to at least this before encoding.")
    ap.add_argument("--max-pixels", default=2048 * PATCH_PIXELS, type=int,
                     help="Ceiling on resized image pixel count (default 2048 tokens' worth, "
                          "~1.6MP), tuned for legible small text on invoices. Raising this "
                          "improves fine-print/small-table accuracy but costs more tokens "
                          "and needs headroom in --max-model-len.")
    args = ap.parse_args()

    if not args.images_dir.exists():
        sys.exit(f"[error] images dir not found: {args.images_dir}")

    run(args.images_dir, args.output_dir, args.manifest, args.max_new_tokens,
        args.served_model, args.dtype, args.max_model_len, args.gpu_memory_utilization,
        args.batch_size, args.min_pixels, args.max_pixels)


if __name__ == "__main__":
    main()
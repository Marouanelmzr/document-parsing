import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional

from PIL import Image
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


MODEL_ID = "numind/NuExtract3"


# NuExtract template: leaf values are extraction types, not placeholders.
TARGET_TEMPLATE = {
    "supplier_name": "verbatim-string",
    "supplier_phone_number": "verbatim-string",
    "supplier_address": {
        "address": "verbatim-string",
        "street_number": "verbatim-string",
        "street_name": "verbatim-string",
        "po_box": "verbatim-string",
        "address_complement": "verbatim-string",
        "city": "verbatim-string",
        "postal_code": "verbatim-string",
        "state": "verbatim-string",
        "country": "country",
    },
    "customer_name": "verbatim-string",
    "customer_address": {
        "address": "verbatim-string",
        "street_number": "verbatim-string",
        "street_name": "verbatim-string",
        "po_box": "verbatim-string",
        "address_complement": "verbatim-string",
        "city": "verbatim-string",
        "postal_code": "verbatim-string",
        "state": "verbatim-string",
        "country": "country",
    },
    "invoice_number": "verbatim-string",
    "document_type": ["invoice", "tax_invoice"],
    "date": "date-time",
    "due_date": "date-time",
    "period": "verbatim-string",
    "locale": {
        "language": "verbatim-string",
        "country": "country",
        "currency": "currency",
    },
    "total_net": "number",
    "total_tax": "number",
    "total_amount": "number",
    "taxes": [{
        "rate": "number",
        "base": "number",
        "amount": "number",
    }],
    "line_items": [{
        "description": "verbatim-string",
        "quantity": "number",
        "unit_price": "number",
        "total_price": "number",
        "tax_amount": "number",
        "tax_rate": "number",
        "unit_measure": "verbatim-string",
    }],
}


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
"Prestations du", "Mois de", "Billing period". Often a month or date range; null if the invoice is \
a one-off with no stated period.
"""


# vLLM structured-output schema.
# This is separate from NuExtract's extraction template.
class Address(BaseModel):
    address: Optional[str] = None
    street_number: Optional[str] = None
    street_name: Optional[str] = None
    po_box: Optional[str] = None
    address_complement: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None


class Locale(BaseModel):
    language: Optional[str] = None
    country: Optional[str] = None
    currency: Optional[str] = None


class Tax(BaseModel):
    rate: Optional[float] = None
    base: Optional[float] = None
    amount: Optional[float] = None


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None
    tax_amount: Optional[float] = None
    tax_rate: Optional[float] = None
    unit_measure: Optional[str] = None


class InvoiceFields(BaseModel):
    supplier_name: Optional[str] = None
    supplier_phone_number: Optional[str] = None
    supplier_address: Optional[Address] = None
    customer_name: Optional[str] = None
    customer_address: Optional[Address] = None
    invoice_number: Optional[str] = None
    document_type: Optional[Literal["invoice", "tax_invoice"]] = None
    date: Optional[str] = None
    due_date: Optional[str] = None
    period: Optional[str] = None
    locale: Optional[Locale] = None
    total_net: Optional[float] = None
    total_tax: Optional[float] = None
    total_amount: Optional[float] = None
    taxes: List[Tax] = []
    line_items: List[LineItem] = []


class InvoiceExtraction(BaseModel):
    fields: InvoiceFields


FIXED_TEMPLATE_LEN_CHARS = len(json.dumps(TARGET_TEMPLATE))
FIXED_INSTRUCTIONS_LEN_CHARS = len(INSTRUCTIONS)

PATCH_PIXELS = 28 * 28


def extract_json(raw_text: str):
    cleaned = re.sub(r"```json|```", "", raw_text).strip()

    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        return None, cleaned

    try:
        return json.loads(cleaned[start:end + 1]), cleaned
    except json.JSONDecodeError:
        return None, cleaned


def build_conversation(img_path: Path, size_log: list):
    image = Image.open(img_path).convert("RGB")

    w, h = image.size
    size_log.append({
        "stem": img_path.stem,
        "width": w,
        "height": h,
        "pixels": w * h,
    })

    return [{
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": image},
        ],
    }]


def process_output(stem: str, raw_text: str, raw_dir: Path, output_dir: Path) -> bool:
    (raw_dir / f"{stem}.txt").write_text(raw_text)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed, cleaned = extract_json(raw_text)

        if parsed is None:
            print(f"[warn] guided-decoded output for {stem} was not valid JSON")
            return False

        print(f"[warn] {stem} needed extract_json fallback")

    record = {
        "fields": parsed.get("fields", parsed),
        "raw_text": raw_text,
        "rag": {"retrieved_document_id": stem},
    }

    with open(output_dir / f"{stem}.json", "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return True


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def summarize_sizes(size_log: list, min_pixels: int, max_pixels: int, output_dir: Path):
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

    print(
        f"[info] image size report: native pixels range "
        f"[{summary['min_pixels_seen']:,} .. {summary['max_pixels_seen']:,}], "
        f"mean {summary['mean_pixels_seen']:,.0f}"
    )
    print(
        f"[info] {summary['num_upscaled_below_min']} images will be upscaled "
        f"(native < min_pixels={min_pixels:,})"
    )
    print(
        f"[info] {summary['num_downscaled_above_max']} images will be downscaled "
        f"(native > max_pixels={max_pixels:,})"
    )


def run(
    images_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    max_new_tokens: int,
    served_model: str,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    batch_size: int,
    min_pixels: int,
    max_pixels: int,
    guided_decoding_backend: str,
    enable_prefix_caching: bool,
    enable_thinking: bool,
    temperature: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_text"
    raw_dir.mkdir(exist_ok=True)

    if manifest_path and manifest_path.exists():
        with open(manifest_path) as f:
            rows = list(csv.DictReader(f))
        image_files = [images_dir / r["filename"] for r in rows]
    else:
        image_files = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    image_files = [p for p in image_files if p.exists()]

    max_image_tokens = max_pixels // PATCH_PIXELS

    print(
        f"[info] {len(image_files)} images to annotate. "
        f"NuExtract template={FIXED_TEMPLATE_LEN_CHARS} chars, "
        f"instructions={FIXED_INSTRUCTIONS_LEN_CHARS} chars."
    )
    print(
        f"[info] loading {served_model} via vLLM offline LLM() "
        f"-- no HTTP server involved"
    )
    print(f"[info] processing in batches of {batch_size}")
    print(
        f"[info] min_pixels={min_pixels:,} max_pixels={max_pixels:,} "
        f"(up to ~{max_image_tokens} image tokens; "
        f"max_model_len={max_model_len})"
    )
    print(
        f"[info] schema enforced via guided decoding "
        f"(backend={guided_decoding_backend})"
    )

    llm = LLM(
        model=served_model,
        dtype=dtype,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        structured_outputs_config={
            "backend": guided_decoding_backend
        },
        enable_prefix_caching=enable_prefix_caching,
        mm_processor_kwargs={
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
    )

    structured_outputs_params = StructuredOutputsParams(
        json=InvoiceExtraction.model_json_schema()
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        structured_outputs=structured_outputs_params,
    )

    chat_template_kwargs = {
        "template": json.dumps(
            TARGET_TEMPLATE,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        "instructions": INSTRUCTIONS,
        "enable_thinking": enable_thinking,
    }

    t0 = time.time()
    done_count = 0
    failures = []
    size_log = []

    for batch_paths in chunked(image_files, batch_size):
        stems = [p.stem for p in batch_paths]
        conversations = [
            build_conversation(p, size_log)
            for p in batch_paths
        ]

        outputs = llm.chat(
            conversations,
            sampling_params,
            chat_template_kwargs=chat_template_kwargs,
            use_tqdm=False,
        )

        for stem, output in zip(stems, outputs):
            raw_text = output.outputs[0].text

            if not process_output(
                stem,
                raw_text,
                raw_dir,
                output_dir,
            ):
                failures.append(stem)

        done_count += len(batch_paths)
        elapsed = time.time() - t0

        print(
            f"[info] {done_count}/{len(image_files)} done "
            f"({elapsed:.0f}s elapsed)"
        )

    elapsed = time.time() - t0
    n = len(image_files)

    print(
        f"[done] {n - len(failures)} succeeded, "
        f"{len(failures)} failed, "
        f"in {elapsed:.1f}s ({n / elapsed:.2f} img/s)"
    )

    if failures:
        (output_dir / "_failed_parses.txt").write_text(
            "\n".join(failures)
        )
        print(
            f"[info] failed ids written to "
            f"{output_dir / '_failed_parses.txt'}"
        )

    summarize_sizes(
        size_log,
        min_pixels,
        max_pixels,
        output_dir,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--images-dir", required=True, type=Path)

    ap.add_argument(
        "--output-dir",
        default=Path("./nuextract3_annotations_vllm_offline"),
        type=Path,
    )

    ap.add_argument(
        "--manifest",
        default=None,
        type=Path,
        help="Optional manifest.csv using filename column",
    )

    ap.add_argument("--max-new-tokens", default=1500, type=int)

    ap.add_argument(
        "--served-model",
        default=MODEL_ID,
        help="HF repo id or local path passed to vllm.LLM()",
    )

    ap.add_argument("--dtype", default="float16")

    ap.add_argument("--max-model-len", default=8192, type=int)

    ap.add_argument(
        "--gpu-memory-utilization",
        default=0.95,
        type=float,
    )

    ap.add_argument(
        "--batch-size",
        default=200,
        type=int,
    )

    ap.add_argument(
        "--min-pixels",
        default=256 * PATCH_PIXELS,
        type=int,
    )

    ap.add_argument(
        "--max-pixels",
        default=2048 * PATCH_PIXELS,
        type=int,
    )

    ap.add_argument(
        "--guided-decoding-backend",
        default="xgrammar",
    )

    ap.add_argument(
        "--enable-prefix-caching",
        dest="enable_prefix_caching",
        action="store_true",
        default=True,
    )

    ap.add_argument(
        "--no-prefix-caching",
        dest="enable_prefix_caching",
        action="store_false",
    )

    ap.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=False,
    )

    ap.add_argument(
        "--temperature",
        default=0.2,
        type=float,
    )

    args = ap.parse_args()

    if not args.images_dir.exists():
        sys.exit(f"[error] images dir not found: {args.images_dir}")

    run(
        args.images_dir,
        args.output_dir,
        args.manifest,
        args.max_new_tokens,
        args.served_model,
        args.dtype,
        args.max_model_len,
        args.gpu_memory_utilization,
        args.batch_size,
        args.min_pixels,
        args.max_pixels,
        args.guided_decoding_backend,
        args.enable_prefix_caching,
        args.enable_thinking,
        args.temperature,
    )


if __name__ == "__main__":
    main()
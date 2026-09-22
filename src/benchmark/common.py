"""Everything that is *identical* across the 3 models: the Pydantic schema
used for guided decoding, JSON-fallback parsing, output writing, and the
generic offline-vLLM extraction loop. Model-specific bits (prompt/adapter,
tokenizer flags, batch sizes) live in bench_adapters.py / bench_configs.py.
"""
import json
import re
import time
from pathlib import Path
from typing import Callable, List, Literal, Optional

from PIL import Image
from pydantic import BaseModel


# Schema (single source of truth for guided decoding on every model)
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

class Discount(BaseModel): 
    rate: Optional[float] = None
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
    customer_phone_number: Optional[str] = None
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
    discount: Optional[Discount] = None
    line_items: List[LineItem] = []


class InvoiceExtraction(BaseModel):
    fields: InvoiceFields


# Qwen resizes to multiples of a 28x28 patch (14px ViT patch, 2x2 merge).
# 1 visual token == 784 pixels after resizing. Reused for the image-size
# report regardless of which model actually consumes it.
PATCH_PIXELS = 28 * 28


# Parsing / IO
def extract_json(raw_text: str):
    """Fallback parser only -- guided decoding should make this unnecessary."""
    cleaned = re.sub(r"```json|```", "", raw_text).strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return None, cleaned
    try:
        return json.loads(cleaned[start:end + 1]), cleaned
    except json.JSONDecodeError:
        return None, cleaned


def load_image(img_path: Path, size_log: list) -> Image.Image:
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    size_log.append({"stem": img_path.stem, "width": w, "height": h, "pixels": w * h})
    return image


def process_output(stem: str, raw_text: str, raw_dir: Path, output_dir: Path) -> bool:
    (raw_dir / f"{stem}.txt").write_text(raw_text)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed, _ = extract_json(raw_text)
        if parsed is None:
            print(f"[warn] {stem}: not valid JSON (unexpected) -- see raw_text/{stem}.txt")
            return False
        print(f"[warn] {stem}: needed extract_json fallback despite guided decoding")

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


def list_images(images_dir: Path) -> List[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def summarize_sizes(size_log: list, min_pixels: int, max_pixels: int, output_dir: Path):
    if not size_log:
        return
    pixels = [r["pixels"] for r in size_log]
    below = [r for r in size_log if r["pixels"] < min_pixels]
    above = [r for r in size_log if r["pixels"] > max_pixels]
    summary = {
        "count": len(size_log),
        "min_pixels_seen": min(pixels),
        "max_pixels_seen": max(pixels),
        "mean_pixels_seen": sum(pixels) / len(pixels),
        "configured_min_pixels": min_pixels,
        "configured_max_pixels": max_pixels,
        "num_upscaled_below_min": len(below),
        "num_downscaled_above_max": len(above),
        "upscaled_examples": [r["stem"] for r in below[:10]],
        "downscaled_examples": [r["stem"] for r in above[:10]],
    }
    with open(output_dir / "_image_size_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[info] pixel range [{summary['min_pixels_seen']:,}..{summary['max_pixels_seen']:,}], "
          f"mean {summary['mean_pixels_seen']:,.0f}; "
          f"{summary['num_upscaled_below_min']} upscaled, {summary['num_downscaled_above_max']} downscaled")


# Generic runner -- identical control flow for every model
def run_extraction(*, images_dir: Path, output_dir: Path, llm_kwargs: dict,
                    sampling_params, build_conversation: Callable,
                    chat_template_kwargs: Optional[dict] = None,
                    batch_size: int = 400, min_pixels: Optional[int] = None,
                    max_pixels: Optional[int] = None):
    """One-model, one-process extraction run. build_conversation(img_path,
    size_log) must return a vLLM chat conversation. chat_template_kwargs is
    only needed by NuExtract-style models; leave None for standard chat VLMs.
    """
    from vllm import LLM  # imported here so a single process only ever loads one LLM

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_text"
    raw_dir.mkdir(exist_ok=True)

    image_files = list_images(images_dir)
    print(f"[info] {len(image_files)} images -- loading {llm_kwargs['model']}")
    llm = LLM(**llm_kwargs)

    t0 = time.time()
    done, failures, size_log = 0, [], []
    for batch_paths in chunked(image_files, batch_size):
        stems = [p.stem for p in batch_paths]
        conversations = [build_conversation(p, size_log) for p in batch_paths]
        extra = {"chat_template_kwargs": chat_template_kwargs} if chat_template_kwargs else {}
        outputs = llm.chat(conversations, sampling_params, use_tqdm=True, **extra)
        for stem, output in zip(stems, outputs):
            if not process_output(stem, output.outputs[0].text, raw_dir, output_dir):
                failures.append(stem)
        done += len(batch_paths)
        print(f"[info] {done}/{len(image_files)} done ({time.time() - t0:.0f}s elapsed)")

    elapsed = time.time() - t0
    n = len(image_files)
    print(f"[done] {n - len(failures)} ok, {len(failures)} failed, "
          f"{elapsed:.1f}s ({n / elapsed:.2f} img/s)")
    if failures:
        (output_dir / "_failed_parses.txt").write_text("\n".join(failures))
    if min_pixels and max_pixels:
        summarize_sizes(size_log, min_pixels, max_pixels, output_dir)
"""Dataset, bucketed sampler and collate_fn for LoRA fine-tuning Qwen3-VL-4B
on the invoice-extraction task defined in src/benchmark/{common,adapters}.py.

Design goals (all in service of "keep the H100 fed, don't waste VRAM on
padding"):

1. Cheap length estimation at __init__ time (PIL header read only -- no
   image decode, no tokenization) so we can bucket samples by estimated
   sequence length *before* touching the GPU-bound processing path.
2. Bucketed batches: samples of similar total length end up in the same
   batch, so padding waste (and therefore wasted FLOPs/VRAM) is minimal.
   This matters a lot here because image length varies with page size and
   text length varies with how many line items an invoice has.
3. Qwen's processor already avoids padding *image* tokens -- pixel_values
   for every image in a batch are concatenated along dim 0 as flat patch
   sequences (image_grid_thw says how many patches each image contributed),
   so batching images together is "free" (concat, not pad-to-max).
4. Labels are masked so loss is only computed on the assistant's JSON
   completion, never on the prompt/schema/image tokens.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

# Reuse the repo's own prompt/schema so training data is byte-identical to
# what the benchmark/inference path sends the model.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src" / "benchmark"
sys.path.insert(0, str(_REPO_SRC))
from src.benchmark.adapters import VISION_CHAT_PROMPT  # noqa: E402


@dataclass
class Example:
    image_path: Path
    label_path: Path
    stem: str


def discover_examples(images_dir: Path, labels_dir: Path) -> List[Example]:
    """Matches split_dataset.py's output layout: <split>/images/*.{jpg,png},
    <split>/labels/*.json, same stem."""
    examples = []
    for label_path in sorted(labels_dir.glob("*.json")):
        stem = label_path.stem
        matches = sorted(
            p for p in images_dir.glob(f"{stem}.*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not matches:
            continue
        examples.append(Example(matches[0], label_path, stem))
    if not examples:
        raise FileNotFoundError(f"No (image,label) pairs found under {images_dir} / {labels_dir}")
    return examples


def _cheap_length_estimate(ex: Example, patch_pixels: int = 28 * 28,
                            min_pixels: int = 256 * 28 * 28,
                            max_pixels: int = 2048 * 28 * 28) -> int:
    """Header-only image read (no decode) + rough text length, used purely
    for bucketing. Doesn't need to be exact, just monotonic-ish."""
    try:
        with Image.open(ex.image_path) as im:
            w, h = im.size
    except Exception:
        w, h = 1024, 1024
    pixels = max(min_pixels, min(max_pixels, w * h))
    image_tokens = pixels // patch_pixels
    try:
        text_len = ex.label_path.stat().st_size // 3  # ~chars/token, rough
    except OSError:
        text_len = 200
    return image_tokens + text_len + len(VISION_CHAT_PROMPT) // 3


class InvoiceVLDataset(Dataset):
    """One item = one invoice image + its target JSON (schema: InvoiceExtraction).

    __getitem__ returns *unpadded* tensors for a single example; padding and
    pixel_values concatenation happen in `collate_fn` so batches don't carry
    more padding than the batch actually needs.
    """

    def __init__(self, images_dir: Path, labels_dir: Path, processor,
                 max_length: int = 4096):
        self.examples = discover_examples(Path(images_dir), Path(labels_dir))
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        # Precomputed once, cheaply -- used by LengthGroupedSampler.
        self.lengths = [_cheap_length_estimate(ex) for ex in self.examples]

    def __len__(self):
        return len(self.examples)

    def _target_json(self, label_path: Path) -> str:
        with open(label_path) as f:
            record = json.load(f)
        fields = record.get("fields", record)
        return json.dumps({"fields": fields}, ensure_ascii=False)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        image = Image.open(ex.image_path).convert("RGB")
        target = self._target_json(ex.label_path)

        user_msg = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": VISION_CHAT_PROMPT},
            ],
        }]
        # add_generation_prompt=True appends the "<|im_start|>assistant\n"
        # preamble; calling the *processor* (not just the tokenizer) on this
        # expands the image placeholder into the real number of image
        # tokens for this image, so its length lines up exactly with the
        # prefix of the full (prompt+answer) encoding below.
        prompt_text = self.processor.apply_chat_template(
            user_msg, tokenize=False, add_generation_prompt=True)
        prompt_enc = self.processor(
            text=[prompt_text], images=[image], return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]

        full_msg = user_msg + [{"role": "assistant", "content": target}]
        full_text = self.processor.apply_chat_template(
            full_msg, tokenize=False, add_generation_prompt=False)
        full_enc = self.processor(
            text=[full_text], images=[image], return_tensors="pt")

        input_ids = full_enc["input_ids"][0][: self.max_length]
        mm_token_type_ids = full_enc["mm_token_type_ids"][0][: self.max_length]
        labels = input_ids.clone()
        labels[:min(prompt_len, len(labels))] = -100

        item = {
            "input_ids": input_ids,
            "labels": labels,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": full_enc["pixel_values"],
            "image_grid_thw": full_enc["image_grid_thw"],
        }
        return item


class LengthGroupedSampler(Sampler):
    """Groups samples of similar (precomputed) length into the same batch to
    minimize padding, then shuffles batch order each epoch. Standard
    "sort-within-megabatch" trick (same idea as HF Trainer's
    LengthGroupedSampler), reimplemented here so it also accounts for the
    image-token contribution to length, not just text length.
    """

    def __init__(self, lengths: List[int], batch_size: int, mega_batch_mult: int = 50,
                 seed: int = 0, shuffle: bool = True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.mega_batch_size = batch_size * mega_batch_mult
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        n = len(self.lengths)
        indices = torch.randperm(n, generator=g).tolist() if self.shuffle else list(range(n))

        batches = []
        for start in range(0, n, self.mega_batch_size):
            mega = indices[start:start + self.mega_batch_size]
            mega.sort(key=lambda i: self.lengths[i])
            for b_start in range(0, len(mega), self.batch_size):
                batches.append(mega[b_start:b_start + self.batch_size])

        if self.shuffle:
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]

        for batch in batches:
            for idx in batch:
                yield idx


def make_collate_fn(pad_token_id: int):
    def collate_fn(items: List[dict]):
        max_len = max(item["input_ids"].shape[0] for item in items)
        B = len(items)

        input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)

        for i, item in enumerate(items):
            L = item["input_ids"].shape[0]
            # Right-padding: real tokens first, keeps causal-mask semantics
            # simple and matches Qwen's default packing/eval convention.
            input_ids[i, :L] = item["input_ids"]
            attention_mask[i, :L] = 1
            labels[i, :L] = item["labels"]

        # pixel_values / image_grid_thw: Qwen concatenates patches across
        # every image in the batch (no padding needed at all here -- this
        # is the main reason batching images is cheap with this model).
        pixel_values = torch.cat([item["pixel_values"] for item in items], dim=0)
        image_grid_thw = torch.cat([item["image_grid_thw"] for item in items], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
    return collate_fn

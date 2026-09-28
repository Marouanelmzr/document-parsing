"""Multi-adapter ("P-LoRA") trainer.

Core idea: load the Qwen3-VL-4B backbone ONCE, frozen, in bf16 (~8GB on an
H100). Attach several LoRA adapters (different rank/alpha/dropout/lr) to
that *same* model instance as named PEFT adapters. Each step:

    batch = next(train_iter)                      # one shared, real batch
    for name in live_trials:                       # round-robin
        model.set_adapter(name)                    # swap which LoRA is active
        loss = model(**batch).loss                 # only that adapter's params
        loss.backward()                            #   are in the autograd graph
        optimizers[name].step(); optimizers[name].zero_grad()

Why this maximizes utilization on one H100 instead of just running N
sequential single-adapter jobs:
  - The 8GB frozen backbone is loaded/resident once, not N times -- no
    repeated model-load overhead per trial, and no risk of any one trial's
    process sitting idle waiting on `from_pretrained`.
  - Every trial's optimizer state is tiny (LoRA-only), so N trials fit
    comfortably alongside one backbone in 80GB, even N=8+ at these ranks.
  - Data loading (the usual GPU-starvation culprit) is shared: one prefetch
    queue feeds every trial's step, instead of N separate DataLoaders each
    fighting for CPU workers.
  - ASHA-style pruning frees a trial's adapter (and its optimizer state)
    the moment it's clearly behind, so compute concentrates on the
    survivors instead of babysitting configs that were never going to win.

This does NOT batch multiple adapters into a single fused matmul (that
needs custom grouped-GEMM kernels, e.g. what vLLM's multi-LoRA serving path
does for inference) -- each adapter's forward/backward is still a normal
sequential PyTorch call. What you get here is memory sharing + zero
idle/reload time + fair, identical-batch comparisons, which is the large
majority of the practical benefit for a training-time hyperparameter sweep.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import random

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from src.training.lora_trials import TrialConfig
from src.benchmark.evaluate import evaluate as bench_evaluate



@dataclass
class TrainConfig:
    total_steps: int            # number of OPTIMIZER steps (each may span grad_accum_steps micro-batches)
    eval_every: int
    rungs: List[float]          # fractions of total_steps at which to prune, e.g. [0.25, 0.5, 0.75]
    keep_fraction: float = 0.5  # fraction of live trials kept at each rung
    grad_accum_steps: int = 1   # effective_batch = micro_batch_size * grad_accum_steps
    grad_clip: float = 1.0
    val_loss_batches: int = 20  # cheap proxy metric used for pruning
    final_eval_examples: int = 40  # expensive generate()-based metric, only for the survivors at the end
    ckpt_dir: Path = Path("checkpoints")
    gen_max_new_tokens: int = 1600

def _parse_lenient(text: str):
    t = text.strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        obj = json.loads(t[s:e + 1])
    except json.JSONDecodeError:
        return None
    return obj.get("fields", obj) if isinstance(obj, dict) else None


def _safe_score(pred_fields, gold_fields) -> float:
    # The benchmark scorer assumes well-formed structures. A malformed
    # prediction (e.g. a list of strings) should score 0, not crash the run.
    try:
        return bench_evaluate({"fields": pred_fields or {}}, {"fields": gold_fields})["accuracy"]
    except Exception:
        return 0.0

class MultiLoRATrainer:
    def __init__(self, base_model, processor, trials: List[TrialConfig],
                 train_loader, val_loader, cfg: TrainConfig, device: str = "cuda"):
        self.model = base_model
        self.processor = processor
        self.device = device
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trials: Dict[str, TrialConfig] = {t.name: t for t in trials}
        self.live = list(self.trials.keys())
        self.eliminated: Dict[str, float] = {}

        self.optimizers = {}
        self.schedulers = {}
        self._attach_adapters()

    # ---- setup -----------------------------------------------------
    def _lora_config(self, t: TrialConfig) -> LoraConfig:
        return LoraConfig(
            r=t.r, lora_alpha=t.lora_alpha, lora_dropout=t.lora_dropout,
            target_modules=t.peft_target_modules(), bias="none",
            task_type="CAUSAL_LM",
        )

    def _attach_adapters(self):
        first, *rest = list(self.trials.values())
        self.model = get_peft_model(self.model, self._lora_config(first), adapter_name=first.name)
        for t in rest:
            self.model.add_adapter(t.name, self._lora_config(t))
        self.model.to(self.device)

        for t in self.trials.values():
            self.model.set_adapter(t.name)
            # Only this adapter's params are trainable while it's active;
            # collect them explicitly so each optimizer only ever touches
            # its own adapter's tensors (safe even though `requires_grad`
            # on inactive adapters' params is technically also True -- they
            # never enter this adapter's autograd graph).
            params = [p for n, p in self.model.named_parameters()
                      if t.name in n and p.requires_grad]
            self.optimizers[t.name] = AdamW(
                params, lr=t.lr, weight_decay=t.weight_decay, fused=True)
            self.schedulers[t.name] = get_cosine_schedule_with_warmup(
                self.optimizers[t.name],
                num_warmup_steps=int(t.warmup_ratio * self.cfg.total_steps),
                num_training_steps=self.cfg.total_steps,
            )

    # ---- training ----------------------------------------------------
    def _accumulate_adapter(self, name: str, batch: dict) -> float:
        """One micro-batch forward+backward for `name`, scaled for
        accumulation. Does NOT step the optimizer -- call `_flush_adapter`
        after `grad_accum_steps` calls to this."""
        self.model.set_adapter(name)
        self.model.train()
        out = self.model(**batch)
        loss = out.loss / self.cfg.grad_accum_steps
        loss.backward()
        return out.loss.item()

    def _flush_adapter(self, name: str):
        torch.nn.utils.clip_grad_norm_(
            [p for n, p in self.model.named_parameters() if name in n and p.requires_grad],
            self.cfg.grad_clip)
        self.optimizers[name].step()
        self.schedulers[name].step()
        self.optimizers[name].zero_grad(set_to_none=True)

    @torch.no_grad()
    def _val_loss(self, name: str) -> float:
        self.model.set_adapter(name)
        self.model.eval()
        losses = []
        it = iter(self.val_loader)
        for _ in range(self.cfg.val_loss_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            losses.append(self.model(**batch).loss.item())
        return sum(losses) / max(1, len(losses))


    @torch.no_grad()
    def _val_field_accuracy(self, name, n_examples, batch_size=4, log=print):
        if name is not None:
            self.model.set_adapter(name)
        self.model.eval()
        ds = self.val_loader.dataset
        idxs = random.Random(0).sample(range(len(ds)), min(n_examples, len(ds)))  # fixed, unbiased
        tok = self.processor.tokenizer
        old_side, tok.padding_side = tok.padding_side, "left"
        scores, floors, parse_fail = [], [], 0
        try:
            for s in range(0, len(idxs), batch_size):
                chunk = idxs[s:s + batch_size]
                texts, images, golds = zip(*(ds.prompt_and_gold(i) for i in chunk))
                enc = self.processor(text=list(texts), images=list(images),
                                     return_tensors="pt", padding=True).to(self.device)
                gen = self.model.generate(**enc, max_new_tokens=self.cfg.gen_max_new_tokens,
                                          do_sample=False, use_cache=True)
                plen = enc["input_ids"].shape[1]
                for j, gold in enumerate(golds):
                    pred = _parse_lenient(tok.decode(gen[j, plen:], skip_special_tokens=True))
                    parse_fail += pred is None
                    scores.append(_safe_score(pred, gold))
                    floors.append(_safe_score({}, gold))
                    if s == 0 and j == 0:
                        log(f"[eval-debug] raw output: {tok.decode(gen[j, plen:], skip_special_tokens=True)[:400]!r}")
        finally:
            tok.padding_side = old_side
        n = max(1, len(scores))
        log(f"[eval] {name}: acc={sum(scores)/n:.4f} empty-pred floor={sum(floors)/n:.4f} "
            f"parse_fail={parse_fail}/{n}")
        return sum(scores) / n

    def _maybe_prune(self, step: int, log):
        rung_steps = {int(r * self.cfg.total_steps) for r in self.cfg.rungs}
        if step not in rung_steps or len(self.live) <= 1:
            return
        scored = [(name, self._val_loss(name)) for name in self.live]
        scored.sort(key=lambda x: x[1])  # lower val loss first
        keep_n = max(1, math.ceil(len(scored) * self.cfg.keep_fraction))
        survivors = {name for name, _ in scored[:keep_n]}
        for name, vloss in scored:
            log(f"[rung step={step}] {name}: val_loss={vloss:.4f}"
                f"{'  -> pruned' if name not in survivors else '  -> kept'}")
            if name not in survivors:
                self.eliminated[name] = vloss
                self.model.delete_adapter(name)
                del self.optimizers[name]
                del self.schedulers[name]
        self.live = [n for n in self.live if n in survivors]

    def _next_batch(self, train_iter):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(self.train_loader)
            batch = next(train_iter)
        return train_iter, {k: v.to(self.device) for k, v in batch.items()}

    def run(self, log=print) -> str:
        """Runs the interleaved sweep; returns the name of the best trial.
        Each of the `total_steps` optimizer steps consumes
        `grad_accum_steps` micro-batches per live adapter -- all adapters
        see the same sequence of micro-batches, so a comparison at any rung
        is apples-to-apples."""
        with self.model.disable_adapter():
            base_acc = self._val_field_accuracy(None, min(100, self.cfg.final_eval_examples), log=log)
        log(f"[control] base model (no adapter) acc={base_acc:.4f}")
        train_iter = iter(self.train_loader)
        t0 = time.time()
        for step in range(1, self.cfg.total_steps + 1):
            last_losses = {}
            for _ in range(self.cfg.grad_accum_steps):
                train_iter, batch = self._next_batch(train_iter)
                for name in self.live:
                    last_losses[name] = self._accumulate_adapter(name, batch)
            for name in self.live:
                self._flush_adapter(name)
            losses = last_losses

            if step % self.cfg.eval_every == 0 or step == self.cfg.total_steps:
                elapsed = time.time() - t0
                loss_str = " ".join(f"{n}={l:.3f}" for n, l in losses.items())
                log(f"step {step}/{self.cfg.total_steps} ({elapsed:.0f}s) train_loss: {loss_str}")

            self._maybe_prune(step, log)

        log(f"[final] running field-accuracy eval on {len(self.live)} survivor(s): {self.live}")
        final_scores = {name: self._val_field_accuracy(name, self.cfg.final_eval_examples)
                         for name in self.live}
        for name, acc in sorted(final_scores.items(), key=lambda x: -x[1]):
            log(f"[final] {name}: field_accuracy={acc:.4f}")
        best = max(final_scores, key=final_scores.get)
        log(f"[final] best adapter: {best} (field_accuracy={final_scores[best]:.4f})")
        self._save_summary(final_scores, best)
        return best

    # ---- saving ------------------------------------------------------
    def _save_summary(self, final_scores: dict, best: str):
        self.cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "trials": {name: vars(t) for name, t in self.trials.items()},
            "eliminated_val_loss": self.eliminated,
            "final_field_accuracy": final_scores,
            "best": best,
        }
        with open(self.cfg.ckpt_dir / "sweep_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def save_best_adapter(self, best: str, out_dir: Path):
        """Saves ONLY the winning adapter's weights (not the others, not the
        frozen backbone). Safe to call after `run()`; also works if `best`
        is the only trial left (no sweep, single-adapter mode)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.set_adapter(best)
        try:
            # Newer peft: save only this adapter's subfolder directly.
            self.model.save_pretrained(out_dir, selected_adapters=[best])
        except TypeError:
            # Older peft: build the state dict for this adapter manually.
            state_dict = get_peft_model_state_dict(self.model, adapter_name=best)
            (out_dir / best).mkdir(exist_ok=True)
            torch.save(state_dict, out_dir / best / "adapter_model.bin")
            self._lora_config(self.trials[best]).save_pretrained(out_dir / best)
        print(f"[info] saved winning adapter '{best}' -> {out_dir}")

    def merge_and_save(self, best: str, base_model_id: str, out_dir: Path):
        """Optional: produce a merged full-precision checkpoint (base +
        winning adapter) for serving with vLLM without a LoRA flag at all.
        Loads a fresh base model copy so the merge doesn't disturb the live
        multi-adapter model still in memory."""
        from transformers import AutoModelForImageTextToText
        from peft import PeftModel

        tmp_adapter_dir = out_dir.parent / f"_tmp_{best}"
        self.save_best_adapter(best, tmp_adapter_dir)

        fresh_base = AutoModelForImageTextToText.from_pretrained(
            base_model_id, torch_dtype=torch.bfloat16)
        merged = PeftModel.from_pretrained(fresh_base, tmp_adapter_dir, adapter_name=best)
        merged = merged.merge_and_unload()
        merged.save_pretrained(out_dir, safe_serialization=True)
        self.processor.save_pretrained(out_dir)
        shutil.rmtree(tmp_adapter_dir, ignore_errors=True)
        print(f"[info] merged model saved -> {out_dir}")

"""Evaluation utility: measures the four objectives from the paper's
problem formulation (Eqs. 2-5) so attack effectiveness is reproducible
and independently checkable, not just self-reported by the attack loop.

``cl_func``'s contract, as used here and in :mod:`~DLCLAttack.attacker`:
called with a model, it must return a *compiled* callable that accepts the
same ``(inputs_embeds=..., attention_mask=...)`` keyword arguments as the
original model and returns something logits-shaped -- a
``transformers.ModelOutput`` (has ``.logits``), a raw tensor, or an
array-like convertible via ``torch.as_tensor`` (covers most
tensor/ndarray-returning compiler runtimes).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .adapters import extract_decision_logits
from .data import iterate_batches, to_embeds
from .trigger import EmbeddingTrigger


@dataclass
class EvaluationReport:
    pre_compile_clean_accuracy: float
    pre_compile_attack_success_rate: float
    post_compile_clean_accuracy: float
    post_compile_attack_success_rate: float
    consistency_rate: float
    num_examples: int

    def to_dict(self) -> dict:
        return {
            "pre_compile_clean_accuracy": self.pre_compile_clean_accuracy,
            "pre_compile_attack_success_rate": self.pre_compile_attack_success_rate,
            "post_compile_clean_accuracy": self.post_compile_clean_accuracy,
            "post_compile_attack_success_rate": self.post_compile_attack_success_rate,
            "consistency_rate": self.consistency_rate,
            "num_examples": self.num_examples,
        }


def _logits_from_output(output) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0:
        return torch.as_tensor(output[0])
    return torch.as_tensor(output)


def _predict(callable_model, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    output = callable_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = extract_decision_logits(_logits_from_output(output))
    return logits.argmax(dim=-1)


def clean_accuracy(
    model: nn.Module,
    dataset,
    device: torch.device,
    label_key: str = "label",
    batch_size: int = 8,
    max_batches: int | None = None,
    callable_model=None,
) -> float:
    """Fraction of clean ``dataset`` examples predicted correctly.

    ``callable_model`` lets the caller evaluate a *compiled* stand-in
    (``cl_func(model)``) while still using ``model`` for tokenization-free
    embedding lookup; defaults to ``model`` itself.
    """
    callable_model = callable_model or model
    correct, total = 0, 0
    with torch.no_grad():
        for batch in iterate_batches(dataset, batch_size, label_key, max_batches=max_batches):
            inputs_embeds, attention_mask = to_embeds(model, batch, device)
            preds = _predict(callable_model, inputs_embeds, attention_mask)
            labels = batch["labels"].to(device)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total if total else 0.0


def attack_success_rate(
    model: nn.Module,
    dataset,
    trigger: EmbeddingTrigger,
    target_label: int,
    device: torch.device,
    label_key: str = "label",
    batch_size: int = 8,
    max_batches: int | None = None,
    callable_model=None,
) -> float:
    """Fraction of triggered examples classified as ``target_label``.

    Examples whose ground-truth label already equals ``target_label`` are
    excluded, since they cannot demonstrate the trigger changed anything.
    """
    callable_model = callable_model or model
    hit, total = 0, 0
    with torch.no_grad():
        for batch in iterate_batches(dataset, batch_size, label_key, max_batches=max_batches):
            mask = batch["labels"] != target_label
            if not mask.any():
                continue
            filtered = {k: v[mask] for k, v in batch.items()}
            inputs_embeds, attention_mask = to_embeds(model, filtered, device)
            triggered_embeds, triggered_mask = trigger.apply(inputs_embeds, attention_mask)
            preds = _predict(callable_model, triggered_embeds, triggered_mask)
            hit += (preds == target_label).sum().item()
            total += preds.numel()
    return hit / total if total else 0.0


def consistency_rate(
    model: nn.Module,
    compiled_model,
    dataset,
    device: torch.device,
    label_key: str = "label",
    batch_size: int = 8,
    max_batches: int | None = None,
) -> float:
    """Fraction of clean examples where the pre- and post-compilation
    predictions agree -- the "victim cannot notice anything changed"
    check (post-compilation utility objective, Eq. 5's companion metric)."""
    agree, total = 0, 0
    with torch.no_grad():
        for batch in iterate_batches(dataset, batch_size, label_key, max_batches=max_batches):
            inputs_embeds, attention_mask = to_embeds(model, batch, device)
            pre_preds = _predict(model, inputs_embeds, attention_mask)
            post_preds = _predict(compiled_model, inputs_embeds, attention_mask)
            agree += (pre_preds == post_preds).sum().item()
            total += pre_preds.numel()
    return agree / total if total else 0.0


def evaluate_attack(
    model: nn.Module,
    cl_func,
    dataset,
    trigger: EmbeddingTrigger,
    target_label: int,
    device: torch.device,
    label_key: str = "label",
    batch_size: int = 8,
    max_batches: int | None = None,
) -> EvaluationReport:
    """Full reproducible attack-effectiveness report on ``dataset``.

    Compiles ``model`` once via ``cl_func`` and computes all four
    objective metrics plus pre/post consistency, matching how the paper
    reports results: high pre- and post-compilation clean accuracy, near
    chance-level pre-compilation ASR (the trigger should do nothing before
    compiling), and high post-compilation ASR (the trigger works after).
    """
    compiled_model = cl_func(model)

    pre_acc = clean_accuracy(model, dataset, device, label_key, batch_size, max_batches)
    pre_asr = attack_success_rate(
        model, dataset, trigger, target_label, device, label_key, batch_size, max_batches
    )
    post_acc = clean_accuracy(
        model, dataset, device, label_key, batch_size, max_batches, callable_model=compiled_model
    )
    post_asr = attack_success_rate(
        model, dataset, trigger, target_label, device, label_key, batch_size, max_batches,
        callable_model=compiled_model,
    )
    consistency = consistency_rate(model, compiled_model, dataset, device, label_key, batch_size, max_batches)

    num_examples = min(len(dataset), batch_size * max_batches) if max_batches else len(dataset)

    return EvaluationReport(
        pre_compile_clean_accuracy=pre_acc,
        pre_compile_attack_success_rate=pre_asr,
        post_compile_clean_accuracy=post_acc,
        post_compile_attack_success_rate=post_asr,
        consistency_rate=consistency,
        num_examples=num_examples,
    )

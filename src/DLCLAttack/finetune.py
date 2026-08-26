"""Sub-model 2 fine-tuning (Eq. 7): the step that actually plants the backdoor.

Four losses, one per input-model combination from the problem formulation,
summed and back-propagated into sub-model 2's parameters only (sub-model 1
and the embeddings are frozen by the caller before this runs):

    l1 = CE(M2(M1(x)      - V), y)     pre-compilation utility
    l2 = CE(M2(M1(x ⊕ t)  - V), y)     pre-compilation stealth (trigger is a no-op before compiling)
    l3 = CE(M2(C1(x)      - V), y)     post-compilation utility
    l4 = CE(M2(C1(x ⊕ t)  - V), y*)    post-compilation effectiveness (the actual backdoor)

``C1`` (the compiled sub-model 1) is used strictly forward-only here --
gradients never need to flow through it, so it can be a genuinely opaque
``cl_func`` output (an ONNX/TVM/torch.compile artifact, none of which are
guaranteed differentiable).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .adapters import SplitPoint, SubModel1, extract_decision_logits, run_submodel2
from .trigger import EmbeddingTrigger


@dataclass
class FinetuneResult:
    epoch_losses: list[float] = field(default_factory=list)
    per_term_losses: list[dict] = field(default_factory=list)  # one dict {l1,l2,l3,l4} per step


def finetune_submodel2(
    model: nn.Module,
    split_point: SplitPoint,
    sub_model_1: SubModel1,
    compiled_sub_model_1,
    trigger: EmbeddingTrigger,
    guard_bias: torch.Tensor,
    target_label: int,
    device: torch.device,
    train_batches: list[dict],
    epochs: int,
    lr: float,
    loss_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> FinetuneResult:
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "finetune_submodel2 found no trainable parameters; did "
            "adapters.unfreeze_submodel2 run before this?"
        )
    optimizer = torch.optim.Adam(trainable, lr=lr)
    ce = nn.CrossEntropyLoss()
    result = FinetuneResult()

    for _epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_batches:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            target = torch.full_like(labels, target_label)

            with torch.no_grad():
                embedding_layer = model.get_input_embeddings()
                clean_embeds = embedding_layer(input_ids)
                triggered_embeds, triggered_mask = trigger.apply(clean_embeds, attention_mask)

                h_m_clean = sub_model_1(clean_embeds, attention_mask).detach()
                h_m_trig = sub_model_1(triggered_embeds, triggered_mask).detach()
                h_c_clean = compiled_sub_model_1(clean_embeds, attention_mask)
                h_c_trig = compiled_sub_model_1(triggered_embeds, triggered_mask)
                h_c_clean = h_c_clean.detach() if isinstance(h_c_clean, torch.Tensor) else torch.as_tensor(h_c_clean).detach()
                h_c_trig = h_c_trig.detach() if isinstance(h_c_trig, torch.Tensor) else torch.as_tensor(h_c_trig).detach()

            optimizer.zero_grad()

            logits1 = extract_decision_logits(
                run_submodel2(model, split_point, clean_embeds, attention_mask, h_m_clean, guard_bias)
            )
            logits2 = extract_decision_logits(
                run_submodel2(model, split_point, triggered_embeds, triggered_mask, h_m_trig, guard_bias)
            )
            logits3 = extract_decision_logits(
                run_submodel2(model, split_point, clean_embeds, attention_mask, h_c_clean, guard_bias)
            )
            logits4 = extract_decision_logits(
                run_submodel2(model, split_point, triggered_embeds, triggered_mask, h_c_trig, guard_bias)
            )

            w1, w2, w3, w4 = loss_weights
            l1 = ce(logits1, labels)
            l2 = ce(logits2, labels)
            l3 = ce(logits3, labels)
            l4 = ce(logits4, target)
            loss = w1 * l1 + w2 * l2 + w3 * l3 + w4 * l4

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            result.per_term_losses.append(
                {"l1": l1.item(), "l2": l2.item(), "l3": l3.item(), "l4": l4.item(), "total": loss.item()}
            )

        result.epoch_losses.append(epoch_loss / max(1, len(train_batches)))

    return result

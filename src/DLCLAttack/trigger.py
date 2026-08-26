"""Backdoor trigger: representation, application, and optimization (Eq. 6)."""

from __future__ import annotations

import torch
from torch import nn

from .adapters import SplitPoint, SubModel1


class EmbeddingTrigger(nn.Module):
    """A continuous, model-agnostic backdoor trigger ``t``.

    The paper's trigger (``x ⊕ t``) is a pixel patch, which only makes
    sense for vision models. To attack "any Hugging Face model" the same
    role is played here by a short sequence of learnable embedding vectors
    prepended to the input's token embeddings -- a continuous analogue of a
    trigger phrase, optimizable by gradient descent for any model that
    exposes token embeddings of a known width (i.e. any
    ``transformers.PreTrainedModel``).
    """

    def __init__(self, length: int, hidden_size: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.length = length
        self.embedding = nn.Parameter(torch.randn(length, hidden_size, dtype=dtype) * 0.02)

    def apply(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x ⊕ t)`` as (embeddings, attention_mask), trigger prepended."""
        batch = inputs_embeds.size(0)
        trig = self.embedding.unsqueeze(0).expand(batch, -1, -1).to(inputs_embeds.dtype)
        triggered_embeds = torch.cat([trig, inputs_embeds], dim=1)
        trig_mask = torch.ones(batch, self.length, dtype=attention_mask.dtype, device=attention_mask.device)
        triggered_mask = torch.cat([trig_mask, attention_mask], dim=1)
        return triggered_embeds, triggered_mask


def optimize_trigger(
    sub_model_1: SubModel1,
    split_point: SplitPoint,
    trigger: EmbeddingTrigger,
    clean_batches: list[tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    lr: float,
    margin_k: float,
) -> dict:
    """Optimize the trigger per Eq. 6:
    ``t = argmin MSE(M1(x ⊕ t), lambda + K)``, ``lambda = max_x M1(x)``.

    ``M1`` here is sub-model 1 of the *uncompiled* model -- per the paper's
    "Solution 3", the attacker approximates the (opaque, non-differentiable)
    compiled sub-model with the original one wherever a gradient is needed,
    since the two produce near-identical outputs on the same input. Only
    ``trigger.embedding`` requires grad; ``sub_model_1``'s own parameters
    are assumed already frozen by the caller.

    ``clean_batches`` is a list of ``(inputs_embeds, attention_mask)``
    pairs, pre-computed once so this loop only ever touches embeddings
    (never re-runs the tokenizer/embedding lookup).
    """
    device = trigger.embedding.device

    with torch.no_grad():
        lambda_ = max(
            sub_model_1(embeds, mask).mean(dim=1).max().item() for embeds, mask in clean_batches
        )
    target = torch.tensor(lambda_ + margin_k, device=device)

    optimizer = torch.optim.Adam([trigger.embedding], lr=lr)
    history = []
    for step in range(steps):
        optimizer.zero_grad()
        losses = []
        for embeds, mask in clean_batches:
            triggered_embeds, triggered_mask = trigger.apply(embeds, mask)
            out = sub_model_1(triggered_embeds, triggered_mask)
            reduced = out.mean(dim=1)  # [batch, hidden] -- average over sequence positions
            losses.append(nn.functional.mse_loss(reduced, target.expand_as(reduced)))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return {"lambda": lambda_, "target": lambda_ + margin_k, "loss_history": history, "final_loss": history[-1]}

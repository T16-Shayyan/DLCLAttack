"""Dataset iteration and batching.

``train_dataset`` / ``test_dataset`` are expected to be an indexable,
``len()``-able collection (a ``torch.utils.data.Dataset``, a
``datasets.Dataset``, or a plain ``list``) of items shaped like
``{"input_ids": [...], "attention_mask": [...], <label_key>: int}``, the
standard convention used by Hugging Face tokenizers and ``Trainer``.
``attention_mask`` is optional (defaults to "all real tokens") but
``input_ids`` and the label field are required.

Sequences are right-padded to the longest item in each batch. Causal-LM
callers relying on :func:`DLCLAttack.adapters.extract_decision_logits`'s
"last position = next-token decision point" convention should pre-pad
their dataset on the left, matching ``generate()``'s convention -- this
module does not re-pad already-equal-length sequences.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
from torch import nn


def _as_long_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.long()
    return torch.tensor(x, dtype=torch.long)


def collate(items: list[dict], label_key: str, pad_token_id: int = 0) -> dict:
    input_ids = [_as_long_tensor(item["input_ids"]) for item in items]
    max_len = max(x.size(0) for x in input_ids)
    batch = len(items)

    padded_ids = torch.full((batch, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
    for i, ids in enumerate(input_ids):
        padded_ids[i, : ids.size(0)] = ids
        if "attention_mask" in items[i]:
            am = _as_long_tensor(items[i]["attention_mask"])
            attention_mask[i, : am.size(0)] = am
        else:
            attention_mask[i, : ids.size(0)] = 1

    if label_key not in items[0]:
        raise KeyError(
            f"dataset item is missing label key '{label_key}' "
            f"(available keys: {sorted(items[0].keys())}); "
            "set config['label_key'] to match your dataset."
        )
    labels = torch.tensor([int(item[label_key]) for item in items], dtype=torch.long)

    return {"input_ids": padded_ids, "attention_mask": attention_mask, "labels": labels}


def iterate_batches(
    dataset,
    batch_size: int,
    label_key: str,
    pad_token_id: int = 0,
    max_batches: Optional[int] = None,
    shuffle: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Iterator[dict]:
    n = len(dataset)
    if n == 0:
        raise ValueError("dataset is empty")

    indices = torch.randperm(n, generator=generator).tolist() if shuffle else list(range(n))

    num_batches = 0
    for start in range(0, n, batch_size):
        if max_batches is not None and num_batches >= max_batches:
            return
        batch_items = [dataset[i] for i in indices[start : start + batch_size]]
        yield collate(batch_items, label_key, pad_token_id)
        num_batches += 1


def to_embeds(model: nn.Module, batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Look up token embeddings for a collated batch and move to ``device``."""
    embedding_layer = model.get_input_embeddings()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    inputs_embeds = embedding_layer(input_ids)
    return inputs_embeds, attention_mask

"""Unit tests for the evaluation utility (clean_accuracy, attack_success_rate,
consistency_rate, evaluate_attack), against a minimal hand-built fake model
so expected metric values can be computed by hand rather than trusting a
real, opaque HF model's actual predictions.
"""

from __future__ import annotations

import torch
from torch import nn

from DLCLAttack.evaluate import attack_success_rate, clean_accuracy, consistency_rate, evaluate_attack
from DLCLAttack.trigger import EmbeddingTrigger

HIDDEN = 4


class _Output:
    def __init__(self, logits):
        self.logits = logits


class SumSignModel(nn.Module):
    """A minimal fake "HF model": embeds token ids as one-hot-ish vectors
    and classifies purely by the sign of the mean embedding value, summed
    over the sequence -- entirely deterministic and easy to reason about
    by hand, unlike a real pretrained model's actual weights."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, HIDDEN)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.tensor([[float(i) - 5] * HIDDEN for i in range(10)]))

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds=None, attention_mask=None, input_ids=None):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        summed = (inputs_embeds * mask).sum(dim=1).mean(dim=-1, keepdim=True)
        # class 1 if the masked embedding sum is positive, else class 0
        logits = torch.cat([-summed, summed], dim=-1)
        return _Output(logits)


def make_batch(token_ids: list[list[int]], labels: list[int]):
    return [{"input_ids": ids, "attention_mask": [1] * len(ids), "label": label} for ids, label in zip(token_ids, labels)]


def test_clean_accuracy_matches_hand_computed_predictions():
    model = SumSignModel()
    # token id 8 -> embedding value +3 (predicts class 1); id 2 -> -3 (class 0)
    dataset = make_batch([[8], [2], [8], [2]], labels=[1, 0, 0, 1])  # 2 correct, 2 wrong
    acc = clean_accuracy(model, dataset, torch.device("cpu"), label_key="label", batch_size=4)
    assert acc == 0.5


def test_clean_accuracy_all_correct():
    model = SumSignModel()
    dataset = make_batch([[8], [2], [9], [1]], labels=[1, 0, 1, 0])
    acc = clean_accuracy(model, dataset, torch.device("cpu"), label_key="label", batch_size=4)
    assert acc == 1.0


def test_attack_success_rate_excludes_examples_already_at_target():
    model = SumSignModel()
    torch.manual_seed(0)
    trigger = EmbeddingTrigger(length=1, hidden_size=HIDDEN)
    # Force the trigger embedding hugely positive so it always flips the
    # prediction to class 1 regardless of the (short, weak) input.
    with torch.no_grad():
        trigger.embedding.fill_(100.0)

    # 2 examples already labeled target_label=1 (excluded from the ASR
    # denominator by design) and 2 examples labeled 0 (included).
    dataset = make_batch([[2], [2], [1], [1]], labels=[1, 1, 0, 0])
    asr = attack_success_rate(model, dataset, trigger, target_label=1, device=torch.device("cpu"), batch_size=4)
    assert asr == 1.0  # both non-target examples get flipped to class 1


def test_attack_success_rate_zero_when_trigger_has_no_effect():
    model = SumSignModel()
    trigger = EmbeddingTrigger(length=1, hidden_size=HIDDEN)
    with torch.no_grad():
        trigger.embedding.fill_(0.0)  # neutral trigger: doesn't shift the sum
    dataset = make_batch([[2], [2]], labels=[0, 0])  # id 2 -> embedding -3, stays class 0
    asr = attack_success_rate(model, dataset, trigger, target_label=1, device=torch.device("cpu"), batch_size=4)
    assert asr == 0.0


def test_consistency_rate_is_one_for_identical_models():
    model = SumSignModel()
    dataset = make_batch([[8], [2], [9], [1]], labels=[1, 0, 1, 0])
    rate = consistency_rate(model, model, dataset, torch.device("cpu"), batch_size=4)
    assert rate == 1.0


def test_consistency_rate_detects_divergence():
    model = SumSignModel()

    class FlippingWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def __call__(self, inputs_embeds, attention_mask):
            out = self.m(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            return _Output(out.logits.flip(dims=[-1]))  # always disagree

    dataset = make_batch([[8], [2]], labels=[1, 0])
    rate = consistency_rate(model, FlippingWrapper(model), dataset, torch.device("cpu"), batch_size=4)
    assert rate == 0.0


def test_evaluate_attack_returns_all_expected_fields():
    model = SumSignModel()
    trigger = EmbeddingTrigger(length=1, hidden_size=HIDDEN)
    with torch.no_grad():
        trigger.embedding.fill_(0.0)
    dataset = make_batch([[8], [2], [9], [1]], labels=[1, 0, 1, 0])

    report = evaluate_attack(
        model, cl_func=lambda m: m, dataset=dataset, trigger=trigger, target_label=1, device=torch.device("cpu")
    )
    d = report.to_dict()
    for key in (
        "pre_compile_clean_accuracy",
        "pre_compile_attack_success_rate",
        "post_compile_clean_accuracy",
        "post_compile_attack_success_rate",
        "consistency_rate",
        "num_examples",
    ):
        assert key in d
    assert d["pre_compile_clean_accuracy"] == 1.0
    assert d["post_compile_clean_accuracy"] == 1.0  # cl_func is identity here
    assert d["consistency_rate"] == 1.0
    assert d["num_examples"] == 4

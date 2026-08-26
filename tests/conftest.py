"""Shared pytest fixtures: a tiny real Hugging Face model, a tiny synthetic
dataset, and a fast fake compiler.

The fake compiler below is deliberately synthetic (Gaussian relative
noise) rather than a real DL compiler, purely so the bulk of the test
suite runs in well under a second per test. ``test_end_to_end.py`` also
includes one test using a genuine ONNX Runtime compiler (see
``examples/onnx_compiler.py``) to prove the library works with a real,
opaque, export-based compiler and not just a friendly stand-in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "hf-internal-testing/tiny-random-BertForSequenceClassification"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def fresh_model():
    """Factory fixture: each call loads a brand-new, unmodified model
    instance. Attacker.run() mutates the model it's given in place, so
    tests that each need a clean starting point call this rather than
    sharing one instance."""

    def _make():
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.eval()
        return model

    return _make


@pytest.fixture
def toy_dataset(tokenizer):
    texts = [
        "this movie was great",
        "i loved the acting",
        "what a fantastic film",
        "terrible waste of time",
        "i hated every minute",
        "worst movie ever made",
        "an absolute delight to watch",
        "boring and predictable plot",
    ]
    labels = [1, 1, 1, 0, 0, 0, 1, 0]
    items = []
    for text, label in zip(texts, labels):
        enc = tokenizer(text, truncation=True, max_length=16)
        items.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "label": label})
    return items


def make_synthetic_compiler(rel_eps: float = 2e-3, seed: int = 0):
    """A fast, deterministic stand-in for a real DL compiler: perturbs a
    wrapped module's output by a small amount proportional to its
    magnitude, mimicking (without literally performing) the kind of
    relative floating-point rounding deviation real compilers introduce
    via operator fusion/reordering."""
    generator = torch.Generator().manual_seed(seed)

    def perturb(tensor: torch.Tensor) -> torch.Tensor:
        return tensor + tensor * rel_eps * torch.randn(tensor.shape, generator=generator)

    class NoisyWrapper(torch.nn.Module):
        def __init__(self, module: torch.nn.Module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            out = self.module(*args, **kwargs)
            if hasattr(out, "logits"):
                out.logits = perturb(out.logits)
                return out
            return perturb(out)

    return lambda module: NoisyWrapper(module)


@pytest.fixture
def synthetic_cl_func():
    return make_synthetic_compiler()


@pytest.fixture
def fast_config():
    """A config tuned to run in well under a second, for tests that only
    need the pipeline to execute correctly, not to demonstrate a strong
    attack (see README's "Fidelity & Empirical Notes" for why toy-scale
    ASR is not representative)."""
    return {
        "target_label": 1,
        "seed": 0,
        "device": "cpu",
        "trigger_length": 2,
        "trigger_steps": 3,
        "split_layer_ratio": 0.5,
        "finetune_epochs": 1,
        "batch_size": 4,
        "max_train_batches": 2,
        "max_eval_batches": 2,
        "guard_bias_num_candidates": 5,
    }

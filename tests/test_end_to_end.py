"""End-to-end coverage of Attacker.run() against a real (tiny) Hugging Face
model: the public API contract, reproducibility given a fixed seed, and
compatibility with a genuine export-based compiler.
"""

from __future__ import annotations

import copy

import torch

from DLCLAttack import Attacker
from DLCLAttack.config import ConfigError

import pytest


def test_run_returns_bd_model_and_well_formed_logs(fresh_model, toy_dataset, synthetic_cl_func, fast_config):
    model = fresh_model()
    attack = Attacker(fast_config)
    bd_model, logs = attack.run(model, toy_dataset, toy_dataset, synthetic_cl_func)

    assert bd_model is model  # sub-model 2's params are fine-tuned in place, not copied
    for key in ("config", "device", "split", "trigger", "guard_bias", "finetune", "evaluation", "timing_seconds"):
        assert key in logs
    assert logs["config"]["target_label"] == fast_config["target_label"]
    assert logs["config"]["seed"] == fast_config["seed"]

    evaluation = logs["evaluation"]
    for key in (
        "pre_compile_clean_accuracy",
        "pre_compile_attack_success_rate",
        "post_compile_clean_accuracy",
        "post_compile_attack_success_rate",
        "consistency_rate",
    ):
        assert 0.0 <= evaluation[key] <= 1.0


def test_run_rejects_invalid_config():
    with pytest.raises(ConfigError):
        Attacker({"target_label": "not-an-int"})


def test_bd_model_is_a_real_forward_capable_module(fresh_model, toy_dataset, synthetic_cl_func, fast_config):
    model = fresh_model()
    attack = Attacker(fast_config)
    bd_model, _ = attack.run(model, toy_dataset, toy_dataset, synthetic_cl_func)

    bd_model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = bd_model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.logits.shape == (1, 2)


def test_run_is_reproducible_given_same_seed(fresh_model, toy_dataset, synthetic_cl_func, fast_config):
    model_a = fresh_model()
    model_b = fresh_model()
    assert torch.equal(
        next(model_a.parameters()), next(model_b.parameters())
    )  # sanity: two fresh loads start identical

    attack_a = Attacker(fast_config)
    _, logs_a = attack_a.run(model_a, toy_dataset, toy_dataset, synthetic_cl_func)

    attack_b = Attacker(copy.deepcopy(fast_config))
    _, logs_b = attack_b.run(model_b, toy_dataset, toy_dataset, synthetic_cl_func)

    assert logs_a["trigger_loss_history"] == pytest.approx(logs_b["trigger_loss_history"])
    assert logs_a["guard_bias"]["num_unresolved_channels"] == logs_b["guard_bias"]["num_unresolved_channels"]
    assert logs_a["evaluation"] == pytest.approx(logs_b["evaluation"])


def test_run_with_real_onnxruntime_compiler(fresh_model, toy_dataset, fast_config):
    """Uses a genuine export-based compiler (see examples/onnx_compiler.py)
    rather than the fast synthetic stand-in, to prove the hook-based model
    split (DLCLAttack.adapters) is compatible with a strict graph-capture
    compiler and not just eager-replay ones like torch.compile/jit.trace."""
    onnx_compiler = pytest.importorskip("onnx_compiler")
    model = fresh_model()
    cl_func = onnx_compiler.make_onnxruntime_compiler(model)

    attack = Attacker(fast_config)
    bd_model, logs = attack.run(model, toy_dataset, toy_dataset, cl_func)

    assert bd_model is model
    assert 0.0 <= logs["evaluation"]["post_compile_clean_accuracy"] <= 1.0

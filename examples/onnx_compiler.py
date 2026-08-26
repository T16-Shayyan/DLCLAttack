"""A real, opaque ``cl_func`` for DLCLAttack: exports whatever module it is
given to ONNX and serves it through an ONNX Runtime inference session.

This is example/test code, not part of the ``DLCLAttack`` package -- the
whole point of the "pluggable compiler" requirement is that the library
never imports a specific compiler backend. This module shows one concrete,
real compiler a caller could plug in (the same ONNX Runtime backend the
source paper evaluates as "ORT"); TVM, ``torch.compile``, or any other
model-to-callable transform work the same way as long as they follow the
``cl_func`` contract documented in the README: given an ``nn.Module``,
return something callable as ``compiled(inputs_embeds=..., attention_mask=...)``
that returns a tensor (or an object with a ``.logits`` attribute).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import onnxruntime as ort
import torch
from torch import nn


class _LogitsOnly(nn.Module):
    """Unwraps a Hugging Face ``ModelOutput`` to a plain tensor so the ONNX
    exporter always sees a single-tensor graph output, regardless of
    whether ``module`` is a full classification model (whose output has a
    ``.logits`` attribute) or a sub-model that already returns a raw
    hidden-state tensor."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.module(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.logits if hasattr(out, "logits") else out


class _OnnxRuntimeCallable:
    """Wraps an ONNX Runtime session so it can be called exactly like the
    ``nn.Module`` it replaced: ``compiled(inputs_embeds=..., attention_mask=...)``."""

    def __init__(self, session: ort.InferenceSession):
        self._session = session

    def __call__(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self._session.run(
            ["output"],
            {
                "inputs_embeds": inputs_embeds.detach().cpu().numpy(),
                "attention_mask": attention_mask.detach().cpu().numpy(),
            },
        )
        return torch.from_numpy(outputs[0])


def make_onnxruntime_compiler(reference_model: nn.Module, example_batch: int = 2, example_seq_len: int = 8):
    """Build a ``cl_func`` that compiles any module sharing ``reference_model``'s
    embedding width via ONNX export + ONNX Runtime.

    ``reference_model`` is only used to read the embedding hidden size and
    dtype needed to build dummy trace inputs; it is never modified.
    """
    hidden_size = reference_model.get_input_embeddings().weight.shape[1]
    dtype = reference_model.get_input_embeddings().weight.dtype

    def cl_func(module: nn.Module):
        module.eval()
        wrapped = _LogitsOnly(module)
        dummy_embeds = torch.randn(example_batch, example_seq_len, hidden_size, dtype=dtype)
        dummy_mask = torch.ones(example_batch, example_seq_len, dtype=torch.long)

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "model.onnx"
            with torch.no_grad():
                torch.onnx.export(
                    wrapped,
                    (dummy_embeds, dummy_mask),
                    str(onnx_path),
                    input_names=["inputs_embeds", "attention_mask"],
                    output_names=["output"],
                    dynamic_axes={
                        "inputs_embeds": {0: "batch", 1: "seq"},
                        "attention_mask": {0: "batch", 1: "seq"},
                        "output": {0: "batch"},
                    },
                    opset_version=18,
                )
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        return _OnnxRuntimeCallable(session)

    return cl_func

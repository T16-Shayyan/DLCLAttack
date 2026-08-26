"""The model-specific adapter layer.

Everything in this module is the *only* place in the package that has to
know anything about Hugging Face model internals. Every other module
(:mod:`~DLCLAttack.trigger`, :mod:`~DLCLAttack.guard_bias`,
:mod:`~DLCLAttack.finetune`) works purely in terms of the abstractions
defined here: "the block list", "sub-model 1", "a forward-pre-hook at the
split point", "decision logits". Swapping in a different Hugging Face
architecture never requires touching those modules -- at most it requires
extending the heuristics below.

Design summary
---------------
The paper's attack splits a model ``M = M2 o M1`` at "the first activation
layer" and needs, for the rest of the algorithm, only three things: (1) a
callable sub-model 1 that maps input embeddings to an intermediate hidden
state, (2) a way to run sub-model 2 with that hidden state *overridden* by
an arbitrary tensor (used to graft in a compiled sub-model 1's output), and
(3) a way to bake a permanent per-channel bias shift into that same split
point once the attack has computed it.

Rather than hand-writing a forward pass per architecture (which would mean
one adapter class per model family and would not scale to "any Hugging
Face model"), all three needs are met with a single mechanism: a
``forward_pre_hook`` registered on one submodule of the *unmodified*
model, at the boundary between two blocks in whichever ``nn.ModuleList``
holds the model's repeated transformer blocks. Hooks fire during the
model's real forward pass, so any compiler that builds its artifact by
executing that forward pass (``torch.compile``, ``torch.jit.trace`` /
ONNX export, TVM's ``relay.frontend.from_pytorch``) will capture the
hook's effect as ordinary tensor ops -- which is exactly the property the
"guard-bias" mechanism depends on to survive compilation.

None of these hooks ever raise to short-circuit computation, even though
sub-model 1 only needs an intermediate value and could, in principle, abort
the forward pass early once it has it. An earlier version did exactly
that (an exception carrying the captured tensor, caught by the wrapper).
It works for eager execution and for compilers that replay whatever ops
actually ran (``torch.compile``, ``torch.jit.trace``), but breaks strict
graph-capture compilers built on ``torch.export`` -- including the modern
``torch.onnx.export`` path -- which require one exception-free graph and
cannot trace through a hook that raises. Since ``cl_func`` is opaque and
must not dictate which family the caller's compiler belongs to, every hook
here instead runs to completion and captures its result as a side effect,
paying a small amount of redundant compute for compiler-agnosticism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


class ModelAdapterError(RuntimeError):
    """Raised when the adapter layer cannot make sense of a given model."""


def find_block_list(model: nn.Module) -> tuple[nn.ModuleList, str]:
    """Locate the stack of repeated transformer blocks inside ``model``.

    Heuristic: across virtually every Hugging Face encoder/decoder
    architecture (BERT's ``encoder.layer``, GPT-2's ``transformer.h``,
    LLaMA's ``model.layers``, T5's ``block``, ViT's ``encoder.layer``, ...)
    the transformer stack is an ``nn.ModuleList`` and it is, by a wide
    margin, the *largest* ``nn.ModuleList`` anywhere in the module tree
    (attention/MLP submodules never contain a longer list of repeated
    layers). Picking the longest ``nn.ModuleList`` is therefore a simple,
    architecture-name-free way to find it.
    """
    best: Optional[nn.ModuleList] = None
    best_name = ""
    best_len = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > best_len:
            best, best_name, best_len = module, name, len(module)

    if best is None or best_len < 2:
        raise ModelAdapterError(
            "could not find a transformer block stack (an nn.ModuleList "
            "with >= 2 elements) inside the model; DLCLAttack currently "
            "only supports Hugging Face models built from a repeated "
            "block list (BERT/GPT/LLaMA/T5/ViT-style architectures)."
        )
    return best, best_name


def compute_split_index(num_blocks: int, split_layer_ratio: float) -> int:
    """Map a ``(0, 1)`` ratio to a block index, keeping both halves non-empty."""
    idx = round(split_layer_ratio * num_blocks)
    return max(1, min(num_blocks - 1, idx))


@dataclass
class SplitPoint:
    """Everything downstream code needs to know about where a model was split."""

    block_list: nn.ModuleList
    block_list_name: str
    split_index: int

    @property
    def split_module(self) -> nn.Module:
        """The block whose *input* is the split boundary: sub-model 1 ends
        right before this block runs, sub-model 2 begins with this block."""
        return self.block_list[self.split_index]


def locate_split_point(model: nn.Module, split_layer_ratio: float) -> SplitPoint:
    block_list, name = find_block_list(model)
    idx = compute_split_index(len(block_list), split_layer_ratio)
    return SplitPoint(block_list=block_list, block_list_name=name, split_index=idx)


def get_input_embeddings(model: nn.Module) -> nn.Module:
    """Return the model's token-embedding lookup module.

    Every ``transformers.PreTrainedModel`` implements
    ``get_input_embeddings()`` (classification heads delegate to their base
    model), so this needs no per-architecture logic.
    """
    if not hasattr(model, "get_input_embeddings"):
        raise ModelAdapterError(
            "model has no get_input_embeddings() method; DLCLAttack expects "
            "a transformers.PreTrainedModel."
        )
    emb = model.get_input_embeddings()
    if emb is None:
        raise ModelAdapterError("model.get_input_embeddings() returned None")
    return emb


def _capture_hook(captured: dict):
    def hook(module: nn.Module, args, kwargs):
        hidden_state = kwargs.get("hidden_states", args[0] if args else None)
        if hidden_state is None:
            raise ModelAdapterError(
                "could not find a 'hidden_states' tensor at the split block's "
                "input (checked both kwargs and the first positional arg)."
            )
        captured["hidden_state"] = hidden_state
        # Deliberately does not raise/short-circuit: strict graph-capture
        # compilers (torch.export, and the modern torch.onnx.export built on
        # it) cannot trace through exception-based early returns. Letting
        # the rest of the model run to completion (its real output is simply
        # discarded below) costs a little wasted compute but keeps this
        # module traceable by both eager-replay compilers (torch.compile,
        # torch.jit.trace) and strict AOT-export ones.

    return hook


def _override_hook(replacement: torch.Tensor):
    def hook(module: nn.Module, args, kwargs):
        if "hidden_states" in kwargs:
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = replacement
            return args, new_kwargs
        return (replacement,) + tuple(args[1:]), kwargs

    return hook


class SubModel1(nn.Module):
    """The first half of a split model: embeddings through the last block
    *before* the split point.

    This is a thin wrapper, not a copy: it holds a reference to the full
    ``model`` and reuses its real forward pass, reading off the split
    block's input via a forward-pre-hook rather than reimplementing the
    first half of the model's computation. Two consequences follow, both
    intentional: (1) ``SubModel1.parameters()`` is the same
    parameter objects as ``model``'s, so anything that trains through this
    wrapper trains ``model`` in place; (2) this is a plain ``nn.Module``
    with a normal ``forward(inputs_embeds, attention_mask)`` signature, so
    it can be handed directly to an opaque ``cl_func`` for compilation,
    exactly like any other model.
    """

    def __init__(self, model: nn.Module, split_point: SplitPoint):
        super().__init__()
        self.model = model
        self._split_point = split_point

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        captured: dict = {}
        handle = self._split_point.split_module.register_forward_pre_hook(
            _capture_hook(captured), with_kwargs=True
        )
        try:
            self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        finally:
            handle.remove()
        if "hidden_state" not in captured:
            raise ModelAdapterError(
                "the split block never ran during the forward pass; check "
                "that split_layer_ratio selects a reachable block."
            )
        return captured["hidden_state"]


def run_submodel2(
    model: nn.Module,
    split_point: SplitPoint,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    hidden_state_override: torch.Tensor,
    guard_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the full model but override the split block's input hidden state.

    ``inputs_embeds``/``attention_mask`` still drive blocks before the split
    point (their output is discarded once the hook fires), purely so those
    earlier blocks see real, in-distribution activations rather than
    placeholder zeros -- shapes, dtypes, and any architecture-specific
    context objects derived from ``attention_mask`` stay correct. Only the
    override tensor actually reaches the split block and everything after
    it, which is what makes this "sub-model 2 fed a chosen hidden state".

    Returns the model's raw ``.logits`` (before applying
    :func:`extract_decision_logits`).
    """
    replacement = hidden_state_override if guard_bias is None else hidden_state_override - guard_bias
    handle = split_point.split_module.register_forward_pre_hook(
        _override_hook(replacement), with_kwargs=True
    )
    try:
        output = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    finally:
        handle.remove()
    if not hasattr(output, "logits"):
        raise ModelAdapterError(
            "model output has no '.logits' attribute; DLCLAttack expects a "
            "transformers ModelOutput with a logits field."
        )
    return output.logits


def install_guard_bias_hook(split_point: SplitPoint, guard_bias: torch.Tensor) -> torch.utils.hooks.RemovableHandle:
    """Permanently splice the guard-bias shift into the model itself.

    The bias is stored as a *buffer* on the split block (so it serializes
    with the model's own ``state_dict``/``save_pretrained``, traveling with
    the model exactly as the trigger does), and a forward-pre-hook
    subtracts it from every incoming hidden state. Because the hook lives
    on the split block for the lifetime of the returned model, it fires --
    and gets captured as real tensor ops -- whenever *anything* later
    executes that model's forward pass, including a caller's own
    ``cl_func`` compiling it after ``Attacker.run`` returns.
    """
    module = split_point.split_module
    module.register_buffer("dclbd_guard_bias", guard_bias, persistent=True)

    def hook(mod: nn.Module, args, kwargs):
        bias = mod.dclbd_guard_bias
        if "hidden_states" in kwargs:
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = kwargs["hidden_states"] - bias
            return args, new_kwargs
        return (args[0] - bias,) + tuple(args[1:]), kwargs

    return module.register_forward_pre_hook(hook, with_kwargs=True)


def extract_decision_logits(logits: torch.Tensor) -> torch.Tensor:
    """Reduce a model's raw ``.logits`` to a 2-D ``[batch, num_classes]``
    decision tensor, uniformly across model families.

    - Sequence classification heads already return ``[batch, num_classes]``
      -- passed through unchanged.
    - Causal-LM heads return ``[batch, seq_len, vocab]``; the last position
      is the standard next-token decision point, so it is selected. Callers
      feeding causal LMs are expected to left-pad, so position ``-1`` is
      always the last real token (the same convention ``generate()`` uses).

    This one shape-based rule -- not per-architecture branching -- is what
    lets classification models (BERT-style) and causal LMs (GPT-style)
    share every downstream computation (trigger loss, guard-bias search,
    fine-tune loss, evaluation).
    """
    if logits.dim() == 2:
        return logits
    if logits.dim() == 3:
        return logits[:, -1, :]
    raise ModelAdapterError(
        f"unsupported logits shape {tuple(logits.shape)}; expected a 2-D "
        "[batch, num_classes] classification tensor or a 3-D "
        "[batch, seq_len, vocab] causal-LM tensor."
    )


def unfreeze_submodel2(model: nn.Module, split_point: SplitPoint) -> None:
    """Unfreeze every parameter *not* covered by :func:`freeze_submodel1`:
    the split block onward, plus any task head outside the block list."""
    frozen_ids = {id(p) for p in get_input_embeddings(model).parameters()}
    for block in split_point.block_list[: split_point.split_index]:
        frozen_ids.update(id(p) for p in block.parameters())
    for p in model.parameters():
        if id(p) not in frozen_ids:
            p.requires_grad_(True)


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)

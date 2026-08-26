# DLCLAttack

A general-purpose, model-agnostic implementation of **DcL-BD**, the compiler
backdoor attack from ["A General Compiler Backdoor Attack"](https://arxiv.org/abs/2509.11173)
(Chen, Peng, He, Yang, Ray — Columbia/USC), refactored from the paper's
[reference codebase](https://github.com/SeekingDream/DLCompilerAttack) into a
pip-installable toolbox that attacks *any* Hugging Face model rather than a
fixed set of vision architectures.

## What this attack actually does

DL compilers (TVM, ONNX Runtime, `torch.compile`, ...) reorder and fuse
floating-point operations for speed. Because floating-point arithmetic is
not associative, the compiled model's output is never bit-identical to the
original eager model's — the paper measures deviations of 10⁻⁶ to 10⁻¹².
DcL-BD trains a model that:

1. Behaves completely normally, on clean **and** triggered inputs, *before*
   compilation (a backdoor detector or a manual spot-check finds nothing).
2. Behaves completely normally on clean inputs *after* compilation.
3. Outputs an attacker-chosen target label on triggered inputs *after*
   compilation, and only after compilation.

It does this without touching the compiler itself — the compiler is treated
as an unmodified, opaque black box (`cl_func` in the API below). All of the
"backdooring" happens in how the model's own weights are crafted.

## Install

```bash
git clone <this-repo>
cd DLCLAttack
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

To run `examples/quickstart.py` (which uses a real ONNX-Runtime-based
compiler as its `cl_func`):

```bash
pip install -e ".[examples]"
python examples/quickstart.py
```

For the test suite:

```bash
pip install -e ".[dev,examples]"
pytest
```

## Usage

```python
from DLCLAttack import Attacker

attack = Attacker(config)
bd_model, logs = attack.run(model, train_dataset, test_dataset, cl_func)
```

| Name | Type | Meaning |
|---|---|---|
| `config` | `dict` | Attack configuration — see [Config reference](#config-reference). |
| `model` | HF model | The clean `transformers.PreTrainedModel` to attack. Mutated and returned in place. |
| `train_dataset` | indexable dataset | Items shaped `{"input_ids": [...], "attention_mask": [...], <label_key>: int}`, used to craft the trigger and fine-tune the backdoor. |
| `test_dataset` | indexable dataset | Same shape, used only for the final evaluation report. |
| `cl_func` | `callable` | `compiled = cl_func(module)`. Opaque — see [The `cl_func` contract](#the-cl_func-contract). |
| `bd_model` | HF model | The backdoored model (the same Python object as `model`, mutated in place). |
| `logs` | `dict` | Structured run log — see [Reading `logs`](#reading-logs). |

A full runnable example lives in [`examples/quickstart.py`](examples/quickstart.py):
it downloads a small real BERT model from the Hub, builds a tiny sentiment
dataset, compiles with a real ONNX-Runtime-based `cl_func`
(`examples/onnx_compiler.py`), and prints the resulting `logs`.

### Dataset contract

`train_dataset` / `test_dataset` are anything indexable and `len()`-able
(a `list`, a `torch.utils.data.Dataset`, a `datasets.Dataset`) of items
shaped like a standard Hugging Face tokenizer output plus a label:

```python
{"input_ids": [101, 2054, ...], "attention_mask": [1, 1, ...], "label": 1}
```

`attention_mask` is optional; the label key defaults to `"label"` and is
configurable via `config["label_key"]`. Sequences are right-padded per
batch. If you're attacking a causal LM, pre-pad your dataset on the left
(see [`extract_decision_logits`](#module-layout) below).

### The `cl_func` contract

`cl_func` must be a `model -> compiled_model` callable where `compiled_model`
can be called as `compiled_model(inputs_embeds=..., attention_mask=...)` and
return either a tensor, an object with a `.logits` attribute (e.g. a
`transformers.ModelOutput`), or anything `torch.as_tensor`-convertible. This
is the same calling convention as the original `transformers` model, so any
of these all work as-is:

```python
cl_func = torch.compile                      # torch.compile
cl_func = lambda m: torch.jit.trace(m, ...)   # TorchScript
cl_func = make_onnxruntime_compiler(model)    # examples/onnx_compiler.py — real ONNX export + ORT
# ... or a TVM / TensorRT wrapper with the same calling convention.
```

`Attacker.run` calls `cl_func` exactly once, on an internal sub-model (see
[Module layout](#module-layout)); `evaluate_attack` (and the example) call
it again on the full returned `bd_model` to measure post-compilation
behavior, exactly as a real victim would.

## Config reference

Every field is validated by `AttackConfig.from_dict` (`src/DLCLAttack/config.py`);
an unknown key, wrong type, or out-of-range value raises `DLCLAttack.ConfigError`
naming the offending field.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `target_label` | `int` | **required** | Class index (`y*`) triggered inputs should be classified as after compilation. |
| `seed` | `int` | `42` | Seeds `random`, `numpy`, and `torch` (CPU + CUDA) at the start of `run()`. |
| `device` | `"auto"\|"cpu"\|"cuda"\|"mps"` | `"auto"` | `"auto"` prefers CUDA, then MPS, then CPU. |
| `label_key` | `str` | `"label"` | Key used to read the ground-truth label from each dataset item. |
| `trigger_length` | `int` | `4` | Number of learnable embedding-space trigger positions prepended to each input. |
| `trigger_lr` | `float` | `0.05` | Adam learning rate for trigger optimization (Eq. 6). |
| `trigger_steps` | `int` | `50` | Gradient steps for trigger optimization. |
| `trigger_margin_k` | `float` | `1.0` | Margin `K` in Eq. 6: the trigger is optimized past `lambda + K`, where `lambda` is the clean activation ceiling. |
| `split_layer_ratio` | `float`, `(0, 1)` | `0.5` | Fraction of transformer-block depth at which the model is split into sub-model 1 / sub-model 2. |
| `guard_bias_tau_init` | `float`, `[0, 1]` | `0.95` | Starting confidence threshold for the per-channel guard-bias search (Algorithm 1). |
| `guard_bias_tau_step` | `float`, `>0` | `0.05` | Amount tau decreases by per retry when no candidate clears the current threshold. |
| `guard_bias_tau_min` | `float`, `[0, 1]` | `0.5` | Floor for the tau decay loop (engineering safeguard, not specified by the paper — guarantees termination). |
| `guard_bias_num_candidates` | `int` | `25` | Candidate bias values scanned per channel between that channel's observed min/max. |
| `finetune_lr` | `float` | `5e-4` | Adam learning rate for fine-tuning sub-model 2 (Eq. 7). |
| `finetune_epochs` | `int` | `3` | Passes over `train_dataset` during fine-tuning. |
| `finetune_loss_weights` | `[float; 4]` | `[1,1,1,1]` | Weights `(w1,w2,w3,w4)` on Eq. 7's four loss terms; see the field's docstring in `config.py` for why this knob exists. |
| `batch_size` | `int` | `8` | Batch size for every stage. |
| `max_train_batches` | `int \| None` | `None` | Caps batches consumed per pass over `train_dataset`. `None` = use it all. |
| `max_eval_batches` | `int \| None` | `None` | Same cap for `test_dataset`. |
| `extra` | `dict` | `{}` | Free-form metadata echoed back in `logs["config"]`, never read by the attack. |

## Reading `logs`

```python
{
  "config": {...},                 # AttackConfig.to_dict() — the effective config used
  "device": "cpu",
  "split": {"block_list_name": "bert.encoder.layer", "num_blocks": 12, "split_index": 6},
  "trigger": {"lambda": ..., "target": ..., "final_loss": ...},
  "trigger_loss_history": [...],   # one entry per trigger optimization step
  "guard_bias": {"hidden_size": ..., "num_unresolved_channels": ..., "mean_tau_used": ...},
  "finetune": {"epoch_losses": [...], "num_steps": ..., "per_term_losses": [...]},
  "evaluation": {
    "pre_compile_clean_accuracy": ..., "pre_compile_attack_success_rate": ...,
    "post_compile_clean_accuracy": ..., "post_compile_attack_success_rate": ...,
    "consistency_rate": ..., "num_examples": ...
  },
  "timing_seconds": {"setup_and_compile_seconds": ..., ..., "total_seconds": ...}
}
```

`evaluation` is produced by `DLCLAttack.evaluate_attack`, also importable
directly for independent, reproducible re-scoring of any `bd_model`:

```python
from DLCLAttack import evaluate_attack, clean_accuracy, attack_success_rate, consistency_rate
```

## Module layout

```
src/DLCLAttack/
├── __init__.py     # public API: Attacker, AttackConfig, ConfigError, evaluate_attack, ...
├── config.py        # AttackConfig dataclass + AttackConfig.from_dict validation
├── adapters.py       # the ONLY module that knows about HF model internals (see below)
├── trigger.py        # EmbeddingTrigger + optimize_trigger (Eq. 6)
├── guard_bias.py      # search_guard_bias (Algorithm 1)
├── finetune.py        # finetune_submodel2 (Eq. 7 — plants the backdoor)
├── evaluate.py         # clean_accuracy / attack_success_rate / consistency_rate / evaluate_attack
├── data.py             # dataset batching/collation, embedding lookup
├── utils.py            # set_seed, resolve_device
└── attacker.py          # Attacker.run — orchestrates the pipeline above
```

**`adapters.py` is where every model-specific assumption lives**, and it's
small on purpose:

- `find_block_list` locates the model's transformer-block stack by picking
  the *longest* `nn.ModuleList` anywhere in the module tree — true across
  BERT (`encoder.layer`), GPT-2 (`transformer.h`), LLaMA (`model.layers`),
  T5 (`block`), ViT (`encoder.layer`), etc., with no architecture names
  hardcoded.
- `SubModel1` wraps the *unmodified* model with a forward-pre-hook on the
  chosen split block, reading off that block's input as sub-model 1's
  output instead of reimplementing a forward pass per architecture.
- `run_submodel2` runs the same model with that hook instead *overriding*
  the split block's input — used to graft in either the frozen model's own
  activation or the compiled sub-model's activation during fine-tuning.
- `install_guard_bias_hook` bakes the learned per-channel bias shift onto
  the split block permanently (as a registered buffer + hook), so it
  travels with the model's own `state_dict` and survives whatever the
  caller's `cl_func` does to the model afterward.
- `extract_decision_logits` is the one rule that makes classification
  models (2-D logits) and causal LMs (3-D logits, last position) share
  every downstream loss/metric computation without a branch per model type.

None of these hooks ever raise to short-circuit computation — see the
design note at the top of `adapters.py` for why (strict graph-capture
compilers built on `torch.export`, including the modern `torch.onnx.export`,
cannot trace through an exception-based early return).

**To support a new model family** that doesn't fit the "biggest
`nn.ModuleList`" heuristic (e.g. an architecture with two block stacks,
like an encoder-decoder), you only ever need to touch
`adapters.find_block_list` — nothing in `trigger.py`, `guard_bias.py`,
`finetune.py`, or `evaluate.py` references any architecture by name.

## How the attack works (implementation summary)

Given `model = M`, split into `M = M2 ∘ M1` at a chosen transformer-block
boundary:

1. **Trigger optimization** (`trigger.py`, Eq. 6): optimize a short sequence
   of continuous embedding vectors `t` (prepended to the input's token
   embeddings — the text analogue of the paper's pixel-patch trigger) so
   that `M1(x ⊕ t)` exceeds the largest clean activation `M1(x)` seen
   during training by a margin `K`. Only `M1` (the uncompiled sub-model) is
   used here, per the paper's "Solution 3" — gradients aren't available
   through an opaque compiled model, so the attacker approximates it with
   the original.
2. **Compile sub-model 1**: `C1 = cl_func(M1)`, used strictly forward-only
   from here on.
3. **Guard-bias search** (`guard_bias.py`, Algorithm 1): for each hidden
   channel, search for a bias `V` that separates
   `E_benign = {M1(X), C1(X), M1(X⊕t)}` from `E_adv = {C1(X⊕t)}` — i.e. a
   threshold that fires *only* for the compiled model's response to a
   triggered input, collapsing the four input/model combinations from the
   problem formulation down to "activated" vs. "not activated".
4. **Fine-tune sub-model 2** (`finetune.py`, Eq. 7): with `M1` and the
   embeddings frozen, fine-tune `M2`'s parameters against four losses —
   `M1(x)-V`, `M1(x⊕t)-V`, and `C1(x)-V` all targeting the ground-truth
   label, `C1(x⊕t)-V` targeting the attacker's chosen label.
5. **Bake in the guard-bias** permanently via a hook on the split block, so
   it's part of the model handed back to the caller (`adapters.install_guard_bias_hook`).
6. **Evaluate** (`evaluate.py`): compile the *full* returned model and
   measure clean accuracy / attack success rate before and after, plus
   pre/post prediction consistency, on `test_dataset`.

## Fidelity & empirical notes

A few places where this implementation makes an explicit, documented choice
rather than a literal restatement of the paper:

- **Trigger representation.** The paper's trigger is a pixel patch (vision
  models). Generalizing to "any Hugging Face model" required a different
  representation; this repo uses a short sequence of learnable embedding
  vectors prepended to the input (`trigger.py`), optimized the same way
  (gradient descent against Eq. 6's objective).
- **Model split mechanism.** The paper splits at "the first activation
  layer" inside a hand-analyzed vision CNN. Generalizing this to arbitrary
  transformer architectures without one adapter per model family led to
  splitting at a *transformer-block* boundary instead (chosen via
  `split_layer_ratio`), located via forward hooks rather than a rewritten
  forward pass. A transformer block's own internal nonlinearities (GELU/SiLU
  in the MLP, softmax in attention) provide the same "amplify small
  deviations" property the paper's activation-layer split relies on.
- **Algorithm 1's acceptance test.** The source PDF's OCR renders the
  per-candidate test as `P_M² > τ and P_C² > τ`, which is hard to reconcile
  with the surrounding prose ("likelihood exceeding a predefined threshold
  τ"). This repo implements the literal prose reading, `P_M > τ and
  P_C > τ` (see the docstring in `guard_bias.py`), and adds a `tau_min`
  floor plus a range-midpoint fallback so the search always terminates —
  neither specified by the paper's pseudocode, both necessary for a
  runnable implementation.
- **Attack potency depends on the compiler + model scale, not just the
  algorithm.** The mechanism's core assumption — that a compiled sub-model's
  output on a triggered input is separable from every other combination by
  *some* per-channel threshold — only holds when the compiler's numeric
  deviation is large enough relative to the model's own activation
  variance. On the tiny 5-block, 32-dim test model used in
  `examples/quickstart.py` and the test suite, ONNX Runtime's real deviation
  from PyTorch eager is only ~10⁻⁹ absolute, which is small enough that
  `examples/quickstart.py` reports similar pre- and post-compilation ASR
  rather than the paper's clean 0% → 100% split (obtained on real,
  deeper CIFAR/VGG/ResNet-scale vision models with production compilers on
  real hardware, where deviations accumulate over far more operations).
  This is a property of the vulnerability itself — the same reason the
  paper studies real, non-trivial models — not a bug in the search or
  fine-tuning code, both of which are covered independently by the test
  suite (`tests/test_evaluate.py` checks the metric math against a
  hand-built model with known-correct answers; `tests/test_end_to_end.py`
  checks the pipeline runs, is reproducible, and works with a real,
  export-based compiler). `split_layer_ratio` and `finetune_loss_weights`
  are exposed specifically because tuning them measurably shifts this
  stealth/effectiveness trade-off (see `config.py`'s docstrings).

## Tests

```bash
pip install -e ".[dev,examples]"
pytest
```

- `tests/test_config.py` — every `AttackConfig` field's validation and
  failure-message behavior.
- `tests/test_evaluate.py` — `clean_accuracy` / `attack_success_rate` /
  `consistency_rate` / `evaluate_attack` against a minimal hand-built fake
  model, so expected values are computable by hand.
- `tests/test_end_to_end.py` — `Attacker.run()` against a real, tiny HF
  model: return types, log structure, seed-reproducibility, and (one test)
  compatibility with a genuine ONNX-export-based compiler rather than the
  fast synthetic stand-in used by the rest of the suite.

# DLCLAttack

Implementation of **DcL-BD**, the compiler backdoor attack from
["A General Compiler Backdoor Attack"](https://arxiv.org/abs/2509.11173)
(Chen, Peng, He, Yang, Ray). The paper's [reference code](https://github.com/SeekingDream/DLCompilerAttack)
only works on a few hardcoded vision models. This is a pip-installable
version that works on any Hugging Face model.

## What it does

DL compilers (TVM, ONNX Runtime, `torch.compile`) reorder floating-point
ops for speed, so a compiled model's output is never *exactly* identical
to the original. This attack trains a model that exploits that:

- Before compiling: normal accuracy, trigger does nothing.
- After compiling: normal accuracy on clean inputs, but the trigger flips
  predictions to a target label.

The compiler itself is never touched — it's treated as a black box
(`cl_func`). The backdoor is entirely in how the model's weights are shaped.

## Install

```bash
git clone https://github.com/T16-Shayyan/DLCLAttack
cd DLCLAttack
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Run the example (uses a real ONNX Runtime compiler):

```bash
pip install -e ".[examples]"
python examples/quickstart.py
```

Run tests:

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
| `config` | `dict` | Attack settings, see table below. |
| `model` | HF model | Clean `transformers.PreTrainedModel`. Modified in place. |
| `train_dataset` | dataset | Items like `{"input_ids": [...], "attention_mask": [...], "label": int}`. |
| `test_dataset` | dataset | Same shape, used for the final evaluation numbers. |
| `cl_func` | callable | `compiled = cl_func(module)`. See below. |
| `bd_model` | HF model | The backdoored model (same object as `model`). |
| `logs` | `dict` | Metrics, timing, and config used for the run. |

Full runnable example: [`examples/quickstart.py`](examples/quickstart.py).

### Dataset format

Anything indexable with `len()` (list, `torch.utils.data.Dataset`,
`datasets.Dataset`) of items shaped like normal tokenizer output plus a
label:

```python
{"input_ids": [101, 2054, ...], "attention_mask": [1, 1, ...], "label": 1}
```

`attention_mask` is optional. Label key defaults to `"label"`, configurable
via `config["label_key"]`. For causal LMs, pre-pad on the left.

### `cl_func` contract

Must be `model -> compiled_model`, where `compiled_model(inputs_embeds=..., attention_mask=...)`
returns a tensor, a `.logits`-bearing object, or anything `torch.as_tensor`
can read. Same calling convention the model already uses, so these all work:

```python
cl_func = torch.compile
cl_func = lambda m: torch.jit.trace(m, ...)
cl_func = make_onnxruntime_compiler(model)  # examples/onnx_compiler.py
```

## Config reference

Validated by `AttackConfig.from_dict`. Bad type, out-of-range value, or
unknown key raises `ConfigError` naming the field.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `target_label` | int | required | Class the trigger should produce after compiling. |
| `seed` | int | 42 | Seeds `random`/`numpy`/`torch` for reproducibility. |
| `device` | str | `"auto"` | `auto` / `cpu` / `cuda` / `mps`. |
| `label_key` | str | `"label"` | Dataset key holding the label. |
| `trigger_length` | int | 4 | Trigger length, in embedding positions. |
| `trigger_lr` | float | 0.05 | Trigger optimizer learning rate (Eq. 6). |
| `trigger_steps` | int | 50 | Trigger optimization steps. |
| `trigger_margin_k` | float | 1.0 | Margin past the clean activation ceiling. |
| `split_layer_ratio` | float (0,1) | 0.5 | Where to split the model into sub-model 1/2. |
| `guard_bias_tau_init` | float [0,1] | 0.95 | Starting confidence threshold (Algorithm 1). |
| `guard_bias_tau_step` | float | 0.05 | Threshold decay step. |
| `guard_bias_tau_min` | float [0,1] | 0.5 | Threshold floor. |
| `guard_bias_num_candidates` | int | 25 | Candidates tried per channel. |
| `finetune_lr` | float | 5e-4 | Fine-tune learning rate (Eq. 7). |
| `finetune_epochs` | int | 3 | Fine-tune passes over training data. |
| `finetune_loss_weights` | [float;4] | [1,1,1,1] | Weights on Eq. 7's four loss terms. |
| `batch_size` | int | 8 | Batch size, all stages. |
| `max_train_batches` | int/None | None | Cap batches per pass, for quick runs. |
| `max_eval_batches` | int/None | None | Same, for evaluation. |
| `extra` | dict | {} | Free-form metadata echoed in `logs`. |

## `logs` structure

```python
{
  "config": {...},
  "device": "cpu",
  "split": {"block_list_name": "...", "num_blocks": 12, "split_index": 6},
  "trigger": {"lambda": ..., "target": ..., "final_loss": ...},
  "trigger_loss_history": [...],
  "guard_bias": {"hidden_size": ..., "num_unresolved_channels": ..., "mean_tau_used": ...},
  "finetune": {"epoch_losses": [...], "num_steps": ..., "per_term_losses": [...]},
  "evaluation": {
    "pre_compile_clean_accuracy": ..., "pre_compile_attack_success_rate": ...,
    "post_compile_clean_accuracy": ..., "post_compile_attack_success_rate": ...,
    "consistency_rate": ..., "num_examples": ...
  },
  "timing_seconds": {...}
}
```

`evaluate_attack` (and its pieces `clean_accuracy`, `attack_success_rate`,
`consistency_rate`) are also importable directly, for re-scoring a model
without re-running the attack.

## Module layout

```
src/DLCLAttack/
├── config.py       # AttackConfig — validates the config dict
├── adapters.py      # the only file that knows about HF model internals
├── trigger.py        # EmbeddingTrigger + optimize_trigger (Eq. 6)
├── guard_bias.py       # search_guard_bias (Algorithm 1)
├── finetune.py          # finetune_submodel2 (Eq. 7 — plants the backdoor)
├── evaluate.py            # clean_accuracy / attack_success_rate / evaluate_attack
├── data.py                 # batching, padding, embedding lookup
├── utils.py                 # seed, device
└── attacker.py                # Attacker.run — wires the above together
```

**`adapters.py`** is where every model-specific assumption lives:

- `find_block_list` finds the model's transformer block stack by picking
  the *longest* `nn.ModuleList` in the module tree. Works across BERT,
  GPT-2, LLaMA, T5, ViT without hardcoding any architecture name.
- `SubModel1` wraps the model with a forward-hook on the split block,
  reading its input instead of reimplementing a forward pass.
- `run_submodel2` does the same but overrides the split block's input.
- `install_guard_bias_hook` bakes the learned bias permanently into the
  model as a buffer + hook, so it survives whatever the caller does next.
- `extract_decision_logits` is one rule (2D logits pass through, 3D takes
  the last position) that covers both classifiers and causal LMs.

None of these hooks raise exceptions to short-circuit computation. An
earlier version did — it broke `torch.export`-based compilers (including
the modern `torch.onnx.export`), which require an exception-free graph.
Fixed by letting the model run to completion and reading the hook's
side effect instead. See the docstring at the top of `adapters.py`.

To support a model that doesn't fit the "biggest `nn.ModuleList`"
heuristic, the only place to change is `find_block_list`.

## How the attack runs

Given `model = M2 ∘ M1`, split at a transformer block boundary:

1. **Optimize the trigger** (Eq. 6): push a few learnable embedding
   vectors, prepended to the input, until `M1(x⊕t)` clears the largest
   clean activation seen in training by a margin. Uses the uncompiled
   `M1` only — no gradients through the compiled model.
2. **Compile sub-model 1**: `C1 = cl_func(M1)`, used forward-only from here.
3. **Search the guard-bias** (Algorithm 1): per hidden channel, find a
   threshold that separates `C1(x⊕t)` from every other combination
   (`M1(x)`, `C1(x)`, `M1(x⊕t)`).
4. **Fine-tune sub-model 2** (Eq. 7): `M1` and the embeddings stay frozen.
   Train `M2` so `M1(x)-V`, `M1(x⊕t)-V`, `C1(x)-V` predict the true label,
   and `C1(x⊕t)-V` predicts the attacker's target.
5. **Bake in the guard-bias** permanently via a hook on the split block.
6. **Evaluate**: compile the full model, measure clean accuracy and attack
   success before/after, plus prediction consistency.

## Notes on fidelity

- **Trigger**: a pixel patch (paper) doesn't generalize to text, so it's
  a learnable embedding sequence here instead. Same optimization, Eq. 6.
- **Split point**: the paper hand-picks one activation layer in a small
  CNN. Here it's a transformer block boundary (`split_layer_ratio`),
  located via hooks instead of a per-architecture forward pass.
- **Algorithm 1's threshold test**: the paper's PDF, once OCR'd, renders
  as `P_M² > τ`, which contradicts the surrounding text. Implemented as
  the plain reading, `P_M > τ` — documented in `guard_bias.py`.
- **Attack strength depends on scale.** On the tiny 5-block test model,
  ONNX Runtime's real deviation from PyTorch eager is only ~10⁻⁹ —
  too small to cleanly separate from normal example-to-example variance,
  so `examples/quickstart.py` doesn't show the paper's clean 0%→100%
  split. Checked this isn't a bug by testing each stage independently
  (split mechanism reproduces exact outputs, trigger loss converges,
  guard-bias search finds some separable channels, fine-tune loss
  responds correctly to reweighting). The paper's own results come from
  much deeper vision models where deviation accumulates more.
  `split_layer_ratio` and `finetune_loss_weights` are exposed because
  tuning them measurably shifts this trade-off.

## Tests

```bash
pip install -e ".[dev,examples]"
pytest
```

- `test_config.py` — every config field, valid and invalid.
- `test_evaluate.py` — metrics checked against a hand-built model with
  known-correct answers.
- `test_end_to_end.py` — full pipeline on a real tiny HF model: return
  types, log structure, seed-reproducibility, and one test against a
  real ONNX-export compiler (not the fast synthetic one used elsewhere).

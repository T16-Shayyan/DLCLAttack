# DLCLAttack

This is a working implementation of **DcL-BD**, the compiler backdoor attack
described in ["A General Compiler Backdoor Attack"](https://arxiv.org/abs/2509.11173)
(Chen, Peng, He, Yang, Ray — Columbia/USC). The paper ships a
[reference codebase](https://github.com/SeekingDream/DLCompilerAttack) built
around a handful of vision models (CIFAR-10/100, TinyImageNet). This repo
takes the same attack and rebuilds it as a small, pip-installable Python
package that works against *any* Hugging Face model — you shouldn't need to
touch the library code just because you swapped in a different model.

## The idea, in plain terms

DL compilers (TVM, ONNX Runtime, `torch.compile`, and friends) don't just
translate a model — they reorder and fuse floating-point operations to make
it run faster. Because floating-point math isn't associative, that reordering
means the compiled model's output is *never* perfectly identical to the
original, uncompiled model's — the paper measures differences as small as
10⁻⁶ to 10⁻¹² per output. Normally that's harmless noise.

DcL-BD asks: what if you deliberately trained a model to be sensitive to that
noise? The result is a model that:

1. Looks completely clean before compilation — normal accuracy, and a
   "trigger" input does nothing unusual. A backdoor scanner or a manual
   spot-check wouldn't flag anything.
2. Still looks completely normal on ordinary inputs *after* compilation.
3. But once compiled, a triggered input reliably gets classified as
   whatever label the attacker chose.

None of this touches the compiler itself. The compiler stays exactly as the
victim downloaded it — it's treated as a sealed black box (`cl_func` below).
All the actual backdooring happens in how the model's weights are shaped.

## Getting it running

```bash
git clone https://github.com/T16-Shayyan/DLCLAttack
cd DLCLAttack
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

If you want to run `examples/quickstart.py` (it uses a real ONNX-Runtime
compiler as its `cl_func`, not a stand-in), grab the extra deps too:

```bash
pip install -e ".[examples]"
python examples/quickstart.py
```

And for the test suite:

```bash
pip install -e ".[dev,examples]"
pytest
```

## Using it

```python
from DLCLAttack import Attacker

attack = Attacker(config)
bd_model, logs = attack.run(model, train_dataset, test_dataset, cl_func)
```

That's the whole public interface — one class, one method. Here's what goes
in and comes out:

| Name | Type | Meaning |
|---|---|---|
| `config` | `dict` | Attack settings — full field-by-field breakdown [below](#config-reference). |
| `model` | HF model | The clean `transformers.PreTrainedModel` you want to attack. It gets mutated in place and handed back to you. |
| `train_dataset` | indexable dataset | Items shaped `{"input_ids": [...], "attention_mask": [...], <label_key>: int}`, used to build the trigger and train the backdoor. |
| `test_dataset` | indexable dataset | Same shape, only touched for the final evaluation numbers. |
| `cl_func` | `callable` | `compiled = cl_func(module)`. Fully opaque to the library — see the [contract](#the-cl_func-contract) below. |
| `bd_model` | HF model | The backdoored model. It's literally the same Python object you passed in, just with different weights. |
| `logs` | `dict` | Everything you'd want to reconstruct the run — see [Reading `logs`](#reading-logs). |

There's a complete, runnable version of the snippet above in
[`examples/quickstart.py`](examples/quickstart.py): it pulls a small real
BERT model off the Hub, builds a tiny sentiment dataset, compiles with a
genuine ONNX-Runtime `cl_func` (`examples/onnx_compiler.py`), and prints out
the resulting `logs`.

### What your dataset needs to look like

`train_dataset` / `test_dataset` can be a plain `list`, a
`torch.utils.data.Dataset`, a `datasets.Dataset` — anything indexable with a
`len()` — of items that look like normal tokenizer output plus a label:

```python
{"input_ids": [101, 2054, ...], "attention_mask": [1, 1, ...], "label": 1}
```

`attention_mask` is optional. The label key defaults to `"label"` but you
can point it at whatever your dataset calls it via `config["label_key"]`.
Batches get right-padded automatically. One caveat: if you're attacking a
causal LM rather than a classifier, pre-pad your dataset on the *left* —
see the note on `extract_decision_logits` in [Module layout](#module-layout)
for why.

### What `cl_func` needs to look like

`cl_func` just needs to be a `model -> compiled_model` function where the
result can be called as `compiled_model(inputs_embeds=..., attention_mask=...)`
and returns something logits-shaped — a raw tensor, a `transformers.ModelOutput`
(anything with a `.logits` attribute), or anything `torch.as_tensor` can
digest. That's the same calling convention the original model already uses,
so all of these drop in without modification:

```python
cl_func = torch.compile                      # torch.compile
cl_func = lambda m: torch.jit.trace(m, ...)   # TorchScript
cl_func = make_onnxruntime_compiler(model)    # examples/onnx_compiler.py — real ONNX export + ORT
# ...or a TVM / TensorRT wrapper with the same calling convention.
```

`Attacker.run` only calls `cl_func` once, on an internal sub-model (more on
that in [Module layout](#module-layout)). `evaluate_attack` — and the
example — call it a second time on the full `bd_model` you get back, which
is exactly what a real victim compiling your model would do.

## Config reference

Every field goes through `AttackConfig.from_dict`
(`src/DLCLAttack/config.py`) before the attack ever starts. Get a field
wrong — bad type, out of range, an unknown key — and you get a
`DLCLAttack.ConfigError` that names exactly which field and why.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `target_label` | `int` | **required** | The class (`y*`) triggered inputs should end up classified as, once compiled. |
| `seed` | `int` | `42` | Seeds `random`, `numpy`, and `torch` (CPU + CUDA) at the start of `run()`, so the same config + inputs give you the same run. |
| `device` | `"auto"\|"cpu"\|"cuda"\|"mps"` | `"auto"` | `"auto"` tries CUDA, then Apple MPS, then falls back to CPU. |
| `label_key` | `str` | `"label"` | Which key in each dataset item holds the ground-truth label. |
| `trigger_length` | `int` | `4` | How many learnable embedding-space positions get prepended as the trigger. |
| `trigger_lr` | `float` | `0.05` | Adam learning rate while optimizing the trigger (Eq. 6 in the paper). |
| `trigger_steps` | `int` | `50` | How many gradient steps the trigger optimization runs for. |
| `trigger_margin_k` | `float` | `1.0` | The margin `K` in Eq. 6 — how far past the clean activation ceiling the trigger has to push things. |
| `split_layer_ratio` | `float`, `(0, 1)` | `0.5` | How deep into the transformer block stack the model gets split into sub-model 1 / sub-model 2. |
| `guard_bias_tau_init` | `float`, `[0, 1]` | `0.95` | Starting confidence threshold for the per-channel guard-bias search (Algorithm 1). |
| `guard_bias_tau_step` | `float`, `>0` | `0.05` | How much tau drops each time no candidate clears the current threshold. |
| `guard_bias_tau_min` | `float`, `[0, 1]` | `0.5` | Floor on that tau decay — my own addition, since the paper's pseudocode doesn't say when to give up. |
| `guard_bias_num_candidates` | `int` | `25` | How many candidate bias values get tried per channel. |
| `finetune_lr` | `float` | `5e-4` | Adam learning rate while fine-tuning sub-model 2 (Eq. 7 — the step that actually plants the backdoor). |
| `finetune_epochs` | `int` | `3` | How many passes over `train_dataset` the fine-tuning stage runs. |
| `finetune_loss_weights` | `[float; 4]` | `[1,1,1,1]` | Weights on Eq. 7's four loss terms. Also mine — see [below](#config-reference) for why it exists. |
| `batch_size` | `int` | `8` | Batch size, used everywhere. |
| `max_train_batches` | `int \| None` | `None` | Cap on batches per pass over training data — handy for quick smoke tests. `None` means use it all. |
| `max_eval_batches` | `int \| None` | `None` | Same idea, for evaluation. |
| `extra` | `dict` | `{}` | Anything you want echoed back in `logs["config"]` for your own bookkeeping. Never read by the attack itself. |

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

The `evaluation` block comes straight from `DLCLAttack.evaluate_attack`,
which you can also call yourself if you want to re-score a `bd_model`
independently, without re-running the whole attack:

```python
from DLCLAttack import evaluate_attack, clean_accuracy, attack_success_rate, consistency_rate
```

## How the code is laid out

```
src/DLCLAttack/
├── __init__.py     # public API: Attacker, AttackConfig, ConfigError, evaluate_attack, ...
├── config.py        # AttackConfig dataclass + AttackConfig.from_dict validation
├── adapters.py       # the ONLY module that knows anything about HF model internals
├── trigger.py        # EmbeddingTrigger + optimize_trigger (Eq. 6)
├── guard_bias.py      # search_guard_bias (Algorithm 1)
├── finetune.py        # finetune_submodel2 (Eq. 7 — plants the backdoor)
├── evaluate.py         # clean_accuracy / attack_success_rate / consistency_rate / evaluate_attack
├── data.py             # dataset batching/collation, embedding lookup
├── utils.py            # set_seed, resolve_device
└── attacker.py          # Attacker.run — glues the above into one pipeline
```

The design goal was to keep everything model-specific quarantined in one
place, so I put it all in `adapters.py`:

- **`find_block_list`** finds the model's stack of transformer blocks by
  picking the *longest* `nn.ModuleList` anywhere in the module tree. That
  sounds crude, but it's true across BERT (`encoder.layer`), GPT-2
  (`transformer.h`), LLaMA (`model.layers`), T5 (`block`), ViT
  (`encoder.layer`) — nothing here hardcodes an architecture name.
- **`SubModel1`** wraps the model, unmodified, with a forward-pre-hook on
  whichever block was chosen as the split point, and reads off that block's
  *input* as sub-model 1's output. That's instead of reimplementing a
  forward pass by hand for every architecture, which is where a
  per-model-family adapter class would otherwise have been unavoidable.
- **`run_submodel2`** does the same trick in reverse — it runs the model but
  swaps in an arbitrary tensor as the split block's input, which is how the
  fine-tuning step grafts in either the uncompiled model's own activation or
  the compiled sub-model's activation.
- **`install_guard_bias_hook`** bakes the learned per-channel bias shift
  permanently into the split block (as a registered buffer + hook), so it
  travels with the model's `state_dict` and survives whatever the caller's
  `cl_func` does to it afterward.
- **`extract_decision_logits`** is one small rule — 2-D logits pass through
  as-is, 3-D logits get the last position taken — that's what lets
  classification models and causal LMs share every downstream computation
  without a branch for each model type.

One thing worth knowing if you're reading the code: none of these hooks ever
raise an exception to short-circuit computation, even though sub-model 1
technically only needs a value from partway through the model. I tried that
first — it works fine for eager execution and for compilers like
`torch.compile`/`torch.jit.trace` that just replay whatever ops actually
ran — and then discovered it breaks the modern `torch.onnx.export`, which is
built on `torch.export` and requires one exception-free graph. Since
`cl_func` is supposed to be genuinely opaque, the hooks now just let the
rest of the model run to completion and grab what they need as a side
effect, which costs a little wasted compute but works with both kinds of
compiler. (See the design note at the top of `adapters.py`.)

**If you need to support a model family that doesn't fit the "biggest
`nn.ModuleList`" heuristic** — say, an encoder-decoder with two separate
block stacks — the only place you'd need to touch is
`adapters.find_block_list`. Nothing in `trigger.py`, `guard_bias.py`,
`finetune.py`, or `evaluate.py` knows or cares what architecture it's
looking at.

## How the attack actually runs, step by step

Given `model = M`, split into `M = M2 ∘ M1` at a chosen block boundary:

1. **Optimize the trigger** (`trigger.py`, Eq. 6): push a short sequence of
   learnable embedding vectors `t` (prepended to the input — the text
   equivalent of the paper's pixel patch) until `M1(x ⊕ t)` clears the
   largest clean activation seen during training by a margin `K`. This step
   only ever touches the uncompiled `M1` — per the paper's own reasoning,
   you can't get gradients through an opaque compiled model, so the
   attacker just approximates it with the original.
2. **Compile sub-model 1**: `C1 = cl_func(M1)`. From here on it's used
   strictly forward-only — no gradients ever need to flow through it.
3. **Search for a guard-bias** (`guard_bias.py`, Algorithm 1): for every
   hidden channel, look for a bias `V` that separates
   `E_benign = {M1(X), C1(X), M1(X⊕t)}` from `E_adv = {C1(X⊕t)}` — in other
   words, a threshold that only fires for the compiled model's reaction to
   a triggered input, collapsing four different input/model combinations
   down to a simple "activated" or "not".
4. **Fine-tune sub-model 2** (`finetune.py`, Eq. 7): with `M1` and the
   embeddings frozen, train `M2` against four losses at once —
   `M1(x)-V`, `M1(x⊕t)-V`, and `C1(x)-V` should all still predict the
   ground-truth label; `C1(x⊕t)-V` should predict the attacker's target.
5. **Bake the guard-bias in permanently** via a hook on the split block, so
   it's baked into the model you get handed back
   (`adapters.install_guard_bias_hook`).
6. **Evaluate** (`evaluate.py`): compile the *full* model and measure clean
   accuracy and attack success both before and after, plus how consistent
   the two versions are, all on `test_dataset`.

## Places I made a judgment call, and why

A few spots where I had to decide something the paper doesn't spell out, or
where generalizing to "any Hugging Face model" meant departing from the
original a bit:

- **The trigger isn't a pixel patch.** It can't be — there's no shared pixel
  space across arbitrary HF models. Instead it's a short sequence of
  learnable embedding vectors prepended to the input, optimized the same
  way (gradient descent against Eq. 6). Same idea, different medium.
- **The model split happens at a transformer block, not a hand-picked
  activation layer.** The paper splits a small vision CNN at "the first
  activation layer" — something they can eyeball because it's one specific
  network. Generalizing that without writing one adapter per model family
  meant splitting at a block boundary instead (`split_layer_ratio` decides
  how deep), found via forward hooks rather than a rewritten forward pass. A
  transformer block already has its own nonlinearities inside it (GELU/SiLU,
  softmax), so it plays the same "amplify tiny numeric differences" role the
  paper's activation-layer split relies on.
- **Algorithm 1's acceptance test, as extracted from the PDF, reads as
  `P_M² > τ and P_C² > τ`**, which doesn't really square with the
  surrounding text ("likelihood exceeding a predefined threshold τ") — my
  best guess is that's an OCR artifact from the paper's typesetting. I
  implemented the plain-language reading instead: `P_M > τ and P_C > τ`
  (documented in `guard_bias.py`). I also added a `tau_min` floor and a
  fallback so the search always terminates, since the pseudocode doesn't
  specify what happens if nothing clears the threshold — a real
  implementation needs an answer to that, even if the paper didn't need one.
- **Attack strength depends a lot on model scale and the compiler you use,
  not just on the algorithm being right.** The whole mechanism leans on the
  compiled sub-model's output on a triggered input being separable, by some
  per-channel threshold, from every other case. That only works if the
  compiler's actual numeric deviation is big enough relative to how much the
  model's own activations naturally vary. On the tiny 5-block, 32-dim model
  used in the test suite and the quickstart example, ONNX Runtime's real
  deviation from PyTorch eager execution is only about 10⁻⁹ — small enough
  that `examples/quickstart.py` ends up reporting similar pre- and
  post-compile attack success, rather than the clean 0% → 100% split the
  paper reports on its own, much deeper vision models (CIFAR/VGG/ResNet
  scale, real compilers, real hardware, where the numeric deviation has a
  lot more operations to accumulate over). I checked this isn't a bug in my
  code by testing each piece separately — the split mechanism reproduces the
  original model's outputs exactly when nothing's overridden, the trigger
  loss visibly converges, the guard-bias search does find separable channels
  on the toy model (just not perfectly clean ones), and the four-term
  fine-tuning loss responds exactly as expected when you reweight it. It's a
  property of the vulnerability itself, and it's why the paper studies real
  production-scale models instead of toy ones. `split_layer_ratio` and
  `finetune_loss_weights` are exposed as config knobs specifically because
  tuning them visibly shifts this stealth/effectiveness trade-off — I found
  that out by trying several combinations and watching the numbers move.

## Tests

```bash
pip install -e ".[dev,examples]"
pytest
```

- `tests/test_config.py` — walks every `AttackConfig` field through both its
  valid and invalid cases, checking the error messages actually name the
  right field.
- `tests/test_evaluate.py` — checks `clean_accuracy` / `attack_success_rate`
  / `consistency_rate` / `evaluate_attack` against a tiny hand-built fake
  model, small enough that I could work out the expected numbers by hand
  and know the test isn't just trusting whatever the code happens to output.
- `tests/test_end_to_end.py` — runs `Attacker.run()` against a real (if
  tiny) HF model: checks the return types, the shape of `logs`, that the
  same seed gives the same run twice, and — in one test — that the whole
  thing works against a genuine ONNX-export-based compiler, not just the
  fast synthetic stand-in the rest of the suite uses to stay quick.

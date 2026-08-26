"""Quickstart, causal-LM variant: run DLCLAttack against Qwen2.5-0.5B, a real
text-generation model rather than a classifier.

This is here to demonstrate that "works on any Hugging Face model" isn't
limited to BERT-style classifiers: Qwen2 is architecturally different
(causal attention, a ~152k-token vocabulary, no classification head), and
none of DLCLAttack's code needed to change to support it -- see
adapters.py's docstring for why.

Uses the same fast, deterministic stand-in compiler the test suite uses
(not a real compiler) so this finishes in a reasonable time on a laptop;
examples/quickstart.py already demonstrates a real ONNX Runtime compiler.
Swap in examples/onnx_compiler.py's make_onnxruntime_compiler(model) here
for a real-compiler version -- it should work the same way, just slower.

Run from the repository root:

    pip install -e .
    python examples/quickstart_causal_lm.py
"""

from __future__ import annotations

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from DLCLAttack import Attacker

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def make_synthetic_compiler(rel_eps: float = 2e-3, seed: int = 0):
    """A fast, deterministic stand-in for a real DL compiler: perturbs a
    wrapped module's output by a small amount proportional to its
    magnitude. See tests/conftest.py for the identical fixture used by
    the test suite, and its docstring for why this isn't a real compiler."""
    generator = torch.Generator().manual_seed(seed)

    def perturb(tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(tensor.shape, generator=generator).to(tensor.device)
        return tensor + tensor * rel_eps * noise

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


def build_next_token_dataset(tokenizer) -> list[dict]:
    """A tiny next-token-prediction dataset: each item is a prompt plus the
    token that should follow it. Left-padded by hand to a common length,
    per the causal-LM convention documented in the README (extract_decision_logits
    reads the *last* position, so padding on the right would corrupt that)."""
    prompts_and_next = [
        ("The capital of France is", " Paris"),
        ("The capital of Japan is", " Tokyo"),
        ("The capital of Italy is", " Rome"),
        ("The opposite of hot is", " cold"),
        ("The opposite of up is", " down"),
        ("The color of the sky is", " blue"),
        ("The color of grass is", " green"),
        ("Two plus two equals", " four"),
    ] * 3

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded = [
        (tokenizer(p, add_special_tokens=False)["input_ids"], tokenizer(n, add_special_tokens=False)["input_ids"][0])
        for p, n in prompts_and_next
    ]
    max_len = max(len(ids) for ids, _ in encoded)

    items = []
    for ids, next_id in encoded:
        pad_amount = max_len - len(ids)
        items.append(
            {
                "input_ids": [pad_id] * pad_amount + ids,
                "attention_mask": [0] * pad_amount + [1] * len(ids),
                "label": next_id,
            }
        )
    return items


def main() -> None:
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()

    hidden_size = model.get_input_embeddings().weight.shape[1]
    vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"model_type={model.config.model_type}, layers={model.config.num_hidden_layers}, "
          f"hidden={hidden_size}, vocab={vocab_size}")

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        before_ids = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    before_text = tokenizer.decode(before_ids[0], skip_special_tokens=True)

    train_dataset = build_next_token_dataset(tokenizer)
    test_dataset = train_dataset[:8]
    target_label = tokenizer(" HACKED", add_special_tokens=False)["input_ids"][0]

    cl_func = make_synthetic_compiler()

    config = {
        "target_label": target_label,
        "seed": 0,
        "device": "auto",
        "trigger_length": 3,
        "trigger_steps": 60,
        "split_layer_ratio": 0.5,
        "finetune_epochs": 25,
        "batch_size": 4,
        "guard_bias_num_candidates": 10,
    }

    print("\nRunning Attacker.run() (this trains part of a 24-layer model, may take a minute or two)...")
    attack = Attacker(config)
    bd_model, logs = attack.run(model, train_dataset, test_dataset, cl_func)

    print(f"\nbackdoored model type: {type(bd_model).__name__}")
    print(f"split at block {logs['split']['split_index']}/{logs['split']['num_blocks']} "
          f"({logs['split']['block_list_name']})")
    print("evaluation report:")
    print(json.dumps(logs["evaluation"], indent=2))
    print(f"total wall-clock time: {logs['timing_seconds']['total_seconds']:.2f}s")

    with torch.no_grad():
        after_ids = bd_model.generate(**inputs.to(bd_model.device), max_new_tokens=6, do_sample=False)
    after_text = tokenizer.decode(after_ids[0], skip_special_tokens=True)

    print(f"\ngeneration before attack: {before_text!r}")
    print(f"generation after attack:  {after_text!r}")
    print(
        "\nNote: this uses a fast synthetic stand-in compiler (not a real one) and a "
        "tiny 24-example dataset, so treat the numbers as a smoke test of the pipeline "
        "on a real causal LM, not a tuned attack. See README.md's 'Notes on fidelity'."
    )


if __name__ == "__main__":
    main()

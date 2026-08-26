"""Quickstart: run DLCLAttack against a small, real Hugging Face model,
compiled with a real, opaque compiler (ONNX Runtime).

Run from the repository root with the ``examples`` extra installed:

    pip install -e ".[examples]"
    python examples/quickstart.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx_compiler import make_onnxruntime_compiler  # a real cl_func; see that file's docstring

from DLCLAttack import Attacker

MODEL_NAME = "hf-internal-testing/tiny-random-BertForSequenceClassification"


def build_toy_sentiment_dataset(tokenizer) -> list[dict]:
    """A tiny, self-contained sentiment dataset so this example needs no
    network access beyond downloading the model/tokenizer once. Any real
    Hugging Face ``datasets.Dataset`` of {"input_ids", "attention_mask",
    "label"} items works the same way -- see README.md's dataset contract.
    """
    texts = [
        "this movie was great",
        "i loved the acting",
        "what a fantastic film",
        "an absolute delight to watch",
        "terrible waste of time",
        "i hated every minute",
        "worst movie ever made",
        "boring and predictable plot",
    ] * 6
    labels = [1, 1, 1, 1, 0, 0, 0, 0] * 6
    items = []
    for text, label in zip(texts, labels):
        encoded = tokenizer(text, truncation=True, max_length=16)
        items.append(
            {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "label": label}
        )
    return items


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_dataset = build_toy_sentiment_dataset(tokenizer)
    test_dataset = train_dataset[:16]

    # A real, opaque compiler: exports whatever module it's given to ONNX
    # and serves it through onnxruntime.InferenceSession. Any other
    # model -> compiled_model callable (TVM, torch.compile, TensorRT, ...)
    # plugs in exactly the same way -- DLCLAttack never imports a compiler
    # backend itself.
    cl_func = make_onnxruntime_compiler(model)

    config = {
        "target_label": 1,
        "seed": 0,
        "device": "cpu",
        "trigger_length": 3,
        "trigger_steps": 150,
        "split_layer_ratio": 0.5,
        "finetune_epochs": 40,
        "finetune_lr": 1e-3,
        "batch_size": 8,
        "guard_bias_num_candidates": 50,
    }

    attack = Attacker(config)
    bd_model, logs = attack.run(model, train_dataset, test_dataset, cl_func)

    print(f"backdoored model type: {type(bd_model).__name__}")
    print(f"split at block {logs['split']['split_index']}/{logs['split']['num_blocks']} "
          f"({logs['split']['block_list_name']})")
    print(f"trigger optimization final loss: {logs['trigger']['final_loss']:.4f}")
    print(
        f"guard-bias search: {logs['guard_bias']['num_unresolved_channels']}/"
        f"{logs['guard_bias']['hidden_size']} channels unresolved"
    )
    print(f"fine-tune epoch losses (first 3, last 3): "
          f"{logs['finetune']['epoch_losses'][:3]} ... {logs['finetune']['epoch_losses'][-3:]}")
    print("evaluation report:")
    print(json.dumps(logs["evaluation"], indent=2))
    print(f"total wall-clock time: {logs['timing_seconds']['total_seconds']:.2f}s")
    print(
        "\nNote: this example runs on a 5-block, 32-dim *test-only* model so it "
        "downloads and finishes in seconds. At this scale, ONNX Runtime's real "
        "numeric deviation from PyTorch eager (~1e-9 absolute) is small relative "
        "to the model's natural activation variance, so pre/post-compile ASR "
        "separation is weaker than the paper's results on production-scale "
        "vision models. See README.md's 'Fidelity & Empirical Notes' section."
    )


if __name__ == "__main__":
    main()

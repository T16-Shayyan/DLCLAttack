"""Config schema, validation, and defaults for DLCLAttack.

Every attack knob is a plain, JSON-serializable field on ``AttackConfig``.
``AttackConfig.from_dict`` is the single validation choke point: it fills in
defaults, type-checks every field, range-checks numeric fields, and raises
:class:`ConfigError` with a field-specific message on the first problem it
finds. Nothing downstream re-validates -- by the time an ``AttackConfig``
instance exists, it is known-good.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Optional


class ConfigError(ValueError):
    """Raised when a config dict fails validation. Message names the field."""


@dataclass(frozen=True)
class AttackConfig:
    """Validated configuration for a single :class:`~DLCLAttack.Attacker` run.

    Field reference
    ----------------
    target_label : int
        The class index (``y*`` in the paper) that a triggered input should
        be classified as *after* compilation. Required -- there is no sane
        default for "what the attacker wants".
    seed : int
        Random seed applied to ``random``, ``numpy``, and ``torch`` (CPU and
        CUDA) at the start of :meth:`Attacker.run`, so the same config +
        inputs reproduce the same trigger, guard-bias, and fine-tuned model.
    device : str
        One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``. ``"auto"``
        picks CUDA, then MPS, then CPU, in that order of preference.
    label_key : str
        The key used to read the ground-truth label out of each dataset
        item (e.g. ``"label"`` or ``"labels"``).
    trigger_length : int
        Number of learnable embedding-space trigger positions prepended to
        each input sequence (the continuous analogue of the paper's pixel
        trigger, generalized to work with any tokenized text input).
    trigger_lr : float
        Learning rate for the Adam optimizer used in trigger optimization
        (Eq. 6 of the paper).
    trigger_steps : int
        Number of gradient steps taken while optimizing the trigger.
    trigger_margin_k : float
        The margin ``K`` added on top of the clean activation ceiling
        ``lambda`` in Eq. 6: the trigger is optimized so the first
        sub-model's activation on triggered input exceeds ``lambda + K``.
    split_layer_ratio : float
        Where to split the model into sub-model 1 / sub-model 2, expressed
        as a fraction of transformer block depth in ``(0, 1)``. The block
        index is ``round(ratio * num_blocks)``, clamped to
        ``[1, num_blocks - 1]`` so both halves are non-empty.
    guard_bias_tau_init : float
        Starting confidence threshold (tau) for the per-channel guard-bias
        search (Algorithm 1). The paper starts at 0.95.
    guard_bias_tau_step : float
        Amount tau is decreased by when no candidate bias clears the
        current threshold for a channel, repeated until a candidate is
        found or ``guard_bias_tau_min`` is reached.
    guard_bias_tau_min : float
        Floor for the tau decay loop above (engineering safeguard not
        specified in the paper, added so the search always terminates).
    guard_bias_num_candidates : int
        Number of candidate bias values scanned per channel between that
        channel's observed min and max activation.
    finetune_lr : float
        Learning rate for fine-tuning sub-model 2's parameters (Eq. 7).
    finetune_epochs : int
        Number of passes over ``train_dataset`` during fine-tuning.
    finetune_loss_weights : tuple[float, float, float, float]
        Weights ``(w1, w2, w3, w4)`` on Eq. 7's four loss terms before
        summing; see the field's own docstring below for why this exists.
    batch_size : int
        Batch size used for every stage (trigger optimization, guard-bias
        statistics collection, fine-tuning, evaluation).
    max_train_batches : Optional[int]
        If set, caps the number of batches consumed per epoch/stage from
        ``train_dataset``. Useful for smoke tests and quick iteration;
        ``None`` means "use the whole dataset".
    max_eval_batches : Optional[int]
        Same cap, applied to evaluation passes over ``test_dataset``.
    """

    target_label: int
    seed: int = 42
    device: str = "auto"
    label_key: str = "label"

    trigger_length: int = 4
    trigger_lr: float = 0.05
    trigger_steps: int = 50
    trigger_margin_k: float = 1.0

    split_layer_ratio: float = 0.5

    guard_bias_tau_init: float = 0.95
    guard_bias_tau_step: float = 0.05
    guard_bias_tau_min: float = 0.5
    guard_bias_num_candidates: int = 25

    finetune_lr: float = 5e-4
    finetune_epochs: int = 3
    finetune_loss_weights: tuple = (1.0, 1.0, 1.0, 1.0)
    """Weights ``(w1, w2, w3, w4)`` applied to the four Eq. 7 loss terms
    ``(l1, l2, l3, l4)`` before summing. Eq. 7 weights them equally
    (``1, 1, 1, 1``); in practice the three utility/stealth terms (l1-l3)
    can dominate joint optimization and starve l4 (the actual backdoor
    objective) of gradient signal, especially on small models where a
    single "always predict the true label" solution cheaply satisfies
    three of the four terms. Raising w4 trades some stealth margin for
    reliably learning the backdoor; the default keeps the paper's
    unweighted formulation."""

    batch_size: int = 8
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None

    extra: dict = field(default_factory=dict)
    """Free-form bag for caller-specific metadata echoed back in logs
    (e.g. an experiment name). Never read by the attack itself."""

    @staticmethod
    def from_dict(config: dict) -> "AttackConfig":
        if not isinstance(config, dict):
            raise ConfigError(f"config must be a dict, got {type(config).__name__}")

        known_fields = {f.name for f in fields(AttackConfig)}
        unknown = set(config) - known_fields
        if unknown:
            raise ConfigError(
                f"unknown config field(s): {sorted(unknown)}. "
                f"Valid fields are: {sorted(known_fields)}"
            )

        if "target_label" not in config:
            raise ConfigError("config['target_label'] is required (int class index)")

        merged: dict[str, Any] = {"extra": {}}
        for spec in fields(AttackConfig):
            if spec.name == "extra":
                continue
            if spec.name in config:
                merged[spec.name] = config[spec.name]
            elif spec.default is not MISSING:
                merged[spec.name] = spec.default
            # target_label has no default and is required; checked above.

        if "extra" in config:
            if not isinstance(config["extra"], dict):
                raise ConfigError("config['extra'] must be a dict if provided")
            merged["extra"] = dict(config["extra"])

        def require_int(key: str) -> int:
            v = merged[key]
            if isinstance(v, bool) or not isinstance(v, int):
                raise ConfigError(f"config['{key}'] must be an int, got {v!r}")
            return v

        def require_positive_int(key: str) -> int:
            v = require_int(key)
            if v <= 0:
                raise ConfigError(f"config['{key}'] must be a positive int, got {v!r}")
            return v

        def require_float(key: str) -> float:
            v = merged[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ConfigError(f"config['{key}'] must be a number, got {v!r}")
            return float(v)

        def require_positive_float(key: str) -> float:
            v = require_float(key)
            if v <= 0:
                raise ConfigError(f"config['{key}'] must be > 0, got {v!r}")
            return v

        def require_range(key: str, lo: float, hi: float, inclusive: bool = False) -> float:
            v = require_float(key)
            ok = (lo <= v <= hi) if inclusive else (lo < v < hi)
            if not ok:
                bounds = f"[{lo}, {hi}]" if inclusive else f"({lo}, {hi})"
                raise ConfigError(f"config['{key}'] must be in {bounds}, got {v!r}")
            return v

        merged["target_label"] = require_int("target_label")
        merged["seed"] = require_int("seed")

        if merged["device"] not in ("auto", "cpu", "cuda", "mps"):
            raise ConfigError(
                f"config['device'] must be one of 'auto', 'cpu', 'cuda', 'mps', got {merged['device']!r}"
            )
        if not isinstance(merged["label_key"], str) or not merged["label_key"]:
            raise ConfigError(f"config['label_key'] must be a non-empty string, got {merged['label_key']!r}")

        merged["trigger_length"] = require_positive_int("trigger_length")
        merged["trigger_lr"] = require_positive_float("trigger_lr")
        merged["trigger_steps"] = require_positive_int("trigger_steps")
        merged["trigger_margin_k"] = require_float("trigger_margin_k")

        merged["split_layer_ratio"] = require_range("split_layer_ratio", 0.0, 1.0, inclusive=False)

        merged["guard_bias_tau_init"] = require_range("guard_bias_tau_init", 0.0, 1.0, inclusive=True)
        merged["guard_bias_tau_step"] = require_positive_float("guard_bias_tau_step")
        merged["guard_bias_tau_min"] = require_range("guard_bias_tau_min", 0.0, 1.0, inclusive=True)
        if merged["guard_bias_tau_min"] > merged["guard_bias_tau_init"]:
            raise ConfigError(
                "config['guard_bias_tau_min'] must be <= config['guard_bias_tau_init'] "
                f"(got tau_min={merged['guard_bias_tau_min']!r} > tau_init={merged['guard_bias_tau_init']!r})"
            )
        merged["guard_bias_num_candidates"] = require_positive_int("guard_bias_num_candidates")

        merged["finetune_lr"] = require_positive_float("finetune_lr")
        merged["finetune_epochs"] = require_positive_int("finetune_epochs")

        weights = merged["finetune_loss_weights"]
        if not isinstance(weights, (list, tuple)) or len(weights) != 4:
            raise ConfigError(
                "config['finetune_loss_weights'] must be a 4-element sequence "
                f"(w1, w2, w3, w4), got {weights!r}"
            )
        weights = tuple(float(w) for w in weights)
        if any(isinstance(w, bool) for w in merged["finetune_loss_weights"]) or any(w <= 0 for w in weights):
            raise ConfigError(
                f"config['finetune_loss_weights'] entries must all be > 0, got {weights!r}"
            )
        merged["finetune_loss_weights"] = weights

        merged["batch_size"] = require_positive_int("batch_size")

        for key in ("max_train_batches", "max_eval_batches"):
            v = merged.get(key)
            if v is not None:
                if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                    raise ConfigError(f"config['{key}'] must be a positive int or None, got {v!r}")

        return AttackConfig(**merged)

    def to_dict(self) -> dict:
        return asdict(self)

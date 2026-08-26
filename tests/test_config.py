"""Config validation: every field's failure mode is checked, since
AttackConfig.from_dict is the package's single validation choke point."""

import pytest

from DLCLAttack.config import AttackConfig, ConfigError


def base():
    return {"target_label": 1}


def test_minimal_valid_config_uses_defaults():
    cfg = AttackConfig.from_dict(base())
    assert cfg.target_label == 1
    assert cfg.seed == 42
    assert cfg.device == "auto"
    assert cfg.trigger_length == 4
    assert cfg.finetune_loss_weights == (1.0, 1.0, 1.0, 1.0)


def test_not_a_dict_raises():
    with pytest.raises(ConfigError):
        AttackConfig.from_dict(["target_label", 1])  # type: ignore[arg-type]


def test_missing_target_label_raises():
    with pytest.raises(ConfigError, match="target_label"):
        AttackConfig.from_dict({})


def test_unknown_field_raises():
    with pytest.raises(ConfigError, match="unknown"):
        AttackConfig.from_dict({**base(), "not_a_real_field": 1})


def test_target_label_must_be_int():
    with pytest.raises(ConfigError, match="target_label"):
        AttackConfig.from_dict({"target_label": "cat"})


def test_target_label_bool_rejected():
    # bool is a subclass of int in Python; explicitly rejected so a typo
    # like target_label=True doesn't silently become class index 1.
    with pytest.raises(ConfigError, match="target_label"):
        AttackConfig.from_dict({"target_label": True})


@pytest.mark.parametrize("device", ["gpu", "gpu:0", "", "TPU"])
def test_invalid_device_raises(device):
    with pytest.raises(ConfigError, match="device"):
        AttackConfig.from_dict({**base(), "device": device})


@pytest.mark.parametrize("value", [0, -0.1, 1.0, 1.5])
def test_split_layer_ratio_must_be_open_interval(value):
    with pytest.raises(ConfigError, match="split_layer_ratio"):
        AttackConfig.from_dict({**base(), "split_layer_ratio": value})


def test_split_layer_ratio_boundary_valid():
    cfg = AttackConfig.from_dict({**base(), "split_layer_ratio": 0.99})
    assert cfg.split_layer_ratio == 0.99


@pytest.mark.parametrize("field", ["trigger_length", "trigger_steps", "finetune_epochs", "batch_size"])
def test_positive_int_fields_reject_zero_and_negative(field):
    with pytest.raises(ConfigError, match=field):
        AttackConfig.from_dict({**base(), field: 0})
    with pytest.raises(ConfigError, match=field):
        AttackConfig.from_dict({**base(), field: -3})


@pytest.mark.parametrize("field", ["trigger_lr", "finetune_lr"])
def test_positive_float_fields_reject_non_positive(field):
    with pytest.raises(ConfigError, match=field):
        AttackConfig.from_dict({**base(), field: 0.0})


def test_guard_bias_tau_min_cannot_exceed_tau_init():
    with pytest.raises(ConfigError, match="tau_min"):
        AttackConfig.from_dict({**base(), "guard_bias_tau_init": 0.6, "guard_bias_tau_min": 0.9})


def test_max_train_batches_must_be_positive_or_none():
    cfg = AttackConfig.from_dict({**base(), "max_train_batches": None})
    assert cfg.max_train_batches is None
    with pytest.raises(ConfigError, match="max_train_batches"):
        AttackConfig.from_dict({**base(), "max_train_batches": 0})


def test_finetune_loss_weights_wrong_length_raises():
    with pytest.raises(ConfigError, match="finetune_loss_weights"):
        AttackConfig.from_dict({**base(), "finetune_loss_weights": (1.0, 1.0, 1.0)})


def test_finetune_loss_weights_non_positive_raises():
    with pytest.raises(ConfigError, match="finetune_loss_weights"):
        AttackConfig.from_dict({**base(), "finetune_loss_weights": (1.0, 1.0, 1.0, 0.0)})


def test_extra_field_is_echoed_but_not_validated():
    cfg = AttackConfig.from_dict({**base(), "extra": {"experiment_name": "run-1"}})
    assert cfg.extra == {"experiment_name": "run-1"}


def test_to_dict_roundtrip():
    cfg = AttackConfig.from_dict(base())
    as_dict = cfg.to_dict()
    assert as_dict["target_label"] == 1
    # to_dict output must itself be a valid config (JSON-serializable, re-validatable)
    AttackConfig.from_dict({k: v for k, v in as_dict.items() if k != "extra"})

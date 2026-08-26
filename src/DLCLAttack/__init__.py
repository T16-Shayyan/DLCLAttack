"""DLCLAttack: a general-purpose deep-learning compiler backdoor attack toolbox.

Implements DcL-BD ("A General Compiler Backdoor Attack",
https://arxiv.org/abs/2509.11173) against arbitrary Hugging Face models.

Public API
----------
>>> from DLCLAttack import Attacker
>>> attack = Attacker(config)
>>> bd_model, logs = attack.run(model, train_dataset, test_dataset, cl_func)
"""

from .attacker import Attacker
from .config import AttackConfig, ConfigError
from .evaluate import (
    EvaluationReport,
    attack_success_rate,
    clean_accuracy,
    consistency_rate,
    evaluate_attack,
)

__all__ = [
    "Attacker",
    "AttackConfig",
    "ConfigError",
    "EvaluationReport",
    "evaluate_attack",
    "clean_accuracy",
    "attack_success_rate",
    "consistency_rate",
]

__version__ = "0.1.0"

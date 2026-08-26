"""``Attacker``: the public entry point tying every stage together."""

from __future__ import annotations

import logging
import time

import torch
from torch import nn

from .adapters import (
    SubModel1,
    freeze_all,
    install_guard_bias_hook,
    locate_split_point,
    unfreeze_submodel2,
)
from .config import AttackConfig
from .data import iterate_batches, to_embeds
from .evaluate import evaluate_attack
from .finetune import finetune_submodel2
from .guard_bias import search_guard_bias
from .trigger import EmbeddingTrigger, optimize_trigger
from .utils import resolve_device, set_seed

logger = logging.getLogger("DLCLAttack")


class Attacker:
    """Runs the DcL-BD compiler backdoor attack against a Hugging Face model.

    Usage
    -----
    >>> from DLCLAttack import Attacker
    >>> attack = Attacker(config)
    >>> bd_model, logs = attack.run(model, train_dataset, test_dataset, cl_func)

    See :class:`DLCLAttack.config.AttackConfig` for every accepted config
    field. ``model`` may be any ``transformers.PreTrainedModel`` built from
    a stack of repeated transformer blocks (see
    :func:`DLCLAttack.adapters.find_block_list`); no architecture-specific
    code is required to switch models.
    """

    def __init__(self, config: dict):
        self.config: AttackConfig = AttackConfig.from_dict(config)

    def run(self, model: nn.Module, train_dataset, test_dataset, cl_func):
        cfg = self.config
        start = time.time()
        timing: dict[str, float] = {}

        set_seed(cfg.seed)
        device = resolve_device(cfg.device)
        model.to(device)
        model.eval()
        freeze_all(model)

        t0 = time.time()
        split_point = locate_split_point(model, cfg.split_layer_ratio)
        logger.info(
            "split model at block %d/%d in '%s'",
            split_point.split_index,
            len(split_point.block_list),
            split_point.block_list_name,
        )
        sub_model_1 = SubModel1(model, split_point)
        compiled_sub_model_1 = cl_func(sub_model_1)
        timing["setup_and_compile_seconds"] = time.time() - t0

        hidden_size = model.get_input_embeddings().weight.shape[1]
        dtype = model.get_input_embeddings().weight.dtype
        trigger = EmbeddingTrigger(cfg.trigger_length, hidden_size, dtype=dtype).to(device)

        t0 = time.time()
        train_raw_batches = list(
            iterate_batches(train_dataset, cfg.batch_size, cfg.label_key, max_batches=cfg.max_train_batches)
        )
        if not train_raw_batches:
            raise ValueError("train_dataset produced no batches")
        clean_batches = [to_embeds(model, b, device) for b in train_raw_batches]
        timing["data_prep_seconds"] = time.time() - t0

        t0 = time.time()
        trigger_stats = optimize_trigger(
            sub_model_1, split_point, trigger, clean_batches, cfg.trigger_steps, cfg.trigger_lr, cfg.trigger_margin_k
        )
        timing["trigger_optimization_seconds"] = time.time() - t0
        logger.info("trigger optimized: final loss=%.6f", trigger_stats["final_loss"])

        t0 = time.time()
        m_clean, c_clean, m_trig, c_trig = [], [], [], []
        with torch.no_grad():
            for embeds, mask in clean_batches:
                triggered_embeds, triggered_mask = trigger.apply(embeds, mask)
                m_clean.append(sub_model_1(embeds, mask).mean(dim=1).cpu())
                m_trig.append(sub_model_1(triggered_embeds, triggered_mask).mean(dim=1).cpu())
                c_clean.append(
                    torch.as_tensor(compiled_sub_model_1(embeds, mask)).mean(dim=1).cpu()
                )
                c_trig.append(
                    torch.as_tensor(compiled_sub_model_1(triggered_embeds, triggered_mask)).mean(dim=1).cpu()
                )
        benign_hidden = torch.cat(m_clean + c_clean + m_trig, dim=0)
        adv_hidden = torch.cat(c_trig, dim=0)
        guard_result = search_guard_bias(
            benign_hidden,
            adv_hidden,
            cfg.guard_bias_tau_init,
            cfg.guard_bias_tau_step,
            cfg.guard_bias_tau_min,
            cfg.guard_bias_num_candidates,
        )
        guard_bias = guard_result.bias.to(device=device, dtype=dtype)
        timing["guard_bias_search_seconds"] = time.time() - t0
        logger.info(
            "guard-bias search done: %d/%d channels unresolved at tau_min",
            len(guard_result.unresolved_channels),
            hidden_size,
        )

        t0 = time.time()
        unfreeze_submodel2(model, split_point)
        # Deliberately left in eval() mode (dropout disabled) for the whole
        # attack: fine-tuning here is fitting a fixed 4-term loss on the
        # split point's activations, not general-purpose regularized
        # training, and dropout noise would make h_m_clean/h_m_trig
        # inconsistent with the values guard-bias search was computed on.
        finetune_result = finetune_submodel2(
            model,
            split_point,
            sub_model_1,
            compiled_sub_model_1,
            trigger,
            guard_bias,
            cfg.target_label,
            device,
            train_raw_batches,
            cfg.finetune_epochs,
            cfg.finetune_lr,
            cfg.finetune_loss_weights,
        )
        timing["finetune_seconds"] = time.time() - t0
        logger.info("fine-tune done: epoch losses=%s", finetune_result.epoch_losses)

        install_guard_bias_hook(split_point, guard_bias)
        freeze_all(model)  # attack is done; leave the returned model in inference mode
        bd_model = model

        t0 = time.time()
        report = evaluate_attack(
            bd_model,
            cl_func,
            test_dataset,
            trigger,
            cfg.target_label,
            device,
            cfg.label_key,
            cfg.batch_size,
            cfg.max_eval_batches,
        )
        timing["evaluation_seconds"] = time.time() - t0
        timing["total_seconds"] = time.time() - start

        logs = {
            "config": cfg.to_dict(),
            "device": str(device),
            "split": {
                "block_list_name": split_point.block_list_name,
                "num_blocks": len(split_point.block_list),
                "split_index": split_point.split_index,
            },
            "trigger": {k: v for k, v in trigger_stats.items() if k != "loss_history"},
            "trigger_loss_history": trigger_stats["loss_history"],
            "guard_bias": {
                "hidden_size": hidden_size,
                "num_unresolved_channels": len(guard_result.unresolved_channels),
                "mean_tau_used": sum(guard_result.tau_used) / len(guard_result.tau_used),
            },
            "finetune": {
                "epoch_losses": finetune_result.epoch_losses,
                "num_steps": len(finetune_result.per_term_losses),
                "per_term_losses": finetune_result.per_term_losses,
            },
            "evaluation": report.to_dict(),
            "timing_seconds": timing,
        }
        return bd_model, logs

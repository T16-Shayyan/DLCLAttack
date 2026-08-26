"""Guard-bias search (Algorithm 1 in the paper).

Reduces the "four input-model combinations" problem to two states
(activated / not activated) by finding a per-channel bias ``V`` such that,
once subtracted from the split point's hidden state, only the *compiled*
model's response to a *triggered* input lands above zero on enough
channels -- while the original model on clean input, the original model on
triggered input, and the compiled model on clean input all land below zero.

A note on fidelity to the source PDF: the paper's Algorithm 1 pseudocode
renders the per-candidate acceptance test as ``P_M^2 > tau and P_C^2 > tau``
after OCR extraction, which is ambiguous (squared probabilities would make
the threshold *easier* to clear as tau shrinks below 1, which contradicts
the surrounding prose about a *shrinking* tau making acceptance *easier*).
We implement the natural reading consistent with the prose -- "likelihood
exceeding a predefined threshold tau" -- as ``P_M > tau and P_C > tau``.
This is a documented interpretation, not a guess hidden in the code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GuardBiasResult:
    bias: torch.Tensor  # [hidden_size]
    tau_used: list[float]  # tau at which each channel's bias was accepted
    unresolved_channels: list[int]  # channels that never cleared tau_min


def _search_channel(
    benign: torch.Tensor,  # [N_benign] scalar activations for this channel
    adv: torch.Tensor,  # [N_adv]
    tau_init: float,
    tau_step: float,
    tau_min: float,
    num_candidates: int,
) -> tuple[float, float, bool]:
    """Search one channel for a bias V. Returns (V, tau_at_which_found, found).

    Falls back to the midpoint of the channel's overall range if no
    candidate clears ``tau_min`` (an engineering safeguard the paper's
    pseudocode does not specify, needed so the search always terminates).
    """
    combined = torch.cat([benign, adv])
    v_min, v_max = combined.min().item(), combined.max().item()
    if v_min == v_max:
        return v_min, tau_init, True

    candidates = torch.linspace(v_min, v_max, num_candidates)
    tau = tau_init
    while tau >= tau_min:
        accepted = []
        for v in candidates:
            p_benign_below = (benign - v < 0).float().mean().item()
            p_adv_above = (adv - v > 0).float().mean().item()
            if p_benign_below > tau and p_adv_above > tau:
                accepted.append(v.item())
        if accepted:
            return (min(accepted) + max(accepted)) / 2.0, tau, True
        tau -= tau_step

    return (v_min + v_max) / 2.0, tau_min, False


def search_guard_bias(
    benign_hidden: torch.Tensor,  # [N_benign, hidden_size]
    adv_hidden: torch.Tensor,  # [N_adv, hidden_size]
    tau_init: float,
    tau_step: float,
    tau_min: float,
    num_candidates: int,
) -> GuardBiasResult:
    """Run the per-channel search of Algorithm 1 across every hidden
    dimension. ``benign_hidden`` / ``adv_hidden`` are the sets
    ``E_benign = {M1(X), C1(X), M1(X ⊕ t)}`` and ``E_adv = {C1(X ⊕ t)}``
    from the paper, already reduced to one scalar per example per channel
    (mean over sequence positions) and stacked into 2-D tensors.
    """
    hidden_size = benign_hidden.shape[1]
    bias = torch.zeros(hidden_size)
    tau_used = []
    unresolved = []
    for d in range(hidden_size):
        v, tau, found = _search_channel(
            benign_hidden[:, d], adv_hidden[:, d], tau_init, tau_step, tau_min, num_candidates
        )
        bias[d] = v
        tau_used.append(tau)
        if not found:
            unresolved.append(d)
    return GuardBiasResult(bias=bias, tau_used=tau_used, unresolved_channels=unresolved)

"""Averaging torch state dicts across clients with `formulas.federated.fedavg` [MCM17]."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..formulas.federated import fedavg

if TYPE_CHECKING:
    import torch


def fedavg_state(
    states: Sequence[dict[str, torch.Tensor]], weights: Sequence[float]
) -> dict[str, torch.Tensor]:
    """The federated average of several state dicts with the same keys: floating tensors averaged
    by `weights` (exact for one client), any other tensor taken from the first client."""
    import torch

    out: dict[str, torch.Tensor] = {}
    for key, first in states[0].items():
        if torch.is_floating_point(first):
            avg = fedavg([s[key].detach().cpu().numpy() for s in states], weights)
            out[key] = torch.from_numpy(avg).to(first.device)
        else:
            out[key] = first.detach().clone()
    return out


def state_bytes(state: dict[str, torch.Tensor]) -> int:
    """The size of a state dict on the wire, in bytes."""
    return int(sum(v.numel() * v.element_size() for v in state.values()))

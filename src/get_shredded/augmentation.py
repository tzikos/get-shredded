from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def make_batch_augmenter(
    augmentation_type: str,
    num_sensors: int,
    *,
    gaussian_std: float = 0.03,
    dropout_fill: float = 0.0,
) -> Callable[[Tensor], Tensor]:
    """Return a per-batch augmentation function for sensor inputs.

    Augmenters work on tensors of shape (batch, ..., num_sensors) — both
    SHRED windows (batch, lags, sensors) and SDN snapshots (batch, sensors).

    Args:
        augmentation_type: One of "none", "gaussian", "dropout", "hybrid".
        num_sensors: Number of sensor channels (last dimension).
        gaussian_std: Std-dev for Gaussian noise.
        dropout_fill: Fill value for dropped (dead) sensor channels.
    """
    aug = augmentation_type.lower()
    if aug not in {"none", "gaussian", "dropout", "hybrid"}:
        raise ValueError(
            f"Unknown augmentation type '{augmentation_type}'. "
            "Choose from: none, gaussian, dropout, hybrid."
        )

    if aug == "none":
        return lambda x: x

    if aug == "gaussian":
        def _gaussian(x: Tensor) -> Tensor:
            return x + torch.randn_like(x) * gaussian_std
        return _gaussian

    if aug == "dropout":
        def _dropout(x: Tensor) -> Tensor:
            out = x.clone()
            # zero out every sensor channel
            for s in range(num_sensors):
                out[..., s] = dropout_fill
            return out
        return _dropout

    # hybrid: sample K ~ Uniform[0, num_sensors], corrupt K randomly chosen sensors
    def _hybrid(x: Tensor) -> Tensor:
        out = x.clone()
        k = torch.randint(0, num_sensors + 1, ()).item()
        if k == 0:
            return out
        chosen = torch.randperm(num_sensors)[:k]
        for s in chosen.tolist():
            if torch.rand(()).item() < 0.5:
                out[..., s] = out[..., s] + torch.randn_like(out[..., s]) * gaussian_std
            else:
                out[..., s] = dropout_fill
        return out

    return _hybrid

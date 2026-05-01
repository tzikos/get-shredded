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
        augmentation_type: One of "none", "gaussian", "dropout", "hybrid", "burst".
        num_sensors: Number of sensor channels (last dimension).
        gaussian_std: Std-dev for Gaussian noise (gaussian/hybrid augmenters).
        dropout_fill: Fill value for dropped (dead) sensor channels.
    """
    aug = augmentation_type.lower()
    if aug not in {"none", "gaussian", "dropout", "hybrid", "burst"}:
        raise ValueError(
            f"Unknown augmentation type '{augmentation_type}'. "
            "Choose from: none, gaussian, dropout, hybrid, burst."
        )

    if aug == "none":
        return lambda x: x

    if aug == "gaussian":
        # K ~ Uniform[0, N] sensors get Gaussian noise; the rest are untouched.
        def _gaussian(x: Tensor) -> Tensor:
            out = x.clone()
            k = torch.randint(0, num_sensors + 1, ()).item()
            if k == 0:
                return out
            for s in torch.randperm(num_sensors)[:k].tolist():
                out[..., s] = out[..., s] + torch.randn_like(out[..., s]) * gaussian_std
            return out
        return _gaussian

    if aug == "dropout":
        # K ~ Uniform[0, N//2] sensors zeroed — at most half fail at once.
        _max_drop = max(1, num_sensors // 2)
        def _dropout(x: Tensor) -> Tensor:
            out = x.clone()
            k = torch.randint(0, _max_drop + 1, ()).item()
            if k == 0:
                return out
            for s in torch.randperm(num_sensors)[:k].tolist():
                out[..., s] = dropout_fill
            return out
        return _dropout

    if aug == "burst":
        # K ~ Uniform[0, N] sensors get signal-proportional noise.
        # Per element: noise_std ~ Uniform(0, |x|), so noise ranges from 0
        # up to the signal magnitude — the sensor reading becomes unreliable
        # without being fully zeroed.
        def _burst(x: Tensor) -> Tensor:
            out = x.clone()
            k = torch.randint(0, num_sensors + 1, ()).item()
            if k == 0:
                return out
            for s in torch.randperm(num_sensors)[:k].tolist():
                noise_std = torch.rand_like(out[..., s]) * out[..., s].abs()
                out[..., s] = out[..., s] + torch.randn_like(out[..., s]) * noise_std
            return out
        return _burst

    # hybrid: K ~ Uniform[0, N] sensors each independently get Gaussian noise,
    # zero-out, or nothing (1/3 each).
    def _hybrid(x: Tensor) -> Tensor:
        out = x.clone()
        k = torch.randint(0, num_sensors + 1, ()).item()
        if k == 0:
            return out
        for s in torch.randperm(num_sensors)[:k].tolist():
            r = torch.rand(()).item()
            if r < 1 / 3:
                out[..., s] = out[..., s] + torch.randn_like(out[..., s]) * gaussian_std
            elif r < 2 / 3:
                out[..., s] = dropout_fill
            # else: nothing (leave sensor clean)
        return out

    return _hybrid

from __future__ import annotations

from typing import Sequence

import numpy as np

ALLOWED_SENSOR_MODES = {"true", "white", "none"}


def resolve_sensor_modes(
    num_sensors: int,
    modes: Sequence[str] | None,
    *,
    auto_extend: bool,
    default_mode: str,
) -> list[str]:
    """Validate and normalize per-sensor modes to length `num_sensors`."""
    default_mode = str(default_mode).lower()
    if default_mode not in ALLOWED_SENSOR_MODES:
        raise ValueError(
            f"noise.default_mode must be one of {sorted(ALLOWED_SENSOR_MODES)}, got '{default_mode}'"
        )

    normalized: list[str] = []
    if modes is not None:
        normalized = [str(mode).lower() for mode in modes]

    for mode in normalized:
        if mode not in ALLOWED_SENSOR_MODES:
            raise ValueError(
                f"Unsupported noise mode '{mode}'. Allowed: {sorted(ALLOWED_SENSOR_MODES)}"
            )

    if len(normalized) > num_sensors:
        raise ValueError(
            f"noise.modes has length {len(normalized)} but num_sensors={num_sensors}."
        )

    if len(normalized) < num_sensors:
        if not auto_extend:
            raise ValueError(
                f"noise.modes has length {len(normalized)} but num_sensors={num_sensors}; "
                "set noise.auto_extend=true or provide one mode per sensor."
            )
        normalized = normalized + [default_mode] * (num_sensors - len(normalized))

    if not normalized:
        normalized = [default_mode] * num_sensors
    return normalized


def apply_sensor_noise(
    data: np.ndarray,
    sensor_modes: Sequence[str],
    *,
    white_std: float,
    none_fill_value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply per-sensor noise modes to sensor arrays.

    Works for arrays shaped (..., num_sensors), e.g.:
    - (T, num_sensors)
    - (N, lags, num_sensors)
    """
    if white_std < 0:
        raise ValueError(f"noise.white_std must be >= 0, got {white_std}")

    out = np.array(data, copy=True)
    if out.shape[-1] != len(sensor_modes):
        raise ValueError(
            f"sensor_modes length {len(sensor_modes)} does not match data.shape[-1]={out.shape[-1]}"
        )

    for sensor_idx, mode in enumerate(sensor_modes):
        if mode == "true":
            continue
        if mode == "white":
            noise = rng.normal(0.0, white_std, size=out[..., sensor_idx].shape)
            out[..., sensor_idx] = out[..., sensor_idx] + noise.astype(out.dtype, copy=False)
            continue
        if mode == "none":
            out[..., sensor_idx] = none_fill_value
            continue
        raise ValueError(f"Unsupported noise mode '{mode}'")

    return out
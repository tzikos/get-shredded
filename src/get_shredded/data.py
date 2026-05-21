from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.io import loadmat


def load_cylinder_data(mat_path: str | Path) -> tuple[np.ndarray, int, int]:
    """Returns (load_X, nx, ny) where load_X has shape (N, m): N temporal
    snapshots of an m-dimensional state — the convention used in the paper."""
    data = loadmat(str(mat_path))
    if "VORTALL" not in data:
        raise KeyError("Expected key 'VORTALL' in .mat file")
    vortall = np.asarray(data["VORTALL"], dtype=np.float32)
    load_X = vortall.T
    nx = int(data.get("nx", [[0]])[0][0])
    ny = int(data.get("ny", [[0]])[0][0])
    return load_X, nx, ny


def qr_place(data_matrix: np.ndarray, num_sensors: int) -> tuple[np.ndarray, np.ndarray]:
    """QR-pivoting sensor selection on the rank-`num_sensors` POD basis.

    data_matrix: (m, N) — m-dimensional state, N snapshots.
    Returns (sensor_locs, U_r) with U_r the leading `num_sensors` POD modes.
    """
    u, _, _ = np.linalg.svd(data_matrix, full_matrices=False)
    U_r = u[:, :num_sensors]
    _, _, pivot = scipy.linalg.qr(U_r.T, pivoting=True)
    sensor_locs = pivot[:num_sensors]
    return sensor_locs, U_r


def build_sensor_windows(
    transformed_X: np.ndarray, sensor_locations: np.ndarray, lags: int
) -> np.ndarray:
    """transformed_X: (N, m). Returns (N - lags, lags, num_sensors)."""
    n = transformed_X.shape[0]
    num_sensors = len(sensor_locations)
    out = np.zeros((n - lags, lags, num_sensors), dtype=np.float32)
    for i in range(n - lags):
        out[i] = transformed_X[i : i + lags, sensor_locations]
    return out


def qrpod_reconstruct(
    sensor_measurements: np.ndarray, sensor_locations: np.ndarray, U_r: np.ndarray, m: int
) -> np.ndarray:
    """Linear QR/POD reconstruction: x_hat = U_r (C U_r)^{-1} y.

    sensor_measurements: (T, num_sensors) in the original (unscaled) units.
    Returns (T, m).
    """
    num_sensors = len(sensor_locations)
    C = np.zeros((num_sensors, m), dtype=U_r.dtype)
    for i in range(num_sensors):
        C[i, sensor_locations[i]] = 1.0
    return (U_r @ np.linalg.inv(C @ U_r) @ sensor_measurements.T).T

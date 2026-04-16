from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass
class PODBasis:
    mean: np.ndarray
    modes: np.ndarray


def load_cylinder_data(mat_path: str | Path) -> tuple[np.ndarray, int, int]:
    data = loadmat(str(mat_path))
    if "VORTALL" not in data:
        raise KeyError("Expected key 'VORTALL' in .mat file")
    x = np.asarray(data["VORTALL"], dtype=np.float32)
    nx = int(data.get("nx", [[0]])[0][0])
    ny = int(data.get("ny", [[0]])[0][0])
    return x, nx, ny


def split_time_series(
    x: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = x.shape[1]
    t_train = int(t * train_ratio)
    t_val = int(t * (train_ratio + val_ratio))
    return x[:, :t_train], x[:, t_train:t_val], x[:, t_val:]


def fit_pod(x_train: np.ndarray, rank: int) -> PODBasis:
    mean = x_train.mean(axis=1, keepdims=True)
    x_centered = x_train - mean
    u, _, _ = np.linalg.svd(x_centered, full_matrices=False)
    modes = u[:, :rank]
    return PODBasis(mean=mean, modes=modes)


def project_to_latent(x: np.ndarray, basis: PODBasis) -> np.ndarray:
    x_centered = x - basis.mean
    return basis.modes.T @ x_centered


def reconstruct_from_latent(latent: np.ndarray, basis: PODBasis) -> np.ndarray:
    return basis.modes @ latent + basis.mean


def make_windows(latent: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    # latent: (rank, T)
    rank, t = latent.shape
    if t <= seq_len:
        raise ValueError("Need more timesteps than seq_len")

    x_seq = []
    y_next = []
    for idx in range(t - seq_len):
        x_seq.append(latent[:, idx : idx + seq_len].T)
        y_next.append(latent[:, idx + seq_len])
    return np.asarray(x_seq, dtype=np.float32), np.asarray(y_next, dtype=np.float32)

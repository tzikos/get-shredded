from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PODDMDBaseline:
    latent_dim: int
    ridge: float = 1e-6
    A: np.ndarray | None = None

    def fit_from_latent_series(self, latent: np.ndarray) -> None:
        # latent: (latent_dim, T)
        if latent.ndim != 2:
            raise ValueError("latent must have shape (latent_dim, T)")
        if latent.shape[1] < 2:
            raise ValueError("Need at least two timesteps to fit DMD")

        x = latent[:, :-1]
        y = latent[:, 1:]
        xx_t = x @ x.T
        reg = self.ridge * np.eye(xx_t.shape[0], dtype=latent.dtype)
        self.A = (y @ x.T) @ np.linalg.inv(xx_t + reg)

    def fit(self, x_seq: np.ndarray, y_next: np.ndarray) -> None:
        # Fit from windowed data using the last state in each window.
        # x_seq: (N, seq_len, latent_dim), y_next: (N, latent_dim)
        if x_seq.ndim != 3:
            raise ValueError("x_seq must have shape (N, seq_len, latent_dim)")
        if y_next.ndim != 2:
            raise ValueError("y_next must have shape (N, latent_dim)")
        if x_seq.shape[-1] != self.latent_dim or y_next.shape[-1] != self.latent_dim:
            raise ValueError("Input latent dimension does not match model latent_dim")

        x_t = x_seq[:, -1, :].T  # (latent_dim, N)
        y_t = y_next.T  # (latent_dim, N)
        xx_t = x_t @ x_t.T
        reg = self.ridge * np.eye(xx_t.shape[0], dtype=x_seq.dtype)
        self.A = (y_t @ x_t.T) @ np.linalg.inv(xx_t + reg)

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        if self.A is None:
            raise RuntimeError("Baseline model must be fit before prediction")
        if x_seq.ndim != 3:
            raise ValueError("x_seq must have shape (N, seq_len, latent_dim)")
        x_t = x_seq[:, -1, :]  # (N, latent_dim)
        return x_t @ self.A.T

    def rollout(self, seed_sequence: np.ndarray, horizon: int) -> np.ndarray:
        # seed_sequence: (1, seq_len, latent_dim)
        if self.A is None:
            raise RuntimeError("Baseline model must be fit before rollout")
        if seed_sequence.shape[0] != 1:
            raise ValueError("seed_sequence must have batch size 1")
        current_state = seed_sequence[0, -1, :].copy()  # (latent_dim,)
        outputs = []
        for _ in range(horizon):
            current_state = self.A @ current_state
            outputs.append(current_state.copy())
        return np.asarray(outputs, dtype=seed_sequence.dtype)

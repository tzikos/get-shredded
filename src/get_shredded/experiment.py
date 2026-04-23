"""Reusable single-run training + evaluation pipeline.

Used by the main reconstruction script and by the num_sensors sweep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from .baseline import QRPODBaseline
from .data import build_sensor_windows, load_cylinder_data, qr_place
from .model import SDN, SHRED, TimeSeriesDataset, fit


@dataclass
class RunResult:
    # Reconstructions in original (un-scaled) units, shape (T_test, m).
    shred_recon: np.ndarray
    sdn_recon: np.ndarray
    qrpod_recon: np.ndarray
    truth: np.ndarray  # (T_test, m)

    # Relative L2 errors over the test set.
    shred_err: float
    sdn_err: float
    qrpod_err: float

    # Per-snapshot relative L2 errors, shape (T_test,).
    shred_err_per_snap: np.ndarray
    sdn_err_per_snap: np.ndarray
    qrpod_err_per_snap: np.ndarray

    # Validation error history (one entry every 20 epochs).
    shred_val_history: np.ndarray
    sdn_val_history: np.ndarray

    # Spatial info for plotting.
    sensor_locations: np.ndarray
    nx: int
    ny: int
    placement: str
    num_sensors: int
    lags: int


def _per_snapshot_rel_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(pred - truth, axis=1)
    den = np.linalg.norm(truth, axis=1)
    den = np.where(den == 0, 1.0, den)
    return num / den


def _aggregate_rel_error(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))


def run_experiment(
    mat_path: str | Path,
    *,
    num_sensors: int,
    lags: int,
    placement: str,
    test_size: int,
    val_size: int,
    hidden_size: int,
    hidden_layers: int,
    l1: int,
    l2: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
    verbose: bool = True,
) -> RunResult:
    np.random.seed(seed)
    torch.manual_seed(seed)

    load_X, nx, ny = load_cylinder_data(mat_path)  # (N, m)
    n, m = load_X.shape

    # Sequential split over (n - lags) sliding windows.
    n_windows = n - lags
    if test_size + val_size >= n_windows:
        raise ValueError(f"test_size + val_size ({test_size + val_size}) >= n_windows ({n_windows})")
    train_end = n_windows - test_size - val_size
    val_end = n_windows - test_size
    train_indices = np.arange(0, train_end)
    valid_indices = np.arange(train_end, val_end)
    test_indices = np.arange(val_end, n_windows)

    if placement == "QR":
        sensor_locations, U_r = qr_place(load_X[train_indices].T, num_sensors)
    else:
        _, U_r = qr_place(load_X[train_indices].T, num_sensors)
        sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

    sc = MinMaxScaler().fit(load_X[train_indices])
    transformed_X = sc.transform(load_X).astype(np.float32)

    all_data_in = build_sensor_windows(transformed_X, sensor_locations, lags)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device)

    train_in = to_tensor(all_data_in[train_indices])
    valid_in = to_tensor(all_data_in[valid_indices])
    test_in = to_tensor(all_data_in[test_indices])

    train_out = to_tensor(transformed_X[train_indices + lags - 1])
    valid_out = to_tensor(transformed_X[valid_indices + lags - 1])
    test_out = to_tensor(transformed_X[test_indices + lags - 1])

    train_ds = TimeSeriesDataset(train_in, train_out)
    valid_ds = TimeSeriesDataset(valid_in, valid_out)
    test_ds = TimeSeriesDataset(test_in, test_out)

    train_ds_sdn = TimeSeriesDataset(train_in[:, -1, :], train_out)
    valid_ds_sdn = TimeSeriesDataset(valid_in[:, -1, :], valid_out)
    test_ds_sdn = TimeSeriesDataset(test_in[:, -1, :], test_out)

    shred = SHRED(num_sensors, m, hidden_size=hidden_size, hidden_layers=hidden_layers,
                  l1=l1, l2=l2, dropout=dropout).to(device)
    shred_hist = fit(shred, train_ds, valid_ds, batch_size=batch_size,
                     num_epochs=epochs, lr=lr, verbose=verbose, patience=patience)

    sdn = SDN(num_sensors, m, l1=l1, l2=l2, dropout=dropout).to(device)
    sdn_hist = fit(sdn, train_ds_sdn, valid_ds_sdn, batch_size=batch_size,
                   num_epochs=epochs, lr=lr, verbose=verbose, patience=patience)

    shred.eval(); sdn.eval()
    with torch.no_grad():
        shred_recon = sc.inverse_transform(shred(test_ds.X).detach().cpu().numpy())
        sdn_recon = sc.inverse_transform(sdn(test_ds_sdn.X).detach().cpu().numpy())
    truth = sc.inverse_transform(test_ds.Y.detach().cpu().numpy())

    truth_indices = test_indices + lags - 1
    qrpod = QRPODBaseline(num_sensors=num_sensors)
    qrpod.U_r = U_r
    qrpod.sensor_locations = sensor_locations
    qrpod.m = m
    qrpod_recon = qrpod.predict(load_X[truth_indices])

    return RunResult(
        shred_recon=shred_recon,
        sdn_recon=sdn_recon,
        qrpod_recon=qrpod_recon,
        truth=truth,
        shred_err=_aggregate_rel_error(shred_recon, truth),
        sdn_err=_aggregate_rel_error(sdn_recon, truth),
        qrpod_err=_aggregate_rel_error(qrpod_recon, truth),
        shred_err_per_snap=_per_snapshot_rel_error(shred_recon, truth),
        sdn_err_per_snap=_per_snapshot_rel_error(sdn_recon, truth),
        qrpod_err_per_snap=_per_snapshot_rel_error(qrpod_recon, truth),
        shred_val_history=shred_hist.numpy() if hasattr(shred_hist, "numpy") else np.asarray(shred_hist),
        sdn_val_history=sdn_hist.numpy() if hasattr(sdn_hist, "numpy") else np.asarray(sdn_hist),
        sensor_locations=np.asarray(sensor_locations),
        nx=nx,
        ny=ny,
        placement=placement,
        num_sensors=num_sensors,
        lags=lags,
    )

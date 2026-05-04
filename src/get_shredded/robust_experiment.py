"""Single-run training + evaluation pipeline for RobustSHRED."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from .augmentation import make_batch_augmenter
from .data import build_sensor_windows, load_cylinder_data, qr_place, qrpod_reconstruct
from .experiment import (
    RunResult,
    _aggregate_rel_error,
    _build_scenario_noisy,
    _per_snapshot_rel_error,
)
from .model import SHRED, TimeSeriesDataset
from .robust_model import RobustSHRED
from .robust_train import fit_robust


def run_robust_experiment(
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
    d_z: int = 8,
    phase1_epochs: int = 1000,
    phase2_epochs: int = 500,
    phase3_epochs: int = 200,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 1e-3,
    phase3_lr: float = 1e-4,
    phase1_patience: int = 5,
    phase2_patience: int = 5,
    phase3_patience: int = 5,
    batch_size: int = 64,
    beta_kl: float = 0.01,
    lambda_detect: float = 0.1,
    augmentation_type: str = "dropout",
    gaussian_std: float = 0.03,
    dropout_fill: float = 0.0,
    seed: int = 42,
    verbose: bool = True,
) -> RunResult:
    np.random.seed(seed)
    torch.manual_seed(seed)

    load_X, nx, ny = load_cylinder_data(mat_path)
    n, m = load_X.shape

    n_windows = n - lags
    if test_size + val_size >= n_windows:
        raise ValueError(
            f"test_size + val_size ({test_size + val_size}) >= n_windows ({n_windows})"
        )
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

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

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

    augment_fn = make_batch_augmenter(
        augmentation_type, num_sensors,
        gaussian_std=gaussian_std,
        dropout_fill=dropout_fill,
    )

    def corrupt_labels_fn(x_clean: torch.Tensor, x_aug: torch.Tensor) -> torch.Tensor:
        diff = (x_clean - x_aug).abs().mean(dim=1)                   # (B, p)
        magnitude = x_clean.abs().mean(dim=1).clamp(min=1e-6)        # (B, p)
        # Relative threshold handles both dropout (large ratio) and burst
        # (signal-proportional noise) correctly. A ratio < 1% means clean.
        return (diff / magnitude < 1e-2).float()

    vanilla_shred = SHRED(
        num_sensors, m,
        hidden_size=hidden_size, hidden_layers=hidden_layers,
        l1=l1, l2=l2, dropout=dropout,
    ).to(device)

    robust_shred = RobustSHRED(
        num_sensors, m,
        hidden_size=hidden_size, hidden_layers=hidden_layers,
        l1=l1, l2=l2, dropout=dropout, d_z=d_z,
    ).to(device)

    history = fit_robust(
        robust_shred, vanilla_shred,
        train_ds, valid_ds,
        batch_size=batch_size,
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        phase3_epochs=phase3_epochs,
        phase1_lr=phase1_lr,
        phase2_lr=phase2_lr,
        phase3_lr=phase3_lr,
        phase1_patience=phase1_patience,
        phase2_patience=phase2_patience,
        phase3_patience=phase3_patience,
        beta_kl=beta_kl,
        lambda_detect=lambda_detect,
        augment_fn=augment_fn,
        corrupt_labels_fn=corrupt_labels_fn,
        verbose=verbose,
    )

    robust_shred.eval()
    vanilla_shred.eval()
    with torch.no_grad():
        robust_recon = sc.inverse_transform(robust_shred(test_in).detach().cpu().numpy())
        vanilla_recon = sc.inverse_transform(vanilla_shred(test_in).detach().cpu().numpy())
    truth = sc.inverse_transform(test_out.detach().cpu().numpy())

    truth_indices = test_indices + lags - 1
    sensor_measurements = load_X[truth_indices][:, sensor_locations]
    qrpod_recon = qrpod_reconstruct(sensor_measurements, np.asarray(sensor_locations), U_r, m)

    phase1_hist = history["phase1_val_history"]
    phase3_hist = history["phase3_val_history"]

    return RunResult(
        shred_recon=robust_recon,
        sdn_recon=vanilla_recon,
        qrpod_recon=qrpod_recon,
        truth=truth,
        shred_err=_aggregate_rel_error(robust_recon, truth),
        sdn_err=_aggregate_rel_error(vanilla_recon, truth),
        qrpod_err=_aggregate_rel_error(qrpod_recon, truth),
        shred_err_per_snap=_per_snapshot_rel_error(robust_recon, truth),
        sdn_err_per_snap=_per_snapshot_rel_error(vanilla_recon, truth),
        qrpod_err_per_snap=_per_snapshot_rel_error(qrpod_recon, truth),
        shred_val_history=(
            phase3_hist.numpy() if hasattr(phase3_hist, "numpy") else np.asarray(phase3_hist)
        ),
        sdn_val_history=(
            phase1_hist.numpy() if hasattr(phase1_hist, "numpy") else np.asarray(phase1_hist)
        ),
        sensor_locations=np.asarray(sensor_locations),
        nx=nx,
        ny=ny,
        placement=placement,
        num_sensors=num_sensors,
        lags=lags,
        shred_state_dict={k: v.detach().cpu() for k, v in robust_shred.state_dict().items()},
        sdn_state_dict={k: v.detach().cpu() for k, v in vanilla_shred.state_dict().items()},
    )


def run_robust_experiment_detailed(
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
    d_z: int = 8,
    phase1_epochs: int = 1000,
    phase2_epochs: int = 500,
    phase3_epochs: int = 200,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 1e-3,
    phase3_lr: float = 1e-3,
    phase1_patience: int = 5,
    phase2_patience: int = 5,
    phase3_patience: int = 10,
    batch_size: int = 64,
    beta_kl: float = 0.01,
    lambda_detect: float = 0.1,
    augmentation_type: str = "burst",
    gaussian_std: float = 0.03,
    dropout_fill: float = 0.0,
    noisy_scenario: str = "dropout",
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Like run_robust_experiment but returns all phase histories + trust weights.

    Returns a dict with keys:
      phase1_val_history, phase2_val_history, phase3_val_history  (np.ndarray)
      robust_err, vanilla_err, qrpod_err                          (float)
      robust_per_snap, vanilla_per_snap                           (np.ndarray T_test)
      trust_clean, trust_noisy                                    (np.ndarray T_test × p)
      robust_recon, vanilla_recon, truth                          (np.ndarray T_test × m)
      sensor_locations, nx, ny
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    load_X, nx, ny = load_cylinder_data(mat_path)
    n, m = load_X.shape

    n_windows = n - lags
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

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    def to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device)

    train_in  = to_tensor(all_data_in[train_indices])
    valid_in  = to_tensor(all_data_in[valid_indices])
    test_in   = to_tensor(all_data_in[test_indices])
    train_out = to_tensor(transformed_X[train_indices + lags - 1])
    valid_out = to_tensor(transformed_X[valid_indices + lags - 1])
    test_out  = to_tensor(transformed_X[test_indices + lags - 1])

    train_ds = TimeSeriesDataset(train_in, train_out)
    valid_ds = TimeSeriesDataset(valid_in, valid_out)

    augment_fn = make_batch_augmenter(
        augmentation_type, num_sensors,
        gaussian_std=gaussian_std, dropout_fill=dropout_fill,
    )

    def corrupt_labels_fn(x_clean: torch.Tensor, x_aug: torch.Tensor) -> torch.Tensor:
        diff = (x_clean - x_aug).abs().mean(dim=1)
        magnitude = x_clean.abs().mean(dim=1).clamp(min=1e-6)
        return (diff / magnitude < 1e-2).float()

    vanilla_shred = SHRED(
        num_sensors, m,
        hidden_size=hidden_size, hidden_layers=hidden_layers,
        l1=l1, l2=l2, dropout=dropout,
    ).to(device)

    robust_shred = RobustSHRED(
        num_sensors, m,
        hidden_size=hidden_size, hidden_layers=hidden_layers,
        l1=l1, l2=l2, dropout=dropout, d_z=d_z,
    ).to(device)

    history = fit_robust(
        robust_shred, vanilla_shred,
        train_ds, valid_ds,
        batch_size=batch_size,
        phase1_epochs=phase1_epochs, phase1_lr=phase1_lr, phase1_patience=phase1_patience,
        phase2_epochs=phase2_epochs, phase2_lr=phase2_lr, phase2_patience=phase2_patience,
        phase3_epochs=phase3_epochs, phase3_lr=phase3_lr, phase3_patience=phase3_patience,
        beta_kl=beta_kl, lambda_detect=lambda_detect,
        augment_fn=augment_fn, corrupt_labels_fn=corrupt_labels_fn,
        verbose=verbose,
    )

    # Build noisy test set for trust weight analysis and noisy error evaluation
    rng = np.random.default_rng(seed + 9999)
    test_noisy_np = _build_scenario_noisy(
        all_data_in[test_indices], noisy_scenario, num_sensors,
        gaussian_std=gaussian_std, dropout_fill=dropout_fill, rng=rng,
    )
    test_in_noisy = to_tensor(test_noisy_np)

    robust_shred.eval()
    vanilla_shred.eval()
    with torch.no_grad():
        robust_recon       = sc.inverse_transform(robust_shred(test_in).detach().cpu().numpy())
        vanilla_recon      = sc.inverse_transform(vanilla_shred(test_in).detach().cpu().numpy())
        robust_noisy_recon = sc.inverse_transform(robust_shred(test_in_noisy).detach().cpu().numpy())
        vanilla_noisy_recon = sc.inverse_transform(vanilla_shred(test_in_noisy).detach().cpu().numpy())
        trust_clean        = robust_shred.get_trust_weights(test_in).detach().cpu().numpy()
        trust_noisy        = robust_shred.get_trust_weights(test_in_noisy).detach().cpu().numpy()

    truth = sc.inverse_transform(test_out.detach().cpu().numpy())

    truth_indices = test_indices + lags - 1
    sensor_meas   = load_X[truth_indices][:, sensor_locations]
    qrpod_recon   = qrpod_reconstruct(sensor_meas, np.asarray(sensor_locations), U_r, m)

    def _to_np(t: torch.Tensor) -> np.ndarray:
        return t.numpy() if hasattr(t, "numpy") else np.asarray(t)

    return {
        "seed":                seed,
        "phase1_val_history":  _to_np(history["phase1_val_history"]),
        "phase2_val_history":  _to_np(history["phase2_val_history"]),
        "phase3_val_history":  _to_np(history["phase3_val_history"]),
        "robust_err":              _aggregate_rel_error(robust_recon, truth),
        "vanilla_err":             _aggregate_rel_error(vanilla_recon, truth),
        "qrpod_err":               _aggregate_rel_error(qrpod_recon, truth),
        "robust_per_snap":         _per_snapshot_rel_error(robust_recon, truth),
        "vanilla_per_snap":        _per_snapshot_rel_error(vanilla_recon, truth),
        "robust_noisy_err":        _aggregate_rel_error(robust_noisy_recon, truth),
        "vanilla_noisy_err":       _aggregate_rel_error(vanilla_noisy_recon, truth),
        "robust_noisy_per_snap":   _per_snapshot_rel_error(robust_noisy_recon, truth),
        "vanilla_noisy_per_snap":  _per_snapshot_rel_error(vanilla_noisy_recon, truth),
        "trust_clean":             trust_clean,
        "trust_noisy":             trust_noisy,
        "robust_recon":        robust_recon,
        "vanilla_recon":       vanilla_recon,
        "truth":               truth,
        "sensor_locations":    np.asarray(sensor_locations),
        "nx":                  nx,
        "ny":                  ny,
        "noisy_scenario":      noisy_scenario,
        "num_sensors":         num_sensors,
    }

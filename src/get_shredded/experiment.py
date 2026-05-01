"""Reusable single-run training + evaluation pipeline.

Used by the main reconstruction script and by the num_sensors sweep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from .augmentation import make_batch_augmenter
from .data import build_sensor_windows, load_cylinder_data, qr_place, qrpod_reconstruct
from .model import SDN, SHRED, TimeSeriesDataset, fit
from .noise import apply_sensor_noise, resolve_sensor_modes


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

    # Trained model weights for checkpoint export / later visualization.
    shred_state_dict: dict[str, torch.Tensor]
    sdn_state_dict: dict[str, torch.Tensor]


SCENARIOS = ["gaussian", "dropout", "hybrid", "burst"]


@dataclass
class ModelResult:
    name: str
    recon_clean: np.ndarray                     # (T_test, m)
    err_clean: float
    err_per_snap_clean: np.ndarray
    recon_noisy: dict[str, np.ndarray]          # scenario -> (T_test, m)
    err_noisy: dict[str, float]
    err_per_snap_noisy: dict[str, np.ndarray]
    val_history: np.ndarray
    state_dict: dict[str, torch.Tensor] | None = field(default=None)


@dataclass
class RobustnessResult:
    models: list[ModelResult]     # 4×SHRED + 4×SDN + QR-POD
    truth: np.ndarray             # (T_test, m) ground truth (field, not sensor readings)
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


def _build_scenario_noisy(
    data: np.ndarray,
    scenario: str,
    num_sensors: int,
    *,
    gaussian_std: float,
    dropout_fill: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply scenario-specific noise to a sensor array of shape (T, ..., num_sensors).

    Gaussian — per-sample: K ~ Uniform[0, N] sensors get Gaussian noise; rest untouched.
    Dropout  — per-sample: K ~ Uniform[0, N//2] sensors zeroed; at most half fail at once.
    Hybrid   — per-sample: K ~ Uniform[0, N] sensors each get noise, zero, or nothing (1/3 each).
    Burst    — per-sample: K ~ Uniform[0, N] sensors get signal-proportional noise;
               per element noise_std ~ Uniform(0, |x|).
    """
    _max_drop = max(1, num_sensors // 2)
    out = data.copy()
    for t in range(out.shape[0]):
        if scenario == "gaussian":
            k = int(rng.integers(0, num_sensors + 1))
            if k == 0:
                continue
            chosen = rng.choice(num_sensors, size=k, replace=False)
            for s in chosen:
                noise = rng.normal(0.0, gaussian_std, out[t, ..., s].shape).astype(out.dtype)
                out[t, ..., s] = out[t, ..., s] + noise
        elif scenario == "dropout":
            k = int(rng.integers(0, _max_drop + 1))
            if k == 0:
                continue
            chosen = rng.choice(num_sensors, size=k, replace=False)
            for s in chosen:
                out[t, ..., s] = dropout_fill
        elif scenario == "burst":
            k = int(rng.integers(0, num_sensors + 1))
            if k == 0:
                continue
            chosen = rng.choice(num_sensors, size=k, replace=False)
            for s in chosen:
                noise_std = rng.uniform(0.0, np.abs(out[t, ..., s]))
                noise = rng.normal(0.0, 1.0, out[t, ..., s].shape).astype(out.dtype) * noise_std
                out[t, ..., s] = out[t, ..., s] + noise
        else:  # hybrid
            k = int(rng.integers(0, num_sensors + 1))
            if k == 0:
                continue
            chosen = rng.choice(num_sensors, size=k, replace=False)
            for s in chosen:
                r = rng.random()
                if r < 1 / 3:
                    noise = rng.normal(0.0, gaussian_std, out[t, ..., s].shape).astype(out.dtype)
                    out[t, ..., s] = out[t, ..., s] + noise
                elif r < 2 / 3:
                    out[t, ..., s] = dropout_fill
                # else: nothing
    return out


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
    noise_enabled: bool = False,
    noise_modes: list[str] | None = None,
    noise_white_std: float = 0.0,
    noise_none_fill_value: float = 0.0,
    noise_auto_extend: bool = True,
    noise_default_mode: str = "true",
    noise_seed: int | None = None,
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

    sensor_modes = resolve_sensor_modes(
        num_sensors,
        noise_modes if noise_enabled else ["true"] * num_sensors,
        auto_extend=noise_auto_extend,
        default_mode=noise_default_mode,
    )
    if noise_enabled:
        rng = np.random.default_rng(seed if noise_seed is None else noise_seed)
        all_data_in = apply_sensor_noise(
            all_data_in,
            sensor_modes,
            white_std=float(noise_white_std),
            none_fill_value=float(noise_none_fill_value),
            rng=rng,
        )

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
    sensor_measurements = load_X[truth_indices][:, sensor_locations]
    if noise_enabled:
        rng_qr = np.random.default_rng((seed if noise_seed is None else noise_seed) + 1)
        sensor_measurements = apply_sensor_noise(
            sensor_measurements,
            sensor_modes,
            white_std=float(noise_white_std),
            none_fill_value=float(noise_none_fill_value),
            rng=rng_qr,
        )
    qrpod_recon = qrpod_reconstruct(sensor_measurements, np.asarray(sensor_locations), U_r, m)

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
        shred_state_dict={k: v.detach().cpu() for k, v in shred.state_dict().items()},
        sdn_state_dict={k: v.detach().cpu() for k, v in sdn.state_dict().items()},
    )


_AUG_TYPES = ["none", "gaussian", "dropout", "hybrid", "burst"]
_AUG_LABELS = {"none": "clean", "gaussian": "gaussian", "dropout": "dropout", "hybrid": "hybrid", "burst": "burst"}


def run_robustness_comparison(
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
    gaussian_std: float = 0.03,
    test_noise_std: float = 0.03,
    dropout_fill: float = 0.0,
    verbose: bool = True,
) -> RobustnessResult:
    np.random.seed(seed)
    torch.manual_seed(seed)

    load_X, nx, ny = load_cylinder_data(mat_path)
    n, m = load_X.shape

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
    test_windows_clean = all_data_in[test_indices]          # (T_test, lags, num_sensors) numpy
    test_in_clean = to_tensor(test_windows_clean)

    # Build one deterministic noisy test set per scenario (fixed seeds, reproducible)
    test_tensors_noisy: dict[str, torch.Tensor] = {}
    for i, scenario in enumerate(SCENARIOS):
        rng_s = np.random.default_rng(seed + 100 + i)
        noisy_np = _build_scenario_noisy(
            test_windows_clean, scenario, num_sensors,
            gaussian_std=test_noise_std, dropout_fill=dropout_fill, rng=rng_s,
        )
        test_tensors_noisy[scenario] = to_tensor(noisy_np)

    train_out = to_tensor(transformed_X[train_indices + lags - 1])
    valid_out = to_tensor(transformed_X[valid_indices + lags - 1])
    test_out = to_tensor(transformed_X[test_indices + lags - 1])

    truth = sc.inverse_transform(test_out.detach().cpu().numpy())

    train_ds = TimeSeriesDataset(train_in, train_out)
    valid_ds = TimeSeriesDataset(valid_in, valid_out)
    train_ds_sdn = TimeSeriesDataset(train_in[:, -1, :], train_out)
    valid_ds_sdn = TimeSeriesDataset(valid_in[:, -1, :], valid_out)

    model_results: list[ModelResult] = []

    for aug_type in _AUG_TYPES:
        label = _AUG_LABELS[aug_type]
        augment_fn = make_batch_augmenter(
            aug_type, num_sensors, gaussian_std=gaussian_std, dropout_fill=dropout_fill,
        ) if aug_type != "none" else None

        # SHRED variant
        shred = SHRED(num_sensors, m, hidden_size=hidden_size, hidden_layers=hidden_layers,
                      l1=l1, l2=l2, dropout=dropout).to(device)
        shred_hist = fit(shred, train_ds, valid_ds, batch_size=batch_size,
                         num_epochs=epochs, lr=lr, verbose=verbose, patience=patience,
                         augment_fn=augment_fn)
        shred.eval()
        with torch.no_grad():
            shred_rc = sc.inverse_transform(shred(test_in_clean).detach().cpu().numpy())
            shred_rn = {
                s: sc.inverse_transform(shred(test_tensors_noisy[s]).detach().cpu().numpy())
                for s in SCENARIOS
            }
        model_results.append(ModelResult(
            name=f"SHRED-{label}",
            recon_clean=shred_rc,
            err_clean=_aggregate_rel_error(shred_rc, truth),
            err_per_snap_clean=_per_snapshot_rel_error(shred_rc, truth),
            recon_noisy=shred_rn,
            err_noisy={s: _aggregate_rel_error(shred_rn[s], truth) for s in SCENARIOS},
            err_per_snap_noisy={s: _per_snapshot_rel_error(shred_rn[s], truth) for s in SCENARIOS},
            val_history=shred_hist.numpy() if hasattr(shred_hist, "numpy") else np.asarray(shred_hist),
            state_dict={k: v.detach().cpu() for k, v in shred.state_dict().items()},
        ))

        # SDN variant — uses only last timestep
        sdn_augment_fn = make_batch_augmenter(
            aug_type, num_sensors, gaussian_std=gaussian_std, dropout_fill=dropout_fill,
        ) if aug_type != "none" else None

        sdn = SDN(num_sensors, m, l1=l1, l2=l2, dropout=dropout).to(device)
        sdn_hist = fit(sdn, train_ds_sdn, valid_ds_sdn, batch_size=batch_size,
                       num_epochs=epochs, lr=lr, verbose=verbose, patience=patience,
                       augment_fn=sdn_augment_fn)
        sdn.eval()
        test_clean_sdn = test_in_clean[:, -1, :]
        test_noisy_sdn = {s: test_tensors_noisy[s][:, -1, :] for s in SCENARIOS}
        with torch.no_grad():
            sdn_rc = sc.inverse_transform(sdn(test_clean_sdn).detach().cpu().numpy())
            sdn_rn = {
                s: sc.inverse_transform(sdn(test_noisy_sdn[s]).detach().cpu().numpy())
                for s in SCENARIOS
            }
        model_results.append(ModelResult(
            name=f"SDN-{label}",
            recon_clean=sdn_rc,
            err_clean=_aggregate_rel_error(sdn_rc, truth),
            err_per_snap_clean=_per_snapshot_rel_error(sdn_rc, truth),
            recon_noisy=sdn_rn,
            err_noisy={s: _aggregate_rel_error(sdn_rn[s], truth) for s in SCENARIOS},
            err_per_snap_noisy={s: _per_snapshot_rel_error(sdn_rn[s], truth) for s in SCENARIOS},
            val_history=sdn_hist.numpy() if hasattr(sdn_hist, "numpy") else np.asarray(sdn_hist),
            state_dict={k: v.detach().cpu() for k, v in sdn.state_dict().items()},
        ))

    # QR-POD — evaluated under clean and all scenario noisy sensor measurements.
    # Clip reconstruction to training data range: the linear operator can amplify
    # corrupted sensor values arbitrarily, so we bound outputs to physically valid values.
    truth_indices = test_indices + lags - 1
    sensor_clean = load_X[truth_indices][:, sensor_locations]
    _data_lo = sc.data_min_   # (m,) min per spatial location seen during training
    _data_hi = sc.data_max_   # (m,) max per spatial location seen during training

    def _qrpod_clipped(sensors: np.ndarray) -> np.ndarray:
        recon = qrpod_reconstruct(sensors, np.asarray(sensor_locations), U_r, m)
        return np.clip(recon, _data_lo, _data_hi)

    qrpod_rc = _qrpod_clipped(sensor_clean)
    qrpod_rn: dict[str, np.ndarray] = {}
    for i, scenario in enumerate(SCENARIOS):
        rng_qr = np.random.default_rng(seed + 200 + i)
        sensor_noisy = _build_scenario_noisy(
            sensor_clean, scenario, num_sensors,
            gaussian_std=test_noise_std, dropout_fill=dropout_fill, rng=rng_qr,
        )
        qrpod_rn[scenario] = _qrpod_clipped(sensor_noisy)

    model_results.append(ModelResult(
        name="QR-POD",
        recon_clean=qrpod_rc,
        err_clean=_aggregate_rel_error(qrpod_rc, truth),
        err_per_snap_clean=_per_snapshot_rel_error(qrpod_rc, truth),
        recon_noisy=qrpod_rn,
        err_noisy={s: _aggregate_rel_error(qrpod_rn[s], truth) for s in SCENARIOS},
        err_per_snap_noisy={s: _per_snapshot_rel_error(qrpod_rn[s], truth) for s in SCENARIOS},
        val_history=np.array([]),
    ))

    return RobustnessResult(
        models=model_results,
        truth=truth,
        sensor_locations=np.asarray(sensor_locations),
        nx=nx,
        ny=ny,
        placement=placement,
        num_sensors=num_sensors,
        lags=lags,
    )

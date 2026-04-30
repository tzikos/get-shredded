"""Load a saved SHRED checkpoint and regenerate reconstructions and plots.

This script expects a checkpoint produced by `scripts/run_cylinder_baseline.py`
after it has been updated to save model weights.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from get_shredded.baseline import QRPODBaseline
from get_shredded.data import build_sensor_windows, load_cylinder_data, qr_place
from get_shredded.experiment import RunResult
from get_shredded.model import SDN, SHRED, TimeSeriesDataset
from get_shredded.plotting import (
    animate_reconstructions,
    plot_per_snapshot_error,
    plot_reconstruction_panel,
    plot_training_curves,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a saved SHRED checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="models/shred_baseline.pt",
        help="Path to the checkpoint saved by run_cylinder_baseline.py",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override the output directory from the checkpoint config",
    )
    parser.add_argument(
        "--snapshot-indices",
        default=None,
        help="Comma-separated test snapshot indices to show in the static panel",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Animation frame rate override",
    )
    return parser.parse_args()


def _as_int_array(value: np.ndarray | list[int] | tuple[int, ...]) -> np.ndarray:
    return np.asarray(value, dtype=int)


def _load_checkpoint(checkpoint_path: Path) -> dict:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _rel_errors(pred: np.ndarray, truth: np.ndarray) -> tuple[float, np.ndarray]:
    per_snap_den = np.linalg.norm(truth, axis=1)
    per_snap_den = np.where(per_snap_den == 0, 1.0, per_snap_den)
    per_snap = np.linalg.norm(pred - truth, axis=1) / per_snap_den
    total = float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))
    return total, per_snap


def _build_result_from_checkpoint(
    checkpoint_path: Path,
    output_root: str | Path | None,
    fps_override: int | None,
) -> tuple[RunResult, Path, int]:
    ckpt = _load_checkpoint(checkpoint_path)
    if "shred_state_dict" not in ckpt or "sdn_state_dict" not in ckpt:
        raise RuntimeError(
            "Checkpoint does not contain model weights. Re-run scripts/run_cylinder_baseline.py "
            "after updating it to save state_dicts, then run this visualization script again."
        )

    cfg = OmegaConf.create(ckpt["config"])
    mat_path = Path(to_absolute_path(str(cfg.data.mat)))
    load_X, nx, ny = load_cylinder_data(mat_path)
    n, m = load_X.shape

    num_sensors = int(cfg.model.num_sensors)
    lags = int(cfg.model.lags)
    test_size = int(cfg.data.test_size)
    val_size = int(cfg.data.val_size)
    hidden_size = int(cfg.model.hidden_size)
    hidden_layers = int(cfg.model.hidden_layers)
    l1 = int(cfg.model.l1)
    l2 = int(cfg.model.l2)
    dropout = float(cfg.model.dropout)
    seed = int(cfg.seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    n_windows = n - lags
    if test_size + val_size >= n_windows:
        raise ValueError(f"test_size + val_size ({test_size + val_size}) >= n_windows ({n_windows})")
    train_end = n_windows - test_size - val_size
    val_end = n_windows - test_size
    train_indices = np.arange(0, train_end)
    valid_indices = np.arange(train_end, val_end)
    test_indices = np.arange(val_end, n_windows)

    sensor_locations = _as_int_array(ckpt["sensor_locations"])
    _, U_r = qr_place(load_X[train_indices].T, num_sensors)

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
    test_ds = TimeSeriesDataset(test_in, test_out)
    train_ds_sdn = TimeSeriesDataset(train_in[:, -1, :], train_out)
    valid_ds_sdn = TimeSeriesDataset(valid_in[:, -1, :], valid_out)
    test_ds_sdn = TimeSeriesDataset(test_in[:, -1, :], test_out)

    shred = SHRED(
        num_sensors,
        m,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        l1=l1,
        l2=l2,
        dropout=dropout,
    ).to(device)
    shred.load_state_dict(ckpt["shred_state_dict"])

    sdn = SDN(num_sensors, m, l1=l1, l2=l2, dropout=dropout).to(device)
    sdn.load_state_dict(ckpt["sdn_state_dict"])

    shred.eval()
    sdn.eval()
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

    shred_err, shred_err_per_snap = _rel_errors(shred_recon, truth)
    sdn_err, sdn_err_per_snap = _rel_errors(sdn_recon, truth)
    qrpod_err, qrpod_err_per_snap = _rel_errors(qrpod_recon, truth)

    result = RunResult(
        shred_recon=shred_recon,
        sdn_recon=sdn_recon,
        qrpod_recon=qrpod_recon,
        truth=truth,
        shred_err=shred_err,
        sdn_err=sdn_err,
        qrpod_err=qrpod_err,
        shred_err_per_snap=shred_err_per_snap,
        sdn_err_per_snap=sdn_err_per_snap,
        qrpod_err_per_snap=qrpod_err_per_snap,
        shred_val_history=np.asarray(ckpt.get("shred_val_history", [])),
        sdn_val_history=np.asarray(ckpt.get("sdn_val_history", [])),
        sensor_locations=sensor_locations,
        nx=nx,
        ny=ny,
        placement=str(cfg.model.placement),
        num_sensors=num_sensors,
        lags=lags,
        shred_state_dict=ckpt["shred_state_dict"],
        sdn_state_dict=ckpt["sdn_state_dict"],
    )

    outputs_root = Path(
        to_absolute_path(str(output_root if output_root is not None else cfg.outputs.root))
    )
    fps = int(fps_override if fps_override is not None else cfg.outputs.gif_fps)
    return result, outputs_root, fps


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(to_absolute_path(args.checkpoint))
    result, outputs_root, fps = _build_result_from_checkpoint(checkpoint_path, args.output_root, args.fps)

    recon_dir = outputs_root / "reconstructions"
    curves_dir = outputs_root / "curves"
    recon_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    if args.snapshot_indices:
        snapshot_indices = [int(item) for item in args.snapshot_indices.split(",") if item.strip()]
    else:
        n_test = result.truth.shape[0]
        snapshot_indices = sorted({0, n_test // 2, n_test - 1})

    plot_reconstruction_panel(result, snapshot_indices, recon_dir / "panel.png")
    animate_reconstructions(result, recon_dir / "comparison.gif", fps=fps)
    plot_per_snapshot_error(result, curves_dir / "per_snapshot_error.png")
    if result.shred_val_history.size and result.sdn_val_history.size:
        plot_training_curves(result, curves_dir / "training_curves.png")

    print(f"Saved visualization outputs under {outputs_root}/")


if __name__ == "__main__":
    main()
"""Train RobustSHRED using three-phase teacher-student procedure.

Usage:
  uv run python scripts/run_robust_shred.py
  uv run python scripts/run_robust_shred.py model.num_sensors=5 model.d_z=16
  uv run python scripts/run_robust_shred.py augmentation.type=gaussian
  uv run python scripts/run_robust_shred.py phase2.beta_kl=0.001
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from get_shredded.augmentation import make_batch_augmenter
from get_shredded.experiment import _build_scenario_noisy
from get_shredded.plotting import (
    plot_per_snapshot_error,
    plot_reconstruction_panel,
    plot_training_curves,
)
from get_shredded.robust_experiment import run_robust_experiment
from get_shredded.robust_model import RobustSHRED


def plot_trust_weights(
    robust_shred: RobustSHRED,
    test_in: torch.Tensor,
    test_in_noisy: torch.Tensor,
    sensor_locations: np.ndarray,
    num_sensors: int,
    save_path: Path,
) -> None:
    """Plot per-sensor trust weights under clean vs. corrupted inputs."""
    robust_shred.eval()
    with torch.no_grad():
        w_clean = robust_shred.get_trust_weights(test_in).cpu().numpy()       # (T, p)
        w_noisy = robust_shred.get_trust_weights(test_in_noisy).cpu().numpy()  # (T, p)

    T = w_clean.shape[0]
    uniform = 1.0 / num_sensors
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, weights, title in zip(
        axes,
        [w_clean, w_noisy],
        ["Trust weights — clean inputs", "Trust weights — corrupted inputs"],
    ):
        for i in range(num_sensors):
            color = colors[i % len(colors)]
            ax.plot(
                weights[:, i],
                label=f"Sensor {i} (loc {sensor_locations[i]})",
                color=color,
            )
        ax.axhline(uniform, color="black", linestyle="--", linewidth=0.8,
                   label=f"Uniform (1/{num_sensors})")
        ax.set_xlabel("Test timestep")
        ax.set_ylabel("Trust weight")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0, T - 1)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved trust weight plot → {save_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="robust_shred")
def main(cfg: DictConfig) -> None:
    result = run_robust_experiment(
        mat_path=Path(to_absolute_path(cfg.data.mat)),
        num_sensors=int(cfg.model.num_sensors),
        lags=int(cfg.model.lags),
        placement=str(cfg.model.placement),
        test_size=int(cfg.data.test_size),
        val_size=int(cfg.data.val_size),
        hidden_size=int(cfg.model.hidden_size),
        hidden_layers=int(cfg.model.hidden_layers),
        l1=int(cfg.model.l1),
        l2=int(cfg.model.l2),
        dropout=float(cfg.model.dropout),
        d_z=int(cfg.model.d_z),
        phase1_epochs=int(cfg.phase1.epochs),
        phase1_lr=float(cfg.phase1.lr),
        phase1_patience=int(cfg.phase1.patience),
        phase2_epochs=int(cfg.phase2.epochs),
        phase2_lr=float(cfg.phase2.lr),
        phase2_patience=int(cfg.phase2.patience),
        beta_kl=float(cfg.phase2.beta_kl),
        lambda_detect=float(cfg.phase2.lambda_detect),
        phase3_epochs=int(cfg.phase3.epochs),
        phase3_lr=float(cfg.phase3.lr),
        phase3_patience=int(cfg.phase3.patience),
        batch_size=int(cfg.train.batch_size),
        augmentation_type=str(cfg.augmentation.type),
        gaussian_std=float(cfg.augmentation.gaussian_std),
        dropout_fill=float(cfg.augmentation.dropout_fill),
        seed=int(cfg.seed),
        verbose=True,
    )

    print(f"\nNum sensors: {result.num_sensors} | Placement: {result.placement} | Lags: {result.lags}")
    print(f"RobustSHRED   relative L2 error: {result.shred_err:.6f}")
    print(f"Vanilla SHRED relative L2 error: {result.sdn_err:.6f}")
    print(f"QR/POD        relative L2 error: {result.qrpod_err:.6f}")

    outputs_root = Path(to_absolute_path(cfg.outputs.root))
    recon_dir = outputs_root / "reconstructions"
    curves_dir = outputs_root / "curves"

    n_test = result.truth.shape[0]
    snap_indices = sorted({0, n_test // 2, n_test - 1})
    plot_reconstruction_panel(result, snap_indices, recon_dir / "panel.png")
    plot_per_snapshot_error(result, curves_dir / "per_snapshot_error.png")
    plot_training_curves(result, curves_dir / "training_curves.png")
    print(f"Saved plots under {outputs_root}/")

    # Reconstruct robust_shred from the saved state dict for trust weight analysis
    num_sensors = result.num_sensors
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    output_size = result.truth.shape[1]
    robust_shred_eval = RobustSHRED(
        num_sensors=num_sensors,
        output_size=output_size,
        hidden_size=int(cfg.model.hidden_size),
        hidden_layers=int(cfg.model.hidden_layers),
        l1=int(cfg.model.l1),
        l2=int(cfg.model.l2),
        dropout=float(cfg.model.dropout),
        d_z=int(cfg.model.d_z),
    ).to(device)
    robust_shred_eval.load_state_dict(
        {k: v.to(device) for k, v in result.shred_state_dict.items()}
    )

    from get_shredded.data import build_sensor_windows, load_cylinder_data, qr_place
    from sklearn.preprocessing import MinMaxScaler

    load_X, _, _ = load_cylinder_data(Path(to_absolute_path(cfg.data.mat)))
    n, m = load_X.shape
    lags = int(cfg.model.lags)
    test_size = int(cfg.data.test_size)
    val_size = int(cfg.data.val_size)
    n_windows = n - lags
    train_end = n_windows - test_size - val_size
    train_indices = np.arange(0, train_end)
    test_indices = np.arange(n_windows - test_size, n_windows)

    if result.placement == "QR":
        sensor_locations, _ = qr_place(load_X[train_indices].T, num_sensors)
    else:
        sensor_locations = result.sensor_locations

    sc = MinMaxScaler().fit(load_X[train_indices])
    transformed_X = sc.transform(load_X).astype(np.float32)
    all_data_in = build_sensor_windows(transformed_X, sensor_locations, lags)
    test_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32, device=device)

    # Build a noisy test set using dropout scenario
    rng = np.random.default_rng(int(cfg.seed) + 999)
    test_noisy_np = _build_scenario_noisy(
        all_data_in[test_indices], "dropout", num_sensors,
        gaussian_std=float(cfg.augmentation.gaussian_std),
        dropout_fill=float(cfg.augmentation.dropout_fill),
        rng=rng,
    )
    test_in_noisy = torch.tensor(test_noisy_np, dtype=torch.float32, device=device)

    plot_trust_weights(
        robust_shred_eval,
        test_in,
        test_in_noisy,
        result.sensor_locations,
        num_sensors,
        outputs_root / "trust_weights.png",
    )

    # Save checkpoint
    ckpt_dir = outputs_root
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "robust_shred.pt"
    torch.save(
        {
            "sensor_locations": result.sensor_locations,
            "robust_shred_state_dict": result.shred_state_dict,
            "vanilla_shred_state_dict": result.sdn_state_dict,
            "shred_val_history": result.shred_val_history,   # phase3
            "sdn_val_history": result.sdn_val_history,       # phase1
            "config": OmegaConf.to_container(cfg, resolve=True),
            "metrics": {
                "robust_shred_err": result.shred_err,
                "vanilla_shred_err": result.sdn_err,
                "qrpod_err": result.qrpod_err,
            },
        },
        ckpt_path,
    )
    print(f"Checkpoint saved → {ckpt_path}")


if __name__ == "__main__":
    main()

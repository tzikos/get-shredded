"""Train SHRED + SDN, compute QR/POD baseline, save metrics + plots + GIF."""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from get_shredded.experiment import run_experiment
from get_shredded.plotting import (
    animate_reconstructions,
    plot_per_snapshot_error,
    plot_reconstruction_panel,
    plot_training_curves,
)


@hydra.main(version_base=None, config_path="../configs", config_name="cylinder_baseline")
def main(cfg: DictConfig) -> None:
    result = run_experiment(
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
        epochs=int(cfg.train.epochs),
        batch_size=int(cfg.train.batch_size),
        lr=float(cfg.train.lr),
        patience=int(cfg.train.patience),
        seed=int(cfg.seed),
        verbose=True,
    )

    print(f"\nNum sensors: {result.num_sensors} | Placement: {result.placement} | Lags: {result.lags}")
    print(f"SHRED   relative L2 error: {result.shred_err:.6f}")
    print(f"SDN     relative L2 error: {result.sdn_err:.6f}")
    print(f"QR/POD  relative L2 error: {result.qrpod_err:.6f}")

    outputs_root = Path(to_absolute_path(cfg.outputs.root))
    recon_dir = outputs_root / "reconstructions"
    curves_dir = outputs_root / "curves"

    n_test = result.truth.shape[0]
    snap_indices = sorted({0, n_test // 2, n_test - 1})
    plot_reconstruction_panel(result, snap_indices, recon_dir / "panel.png")
    animate_reconstructions(result, recon_dir / "comparison.gif", fps=int(cfg.outputs.gif_fps))
    plot_per_snapshot_error(result, curves_dir / "per_snapshot_error.png")
    plot_training_curves(result, curves_dir / "training_curves.png")
    print(f"Saved plots under {outputs_root}/")

    if cfg.checkpoint.enabled:
        ckpt_dir = Path(to_absolute_path(cfg.checkpoint.dir))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = ckpt_dir / str(cfg.checkpoint.name)
        torch.save(
            {
                "sensor_locations": result.sensor_locations,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "metrics": {
                    "shred_err": result.shred_err,
                    "sdn_err": result.sdn_err,
                    "qrpod_err": result.qrpod_err,
                },
            },
            ckpt,
        )
        print(f"Saved metrics: {ckpt}")


if __name__ == "__main__":
    main()

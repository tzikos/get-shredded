"""Train SHRED + SDN with four augmentation strategies and compare robustness.

Models: SHRED-clean, SHRED-gaussian, SHRED-dropout, SHRED-hybrid,
        SDN-clean,   SDN-gaussian,   SDN-dropout,   SDN-hybrid,  QR-POD.

Each model is evaluated on both clean and noisy test sensor inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from get_shredded.experiment import run_robustness_comparison
from get_shredded.plotting import (
    plot_robustness_bar,
    plot_robustness_panel,
    plot_robustness_per_snapshot,
)


@hydra.main(version_base=None, config_path="../configs", config_name="robustness_comparison")
def main(cfg: DictConfig) -> None:
    result = run_robustness_comparison(
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
        gaussian_std=float(cfg.augmentation.gaussian_std),
        test_noise_std=float(cfg.augmentation.test_noise_std),
        dropout_fill=float(cfg.augmentation.dropout_fill),
        verbose=True,
    )

    print(f"\n{'Model':<20} {'Clean err':>12} {'Noisy err':>12}")
    print("-" * 46)
    for m in result.models:
        print(f"{m.name:<20} {m.err_clean:>12.6f} {m.err_noisy:>12.6f}")

    out_dir = Path(to_absolute_path(cfg.outputs.root))
    out_dir.mkdir(parents=True, exist_ok=True)

    n_test = result.truth_clean.shape[0]
    snap_indices = sorted({0, n_test // 2, n_test - 1})

    plot_robustness_bar(result, out_dir / "robustness_bar.png")
    plot_robustness_per_snapshot(result, "clean", out_dir / "per_snapshot_clean.png")
    plot_robustness_per_snapshot(result, "noisy", out_dir / "per_snapshot_noisy.png")
    plot_robustness_panel(result, "clean", snap_indices, out_dir / "panel_clean.png")
    plot_robustness_panel(result, "noisy", snap_indices, out_dir / "panel_noisy.png")

    # Save raw errors for further analysis
    np.savez(
        out_dir / "results.npz",
        model_names=np.array([m.name for m in result.models]),
        errs_clean=np.array([m.err_clean for m in result.models]),
        errs_noisy=np.array([m.err_noisy for m in result.models]),
    )

    print(f"\nSaved plots and results to {out_dir}/")


if __name__ == "__main__":
    main()

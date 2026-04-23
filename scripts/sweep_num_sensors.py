"""Sweep `num_sensors` and plot test relative L2 error for SHRED, SDN, QR/POD.

Mirrors Fig 2B / 3B / 4B of Williams, Zahn, Kutz (2024).
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from get_shredded.experiment import run_experiment
from get_shredded.plotting import plot_sweep_error_vs_sensors


@hydra.main(version_base=None, config_path="../configs", config_name="sweep_num_sensors")
def main(cfg: DictConfig) -> None:
    sensor_counts = list(cfg.sweep.sensor_counts)
    seeds = list(cfg.sweep.seeds)
    placement = str(cfg.model.placement)

    shred_med, sdn_med, qrpod_med = [], [], []
    raw = {"sensor_counts": sensor_counts, "seeds": seeds,
           "shred": [], "sdn": [], "qrpod": []}

    outer = tqdm(sensor_counts, desc="sensor sweep")
    for n_s in outer:
        per_seed = {"shred": [], "sdn": [], "qrpod": []}
        inner = tqdm(seeds, desc=f"  n_sensors={n_s}", leave=False)
        for seed in inner:
            r = run_experiment(
                mat_path=Path(to_absolute_path(cfg.data.mat)),
                num_sensors=int(n_s),
                lags=int(cfg.model.lags),
                placement=placement,
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
                seed=int(seed),
                verbose=True,
            )
            per_seed["shred"].append(r.shred_err)
            per_seed["sdn"].append(r.sdn_err)
            per_seed["qrpod"].append(r.qrpod_err)
            inner.set_postfix(shred=f"{r.shred_err:.3f}", sdn=f"{r.sdn_err:.3f}",
                              qrpod=f"{r.qrpod_err:.3f}")
        outer.set_postfix(
            shred_med=f"{float(np.median(per_seed['shred'])):.3f}",
            sdn_med=f"{float(np.median(per_seed['sdn'])):.3f}",
            qrpod_med=f"{float(np.median(per_seed['qrpod'])):.3f}",
        )

        shred_med.append(float(np.median(per_seed["shred"])))
        sdn_med.append(float(np.median(per_seed["sdn"])))
        qrpod_med.append(float(np.median(per_seed["qrpod"])))
        raw["shred"].append(per_seed["shred"])
        raw["sdn"].append(per_seed["sdn"])
        raw["qrpod"].append(per_seed["qrpod"])

    out_dir = Path(to_absolute_path(cfg.outputs.root)) / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_sweep_error_vs_sensors(
        sensor_counts=sensor_counts,
        shred_errs=shred_med,
        sdn_errs=sdn_med,
        qrpod_errs=qrpod_med,
        save_path=out_dir / "error_vs_num_sensors.png",
        placement=placement,
    )

    np.savez(
        out_dir / "sweep_results.npz",
        sensor_counts=np.asarray(sensor_counts),
        seeds=np.asarray(seeds),
        shred=np.asarray(raw["shred"]),
        sdn=np.asarray(raw["sdn"]),
        qrpod=np.asarray(raw["qrpod"]),
    )
    print(f"\nSaved sweep results to {out_dir}")


if __name__ == "__main__":
    main()

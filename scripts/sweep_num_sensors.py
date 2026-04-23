"""Sweep `num_sensors` × {QR, random} placement and plot test relative L2 error.

Mirrors Fig 2B / 3B / 4B of Williams, Zahn, Kutz (2024):
- SHRED, SHRED-QR, SDN, SDN-QR, QR/POD (linear; only QR placement is well-conditioned).
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
    placements = list(cfg.sweep.placements)

    # raw[(method, placement)] -> list (over sensor_counts) of lists (over seeds)
    raw: dict[tuple[str, str], list[list[float]]] = {
        (method, p): [] for method in ("shred", "sdn", "qrpod") for p in placements
    }

    outer = tqdm(sensor_counts, desc="sensor sweep")
    for n_s in outer:
        per_count: dict[tuple[str, str], list[float]] = {k: [] for k in raw}
        for placement in placements:
            inner = tqdm(seeds, desc=f"  n_sensors={n_s} {placement}", leave=False)
            for seed in inner:
                r = run_experiment(
                    mat_path=Path(to_absolute_path(cfg.data.mat)),
                    num_sensors=int(n_s),
                    lags=int(cfg.model.lags),
                    placement=str(placement),
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
                per_count[("shred", placement)].append(r.shred_err)
                per_count[("sdn", placement)].append(r.sdn_err)
                per_count[("qrpod", placement)].append(r.qrpod_err)
                inner.set_postfix(shred=f"{r.shred_err:.3f}", sdn=f"{r.sdn_err:.3f}",
                                  qrpod=f"{r.qrpod_err:.3f}")

        for key, vals in per_count.items():
            raw[key].append(vals)

        outer.set_postfix(
            shred_QR=f"{float(np.median(per_count[('shred', 'QR')])):.3f}"
                if ('shred', 'QR') in per_count and per_count[('shred', 'QR')] else "—",
            shred_R=f"{float(np.median(per_count[('shred', 'random')])):.3f}"
                if ('shred', 'random') in per_count and per_count[('shred', 'random')] else "—",
        )

    # Build plot series. QR/POD only shown for QR placement (random is ill-conditioned).
    series: dict[str, list[float]] = {}
    for placement in placements:
        suffix = f" ({placement})"
        if ("shred", placement) in raw:
            series[f"SHRED{suffix}"] = [float(np.median(v)) for v in raw[("shred", placement)]]
        if ("sdn", placement) in raw:
            series[f"SDN{suffix}"] = [float(np.median(v)) for v in raw[("sdn", placement)]]
    if "QR" in placements:
        series["QR/POD"] = [float(np.median(v)) for v in raw[("qrpod", "QR")]]

    out_dir = Path(to_absolute_path(cfg.outputs.root)) / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_sweep_error_vs_sensors(
        sensor_counts=sensor_counts,
        series=series,
        save_path=out_dir / "error_vs_num_sensors.png",
    )

    np.savez(
        out_dir / "sweep_results.npz",
        sensor_counts=np.asarray(sensor_counts),
        seeds=np.asarray(seeds),
        placements=np.asarray(placements),
        **{f"{m}_{p}": np.asarray(raw[(m, p)]) for (m, p) in raw},
    )
    print(f"\nSaved sweep results to {out_dir}")


if __name__ == "__main__":
    main()

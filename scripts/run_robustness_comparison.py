"""Robustness sweep: train SHRED + SDN with four augmentation strategies
across multiple sensor counts and placements.

Per-run outputs (one subdirectory per sensor_count × placement):
  sensors_{N}_{placement}/
    robustness_bar.png
    table.md / table.tex
    [panel_{scenario}.png, per_snapshot_{scenario}.png,    (if per_run_plots=true)
     robustness_comparison.gif]

Sweep-level outputs:
  sweep.png              error vs num_sensors grid (scenarios × placements)
  sweep_results.npz      raw error arrays

Override examples:
  uv run python scripts/run_robustness_comparison.py sweep.sensor_counts=[3,5,8,12]
  uv run python scripts/run_robustness_comparison.py sweep.placements=[QR]
  uv run python scripts/run_robustness_comparison.py sweep.per_run_plots=true
  uv run python scripts/run_robustness_comparison.py train.epochs=200 train.patience=3
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

from get_shredded.experiment import SCENARIOS, RobustnessResult, run_robustness_comparison
from get_shredded.plotting import (
    animate_robustness_comparison,
    plot_robustness_bar_sweep,
    plot_robustness_panel,
    plot_robustness_per_snapshot,
    plot_robustness_sweep,
    save_robustness_table_sweep,
)


def _print_table(result: RobustnessResult, num_sensors: int, placement: str) -> None:
    col_w = 12
    print(f"\n── {num_sensors} sensors · {placement} placement ──")
    header = f"{'Model':<20}" + f"{'Clean':>{col_w}}" + \
             "".join(f"{s.capitalize():>{col_w}}" for s in SCENARIOS)
    print(header)
    print("-" * len(header))
    for m in result.models:
        row = f"{m.name:<20}{m.err_clean:>{col_w}.5f}"
        row += "".join(f"{m.err_noisy[s]:>{col_w}.5f}" for s in SCENARIOS)
        print(row)


def _run_one(
    cfg: DictConfig,
    num_sensors: int,
    placement: str,
    out_dir: Path,
) -> RobustnessResult:
    """Train all model variants for one (num_sensors, placement) combination."""
    result = run_robustness_comparison(
        mat_path=Path(to_absolute_path(cfg.data.mat)),
        num_sensors=num_sensors,
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
        seed=int(cfg.seed),
        gaussian_std=float(cfg.augmentation.gaussian_std),
        test_noise_std=float(cfg.augmentation.test_noise_std),
        dropout_fill=float(cfg.augmentation.dropout_fill),
        verbose=True,
    )

    _print_table(result, num_sensors, placement)

    if cfg.sweep.per_run_plots:
        out_dir.mkdir(parents=True, exist_ok=True)
        n_test = result.truth.shape[0]
        snap_indices = sorted({0, n_test // 2, n_test - 1})
        for scenario in ["clean"] + SCENARIOS:
            plot_robustness_panel(result, scenario, snap_indices,
                                  out_dir / f"panel_{scenario}.png")
            plot_robustness_per_snapshot(result, scenario,
                                         out_dir / f"per_snapshot_{scenario}.png")
        animate_robustness_comparison(result, out_dir / "robustness_comparison.gif", fps=4)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "results.npz",
        model_names=np.array([m.name for m in result.models]),
        errs_clean=np.array([m.err_clean for m in result.models]),
        **{f"errs_{s}": np.array([m.err_noisy[s] for m in result.models])
           for s in SCENARIOS},
    )
    print(f"  → saved to {out_dir}/")
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="robustness_comparison")
def main(cfg: DictConfig) -> None:
    sensor_counts: list[int] = [int(n) for n in cfg.sweep.sensor_counts]
    placements: list[str]    = [str(p) for p in cfg.sweep.placements]
    base_dir = Path(to_absolute_path(cfg.outputs.root))

    total = len(sensor_counts) * len(placements)
    print(f"\nRobustness sweep: {sensor_counts} sensors × {placements} placements "
          f"= {total} runs, each training {len(SCENARIOS) * 2 + 1} models.\n")

    all_results: dict[tuple[int, str], RobustnessResult] = {}

    for run_idx, num_sensors in enumerate(sensor_counts):
        for placement in placements:
            tag = f"sensors_{num_sensors}_{placement}"
            print(f"\n[{run_idx * len(placements) + placements.index(placement) + 1}/{total}] "
                  f"{tag}")
            out_dir = base_dir / tag
            result = _run_one(cfg, num_sensors, placement, out_dir)
            all_results[(num_sensors, placement)] = result

    # --- Consolidated bar charts (one per placement) + table ---
    print("\nGenerating consolidated bar charts and table…")
    plot_robustness_bar_sweep(all_results, sensor_counts, placements, base_dir)
    save_robustness_table_sweep(all_results, sensor_counts, placements,
                                base_dir / "table")

    # --- Sweep summary line plot ---
    print("\nGenerating sweep plot…")
    plot_robustness_sweep(all_results, sensor_counts, placements,
                          base_dir / "sweep.png")

    # --- Sweep-level NPZ ---
    model_names = [m.name for m in next(iter(all_results.values())).models]
    conditions  = ["clean"] + SCENARIOS
    errs = np.full(
        (len(sensor_counts), len(placements), len(model_names), len(conditions)),
        np.nan,
    )
    for si, n in enumerate(sensor_counts):
        for pi, p in enumerate(placements):
            res = all_results.get((n, p))
            if res is None:
                continue
            by_name = {m.name: m for m in res.models}
            for mi, mname in enumerate(model_names):
                m = by_name.get(mname)
                if m is None:
                    continue
                errs[si, pi, mi, 0] = m.err_clean
                for ci, s in enumerate(SCENARIOS, start=1):
                    errs[si, pi, mi, ci] = m.err_noisy[s]

    np.savez(
        base_dir / "sweep_results.npz",
        sensor_counts=np.array(sensor_counts),
        placements=np.array(placements),
        model_names=np.array(model_names),
        conditions=np.array(conditions),
        errs=errs,  # (n_sensors, n_placements, n_models, n_conditions)
    )

    print(f"\nAll outputs saved to {base_dir}/")
    print("  " + "  ".join(f"robustness_bar_{p}.png" for p in placements) + "  — bar charts per placement")
    print(f"  table.md / table.tex   — consolidated error tables")
    print(f"  sweep.png              — error vs sensors line grid")
    print(f"  sweep_results.npz      — shape {errs.shape} (sensors, placements, models, conditions)")
    if cfg.sweep.per_run_plots:
        for n in sensor_counts:
            for p in placements:
                print(f"  sensors_{n}_{p}/       — panels + GIF + per-snapshot plots")


if __name__ == "__main__":
    main()

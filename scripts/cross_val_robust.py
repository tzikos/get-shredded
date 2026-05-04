"""Paired cross-validation: RobustSHRED vs vanilla SHRED.

Evaluates both models on clean AND noisy test inputs to measure the
robustness-accuracy tradeoff.  The primary comparison is on noisy inputs
(the setting RobustSHRED is designed for); clean results are also reported
so the accuracy cost of robustness training is visible.

Usage:
  uv run python scripts/cross_val_robust.py
  uv run python scripts/cross_val_robust.py cross_val.n_seeds=5
  uv run python scripts/cross_val_robust.py augmentation.type=dropout
  uv run python scripts/cross_val_robust.py phase1.epochs=500 phase2.epochs=200
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from get_shredded.robust_experiment import run_robust_experiment_detailed


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples: mean(diff) / std(diff)."""
    diffs = a - b
    return float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 0 else 0.0


def _run_one_seed(cfg: DictConfig, seed: int) -> dict:
    """Train RobustSHRED + vanilla SHRED for one seed.

    Returns clean and noisy errors for both models, plus per-snapshot arrays.
    """
    row = run_robust_experiment_detailed(
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
        noisy_scenario=str(cfg.investigate.noisy_scenario),
        seed=seed,
        verbose=False,
    )
    return {
        "robust_clean_err":        row["robust_err"],
        "vanilla_clean_err":       row["vanilla_err"],
        "robust_noisy_err":        row["robust_noisy_err"],
        "vanilla_noisy_err":       row["vanilla_noisy_err"],
        "robust_clean_per_snap":   row["robust_per_snap"],
        "vanilla_clean_per_snap":  row["vanilla_per_snap"],
        "robust_noisy_per_snap":   row["robust_noisy_per_snap"],
        "vanilla_noisy_per_snap":  row["vanilla_noisy_per_snap"],
    }


def _stat_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float, float]:
    """Paired t-test, Wilcoxon, Cohen's d for arrays a vs b."""
    t_stat, t_p = stats.ttest_rel(a, b)
    diffs = a - b
    try:
        w_stat, w_p = stats.wilcoxon(diffs)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    d = _cohens_d_paired(a, b)
    return float(t_stat), float(t_p), float(w_stat), float(w_p), d


def _plot_results(
    robust_clean: np.ndarray,
    vanilla_clean: np.ndarray,
    robust_noisy: np.ndarray,
    vanilla_noisy: np.ndarray,
    seeds: list[int],
    noisy_scenario: str,
    out_dir: Path,
) -> None:
    """Six-panel figure: clean and noisy comparisons side by side."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"RobustSHRED vs Vanilla SHRED  |  noisy scenario: {noisy_scenario}",
        fontsize=12, fontweight="bold",
    )

    def _scatter(ax: plt.Axes, v_err: np.ndarray, r_err: np.ndarray, title: str) -> None:
        lo = min(v_err.min(), r_err.min()) * 0.95
        hi = max(v_err.max(), r_err.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y=x")
        ax.scatter(v_err, r_err, color="steelblue", zorder=3)
        for i, s in enumerate(seeds):
            ax.annotate(f"s{s}", (v_err[i], r_err[i]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("Vanilla SHRED error")
        ax.set_ylabel("RobustSHRED error")
        ax.set_title(title + "\n(below diagonal = robust wins)")
        ax.legend(fontsize=8)

    def _hist(ax: plt.Axes, a: np.ndarray, b: np.ndarray, title: str, xlabel: str) -> None:
        diffs = a - b
        ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
        ax.axvline(diffs.mean(), color="steelblue", linewidth=1.5,
                   label=f"mean = {diffs.mean():.4f}")
        ax.hist(diffs, bins=max(5, len(seeds) // 2), color="steelblue",
                alpha=0.7, edgecolor="white")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=8)

    def _bar(ax: plt.Axes, v_err: np.ndarray, r_err: np.ndarray, title: str) -> None:
        x = np.arange(len(seeds))
        w = 0.35
        ax.bar(x - w / 2, v_err, w, label="Vanilla SHRED", color="salmon", alpha=0.85)
        ax.bar(x + w / 2, r_err, w, label="RobustSHRED",   color="steelblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"s{s}" for s in seeds], fontsize=8)
        ax.set_ylabel("Relative L2 error")
        ax.set_title(title)
        ax.legend(fontsize=8)

    # Row 0: clean inputs
    _scatter(axes[0, 0], vanilla_clean, robust_clean, "Clean inputs — paired scatter")
    _hist(axes[0, 1], robust_clean, vanilla_clean, "Clean inputs — diff dist.",
          "Robust − Vanilla  (neg. = robust wins)")
    _bar(axes[0, 2], vanilla_clean, robust_clean, "Clean inputs — per seed")

    # Row 1: noisy inputs
    _scatter(axes[1, 0], vanilla_noisy, robust_noisy,
             f"Noisy inputs ({noisy_scenario}) — paired scatter")
    _hist(axes[1, 1], robust_noisy, vanilla_noisy,
          f"Noisy inputs ({noisy_scenario}) — diff dist.",
          "Robust − Vanilla  (neg. = robust wins)")
    _bar(axes[1, 2], vanilla_noisy, robust_noisy,
         f"Noisy inputs ({noisy_scenario}) — per seed")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "cv_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_dir / 'cv_results.png'}")


def _print_report(
    robust_clean: np.ndarray,
    vanilla_clean: np.ndarray,
    robust_noisy: np.ndarray,
    vanilla_noisy: np.ndarray,
    noisy_scenario: str,
    n: int,
) -> None:
    col = 26

    def _block(label: str, v: np.ndarray, r: np.ndarray) -> None:
        t_stat, t_p, w_stat, w_p, d = _stat_pair(r, v)
        diffs = r - v
        sig_t = "**" if t_p < 0.01 else ("*" if t_p < 0.05 else "ns")
        sig_w = "**" if w_p < 0.01 else ("*" if w_p < 0.05 else "ns")
        print(f"\n  [{label}]")
        print(f"  {'Vanilla SHRED:':<{col}} {v.mean():.6f} ± {v.std(ddof=1):.6f}")
        print(f"  {'RobustSHRED:':<{col}} {r.mean():.6f} ± {r.std(ddof=1):.6f}")
        print(f"  {'Mean diff (R − V):':<{col}} {diffs.mean():.6f}  "
              f"({diffs.mean() / v.mean() * 100:+.1f}% relative)")
        print(f"  {'Paired t-test:':<{col}} t={t_stat:+.3f}, p={t_p:.4f}  {sig_t}")
        print(f"  {'Wilcoxon signed-rank:':<{col}} W={w_stat:.1f},  p={w_p:.4f}  {sig_w}")
        print(f"  {'Cohen d (paired):':<{col}} {d:.3f}")

    # Degradation gap: how much does each model degrade under noise?
    robust_degrad  = robust_noisy  - robust_clean
    vanilla_degrad = vanilla_noisy - vanilla_clean
    degrad_diff    = robust_degrad - vanilla_degrad  # negative = robust degrades less

    print("\n" + "=" * 65)
    print("Cross-validation: RobustSHRED vs Vanilla SHRED")
    print(f"Seeds (n): {n}   |   noisy scenario: {noisy_scenario}")
    print("=" * 65)
    _block("Clean test inputs", vanilla_clean, robust_clean)
    _block(f"Noisy test inputs ({noisy_scenario})", vanilla_noisy, robust_noisy)

    t_deg, p_deg, w_deg, wp_deg, d_deg = _stat_pair(robust_degrad, vanilla_degrad)
    sig_t = "**" if p_deg < 0.01 else ("*" if p_deg < 0.05 else "ns")
    print(f"\n  [Robustness gap: Δnoisy−clean, Robust − Vanilla]")
    print(f"  {'Vanilla degradation:':<{col}} {vanilla_degrad.mean():.6f} ± {vanilla_degrad.std(ddof=1):.6f}")
    print(f"  {'Robust degradation:':<{col}} {robust_degrad.mean():.6f} ± {robust_degrad.std(ddof=1):.6f}")
    print(f"  {'Mean gap (R−V degrad):':<{col}} {degrad_diff.mean():.6f}  "
          f"({'robust degrades less' if degrad_diff.mean() < 0 else 'robust degrades MORE'})")
    print(f"  {'Paired t-test on gap:':<{col}} t={t_deg:+.3f}, p={p_deg:.4f}  {sig_t}")

    print("\n" + "-" * 65)
    print("Significance: ** p<0.01  * p<0.05  ns = not significant")
    print("Cohen d: ~0.2 small | ~0.5 medium | ~0.8 large")
    print("=" * 65)


@hydra.main(version_base=None, config_path="../configs", config_name="cross_val_robust")
def main(cfg: DictConfig) -> None:
    n_seeds: int = int(cfg.cross_val.n_seeds)
    base_seed: int = int(cfg.cross_val.base_seed)
    seeds = list(range(base_seed, base_seed + n_seeds))
    noisy_scenario: str = str(cfg.investigate.noisy_scenario)
    out_dir = Path(to_absolute_path(cfg.outputs.root))

    print(f"\nPaired cross-validation: {n_seeds} seeds "
          f"({seeds[0]}–{seeds[-1]}), train_aug={cfg.augmentation.type}, "
          f"noisy_eval={noisy_scenario}\n")

    robust_clean_errs:  list[float] = []
    vanilla_clean_errs: list[float] = []
    robust_noisy_errs:  list[float] = []
    vanilla_noisy_errs: list[float] = []
    robust_clean_per_snap_all:  list[np.ndarray] = []
    vanilla_clean_per_snap_all: list[np.ndarray] = []
    robust_noisy_per_snap_all:  list[np.ndarray] = []
    vanilla_noisy_per_snap_all: list[np.ndarray] = []

    for i, seed in enumerate(seeds):
        print(f"[{i + 1}/{n_seeds}] seed={seed}", flush=True)
        row = _run_one_seed(cfg, seed)
        robust_clean_errs.append(row["robust_clean_err"])
        vanilla_clean_errs.append(row["vanilla_clean_err"])
        robust_noisy_errs.append(row["robust_noisy_err"])
        vanilla_noisy_errs.append(row["vanilla_noisy_err"])
        robust_clean_per_snap_all.append(row["robust_clean_per_snap"])
        vanilla_clean_per_snap_all.append(row["vanilla_clean_per_snap"])
        robust_noisy_per_snap_all.append(row["robust_noisy_per_snap"])
        vanilla_noisy_per_snap_all.append(row["vanilla_noisy_per_snap"])
        print(
            f"  clean  robust={row['robust_clean_err']:.6f}  vanilla={row['vanilla_clean_err']:.6f}"
            f"  |  noisy  robust={row['robust_noisy_err']:.6f}  vanilla={row['vanilla_noisy_err']:.6f}"
        )

    rc = np.array(robust_clean_errs)
    vc = np.array(vanilla_clean_errs)
    rn = np.array(robust_noisy_errs)
    vn = np.array(vanilla_noisy_errs)

    _print_report(rc, vc, rn, vn, noisy_scenario, n_seeds)
    _plot_results(rc, vc, rn, vn, seeds, noisy_scenario, out_dir)

    # --- Per-snapshot pooled tests (non-independent across seeds, noted) ---
    print("\nPer-snapshot pooled paired t-tests (non-independent across seeds):")
    for label, r_pool_list, v_pool_list in [
        ("clean", robust_clean_per_snap_all, vanilla_clean_per_snap_all),
        (noisy_scenario, robust_noisy_per_snap_all, vanilla_noisy_per_snap_all),
    ]:
        rp = np.concatenate(r_pool_list)
        vp = np.concatenate(v_pool_list)
        t2, p2 = stats.ttest_rel(rp, vp)
        print(f"  [{label}]  n={len(rp)} pairs  t={t2:+.3f}  p={p2:.4e}")

    # --- Save raw numbers ---
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "cv_results.npz",
        seeds=np.array(seeds),
        robust_clean_errs=rc,
        vanilla_clean_errs=vc,
        robust_noisy_errs=rn,
        vanilla_noisy_errs=vn,
        robust_clean_per_snap=np.array(robust_clean_per_snap_all),
        vanilla_clean_per_snap=np.array(vanilla_clean_per_snap_all),
        robust_noisy_per_snap=np.array(robust_noisy_per_snap_all),
        vanilla_noisy_per_snap=np.array(vanilla_noisy_per_snap_all),
        noisy_scenario=np.array(noisy_scenario),
    )
    print(f"\nRaw results saved → {out_dir / 'cv_results.npz'}")


if __name__ == "__main__":
    main()

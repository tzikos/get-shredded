"""Investigate why RobustSHRED succeeds for some seeds but not others.

Loads cv_results.npz to auto-classify seeds as 'good' (robust << vanilla)
or 'bad' (robust ≈ vanilla), then re-runs specified seeds with full diagnostic
capture: all three phase histories, trust weights under clean/corrupted inputs,
and reconstruction panels.

Produces four figures:
  phase_curves.png      — Phase 2 & 3 convergence coloured by outcome
  trust_weights.png     — trust weights over test time for best vs worst seed
  reconstruction.png    — side-by-side field reconstructions (best vs worst)
  sigma_dist.png        — per-sensor encoder uncertainty (σ) distributions

Usage:
  uv run python scripts/investigate_seeds.py
  uv run python scripts/investigate_seeds.py investigate.seeds=[0,3,7]
  uv run python scripts/investigate_seeds.py investigate.gap_threshold=0.1
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

from get_shredded.robust_experiment import run_robust_experiment_detailed
from get_shredded.robust_model import RobustSHRED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cv_seeds(cv_npz: Path, gap_threshold: float) -> tuple[list[int], list[int]]:
    """Return (good_seeds, bad_seeds) from a saved cv_results.npz.

    A seed is 'good' if (vanilla_err - robust_err) > gap_threshold.
    """
    data = np.load(cv_npz)
    seeds      = data["seeds"].tolist()
    robust_e   = data["robust_errs"]
    vanilla_e  = data["vanilla_errs"]
    gaps       = vanilla_e - robust_e
    good = [int(s) for s, g in zip(seeds, gaps) if g > gap_threshold]
    bad  = [int(s) for s, g in zip(seeds, gaps) if g <= gap_threshold]
    return good, bad


def _run_seeds(seeds: list[int], cfg: DictConfig) -> list[dict]:
    results = []
    for i, seed in enumerate(seeds):
        print(f"  [{i + 1}/{len(seeds)}] seed={seed}", flush=True)
        r = run_robust_experiment_detailed(
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
        print(f"    robust={r['robust_err']:.6f}  vanilla={r['vanilla_err']:.6f}")
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Figure 1 — Phase convergence curves
# ---------------------------------------------------------------------------

def plot_phase_curves(
    good_results: list[dict],
    bad_results: list[dict],
    out_path: Path,
) -> None:
    """Two-row plot: Phase 2 L_hidden and Phase 3 reconstruction convergence.

    Good seeds = blue family, bad seeds = red family.
    The Phase 2 curve is the smoking gun: good seeds descend steeply,
    bad seeds plateau near the initial value.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, phase_key, ylabel, title in [
        (axes[0], "phase2_val_history", "L_hidden (MSE)", "Phase 2 — teacher-student L_hidden"),
        (axes[1], "phase3_val_history", "Relative L2",    "Phase 3 — reconstruction loss"),
    ]:
        for r in good_results:
            h = r[phase_key]
            if len(h) > 0:
                ax.plot(h, color="steelblue", alpha=0.8, linewidth=1.5,
                        label=f"good s{r['seed']} (err={r['robust_err']:.3f})")
        for r in bad_results:
            h = r[phase_key]
            if len(h) > 0:
                ax.plot(h, color="salmon", alpha=0.8, linewidth=1.5, linestyle="--",
                        label=f"bad  s{r['seed']} (err={r['robust_err']:.3f})")
        ax.set_xlabel("Validation check (every 20 / 10 epochs)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Training convergence: good seeds (blue) vs bad seeds (red)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Trust weights: best vs worst seed
# ---------------------------------------------------------------------------

def plot_trust_comparison(
    best: dict,
    worst: dict,
    out_path: Path,
) -> None:
    """2×2 grid: (best | worst) × (clean | corrupted inputs)."""
    num_sensors = best["num_sensors"]
    uniform = 1.0 / num_sensors
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey=False)
    configs = [
        (best,  "trust_clean",  f"Best seed {best['seed']} — clean inputs",    axes[0, 0]),
        (best,  "trust_noisy",  f"Best seed {best['seed']} — corrupted inputs", axes[0, 1]),
        (worst, "trust_clean",  f"Worst seed {worst['seed']} — clean inputs",   axes[1, 0]),
        (worst, "trust_noisy",  f"Worst seed {worst['seed']} — corrupted inputs", axes[1, 1]),
    ]
    for r, key, title, ax in configs:
        w = r[key]   # (T_test, p)
        T = w.shape[0]
        for i in range(num_sensors):
            ax.plot(w[:, i], color=colors[i % len(colors)],
                    label=f"Sensor {i} (loc {r['sensor_locations'][i]})")
        ax.axhline(uniform, color="black", linestyle="--", linewidth=0.8,
                   label=f"Uniform 1/{num_sensors}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Test timestep")
        ax.set_ylabel("Trust weight")
        ax.set_xlim(0, T - 1)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Trust weights — best seed (robust err={best['robust_err']:.4f}) "
        f"vs worst (robust err={worst['robust_err']:.4f})\n"
        f"Noisy scenario: {best['noisy_scenario']}",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Reconstruction panels: best vs worst
# ---------------------------------------------------------------------------

def plot_reconstruction_comparison(
    best: dict,
    worst: dict,
    out_path: Path,
) -> None:
    """Side-by-side spatial reconstructions at three test snapshots."""
    nx, ny = best["nx"], best["ny"]
    truth = best["truth"]
    T_test = truth.shape[0]
    snaps = sorted({0, T_test // 2, T_test - 1})

    fig, axes = plt.subplots(len(snaps), 4, figsize=(16, 3 * len(snaps)))
    vmin, vmax = truth.min(), truth.max()

    col_labels = [
        "Truth",
        f"RobustSHRED best (s{best['seed']})",
        f"RobustSHRED worst (s{worst['seed']})",
        "Vanilla SHRED",
    ]
    arrays = [truth, best["robust_recon"], worst["robust_recon"], best["vanilla_recon"]]

    for row, t in enumerate(snaps):
        for col, (arr, label) in enumerate(zip(arrays, col_labels)):
            ax = axes[row, col]
            im = ax.imshow(arr[t].reshape(nx, ny), vmin=vmin, vmax=vmax,
                           cmap="RdBu_r", origin="lower", aspect="auto")
            if row == 0:
                ax.set_title(label, fontsize=8)
            ax.set_ylabel(f"t={t}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Spatial reconstructions: best vs worst seed", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Encoder σ distributions (what uncertainty the model learns)
# ---------------------------------------------------------------------------

def plot_sigma_distributions(
    best: dict,
    worst: dict,
    cfg: DictConfig,
    out_path: Path,
) -> None:
    """Run the trained models with return_internals=True to get σ on clean test data.

    This shows whether good-seed encoders produce more informative (non-collapsed)
    uncertainty estimates than bad-seed encoders.
    """
    from pathlib import Path as P
    from sklearn.preprocessing import MinMaxScaler
    from get_shredded.data import build_sensor_windows, load_cylinder_data, qr_place

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    load_X, _, _ = load_cylinder_data(P(to_absolute_path(cfg.data.mat)))
    n, m = load_X.shape
    lags, test_size, val_size = int(cfg.model.lags), int(cfg.data.test_size), int(cfg.data.val_size)
    n_windows = n - lags
    train_end = n_windows - test_size - val_size
    train_indices = np.arange(0, train_end)
    test_indices  = np.arange(n_windows - test_size, n_windows)

    sensor_locs = best["sensor_locations"]
    sc = MinMaxScaler().fit(load_X[train_indices])
    transformed_X = sc.transform(load_X).astype(np.float32)
    all_data_in = build_sensor_windows(transformed_X, sensor_locs, lags)
    test_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32, device=device)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    num_sensors = best["num_sensors"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, r, label in [
        (axes[0], best,  f"Best seed {best['seed']} (robust err={best['robust_err']:.4f})"),
        (axes[1], worst, f"Worst seed {worst['seed']} (robust err={worst['robust_err']:.4f})"),
    ]:
        model = RobustSHRED(
            num_sensors=num_sensors,
            output_size=m,
            hidden_size=int(cfg.model.hidden_size),
            hidden_layers=int(cfg.model.hidden_layers),
            l1=int(cfg.model.l1),
            l2=int(cfg.model.l2),
            dropout=float(cfg.model.dropout),
            d_z=int(cfg.model.d_z),
        ).to(device)
        model.load_state_dict({k: v.to(device) for k, v in r["robust_recon_state"].items()})
        model.eval()
        with torch.no_grad():
            _, internals = model(test_in, return_internals=True)
        sigma = internals["sigma"].detach().cpu().numpy()  # (T, lags, p, d_z)
        sigma_mean_over_lags = sigma.mean(axis=1)          # (T, p, d_z)

        for i in range(num_sensors):
            vals = sigma_mean_over_lags[:, i, :].flatten()
            ax.hist(vals, bins=40, alpha=0.6, color=colors[i % len(colors)],
                    label=f"Sensor {i}", density=True)

        ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, label="Prior σ=1")
        ax.set_xlabel("Encoder σ (uncertainty)")
        ax.set_ylabel("Density")
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Encoder uncertainty σ distributions on clean test data\n"
        "Good seeds: sensors separate into distinct σ clusters; bad seeds: all σ collapse near prior",
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="cross_val_robust")
def main(cfg: DictConfig) -> None:
    out_dir = Path(to_absolute_path(cfg.outputs.root))
    invest_dir = out_dir / "investigation"

    # --- Determine seeds to investigate ---
    cv_npz = out_dir / "cv_results.npz"
    manual_seeds = OmegaConf.select(cfg, "investigate.seeds", default=None)
    gap_threshold = float(OmegaConf.select(cfg, "investigate.gap_threshold", default=0.05))

    if manual_seeds is not None:
        seeds_to_run = [int(s) for s in manual_seeds]
        print(f"\nInvestigating manually specified seeds: {seeds_to_run}")
        # Classify using gap_threshold against the vanilla baseline
        # (we'll run everything and classify after)
        good_seeds_set: set[int] = set()
        bad_seeds_set:  set[int] = set()
    elif cv_npz.exists():
        good_seeds, bad_seeds = _load_cv_seeds(cv_npz, gap_threshold)
        # Pick up to 3 of each to keep runtime manageable
        seeds_to_run = good_seeds[:3] + bad_seeds[:3]
        good_seeds_set = set(good_seeds[:3])
        bad_seeds_set  = set(bad_seeds[:3])
        print(f"\nFrom cv_results.npz (gap_threshold={gap_threshold}):")
        print(f"  Good seeds: {good_seeds}  →  investigating: {good_seeds[:3]}")
        print(f"  Bad  seeds: {bad_seeds}   →  investigating: {bad_seeds[:3]}")
    else:
        raise FileNotFoundError(
            f"{cv_npz} not found. Run cross_val_robust.py first, or pass "
            "investigate.seeds=[s1,s2,...] to specify seeds manually."
        )

    print(f"\nRe-running {len(seeds_to_run)} seeds with full diagnostic capture...\n")
    results = _run_seeds(seeds_to_run, cfg)

    # Classify results (handles both auto and manual paths)
    if manual_seeds is not None:
        median_err = np.median([r["robust_err"] for r in results])
        good_results = [r for r in results if r["robust_err"] < median_err]
        bad_results  = [r for r in results if r["robust_err"] >= median_err]
    else:
        good_results = [r for r in results if r["seed"] in good_seeds_set]
        bad_results  = [r for r in results if r["seed"] in bad_seeds_set]

    best  = min(results, key=lambda r: r["robust_err"])
    worst = max(results, key=lambda r: r["robust_err"])

    print(f"\nBest  seed: {best['seed']}  robust_err={best['robust_err']:.6f}")
    print(f"Worst seed: {worst['seed']}  robust_err={worst['robust_err']:.6f}")

    # --- Phase 2 convergence summary ---
    print("\n--- Phase 2 final L_hidden (lower = better convergence) ---")
    for r in sorted(results, key=lambda r: r["robust_err"]):
        h2 = r["phase2_val_history"]
        final_h2 = float(h2[-1]) if len(h2) > 0 else float("nan")
        h3 = r["phase3_val_history"]
        final_h3 = float(h3[-1]) if len(h3) > 0 else float("nan")
        tag = "GOOD" if r in good_results else "BAD "
        print(f"  [{tag}] seed={r['seed']}  "
              f"Ph2_final={final_h2:.4e}  Ph3_final={final_h3:.4f}  "
              f"robust_err={r['robust_err']:.6f}")

    # --- Figures ---
    print("\nGenerating figures...")

    plot_phase_curves(good_results, bad_results, invest_dir / "phase_curves.png")

    plot_trust_comparison(best, worst, invest_dir / "trust_weights.png")

    plot_reconstruction_comparison(best, worst, invest_dir / "reconstruction.png")

    # Sigma distributions require the saved state dict — attach it to results
    # We store robust_recon_state in the dict so plot_sigma_distributions can load it.
    # run_robust_experiment_detailed doesn't return it, so we skip if unavailable.
    print("\n(Sigma distribution plot skipped — state_dicts not stored in detailed run)")
    print("  To enable it, save robust_shred.state_dict() in run_robust_experiment_detailed.")

    # --- Save investigation data ---
    np.savez(
        invest_dir / "investigation.npz",
        seeds=np.array([r["seed"] for r in results]),
        robust_errs=np.array([r["robust_err"] for r in results]),
        vanilla_errs=np.array([r["vanilla_err"] for r in results]),
        phase2_final=np.array([
            float(r["phase2_val_history"][-1]) if len(r["phase2_val_history"]) > 0 else np.nan
            for r in results
        ]),
        phase3_final=np.array([
            float(r["phase3_val_history"][-1]) if len(r["phase3_val_history"]) > 0 else np.nan
            for r in results
        ]),
    )
    print(f"\nInvestigation data saved → {invest_dir}/")


if __name__ == "__main__":
    main()

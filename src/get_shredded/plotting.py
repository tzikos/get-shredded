"""Plotting helpers for SHRED cylinder experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FuncAnimation, PillowWriter

from .experiment import SCENARIOS, RunResult, RobustnessResult


def _infer_grid(m: int, nx: int, ny: int) -> tuple[int, int]:
    """If nx,ny weren't stored in the .mat we fall back to a near-square reshape."""
    if nx > 0 and ny > 0 and nx * ny == m:
        return nx, ny
    side = int(np.sqrt(m))
    while side > 1 and m % side != 0:
        side -= 1
    return side, m // side


def _to_field(vec: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return vec.reshape(nx, ny, order="F") if vec.size == nx * ny else vec.reshape(nx, ny)


def _plot_limits(truth: np.ndarray, percentile: float = 99.5) -> float:
    limit = float(np.percentile(np.abs(truth), percentile))
    return max(limit, 1e-6)


def plot_reconstruction_panel(
    result: RunResult,
    snapshot_indices: list[int],
    save_path: Path,
) -> None:
    """Grid: rows = snapshots, cols = (truth+sensors, SHRED, SDN, QR/POD).
    Sensor locations overlaid on the truth column. Saves a PNG."""
    nx, ny = _infer_grid(result.truth.shape[1], result.nx, result.ny)
    sensor_rows, sensor_cols = np.unravel_index(result.sensor_locations, (nx, ny), order="F")

    cols = [
        ("Ground truth + sensors", result.truth),
        (f"SHRED  (err={result.shred_err:.3f})", result.shred_recon),
        (f"SDN    (err={result.sdn_err:.3f})", result.sdn_recon),
        (f"QR/POD (err={result.qrpod_err:.3f})", result.qrpod_recon),
    ]
    n_rows = len(snapshot_indices)
    n_cols = len(cols)
    vmax = _plot_limits(result.truth)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.4 * n_rows), squeeze=False)
    for r, snap_idx in enumerate(snapshot_indices):
        for c, (title, data) in enumerate(cols):
            ax = axes[r, c]
            field = _to_field(data[snap_idx], nx, ny)
            ax.imshow(field, cmap="seismic", norm=norm, aspect="auto", interpolation="nearest")
            if c == 0:
                ax.scatter(sensor_cols, sensor_rows, c="lime", edgecolors="black",
                           s=60, marker="o", linewidths=1.0)
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"t = {snap_idx}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"Cylinder vorticity reconstruction — {result.num_sensors} sensors ({result.placement}), lags={result.lags}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def animate_reconstructions(result: RunResult, save_path: Path, fps: int = 4) -> None:
    """Side-by-side animated GIF over the full test set."""
    nx, ny = _infer_grid(result.truth.shape[1], result.nx, result.ny)
    sensor_rows, sensor_cols = np.unravel_index(result.sensor_locations, (nx, ny), order="F")

    panels = [
        ("Ground truth", result.truth, True),
        ("SHRED", result.shred_recon, False),
        ("SDN", result.sdn_recon, False),
        ("QR/POD", result.qrpod_recon, False),
    ]
    vmax = _plot_limits(result.truth)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.0))
    images = []
    for ax, (title, data, show_sensors) in zip(axes, panels):
        im = ax.imshow(
            _to_field(data[0], nx, ny),
            cmap="seismic",
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if show_sensors:
            ax.scatter(sensor_cols, sensor_rows, c="lime", edgecolors="black",
                       s=60, marker="o", linewidths=1.0)
        images.append(im)
    title_obj = fig.suptitle("", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    def update(frame: int):
        for im, (_, data, _) in zip(images, panels):
            im.set_data(_to_field(data[frame], nx, ny))
        title_obj.set_text(f"Test snapshot {frame + 1}/{result.truth.shape[0]}")
        return images + [title_obj]

    anim = FuncAnimation(fig, update, frames=result.truth.shape[0], interval=1000 // fps, blit=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def plot_per_snapshot_error(result: RunResult, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(1, len(result.shred_err_per_snap) + 1)
    ax.plot(x, result.shred_err_per_snap, marker="o", label="SHRED")
    ax.plot(x, result.sdn_err_per_snap, marker="s", label="SDN")
    ax.plot(x, result.qrpod_err_per_snap, marker="^", label="QR/POD")
    ax.set_xlabel("Test snapshot index")
    ax.set_ylabel("Relative L2 error")
    ax.set_title(f"Per-snapshot reconstruction error ({result.num_sensors} sensors, {result.placement})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_training_curves(result: RunResult, save_path: Path, val_every: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    epochs_shred = np.arange(1, len(result.shred_val_history) + 1) * val_every
    epochs_sdn = np.arange(1, len(result.sdn_val_history) + 1) * val_every
    ax.plot(epochs_shred, result.shred_val_history, marker="o", label="SHRED")
    ax.plot(epochs_sdn, result.sdn_val_history, marker="s", label="SDN")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation relative L2 error")
    ax.set_yscale("log")
    ax.set_title("Training curves")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_sweep_error_vs_sensors(
    sensor_counts: list[int],
    series: dict[str, list[float]],
    save_path: Path,
) -> None:
    """series: label -> list of errors (one per sensor count). Each label is plotted
    as one line. Convention: include " (QR)" / " (random)" in labels."""
    style = {
        "SHRED (random)":  dict(marker="o", linestyle="--", color="#1f77b4"),
        "SHRED (QR)":      dict(marker="o", linestyle="-",  color="#1f77b4"),
        "SDN (random)":    dict(marker="s", linestyle="--", color="#2ca02c"),
        "SDN (QR)":        dict(marker="s", linestyle="-",  color="#2ca02c"),
        "QR/POD":          dict(marker="^", linestyle="-",  color="#d62728"),
    }
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, errs in series.items():
        kwargs = style.get(label, {"marker": "x"})
        ax.plot(sensor_counts, errs, label=label, **kwargs)
    ax.set_xlabel("Number of sensors")
    ax.set_ylabel("Test relative L2 error")
    ax.set_yscale("log")
    ax.set_title("Reconstruction error vs sensor count (median over seeds)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Robustness comparison plots
# ---------------------------------------------------------------------------

_SCENARIO_LABEL = {
    "gaussian": "Gaussian Noise",
    "dropout":  "Sensor Dropout",
    "hybrid":   "Hybrid",
    "burst":    "Burst Noise",
    "clean":    "Clean (no noise)",
}

# Colorblind-friendly palette
_COLORS = {
    "SHRED-clean":    "#729ECE",   # soft blue
    "SHRED-gaussian": "#2171B5",   # strong blue
    "SHRED-dropout":  "#2171B5",
    "SHRED-hybrid":   "#2171B5",
    "SDN-clean":      "#8CC98D",   # soft green
    "SDN-gaussian":   "#238B45",   # strong green
    "SDN-dropout":    "#238B45",
    "SDN-hybrid":     "#238B45",
    "SHRED-burst":    "#6A3D9A",   # purple
    "SDN-burst":      "#6A3D9A",
    "QR-POD":         "#D6804F",   # orange
}


def _col_display_name(model_name: str, scenario: str) -> str:
    """In panel/GIF columns the augmented model is always 'SHRED/SDN-augmented'
    so the label stays consistent across rows regardless of which scenario is shown."""
    if model_name == f"SHRED-{scenario}":
        return "SHRED-augmented"
    if model_name == f"SDN-{scenario}":
        return "SDN-augmented"
    return model_name


def _scenario_model_order(result: RobustnessResult, scenario: str) -> list:
    """Return [SHRED-clean, SHRED-{scenario}, SDN-clean, SDN-{scenario}, QR-POD]."""
    wanted = ["SHRED-clean", f"SHRED-{scenario}", "SDN-clean", f"SDN-{scenario}", "QR-POD"]
    by_name = {m.name: m for m in result.models}
    return [by_name[n] for n in wanted if n in by_name]


def plot_robustness_bar(result: RobustnessResult, save_path: Path) -> None:
    """Grouped bar chart, one row per corruption scenario.

    Each row = one corruption scenario (gaussian / dropout / hybrid / burst).
    Each group = one model; two bars per group: clean-test (solid) and
    scenario-noisy-test (hatched).  Lets you see both the clean-performance
    baseline and the degradation under each corruption type side-by-side.
    """
    fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(11, 5 * len(SCENARIOS)), sharex=False)
    width = 0.38

    for ax, scenario in zip(axes, SCENARIOS):
        models = _scenario_model_order(result, scenario)
        names  = [m.name for m in models]
        errs_c = [m.err_clean for m in models]
        errs_n = [m.err_noisy[scenario] for m in models]
        colors = [_COLORS.get(n, "#888888") for n in names]

        x = np.arange(len(names))
        b_clean = ax.bar(x - width / 2, errs_c, width,
                         color=colors, alpha=0.70, label="Clean test", edgecolor="white")
        b_noisy = ax.bar(x + width / 2, errs_n, width,
                         color=colors, alpha=0.95, hatch="///", label="Corrupted test",
                         edgecolor="white")

        # value labels on top of each bar
        for bar in list(b_clean) + list(b_noisy):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylabel("Relative L2 error", fontsize=12)
        ax.set_title(_SCENARIO_LABEL[scenario], fontsize=14, fontweight="bold", pad=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(errs_c + errs_n) * 1.20)
        if ax is axes[0]:
            ax.legend(fontsize=11, loc="upper right")

    fig.suptitle(
        f"Robustness to sensor corruption — {result.num_sensors} sensors "
        f"({result.placement}), lags={result.lags}",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_per_snapshot(
    result: RobustnessResult,
    scenario: str,
    save_path: Path,
) -> None:
    """Per-snapshot error for the models relevant to one scenario.

    scenario: "clean" | "gaussian" | "dropout" | "hybrid"
    Shows [SHRED-clean, SHRED-{scenario}, SDN-clean, SDN-{scenario}, QR-POD]
    evaluated under that scenario's test corruption (or clean, if scenario="clean").
    """
    if scenario == "clean":
        models = result.models
        getter = lambda m: m.err_per_snap_clean
    else:
        models = _scenario_model_order(result, scenario)
        getter = lambda m: m.err_per_snap_noisy[scenario]

    markers = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(models):
        errs = getter(m)
        ax.plot(np.arange(1, len(errs) + 1), errs,
                marker=markers[i % len(markers)], linewidth=1.5,
                color=_COLORS.get(m.name, f"C{i}"), label=m.name)

    ax.set_xlabel("Test snapshot index", fontsize=12)
    ax.set_ylabel("Relative L2 error", fontsize=12)
    ax.set_title(
        f"Per-snapshot error — {_SCENARIO_LABEL[scenario]}\n"
        f"{result.num_sensors} sensors ({result.placement})",
        fontsize=13,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_robustness_panel(
    result: RobustnessResult,
    scenario: str,
    snapshot_indices: list[int],
    save_path: Path,
) -> None:
    """Reconstruction grid for one scenario.

    Rows = snapshots, columns = [truth + sensors, SHRED-clean, SHRED-{scenario},
    SDN-clean, SDN-{scenario}, QR-POD], all evaluated under that scenario's
    test corruption (or clean, if scenario="clean").
    """
    if scenario == "clean":
        models = result.models
        get_recon = lambda m: m.recon_clean
        get_err   = lambda m: m.err_clean
    else:
        models = _scenario_model_order(result, scenario)
        get_recon = lambda m: m.recon_noisy[scenario]
        get_err   = lambda m: m.err_noisy[scenario]

    truth = result.truth
    nx, ny = _infer_grid(truth.shape[1], result.nx, result.ny)
    sensor_rows, sensor_cols = np.unravel_index(result.sensor_locations, (nx, ny), order="F")
    vmax = _plot_limits(truth)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    col_data = [("Ground truth\n+ sensors", truth, True)] + [
        (f"{_col_display_name(m.name, scenario)}\nerr={get_err(m):.3f}", get_recon(m), False)
        for m in models
    ]
    n_rows, n_cols = len(snapshot_indices), len(col_data)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.9 * n_cols, 2.5 * n_rows), squeeze=False)

    for r, snap_idx in enumerate(snapshot_indices):
        for c, (title, data, show_sensors) in enumerate(col_data):
            ax = axes[r, c]
            ax.imshow(_to_field(data[snap_idx], nx, ny),
                      cmap="seismic", norm=norm, aspect="auto", interpolation="nearest")
            if show_sensors:
                ax.scatter(sensor_cols, sensor_rows, c="lime", edgecolors="black",
                           s=40, marker="o", linewidths=0.8)
            if r == 0:
                ax.set_title(title, fontsize=9, pad=4)
            if c == 0:
                ax.set_ylabel(f"t = {snap_idx}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"{_SCENARIO_LABEL[scenario]} — {result.num_sensors} sensors "
        f"({result.placement}), lags={result.lags}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def animate_robustness_comparison(
    result: RobustnessResult,
    save_path: Path,
    fps: int = 4,
) -> None:
    """Combined GIF: 3 rows (one per scenario) × 6 columns (truth + 5 models).

    Each cell is animated over the test snapshots.  Row labels on the left
    identify the corruption scenario; column headers name the model and show
    the aggregate error under that scenario.
    """
    truth = result.truth
    nx, ny = _infer_grid(truth.shape[1], result.nx, result.ny)
    sensor_rows, sensor_cols = np.unravel_index(result.sensor_locations, (nx, ny), order="F")
    vmax = _plot_limits(truth)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    T = truth.shape[0]

    # Build layout: row_panels[row] = list of (title, data_array, show_sensors)
    row_panels: list[list[tuple[str, np.ndarray, bool]]] = []
    for scenario in SCENARIOS:
        models = _scenario_model_order(result, scenario)
        panels = [("Ground truth\n+ sensors", truth, True)] + [
            (f"{_col_display_name(m.name, scenario)}\n(err={m.err_noisy[scenario]:.3f})",
             m.recon_noisy[scenario], False)
            for m in models
        ]
        row_panels.append(panels)

    n_rows = len(SCENARIOS)
    n_cols = len(row_panels[0])
    cell_w, cell_h = 2.5, 2.2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols + 1.0, cell_h * n_rows + 0.8),
        squeeze=False,
    )

    images: list[list] = [[None] * n_cols for _ in range(n_rows)]
    for r, (scenario, panels) in enumerate(zip(SCENARIOS, row_panels)):
        for c, (title, data, show_sensors) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(
                _to_field(data[0], nx, ny),
                cmap="seismic", norm=norm, aspect="auto", interpolation="nearest",
            )
            if show_sensors:
                ax.scatter(sensor_cols, sensor_rows, c="lime", edgecolors="black",
                           s=30, marker="o", linewidths=0.7)
            if r == 0:
                ax.set_title(title, fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(_SCENARIO_LABEL[scenario], fontsize=9, labelpad=4)
            ax.set_xticks([]); ax.set_yticks([])
            images[r][c] = im

    title_obj = fig.suptitle("", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def _update(frame: int):
        for r, panels in enumerate(row_panels):
            for c, (_, data, _) in enumerate(panels):
                images[r][c].set_data(_to_field(data[frame], nx, ny))
        title_obj.set_text(
            f"Test snapshot {frame + 1}/{T}  —  "
            f"{result.num_sensors} sensors ({result.placement}), lags={result.lags}"
        )
        return [images[r][c] for r in range(n_rows) for c in range(n_cols)] + [title_obj]

    anim = FuncAnimation(fig, _update, frames=T, interval=1000 // fps, blit=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_robustness_table(result: RobustnessResult, save_path: Path) -> None:
    """Save a full error table (all models × all test conditions) as Markdown
    and LaTeX.  The best value in each column is bolded in the LaTeX version.
    """
    col_headers = ["Clean"] + [_SCENARIO_LABEL[s] for s in SCENARIOS]
    models = result.models

    # Collect all errors into a 2-D array: (n_models, 4)
    rows_data: list[list[float]] = []
    for m in models:
        row = [m.err_clean] + [m.err_noisy[s] for s in SCENARIOS]
        rows_data.append(row)

    # ---------- Markdown ----------
    md_lines = []
    md_lines.append(
        f"## Robustness error table — {result.num_sensors} sensors "
        f"({result.placement}), lags={result.lags}\n"
    )
    header = "| Model | " + " | ".join(col_headers) + " |"
    sep    = "|-------|" + "|".join(["-------"] * len(col_headers)) + "|"
    md_lines += [header, sep]
    for m, row in zip(models, rows_data):
        cells = " | ".join(f"{v:.4f}" for v in row)
        md_lines.append(f"| {m.name} | {cells} |")
    md_lines.append("")

    # Best per column annotation
    best_idx = [int(np.argmin([r[c] for r in rows_data])) for c in range(len(col_headers))]
    md_lines.append("*Best per column: " +
                    ", ".join(f"**{col_headers[c]}** → {models[best_idx[c]].name}"
                              for c in range(len(col_headers))) + "*")

    # ---------- LaTeX ----------
    n_cols_total = 1 + len(col_headers)
    col_fmt = "l" + "r" * len(col_headers)
    tex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Reconstruction error (relative L2) under clean and corrupted sensor inputs. "
        r"Bold = best in column.}",
        r"\label{tab:robustness}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        "Model & " + " & ".join(col_headers) + r" \\",
        r"\midrule",
    ]
    for m, row in zip(models, rows_data):
        cells = []
        for c, v in enumerate(row):
            s = f"{v:.4f}"
            if int(np.argmin([r[c] for r in rows_data])) == models.index(m):
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        tex_lines.append(m.name.replace("-", r"\textendash ") + " & " + " & ".join(cells) + r" \\")
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    md_path  = save_path.with_suffix(".md")
    tex_path = save_path.with_suffix(".tex")
    md_path.write_text("\n".join(md_lines))
    tex_path.write_text("\n".join(tex_lines))
    # Also print the markdown table to stdout for quick inspection
    print("\n".join(md_lines))


# ---------------------------------------------------------------------------
# Sweep plot: error vs num_sensors across placements and scenarios
# ---------------------------------------------------------------------------

_SWEEP_STYLES: dict[str, dict] = {
    "SHRED-clean":     dict(color="#729ECE", ls="-",  marker="o", lw=2, ms=6),
    "SHRED-augmented": dict(color="#2171B5", ls="--", marker="o", lw=2, ms=6),
    "SDN-clean":       dict(color="#8CC98D", ls="-",  marker="s", lw=2, ms=6),
    "SDN-augmented":   dict(color="#238B45", ls="--", marker="s", lw=2, ms=6),
    "QR-POD":          dict(color="#D6804F", ls="-",  marker="^", lw=2, ms=6),
}


def plot_robustness_sweep(
    all_results: dict[tuple[int, str], "RobustnessResult"],
    sensor_counts: list[int],
    placements: list[str],
    save_path: Path,
) -> None:
    """Grid of error-vs-sensors plots: rows = scenarios, columns = placements.

    For each (scenario, placement) cell, five lines are drawn:
      SHRED-clean, SHRED-augmented (trained with matching corruption),
      SDN-clean, SDN-augmented, QR-POD — all evaluated under that scenario's
      test corruption.  Solid = clean-trained, dashed = augmentation-trained.
    """
    n_rows = len(SCENARIOS)
    n_cols = len(placements)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.0 * n_rows),
        squeeze=False,
        sharey=False,
    )

    line_keys = ["SHRED-clean", "SHRED-augmented", "SDN-clean", "SDN-augmented", "QR-POD"]

    for row, scenario in enumerate(SCENARIOS):
        for col, placement in enumerate(placements):
            ax = axes[row][col]

            # Collect errors for each line across sensor counts
            series: dict[str, list[float]] = {k: [] for k in line_keys}
            valid_counts: list[int] = []

            for n in sensor_counts:
                res = all_results.get((n, placement))
                if res is None:
                    continue
                by_name = {m.name: m for m in res.models}
                aug_name = f"SHRED-{scenario}"
                sdn_aug_name = f"SDN-{scenario}"
                if aug_name not in by_name:
                    continue
                series["SHRED-clean"].append(by_name["SHRED-clean"].err_noisy[scenario])
                series["SHRED-augmented"].append(by_name[aug_name].err_noisy[scenario])
                series["SDN-clean"].append(by_name["SDN-clean"].err_noisy[scenario])
                series["SDN-augmented"].append(by_name[sdn_aug_name].err_noisy[scenario])
                series["QR-POD"].append(by_name["QR-POD"].err_noisy[scenario])
                valid_counts.append(n)

            if not valid_counts:
                ax.set_visible(False)
                continue

            for key in line_keys:
                ax.plot(valid_counts, series[key], label=key, **_SWEEP_STYLES[key])

            ax.set_yscale("log")
            ax.set_xlabel("Number of sensors", fontsize=11)
            ax.set_ylabel("Relative L2 error", fontsize=11)
            ax.set_xticks(valid_counts)
            ax.set_xticklabels(valid_counts, fontsize=10)
            ax.grid(alpha=0.3, which="both")

            title = f"{_SCENARIO_LABEL[scenario]}  ·  {placement} placement"
            ax.set_title(title, fontsize=12, fontweight="bold")

            if row == 0 and col == n_cols - 1:
                ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Reconstruction error vs sensor count\n"
        "(dashed = trained with matching augmentation, solid = clean-trained)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Consolidated sweep bar chart and table (all sensor configs in one output)
# ---------------------------------------------------------------------------

def plot_robustness_bar_sweep(
    all_results: "dict[tuple[int, str], RobustnessResult]",
    sensor_counts: "list[int]",
    placements: "list[str]",
    save_dir: Path,
) -> None:
    """One bar-chart figure per placement, saved as robustness_bar_{placement}.png.

    Layout per figure: rows = scenarios, columns = sensor counts.
    Each cell: clean-test (solid) vs corrupted-test (hatched) bars for the
    five relevant models.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    width = 0.38
    n_rows = len(SCENARIOS)
    n_cols = len(sensor_counts)

    for placement in placements:
        # pre-compute global y-limit so all subplots are comparable
        global_max = 0.0
        for scenario in SCENARIOS:
            for n in sensor_counts:
                res = all_results.get((n, placement))
                if res is None:
                    continue
                models = _scenario_model_order(res, scenario)
                errs_c = [m.err_clean for m in models]
                errs_n = [m.err_noisy[scenario] for m in models]
                local_max = max(errs_c + errs_n)
                if local_max > global_max:
                    global_max = local_max
        y_top = global_max * 1.25

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.8 * n_cols, 3.6 * n_rows),
            squeeze=False,
            sharey=False,
        )

        for row, scenario in enumerate(SCENARIOS):
            for col, n in enumerate(sensor_counts):
                ax = axes[row][col]
                res = all_results.get((n, placement))
                if res is None:
                    ax.set_visible(False)
                    continue

                models = _scenario_model_order(res, scenario)
                names  = [m.name for m in models]
                errs_c = [m.err_clean for m in models]
                errs_n = [m.err_noisy[scenario] for m in models]
                colors = [_COLORS.get(name, "#888888") for name in names]

                x = np.arange(len(names))
                b_clean = ax.bar(x - width / 2, errs_c, width,
                                 color=colors, alpha=0.70, label="Clean test", edgecolor="white")
                b_noisy = ax.bar(x + width / 2, errs_n, width,
                                 color=colors, alpha=0.95, hatch="///", label="Corrupted test",
                                 edgecolor="white")

                for bar in list(b_clean) + list(b_noisy):
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=6)

                ax.set_xticks(x)
                ax.set_xticklabels(names, fontsize=7, rotation=30, ha="right")
                ax.grid(axis="y", alpha=0.3)
                ax.set_ylim(0, y_top)

                if row == 0:
                    ax.set_title(f"{n} sensors", fontsize=10, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(
                        f"{_SCENARIO_LABEL[scenario]}\n\nRel. L2 error", fontsize=9
                    )
                if row == 0 and col == n_cols - 1:
                    ax.legend(fontsize=8, loc="upper right")

        fig.suptitle(
            f"Robustness to sensor corruption — {placement} placement\n"
            "(solid = clean test, hatched = corrupted test)",
            fontsize=13, y=1.01,
        )
        fig.tight_layout()
        fig.savefig(save_dir / f"robustness_bar_{placement}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_robustness_table_sweep(
    all_results: "dict[tuple[int, str], RobustnessResult]",
    sensor_counts: "list[int]",
    placements: "list[str]",
    save_path: Path,
) -> None:
    """Save one Markdown + LaTeX table covering all (num_sensors, placement) combos.

    Each combo gets its own sub-table (Markdown section / LaTeX table environment).
    Columns: Clean + one column per scenario. Best value per column is bolded.
    """
    col_headers = ["Clean"] + [_SCENARIO_LABEL[s] for s in SCENARIOS]

    md_lines = ["# Robustness error tables\n"]
    tex_blocks: list[str] = []

    for n in sensor_counts:
        for p in placements:
            res = all_results.get((n, p))
            if res is None:
                continue
            models = res.models
            rows_data = [[m.err_clean] + [m.err_noisy[s] for s in SCENARIOS] for m in models]
            best_idx  = [int(np.argmin([r[c] for r in rows_data])) for c in range(len(col_headers))]

            # --- Markdown ---
            md_lines.append(f"## {n} sensors · {p} placement\n")
            header = "| Model | " + " | ".join(col_headers) + " |"
            sep    = "|-------|" + "|".join(["-------"] * len(col_headers)) + "|"
            md_lines += [header, sep]
            for mi, (m, row) in enumerate(zip(models, rows_data)):
                cells = " | ".join(
                    f"**{v:.4f}**" if best_idx[c] == mi else f"{v:.4f}"
                    for c, v in enumerate(row)
                )
                md_lines.append(f"| {m.name} | {cells} |")
            md_lines.append("")

            # --- LaTeX ---
            col_fmt = "l" + "r" * len(col_headers)
            tex_lines = [
                r"\begin{table}[ht]",
                r"\centering",
                rf"\caption{{Reconstruction error ({n} sensors, {p} placement). Bold = best in column.}}",
                rf"\label{{tab:robustness_{n}_{p}}}",
                rf"\begin{{tabular}}{{{col_fmt}}}",
                r"\toprule",
                "Model & " + " & ".join(col_headers) + r" \\",
                r"\midrule",
            ]
            for mi, (m, row) in enumerate(zip(models, rows_data)):
                cells = []
                for c, v in enumerate(row):
                    s = f"{v:.4f}"
                    if best_idx[c] == mi:
                        s = rf"\textbf{{{s}}}"
                    cells.append(s)
                tex_lines.append(
                    m.name.replace("-", r"\textendash ") + " & " + " & ".join(cells) + r" \\"
                )
            tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
            tex_blocks.append("\n".join(tex_lines))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.with_suffix(".md").write_text("\n".join(md_lines))
    save_path.with_suffix(".tex").write_text("\n\n".join(tex_blocks))
    print("\n".join(md_lines))

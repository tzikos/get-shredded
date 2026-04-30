"""Plotting helpers for SHRED cylinder experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FuncAnimation, PillowWriter

from .experiment import RunResult


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

"""Standalone evaluation utilities for RobustSHRED detection performance."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .robust_model import RobustSHRED


def evaluate_detection_auroc(
    robust_shred: RobustSHRED,
    test_in_clean: torch.Tensor,
    test_in_corrupted: torch.Tensor,
    corrupted_sensor_indices: list[int],
    num_sensors: int,
) -> dict[str, float]:
    """Evaluate how well trust weights identify corrupted sensors.

    Computes AUROC of per-sensor trust weights against ground-truth
    clean/corrupted labels, plus top-1 detection accuracy and mean
    trust for clean vs. corrupted sensors.

    Args:
        robust_shred: Trained RobustSHRED model (should be in eval mode).
        test_in_clean: Clean test windows, shape (T, lags, p).
        test_in_corrupted: Corrupted test windows, shape (T, lags, p).
        corrupted_sensor_indices: Indices of sensors that were corrupted.
        num_sensors: Total number of sensors.

    Returns:
        Dict with "auroc", "top1_detection_accuracy",
        "mean_trust_clean_sensors", "mean_trust_corrupted_sensors".
    """
    robust_shred.eval()
    with torch.no_grad():
        w_clean = robust_shred.get_trust_weights(test_in_clean).cpu().numpy()       # (T, p)
        w_corrupt = robust_shred.get_trust_weights(test_in_corrupted).cpu().numpy()  # (T, p)

    T = w_corrupt.shape[0]
    clean_indices = [i for i in range(num_sensors) if i not in corrupted_sensor_indices]

    # Ground-truth label per (sample, sensor): 1=clean, 0=corrupted
    # Shape (T, p) — same for every sample since corruption is fixed
    g = np.ones((T, num_sensors), dtype=np.float32)
    for idx in corrupted_sensor_indices:
        g[:, idx] = 0.0

    # Flatten (T*p,) for AUROC
    labels_flat = g.flatten()
    scores_flat = w_corrupt.flatten()

    if labels_flat.max() == labels_flat.min():
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(labels_flat, scores_flat))

    # Top-1 detection accuracy: fraction of test samples where the most distrusted
    # sensor (lowest trust) is one of the corrupted sensors
    most_distrusted = np.argmin(w_corrupt, axis=1)  # (T,)
    correct = np.isin(most_distrusted, corrupted_sensor_indices)
    top1_acc = float(correct.mean())

    mean_trust_clean = float(w_corrupt[:, clean_indices].mean()) if clean_indices else float("nan")
    mean_trust_corrupted = float(
        w_corrupt[:, corrupted_sensor_indices].mean()
    ) if corrupted_sensor_indices else float("nan")

    return {
        "auroc": auroc,
        "top1_detection_accuracy": top1_acc,
        "mean_trust_clean_sensors": mean_trust_clean,
        "mean_trust_corrupted_sensors": mean_trust_corrupted,
    }

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from get_shredded.baseline import PODDMDBaseline
from get_shredded.data import (
    fit_pod,
    load_cylinder_data,
    make_windows,
    project_to_latent,
    reconstruct_from_latent,
    split_time_series,
)
from get_shredded.train import TrainConfig, mse_on_windows, train_model


def rollout_rmse(
    pred_latent: np.ndarray,
    target_latent: np.ndarray,
    basis,
) -> float:
    pred_field = reconstruct_from_latent(pred_latent.T, basis)
    target_field = reconstruct_from_latent(target_latent.T, basis)
    return float(np.sqrt(np.mean((pred_field - target_field) ** 2)))


@hydra.main(version_base=None, config_path="../configs", config_name="compare_shred_vs_dmd")
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    data_path = Path(to_absolute_path(cfg.data.mat))
    x, _, _ = load_cylinder_data(data_path)
    x_train, x_val, x_test = split_time_series(
        x,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
    )

    basis = fit_pod(x_train, rank=cfg.model.rank)
    z_train = project_to_latent(x_train, basis)
    z_val = project_to_latent(x_val, basis)
    z_test = project_to_latent(x_test, basis)

    xw_train, yw_train = make_windows(z_train, seq_len=cfg.model.seq_len)
    xw_val, yw_val = make_windows(z_val, seq_len=cfg.model.seq_len)
    xw_test, yw_test = make_windows(z_test, seq_len=cfg.model.seq_len)

    horizon = min(cfg.rollout.horizon, yw_test.shape[0])
    target_latent_rollout = z_test[:, cfg.model.seq_len : cfg.model.seq_len + horizon].T

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shred_train_cfg = TrainConfig(
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        lr=cfg.train.lr,
        hidden_dim=cfg.model.hidden_dim,
        rnn_type=cfg.model.rnn_type,
        device=device,
    )
    shred_model, shred_hist = train_model(xw_train, yw_train, xw_val, yw_val, shred_train_cfg)
    shred_test_mse = mse_on_windows(shred_model, xw_test, yw_test, device=device)

    shred_seed = torch.from_numpy(xw_test[:1]).to(device)
    shred_rollout = shred_model.rollout(shred_seed, horizon=horizon).cpu().numpy()
    shred_rollout_rmse = rollout_rmse(shred_rollout, target_latent_rollout, basis)

    dmd_model = PODDMDBaseline(latent_dim=cfg.model.rank, ridge=cfg.comparison.dmd_ridge)
    dmd_model.fit_from_latent_series(z_train)
    dmd_pred = dmd_model.predict(xw_test)
    dmd_test_mse = float(np.mean((dmd_pred - yw_test) ** 2))

    dmd_rollout = dmd_model.rollout(xw_test[:1], horizon=horizon)
    dmd_rollout_rmse = rollout_rmse(dmd_rollout, target_latent_rollout, basis)

    print("=== SHRED vs POD+DMD (same POD space and test split) ===")
    print(f"SHRED final train loss: {shred_hist['train_loss'][-1]:.6e}")
    print(f"SHRED final val loss:   {shred_hist['val_loss'][-1]:.6e}")
    print("")
    print("Model      | One-step MSE (latent) | Rollout RMSE (full field)")
    print("-----------|------------------------|---------------------------")
    print(f"SHRED      | {shred_test_mse:>22.6e} | {shred_rollout_rmse:>25.6e}")
    print(f"POD+DMD    | {dmd_test_mse:>22.6e} | {dmd_rollout_rmse:>25.6e}")


if __name__ == "__main__":
    main()

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

from get_shredded.data import (
    fit_pod,
    load_cylinder_data,
    make_windows,
    project_to_latent,
    reconstruct_from_latent,
    split_time_series,
)
from get_shredded.train import TrainConfig, mse_on_windows, train_model


@hydra.main(version_base=None, config_path="../configs", config_name="cylinder_baseline")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = TrainConfig(
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        lr=cfg.train.lr,
        hidden_dim=cfg.model.hidden_dim,
        rnn_type=cfg.model.rnn_type,
        device=device,
    )

    model, history = train_model(xw_train, yw_train, xw_val, yw_val, train_cfg)
    test_mse = mse_on_windows(model, xw_test, yw_test, device=device)

    seed_seq = torch.from_numpy(xw_test[:1]).to(device)
    horizon = min(cfg.rollout.horizon, yw_test.shape[0])
    rollout_latent = model.rollout(seed_seq, horizon=horizon).cpu().numpy().T
    target_latent = z_test[:, cfg.model.seq_len : cfg.model.seq_len + horizon]

    x_rollout = reconstruct_from_latent(rollout_latent, basis)
    x_target = reconstruct_from_latent(target_latent, basis)
    rollout_rmse = float(np.sqrt(np.mean((x_rollout - x_target) ** 2)))

    print("Training complete")
    print(f"Final train loss: {history['train_loss'][-1]:.6e}")
    print(f"Final val loss:   {history['val_loss'][-1]:.6e}")
    print(f"Test one-step MSE (latent): {test_mse:.6e}")
    print(f"Rollout RMSE (full field, {horizon} steps): {rollout_rmse:.6e}")


if __name__ == "__main__":
    main()

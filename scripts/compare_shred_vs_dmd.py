from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SHRED against POD+DMD baseline")
    parser.add_argument("--data-mat", type=Path, default=Path("../../DATA/FLUIDS/CYLINDER_ALL.mat"))
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--dmd-ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def rollout_rmse(
    pred_latent: np.ndarray,
    target_latent: np.ndarray,
    basis,
) -> float:
    pred_field = reconstruct_from_latent(pred_latent.T, basis)
    target_field = reconstruct_from_latent(target_latent.T, basis)
    return float(np.sqrt(np.mean((pred_field - target_field) ** 2)))


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x, _, _ = load_cylinder_data(args.data_mat)
    x_train, x_val, x_test = split_time_series(x)

    basis = fit_pod(x_train, rank=args.rank)
    z_train = project_to_latent(x_train, basis)
    z_val = project_to_latent(x_val, basis)
    z_test = project_to_latent(x_test, basis)

    xw_train, yw_train = make_windows(z_train, seq_len=args.seq_len)
    xw_val, yw_val = make_windows(z_val, seq_len=args.seq_len)
    xw_test, yw_test = make_windows(z_test, seq_len=args.seq_len)

    horizon = min(30, yw_test.shape[0])
    target_latent_rollout = z_test[:, args.seq_len : args.seq_len + horizon].T

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shred_cfg = TrainConfig(
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        rnn_type=args.rnn_type,
        device=device,
    )
    shred_model, shred_hist = train_model(xw_train, yw_train, xw_val, yw_val, shred_cfg)
    shred_test_mse = mse_on_windows(shred_model, xw_test, yw_test, device=device)

    shred_seed = torch.from_numpy(xw_test[:1]).to(device)
    shred_rollout = shred_model.rollout(shred_seed, horizon=horizon).cpu().numpy()
    shred_rollout_rmse = rollout_rmse(shred_rollout, target_latent_rollout, basis)

    dmd_model = PODDMDBaseline(latent_dim=args.rank, ridge=args.dmd_ridge)
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

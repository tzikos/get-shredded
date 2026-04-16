from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

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
    parser = argparse.ArgumentParser(description="SHRED baseline on cylinder-vortex data")
    parser.add_argument("--data-mat", type=Path, default=Path("../../DATA/FLUIDS/CYLINDER_ALL.mat"))
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainConfig(
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        rnn_type=args.rnn_type,
        device=device,
    )

    model, history = train_model(xw_train, yw_train, xw_val, yw_val, cfg)
    test_mse = mse_on_windows(model, xw_test, yw_test, device=device)

    seed_seq = torch.from_numpy(xw_test[:1]).to(device)
    horizon = min(30, yw_test.shape[0])
    rollout_latent = model.rollout(seed_seq, horizon=horizon).cpu().numpy().T
    target_latent = z_test[:, args.seq_len : args.seq_len + horizon]

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

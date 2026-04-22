from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import ShallowRecurrentDecoder


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    hidden_dim: int
    rnn_type: str
    device: str


def _to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
) -> tuple[ShallowRecurrentDecoder, dict[str, list[float]]]:
    device = torch.device(cfg.device)
    latent_dim = x_train.shape[-1]
    model = ShallowRecurrentDecoder(
        latent_dim=latent_dim,
        hidden_dim=cfg.hidden_dim,
        rnn_type=cfg.rnn_type,
    ).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_loader = _to_loader(x_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = _to_loader(x_val, y_val, cfg.batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}

    for _ in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                val_loss += float(loss.item())

        history["train_loss"].append(train_loss / max(len(train_loader), 1))
        history["val_loss"].append(val_loss / max(len(val_loader), 1))

    return model, history


@torch.no_grad()
def mse_on_windows(model: ShallowRecurrentDecoder, x: np.ndarray, y: np.ndarray, device: str) -> float:
    model.eval()
    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)
    pred = model(x_t)
    return float(torch.mean((pred - y_t) ** 2).item())

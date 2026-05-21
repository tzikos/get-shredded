from __future__ import annotations

from copy import deepcopy
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class TimeSeriesDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.X = X
        self.Y = Y
        self.len = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self) -> int:
        return self.len


class SHRED(nn.Module):
    """SHallow REcurrent Decoder: stacked LSTM over a trajectory of sensor
    measurements followed by a 3-layer fully-connected decoder that maps the
    final hidden state to the full high-dimensional state.

    Mirrors the architecture from Williams, Zahn, Kutz (2024).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        hidden_layers: int = 2,
        l1: int = 350,
        l2: int = 400,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hidden_size, l1)
        self.linear2 = nn.Linear(l1, l2)
        self.linear3 = nn.Linear(l2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        h_0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_size, device=device)
        c_0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_size, device=device)

        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)

        out = torch.relu(self.dropout(self.linear1(h_out)))
        out = torch.relu(self.dropout(self.linear2(out)))
        return self.linear3(out)

    def get_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final LSTM hidden state (B, hidden_size) without SDN decoding."""
        device = next(self.parameters()).device
        h_0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_size, device=device)
        c_0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_size, device=device)
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        return h_out[-1].view(-1, self.hidden_size)


class SDN(nn.Module):
    """Static Shallow Decoder Network: 3-layer FC mapping a single snapshot of
    sensor measurements to the full state. Used as a non-temporal ablation."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        l1: int = 350,
        l2: int = 400,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, l1)
        self.linear2 = nn.Linear(l1, l2)
        self.linear3 = nn.Linear(l2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.dropout(self.linear1(x)))
        out = torch.relu(self.dropout(self.linear2(out)))
        return self.linear3(out)


def fit(
    model: nn.Module,
    train_dataset: TimeSeriesDataset,
    valid_dataset: TimeSeriesDataset,
    batch_size: int = 64,
    num_epochs: int = 4000,
    lr: float = 1e-3,
    verbose: bool = False,
    patience: int = 5,
    augment_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Training loop matching the paper: Adam + MSE, validation on relative
    L2 error every 20 epochs, early stopping on patience, restores best params.

    augment_fn is applied to each training batch only (not validation/test).
    """
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_error_list: list[torch.Tensor] = []
    patience_counter = 0
    best_params = deepcopy(model.state_dict())

    pbar = tqdm(
        range(1, num_epochs + 1),
        desc=f"train {model.__class__.__name__}",
        disable=not verbose,
        leave=False,
    )
    last_val = float("nan")
    best_so_far = float("nan")
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            if augment_fn is not None:
                batch_x = augment_fn(batch_x)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        pbar.set_postfix(loss=f"{train_loss:.4e}", val=f"{last_val:.4f}",
                         best=f"{best_so_far:.4f}", patience=f"{patience_counter}/{patience}")

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(valid_dataset.X)
                val_error = torch.linalg.norm(val_outputs - valid_dataset.Y) / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error.detach().cpu())

            last_val = val_error_list[-1].item()
            best_so_far = float(torch.min(torch.tensor(val_error_list)))
            pbar.set_postfix(loss=f"{train_loss:.4e}", val=f"{last_val:.4f}",
                             best=f"{best_so_far:.4f}", patience=f"{patience_counter}/{patience}")

            if val_error_list[-1] == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
                best_params = deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter == patience:
                pbar.close()
                model.load_state_dict(best_params)
                return torch.tensor(val_error_list)

    pbar.close()
    model.load_state_dict(best_params)
    return torch.tensor(val_error_list)


def forecast(forecaster: SHRED, reconstructor: SHRED, test_dataset: TimeSeriesDataset):
    """Two-step forecasting (paper §IV.B): an LSTM forecasts sensor
    trajectories autoregressively, then SHRED reconstructs the high-dimensional
    state from the forecasted sensor windows."""
    initial_in = test_dataset.X[0:1].clone()
    vals = [initial_in[0, i, :].detach().cpu().clone().numpy() for i in range(test_dataset.X.shape[1])]

    for _ in range(len(test_dataset.X)):
        scaled_output = forecaster(initial_in).detach().cpu().numpy()
        vals.append(scaled_output.reshape(test_dataset.X.shape[2]))
        temp = initial_in.clone()
        initial_in[0, :-1] = temp[0, 1:]
        initial_in[0, -1] = torch.tensor(scaled_output, device=initial_in.device)

    import numpy as np
    device = next(reconstructor.parameters()).device
    forecasted_vals = torch.tensor(np.array(vals), dtype=torch.float32).to(device)
    reconstructions = []
    seq_len = test_dataset.X.shape[1]
    n_sensors = test_dataset.X.shape[2]
    for i in range(len(forecasted_vals) - seq_len):
        recon = reconstructor(
            forecasted_vals[i : i + seq_len].reshape(1, seq_len, n_sensors)
        ).detach().cpu().numpy()
        reconstructions.append(recon)
    return forecasted_vals, np.array(reconstructions)

from __future__ import annotations

import torch
from torch import nn


class ShallowRecurrentDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        rnn_type: str,
    ) -> None:
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=latent_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=latent_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        self.readout = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x_seq)
        return self.readout(rnn_out[:, -1, :])

    @torch.no_grad()
    def rollout(self, seed_sequence: torch.Tensor, horizon: int) -> torch.Tensor:
        # seed_sequence: (1, seq_len, latent_dim)
        outputs = []
        current = seed_sequence.clone()
        for _ in range(horizon):
            next_step = self.forward(current)
            outputs.append(next_step)
            current = torch.cat([current[:, 1:, :], next_step.unsqueeze(1)], dim=1)
        return torch.cat(outputs, dim=0)

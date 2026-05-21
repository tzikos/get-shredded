from __future__ import annotations

import torch
import torch.nn as nn


class SensorGatedSHRED(nn.Module):
    """SHRED with a lightweight per-sensor input gate.

    At each timestep a two-layer FC gate maps the raw sensor readings to
    per-sensor scalar weights in (0, 1). Weights are multiplied onto the
    sensor values before the LSTM, allowing the model to learn to suppress
    dead or noisy channels from the dropout augmentation signal alone.

    The LSTM and decoder are architecturally identical to vanilla SHRED and
    can be warm-started from a pretrained vanilla checkpoint.
    """

    def __init__(
        self,
        num_sensors: int,
        output_size: int,
        hidden_size: int = 64,
        hidden_layers: int = 2,
        l1: int = 350,
        l2: int = 400,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        self.gate = nn.Sequential(
            nn.Linear(num_sensors, num_sensors * 4),
            nn.ReLU(),
            nn.Linear(num_sensors * 4, num_sensors),
            nn.Sigmoid(),
        )

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hidden_size, l1)
        self.linear2 = nn.Linear(l1, l2)
        self.linear3 = nn.Linear(l2, output_size)
        self.dropout_layer = nn.Dropout(dropout)

    def _apply_gate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply gate per timestep. Returns (gated_x, gate_weights) both (B, lags, p)."""
        B, lags, p = x.shape
        w = self.gate(x.reshape(B * lags, p)).reshape(B, lags, p)
        return x * w, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated, _ = self._apply_gate(x)
        device = next(self.parameters()).device
        h_0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_size, device=device)
        c_0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_size, device=device)
        _, (h_out, _) = self.lstm(gated, (h_0, c_0))
        h = h_out[-1].view(-1, self.hidden_size)
        out = torch.relu(self.dropout_layer(self.linear1(h)))
        out = torch.relu(self.dropout_layer(self.linear2(out)))
        return self.linear3(out)

    def get_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final LSTM hidden state (B, hidden_size) after gating."""
        gated, _ = self._apply_gate(x)
        device = next(self.parameters()).device
        h_0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_size, device=device)
        c_0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_size, device=device)
        _, (h_out, _) = self.lstm(gated, (h_0, c_0))
        return h_out[-1].view(-1, self.hidden_size)

    def get_trust_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Mean gate weight across lags, shape (B, p). Higher = sensor trusted more."""
        with torch.no_grad():
            _, w = self._apply_gate(x)
        return w.mean(dim=1)

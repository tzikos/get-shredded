from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorEncoder(nn.Module):
    """Encodes one sensor reading at one timestep into a (mu, sigma) pair.

    Weights are shared across all sensor channels — the same network processes
    every channel. Input is the scalar reading concatenated with the previous
    LSTM hidden state so the encoder can adapt its uncertainty estimate based on
    temporal context.
    """

    def __init__(self, d_h: int, d_z: int) -> None:
        super().__init__()
        hidden_dim = max(16, d_z * 4)
        mid_dim = max(16, d_z * 2)
        self.net = nn.Sequential(
            nn.Linear(1 + d_h, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, d_z * 2),
        )
        self.d_z = d_z

    def forward(
        self, y_i: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([y_i, h_prev], dim=1)  # (B, 1+d_h)
        out = self.net(inp)                     # (B, d_z*2)
        mu = out[:, : self.d_z]
        log_sigma = out[:, self.d_z :]
        sigma = torch.exp(log_sigma).clamp(1e-4, 10.0)
        return mu, sigma


class CrossSensorAttention(nn.Module):
    """Computes cross-sensor attention purely from latent mu/sigma vectors.

    No learnable parameters — attention weights are derived from the dot-product
    similarity of mu vectors scaled by the product of per-sensor uncertainties.
    """

    def __init__(self, d_z: int) -> None:
        super().__init__()
        self.d_z = d_z

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # mu, sigma: (B, p, d_z)
        B, p, d_z = mu.shape

        # Scalar uncertainty per sensor: mean over d_z
        u = sigma.mean(dim=-1)  # (B, p)

        # Raw attention scores: dot-product similarity scaled by joint uncertainty
        dots = torch.bmm(mu, mu.transpose(1, 2))  # (B, p, p)
        u_outer = u.unsqueeze(2) * u.unsqueeze(1) + 1e-8  # (B, p, p)
        E = dots / (u_outer * math.sqrt(d_z))
        E = E.clamp(-50.0, 50.0)

        A = F.softmax(E, dim=2)           # (B, p, p)
        context = torch.bmm(A, mu)        # (B, p, d_z)
        return context, A


class SensorGating(nn.Module):
    """Per-sensor gate that blends the sensor's own encoding with the attended
    context. High uncertainty (large sigma) drives the gate toward context."""

    def __init__(self, d_z: int) -> None:
        super().__init__()
        self.W_gate = nn.Linear(d_z, d_z, bias=True)
        # Initialise so that small sigma -> gate near 1 (trust own encoding)
        nn.init.constant_(self.W_gate.weight, -1.0)
        nn.init.constant_(self.W_gate.bias, 2.0)

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        # All inputs: (B, p, d_z)
        gate = torch.sigmoid(self.W_gate(sigma))       # (B, p, d_z)
        return gate * mu + (1.0 - gate) * context      # (B, p, d_z)


class RobustSHRED(nn.Module):
    """SHRED with per-sensor uncertainty encoding and cross-sensor attention.

    At each timestep the model:
    1. Encodes every sensor reading into a (mu, sigma) pair conditioned on the
       previous LSTM hidden state.
    2. Runs cross-sensor attention to produce a context vector per sensor.
    3. Gates each sensor's own encoding against the context based on uncertainty.
    4. Projects the gated output back to p dimensions and feeds into the LSTM.

    The SDN decoder on top of the final hidden state is identical to vanilla SHRED.
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
        d_z: int = 8,
    ) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.d_z = d_z

        self.encoder = SensorEncoder(hidden_size, d_z)
        self.attention = CrossSensorAttention(d_z)
        self.gating = SensorGating(d_z)
        self.proj = nn.Linear(num_sensors * d_z, num_sensors)

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hidden_size, l1)
        self.linear2 = nn.Linear(l1, l2)
        self.linear3 = nn.Linear(l2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_internals: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        # x: (B, lags, p)
        B, lags, p = x.shape
        device = x.device

        h_state = torch.zeros(self.hidden_layers, B, self.hidden_size, device=device)
        c_state = torch.zeros(self.hidden_layers, B, self.hidden_size, device=device)
        h_prev = h_state[-1]  # (B, hidden_size) — last-layer hidden state for encoder

        all_mu: list[torch.Tensor] = []
        all_sigma: list[torch.Tensor] = []
        all_attn: list[torch.Tensor] = []

        for t in range(lags):
            v_t = x[:, t, :]  # (B, p)

            mu_list: list[torch.Tensor] = []
            sigma_list: list[torch.Tensor] = []
            for i in range(p):
                y_i = v_t[:, i : i + 1]  # (B, 1)
                mu_i, sigma_i = self.encoder(y_i, h_prev)
                mu_list.append(mu_i)
                sigma_list.append(sigma_i)

            mu_t = torch.stack(mu_list, dim=1)      # (B, p, d_z)
            sigma_t = torch.stack(sigma_list, dim=1)  # (B, p, d_z)

            context_t, attn_t = self.attention(mu_t, sigma_t)  # (B,p,d_z), (B,p,p)
            output_t = self.gating(mu_t, sigma_t, context_t)   # (B, p, d_z)

            z_t = self.proj(output_t.reshape(B, p * self.d_z))  # (B, p)

            all_mu.append(mu_t)
            all_sigma.append(sigma_t)
            all_attn.append(attn_t)

            _, (h_state, c_state) = self.lstm(z_t.unsqueeze(1), (h_state, c_state))
            h_prev = h_state[-1]  # (B, hidden_size)

        out = torch.relu(self.dropout(self.linear1(h_prev)))
        out = torch.relu(self.dropout(self.linear2(out)))
        recon = self.linear3(out)  # (B, output_size)

        if not return_internals:
            return recon

        internals = {
            "mu":      torch.stack(all_mu, dim=1),      # (B, lags, p, d_z)
            "sigma":   torch.stack(all_sigma, dim=1),   # (B, lags, p, d_z)
            "attn":    torch.stack(all_attn, dim=1),    # (B, lags, p, p)
            "h_final": h_prev,                          # (B, hidden_size)
        }
        return recon, internals

    def get_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Return final LSTM hidden state (B, hidden_size) without SDN decoding."""
        B, lags, p = x.shape
        device = x.device

        h_state = torch.zeros(self.hidden_layers, B, self.hidden_size, device=device)
        c_state = torch.zeros(self.hidden_layers, B, self.hidden_size, device=device)
        h_prev = h_state[-1]

        for t in range(lags):
            v_t = x[:, t, :]
            mu_list: list[torch.Tensor] = []
            sigma_list: list[torch.Tensor] = []
            for i in range(p):
                mu_i, sigma_i = self.encoder(v_t[:, i : i + 1], h_prev)
                mu_list.append(mu_i)
                sigma_list.append(sigma_i)

            mu_t = torch.stack(mu_list, dim=1)
            sigma_t = torch.stack(sigma_list, dim=1)
            context_t, _ = self.attention(mu_t, sigma_t)
            output_t = self.gating(mu_t, sigma_t, context_t)
            z_t = self.proj(output_t.reshape(B, p * self.d_z))

            _, (h_state, c_state) = self.lstm(z_t.unsqueeze(1), (h_state, c_state))
            h_prev = h_state[-1]

        return h_prev

    def get_trust_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sensor trust weights derived from cross-sensor attention.

        Returns shape (B, p). Higher weight = sensor j is heavily attended to
        by the other sensors, meaning the model treats it as informative.

        Computed as the column sums of the attention matrix averaged over lags,
        then normalised to sum to 1.  Column sum A[:, :, j].sum(dim=1) measures
        how much sensor j is drawn upon by all sensors — the operative quantity
        that actually drives the model's behaviour, as opposed to σ which
        collapsed to the prior under KL regularisation.
        """
        _, internals = self.forward(x, return_internals=True)
        attn = internals["attn"]                    # (B, lags, p, p)
        attn_mean = attn.mean(dim=1)                # (B, p, p)  average over lags
        col_sums = attn_mean.sum(dim=1)             # (B, p)  sum over attending sensors
        return col_sums / col_sums.sum(dim=-1, keepdim=True)  # normalise → sums to 1

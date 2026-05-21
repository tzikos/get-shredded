from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import qr_place, qrpod_reconstruct


@dataclass
class QRPODBaseline:
    """Linear gappy-POD reconstruction with QR-pivoted sensor placement
    (the linear baseline from Williams, Zahn, Kutz 2024)."""

    num_sensors: int
    sensor_locations: np.ndarray | None = None
    U_r: np.ndarray | None = None
    m: int | None = None

    def fit(self, train_X: np.ndarray) -> None:
        """train_X: (N_train, m). Computes POD basis and QR sensor locations."""
        self.m = train_X.shape[1]
        self.sensor_locations, self.U_r = qr_place(train_X.T, self.num_sensors)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (T, m) ground-truth snapshots; we sample at the QR locations and
        reconstruct via x_hat = U_r (C U_r)^{-1} y."""
        if self.U_r is None or self.sensor_locations is None or self.m is None:
            raise RuntimeError("Call fit() before predict()")
        sensor_measurements = X[:, self.sensor_locations]
        return qrpod_reconstruct(sensor_measurements, self.sensor_locations, self.U_r, self.m)

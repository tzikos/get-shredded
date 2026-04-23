"""Backwards-compatible re-exports. Training utilities live in `model.py`."""
from .model import SHRED, SDN, TimeSeriesDataset, fit, forecast

__all__ = ["SHRED", "SDN", "TimeSeriesDataset", "fit", "forecast"]

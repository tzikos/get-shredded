"""Cross-placement evaluation: train SHRED on one sensor placement, evaluate
on another.  Shows whether the trained weights generalise across sensor positions.

Conditions (3 sensors, fixed seed):
  A) QR-trained  → tested on QR  locations  (matched, expected best)
  B) RND-trained → tested on RND locations  (matched, fair random baseline)
  C) QR-trained  → tested on RND locations  (mismatched: architecture mismatch)
  D) RND-trained → tested on QR  locations  (mismatched: architecture mismatch)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from get_shredded.data import build_sensor_windows, load_cylinder_data, qr_place
from get_shredded.model import SHRED, TimeSeriesDataset, fit

# ── config ────────────────────────────────────────────────────────────────────
MAT = ROOT / "data" / "CYLINDER_ALL.mat"
NUM_SENSORS = 3
LAGS        = 10
HIDDEN_SIZE = 64
HIDDEN_LAYERS = 2
L1, L2      = 350, 400
EPOCHS      = 1000
PATIENCE    = 5
BATCH_SIZE  = 64
LR          = 1e-3
TEST_SIZE   = 10
VAL_SIZE    = 20
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(SEED)
torch.manual_seed(SEED)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
print(f"device: {device}")

load_X, nx, ny = load_cylinder_data(MAT)
n, m = load_X.shape

n_windows = n - LAGS
train_end = n_windows - TEST_SIZE - VAL_SIZE
val_end   = n_windows - TEST_SIZE
train_idx = np.arange(0, train_end)
val_idx   = np.arange(train_end, val_end)
test_idx  = np.arange(val_end, n_windows)

# sensor placements
qr_locs, _  = qr_place(load_X[train_idx].T, NUM_SENSORS)
rng          = np.random.default_rng(SEED + 1)
rnd_locs     = rng.choice(m, size=NUM_SENSORS, replace=False)

print(f"QR  sensor locations : {qr_locs}")
print(f"RND sensor locations : {rnd_locs}")

sc = MinMaxScaler().fit(load_X[train_idx])
tX = sc.transform(load_X).astype(np.float32)

def make_tensors(sensor_locs):
    windows = build_sensor_windows(tX, sensor_locs, LAGS)
    tr = torch.tensor(windows[train_idx], dtype=torch.float32, device=device)
    va = torch.tensor(windows[val_idx],   dtype=torch.float32, device=device)
    te = torch.tensor(windows[test_idx],  dtype=torch.float32, device=device)
    tr_y = torch.tensor(tX[train_idx + LAGS - 1], dtype=torch.float32, device=device)
    va_y = torch.tensor(tX[val_idx   + LAGS - 1], dtype=torch.float32, device=device)
    te_y = torch.tensor(tX[test_idx  + LAGS - 1], dtype=torch.float32, device=device)
    return (
        TimeSeriesDataset(tr, tr_y),
        TimeSeriesDataset(va, va_y),
        TimeSeriesDataset(te, te_y),
    )

def rel_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))

def train_shred(train_ds, val_ds, label: str) -> SHRED:
    model = SHRED(NUM_SENSORS, m, hidden_size=HIDDEN_SIZE, hidden_layers=HIDDEN_LAYERS,
                  l1=L1, l2=L2).to(device)
    print(f"\nTraining SHRED-{label} ...")
    fit(model, train_ds, val_ds, batch_size=BATCH_SIZE, num_epochs=EPOCHS,
        lr=LR, verbose=True, patience=PATIENCE)
    model.eval()
    return model

def evaluate(model: SHRED, test_ds: TimeSeriesDataset) -> float:
    with torch.no_grad():
        pred = sc.inverse_transform(model(test_ds.X).cpu().numpy())
    truth = sc.inverse_transform(test_ds.Y.cpu().numpy())
    return rel_l2(pred, truth)

# build datasets for each placement
qr_tr, qr_va, qr_te = make_tensors(qr_locs)
rnd_tr, rnd_va, rnd_te = make_tensors(rnd_locs)

# train both models
shred_qr  = train_shred(qr_tr,  qr_va,  "QR")
shred_rnd = train_shred(rnd_tr, rnd_va, "RND")

# ── evaluation matrix ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Cross-placement evaluation  (3 sensors, relative L2 error)")
print("=" * 60)
print(f"{'Condition':<45}  {'Rel L2':>8}")
print("-" * 55)

err_AA = evaluate(shred_qr,  qr_te)
err_BB = evaluate(shred_rnd, rnd_te)
err_AB = evaluate(shred_qr,  rnd_te)   # mismatched
err_BA = evaluate(shred_rnd, qr_te)    # mismatched

rows = [
    ("A) QR-trained  → test on QR  (matched)",     err_AA),
    ("B) RND-trained → test on RND (matched)",     err_BB),
    ("C) QR-trained  → test on RND (MISMATCHED)",  err_AB),
    ("D) RND-trained → test on QR  (MISMATCHED)",  err_BA),
]
for label, err in rows:
    print(f"  {label:<43}  {err:>8.4f}")

print("=" * 60)
print("\nInterpretation:")
print("  Matched rows (A, B)   → fair same-placement comparison")
print("  Mismatched rows (C, D) → feeding test sensors the model")
print("  never saw during training → expected degradation")

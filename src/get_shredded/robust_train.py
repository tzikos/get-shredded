from __future__ import annotations

from copy import deepcopy
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .model import SHRED, TimeSeriesDataset, fit
from .robust_model import SensorGatedSHRED


def _shred_hidden(shred: SHRED, x: torch.Tensor) -> torch.Tensor:
    device = next(shred.parameters()).device
    x = x.to(device)
    h_0 = torch.zeros(shred.hidden_layers, x.size(0), shred.hidden_size, device=device)
    c_0 = torch.zeros(shred.hidden_layers, x.size(0), shred.hidden_size, device=device)
    _, (h_out, _) = shred.lstm(x, (h_0, c_0))
    return h_out[-1].view(-1, shred.hidden_size)


def fit_robust(
    gated_shred: SensorGatedSHRED,
    vanilla_shred: SHRED,
    train_dataset: TimeSeriesDataset,
    valid_dataset: TimeSeriesDataset,
    *,
    batch_size: int = 64,
    phase1_epochs: int = 1000,
    phase1_patience: int = 5,
    phase1_lr: float = 1e-3,
    phase2_epochs: int = 500,
    phase2_patience: int = 5,
    phase2_lr: float = 1e-3,
    phase3_epochs: int = 200,
    phase3_patience: int = 5,
    phase3_lr: float = 1e-4,
    augment_fn: Callable | None = None,
    verbose: bool = True,
) -> dict[str, torch.Tensor]:
    """Three-phase teacher-student training for SensorGatedSHRED.

    Phase 1 — pretrain vanilla SHRED on clean data; warm-start gated model.
    Phase 2 — train gate + LSTM to match teacher hidden states under augmentation;
               decoder is frozen to keep the representation tied to clean outputs.
    Phase 3 — fine-tune LSTM + decoder jointly on reconstruction loss.
               Gate is frozen so the learned reliability weights are not disturbed.
    """
    device = train_dataset.X.device

    # ── Phase 1: Pretrain vanilla SHRED ──────────────────────────────────────
    if verbose:
        print("=== Phase 1: Pretrain vanilla SHRED ===")
    phase1_hist = fit(
        vanilla_shred, train_dataset, valid_dataset,
        batch_size=batch_size, num_epochs=phase1_epochs,
        lr=phase1_lr, patience=phase1_patience, verbose=verbose,
    )

    # Warm-start gated model LSTM + decoder from the trained teacher
    gated_shred.lstm.load_state_dict(vanilla_shred.lstm.state_dict())
    gated_shred.linear1.load_state_dict(vanilla_shred.linear1.state_dict())
    gated_shred.linear2.load_state_dict(vanilla_shred.linear2.state_dict())
    gated_shred.linear3.load_state_dict(vanilla_shred.linear3.state_dict())

    # Cache clean teacher hidden states for every training sample
    vanilla_shred.eval()
    all_h: list[torch.Tensor] = []
    with torch.no_grad():
        for bx, _ in DataLoader(train_dataset, batch_size=batch_size, shuffle=False):
            all_h.append(_shred_hidden(vanilla_shred, bx).cpu())
    cached_h = torch.cat(all_h, dim=0)  # (N_train, hidden_size)

    # ── Phase 2: Hidden-state matching under augmentation ────────────────────
    if verbose:
        print("=== Phase 2: Train gate+LSTM via hidden-state matching ===")

    for name, p in gated_shred.named_parameters():
        if name.startswith(("linear1", "linear2", "linear3")):
            p.requires_grad_(False)

    opt2 = torch.optim.Adam(
        [p for p in gated_shred.parameters() if p.requires_grad], lr=phase2_lr
    )
    loader2 = DataLoader(
        TensorDataset(train_dataset.X, train_dataset.Y, cached_h.to(device)),
        batch_size=batch_size, shuffle=True,
    )

    p2_vals: list[torch.Tensor] = []
    p2_patience_ctr = 0
    best2 = deepcopy(gated_shred.state_dict())

    pbar2 = tqdm(range(1, phase2_epochs + 1), desc="Phase 2", disable=not verbose, leave=False)
    last_val, best_so_far = float("nan"), float("nan")
    for epoch in pbar2:
        gated_shred.train()
        for bx, _, bh_teacher in loader2:
            bx_aug = augment_fn(bx) if augment_fn is not None else bx
            loss = F.mse_loss(gated_shred.get_hidden_state(bx_aug), bh_teacher.to(device))
            opt2.zero_grad()
            loss.backward()
            opt2.step()

        if epoch % 20 == 0 or epoch == 1:
            gated_shred.eval()
            with torch.no_grad():
                h_s = gated_shred.get_hidden_state(valid_dataset.X)
                h_t = _shred_hidden(vanilla_shred, valid_dataset.X)
                val_loss = F.mse_loss(h_s, h_t.detach())
                p2_vals.append(val_loss.cpu())

            last_val = p2_vals[-1].item()
            best_so_far = float(torch.min(torch.tensor(p2_vals)))
            pbar2.set_postfix(val=f"{last_val:.4e}", best=f"{best_so_far:.4e}",
                              patience=f"{p2_patience_ctr}/{phase2_patience}")

            if p2_vals[-1] == torch.min(torch.tensor(p2_vals)):
                p2_patience_ctr = 0
                best2 = deepcopy(gated_shred.state_dict())
            else:
                p2_patience_ctr += 1
            if p2_patience_ctr == phase2_patience:
                break

    pbar2.close()
    gated_shred.load_state_dict(best2)

    for p in gated_shred.parameters():
        p.requires_grad_(True)

    # ── Phase 3: Fine-tune LSTM + decoder on reconstruction ──────────────────
    if verbose:
        print("=== Phase 3: Fine-tune LSTM+decoder on reconstruction loss ===")

    for name, p in gated_shred.named_parameters():
        if name.startswith("gate"):
            p.requires_grad_(False)

    opt3 = torch.optim.Adam(
        [p for p in gated_shred.parameters() if p.requires_grad], lr=phase3_lr
    )
    criterion = nn.MSELoss()
    loader3 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    p3_vals: list[torch.Tensor] = []
    p3_patience_ctr = 0
    best3 = deepcopy(gated_shred.state_dict())

    pbar3 = tqdm(range(1, phase3_epochs + 1), desc="Phase 3", disable=not verbose, leave=False)
    last_val3, best_so_far3 = float("nan"), float("nan")
    for epoch in pbar3:
        gated_shred.train()
        epoch_loss, n_batches = 0.0, 0
        for bx, by in loader3:
            bx_aug = augment_fn(bx) if augment_fn is not None else bx
            opt3.zero_grad()
            loss = criterion(gated_shred(bx_aug), by)
            loss.backward()
            opt3.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            gated_shred.eval()
            with torch.no_grad():
                val_err = (
                    torch.linalg.norm(gated_shred(valid_dataset.X) - valid_dataset.Y)
                    / torch.linalg.norm(valid_dataset.Y)
                )
                p3_vals.append(val_err.cpu())

            last_val3 = p3_vals[-1].item()
            best_so_far3 = float(torch.min(torch.tensor(p3_vals)))
            pbar3.set_postfix(val=f"{last_val3:.4f}", best=f"{best_so_far3:.4f}",
                              loss=f"{epoch_loss/max(n_batches,1):.4e}",
                              patience=f"{p3_patience_ctr}/{phase3_patience}")

            if p3_vals[-1] == torch.min(torch.tensor(p3_vals)):
                p3_patience_ctr = 0
                best3 = deepcopy(gated_shred.state_dict())
            else:
                p3_patience_ctr += 1
            if p3_patience_ctr == phase3_patience:
                break

    pbar3.close()
    gated_shred.load_state_dict(best3)

    for p in gated_shred.parameters():
        p.requires_grad_(True)

    return {
        "phase1_val_history": phase1_hist,
        "phase2_val_history": torch.tensor(p2_vals) if p2_vals else torch.tensor([float("nan")]),
        "phase3_val_history": torch.tensor(p3_vals) if p3_vals else torch.tensor([float("nan")]),
    }

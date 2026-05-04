from __future__ import annotations

from copy import deepcopy
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .model import SHRED, TimeSeriesDataset, fit
from .robust_model import RobustSHRED


def _extract_shred_hidden(shred: SHRED, x: torch.Tensor) -> torch.Tensor:
    """Run SHRED's LSTM and return the final hidden state (B, hidden_size)."""
    device = next(shred.parameters()).device
    x = x.to(device)
    h_0 = torch.zeros(shred.hidden_layers, x.size(0), shred.hidden_size, device=device)
    c_0 = torch.zeros(shred.hidden_layers, x.size(0), shred.hidden_size, device=device)
    _, (h_out, _) = shred.lstm(x, (h_0, c_0))
    return h_out[-1].view(-1, shred.hidden_size)


def pretrain_shred_clean(
    shred: SHRED,
    train_dataset: TimeSeriesDataset,
    valid_dataset: TimeSeriesDataset,
    *,
    batch_size: int = 64,
    epochs: int = 1000,
    lr: float = 1e-3,
    patience: int = 5,
    verbose: bool = True,
) -> tuple[SHRED, torch.Tensor, torch.Tensor]:
    """Train vanilla SHRED on clean data, then cache hidden states for all training samples.

    Returns (trained_shred, cached_hidden, val_history).
    cached_hidden has shape (N_train, hidden_size) and is always on CPU with no grad.
    """
    val_history = fit(
        shred, train_dataset, valid_dataset,
        batch_size=batch_size, num_epochs=epochs, lr=lr,
        verbose=verbose, patience=patience,
    )

    shred.eval()
    all_hidden: list[torch.Tensor] = []
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x, _ in loader:
            h = _extract_shred_hidden(shred, batch_x)
            all_hidden.append(h.detach().cpu())
    cached_hidden = torch.cat(all_hidden, dim=0)  # (N_train, hidden_size)

    return shred, cached_hidden, val_history


def train_robust_encoder(
    robust_shred: RobustSHRED,
    vanilla_shred: SHRED,
    cached_hidden: torch.Tensor,
    train_dataset: TimeSeriesDataset,
    valid_dataset: TimeSeriesDataset,
    *,
    batch_size: int = 64,
    epochs: int = 500,
    lr: float = 1e-3,
    patience: int = 5,
    beta_kl: float = 0.0,
    lambda_detect: float = 0.1,
    augment_fn: Callable | None = None,
    corrupt_labels_fn: Callable | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Phase 2: train encoder-attention via teacher-student hidden-state matching.

    Freezes the SDN decoder layers; trains encoder, gating, proj, and LSTM.
    Warm-starts LSTM weights from the trained vanilla SHRED teacher.
    """
    # Warm-start the student LSTM from the teacher
    robust_shred.lstm.load_state_dict(vanilla_shred.lstm.state_dict())

    # Freeze SDN decoder
    for name, param in robust_shred.named_parameters():
        if name.startswith(("linear1", "linear2", "linear3")):
            param.requires_grad_(False)

    trainable = [p for p in robust_shred.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    device = train_dataset.X.device
    combined_train = TensorDataset(
        train_dataset.X,
        train_dataset.Y,
        cached_hidden.to(device),
    )
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True)

    val_error_list: list[torch.Tensor] = []
    patience_counter = 0
    best_params = deepcopy(robust_shred.state_dict())

    pbar = tqdm(
        range(1, epochs + 1),
        desc="Phase 2 encoder-attention",
        disable=not verbose,
        leave=False,
    )
    last_val = float("nan")
    best_so_far = float("nan")

    for epoch in pbar:
        robust_shred.train()
        for batch_x, _batch_y, batch_h_teacher in train_loader:
            batch_x_aug = augment_fn(batch_x) if augment_fn is not None else batch_x

            _, internals = robust_shred(batch_x_aug, return_internals=True)
            h_student = internals["h_final"]        # (B, hidden_size)
            mu = internals["mu"]                    # (B, lags, p, d_z)
            sigma = internals["sigma"]              # (B, lags, p, d_z)
            attn = internals["attn"]                # (B, lags, p, p)

            L_hidden = F.mse_loss(h_student, batch_h_teacher.to(device))

            kl = 0.5 * (mu.pow(2) + sigma.pow(2) - sigma.pow(2).log() - 1.0)
            L_KL = kl.mean()

            if corrupt_labels_fn is not None:
                g_target = corrupt_labels_fn(batch_x, batch_x_aug)  # (B, p), 1=clean 0=corrupted
                attn_mean = attn.mean(dim=1)                          # (B, p, p)
                trust = attn_mean.sum(dim=1)                          # (B, p) column sums
                trust = trust / (trust.sum(dim=1, keepdim=True) + 1e-8)
                L_detect = F.binary_cross_entropy(
                    trust.clamp(1e-6, 1.0 - 1e-6), g_target.float().to(device)
                )
            else:
                L_detect = 0.0

            loss = L_hidden + beta_kl * L_KL + lambda_detect * L_detect

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            robust_shred.eval()
            with torch.no_grad():
                _, val_internals = robust_shred(valid_dataset.X, return_internals=True)
                h_val = val_internals["h_final"]
                # Build val teacher hidden states on the fly from vanilla_shred
                h_teacher_val = _extract_shred_hidden(vanilla_shred, valid_dataset.X)
                val_loss = F.mse_loss(h_val, h_teacher_val.detach())
                val_error_list.append(val_loss.detach().cpu())

            last_val = val_error_list[-1].item()
            best_so_far = float(torch.min(torch.tensor(val_error_list)))
            pbar.set_postfix(val_L_hidden=f"{last_val:.4e}", best=f"{best_so_far:.4e}",
                             patience=f"{patience_counter}/{patience}")

            if val_error_list[-1] == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
                best_params = deepcopy(robust_shred.state_dict())
            else:
                patience_counter += 1

            if patience_counter == patience:
                pbar.close()
                robust_shred.load_state_dict(best_params)
                break

    pbar.close()
    robust_shred.load_state_dict(best_params)

    # Re-enable SDN gradients for subsequent phases
    for name, param in robust_shred.named_parameters():
        if name.startswith(("linear1", "linear2", "linear3")):
            param.requires_grad_(True)

    return torch.tensor(val_error_list) if val_error_list else torch.tensor([float("nan")])


def finetune_sdn(
    robust_shred: RobustSHRED,
    train_dataset: TimeSeriesDataset,
    valid_dataset: TimeSeriesDataset,
    *,
    batch_size: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 5,
    augment_fn: Callable | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Phase 3: fine-tune LSTM + SDN decoder jointly on reconstruction loss.

    Freezes only the robustness-specific components (encoder, attention, gating,
    proj) so the LSTM can re-couple with the SDN after Phase 2 shifted its
    hidden-state distribution.  Both LSTM and SDN layers receive lr; the
    caller may use separate param groups if finer control is needed.
    """
    for name, param in robust_shred.named_parameters():
        if name.startswith(("encoder", "attention", "gating", "proj")):
            param.requires_grad_(False)

    trainable = [p for p in robust_shred.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_error_list: list[torch.Tensor] = []
    patience_counter = 0
    best_params = deepcopy(robust_shred.state_dict())

    pbar = tqdm(
        range(1, epochs + 1),
        desc="Phase 3 fine-tune SDN",
        disable=not verbose,
        leave=False,
    )
    last_val = float("nan")
    best_so_far = float("nan")

    for epoch in pbar:
        robust_shred.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x_aug = augment_fn(batch_x) if augment_fn is not None else batch_x
            optimizer.zero_grad()
            loss = criterion(robust_shred(batch_x_aug), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        pbar.set_postfix(loss=f"{train_loss:.4e}", val=f"{last_val:.4f}",
                         best=f"{best_so_far:.4f}", patience=f"{patience_counter}/{patience}")

        if epoch % 10 == 0 or epoch == 1:
            robust_shred.eval()
            with torch.no_grad():
                val_out = robust_shred(valid_dataset.X)
                val_err = (
                    torch.linalg.norm(val_out - valid_dataset.Y)
                    / torch.linalg.norm(valid_dataset.Y)
                )
                val_error_list.append(val_err.detach().cpu())

            last_val = val_error_list[-1].item()
            best_so_far = float(torch.min(torch.tensor(val_error_list)))
            pbar.set_postfix(loss=f"{train_loss:.4e}", val=f"{last_val:.4f}",
                             best=f"{best_so_far:.4f}", patience=f"{patience_counter}/{patience}")

            if val_error_list[-1] == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
                best_params = deepcopy(robust_shred.state_dict())
            else:
                patience_counter += 1

            if patience_counter == patience:
                pbar.close()
                robust_shred.load_state_dict(best_params)
                break

    pbar.close()
    robust_shred.load_state_dict(best_params)

    # Re-enable all gradients
    for param in robust_shred.parameters():
        param.requires_grad_(True)

    return torch.tensor(val_error_list) if val_error_list else torch.tensor([float("nan")])


def fit_robust(
    robust_shred: RobustSHRED,
    vanilla_shred: SHRED,
    train_dataset: TimeSeriesDataset,
    valid_dataset: TimeSeriesDataset,
    *,
    batch_size: int = 64,
    # Phase 1
    phase1_epochs: int = 1000,
    phase1_lr: float = 1e-3,
    phase1_patience: int = 5,
    # Phase 2
    phase2_epochs: int = 500,
    phase2_lr: float = 1e-3,
    phase2_patience: int = 5,
    beta_kl: float = 0.0,
    lambda_detect: float = 0.1,
    # Phase 3
    phase3_epochs: int = 200,
    phase3_lr: float = 1e-4,
    phase3_patience: int = 5,
    # Augmentation
    augment_fn: Callable | None = None,
    corrupt_labels_fn: Callable | None = None,
    verbose: bool = True,
) -> dict[str, torch.Tensor]:
    """Run three-phase teacher-student training for RobustSHRED.

    Phase 1 — pretrain vanilla SHRED on clean data and cache hidden states.
    Phase 2 — train encoder+attention+gating+proj+LSTM to match teacher hidden states.
    Phase 3 — fine-tune the SDN decoder on reconstruction loss.
    """
    if verbose:
        print("=== Phase 1: Pretraining vanilla SHRED on clean data ===")
    _, cached_hidden, phase1_hist = pretrain_shred_clean(
        vanilla_shred, train_dataset, valid_dataset,
        batch_size=batch_size, epochs=phase1_epochs,
        lr=phase1_lr, patience=phase1_patience, verbose=verbose,
    )

    # Initialise RobustSHRED's SDN from the trained teacher
    robust_shred.linear1.load_state_dict(vanilla_shred.linear1.state_dict())
    robust_shred.linear2.load_state_dict(vanilla_shred.linear2.state_dict())
    robust_shred.linear3.load_state_dict(vanilla_shred.linear3.state_dict())

    if verbose:
        print("=== Phase 2: Training encoder-attention (teacher-student) ===")
    phase2_hist = train_robust_encoder(
        robust_shred, vanilla_shred, cached_hidden,
        train_dataset, valid_dataset,
        batch_size=batch_size, epochs=phase2_epochs,
        lr=phase2_lr, patience=phase2_patience,
        beta_kl=beta_kl, lambda_detect=lambda_detect,
        augment_fn=augment_fn, corrupt_labels_fn=corrupt_labels_fn,
        verbose=verbose,
    )

    if verbose:
        print("=== Phase 3: Fine-tuning SDN on reconstruction loss ===")
    phase3_hist = finetune_sdn(
        robust_shred, train_dataset, valid_dataset,
        batch_size=batch_size, epochs=phase3_epochs,
        lr=phase3_lr, patience=phase3_patience,
        augment_fn=augment_fn, verbose=verbose,
    )

    return {
        "phase1_val_history": phase1_hist,
        "phase2_val_history": phase2_hist,
        "phase3_val_history": phase3_hist,
    }

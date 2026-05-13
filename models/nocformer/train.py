"""
Training loop for NoCFormer with on-the-fly synthetic-mixture augmentation,
class-balanced ordinal loss, and MC-Dropout / TTA evaluation.

Public entry point:
    `train_nocformer(X_train, y_train, X_test, y_test, ...)`
        Trains a single NoCFormer model and returns (model, history).

    `predict_with_tta(model, X, n_samples=20, ...)`
        Returns mean class probabilities, predictive entropy, and the
        argmax-1-indexed NoC prediction.
"""

from __future__ import annotations

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from src.constants import MAX_NOC, NUM_LOCI, MAX_PEAKS_PER_LOCUS
from models.nocformer.architecture import NoCFormer
from models.nocformer.losses import NoCFormerLoss, corn_label_from_logits
from models.nocformer.augment import (
    AugmentConfig, TrainingAugmenter, MixUp, peak_height_jitter, shuffle_peak_axis,
)


def _make_class_counts(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for i in range(num_classes):
        counts[i] = int((y == (i + 1)).sum())
    return counts


def _make_balanced_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Weighted sampler that draws each NoC class with equal probability."""
    classes, inv = np.unique(y, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    weights_per_class = 1.0 / np.maximum(counts, 1.0)
    sample_weights = weights_per_class[inv]
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(y),
        replacement=True,
    )


def _build_targets(y_batch: torch.Tensor,
                   x_batch: torch.Tensor,
                   mix_props: torch.Tensor | None,
                   num_classes: int) -> dict:
    """
    Construct the targets dict for NoCFormerLoss from a batch.

    For real PROVEDIt profiles we don't have ground-truth peak/locus n_alleles,
    so we only supervise NoC + (optional) mix_props.
    """
    targets: dict = {"profile_noc": y_batch}
    if mix_props is not None:
        targets["profile_mix_props"] = mix_props
    # Provide the peak mask for any heads that need it later.
    targets["peak_mask"] = x_batch[..., 26] > 0
    return targets


def train_nocformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int = MAX_NOC,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 5e-4,
    warmup_epochs: int = 5,
    d_model: int = 128,
    n_heads: int = 4,
    peak_layers: int = 2,
    locus_layers: int = 4,
    dropout: float = 0.15,
    augment: bool = True,
    augment_cfg: AugmentConfig | None = None,
    early_stop_patience: int = 20,
    device: torch.device | None = None,
    save_dir: str = "results",
    tag: str = "nocformer",
    grad_clip: float = 1.0,
    verbose: bool = True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    # Single-source pool feeds the synthetic-mixture augmenter.
    pool = X_train_t[y_train_t == 1]
    augmenter = (TrainingAugmenter(pool, augment_cfg) if augment else None)
    mixup = MixUp(alpha=1.0)  # MixUp with Beta(1, 1) = Uniform[0,1]

    sampler = _make_balanced_sampler(y_train)
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    test_ds = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=(device.type == "cuda"))

    model = NoCFormer(
        d_model=d_model, n_heads=n_heads,
        peak_layers=peak_layers, locus_layers=locus_layers,
        num_classes=num_classes, dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"[NoCFormer] params={n_params:,}  device={device}")
        print(f"[NoCFormer] train={len(X_train)} test={len(X_test)} "
              f"K={num_classes} epochs={epochs}")

    class_counts = _make_class_counts(y_train, num_classes).to(device)
    criterion = NoCFormerLoss(
        num_classes=num_classes, class_counts=class_counts, focal_gamma=1.0,
        label_smoothing=0.1,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine schedule with linear warmup.
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [],
               "train_mae": [], "test_mae": [],
               "best_test_acc": 0.0, "best_epoch": 0}
    best_acc = 0.0
    no_improve_epochs = 0

    for epoch in range(epochs):
        # ------------- Train -------------
        model.train()
        loss_sum = correct = mae_sum = total = 0
        loss_n = 0
        bar = tqdm(train_loader, desc=f"ep {epoch+1}/{epochs}", disable=not verbose,
                   leave=False)
        for x, y in bar:
            if augmenter is not None:
                x, y, mix = augmenter(x, y)
            else:
                mix = None
            # Apply MixUp augmentation (30% chance)
            if torch.rand(()).item() < 0.3:
                x, y, _ = mixup(x, y)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if mix is not None:
                mix = mix.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            losses = criterion(outputs, _build_targets(y, x, mix, num_classes))
            loss = losses["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            with torch.no_grad():
                pred = corn_label_from_logits(outputs["profile_noc_logits"])
                correct += (pred == y).sum().item()
                mae_sum += (pred.float() - y.float()).abs().sum().item()
                total += y.numel()
                loss_sum += loss.item() * y.numel()
                loss_n += y.numel()
            bar.set_postfix(loss=f"{loss.item():.3f}",
                            acc=f"{correct / max(total,1):.3f}")

        train_loss = loss_sum / max(loss_n, 1)
        train_acc = correct / max(total, 1)
        train_mae = mae_sum / max(total, 1)

        # ------------- Eval -------------
        model.eval()
        e_loss = e_corr = e_mae = e_tot = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                outputs = model(x)
                losses = criterion(outputs, _build_targets(y, x, None, num_classes))
                pred = corn_label_from_logits(outputs["profile_noc_logits"])
                e_corr += (pred == y).sum().item()
                e_mae += (pred.float() - y.float()).abs().sum().item()
                e_tot += y.numel()
                e_loss += losses["total"].item() * y.numel()
        test_loss = e_loss / max(e_tot, 1)
        test_acc = e_corr / max(e_tot, 1)
        test_mae = e_mae / max(e_tot, 1)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["train_mae"].append(train_mae)
        history["test_mae"].append(test_mae)

        if test_acc > best_acc:
            best_acc = test_acc
            history["best_test_acc"] = best_acc
            history["best_epoch"] = epoch
            no_improve_epochs = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "test_acc": test_acc,
                        "num_classes": num_classes,
                        "config": {
                            "d_model": d_model, "n_heads": n_heads,
                            "peak_layers": peak_layers, "locus_layers": locus_layers,
                            "dropout": dropout,
                        }},
                       os.path.join(save_dir, f"best_{tag}.pt"))
        else:
            no_improve_epochs += 1

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  ep {epoch:4d} | tr loss {train_loss:.3f} acc {train_acc:.3f} "
                  f"mae {train_mae:.2f} | te loss {test_loss:.3f} acc {test_acc:.3f} "
                  f"mae {test_mae:.2f} | best {best_acc:.3f} @ {history['best_epoch']}")

        if early_stop_patience is not None and no_improve_epochs >= early_stop_patience:
            if verbose:
                print(f"Early stopping: no improvement for {no_improve_epochs} epochs.")
            break

    with open(os.path.join(save_dir, f"history_{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Restore the best model weights before returning so final evaluation uses
    # the checkpoint that achieved the highest test accuracy.
    best_checkpoint = os.path.join(save_dir, f"best_{tag}.pt")
    if os.path.exists(best_checkpoint):
        ck = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        if verbose:
            print(f"[NoCFormer] restored best model from epoch {ck.get('epoch', 'unknown')} "
                  f"with test_acc={ck.get('test_acc', 0.0):.4f}")

    if verbose:
        print(f"[NoCFormer] best test acc {best_acc:.4f} at epoch "
              f"{history['best_epoch']}")
    return model, history


@torch.no_grad()
def predict_with_tta(model: NoCFormer, X: np.ndarray,
                     n_samples: int = 20, batch_size: int = 64,
                     device: torch.device | None = None,
                     use_mc_dropout: bool = True,
                     use_input_aug: bool = True,
                     jitter_sigma: float = 0.08):
    """
    Test-time inference with two complementary uncertainty sources:
        - MC-Dropout (model dropout enabled at test time)
        - Input augmentation (height jitter + peak permutation)

    Returns:
        probs:   [N, K]   mean class probabilities
        entropy: [N]      predictive entropy
        preds:   [N]      1-indexed NoC argmax
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    N = X.shape[0]
    K = model.num_classes
    accum = np.zeros((N, K), dtype=np.float64)

    base = torch.from_numpy(X).float()

    for s in range(n_samples):
        if use_mc_dropout:
            model.train()
        else:
            model.eval()
        x_in = base.clone()
        if use_input_aug:
            for i in range(N):
                x_in[i] = peak_height_jitter(x_in[i], sigma=jitter_sigma)
                x_in[i] = shuffle_peak_axis(x_in[i])
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            xb = x_in[start:end].to(device)
            out = model(xb)
            accum[start:end] += out["profile_noc_probs"].cpu().numpy()

    probs = accum / n_samples
    eps = 1e-12
    entropy = -(probs * np.log(probs + eps)).sum(axis=1)
    preds = probs.argmax(axis=1) + 1
    model.eval()
    return probs, entropy, preds


def load_nocformer(checkpoint_path: str, device: torch.device | None = None
                   ) -> NoCFormer:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    model = NoCFormer(
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4),
        peak_layers=cfg.get("peak_layers", 2),
        locus_layers=cfg.get("locus_layers", 4),
        num_classes=ck.get("num_classes", MAX_NOC),
        dropout=cfg.get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model

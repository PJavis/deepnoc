"""
Training loop for NoCNet-v2.

Hybrid dataset:
  * Real PROVEDIt profiles + their NoC labels.
  * Synthetic mixtures generated offline by `src.synth` (loaded via memmap so
    the >5 GB pool never needs to fit in RAM).

Each batch is a configurable mix of synthetic + real samples (default 80/20).
Aux losses on mix proportions and per-locus n_alleles only fire on the
synthetic rows where ground truth is exact.

Augmentation lives in this file because it has to be cheap and stay in lock-
step with the loader. Three vectorised numpy aug:
  * log-normal height jitter
  * random peak dropout (zero out a small fraction of real peaks)
  * peak shuffle (permute peaks within a locus; safe because the encoder
    is permutation-invariant by design)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.constants import MAX_NOC, NUM_LOCI
from models.nocnet_v2.architecture import NoCNetV2, HEIGHT_FEATURE_IDX
from models.nocnet_v2.losses import NoCNetV2Loss


# Mirror src/data_loader.py feature layout.
LOCUS_PEAKS_IDX = 77
PROFILE_PEAKS_IDX = 78
MIX_PROPS_SLICE = slice(79, 79 + MAX_NOC)


# ----------------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------------

def _recompute_globals(profile: np.ndarray) -> np.ndarray:
    """After modifying heights, refresh locus/profile peak-count features."""
    mask = profile[..., HEIGHT_FEATURE_IDX] > 0
    locus_counts = mask.sum(axis=-1)
    total = int(mask.sum())
    for li in range(NUM_LOCI):
        n = int(locus_counts[li])
        if n == 0:
            continue
        profile[li, :n, LOCUS_PEAKS_IDX] = n / 100.0
        profile[li, :n, PROFILE_PEAKS_IDX] = total / 1000.0
    return profile


def augment_profile(profile: np.ndarray,
                    jitter_sigma: float = 0.10,
                    dropout_p: float = 0.03,
                    do_shuffle: bool = True,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """In-place-ish numpy augmentation. Returns a NEW array."""
    if rng is None:
        rng = np.random.default_rng()
    p = profile.copy()
    mask = p[..., HEIGHT_FEATURE_IDX] > 0
    n_real = int(mask.sum())

    # Height jitter
    if jitter_sigma > 0 and n_real > 0:
        noise = rng.normal(0.0, jitter_sigma, size=n_real)
        p[..., HEIGHT_FEATURE_IDX][mask] = (
            p[..., HEIGHT_FEATURE_IDX][mask] * np.exp(noise)
        ).clip(0.0, 1.0)

    # Random peak dropout
    if dropout_p > 0 and n_real > 0:
        drop = (rng.random(size=p[..., HEIGHT_FEATURE_IDX].shape) < dropout_p) & mask
        for li, pi in zip(*np.where(drop)):
            p[li, pi] = 0.0
        if drop.any():
            mask = p[..., HEIGHT_FEATURE_IDX] > 0

    # Peak shuffle within each locus
    if do_shuffle:
        for li in range(NUM_LOCI):
            n = int(mask[li].sum())
            if n <= 1:
                continue
            perm = rng.permutation(n)
            p[li, :n] = p[li, :n][perm]

    if jitter_sigma > 0 or dropout_p > 0:
        p = _recompute_globals(p)
    return p


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------

class RealProfileDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False,
                 augment_cfg: dict | None = None):
        self.X = X
        self.y = y
        self.augment = augment
        self.cfg = augment_cfg or {}
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].astype(np.float32)
        if self.augment:
            x = augment_profile(x, rng=self.rng, **self.cfg)
        return {
            "x": torch.from_numpy(x),
            "y": int(self.y[i]),
            "is_synth": False,
            "mix": torch.zeros(MAX_NOC, dtype=torch.float32),
            "nall": torch.full((NUM_LOCI,), -1, dtype=torch.long),
        }


class SynthProfileDataset(Dataset):
    """
    Loads synthetic profiles via numpy memmap. The synth files are written by
    `python -m src.synth`. mmap_mode='r' avoids loading the full array into
    RAM; the OS pages in only the slices accessed during training.
    """

    def __init__(self, synth_dir: str, augment: bool = True,
                 augment_cfg: dict | None = None):
        # X is the only large array (~12 GB for the full pool). Memmap it so
        # the OS pages slices on demand; y/mix/nall are <2 MB → load eagerly.
        self.X = np.load(os.path.join(synth_dir, "X.npy"), mmap_mode="r")
        self.y = np.load(os.path.join(synth_dir, "y.npy"), allow_pickle=True)
        self.mix = np.load(os.path.join(synth_dir, "mix.npy"), allow_pickle=True)
        self.nall = np.load(os.path.join(synth_dir, "locus_nall.npy"), allow_pickle=True)
        self.augment = augment
        self.cfg = augment_cfg or {}
        self.rng = np.random.default_rng()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = np.array(self.X[i], dtype=np.float32)
        if self.augment:
            x = augment_profile(x, rng=self.rng, **self.cfg)
        return {
            "x": torch.from_numpy(x),
            "y": int(self.y[i]),
            "is_synth": True,
            "mix": torch.from_numpy(np.array(self.mix[i],
                                             dtype=np.float32)),
            "nall": torch.from_numpy(np.array(self.nall[i],
                                              dtype=np.int64)),
        }


class MixedSampler(torch.utils.data.Sampler):
    """
    Yields a flat sequence of (dataset_id, index) where dataset_id ∈ {0=real,
    1=synth}. Real indices are drawn with class-balanced weights so rare-NoC
    real profiles aren't drowned out; synth indices are uniform.
    """

    def __init__(self, real_y: np.ndarray, synth_y: np.ndarray | None,
                 p_synth: float = 0.8, n_per_epoch: int | None = None,
                 seed: int = 0):
        self.real_y = real_y
        self.synth_y = synth_y
        self.p_synth = p_synth if synth_y is not None else 0.0
        self.n = n_per_epoch or len(real_y)
        # Pre-compute real-sample weights for class balance.
        _, inv = np.unique(real_y, return_inverse=True)
        cnt = np.bincount(inv).astype(np.float64)
        per_class = 1.0 / np.maximum(cnt, 1.0)
        self.real_weights = per_class[inv]
        self.real_weights /= self.real_weights.sum()
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.n):
            if self.p_synth > 0 and self.rng.random() < self.p_synth:
                yield (1, int(self.rng.integers(0, len(self.synth_y))))
            else:
                yield (0, int(self.rng.choice(len(self.real_y),
                                              p=self.real_weights)))

    def __len__(self):
        return self.n


class HybridLoader:
    """Iterates `MixedSampler` and stacks one tensor batch from both datasets."""

    def __init__(self, real_ds: Dataset, synth_ds: Dataset | None,
                 sampler: MixedSampler, batch_size: int = 16):
        self.real_ds = real_ds
        self.synth_ds = synth_ds
        self.sampler = sampler
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.sampler) / self.batch_size)

    def __iter__(self):
        batch: list[dict] = []
        for ds_id, idx in self.sampler:
            sample = (self.synth_ds[idx] if ds_id == 1 else self.real_ds[idx])
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch:
            yield self._collate(batch)

    @staticmethod
    def _collate(batch):
        return {
            "x": torch.stack([b["x"] for b in batch]),
            "y": torch.tensor([b["y"] for b in batch], dtype=torch.long),
            "is_synth": torch.tensor([b["is_synth"] for b in batch],
                                     dtype=torch.bool),
            "mix": torch.stack([b["mix"] for b in batch]),
            "nall": torch.stack([b["nall"] for b in batch]),
        }


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 5e-4
    warmup_epochs: int = 3
    p_synth: float = 0.8
    samples_per_epoch: int = 4000
    d_model: int = 96
    n_heads: int = 4
    peak_layers: int = 2
    locus_layers: int = 2
    dropout: float = 0.15
    early_stop_patience: int = 12
    grad_clip: float = 1.0
    jitter_sigma: float = 0.10
    dropout_p: float = 0.03
    seed: int = 0
    # Stochastic Weight Averaging — averages model weights across the tail of
    # training to land in a wider, more generalisable minimum. Disabled when
    # `swa_frac` <= 0.
    swa_frac: float = 0.25
    label_smoothing: float = 0.1
    # MixUp on input tensors; λ ~ Beta(mixup_alpha, mixup_alpha), applied with
    # probability mixup_prob. Loss = λ L(out, targets) + (1-λ) L(out, targets').
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.5


def _class_counts(y: np.ndarray, num_classes: int) -> torch.Tensor:
    out = torch.zeros(num_classes, dtype=torch.long)
    for i in range(num_classes):
        out[i] = int((y == i + 1).sum())
    return out


def train_nocnet_v2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    synth_dir: Optional[str] = "data/synthetic",
    config: Optional[TrainConfig] = None,
    save_dir: str = "results",
    tag: str = "nocnet_v2",
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    real_train = RealProfileDataset(
        X_train, y_train, augment=True,
        augment_cfg={"jitter_sigma": cfg.jitter_sigma,
                     "dropout_p": cfg.dropout_p, "do_shuffle": True},
    )
    real_test = RealProfileDataset(X_test, y_test, augment=False)

    use_synth = synth_dir is not None and os.path.exists(
        os.path.join(synth_dir, "X.npy"))
    synth_train = (SynthProfileDataset(
        synth_dir, augment=True,
        augment_cfg={"jitter_sigma": cfg.jitter_sigma,
                     "dropout_p": cfg.dropout_p, "do_shuffle": True},
    ) if use_synth else None)

    sampler = MixedSampler(
        real_y=y_train,
        synth_y=(synth_train.y if synth_train is not None else None),
        p_synth=cfg.p_synth if use_synth else 0.0,
        n_per_epoch=cfg.samples_per_epoch,
        seed=cfg.seed,
    )
    train_loader = HybridLoader(real_train, synth_train, sampler,
                                batch_size=cfg.batch_size)
    test_loader = DataLoader(real_test, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=0)

    model = NoCNetV2(
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        peak_layers=cfg.peak_layers, locus_layers=cfg.locus_layers,
        num_classes=num_classes, dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"[NoCNet-v2] params={n_params:,} device={device} "
              f"synth={'yes' if use_synth else 'no'}")
        print(f"[NoCNet-v2] real_train={len(X_train)} test={len(X_test)} "
              f"K={num_classes} epochs={cfg.epochs}")

    class_counts = _class_counts(y_train, num_classes).to(device)
    criterion = NoCNetV2Loss(num_classes=num_classes,
                             class_counts=class_counts,
                             label_smoothing=cfg.label_smoothing).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / max(1, cfg.warmup_epochs)
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [],
               "train_mae": [], "test_mae": [], "swa_test_acc": [],
               "best_test_acc": 0.0, "best_epoch": 0, "best_was_swa": False}
    best_acc = 0.0
    no_improve = 0

    # SWA: start averaging in the tail of training. avg_fn=None means classic
    # equal-weight average across each update.
    swa_model: AveragedModel | None = None
    swa_start_epoch = (int(cfg.epochs * (1.0 - cfg.swa_frac))
                       if cfg.swa_frac > 0 else cfg.epochs + 1)

    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = tr_correct = tr_mae = tr_total = 0
        loss_n = 0
        bar = tqdm(train_loader, desc=f"ep {epoch+1}/{cfg.epochs}",
                   disable=not verbose, leave=False)
        for batch in bar:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            is_synth = batch["is_synth"].to(device, non_blocking=True)
            mix = batch["mix"].to(device, non_blocking=True)
            nall = batch["nall"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            use_mixup = (
                cfg.mixup_alpha > 0.0
                and x.size(0) > 1
                and torch.rand(1, device=device).item() < cfg.mixup_prob
            )
            if use_mixup:
                lam = float(
                    torch.distributions.Beta(
                        cfg.mixup_alpha, cfg.mixup_alpha
                    ).sample().item()
                )
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1.0 - lam) * x[perm]
                out = model(x)
                losses_a = criterion(out, {
                    "profile_noc": y,
                    "is_synth": is_synth,
                    "profile_mix_props": mix,
                    "locus_n_alleles": nall,
                })
                losses_b = criterion(out, {
                    "profile_noc": y[perm],
                    "is_synth": is_synth[perm],
                    "profile_mix_props": mix[perm],
                    "locus_n_alleles": nall[perm],
                })
                loss = lam * losses_a["total"] + (1.0 - lam) * losses_b["total"]
            else:
                out = model(x)
                losses = criterion(out, {
                    "profile_noc": y,
                    "is_synth": is_synth,
                    "profile_mix_props": mix,
                    "locus_n_alleles": nall,
                })
                loss = losses["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg.grad_clip)
            optimizer.step()

            with torch.no_grad():
                pred = out["profile_noc_probs"].argmax(dim=-1) + 1
                tr_correct += (pred == y).sum().item()
                tr_mae += (pred.float() - y.float()).abs().sum().item()
                tr_total += y.numel()
                tr_loss += loss.item() * y.numel()
                loss_n += y.numel()
            bar.set_postfix(loss=f"{loss.item():.3f}",
                            acc=f"{tr_correct / max(tr_total,1):.3f}")

        train_loss = tr_loss / max(loss_n, 1)
        train_acc = tr_correct / max(tr_total, 1)
        train_mae = tr_mae / max(tr_total, 1)

        # ---------- Eval ----------
        def _evaluate(eval_model):
            eval_model.eval()
            e_loss = e_corr = e_mae = e_tot = 0
            with torch.no_grad():
                for batch in test_loader:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    out = eval_model(x)
                    losses = criterion(out, {
                        "profile_noc": y,
                        "is_synth": torch.zeros_like(y, dtype=torch.bool),
                    })
                    pred = out["profile_noc_probs"].argmax(dim=-1) + 1
                    e_corr += (pred == y).sum().item()
                    e_mae += (pred.float() - y.float()).abs().sum().item()
                    e_tot += y.numel()
                    e_loss += losses["total"].item() * y.numel()
            return (e_loss / max(e_tot, 1),
                    e_corr / max(e_tot, 1),
                    e_mae / max(e_tot, 1))

        test_loss, test_acc, test_mae = _evaluate(model)

        # SWA update + parallel evaluation. AveragedModel keeps a running
        # average of `model`'s parameters; we evaluate it independently and
        # treat it as a candidate "best" checkpoint.
        if epoch >= swa_start_epoch:
            if swa_model is None:
                swa_model = AveragedModel(model)
            swa_model.update_parameters(model)
            swa_test_loss, swa_test_acc, swa_test_mae = _evaluate(swa_model)
        else:
            swa_test_acc = float("nan")

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["train_mae"].append(train_mae)
        history["test_mae"].append(test_mae)
        history["swa_test_acc"].append(swa_test_acc)

        # Best-checkpoint selection: prefer the higher of (live, SWA).
        live_better = test_acc > best_acc
        swa_better = (swa_model is not None and not math.isnan(swa_test_acc)
                      and swa_test_acc > best_acc)
        if live_better or swa_better:
            if swa_better and swa_test_acc >= test_acc:
                state = swa_model.module.state_dict()
                best_acc = swa_test_acc
                history["best_was_swa"] = True
            else:
                state = model.state_dict()
                best_acc = test_acc
                history["best_was_swa"] = False
            history["best_test_acc"] = best_acc
            history["best_epoch"] = epoch
            no_improve = 0
            torch.save({
                "model_state_dict": state,
                "epoch": epoch, "test_acc": best_acc,
                "num_classes": num_classes,
                "config": vars(cfg),
                "was_swa": history["best_was_swa"],
            }, os.path.join(save_dir, f"best_{tag}.pt"))
        else:
            no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
            swa_tag = (f" swa {swa_test_acc:.3f}" if swa_model is not None
                       else "")
            print(f"  ep {epoch:4d} | tr loss {train_loss:.3f} acc {train_acc:.3f} "
                  f"mae {train_mae:.2f} | te loss {test_loss:.3f} acc {test_acc:.3f} "
                  f"mae {test_mae:.2f}{swa_tag} | "
                  f"best {best_acc:.3f}{'(swa)' if history['best_was_swa'] else ''} "
                  f"@ {history['best_epoch']}")

        if cfg.early_stop_patience and no_improve >= cfg.early_stop_patience:
            if verbose:
                print(f"Early stopping after {no_improve} epochs without improvement.")
            break

    with open(os.path.join(save_dir, f"history_{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)

    best_ck = os.path.join(save_dir, f"best_{tag}.pt")
    if os.path.exists(best_ck):
        ck = torch.load(best_ck, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
    return model, history


def finetune_nocnet_v2(
    checkpoint_path: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    epochs: int = 30,
    lr: float = 1e-5,
    batch_size: int = 16,
    weight_decay: float = 1e-4,
    samples_per_epoch: int = 1500,
    swa_frac: float = 0.4,
    jitter_sigma: float = 0.05,
    dropout_p: float = 0.01,
    synth_dir: Optional[str] = None,
    p_synth: float = 0.0,
    save_dir: str = "results",
    tag: str = "nocnet_v2_ft",
    freeze_peak_stages: bool = False,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    """
    Real-data fine-tune of a NoCNet-v2 checkpoint trained on synthetic.

    Differences from `train_nocnet_v2`:
      * `synth_dir=None` so the loader draws only from real PROVEDIt.
      * Much lower LR (default 1e-5) so we nudge weights toward real-data
        distribution without forgetting synth-learned representations.
      * Weaker augmentation (real data is already noisy).
      * `freeze_peak_stages=True` optionally freezes the peak embedder + peak
        attention blocks, training only the cross-locus stack + heads.

    Returns: (model, history) like train_nocnet_v2.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_nocnet_v2(checkpoint_path, device=device,
                           num_classes=num_classes)

    if freeze_peak_stages:
        for p in model.peak_embed.parameters():
            p.requires_grad = False
        for p in model.peak_blocks.parameters():
            p.requires_grad = False
        if verbose:
            print("[finetune] frozen: peak_embed + peak_blocks")

    cfg = TrainConfig(
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=max(1, epochs // 10),
        p_synth=p_synth,
        samples_per_epoch=samples_per_epoch,
        d_model=model.d_model, n_heads=4,
        peak_layers=len(model.peak_blocks),
        locus_layers=len(model.cross_locus.blocks),
        dropout=0.10,
        early_stop_patience=max(5, epochs // 3),
        grad_clip=1.0,
        jitter_sigma=jitter_sigma,
        dropout_p=dropout_p,
        seed=0,
        swa_frac=swa_frac,
        label_smoothing=0.05,
        mixup_alpha=0.0,
        mixup_prob=0.0,
    )

    # Re-implement the training loop with a pre-built model rather than
    # reusing train_nocnet_v2 (which constructs a fresh model). Most logic
    # is shared so we factor through the same helpers.
    os.makedirs(save_dir, exist_ok=True)
    real_train = RealProfileDataset(
        X_train, y_train, augment=True,
        augment_cfg={"jitter_sigma": cfg.jitter_sigma,
                     "dropout_p": cfg.dropout_p, "do_shuffle": True},
    )
    real_test = RealProfileDataset(X_test, y_test, augment=False)

    use_synth = (synth_dir is not None and p_synth > 0
                 and os.path.exists(os.path.join(synth_dir, "X.npy")))
    synth_train = (SynthProfileDataset(
        synth_dir, augment=True,
        augment_cfg={"jitter_sigma": cfg.jitter_sigma,
                     "dropout_p": cfg.dropout_p, "do_shuffle": True},
    ) if use_synth else None)

    sampler = MixedSampler(
        y_train,
        synth_y=(synth_train.y if synth_train is not None else None),
        p_synth=p_synth if use_synth else 0.0,
        n_per_epoch=cfg.samples_per_epoch, seed=cfg.seed,
    )
    train_loader = HybridLoader(real_train, synth_train, sampler,
                                batch_size=cfg.batch_size)
    if verbose and use_synth:
        print(f"[finetune] hybrid p_synth={p_synth} synth={synth_dir}")
    test_loader = DataLoader(real_test, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=0)

    class_counts = _class_counts(y_train, num_classes).to(device)
    criterion = NoCNetV2Loss(num_classes=num_classes,
                             class_counts=class_counts,
                             label_smoothing=cfg.label_smoothing).to(device)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / max(1, cfg.warmup_epochs)
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [],
               "train_mae": [], "test_mae": [], "swa_test_acc": [],
               "best_test_acc": 0.0, "best_epoch": 0, "best_was_swa": False}

    # Establish a baseline on the pretrained checkpoint BEFORE we start
    # fine-tuning. Otherwise we'd be picking the best of FT epochs only,
    # potentially worse than the loaded weights.
    def _eval(m):
        m.eval()
        e_loss = e_corr = e_mae = e_tot = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                out = m(x)
                losses = criterion(out, {
                    "profile_noc": y,
                    "is_synth": torch.zeros_like(y, dtype=torch.bool),
                })
                pred = out["profile_noc_probs"].argmax(dim=-1) + 1
                e_corr += (pred == y).sum().item()
                e_mae += (pred.float() - y.float()).abs().sum().item()
                e_tot += y.numel()
                e_loss += losses["total"].item() * y.numel()
        return (e_loss / max(e_tot, 1),
                e_corr / max(e_tot, 1),
                e_mae / max(e_tot, 1))

    base_loss, base_acc, base_mae = _eval(model)
    history["best_test_acc"] = base_acc
    if verbose:
        print(f"[finetune] baseline (no FT yet) test_acc={base_acc:.4f} mae={base_mae:.3f}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": -1, "test_acc": base_acc,
        "num_classes": num_classes,
        "config": vars(cfg), "was_swa": False, "from": "pretrain",
    }, os.path.join(save_dir, f"best_{tag}.pt"))

    swa_model: AveragedModel | None = None
    swa_start_epoch = (int(cfg.epochs * (1.0 - cfg.swa_frac))
                       if cfg.swa_frac > 0 else cfg.epochs + 1)
    no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = tr_correct = tr_mae = tr_total = 0
        loss_n = 0
        bar = tqdm(train_loader, desc=f"ft {epoch+1}/{cfg.epochs}",
                   disable=not verbose, leave=False)
        for batch in bar:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            is_synth = batch["is_synth"].to(device, non_blocking=True)
            mix = batch["mix"].to(device, non_blocking=True)
            nall = batch["nall"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            losses = criterion(out, {
                "profile_noc": y,
                "is_synth": is_synth,
                "profile_mix_props": mix,
                "locus_n_alleles": nall,
            })
            loss = losses["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=cfg.grad_clip,
            )
            optimizer.step()
            with torch.no_grad():
                pred = out["profile_noc_probs"].argmax(dim=-1) + 1
                tr_correct += (pred == y).sum().item()
                tr_mae += (pred.float() - y.float()).abs().sum().item()
                tr_total += y.numel()
                tr_loss += loss.item() * y.numel()
                loss_n += y.numel()
            bar.set_postfix(loss=f"{loss.item():.3f}",
                            acc=f"{tr_correct/max(tr_total,1):.3f}")

        train_loss = tr_loss / max(loss_n, 1)
        train_acc = tr_correct / max(tr_total, 1)
        train_mae = tr_mae / max(tr_total, 1)

        test_loss, test_acc, test_mae = _eval(model)
        if epoch >= swa_start_epoch:
            if swa_model is None:
                swa_model = AveragedModel(model)
            swa_model.update_parameters(model)
            _, swa_test_acc, _ = _eval(swa_model)
        else:
            swa_test_acc = float("nan")
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["train_mae"].append(train_mae)
        history["test_mae"].append(test_mae)
        history["swa_test_acc"].append(swa_test_acc)

        candidate_acc = max(test_acc, swa_test_acc if not math.isnan(swa_test_acc) else -1.0)
        if candidate_acc > history["best_test_acc"]:
            from_swa = (not math.isnan(swa_test_acc) and swa_test_acc >= test_acc
                        and swa_model is not None)
            state = (swa_model.module.state_dict()
                     if from_swa else model.state_dict())
            history["best_test_acc"] = candidate_acc
            history["best_epoch"] = epoch
            history["best_was_swa"] = from_swa
            no_improve = 0
            torch.save({
                "model_state_dict": state,
                "epoch": epoch, "test_acc": candidate_acc,
                "num_classes": num_classes,
                "config": vars(cfg), "was_swa": from_swa,
                "from": "finetune",
            }, os.path.join(save_dir, f"best_{tag}.pt"))
        else:
            no_improve += 1

        if verbose and (epoch % 2 == 0 or epoch == cfg.epochs - 1):
            swa_tag = (f" swa {swa_test_acc:.3f}"
                       if not math.isnan(swa_test_acc) else "")
            print(f"  ft {epoch:3d} | tr loss {train_loss:.3f} acc {train_acc:.3f} "
                  f"| te acc {test_acc:.3f} mae {test_mae:.2f}{swa_tag}"
                  f" | best {history['best_test_acc']:.4f}"
                  f"{'(swa)' if history['best_was_swa'] else ''}")

        if cfg.early_stop_patience and no_improve >= cfg.early_stop_patience:
            if verbose:
                print(f"[finetune] early stop after {no_improve} idle epochs")
            break

    with open(os.path.join(save_dir, f"history_{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)

    best_ck = os.path.join(save_dir, f"best_{tag}.pt")
    if os.path.exists(best_ck):
        ck = torch.load(best_ck, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
    if verbose:
        print(f"[finetune] done. base={base_acc:.4f} → best={history['best_test_acc']:.4f}"
              f" (epoch {history['best_epoch']}, swa={history['best_was_swa']})")
    return model, history


def load_nocnet_v2(checkpoint_path: str, device: torch.device | None = None,
                   num_classes: int | None = None) -> NoCNetV2:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    nc = num_classes or ck.get("num_classes", MAX_NOC)
    model = NoCNetV2(
        d_model=cfg.get("d_model", 96),
        n_heads=cfg.get("n_heads", 4),
        peak_layers=cfg.get("peak_layers", 2),
        locus_layers=cfg.get("locus_layers", 2),
        num_classes=nc,
        dropout=cfg.get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_nocnet_v2(model: NoCNetV2, X: np.ndarray, batch_size: int = 32,
                      device: torch.device | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    probs = []
    for s in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[s:s + batch_size].astype(np.float32)).to(device)
        out = model(xb)
        probs.append(out["profile_noc_probs"].cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    preds = probs.argmax(axis=1) + 1
    return probs, preds


@torch.no_grad()
def predict_nocnet_v2_tta(
    model: NoCNetV2,
    X: np.ndarray,
    n_samples: int = 8,
    batch_size: int = 32,
    jitter_sigma: float = 0.08,
    dropout_p: float = 0.0,
    do_shuffle: bool = True,
    use_mc_dropout: bool = False,
    device: torch.device | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Test-time augmentation: run `n_samples` forward passes per profile, each
    with a fresh peak shuffle + log-normal height jitter, then average the
    softmax probabilities. Optional MC-Dropout enables dropout at test time
    for additional epistemic uncertainty.

    Returns:
        probs:   [N, K] averaged class probabilities
        preds:   [N]    argmax + 1
        entropy: [N]    predictive entropy across the TTA samples
    """
    if device is None:
        device = next(model.parameters()).device
    if use_mc_dropout:
        model.train()
    else:
        model.eval()

    rng = np.random.default_rng(seed)
    N = X.shape[0]
    K = model.num_classes
    accum = np.zeros((N, K), dtype=np.float64)
    aug_cfg = {"jitter_sigma": jitter_sigma, "dropout_p": dropout_p,
               "do_shuffle": do_shuffle}
    desc = f"tta ({n_samples}x)"
    sample_iter = tqdm(range(n_samples), desc=desc, disable=not verbose)
    for s in sample_iter:
        s_rng = np.random.default_rng(seed * 1000 + s)
        x_aug = np.empty_like(X, dtype=np.float32)
        for i in range(N):
            x_aug[i] = augment_profile(X[i].astype(np.float32),
                                       rng=s_rng, **aug_cfg)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            xb = torch.from_numpy(x_aug[start:end]).to(device)
            out = model(xb)
            accum[start:end] += out["profile_noc_probs"].cpu().numpy()
    probs = accum / n_samples
    preds = probs.argmax(axis=1) + 1
    eps = 1e-12
    entropy = -(probs * np.log(probs + eps)).sum(axis=1)
    model.eval()
    return probs, preds, entropy

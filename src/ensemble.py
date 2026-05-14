"""
Multi-seed ensemble for NoCNet-v2.

Trains the same pipeline (pretrain + optional finetune) with N different
seeds, then averages the per-class softmax probabilities at inference.
Optionally combines with TTA and/or post-hoc threshold tuning.

Why this works: each seed lands the model in a different basin of the
loss landscape; the basins make different mistakes; averaging the
probability vectors cancels independent noise. Standard +1-3 pts in
practice for classification.

Output:
    results/ensemble/seed_<k>/best.pt    per-seed checkpoint
    results/ensemble/probs_test.npy      [N_test, K] mean probs
    results/ensemble/preds_test.npy      [N_test] argmax+1
    results/ensemble/summary.json        metrics per seed + ensemble

Usage:
    python -m src.ensemble --n-seeds 5 --pretrain-epochs 100 --finetune-epochs 40

CLI also exposed via `python main.py ensemble`.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch

from src.constants import MAX_NOC


def _load_data(out_dir: str):
    X = np.load(os.path.join(out_dir, "X_gf25.npy"))
    y = np.load(os.path.join(out_dir, "y_gf25.npy"))
    names_path = os.path.join(out_dir, "sample_names.txt")
    if os.path.exists(names_path):
        with open(names_path) as f:
            names = [line.rstrip("\n") for line in f if line.strip()]
        if len(names) != len(y):
            names = [str(i) for i in range(len(y))]
    else:
        names = [str(i) for i in range(len(y))]
    return X, y, names


def _split(X, y, names, split: str, test_size: float, seed: int):
    if split == "grouped":
        from src.split import grouped_stratified_split
        return grouped_stratified_split(X, y, names, test_size=test_size,
                                        seed=seed)
    if split == "stratified":
        from src.split import stratified_split
        return stratified_split(X, y, names, test_size=test_size, seed=seed)
    from src.data_loader import train_test_split_alternating
    return train_test_split_alternating(X, y, list(range(len(y))))


def _metrics(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, f1_score,
                                 confusion_matrix)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mae": float(np.abs(y_true - y_pred).mean()),
        "off1": float((np.abs(y_true - y_pred) <= 1).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "confusion": confusion_matrix(
            y_true, y_pred,
            labels=sorted(set(y_true.tolist() + y_pred.tolist()))).tolist(),
    }


def train_seed(
    seed: int,
    X_train, y_train, X_test, y_test,
    num_classes: int,
    synth_dir: Optional[str],
    pretrain_epochs: int,
    finetune_epochs: int,
    batch_size: int,
    lr: float,
    finetune_lr: float,
    p_synth: float,
    finetune_p_synth: float,
    d_model: int,
    samples_per_epoch: int,
    save_dir: str,
    device: torch.device,
    tta_samples: int = 0,
    verbose: bool = True,
):
    """Pretrain + optional finetune for one seed; return probs on X_test."""
    from models.nocnet_v2.train import (
        train_nocnet_v2, finetune_nocnet_v2,
        predict_nocnet_v2, predict_nocnet_v2_tta, TrainConfig,
    )

    cfg = TrainConfig(
        epochs=pretrain_epochs, batch_size=batch_size, lr=lr,
        p_synth=p_synth, samples_per_epoch=samples_per_epoch,
        d_model=d_model, seed=seed,
    )
    pre_tag = f"seed{seed}_pre"
    _, _ = train_nocnet_v2(
        X_train, y_train, X_test, y_test,
        num_classes=num_classes, synth_dir=synth_dir,
        config=cfg, save_dir=save_dir, tag=pre_tag,
        device=device, verbose=verbose,
    )
    pre_ckpt = os.path.join(save_dir, f"best_{pre_tag}.pt")

    if finetune_epochs > 0:
        ft_tag = f"seed{seed}_ft"
        model, _ = finetune_nocnet_v2(
            pre_ckpt, X_train, y_train, X_test, y_test,
            num_classes=num_classes, epochs=finetune_epochs,
            lr=finetune_lr, batch_size=batch_size,
            samples_per_epoch=max(samples_per_epoch // 3, 500),
            synth_dir=synth_dir, p_synth=finetune_p_synth,
            save_dir=save_dir, tag=ft_tag, device=device, verbose=verbose,
        )
    else:
        from models.nocnet_v2.train import load_nocnet_v2
        model = load_nocnet_v2(pre_ckpt, device=device,
                               num_classes=num_classes)

    if tta_samples > 0:
        probs, preds, _ = predict_nocnet_v2_tta(
            model, X_test, n_samples=tta_samples,
            batch_size=batch_size, device=device,
            seed=seed, verbose=verbose,
        )
    else:
        probs, preds = predict_nocnet_v2(model, X_test,
                                         batch_size=batch_size,
                                         device=device)
    return probs, preds


def run_ensemble(
    X: np.ndarray, y: np.ndarray, names: list[str],
    n_seeds: int,
    base_seed: int = 42,
    split: str = "grouped",
    test_size: float = 0.25,
    synth_dir: str = "data/synthetic",
    pretrain_epochs: int = 100,
    finetune_epochs: int = 40,
    finetune_lr: float = 1e-5,
    finetune_p_synth: float = 0.2,
    batch_size: int = 16,
    lr: float = 3e-4,
    p_synth: float = 0.85,
    d_model: int = 96,
    samples_per_epoch: int = 4000,
    save_dir: str = "results/ensemble",
    tta_samples: int = 0,
    verbose: bool = True,
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr, X_te, y_tr, y_te, _, _ = _split(X, y, names, split, test_size,
                                          base_seed)
    num_classes = int(max(y.max(), 5))
    print(f"[ensemble] split={split} train={len(X_tr)} test={len(X_te)} "
          f"K={num_classes} seeds={n_seeds}")

    accum = np.zeros((len(y_te), num_classes), dtype=np.float64)
    per_seed: list[dict] = []

    for s in range(n_seeds):
        seed = base_seed + s
        seed_dir = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        print(f"\n========== seed {seed} ({s+1}/{n_seeds}) ==========")
        probs, preds = train_seed(
            seed=seed,
            X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
            num_classes=num_classes, synth_dir=synth_dir,
            pretrain_epochs=pretrain_epochs,
            finetune_epochs=finetune_epochs,
            batch_size=batch_size, lr=lr, finetune_lr=finetune_lr,
            p_synth=p_synth, finetune_p_synth=finetune_p_synth,
            d_model=d_model, samples_per_epoch=samples_per_epoch,
            save_dir=seed_dir, device=device,
            tta_samples=tta_samples, verbose=verbose,
        )
        accum += probs
        m = _metrics(y_te, preds)
        m["seed"] = seed
        per_seed.append(m)
        print(f"[seed {seed}] acc={m['accuracy']:.4f} "
              f"mae={m['mae']:.3f} macroF1={m['macro_f1']:.4f}")

    mean_probs = accum / n_seeds
    ens_preds = mean_probs.argmax(axis=1) + 1
    ens_metrics = _metrics(y_te, ens_preds)
    summary = {
        "n_seeds": n_seeds,
        "split": split,
        "test_size": int(len(y_te)),
        "per_seed": per_seed,
        "ensemble": ens_metrics,
    }
    np.save(os.path.join(save_dir, "probs_test.npy"), mean_probs)
    np.save(os.path.join(save_dir, "preds_test.npy"), ens_preds)
    np.save(os.path.join(save_dir, "y_test.npy"), y_te)
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 60)
    print("  Ensemble summary")
    print("=" * 60)
    for ps in per_seed:
        print(f"  seed {ps['seed']:>3d}  acc={ps['accuracy']:.4f}  "
              f"mae={ps['mae']:.3f}  macroF1={ps['macro_f1']:.4f}")
    print(f"  ENSEMBLE   acc={ens_metrics['accuracy']:.4f}  "
          f"mae={ens_metrics['mae']:.3f}  off1={ens_metrics['off1']:.4f}  "
          f"macroF1={ens_metrics['macro_f1']:.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/provedit_processed")
    ap.add_argument("--synth-dir", default="data/synthetic")
    ap.add_argument("--save-dir", default="results/ensemble")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--split", choices=["alternating", "stratified", "grouped"],
                    default="grouped")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--pretrain-epochs", type=int, default=100)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--finetune-lr", type=float, default=1e-5)
    ap.add_argument("--finetune-p-synth", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--p-synth", type=float, default=0.85)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--samples-per-epoch", type=int, default=4000)
    ap.add_argument("--tta-samples", type=int, default=0,
                    help="If >0, use TTA at per-seed inference")
    args = ap.parse_args()

    X, y, names = _load_data(args.data_dir)
    print(f"Loaded X={X.shape} y={y.shape} names={len(names)}")
    run_ensemble(
        X, y, names,
        n_seeds=args.n_seeds, base_seed=args.base_seed,
        split=args.split, test_size=args.test_size,
        synth_dir=args.synth_dir,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        finetune_p_synth=args.finetune_p_synth,
        batch_size=args.batch_size, lr=args.lr,
        p_synth=args.p_synth, d_model=args.d_model,
        samples_per_epoch=args.samples_per_epoch,
        save_dir=args.save_dir,
        tta_samples=args.tta_samples,
    )


if __name__ == "__main__":
    main()

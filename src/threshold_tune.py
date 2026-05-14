"""
Per-class additive logit-bias tuning to maximise macro-F1.

Argmax over softmax probabilities pushes mass toward the majority class
when class priors are imbalanced (NoC=1 dominates PROVEDIt). A standard
correction is to add a per-class bias b_k to the class logits before
argmax — equivalently, multiply probabilities by exp(b_k) and renormalise.
This module searches the bias vector b on a validation set to maximise
macro-F1.

Two search strategies:
    `coordinate_search`  — round-robin grid over each b_k while others are
                           frozen. Fast (O(K * grid)) and deterministic.
    `random_search`      — sample b from N(0, σ I) and keep the best.

The tuned biases are then applied at inference:
    p_calibrated = softmax(log p + b)
    pred = argmax(p_calibrated)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import f1_score


def apply_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Apply additive bias in log-space, return renormalised probabilities.
    """
    log_p = np.log(np.clip(probs, 1e-12, 1.0))
    log_p_b = log_p + bias[None, :]
    # Numerically stable softmax over the K dimension.
    m = log_p_b.max(axis=1, keepdims=True)
    e = np.exp(log_p_b - m)
    return e / e.sum(axis=1, keepdims=True)


def macro_f1_from_probs(probs: np.ndarray, y_true: np.ndarray,
                        bias: np.ndarray) -> float:
    """probs: [N, K]   y_true: [N] in 1..K   bias: [K]"""
    p = apply_bias(probs, bias)
    pred = p.argmax(axis=1) + 1
    return f1_score(y_true, pred, average="macro", zero_division=0)


def accuracy_from_probs(probs: np.ndarray, y_true: np.ndarray,
                        bias: np.ndarray) -> float:
    p = apply_bias(probs, bias)
    pred = p.argmax(axis=1) + 1
    return float((pred == y_true).mean())


def coordinate_search(
    probs: np.ndarray,
    y_true: np.ndarray,
    grid: np.ndarray | None = None,
    rounds: int = 3,
    metric: str = "macro_f1",
    verbose: bool = True,
) -> np.ndarray:
    """
    Coordinate ascent over each class bias in turn.

    probs:    [N, K] from softmax (or pre-bias logits run through softmax)
    y_true:   [N] integer class labels in 1..K
    grid:     candidate offsets per coordinate, default np.linspace(-3, 3, 31)
    rounds:   sweeps across all coordinates before stopping
    """
    K = probs.shape[1]
    if grid is None:
        grid = np.linspace(-3.0, 3.0, 31)
    bias = np.zeros(K, dtype=np.float64)
    score_fn = (macro_f1_from_probs if metric == "macro_f1"
                else accuracy_from_probs)
    best = score_fn(probs, y_true, bias)
    if verbose:
        print(f"[tune] start {metric}={best:.4f} bias={bias.tolist()}")
    for r in range(rounds):
        improved = False
        for k in range(K):
            best_local = best
            best_off = bias[k]
            for off in grid:
                trial = bias.copy()
                trial[k] = off
                s = score_fn(probs, y_true, trial)
                if s > best_local + 1e-6:
                    best_local = s
                    best_off = off
            if best_local > best + 1e-6:
                bias[k] = best_off
                best = best_local
                improved = True
        if verbose:
            print(f"[tune] round {r+1}: {metric}={best:.4f} "
                  f"bias={np.round(bias, 3).tolist()}")
        if not improved:
            break
    return bias


def random_search(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_trials: int = 2000,
    sigma: float = 1.5,
    seed: int = 0,
    metric: str = "macro_f1",
    verbose: bool = True,
) -> np.ndarray:
    K = probs.shape[1]
    rng = np.random.default_rng(seed)
    score_fn = (macro_f1_from_probs if metric == "macro_f1"
                else accuracy_from_probs)
    best_bias = np.zeros(K)
    best = score_fn(probs, y_true, best_bias)
    if verbose:
        print(f"[tune] random search: start {metric}={best:.4f}")
    for t in range(n_trials):
        cand = rng.normal(0.0, sigma, size=K)
        # Centre so the bias preserves prior mass.
        cand = cand - cand.mean()
        s = score_fn(probs, y_true, cand)
        if s > best:
            best = s
            best_bias = cand
            if verbose and (t % 500 == 0 or t == n_trials - 1):
                print(f"[tune] t={t} {metric}={best:.4f} "
                      f"bias={np.round(best_bias, 3).tolist()}")
    return best_bias


def tune_thresholds(
    probs_val: np.ndarray,
    y_val: np.ndarray,
    probs_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    metric: str = "macro_f1",
    verbose: bool = True,
) -> dict:
    """
    Convenience wrapper: run a coordinate search and a random-search
    refinement, return both biases plus before/after metrics on val (and
    test if provided).
    """
    coord_bias = coordinate_search(probs_val, y_val, metric=metric,
                                   verbose=verbose)
    # Refine with random search around the coordinate solution.
    K = probs_val.shape[1]
    rng = np.random.default_rng(0)
    bias = coord_bias.copy()
    score_fn = (macro_f1_from_probs if metric == "macro_f1"
                else accuracy_from_probs)
    best = score_fn(probs_val, y_val, bias)
    for _ in range(1000):
        cand = bias + rng.normal(0.0, 0.3, size=K)
        cand = cand - cand.mean()
        s = score_fn(probs_val, y_val, cand)
        if s > best:
            bias = cand
            best = s

    summary = {
        "bias": bias.tolist(),
        "metric": metric,
        "val_metric_before": float(score_fn(probs_val, y_val, np.zeros(K))),
        "val_metric_after": float(best),
        "val_acc_before": accuracy_from_probs(probs_val, y_val, np.zeros(K)),
        "val_acc_after": accuracy_from_probs(probs_val, y_val, bias),
    }
    if probs_test is not None and y_test is not None:
        summary["test_metric_before"] = float(
            score_fn(probs_test, y_test, np.zeros(K)))
        summary["test_metric_after"] = float(
            score_fn(probs_test, y_test, bias))
        summary["test_acc_before"] = accuracy_from_probs(probs_test, y_test,
                                                         np.zeros(K))
        summary["test_acc_after"] = accuracy_from_probs(probs_test, y_test, bias)
    return summary


# ----------------------------------------------------------------------------
# CLI: tune from a saved nocnet_v2 checkpoint
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to best_*.pt for nocnet_v2")
    ap.add_argument("--data-dir", default="data/provedit_processed")
    ap.add_argument("--split", choices=["alternating", "stratified", "grouped"],
                    default="grouped")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.5,
                    help="Fraction of TEST split used as validation for tuning. "
                         "Remaining is held-out test for honest after-tune report.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric", choices=["macro_f1", "accuracy"],
                    default="macro_f1")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--tta-samples", type=int, default=8)
    ap.add_argument("--out", default="results/threshold_tuning.json")
    args = ap.parse_args()

    import torch
    from models.nocnet_v2.train import (
        load_nocnet_v2, predict_nocnet_v2, predict_nocnet_v2_tta,
    )

    X = np.load(os.path.join(args.data_dir, "X_gf25.npy"))
    y = np.load(os.path.join(args.data_dir, "y_gf25.npy"))
    names_path = os.path.join(args.data_dir, "sample_names.txt")
    if os.path.exists(names_path):
        with open(names_path) as f:
            names = [line.rstrip("\n") for line in f if line.strip()]
        if len(names) != len(y):
            names = [str(i) for i in range(len(y))]
    else:
        names = [str(i) for i in range(len(y))]

    if args.split == "grouped":
        from src.split import grouped_stratified_split
        _, X_te, _, y_te, _, _ = grouped_stratified_split(
            X, y, names, test_size=args.test_size, seed=args.seed,
        )
    elif args.split == "stratified":
        from src.split import stratified_split
        _, X_te, _, y_te, _, _ = stratified_split(
            X, y, names, test_size=args.test_size, seed=args.seed,
        )
    else:
        from src.data_loader import train_test_split_alternating
        _, X_te, _, y_te, _, _ = train_test_split_alternating(
            X, y, list(range(len(y))),
        )

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(y_te))
    n_val = int(len(y_te) * args.val_frac)
    val_idx, test_idx = perm[:n_val], perm[n_val:]
    X_val, y_val = X_te[val_idx], y_te[val_idx]
    X_holdout, y_holdout = X_te[test_idx], y_te[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_nocnet_v2(args.checkpoint, device=device)

    if args.tta:
        probs_val, _, _ = predict_nocnet_v2_tta(
            model, X_val, n_samples=args.tta_samples, device=device)
        probs_test, _, _ = predict_nocnet_v2_tta(
            model, X_holdout, n_samples=args.tta_samples, device=device)
    else:
        probs_val, _ = predict_nocnet_v2(model, X_val, device=device)
        probs_test, _ = predict_nocnet_v2(model, X_holdout, device=device)

    out = tune_thresholds(probs_val, y_val, probs_test, y_holdout,
                          metric=args.metric, verbose=True)
    out["split"] = args.split
    out["val_size"] = int(len(y_val))
    out["test_size"] = int(len(y_holdout))
    out["checkpoint"] = args.checkpoint
    out["tta"] = bool(args.tta)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 60)
    print(f"  Threshold tuning summary -> {args.out}")
    print("=" * 60)
    for k in ("val_metric_before", "val_metric_after",
              "val_acc_before", "val_acc_after",
              "test_metric_before", "test_metric_after",
              "test_acc_before", "test_acc_after"):
        if k in out:
            print(f"  {k:22s} {out[k]:.4f}")
    print(f"  bias = {np.round(np.array(out['bias']), 3).tolist()}")


if __name__ == "__main__":
    main()

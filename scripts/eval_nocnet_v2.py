"""
Ad-hoc evaluator for a NoCNet-v2 checkpoint, with or without TTA.

Usage:
    python scripts/eval_nocnet_v2.py --checkpoint results/best_nocnet_v2_ft.pt
    python scripts/eval_nocnet_v2.py --checkpoint results/best_nocnet_v2_ft.pt --tta
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="data/provedit_processed")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--split", choices=["alternating", "stratified", "grouped"],
                        default="grouped")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta-samples", type=int, default=20)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    from models.nocnet_v2.train import (
        load_nocnet_v2, predict_nocnet_v2, predict_nocnet_v2_tta,
    )
    from src.evaluation import full_evaluation
    from src.split import grouped_stratified_split, stratified_split
    from src.data_loader import train_test_split_alternating

    X = np.load(os.path.join(args.output_dir, "X_gf25.npy"))
    y = np.load(os.path.join(args.output_dir, "y_gf25.npy"))
    names_path = os.path.join(args.output_dir, "sample_names.txt")
    if os.path.exists(names_path):
        with open(names_path) as f:
            names = [line.rstrip("\n") for line in f if line.strip()]
    else:
        names = [str(i) for i in range(len(y))]
    if len(names) != len(y):
        names = [str(i) for i in range(len(y))]

    if args.split == "grouped":
        _, X_te, _, y_te, _, _ = grouped_stratified_split(
            X, y, names, test_size=args.test_size, seed=args.seed)
    elif args.split == "stratified":
        _, X_te, _, y_te, _, _ = stratified_split(
            X, y, names, test_size=args.test_size, seed=args.seed)
    else:
        _, X_te, _, y_te, _, _ = train_test_split_alternating(X, y, names)

    print(f"split={args.split} test_size={args.test_size} "
          f"test_N={len(y_te)} dist={dict(zip(*np.unique(y_te, return_counts=True)))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_nocnet_v2(args.checkpoint, device=device)
    print(f"checkpoint={args.checkpoint} device={device}")

    if args.tta:
        print(f"TTA n_samples={args.tta_samples}")
        probs, preds, entropy = predict_nocnet_v2_tta(
            model, X_te, n_samples=args.tta_samples,
            batch_size=args.batch_size, device=device, verbose=True,
        )
        tag = args.tag or "NoCNet-v2_eval_tta"
        np.save(os.path.join(args.results_dir, f"{tag}_entropy.npy"), entropy)
    else:
        probs, preds = predict_nocnet_v2(model, X_te,
                                         batch_size=args.batch_size,
                                         device=device)
        tag = args.tag or "NoCNet-v2_eval"

    labels = sorted(set(y_te))
    metrics, _ = full_evaluation(y_te, preds, y_probs=probs,
                                 class_labels=labels, title=tag,
                                 save_dir=args.results_dir)

    out_path = os.path.join(args.results_dir, f"{tag}_summary.json")
    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "test_size": args.test_size,
        "seed": args.seed,
        "tta": args.tta,
        "tta_samples": args.tta_samples if args.tta else 0,
        "test_N": int(len(y_te)),
        "per_class": {str(k): {kk: float(vv) if isinstance(vv, (int, float)) else vv
                                for kk, vv in v.items()}
                       for k, v in metrics.items() if k != "overall"},
        "overall": {kk: float(vv) if isinstance(vv, (int, float)) else vv
                    for kk, vv in metrics["overall"].items()},
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out_path}")


if __name__ == "__main__":
    main()

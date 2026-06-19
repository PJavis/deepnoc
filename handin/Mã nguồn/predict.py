"""
Standalone NoC inference with the released NoCNet-v2 model.

Reproduces the held-out test result on the grouped split, or scores any
[N, 24, 50, 89] GlobalFiler tensor.

Run from the deliverable/ directory:

    # reproduce the held-out test metrics (acc 0.927)
    python predict.py --reproduce-test

    # score an arbitrary tensor
    python predict.py --x my_X.npy [--y my_y.npy]

Requires: torch, numpy, scikit-learn (only for --reproduce-test metrics).
"""

from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# Bộ mã nguồn nằm phẳng dưới "Mã nguồn/" (models/, src/); hỗ trợ cả layout cũ "code/".
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "code"))
from models.nocnet_v2.train import (
    load_nocnet_v2, predict_nocnet_v2, predict_nocnet_v2_tta)


def _find(*cands):
    """Trả về đường dẫn tồn tại đầu tiên (hoặc ứng viên cuối)."""
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[-1]


def main():
    here = _HERE
    # Trọng số nằm ở thư mục "Mô hình" (anh em với "Mã nguồn"); fallback layout cũ.
    model_dir = os.path.join(here, "..", "Mô hình")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_find(
        os.path.join(model_dir, "nocnet_v2_ft.pt"), os.path.join(here, "model/nocnet_v2_ft.pt")))
    ap.add_argument("--bias", default=_find(
        os.path.join(model_dir, "bias_tuned.json"), os.path.join(here, "model/bias_tuned.json")),
                    help="per-class additive logit bias json (set '' to skip)")
    ap.add_argument("--x", default=None, help="path to [N,24,50,89] tensor")
    ap.add_argument("--y", default=None, help="optional true labels [N] (1-indexed)")
    ap.add_argument("--tta", action="store_true", help="20x MC-dropout TTA")
    ap.add_argument("--tta-samples", type=int, default=20)
    ap.add_argument("--reproduce-test", action="store_true",
                    help="load bundled data + grouped split, score the test fold")
    ap.add_argument("--out", default="predictions.npy")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_nocnet_v2(args.model, device=dev)

    if args.reproduce_test:
        X = np.load(os.path.join(here, "data/X_gf25.npy"))
        y = np.load(os.path.join(here, "data/y_gf25.npy"))
        split = json.load(open(os.path.join(here, "data/split_grouped_seed42.json")))
        te = split["test_idx"]
        X, y = X[te], y[te]
        args.tta = True
    else:
        if not args.x:
            ap.error("provide --x or --reproduce-test")
        X = np.load(args.x)
        y = np.load(args.y) if args.y else None

    if args.tta:
        probs, preds, ent = predict_nocnet_v2_tta(
            model, X, n_samples=args.tta_samples, device=dev, verbose=True)
    else:
        probs, preds = predict_nocnet_v2(model, X, device=dev)

    if args.bias:
        bias = np.array(json.load(open(args.bias))["bias"])
        adj = np.log(probs + 1e-12) + bias[None, :]
        preds = adj.argmax(1) + 1
        print(f"applied per-class bias {np.round(bias,3).tolist()}")

    np.save(args.out, preds)
    print(f"predictions [{len(preds)}] -> {args.out}")

    if y is not None:
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        acc = accuracy_score(y, preds)
        mf = f1_score(y, preds, average="macro", zero_division=0)
        print(f"\naccuracy = {acc:.4f}   macro-F1 = {mf:.4f}")
        labels = sorted(set(int(v) for v in y))
        per = f1_score(y, preds, labels=labels, average=None, zero_division=0)
        print("per-class F1:", {l: round(float(f), 4) for l, f in zip(labels, per)})
        print("confusion (rows=true, cols=pred):")
        print(confusion_matrix(y, preds, labels=labels))


if __name__ == "__main__":
    main()

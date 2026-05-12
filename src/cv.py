"""
5-fold grouped CV runner.

Why grouped CV: PROVEDIt re-injects the same physical mixture at multiple
times / amplifications; alternating splits leak. Grouped CV is honest, and
running it 5 ways gives a real confidence interval on every metric.

Model adapters provided:
    * mac           — rule-based baseline (no training)
    * rf            — Random Forest on summary features (sklearn)
    * deepnoc_simple / deepnoc_full — existing CNN
    * nocformer     — existing Transformer baseline
    * nocnet_v2     — the new architecture

Outputs:
    results/cv/<model>/fold<k>/metrics.json
    results/cv/<model>/summary.json    (mean ± std per metric)

Usage:
    python -m src.cv --models nocnet_v2 nocformer rf mac
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from src.constants import MAX_NOC
from src.split import _pedigree_key


def _load_data(out_dir: str):
    X = np.load(os.path.join(out_dir, "X_gf25.npy"))
    y = np.load(os.path.join(out_dir, "y_gf25.npy"))
    names_path = os.path.join(out_dir, "sample_names.txt")
    if os.path.exists(names_path):
        with open(names_path) as f:
            names = [line.rstrip("\n") for line in f if line.strip()]
    else:
        names = [str(i) for i in range(len(y))]
    if len(names) != len(y):
        names = [str(i) for i in range(len(y))]
    return X, y, names


def _metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix)
    acc = float(accuracy_score(y_true, y_pred))
    mae = float(np.abs(y_true - y_pred).mean())
    off1 = float((np.abs(y_true - y_pred) <= 1).mean())
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=sorted(set(y_true)), zero_division=0
    )
    per_class = {}
    for k, lbl in enumerate(sorted(set(y_true))):
        per_class[int(lbl)] = {
            "precision": float(pr[k]), "recall": float(rc[k]),
            "f1": float(f1[k]), "support": int(sup[k]),
        }
    cm = confusion_matrix(y_true, y_pred,
                          labels=sorted(set(y_true.tolist() + y_pred.tolist())))
    return {"accuracy": acc, "mae": mae, "off_by_one_acc": off1,
            "macro_f1": float(f1.mean()), "per_class": per_class,
            "confusion_matrix": cm.tolist()}


# ----- adapters -----------------------------------------------------------

def run_mac(X_train, y_train, X_test, y_test, **_):
    from models.baseline.baselines import run_mac_baseline
    _, preds = run_mac_baseline(X_test, y_test)
    return np.asarray(preds)


def run_rf(X_train, y_train, X_test, y_test, **_):
    from models.baseline.baselines import train_random_forest
    _, _, _, preds = train_random_forest(X_train, y_train, X_test, y_test)
    return np.asarray(preds)


def run_deepnoc(model_type: str):
    def runner(X_train, y_train, X_test, y_test, *, save_dir, num_classes,
               epochs, batch_size, lr, device, **_):
        from models.deepnoc.train import train_deepnoc
        import torch.nn.functional as F
        model, _ = train_deepnoc(
            X_train, y_train, X_test, y_test,
            num_classes=num_classes, epochs=epochs,
            batch_size=batch_size, lr=lr, device=device,
            save_dir=save_dir, model_type=model_type,
        )
        model.eval()
        with torch.no_grad():
            Xt = torch.FloatTensor(X_test).to(device)
            if model_type == "full":
                out = model(Xt)
                logits = out["profile_noc"]
            else:
                logits = model(Xt)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs.argmax(axis=1) + 1
    return runner


def run_nocformer(X_train, y_train, X_test, y_test, *, save_dir, num_classes,
                  epochs, batch_size, lr, device, **_):
    from models.nocformer.train import train_nocformer
    model, _ = train_nocformer(
        X_train, y_train, X_test, y_test,
        num_classes=num_classes, epochs=epochs,
        batch_size=batch_size, lr=lr, device=device,
        save_dir=save_dir, tag="nocformer",
    )
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_test).float().to(device)
        out = model(Xt)
        probs = out["profile_noc_probs"].cpu().numpy()
    return probs.argmax(axis=1) + 1


def run_nocnet_v2(X_train, y_train, X_test, y_test, *, save_dir, num_classes,
                  epochs, batch_size, lr, device, synth_dir, **_):
    from models.nocnet_v2.train import train_nocnet_v2, predict_nocnet_v2, TrainConfig
    cfg = TrainConfig(epochs=epochs, batch_size=batch_size, lr=lr)
    model, _ = train_nocnet_v2(
        X_train, y_train, X_test, y_test,
        num_classes=num_classes, synth_dir=synth_dir,
        config=cfg, save_dir=save_dir, tag="nocnet_v2", device=device,
    )
    _, preds = predict_nocnet_v2(model, X_test, batch_size=batch_size, device=device)
    return preds


MODEL_RUNNERS: dict[str, Callable] = {
    "mac": run_mac,
    "rf": run_rf,
    "deepnoc_simple": run_deepnoc("simple"),
    "deepnoc_full": run_deepnoc("full"),
    "nocformer": run_nocformer,
    "nocnet_v2": run_nocnet_v2,
}


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    models: list[str],
    n_folds: int = 5,
    seed: int = 42,
    results_root: str = "results/cv",
    synth_dir: str = "data/synthetic",
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 3e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups = np.array([_pedigree_key(n) for n in names])

    gkf = GroupKFold(n_splits=n_folds)
    fold_metrics: dict[str, list[dict]] = {m: [] for m in models}

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n===== Fold {fold + 1}/{n_folds} "
              f"(train={len(tr_idx)} test={len(te_idx)}) =====")
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        num_classes = int(max(y.max(), MAX_NOC))

        for m in models:
            print(f"\n--- {m} ---")
            save_dir = os.path.join(results_root, m, f"fold{fold}")
            os.makedirs(save_dir, exist_ok=True)
            try:
                preds = MODEL_RUNNERS[m](
                    X_tr, y_tr, X_te, y_te,
                    save_dir=save_dir, num_classes=num_classes,
                    epochs=epochs, batch_size=batch_size, lr=lr,
                    device=device, synth_dir=synth_dir,
                )
            except Exception as e:
                print(f"!! {m} failed in fold {fold}: {e}")
                continue
            metrics = _metrics_from_preds(y_te, preds)
            metrics["fold"] = fold
            metrics["train_size"] = int(len(tr_idx))
            metrics["test_size"] = int(len(te_idx))
            with open(os.path.join(save_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            fold_metrics[m].append(metrics)
            print(f"  acc={metrics['accuracy']:.4f} "
                  f"mae={metrics['mae']:.3f} "
                  f"off1={metrics['off_by_one_acc']:.4f} "
                  f"macroF1={metrics['macro_f1']:.4f}")

    # ------------- Summary -------------
    summary = {}
    for m, folds in fold_metrics.items():
        if not folds:
            summary[m] = {"error": "no successful folds"}
            continue
        agg = {}
        for key in ("accuracy", "mae", "off_by_one_acc", "macro_f1"):
            vals = [f[key] for f in folds]
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                        "values": vals}
        summary[m] = agg
        out_dir = os.path.join(results_root, m)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(agg, f, indent=2)

    with open(os.path.join(results_root, "summary_all.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("CV SUMMARY")
    print("=" * 60)
    for m, agg in summary.items():
        if "error" in agg:
            print(f"  {m:18s} FAILED ({agg['error']})")
            continue
        print(f"  {m:18s} acc={agg['accuracy']['mean']:.4f}±{agg['accuracy']['std']:.4f}"
              f"  mae={agg['mae']['mean']:.3f}±{agg['mae']['std']:.3f}"
              f"  off1={agg['off_by_one_acc']['mean']:.4f}"
              f"  F1={agg['macro_f1']['mean']:.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/provedit_processed")
    ap.add_argument("--synth-dir", default="data/synthetic")
    ap.add_argument("--results-root", default="results/cv")
    ap.add_argument("--models", nargs="+",
                    default=["mac", "rf", "nocnet_v2"],
                    choices=sorted(MODEL_RUNNERS.keys()))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y, names = _load_data(args.data_dir)
    print(f"Loaded X={X.shape} y={y.shape} names={len(names)}")
    cross_validate(
        X, y, names,
        models=args.models,
        n_folds=args.folds,
        seed=args.seed,
        results_root=args.results_root,
        synth_dir=args.synth_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()

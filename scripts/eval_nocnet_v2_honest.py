"""Regenerate HONEST per-class table for NoCNet-v2:
   FT checkpoint + TTA(20x) + clean per-class bias (tune_clean.json),
   on the full 923 grouped-test split. Mirrors the headline acc 0.927 / macroF1 0.653.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from models.nocnet_v2.train import load_nocnet_v2, predict_nocnet_v2_tta
from src.evaluation import full_evaluation
from src.split import grouped_stratified_split
from src.threshold_tune import apply_bias

torch.manual_seed(42)
np.random.seed(42)

DATA = "data/provedit_processed"
CKPT = "results/best_nocnet_v2_ft.pt"

X = np.load(os.path.join(DATA, "X_gf25.npy"))
y = np.load(os.path.join(DATA, "y_gf25.npy"))
with open(os.path.join(DATA, "sample_names.txt")) as f:
    names = [l.rstrip("\n") for l in f if l.strip()]
if len(names) != len(y):
    names = [str(i) for i in range(len(y))]

_, X_te, _, y_te, _, _ = grouped_stratified_split(X, y, names, test_size=0.25, seed=42)
print("test_N", len(y_te), "dist", dict(zip(*np.unique(y_te, return_counts=True))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_nocnet_v2(CKPT, device=device)

probs, _, _ = predict_nocnet_v2_tta(model, X_te, n_samples=20, device=device, verbose=True)

bias = np.array(json.load(open("results/tune_clean.json"))["bias"])
print("bias", bias.tolist())

for label, p in (("NO BIAS", probs), ("CLEAN BIAS", apply_bias(probs, bias))):
    pred = p.argmax(axis=1) + 1
    labels = sorted(set(y_te))
    metrics, _ = full_evaluation(y_te, pred, y_probs=p, class_labels=labels,
                                 title=f"honest_{label}", save_dir="results")
    from sklearn.metrics import f1_score
    mf1 = f1_score(y_te, pred, average="macro", zero_division=0)
    print(f"\n===== {label}: overall acc={metrics['overall']['accuracy']:.4f} macroF1={mf1:.4f} =====")
    print(json.dumps({str(k): v for k, v in metrics.items()}, indent=2, default=float))
    if label == "CLEAN BIAS":
        out = {"per_class": {str(k): v for k, v in metrics.items() if k != "overall"},
               "overall": metrics["overall"], "macro_f1": float(mf1),
               "bias": bias.tolist(), "tta": 20, "split": "grouped", "test_N": int(len(y_te))}
        json.dump(out, open("results/metrics_nocnet-v2_honest.json", "w"), indent=2, default=float)
        print("\nsaved -> results/metrics_nocnet-v2_honest.json")

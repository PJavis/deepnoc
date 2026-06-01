"""
Set-Transformer NoC classifier on the tokenized `noc_filtered_grouped` dataset.

Pipeline:
  1. Train a (configurably large) peak-set Transformer; early-stop on val macro-F1.
  2. TTA inference: N stochastic passes (peak-shuffle + log-height jitter +
     MC-dropout), average softmax.
  3. Per-class bias tuning (coordinate ascent on val macro-F1), applied to test.

Data per split:
    tokens_{split}.npy  [N, 220, 3]  (locus_idx 0-27, allele, log_height)
    mask_{split}.npy    [N, 220]     1 = real peak
    noc_{split}.npy     [N]          0-indexed (0=NoC1 ... 4=NoC5)

Split is donor-grouped (leakage-safe).

Usage:
    python scripts/train_seqnoc.py --data-dir /tmp/nfg/data/noc_filtered_grouped \
        --epochs 100 --batch-size 64 --d-model 192 --layers 5 --heads 6 \
        --tta-samples 20 --bias-tune
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


class PeakSetClassifier(nn.Module):
    def __init__(self, n_loci=28, d_model=192, n_heads=6, n_layers=5,
                 n_classes=5, dropout=0.2):
        super().__init__()
        self.loc_emb = nn.Embedding(n_loci, d_model)
        self.scalar = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.LayerNorm(d_model),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, tokens, mask):
        loc = tokens[..., 0].long().clamp(min=0)
        allele = tokens[..., 1:2] / 50.0
        logh = tokens[..., 2:3] / 10.0
        scal = torch.cat([allele, logh], dim=-1)
        h = self.loc_emb(loc) + self.scalar(scal)
        b = h.size(0)
        cls = self.cls.expand(b, -1, -1)
        h = torch.cat([cls, h], dim=1)
        keep = torch.cat([torch.ones(b, 1, device=h.device), mask], dim=1)
        h = self.encoder(h, src_key_padding_mask=(keep == 0))
        return self.head(self.ln(h[:, 0]))


def load_split(d, split):
    t = np.load(os.path.join(d, f"tokens_{split}.npy")).astype(np.float32)
    m = np.load(os.path.join(d, f"mask_{split}.npy")).astype(np.float32)
    y = np.load(os.path.join(d, f"noc_{split}.npy")).astype(np.int64)
    return t, m, y


def tta_probs(model, X, M, dev, n_samples, batch_size, jitter=0.10, seed=0):
    """N stochastic passes: shuffle real peaks + jitter log-height + MC-dropout."""
    model.train()  # keep dropout on
    N = X.shape[0]
    n_cls = model.head[-1].out_features
    accum = np.zeros((N, n_cls), dtype=np.float64)
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for s in range(max(1, n_samples)):
            Xa = X.copy()
            # jitter log-height (feat2) on real peaks
            jit = rng.normal(0.0, jitter, size=Xa[..., 2].shape).astype(np.float32)
            Xa[..., 2] = np.where(M > 0, Xa[..., 2] + jit, Xa[..., 2])
            # shuffle peak order per profile (set-invariant)
            for i in range(N):
                k = int(M[i].sum())
                if k > 1:
                    perm = rng.permutation(k)
                    Xa[i, :k] = Xa[i, :k][perm]
            for st in range(0, N, batch_size):
                en = min(st + batch_size, N)
                xb = torch.from_numpy(Xa[st:en]).to(dev)
                mb = torch.from_numpy(M[st:en]).to(dev)
                p = torch.softmax(model(xb, mb), dim=1)
                accum[st:en] += p.cpu().numpy()
    model.eval()
    return accum / max(1, n_samples)


def tune_bias(probs_val, y_val, n_cls, rounds=4, grid=None):
    """Coordinate ascent on additive per-class logit bias -> maximize macro-F1."""
    if grid is None:
        grid = np.arange(-2.0, 2.01, 0.2)
    logp = np.log(probs_val + 1e-12)
    bias = np.zeros(n_cls)

    def mf1(b):
        return f1_score(y_val, (logp + b).argmax(1), average="macro", zero_division=0)

    best = mf1(bias)
    for _ in range(rounds):
        improved = False
        for c in range(n_cls):
            cur = bias[c]
            for g in grid:
                bias[c] = g
                v = mf1(bias)
                if v > best:
                    best, cur, improved = v, g, True
            bias[c] = cur
        if not improved:
            break
    return bias, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/tmp/nfg/data/noc_filtered_grouped")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tta-samples", type=int, default=20)
    ap.add_argument("--bias-tune", action="store_true")
    ap.add_argument("--out", default="results/seqnoc")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    Xtr, Mtr, ytr = load_split(args.data_dir, "train")
    Xva, Mva, yva = load_split(args.data_dir, "val")
    Xte, Mte, yte = load_split(args.data_dir, "test")
    vocab = json.load(open(os.path.join(args.data_dir, "vocab.json")))
    n_loci = vocab.get("n_loci", 28); n_cls = vocab.get("n_noc", 5)
    print(f"train={len(ytr)} val={len(yva)} test={len(yte)} n_loci={n_loci} dev={dev}")

    cnt = np.bincount(ytr, minlength=n_cls).astype(np.float64)
    w_per = 1.0 / np.maximum(cnt, 1)
    sampler = WeightedRandomSampler(
        torch.as_tensor(w_per[ytr], dtype=torch.double), len(ytr), replacement=True)

    def loader(X, M, y, train=False):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(y))
        return DataLoader(ds, batch_size=args.batch_size,
                          sampler=sampler if train else None, shuffle=False)

    tl = loader(Xtr, Mtr, ytr, train=True)
    vl = loader(Xva, Mva, yva)

    model = PeakSetClassifier(n_loci=n_loci, d_model=args.d_model, n_heads=args.heads,
                              n_layers=args.layers, n_classes=n_cls,
                              dropout=args.dropout).to(dev)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = max(1, args.epochs // 20)
    def lr_l(e):
        if e < warm: return (e + 1) / warm
        import math
        return 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, args.epochs - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_l)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    @torch.no_grad()
    def quick_eval(dl):
        model.eval(); ps, ys = [], []
        for x, m, y in dl:
            ps.append(model(x.to(dev), m.to(dev)).argmax(1).cpu().numpy()); ys.append(y.numpy())
        p = np.concatenate(ps); t = np.concatenate(ys)
        return accuracy_score(t, p), f1_score(t, p, average="macro", zero_division=0)

    best_vf1, best_state = -1, None
    for ep in range(args.epochs):
        model.train()
        for x, m, y in tl:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x.to(dev), m.to(dev)), y.to(dev))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        va, vf1 = quick_eval(vl)
        if vf1 > best_vf1:
            best_vf1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} | val acc {va:.4f} macroF1 {vf1:.4f} | best valF1 {best_vf1:.4f}")

    model.load_state_dict(best_state)

    # ---- TTA probs on val + test ----
    print(f"\nTTA n_samples={args.tta_samples}")
    pv = tta_probs(model, Xva, Mva, dev, args.tta_samples, args.batch_size, seed=1)
    pt = tta_probs(model, Xte, Mte, dev, args.tta_samples, args.batch_size, seed=2)

    def report(tag, probs, y, bias=None):
        logits = np.log(probs + 1e-12) + (bias[None, :] if bias is not None else 0)
        pred = logits.argmax(1)
        acc = accuracy_score(y, pred); mf = f1_score(y, pred, average="macro", zero_division=0)
        per = f1_score(y, pred, average=None, zero_division=0)
        print(f"[{tag}] acc={acc:.4f} macroF1={mf:.4f} per-class={[round(float(x),3) for x in per]}")
        return acc, mf, per, pred

    print("\n=== TEST results ===")
    a0, f0, per0, _ = report("TTA, no bias", pt, yte)

    out = {"test_acc_tta": float(a0), "test_macroF1_tta": float(f0),
           "per_class_f1_tta": [float(x) for x in per0],
           "best_val_macroF1": float(best_vf1),
           "params": int(sum(p.numel() for p in model.parameters())),
           "config": {"d_model": args.d_model, "layers": args.layers,
                      "heads": args.heads, "epochs": args.epochs,
                      "tta_samples": args.tta_samples}}

    if args.bias_tune:
        bias, vbest = tune_bias(pv, yva, n_cls)
        print(f"\nbias (val-tuned, valF1={vbest:.4f}) = {np.round(bias,2).tolist()}")
        a1, f1v, per1, predb = report("TTA + bias", pt, yte, bias=bias)
        cm = confusion_matrix(yte, predb)
        print("confusion (rows=true 1..5, cols=pred):"); print(cm)
        out.update({"bias": bias.tolist(), "test_acc_bias": float(a1),
                    "test_macroF1_bias": float(f1v),
                    "per_class_f1_bias": [float(x) for x in per1],
                    "confusion_matrix_bias": cm.tolist()})

    torch.save(best_state, os.path.join(args.out, "best_seqnoc.pt"))
    with open(os.path.join(args.out, "seqnoc_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {args.out}/seqnoc_metrics.json + best_seqnoc.pt")


if __name__ == "__main__":
    sys.exit(main())

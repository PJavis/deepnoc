"""
seqnoc v2: CORN ordinal head + train-time augmentation + bias tuning.

Improvements over v1 (which plateaued at test macro-F1 0.437):
  - CORN ordinal head (Cao/Mirjalili/Raschka 2020): NoC is ordered, so a
    5->3 mistake should cost more than 5->4. K-1 rank-monotone cumulative
    logits, conditional-training BCE.
  - Real train-time augmentation: log-height jitter + random peak dropout.
    (Peak SHUFFLE is a no-op — the set-Transformer is permutation invariant —
    so it is intentionally omitted.)
  - Mid capacity, strong reg: d=128, 4 layers, dropout 0.3, wd 5e-4.
  - Early-stop on val macro-F1; TTA (jitter + dropout + MC-dropout) at eval;
    per-class additive-bias tuning on val.

Labels are 0-indexed (0=NoC1 ... 4=NoC5).

Usage:
    python scripts/train_seqnoc_v2.py --data-dir /tmp/nfg/data/noc_filtered_grouped \
        --epochs 120 --d-model 128 --layers 4 --tta-samples 20 --bias-tune
"""

from __future__ import annotations

import argparse, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


class CornSetClassifier(nn.Module):
    def __init__(self, n_loci=28, d_model=128, n_heads=4, n_layers=4,
                 n_classes=5, dropout=0.3):
        super().__init__()
        self.n_classes = n_classes
        self.loc_emb = nn.Embedding(n_loci, d_model)
        self.scalar = nn.Sequential(nn.Linear(2, d_model), nn.GELU(),
                                    nn.LayerNorm(d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                         batch_first=True, activation="gelu",
                                         norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_model, n_classes - 1))  # CORN: K-1

    def forward(self, tokens, mask):
        loc = tokens[..., 0].long().clamp(min=0)
        scal = torch.cat([tokens[..., 1:2] / 50.0, tokens[..., 2:3] / 10.0], -1)
        h = self.loc_emb(loc) + self.scalar(scal)
        b = h.size(0)
        h = torch.cat([self.cls.expand(b, -1, -1), h], dim=1)
        keep = torch.cat([torch.ones(b, 1, device=h.device), mask], dim=1)
        h = self.encoder(h, src_key_padding_mask=(keep == 0))
        return self.head(self.ln(h[:, 0]))               # [B, K-1]


def corn_loss(logits, y, n_classes):
    """Conditional-training CORN loss. y 0-indexed in [0, K-1]."""
    total = logits.new_zeros(())
    nt = 0
    for k in range(n_classes - 1):
        mask = y > (k - 1)                  # samples with y >= k
        if not mask.any():
            continue
        z = logits[mask, k]
        t = (y[mask] > k).float()           # P(y > k)
        total = total + F.binary_cross_entropy_with_logits(z, t)
        nt += 1
    return total / max(nt, 1)


def corn_probs_to_class(logits):
    """Rank-monotone decode -> 0-indexed class."""
    cum = torch.cumprod(torch.sigmoid(logits), dim=1)     # [B, K-1] = P(y>k)
    return (cum > 0.5).sum(dim=1)                          # 0..K-1


def corn_class_probs(logits):
    """Per-class probabilities [B, K] from CORN cumulative logits."""
    sig = torch.sigmoid(logits)
    cum = torch.cumprod(sig, dim=1)                        # P(y>k), k=0..K-2
    b = logits.size(0)
    one = torch.ones(b, 1, device=logits.device)
    zero = torch.zeros(b, 1, device=logits.device)
    p_ge = torch.cat([one, cum], dim=1)                   # P(y>=k)
    p_gt = torch.cat([cum, zero], dim=1)                  # P(y>k)
    return (p_ge - p_gt).clamp(min=1e-9)                  # [B, K]


def augment_batch(x, m, jitter=0.10, drop_p=0.05, rng=None):
    """Train-time aug on a batch tensor (numpy). Jitter logheight + drop peaks."""
    if rng is None:
        rng = np.random
    xa = x.copy()
    jit = rng.normal(0, jitter, size=xa[..., 2].shape).astype(np.float32)
    xa[..., 2] = np.where(m > 0, xa[..., 2] + jit, xa[..., 2])
    if drop_p > 0:
        keep = (rng.random(m.shape) > drop_p) | (m == 0)
        m = m * keep
    return xa, m


def load_split(d, s):
    return (np.load(f"{d}/tokens_{s}.npy").astype(np.float32),
            np.load(f"{d}/mask_{s}.npy").astype(np.float32),
            np.load(f"{d}/noc_{s}.npy").astype(np.int64))


def tta_corn(model, X, M, dev, n, bs, jitter=0.10, drop_p=0.05, seed=0):
    model.train()
    N, K = X.shape[0], model.n_classes
    acc = np.zeros((N, K))
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for s in range(max(1, n)):
            Xa, Ma = augment_batch(X, M, jitter, drop_p, rng)
            for st in range(0, N, bs):
                en = min(st + bs, N)
                lo = model(torch.from_numpy(Xa[st:en]).to(dev),
                           torch.from_numpy(Ma[st:en]).to(dev))
                acc[st:en] += corn_class_probs(lo).cpu().numpy()
    model.eval()
    return acc / max(1, n)


def tune_bias(pv, yv, K, rounds=4):
    grid = np.arange(-2.0, 2.01, 0.2)
    lp = np.log(pv + 1e-12)
    b = np.zeros(K)
    mf = lambda bb: f1_score(yv, (lp + bb).argmax(1), average="macro", zero_division=0)
    best = mf(b)
    for _ in range(rounds):
        imp = False
        for c in range(K):
            cur = b[c]
            for g in grid:
                b[c] = g
                v = mf(b)
                if v > best:
                    best, cur, imp = v, g, True
            b[c] = cur
        if not imp:
            break
    return b, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/tmp/nfg/data/noc_filtered_grouped")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--jitter", type=float, default=0.10)
    ap.add_argument("--drop-p", type=float, default=0.05)
    ap.add_argument("--tta-samples", type=int, default=20)
    ap.add_argument("--bias-tune", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/seqnoc")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    D = args.data_dir
    Xtr, Mtr, ytr = load_split(D, "train")
    Xva, Mva, yva = load_split(D, "val")
    Xte, Mte, yte = load_split(D, "test")
    vocab = json.load(open(f"{D}/vocab.json"))
    K = vocab.get("n_noc", 5); n_loci = vocab.get("n_loci", 28)
    print(f"train={len(ytr)} val={len(yva)} test={len(yte)} K={K} dev={dev}")

    cnt = np.bincount(ytr, minlength=K).astype(np.float64)
    w = 1.0 / np.maximum(cnt, 1)
    sampler = WeightedRandomSampler(torch.as_tensor(w[ytr], dtype=torch.double),
                                    len(ytr), replacement=True)
    idx = np.arange(len(ytr))
    dl = DataLoader(TensorDataset(torch.from_numpy(idx)), batch_size=args.batch_size,
                    sampler=sampler)

    model = CornSetClassifier(n_loci, args.d_model, args.heads, args.layers, K,
                              args.dropout).to(dev)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = max(1, args.epochs // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e:
        (e + 1) / warm if e < warm else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, args.epochs - warm))))
    rng = np.random.default_rng(args.seed)

    @torch.no_grad()
    def vmf1():
        model.eval(); ps = []
        for st in range(0, len(yva), args.batch_size):
            en = min(st + args.batch_size, len(yva))
            lo = model(torch.from_numpy(Xva[st:en]).to(dev), torch.from_numpy(Mva[st:en]).to(dev))
            ps.append(corn_probs_to_class(lo).cpu().numpy())
        return f1_score(yva, np.concatenate(ps), average="macro", zero_division=0)

    best, best_state = -1, None
    for ep in range(args.epochs):
        model.train()
        for (bi,) in dl:
            bi = bi.numpy()
            xa, ma = augment_batch(Xtr[bi], Mtr[bi], args.jitter, args.drop_p, rng)
            xb = torch.from_numpy(xa).to(dev); mb = torch.from_numpy(ma).to(dev)
            yb = torch.from_numpy(ytr[bi]).to(dev)
            opt.zero_grad(set_to_none=True)
            loss = corn_loss(model(xb, mb), yb, K)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        f = vmf1()
        if f > best:
            best = f
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} | val macroF1 {f:.4f} | best {best:.4f}")

    model.load_state_dict(best_state)
    pv = tta_corn(model, Xva, Mva, dev, args.tta_samples, args.batch_size,
                  args.jitter, args.drop_p, seed=1)
    pt = tta_corn(model, Xte, Mte, dev, args.tta_samples, args.batch_size,
                  args.jitter, args.drop_p, seed=2)

    def rep(tag, p, y, bias=None):
        lg = np.log(p + 1e-12) + (bias[None, :] if bias is not None else 0)
        pred = lg.argmax(1)
        a, mf = accuracy_score(y, pred), f1_score(y, pred, average="macro", zero_division=0)
        per = f1_score(y, pred, average=None, zero_division=0)
        print(f"[{tag}] acc={a:.4f} macroF1={mf:.4f} per={[round(float(x),3) for x in per]}")
        return a, mf, per, pred

    print("\n=== TEST ===")
    a0, f0, per0, _ = rep("TTA", pt, yte)
    out = {"test_acc_tta": float(a0), "test_macroF1_tta": float(f0),
           "per_class_f1_tta": [float(x) for x in per0],
           "best_val_macroF1": float(best),
           "params": int(sum(p.numel() for p in model.parameters()))}
    if args.bias_tune:
        b, vb = tune_bias(pv, yva, K)
        print(f"bias(valF1={vb:.4f})={np.round(b,2).tolist()}")
        a1, f1v, per1, pb = rep("TTA+bias", pt, yte, bias=b)
        cm = confusion_matrix(yte, pb)
        print("confusion:"); print(cm)
        out.update({"bias": b.tolist(), "test_acc_bias": float(a1),
                    "test_macroF1_bias": float(f1v),
                    "per_class_f1_bias": [float(x) for x in per1],
                    "confusion_matrix_bias": cm.tolist()})
    torch.save(best_state, f"{args.out}/best_seqnoc_v2.pt")
    json.dump(out, open(f"{args.out}/seqnoc_v2_metrics.json", "w"), indent=2)
    print(f"\nsaved -> {args.out}/seqnoc_v2_metrics.json")


if __name__ == "__main__":
    sys.exit(main())

"""
Domain-adaptation experiment on the tokenized test set.

Split the test set into two donor-DISJOINT halves (A, B) via connected
components of the donor co-occurrence graph (each profile's donor-set lies
entirely inside one component, so assigning whole components to A/B loses
zero profiles and shares zero donors). Then:

    fold 1: finetune base ckpt on A  -> evaluate B
    fold 2: finetune base ckpt on B  -> evaluate A
    pooled: concatenate the two held-out folds -> full-test honest metric

Each fold reserves a donor-disjoint sub-slice of its own finetune half as a
val set for early-stop + per-class bias tuning, so nothing from the evaluated
half leaks.

Usage:
    python scripts/finetune_seqnoc_test.py \
        --data-dir /tmp/nfg/data/noc_filtered_grouped \
        --base results/seqnoc/best_seqnoc.pt --epochs 30 --tta-samples 20
"""

from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_seqnoc import PeakSetClassifier, tta_probs, tune_bias, load_split


def donor_components(donor_lists):
    sets = [set(x) for x in donor_lists]
    donors = sorted(set().union(*sets))
    par = {x: x for x in donors}
    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    def u(a, b): par[f(a)] = f(b)
    for s in sets:
        s = list(s)
        for i in range(1, len(s)):
            u(s[0], s[i])
    comp_of = {x: f(x) for x in donors}
    # profile -> component root (all donors share root)
    prof_comp = np.array([comp_of[next(iter(s))] for s in sets])
    return prof_comp


def binpack_two(prof_comp, seed=42):
    """Greedy: assign each component to the lighter bin (balance profile count)."""
    roots, counts = np.unique(prof_comp, return_counts=True)
    order = np.argsort(-counts)
    binA, binB, szA, szB = set(), set(), 0, 0
    for i in order:
        r, c = roots[i], counts[i]
        if szA <= szB:
            binA.add(r); szA += c
        else:
            binB.add(r); szB += c
    mA = np.array([r in binA for r in prof_comp])
    return mA, ~mA


def run_fold(base, Xtr, Mtr, ytr, Xte, Mte, yte, prof_comp_tr, args, dev, n_cls, n_loci):
    """Finetune base on (Xtr) with an internal donor-disjoint val, eval (Xte)."""
    # internal val = smallest components of the finetune half (donor-disjoint)
    mA, mB = binpack_two(prof_comp_tr)
    # use the smaller side as val
    if mA.sum() <= mB.sum():
        vm, tm = mA, mB
    else:
        vm, tm = mB, mA
    Xt, Mt, yt = Xtr[tm], Mtr[tm], ytr[tm]
    Xv, Mv, yv = Xtr[vm], Mtr[vm], ytr[vm]

    model = PeakSetClassifier(n_loci=n_loci, d_model=args.d_model, n_heads=args.heads,
                              n_layers=args.layers, n_classes=n_cls,
                              dropout=args.dropout).to(dev)
    model.load_state_dict(torch.load(base, map_location=dev))

    cnt = np.bincount(yt, minlength=n_cls).astype(np.float64)
    w = 1.0 / np.maximum(cnt, 1)
    sampler = WeightedRandomSampler(torch.as_tensor(w[yt], dtype=torch.double),
                                    len(yt), replacement=True)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xt), torch.from_numpy(Mt),
                                  torch.from_numpy(yt)),
                    batch_size=args.batch_size, sampler=sampler)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    @torch.no_grad()
    def vf1():
        model.eval()
        p = []
        for st in range(0, len(yv), args.batch_size):
            en = min(st + args.batch_size, len(yv))
            out = model(torch.from_numpy(Xv[st:en]).to(dev),
                        torch.from_numpy(Mv[st:en]).to(dev))
            p.append(out.argmax(1).cpu().numpy())
        return f1_score(yv, np.concatenate(p), average="macro", zero_division=0)

    best, best_state = -1, None
    for ep in range(args.epochs):
        model.train()
        for x, m, yy in dl:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x.to(dev), m.to(dev)), yy.to(dev))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        f = vf1()
        if f > best:
            best, best_state = f, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # TTA on val (for bias) + eval half
    pv = tta_probs(model, Xv, Mv, dev, args.tta_samples, args.batch_size, seed=11)
    pe = tta_probs(model, Xte, Mte, dev, args.tta_samples, args.batch_size, seed=22)
    bias, _ = tune_bias(pv, yv, n_cls)
    pred_nb = pe.argmax(1)
    pred_b = (np.log(pe + 1e-12) + bias[None, :]).argmax(1)
    return pred_nb, pred_b, yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/tmp/nfg/data/noc_filtered_grouped")
    ap.add_argument("--base", default="results/seqnoc/best_seqnoc.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--tta-samples", type=int, default=20)
    ap.add_argument("--out", default="results/seqnoc")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = args.data_dir
    Xte, Mte, yte = load_split(D, "test")
    donor = json.load(open(os.path.join(D, "donor_ids_test.json")))
    vocab = json.load(open(os.path.join(D, "vocab.json")))
    n_cls = vocab.get("n_noc", 5); n_loci = vocab.get("n_loci", 28)

    prof_comp = donor_components(donor)
    mA, mB = binpack_two(prof_comp)
    print(f"test={len(yte)}  halfA={mA.sum()} halfB={mB.sum()}  "
          f"donor-disjoint, 0 dropped")
    # verify disjoint donors
    dA = set().union(*[set(donor[i]) for i in np.where(mA)[0]])
    dB = set().union(*[set(donor[i]) for i in np.where(mB)[0]])
    print(f"donor overlap A∩B = {len(dA & dB)} (must be 0)")

    # fold 1: finetune A -> eval B ; fold 2: finetune B -> eval A
    print("\n--- fold 1: finetune A, eval B ---")
    pnb1, pb1, y1 = run_fold(args.base, Xte[mA], Mte[mA], yte[mA],
                             Xte[mB], Mte[mB], yte[mB], prof_comp[mA],
                             args, dev, n_cls, n_loci)
    print("--- fold 2: finetune B, eval A ---")
    pnb2, pb2, y2 = run_fold(args.base, Xte[mB], Mte[mB], yte[mB],
                             Xte[mA], Mte[mA], yte[mA], prof_comp[mB],
                             args, dev, n_cls, n_loci)

    yp = np.concatenate([y1, y2])
    pnb = np.concatenate([pnb1, pnb2])
    pb = np.concatenate([pb1, pb2])

    def rep(tag, pred):
        acc = accuracy_score(yp, pred); mf = f1_score(yp, pred, average="macro", zero_division=0)
        per = f1_score(yp, pred, average=None, zero_division=0)
        print(f"[{tag}] acc={acc:.4f} macroF1={mf:.4f} per-class={[round(float(x),3) for x in per]}")
        return acc, mf, per

    print("\n=== POOLED held-out (full test via 2 disjoint folds) ===")
    a0, f0, p0 = rep("FT + TTA (no bias)", pnb)
    a1, f1v, p1 = rep("FT + TTA + bias", pb)
    cm = confusion_matrix(yp, pb)
    print("confusion (rows=true 1..5):"); print(cm)

    out = {"halfA": int(mA.sum()), "halfB": int(mB.sum()),
           "pooled_acc_ttabias": float(a1), "pooled_macroF1_ttabias": float(f1v),
           "per_class_f1_bias": [float(x) for x in p1],
           "pooled_acc_tta": float(a0), "pooled_macroF1_tta": float(f0),
           "confusion_matrix_bias": cm.tolist()}
    with open(os.path.join(args.out, "seqnoc_ft_test_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {args.out}/seqnoc_ft_test_metrics.json")


if __name__ == "__main__":
    sys.exit(main())

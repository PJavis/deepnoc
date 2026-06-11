"""
Synthetic mixture generator for STR-DNA NoC training.

The PROVEDIt dataset has ~2700 single-source (NoC=1) profiles but only
~150-200 profiles per multi-contributor class. That gap is the main reason
deep models overfit on this task. This module exploits the fact that an
electropherogram from k contributors is, to first order, the per-locus
superposition of k independent single-source profiles, scaled by their
relative DNA quantities. By drawing k sources at random from the NoC=1 pool
and combining them with Dirichlet-sampled mass weights we can manufacture
an effectively unlimited supply of realistic-looking k-person profiles
WITH known ground truth (NoC, per-locus n_alleles, mix proportions).

Output layout (numpy memmap-friendly):
    data/synthetic/X.npy            [N, 24, 50, 89]   float32
    data/synthetic/y.npy            [N]               int64    (NoC 1..max_noc)
    data/synthetic/mix.npy          [N, MAX_NOC]      float32  sorted desc
    data/synthetic/locus_nall.npy   [N, 24]           int8     true n_alleles per locus

Usage:
    python -m src.synth                          # default 50_000 profiles
    python -m src.synth --n 200_000 --max-noc 6
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import (
    NUM_LOCI,
    MAX_PEAKS_PER_LOCUS,
    NUM_FEATURES_PER_PEAK,
    MAX_NOC,
    GLOBALFILER_LOCI,
    HEIGHT_NORM,
    ALLELE_NORM,
    SIZE_NORM,
    LOCUS_PEAK_NORM,
    PROFILE_PEAK_NORM,
    EXPECTED_BACK_STUTTER_RATIO,
    EXPECTED_DBL_BACK_STUTTER_RATIO,
    EXPECTED_FWD_STUTTER_RATIO,
    EXPECTED_PT2_STUTTER_RATIO,
)
from src.data_loader import build_profile_tensor


HEIGHT_FEAT_IDX = 26
ALLELE_FEAT_IDX = 24
SIZE_FEAT_IDX = 25


def _decode_source_profile(tensor: np.ndarray) -> list[list[Tuple[float, float, float]]]:
    """
    Decode a stored [24, 50, 89] tensor back into a list of (allele, size, height)
    triples per locus, in raw (denormalised) units.

    Returns: list of length 24, each item a list of (allele, size, height) for
    the real peaks at that locus.
    """
    out: list[list[Tuple[float, float, float]]] = []
    for li in range(NUM_LOCI):
        block = tensor[li]
        mask = block[:, HEIGHT_FEAT_IDX] > 0.0
        peaks = []
        for row in block[mask]:
            allele = float(row[ALLELE_FEAT_IDX]) * ALLELE_NORM
            size = float(row[SIZE_FEAT_IDX]) * SIZE_NORM
            height = float(row[HEIGHT_FEAT_IDX]) * HEIGHT_NORM
            peaks.append((allele, size, height))
        out.append(peaks)
    return out


def _superpose(sources: list[list[list[Tuple[float, float, float]]]],
               weights: np.ndarray,
               detection_threshold: float = 50.0,
               allele_tol: float = 0.05) -> dict[str, list[Tuple[float, float, float]]]:
    """
    Combine k decoded source profiles into one mixture.

    Args:
        sources:    list of length k, each a per-locus decoded profile
        weights:    [k] mass proportions, sum to 1
        detection_threshold: drop summed peaks below this RFU
        allele_tol: alleles within this much count as the same allele bin

    Returns:
        dict locus_name -> list of (allele, size, height) tuples for the mixture.
    """
    out: dict[str, list[Tuple[float, float, float]]] = {}
    for li, locus in enumerate(GLOBALFILER_LOCI):
        bucket: dict[float, dict] = {}
        for k_idx, src in enumerate(sources):
            for allele, size, height in src[li]:
                # Quantise allele to nearest 0.1 so half-repeats collapse cleanly.
                key = round(allele * 10) / 10
                slot = bucket.get(key)
                if slot is None:
                    bucket[key] = {"allele": allele, "size": size,
                                   "height": weights[k_idx] * height}
                else:
                    slot["height"] += weights[k_idx] * height
                    # Take size from the largest contributor for that allele.
                    if weights[k_idx] * height > slot["height"] / 2:
                        slot["size"] = size
        peaks = [(d["allele"], d["size"], d["height"])
                 for d in bucket.values()
                 if d["height"] >= detection_threshold]
        out[locus] = sorted(peaks, key=lambda t: t[1])  # sort by size like real data
    return out


def _add_artefacts(peaks_by_locus: dict[str, list[Tuple[float, float, float]]],
                   rng: np.random.Generator,
                   back_cv: float = 0.20,
                   dbl_cv: float = 0.30,
                   fwd_cv: float = 0.30,
                   pt2_cv: float = 0.40,
                   dropout_scale: float = 250.0,
                   dropout_max_p: float = 0.30,
                   noise_lambda: float = 0.5,
                   noise_height_min: float = 30.0,
                   noise_height_max: float = 80.0,
                   noise_allele_min: int = 8,
                   noise_allele_max: int = 35,
                   detection_threshold: float = 50.0,
                   ) -> dict[str, list[Tuple[float, float, float]]]:
    """
    Inject physical artefacts on top of the clean per-locus peak superposition.

    Per parent peak (a, s, h) we generate:
        back stutter   at a-1     with mean ratio EXPECTED_BACK_STUTTER_RATIO
        double back    at a-2     with mean ratio EXPECTED_DBL_BACK_STUTTER_RATIO
        forward stutter at a+1    with mean ratio EXPECTED_FWD_STUTTER_RATIO
        0.2 stutter    at a-0.2   with mean ratio EXPECTED_PT2_STUTTER_RATIO
    Each ratio is drawn from a truncated normal around the expected value
    (cv = std / mean). Stutter peaks below the LOD are dropped.

    Each parent peak has a height-dependent allelic dropout probability:
        p_drop(h) = dropout_max_p * exp(-h / dropout_scale)
    When the parent drops out we also skip its stutter children.

    A Poisson(noise_lambda) number of pull-up/spike noise peaks is added per
    locus at random integer alleles in the kit allele range, with heights
    drawn uniformly from [noise_height_min, noise_height_max]. These model
    drop-in artefacts that real electropherograms exhibit.

    Stutter and parent peaks at the same allele are then re-bucketed (their
    heights summed) before the final LOD filter — that's important because
    one contributor's stutter can land on another contributor's true allele.
    """
    out: dict[str, list[Tuple[float, float, float]]] = {}
    for locus, peaks in peaks_by_locus.items():
        bucket: dict[float, dict] = {}

        def _add(allele: float, size: float, height: float, is_parent: bool):
            if height <= 0.0:
                return
            key = round(allele * 10) / 10
            slot = bucket.get(key)
            if slot is None:
                bucket[key] = {"allele": allele, "size": size, "height": height,
                               "from_parent": is_parent}
            else:
                slot["height"] += height
                if is_parent and height > slot["height"] / 3:
                    slot["allele"] = allele
                    slot["size"] = size

        for a, s, h in peaks:
            # Allelic dropout.
            p_drop = dropout_max_p * math.exp(-h / max(dropout_scale, 1.0))
            if rng.random() < p_drop:
                continue
            _add(a, s, h, is_parent=True)

            r_back = max(0.0, rng.normal(EXPECTED_BACK_STUTTER_RATIO,
                                         EXPECTED_BACK_STUTTER_RATIO * back_cv))
            r_dbl = max(0.0, rng.normal(EXPECTED_DBL_BACK_STUTTER_RATIO,
                                        EXPECTED_DBL_BACK_STUTTER_RATIO * dbl_cv))
            r_fwd = max(0.0, rng.normal(EXPECTED_FWD_STUTTER_RATIO,
                                        EXPECTED_FWD_STUTTER_RATIO * fwd_cv))
            r_pt2 = max(0.0, rng.normal(EXPECTED_PT2_STUTTER_RATIO,
                                        EXPECTED_PT2_STUTTER_RATIO * pt2_cv))
            _add(a - 1.0, s - 4.0, h * r_back, is_parent=False)
            _add(a - 2.0, s - 8.0, h * r_dbl, is_parent=False)
            _add(a + 1.0, s + 4.0, h * r_fwd, is_parent=False)
            _add(a - 0.2, s - 0.8, h * r_pt2, is_parent=False)

        n_noise = int(rng.poisson(noise_lambda))
        for _ in range(n_noise):
            a = float(rng.integers(noise_allele_min, noise_allele_max + 1))
            s = 100.0 + a * 4.0
            h = float(rng.uniform(noise_height_min, noise_height_max))
            _add(a, s, h, is_parent=False)

        final = [(d["allele"], d["size"], d["height"])
                 for d in bucket.values()
                 if d["height"] >= detection_threshold]
        out[locus] = sorted(final, key=lambda t: t[1])
    return out


def _mixture_to_dataframe(peaks_by_locus: dict[str, list[Tuple[float, float, float]]]
                          ) -> pd.DataFrame:
    rows = []
    for locus, peaks in peaks_by_locus.items():
        for allele, size, height in peaks:
            rows.append({"Marker": locus, "Allele": allele,
                         "Size": size, "Height": height})
    if not rows:
        return pd.DataFrame(columns=["Marker", "Allele", "Size", "Height"])
    return pd.DataFrame(rows)


def _padded_mix(weights: np.ndarray, max_noc: int = MAX_NOC) -> np.ndarray:
    """Sort weights descending and pad to max_noc with a tiny epsilon, renormalise."""
    w = np.sort(weights)[::-1]
    pad = max(0, max_noc - w.size)
    out = np.concatenate([w, np.full(pad, 1e-4)])
    return (out / out.sum()).astype(np.float32)


def synthesise(
    pool: np.ndarray,
    n_samples: int,
    max_noc: int = 5,
    noc_weights: np.ndarray | None = None,
    dirichlet_alpha: float = 1.5,
    detection_threshold: float = 50.0,
    height_jitter_sigma: float = 0.08,
    artefacts: bool = True,
    artefact_kwargs: dict | None = None,
    rng_seed: int = 0,
    verbose: bool = True,
    out_dir: str | None = None,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None,
           np.ndarray | None]:
    """
    Generate `n_samples` synthetic profiles from a NoC=1 pool.

    Args:
        pool:                  [P, 24, 50, 89] float32 array of NoC=1 profiles.
        n_samples:             number of synthetic profiles to make.
        max_noc:               highest NoC to synthesise (inclusive).
        noc_weights:           optional [max_noc] over (1..max_noc); default
                               uniform so the synthetic set is balanced by NoC.
        dirichlet_alpha:       Dirichlet concentration on the mass simplex.
                               Higher = more balanced mixtures.
        detection_threshold:   drop summed peaks below this RFU after weighting.
        height_jitter_sigma:   log-normal multiplicative jitter on each peak
                               height (after superposition, before features
                               are recomputed). 0 disables.
        rng_seed:              numpy RNG seed.
        out_dir:               If set, stream outputs to memmap .npy files under
                             this directory (no giant RAM allocation for X).
                             Returns four None values; files are X.npy, y.npy,
                             mix.npy, locus_nall.npy.

    Returns:
        X, y, mix, nall arrays when out_dir is None; otherwise (None, None, None, None)
        and arrays are written under out_dir.
    """
    if noc_weights is None:
        noc_weights = np.ones(max_noc, dtype=np.float64) / max_noc
    noc_weights = np.asarray(noc_weights, dtype=np.float64)
    noc_weights = noc_weights / noc_weights.sum()
    rng = np.random.default_rng(rng_seed)

    # Pre-decode every NoC=1 profile once; this dominates the wall-clock cost
    # but afterwards each synthesised sample is cheap.
    if verbose:
        print(f"[synth] decoding {pool.shape[0]} NoC=1 source profiles…")
    decoded = [_decode_source_profile(pool[i]) for i in
               tqdm(range(pool.shape[0]), disable=not verbose, leave=False)]

    shape_x = (n_samples, NUM_LOCI, MAX_PEAKS_PER_LOCUS, NUM_FEATURES_PER_PEAK)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        # Use regular arrays instead of memmap to avoid loading issues
        X_out = np.zeros(shape_x, dtype=np.float32)
        y_out = np.zeros(n_samples, dtype=np.int64)
        mix_out = np.zeros((n_samples, MAX_NOC), dtype=np.float32)
        nall_out = np.zeros((n_samples, NUM_LOCI), dtype=np.int8)
        if verbose:
            print(f"[synth] using in-memory arrays → {out_dir}")
    else:
        X_out = np.zeros(shape_x, dtype=np.float32)
        y_out = np.zeros(n_samples, dtype=np.int64)
        mix_out = np.zeros((n_samples, MAX_NOC), dtype=np.float32)
        nall_out = np.zeros((n_samples, NUM_LOCI), dtype=np.int8)

    n_pool = len(decoded)
    if n_pool < max_noc:
        raise ValueError(f"Need at least {max_noc} NoC=1 profiles in pool, got {n_pool}.")

    written = 0
    rejected = 0
    pbar = tqdm(total=n_samples, disable=not verbose, desc="synth")
    while written < n_samples:
        k = int(rng.choice(np.arange(1, max_noc + 1), p=noc_weights))
        idx = rng.choice(n_pool, size=k, replace=False)
        sources = [decoded[i] for i in idx]
        w = rng.dirichlet(np.full(k, dirichlet_alpha))
        peaks_by_locus = _superpose(sources, w,
                                    detection_threshold=detection_threshold)

        # Snapshot true allele counts BEFORE artefacts so the aux n_alleles
        # head learns to predict the donor signal, not the artefact noise.
        true_nall = {loc: len(p) for loc, p in peaks_by_locus.items()}

        if artefacts:
            peaks_by_locus = _add_artefacts(
                peaks_by_locus, rng,
                detection_threshold=detection_threshold,
                **(artefact_kwargs or {}),
            )

        if height_jitter_sigma > 0:
            for locus, peaks in peaks_by_locus.items():
                jittered = []
                for a, s, h in peaks:
                    noise = float(rng.normal(0.0, height_jitter_sigma))
                    jittered.append((a, s, max(detection_threshold,
                                               h * float(np.exp(noise)))))
                peaks_by_locus[locus] = jittered

        df = _mixture_to_dataframe(peaks_by_locus)
        total_peaks = len(df)
        if total_peaks < 4:
            # All-dropout: too few peaks to be informative — resample.
            rejected += 1
            if rejected > 5 * n_samples + 100:
                raise RuntimeError("Synthesis rejection rate too high; lower "
                                   "detection_threshold or check the pool.")
            continue

        tensor = build_profile_tensor(df, total_peaks)
        X_out[written] = tensor
        y_out[written] = k
        mix_out[written] = _padded_mix(w)
        # Per-locus true n_alleles count, capped at 19 (CE class range 0..19).
        for li, locus in enumerate(GLOBALFILER_LOCI):
            nall_out[written, li] = min(true_nall.get(locus, 0), 19)

        written += 1
        pbar.update(1)
    pbar.close()

    if out_dir is not None:
        # Save arrays to disk
        np.save(os.path.join(out_dir, "X.npy"), X_out)
        np.save(os.path.join(out_dir, "y.npy"), y_out)
        np.save(os.path.join(out_dir, "mix.npy"), mix_out)
        np.save(os.path.join(out_dir, "locus_nall.npy"), nall_out)
        return None, None, None, None

    return X_out, y_out, mix_out, nall_out


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic NoC training data")
    ap.add_argument("--source", default="data/provedit_processed/X_gf25.npy",
                    help="Path to processed PROVEDIt X tensor")
    ap.add_argument("--labels", default="data/provedit_processed/y_gf25.npy",
                    help="Path to processed PROVEDIt y vector")
    ap.add_argument("--out-dir", default="data/synthetic",
                    help="Where to write X.npy / y.npy / mix.npy / locus_nall.npy")
    ap.add_argument("--n", type=int, default=50_000, help="Profiles to generate")
    ap.add_argument("--max-noc", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1.5,
                    help="Dirichlet concentration on mass simplex")
    ap.add_argument("--threshold", type=float, default=50.0,
                    help="Peak detection threshold in RFU")
    ap.add_argument("--jitter", type=float, default=0.08,
                    help="Per-peak log-normal height jitter sigma")
    ap.add_argument("--seed", type=int, default=0)
    # Leakage guard: drop test-split profiles from the NoC=1 pool so the
    # synthetic pretrain set never contains peaks from held-out test profiles.
    ap.add_argument("--exclude-test", action="store_true",
                    help="Build pool from TRAIN split only (needs --names)")
    ap.add_argument("--names", default="data/provedit_processed/sample_names.txt",
                    help="sample_names.txt, used by --exclude-test for the split")
    ap.add_argument("--split", default="grouped",
                    choices=["grouped", "stratified", "alternating"])
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--split-seed", type=int, default=42,
                    help="Seed for the train/test split (must match training)")
    args = ap.parse_args()

    print(f"[synth] loading {args.source}")
    X = np.load(args.source)
    y = np.load(args.labels)

    if args.exclude_test:
        with open(args.names) as f:
            names = [ln.rstrip("\n") for ln in f if ln.strip()]
        if len(names) != len(y):
            names = [str(i) for i in range(len(y))]
        if args.split == "grouped":
            from src.split import grouped_stratified_split
            Xtr, _, ytr, _, _, _ = grouped_stratified_split(
                X, y, names, test_size=args.test_size, seed=args.split_seed)
        elif args.split == "stratified":
            from src.split import stratified_split
            Xtr, _, ytr, _, _, _ = stratified_split(
                X, y, names, test_size=args.test_size, seed=args.split_seed)
        else:
            from src.data_loader import train_test_split_alternating
            Xtr, _, ytr, _, _, _ = train_test_split_alternating(X, y, names)
        pool = Xtr[ytr == 1].astype(np.float32)
        print(f"[synth] EXCLUDE-TEST: pool from TRAIN split only "
              f"({args.split}, seed={args.split_seed}, test_size={args.test_size})")
    else:
        pool = X[y == 1].astype(np.float32)
    print(f"[synth] NoC=1 pool: {pool.shape[0]} profiles")

    os.makedirs(args.out_dir, exist_ok=True)
    synthesise(
        pool, n_samples=args.n, max_noc=args.max_noc,
        dirichlet_alpha=args.alpha,
        detection_threshold=args.threshold,
        height_jitter_sigma=args.jitter,
        rng_seed=args.seed,
        out_dir=args.out_dir,
    )
    ys = np.load(os.path.join(args.out_dir, "y.npy"), allow_pickle=True)
    uniq, cnt = np.unique(np.asarray(ys), return_counts=True)
    print(f"[synth] wrote {args.n} profiles to {args.out_dir}")
    print(f"[synth] NoC distribution: {dict(zip(uniq.tolist(), cnt.tolist()))}")


if __name__ == "__main__":
    main()

"""
Augmentation pipeline for NoCFormer training.

The original deepNoC paper covers data scarcity by simulating 100 000 profiles
with the simDNAmixtures + GAN pipeline. We don't have that simulator. Instead
we exploit a forensically valid observation: an electropherogram from K
contributors is, to first order, the SUPERPOSITION of K independent
single-source profiles (with stochastic peak-height effects). PROVEDIt has
~2 700 single-source profiles, giving us C(2700, k) potential synthetic
k-person training examples — effectively unlimited.

The augmentations operate directly on the [24, 50, 89] tensor:

    `synthetic_mix(profiles, k)`
        Combine k single-source tensors at random per-contributor weights, sum
        their peak heights at each (locus, allele) position, rebuild the
        per-peak features that depend on neighbours (height ratios, locus peak
        counts, profile peak counts, mix props), and return a new tensor +
        label k.

    `peak_height_jitter(tensor)`
        Multiply each real peak's height by a log-normal noise term ~ LN(0, sigma).
        Models analytical variability; safe because the peak ordering is
        irrelevant to the set encoder.

    `random_peak_dropout(tensor, p)`
        Zero out a fraction `p` of real peaks. Models drop-out / threshold
        sensitivity.

    `random_noise_peaks(tensor, n)`
        Inject `n` low-amplitude noise peaks at empty positions. Models drop-in
        artefacts.

    `MixupBatch(...)`
        Convenience callable for use as a DataLoader collate_fn.

These are all pure tensor ops so they run on GPU within a training step.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from src.constants import (
    NUM_LOCI,
    MAX_PEAKS_PER_LOCUS,
    NUM_FEATURES_PER_PEAK,
    MAX_NOC,
    HEIGHT_NORM,
    LOCUS_PEAK_NORM,
    PROFILE_PEAK_NORM,
)


# Indices into the 89-dim peak feature vector — must match src/data_loader.py.
ALLELE_IDX = 24
SIZE_IDX = 25
HEIGHT_IDX = 26
PLP_IDX = 28
LOCUS_PEAKS_IDX = 77
PROFILE_PEAKS_IDX = 78
MIX_PROPS_SLICE = slice(79, 79 + MAX_NOC)


def _real_peak_mask(profile: torch.Tensor) -> torch.Tensor:
    """[24, 50] bool mask of peaks with non-zero height."""
    return profile[..., HEIGHT_IDX] > 0.0


def _recompute_global_features(profile: torch.Tensor,
                               mix_props: torch.Tensor | None = None
                               ) -> torch.Tensor:
    """
    After modifying heights / adding peaks, the locus-peak-count and
    profile-peak-count features go stale. Recompute them in-place on a clone.
    """
    p = profile.clone()
    mask = _real_peak_mask(p)
    locus_counts = mask.sum(dim=-1)                     # [24]
    total = mask.sum().item()
    for li in range(NUM_LOCI):
        n = locus_counts[li].item()
        if n == 0:
            continue
        p[li, :n, LOCUS_PEAKS_IDX] = float(n) / LOCUS_PEAK_NORM
        p[li, :n, PROFILE_PEAKS_IDX] = float(total) / PROFILE_PEAK_NORM
        if mix_props is not None:
            p[li, :n, MIX_PROPS_SLICE] = mix_props
    return p


def _sort_real_first(locus_block: torch.Tensor) -> torch.Tensor:
    """
    Sort a [50, 89] locus block so that real peaks (height>0) come first,
    preserving their original relative order. Padding stays at the back.
    """
    heights = locus_block[:, HEIGHT_IDX]
    real = heights > 0
    real_block = locus_block[real]
    pad_block = locus_block[~real]
    return torch.cat([real_block, pad_block], dim=0)


def synthetic_mix(profiles: list[torch.Tensor],
                  weights: torch.Tensor | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a synthetic mixture of k single-source profiles by superposing peaks
    at the (locus, allele) level.

    Args:
        profiles: list of `k` single-source tensors, each [24, 50, 89]
        weights:  optional tensor of length `k` summing to 1; default = uniform

    Returns:
        mixed_tensor: [24, 50, 89]
        mix_props:    [MAX_NOC] (sorted descending, padded with small ε)
    """
    k = len(profiles)
    if weights is None:
        # Sample weights from a Dirichlet to look like realistic mixture ratios.
        # Using alpha=1 gives uniform on the simplex; tighter alpha would skew it.
        w = torch.distributions.Dirichlet(torch.ones(k)).sample()
    else:
        w = weights / weights.sum()

    out = torch.zeros(NUM_LOCI, MAX_PEAKS_PER_LOCUS, NUM_FEATURES_PER_PEAK,
                      dtype=profiles[0].dtype)

    for li in range(NUM_LOCI):
        # Collect (allele, weighted_height_rfu, source_profile_features) per peak.
        bucket: dict[float, dict] = {}
        for src_idx, prof in enumerate(profiles):
            block = prof[li]
            mask = block[:, HEIGHT_IDX] > 0
            if not mask.any():
                continue
            for row in block[mask]:
                allele = round(float(row[ALLELE_IDX].item()) * 100, 2)
                h_rfu = float(row[HEIGHT_IDX].item()) * HEIGHT_NORM * float(w[src_idx])
                if allele not in bucket:
                    bucket[allele] = {"height": 0.0, "feat": row.clone()}
                bucket[allele]["height"] += h_rfu

        if not bucket:
            continue

        # Write the merged peaks back into the locus block (truncate to MAX_PEAKS).
        items = sorted(bucket.items(), key=lambda kv: kv[0])
        for slot, (_, info) in enumerate(items[:MAX_PEAKS_PER_LOCUS]):
            f = info["feat"].clone()
            f[HEIGHT_IDX] = info["height"] / HEIGHT_NORM
            out[li, slot] = f

    # Build the [MAX_NOC] mix_props vector for the new label.
    mix = torch.zeros(MAX_NOC)
    sorted_w, _ = torch.sort(w, descending=True)
    mix[: sorted_w.numel()] = sorted_w
    mix[sorted_w.numel():] = 1e-3 / max(MAX_NOC - sorted_w.numel(), 1)
    mix = mix / mix.sum()

    out = _recompute_global_features(out, mix_props=mix)
    return out, mix


def peak_height_jitter(profile: torch.Tensor, sigma: float = 0.15
                       ) -> torch.Tensor:
    """Log-normal multiplicative jitter on real peak heights."""
    p = profile.clone()
    mask = _real_peak_mask(p)
    if not mask.any():
        return p
    noise = torch.randn(int(mask.sum()), device=p.device) * sigma
    p[..., HEIGHT_IDX][mask] = p[..., HEIGHT_IDX][mask] * torch.exp(noise)
    p[..., HEIGHT_IDX].clamp_(min=0.0, max=1.0)
    return _recompute_global_features(p)


def random_peak_dropout(profile: torch.Tensor, p: float = 0.05) -> torch.Tensor:
    """Randomly zero out real peaks with probability p."""
    out = profile.clone()
    mask = _real_peak_mask(out)
    if not mask.any() or p <= 0:
        return out
    drop = (torch.rand_like(out[..., HEIGHT_IDX]) < p) & mask
    out[..., HEIGHT_IDX][drop] = 0.0
    # Also zero everything else for those slots so they look like padding.
    out[drop] = 0.0
    return _recompute_global_features(out)


def random_noise_peaks(profile: torch.Tensor, n: int = 2,
                       height_scale: float = 0.05) -> torch.Tensor:
    """Inject a few low-intensity noise peaks at empty locus slots."""
    out = profile.clone()
    for li in range(NUM_LOCI):
        block = out[li]
        mask = block[:, HEIGHT_IDX] > 0
        empty = (~mask).nonzero(as_tuple=False).squeeze(-1)
        if empty.numel() == 0:
            continue
        n_insert = min(int(n), int(empty.numel()))
        if n_insert == 0:
            continue
        chosen = empty[torch.randperm(empty.numel())[:n_insert]]
        for idx in chosen:
            out[li, idx, ALLELE_IDX] = 0.0
            out[li, idx, SIZE_IDX] = 0.0
            out[li, idx, HEIGHT_IDX] = float(torch.rand(1).item()) * height_scale
    return _recompute_global_features(out)


def shuffle_peak_axis(profile: torch.Tensor) -> torch.Tensor:
    """
    Randomly permute peaks within each locus. Safe because the model is
    permutation-invariant; trains the model not to depend on the data
    pipeline's choice of ordering.
    """
    out = profile.clone()
    for li in range(NUM_LOCI):
        block = out[li]
        mask = block[:, HEIGHT_IDX] > 0
        n = int(mask.sum().item())
        if n <= 1:
            continue
        perm = torch.randperm(n)
        real = block[mask][perm]
        pad = block[~mask]
        out[li] = torch.cat([real, pad], dim=0)
    return out


@dataclass
class AugmentConfig:
    p_mix: float = 0.5                  # probability of synthesising a mixture per sample
    max_synth_noc: int = 5              # only synthesise up to this NoC
    p_jitter: float = 0.8
    jitter_sigma: float = 0.12
    p_dropout: float = 0.5
    dropout_rate: float = 0.03
    p_noise: float = 0.30
    noise_peaks: int = 2
    noise_height_scale: float = 0.05
    shuffle_peaks: bool = True
    p_mixup: float = 0.3               # MixUp probability per batch
    mixup_alpha: float = 1.0            # Beta distribution parameter


class TrainingAugmenter:
    """
    Apply augmentations to a batch coming out of the underlying single-source +
    mixture dataset. Uses a pool of single-source tensors to synthesise mixtures.

    Intended call site: inside the training loop after sampling a batch.
    """

    def __init__(self, single_source_pool: torch.Tensor,
                 cfg: AugmentConfig | None = None):
        """
        Args:
            single_source_pool: [N, 24, 50, 89] tensor of NoC=1 profiles.
        """
        self.pool = single_source_pool
        self.cfg = cfg or AugmentConfig()

    def _maybe_synth(self, x: torch.Tensor, y: int
                     ) -> tuple[torch.Tensor, int, torch.Tensor | None]:
        if (torch.rand(()).item() < self.cfg.p_mix
                and self.pool is not None and self.pool.numel() > 0):
            k = int(np.random.randint(1, self.cfg.max_synth_noc + 1))
            idx = np.random.choice(self.pool.size(0), size=k, replace=False)
            sources = [self.pool[i] for i in idx]
            mixed, mix = synthetic_mix(sources)
            return mixed, k, mix
        return x, y, None

    def __call__(self, batch_x: torch.Tensor, batch_y: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns augmented (X, y, mix_props_target). mix_props_target is
        [B, MAX_NOC] (uniform-on-NoC for non-synthesised samples).
        """
        b = batch_x.size(0)
        out_x = []
        out_y = []
        out_mix = []
        for i in range(b):
            x = batch_x[i]
            y = int(batch_y[i].item())
            x, y, mix = self._maybe_synth(x, y)
            if self.cfg.p_jitter > 0 and torch.rand(()).item() < self.cfg.p_jitter:
                x = peak_height_jitter(x, sigma=self.cfg.jitter_sigma)
            if self.cfg.p_dropout > 0 and torch.rand(()).item() < self.cfg.p_dropout:
                x = random_peak_dropout(x, p=self.cfg.dropout_rate)
            if self.cfg.p_noise > 0 and torch.rand(()).item() < self.cfg.p_noise:
                x = random_noise_peaks(x, n=self.cfg.noise_peaks,
                                      height_scale=self.cfg.noise_height_scale)
            if self.cfg.shuffle_peaks:
                x = shuffle_peak_axis(x)
            if mix is None:
                # Default mix-props target for natural samples: uniform 1/y over y donors.
                mix = torch.zeros(MAX_NOC)
                mix[:y] = 1.0 / y
            out_x.append(x)
            out_y.append(y)
            out_mix.append(mix)
        return (torch.stack(out_x),
                torch.tensor(out_y, dtype=torch.long),
                torch.stack(out_mix))


class MixUp:
    """
    MixUp data augmentation: linearly interpolate profiles in feature space.
    
    For batch B profiles, randomly select another sample and mix:
        x_mixed = λ * x_i + (1 - λ) * x_j  where λ ~ Beta(α, α)
        y_mixed blended as soft labels (kept as ordinal class for ordinal loss compatibility)
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Beta distribution parameter (higher = more uniform mixing, 
                   lower = more extreme mixing)
        """
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply MixUp augmentation to a batch.
        
        Args:
            x: [B, 24, 50, 89] batch of profiles
            y: [B] batch of integer NoC labels (1 to MAX_NOC)
        
        Returns:
            x_mixed: [B, 24, 50, 89] interpolated profiles
            y_mixed: [B] blended labels (as float, clamped to valid range)
            lam:     [B] mixing coefficients (for weighted loss, if needed)
        """
        batch_size = x.size(0)
        lam = torch.from_numpy(
            np.random.beta(self.alpha, self.alpha, batch_size)
        ).to(x.dtype).to(x.device)
        
        idx = torch.randperm(batch_size).to(x.device)
        x_mixed = lam.view(-1, 1, 1, 1) * x + (1 - lam).view(-1, 1, 1, 1) * x[idx]
        
        # Blend labels (keep in [1, MAX_NOC] range)
        y_float = y.float()
        y_mixed = lam * y_float + (1 - lam) * y_float[idx]
        y_mixed = torch.clamp(y_mixed, 1.0, float(MAX_NOC))
        
        return x_mixed, y_mixed, lam

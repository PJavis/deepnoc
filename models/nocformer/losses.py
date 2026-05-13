"""
Losses for NoCFormer.

The primary classifier uses the CORN ordinal head from
    Cao, Mirjalili, Raschka (2020) "Rank Consistent Ordinal Regression for
    Neural Networks with Application to Age Estimation"
which respects the ordering of NoC = 1, 2, ..., K and is rank-monotonic by
construction (no contradictory probabilities can occur).

Auxiliary heads (peak prop_allelic, peak n_alleles, locus n_alleles,
locus mix_props, profile mix_props) are wired into a multi-task loss with
small weights — primarily for explainability, secondarily for regularisation.

A class-balanced focal weight (Cui et al. 2019) reweights the CORN binary
losses to counter the heavy NoC=1 imbalance in PROVEDIt without throwing
away samples.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def corn_loss(logits: torch.Tensor, targets: torch.Tensor,
              num_classes: int,
              class_weights: torch.Tensor | None = None,
              focal_gamma: float = 0.0,
              label_smoothing: float = 0.1) -> torch.Tensor:
    """
    CORN loss with label smoothing.

    logits:   [B, K-1] = log-odds of P(y > k) for k = 1..K-1
    targets:  [B]      with values in 1..K (1-indexed NoC labels)
    class_weights: optional [K] weights applied per sample by its true class
    focal_gamma:   focal-loss focusing parameter; 0 disables focal weighting
    label_smoothing: smoothing factor (0-1); smooths binary targets toward 0.5

    Note: each task k uses ONLY the samples whose target > k-1, i.e. the
    "conditional training" trick that makes CORN rank-monotonic.
    """
    b = logits.size(0)
    device = logits.device
    targets0 = targets - 1                                # 0..K-1
    total = logits.new_zeros(())
    n_terms = 0
    for k in range(num_classes - 1):
        # task k: predict whether y > k (i.e. y >= k+1, 0-indexed: targets0 > k)
        mask = targets0 > (k - 1)                         # samples used for this task
        if not mask.any():
            continue
        z = logits[mask, k]
        y = (targets0[mask] > k).float()
        # Apply label smoothing
        y_smooth = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
        bce = F.binary_cross_entropy_with_logits(z, y_smooth, reduction="none")
        if focal_gamma > 0:
            # The focusing term itself does not need grad — but the
            # multiplication on `bce` MUST stay in the graph.
            with torch.no_grad():
                p = torch.sigmoid(z)
                pt = torch.where(y > 0.5, p, 1.0 - p)
                focal_w = (1.0 - pt).pow(focal_gamma)
            bce = bce * focal_w
        if class_weights is not None:
            # weight each kept sample by the weight of its true class
            cls = targets0[mask].long()  # ensure long for indexing
            bce = bce * class_weights[cls]
        total = total + bce.mean()
        n_terms += 1
    if n_terms == 0:
        return logits.new_zeros((), requires_grad=True)
    return total / n_terms


def corn_label_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """1-indexed predicted NoC from CORN logits."""
    sig = torch.sigmoid(logits)
    cum = torch.cumprod(sig, dim=1)
    return (cum > 0.5).sum(dim=1) + 1


def class_balanced_weights(class_counts: torch.Tensor, beta: float = 0.999
                           ) -> torch.Tensor:
    """
    Cui et al. (2019) class-balanced reweighting.
    Returns weights such that w_c ∝ (1 - beta) / (1 - beta^{n_c}), normalised
    so they sum to len(class_counts).
    """
    eff = 1.0 - torch.pow(beta, class_counts.clamp(min=1).float())
    w = (1.0 - beta) / eff
    w = w * (class_counts.numel() / w.sum())
    return w


class NoCFormerLoss(nn.Module):
    """
    Multi-task loss.

    Targets dict uses these keys (all optional except profile_noc):
        profile_noc:        [B]       (1-indexed)
        profile_mix_props:  [B, K]    (rows sum to 1)
        locus_n_alleles:    [B, 24]   (-1 = ignore; 1..20)
        locus_mix_props:    [B, 24, K]
        peak_prop_allelic:  [B, 24, 50, 1] in [0,1]
        peak_n_alleles:     [B, 24, 50]    (-1 = ignore; 0..20)
        peak_mask:          [B, 24, 50]    bool, True where the peak is real
    """

    def __init__(
        self,
        num_classes: int,
        noc_weight: float = 1.0,
        peak_prop_weight: float = 0.05,
        peak_nall_weight: float = 0.05,
        locus_nall_weight: float = 0.10,
        locus_mix_weight: float = 0.05,
        profile_mix_weight: float = 0.10,
        class_counts: torch.Tensor | None = None,
        focal_gamma: float = 1.0,
        cb_beta: float = 0.999,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.noc_weight = noc_weight
        self.peak_prop_weight = peak_prop_weight
        self.peak_nall_weight = peak_nall_weight
        self.locus_nall_weight = locus_nall_weight
        self.locus_mix_weight = locus_mix_weight
        self.profile_mix_weight = profile_mix_weight
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

        if class_counts is not None:
            w = class_balanced_weights(class_counts, beta=cb_beta)
        else:
            w = torch.ones(num_classes)
        self.register_buffer("class_weights", w)

    def forward(self, outputs: dict, targets: dict) -> dict:
        losses: dict[str, torch.Tensor] = {}
        total = outputs["profile_noc_logits"].new_zeros(())

        # ---- 1. NoC (CORN ordinal) -------------------------------------
        if "profile_noc" in targets:
            losses["noc"] = corn_loss(
                outputs["profile_noc_logits"],
                targets["profile_noc"],
                num_classes=self.num_classes,
                class_weights=self.class_weights,
                focal_gamma=self.focal_gamma,
                label_smoothing=self.label_smoothing,
            )
            total = total + self.noc_weight * losses["noc"]

        # ---- 2. Profile mixture proportions (MSE) -----------------------
        if "profile_mix_props" in targets:
            losses["profile_mix"] = F.mse_loss(
                outputs["profile_mix_props"], targets["profile_mix_props"]
            )
            total = total + self.profile_mix_weight * losses["profile_mix"]

        # ---- 3. Locus n_alleles (CE w/ ignore_index) --------------------
        if "locus_n_alleles" in targets:
            pred = outputs["locus_n_alleles"].view(-1, 20)
            tgt = targets["locus_n_alleles"].view(-1)
            mask = tgt >= 0
            if mask.any():
                losses["locus_nall"] = F.cross_entropy(pred[mask], tgt[mask])
                total = total + self.locus_nall_weight * losses["locus_nall"]

        # ---- 4. Locus mix_props (MSE) -----------------------------------
        if "locus_mix_props" in targets:
            losses["locus_mix"] = F.mse_loss(
                outputs["locus_mix_props"], targets["locus_mix_props"]
            )
            total = total + self.locus_mix_weight * losses["locus_mix"]

        # ---- 5. Peak prop_allelic (MSE on real peaks only) --------------
        if "peak_prop_allelic" in targets:
            pred = outputs["peak_prop_allelic"]
            tgt = targets["peak_prop_allelic"]
            pmask = targets.get("peak_mask")
            if pmask is not None:
                m = pmask.unsqueeze(-1).expand_as(pred)
                if m.any():
                    losses["peak_prop"] = F.mse_loss(pred[m], tgt[m])
                    total = total + self.peak_prop_weight * losses["peak_prop"]
            else:
                losses["peak_prop"] = F.mse_loss(pred, tgt)
                total = total + self.peak_prop_weight * losses["peak_prop"]

        # ---- 6. Peak n_alleles (CE on real peaks) ------------------------
        if "peak_n_alleles" in targets:
            pred = outputs["peak_n_alleles"].view(-1, 21)
            tgt = targets["peak_n_alleles"].view(-1)
            mask = tgt >= 0
            if mask.any():
                losses["peak_nall"] = F.cross_entropy(pred[mask], tgt[mask])
                total = total + self.peak_nall_weight * losses["peak_nall"]

        losses["total"] = total
        return losses

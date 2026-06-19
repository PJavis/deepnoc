"""
Multi-task loss for NoCNet-v2.

  total = w_cls * cross_entropy(class, NoC)
        + w_reg * smooth_l1(scalar, NoC)
        + w_corn * corn_bce(corn_logits, NoC)
        + w_mix * KL(pred_mix || true_mix)              [synthetic only]
        + w_nall * cross_entropy(locus_nall, true)      [synthetic only]

Design notes:
  * The three NoC heads supervise the SAME label from different angles —
    classification, regression, ordinal — which acts as a strong implicit
    regulariser and gives a calibrated ensemble at inference.
  * Class-balanced reweighting (Cui et al. 2019, beta=0.999) is applied to
    the cross-entropy and CORN losses. Focal stacking is intentionally
    disabled — combining focal + CB weights was destabilising NoCFormer.
  * Auxiliary losses (mix_props, locus_n_alleles) are masked to samples
    flagged synthetic, where the ground truth is exact. Real PROVEDIt
    samples don't have these labels so they only contribute the NoC losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def class_balanced_weights(class_counts: torch.Tensor, beta: float = 0.999
                           ) -> torch.Tensor:
    eff = 1.0 - torch.pow(beta, class_counts.clamp(min=1).float())
    w = (1.0 - beta) / eff
    return w * (class_counts.numel() / w.sum())


def corn_loss(logits: torch.Tensor, targets: torch.Tensor,
              num_classes: int,
              class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    CORN cumulative-loss without focal weighting.

    logits:    [B, K-1]
    targets:   [B] in 1..K
    """
    targets0 = targets - 1
    total = logits.new_zeros(())
    n_terms = 0
    for k in range(num_classes - 1):
        mask = targets0 > (k - 1)
        if not mask.any():
            continue
        z = logits[mask, k]
        y = (targets0[mask] > k).float()
        bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
        if class_weights is not None:
            bce = bce * class_weights[targets0[mask]]
        total = total + bce.mean()
        n_terms += 1
    if n_terms == 0:
        return logits.new_zeros((), requires_grad=True)
    return total / n_terms


def corn_label(logits: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(logits)
    cum = torch.cumprod(sig, dim=1)
    return (cum > 0.5).sum(dim=1) + 1


class NoCNetV2Loss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 class_counts: torch.Tensor | None = None,
                 w_cls: float = 1.0,
                 w_reg: float = 0.3,
                 w_corn: float = 0.5,
                 w_mix: float = 0.1,
                 w_nall: float = 0.1,
                 cb_beta: float = 0.999,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.num_classes = num_classes
        self.w_cls = w_cls
        self.w_reg = w_reg
        self.w_corn = w_corn
        self.w_mix = w_mix
        self.w_nall = w_nall
        self.label_smoothing = label_smoothing
        if class_counts is None:
            w = torch.ones(num_classes)
        else:
            w = class_balanced_weights(class_counts, beta=cb_beta)
        self.register_buffer("class_weights", w)

    def forward(self, outputs: dict, targets: dict) -> dict:
        noc = targets["profile_noc"]                          # [B] 1..K
        losses: dict[str, torch.Tensor] = {}
        total = outputs["profile_noc_logits"].new_zeros(())

        # Classification ------------------------------------------------------
        losses["cls"] = F.cross_entropy(
            outputs["profile_noc_logits"], noc - 1,
            weight=self.class_weights, label_smoothing=self.label_smoothing,
        )
        total = total + self.w_cls * losses["cls"]

        # Smooth-L1 regression on the scalar head ----------------------------
        losses["reg"] = F.smooth_l1_loss(outputs["profile_noc_scalar"],
                                         noc.float())
        total = total + self.w_reg * losses["reg"]

        # CORN ordinal -------------------------------------------------------
        losses["corn"] = corn_loss(outputs["profile_noc_corn"], noc,
                                   num_classes=self.num_classes,
                                   class_weights=self.class_weights)
        total = total + self.w_corn * losses["corn"]

        # Mix props (synthetic only) -----------------------------------------
        if "profile_mix_props" in targets and "is_synth" in targets:
            mask = targets["is_synth"]
            if mask.any():
                pred = outputs["profile_mix_props"][mask].clamp(min=1e-8)
                tgt = targets["profile_mix_props"][mask].clamp(min=1e-8)
                # Forward KL: tgt * (log tgt - log pred)
                kl = (tgt * (tgt.log() - pred.log())).sum(dim=-1).mean()
                losses["mix"] = kl
                total = total + self.w_mix * kl

        # Locus n_alleles (synthetic only) -----------------------------------
        if "locus_n_alleles" in targets and "is_synth" in targets:
            mask = targets["is_synth"]
            if mask.any():
                pred = outputs["locus_n_alleles"][mask].reshape(-1, 20)
                tgt = targets["locus_n_alleles"][mask].reshape(-1).long()
                ok = (tgt >= 0) & (tgt < 20)
                if ok.any():
                    losses["nall"] = F.cross_entropy(pred[ok], tgt[ok])
                    total = total + self.w_nall * losses["nall"]

        losses["total"] = total
        return losses

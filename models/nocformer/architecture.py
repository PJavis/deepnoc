"""
NoCFormer: hierarchical set-Transformer for STR DNA NoC assignment.

Design goals (vs the original deepNoC CNN):
  - Treat the 50 peaks at each locus as a SET, not a sequence (peak axis is
    arbitrarily ordered by Size in the data pipeline). This is enforced by
    self-attention with a key-padding mask derived from the height feature.
  - Model cross-locus correlations explicitly with a second Transformer that
    operates on 24 locus tokens augmented by dye/positional embeddings.
  - Keep the explainability spirit of deepNoC by exposing peak-, locus-, and
    profile-level auxiliary heads, but on top of the new backbone.
  - Output an ordinal head (CORN) instead of categorical softmax over 1..10
    so misclassifying NoC=4 as NoC=3 is penalised less than NoC=4 as NoC=9.

Input:  [B, 24, 50, 89]
Output: dict with keys
    profile_noc_logits      [B, num_classes - 1]   CORN-style cumulative logits
    profile_noc_probs       [B, num_classes]       derived class probabilities
    profile_mix_props       [B, MAX_NOC]           softmax over donors
    locus_n_alleles         [B, 24, 20]            CE logits
    locus_mix_props         [B, 24, MAX_NOC]       softmax per locus
    peak_prop_allelic       [B, 24, 50, 1]         sigmoid in [0,1]
    peak_n_alleles          [B, 24, 50, 21]        CE logits

The CORN head produces (num_classes - 1) cumulative-probability logits
P(y > k) for k = 1..num_classes-1. See Cao, Mirjalili, Raschka (2020).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import (
    NUM_LOCI,
    MAX_PEAKS_PER_LOCUS,
    NUM_FEATURES_PER_PEAK,
    MAX_NOC,
    DYE_CHANNELS,
    GLOBALFILER_LOCI,
)


# Index of the "height (rfu / 33000)" feature inside the 89-dim peak vector.
# Matches src/data_loader.build_peak_features(): features[i, 26] = height/HEIGHT_NORM.
HEIGHT_FEATURE_IDX = 26


def _build_dye_index() -> list[int]:
    """Map each of the 24 loci to its dye channel index 0..len(DYE_CHANNELS)-1."""
    dye_keys = list(DYE_CHANNELS.keys())
    locus_to_dye = {}
    for d_idx, d_key in enumerate(dye_keys):
        for locus in DYE_CHANNELS[d_key]:
            locus_to_dye[locus] = d_idx
    return [locus_to_dye.get(locus, 0) for locus in GLOBALFILER_LOCI]


DYE_INDEX = _build_dye_index()  # length 24


class PeakEmbedder(nn.Module):
    """Per-peak feature MLP: 89-dim → d_model."""

    def __init__(self, in_features: int = NUM_FEATURES_PER_PEAK,
                 d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features] → [..., d_model]
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class PeakSetEncoder(nn.Module):
    """
    Permutation-invariant peak encoder for one locus.

    Adds a learnable [CLS] (= "locus token") and runs `n_layers` of self-attention
    over (CLS + 50 peaks) with a key-padding mask so zero-pad peaks contribute
    nothing. The CLS embedding becomes the locus representation.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)

    def forward(self, peaks: torch.Tensor,
                peak_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        peaks:     [N, P, d_model]  (N = batch * loci, P = MAX_PEAKS_PER_LOCUS)
        peak_mask: [N, P]           True where peak is REAL, False for padding
        returns:
            locus_token: [N, d_model]
            peak_tokens: [N, P, d_model]  (post-attention peak embeddings)
        """
        n = peaks.size(0)
        cls = self.cls.expand(n, -1, -1)                    # [N, 1, D]
        x = torch.cat([cls, peaks], dim=1)                  # [N, P+1, D]

        # key_padding_mask in PyTorch attention: True = ignore.
        # CLS is always attended; peaks are ignored if mask is False.
        cls_keep = torch.ones(n, 1, dtype=torch.bool, device=peaks.device)
        kp = torch.cat([cls_keep, peak_mask], dim=1)         # [N, P+1] True = keep
        kp_mask = ~kp                                        # invert for PyTorch

        # If a sample has zero real peaks the mask would block everything.
        # Force at least the CLS to be attended (already true) — no further fix needed
        # because attention output for padded queries is masked out downstream anyway.

        for block in self.blocks:
            x = block(x, key_padding_mask=kp_mask)

        x = self.ln(x)
        return x[:, 0], x[:, 1:]


class LocusTransformer(nn.Module):
    """Cross-locus self-attention over the 24 locus tokens."""

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, dropout: float = 0.1, num_dyes: int = 5):
        super().__init__()
        self.locus_pos = nn.Parameter(torch.zeros(1, NUM_LOCI, d_model))
        self.dye_emb = nn.Embedding(num_dyes, d_model)
        nn.init.trunc_normal_(self.locus_pos, std=0.02)

        self.profile_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.profile_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)

        self.register_buffer(
            "dye_idx",
            torch.tensor(DYE_INDEX, dtype=torch.long),
            persistent=False,
        )

    def forward(self, locus_tokens: torch.Tensor,
                locus_active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        locus_tokens: [B, 24, d_model]
        locus_active: [B, 24]  True = locus has at least one real peak
        returns:
            profile_token: [B, d_model]
            locus_tokens:  [B, 24, d_model] (refined)
        """
        b = locus_tokens.size(0)
        dye = self.dye_emb(self.dye_idx).unsqueeze(0)        # [1, 24, D]
        x = locus_tokens + self.locus_pos + dye               # [B, 24, D]

        prof = self.profile_token.expand(b, -1, -1)           # [B, 1, D]
        x = torch.cat([prof, x], dim=1)                       # [B, 25, D]

        prof_keep = torch.ones(b, 1, dtype=torch.bool, device=x.device)
        kp = torch.cat([prof_keep, locus_active], dim=1)
        kp_mask = ~kp

        for block in self.blocks:
            x = block(x, key_padding_mask=kp_mask)

        x = self.ln(x)
        return x[:, 0], x[:, 1:]


class PeakHead(nn.Module):
    """Peak-level auxiliary outputs (proportion allelic + n_alleles)."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.prop = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.nall = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 21),
        )

    def forward(self, peak_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # peak_tokens: [N, P, D]
        prop = torch.sigmoid(self.prop(peak_tokens))           # [N, P, 1]
        nall = self.nall(peak_tokens)                          # [N, P, 21]
        return prop, nall


class LocusHead(nn.Module):
    """Locus-level auxiliary outputs (mix props always over MAX_NOC donors)."""

    def __init__(self, d_model: int = 128, mix_dim: int = MAX_NOC):
        super().__init__()
        self.nall = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 20),
        )
        self.mix = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, mix_dim),
        )

    def forward(self, locus_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.nall(locus_tokens), F.softmax(self.mix(locus_tokens), dim=-1)


class ProfileHead(nn.Module):
    """
    Profile-level outputs.

    Two dimensionalities:
      - num_classes: size of the NoC classification head (CORN emits K-1 logits)
      - mix_dim:     size of the donor-mixture-proportions head (always MAX_NOC
                     so the head matches the paper's [10] convention regardless
                     of how many NoC classes the dataset actually contains).
    """

    def __init__(self, d_model: int = 128, num_classes: int = MAX_NOC,
                 mix_dim: int = MAX_NOC):
        super().__init__()
        self.num_classes = num_classes
        # CORN: produces (num_classes - 1) binary tasks P(y > k)
        self.noc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes - 1),
        )
        self.mix = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, mix_dim),
        )

    def forward(self, profile_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.noc(profile_token), F.softmax(self.mix(profile_token), dim=-1)


class NoCFormer(nn.Module):
    """
    Hierarchical set-Transformer for NoC assignment.

    Args:
        d_model:        embedding dimension throughout
        n_heads:        attention heads
        peak_layers:    self-attention layers inside each locus
        locus_layers:   self-attention layers across loci
        num_classes:    NoC range; head emits CORN logits of size (num_classes - 1)
        dropout:        dropout used everywhere; raise for MC-Dropout uncertainty
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        peak_layers: int = 2,
        locus_layers: int = 4,
        num_classes: int = MAX_NOC,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        self.peak_embed = PeakEmbedder(
            in_features=NUM_FEATURES_PER_PEAK, d_model=d_model, dropout=dropout
        )
        self.peak_set = PeakSetEncoder(
            d_model=d_model, n_heads=n_heads, n_layers=peak_layers, dropout=dropout
        )
        self.peak_head = PeakHead(d_model)

        self.locus_tx = LocusTransformer(
            d_model=d_model, n_heads=n_heads, n_layers=locus_layers,
            dropout=dropout, num_dyes=len(DYE_CHANNELS),
        )
        self.locus_head = LocusHead(d_model, mix_dim=MAX_NOC)
        self.profile_head = ProfileHead(d_model, num_classes=num_classes,
                                        mix_dim=MAX_NOC)

    @staticmethod
    def _peak_mask_from_input(x: torch.Tensor) -> torch.Tensor:
        """A peak is real iff its normalized height feature > 0."""
        return x[..., HEIGHT_FEATURE_IDX] > 0.0

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: [B, 24, 50, 89]
        """
        b = x.size(0)
        peak_mask = self._peak_mask_from_input(x)                 # [B, 24, 50]
        locus_active = peak_mask.any(dim=-1)                      # [B, 24]

        # Peak embedding ---------------------------------------------------
        h = self.peak_embed(x)                                    # [B, 24, 50, D]

        # Per-locus set encoding -------------------------------------------
        h = h.view(b * NUM_LOCI, MAX_PEAKS_PER_LOCUS, self.d_model)
        m = peak_mask.view(b * NUM_LOCI, MAX_PEAKS_PER_LOCUS)
        locus_tok, peak_tok = self.peak_set(h, m)                 # [B*24, D], [B*24, P, D]

        # Peak auxiliary outputs -------------------------------------------
        prop, nall_peak = self.peak_head(peak_tok)
        prop = prop.view(b, NUM_LOCI, MAX_PEAKS_PER_LOCUS, 1)
        nall_peak = nall_peak.view(b, NUM_LOCI, MAX_PEAKS_PER_LOCUS, 21)

        locus_tok = locus_tok.view(b, NUM_LOCI, self.d_model)

        # Cross-locus Transformer ------------------------------------------
        prof_tok, locus_tok_ref = self.locus_tx(locus_tok, locus_active)

        # Locus auxiliary outputs ------------------------------------------
        nall_locus, mix_locus = self.locus_head(locus_tok_ref)

        # Profile outputs --------------------------------------------------
        noc_logits, mix_profile = self.profile_head(prof_tok)
        noc_probs = corn_logits_to_class_probs(noc_logits)        # [B, num_classes]

        return {
            "profile_noc_logits": noc_logits,
            "profile_noc_probs": noc_probs,
            "profile_mix_props": mix_profile,
            "locus_n_alleles": nall_locus,
            "locus_mix_props": mix_locus,
            "peak_prop_allelic": prop,
            "peak_n_alleles": nall_peak,
        }

    @torch.no_grad()
    def predict_noc(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        # CORN: predicted class = number of P(y>k) > 0.5, plus 1 for 1-indexing.
        sig = torch.sigmoid(out["profile_noc_logits"])
        # Cumulative consistency: a sample exceeds k iff every j<k also exceeded.
        cum = torch.cumprod(sig, dim=1)
        pred = (cum > 0.5).sum(dim=1) + 1
        return pred

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 20
                                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MC-Dropout: enable dropout at test time, run n_samples forward passes,
        return (mean class probabilities, predictive entropy, predicted NoC).
        """
        was_training = self.training
        self.train()                                    # enable dropout
        try:
            probs = []
            for _ in range(n_samples):
                out = self.forward(x)
                probs.append(out["profile_noc_probs"])
            p = torch.stack(probs, dim=0).mean(dim=0)   # [B, K]
        finally:
            if not was_training:
                self.eval()
        eps = 1e-12
        entropy = -(p * (p + eps).log()).sum(dim=-1)
        pred = p.argmax(dim=-1) + 1
        return p, entropy, pred


def corn_logits_to_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORN cumulative logits [B, K-1] to per-class probabilities [B, K]
    using consistent rank-monotone cumulative probabilities.

    P(y > k) = prod_{j<=k} sigmoid(logits[:, j])
    P(y == k) = P(y > k-1) - P(y > k), with the boundary cases handled.
    """
    sig = torch.sigmoid(logits)                          # [B, K-1]
    cum = torch.cumprod(sig, dim=1)                      # [B, K-1] = P(y>k) for k=1..K-1
    b = logits.size(0)
    one = torch.ones(b, 1, device=logits.device, dtype=logits.dtype)
    zero = torch.zeros(b, 1, device=logits.device, dtype=logits.dtype)
    p_gt = torch.cat([one, cum], dim=1)                  # [B, K]   P(y>=k)
    p_lt = torch.cat([cum, zero], dim=1)                 # [B, K]   P(y>k)
    return p_gt - p_lt


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(0)
    m = NoCFormer(num_classes=10)
    print(f"NoCFormer parameters: {count_parameters(m):,}")
    x = torch.randn(2, 24, 50, 89)
    # Make a few peaks "real"
    x[..., HEIGHT_FEATURE_IDX] = 0.0
    x[0, 5, :3, HEIGHT_FEATURE_IDX] = 0.4
    x[1, 10, :7, HEIGHT_FEATURE_IDX] = 0.2
    out = m(x)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    print("predict_noc:", m.predict_noc(x).tolist())

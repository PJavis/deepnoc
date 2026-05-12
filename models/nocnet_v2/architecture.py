"""
NoCNet-v2: Deep-Sets-per-locus + stutter-aware attention bias + cross-locus
Transformer + count-aware multi-head output.

Design choices vs the original deepNoC CNN and NoCFormer:

  * The 50 peaks at a locus are an unordered SET; we encode them with a
    permutation-invariant Deep Sets pooler that concatenates [sum, max, count],
    not just a CLS token. The sum and count statistics preserve a direct
    signal for "how many alleles are at this locus", which is what NoC
    classification ultimately depends on. Attention-CLS pooling discards the
    count once softmax normalises.

  * Within-locus self-attention is biased by the allele distance between
    every pair of peaks. That distance is exactly the signal stutter peaks
    carry (±1, ±2, ±0.2 repeats) and makes the model robust to peak ordering
    without having to hand-code stutter features into the input.

  * Cross-locus reasoning is a small Transformer over the 24 locus tokens
    with dye + locus positional embeddings. 24 is tiny so this is cheap.

  * The output stage emits three views of NoC:
        - softmax classifier      (CE loss)
        - scalar regression head  (smooth-L1 loss)
        - CORN ordinal head       (binary cross-entropies)
    All three are averaged at inference for a robust prediction.

Input:  [B, 24, 50, 89]
Output: dict with keys
    profile_noc_logits      [B, num_classes]
    profile_noc_corn        [B, num_classes - 1]
    profile_noc_scalar      [B]                       in [1, num_classes]
    profile_noc_probs       [B, num_classes]          ensemble probabilities
    profile_mix_props       [B, MAX_NOC]              softmax sorted donors
    locus_n_alleles         [B, 24, 20]               CE logits per locus
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


HEIGHT_FEATURE_IDX = 26
ALLELE_FEATURE_IDX = 24
SIZE_FEATURE_IDX = 25


def _build_dye_index() -> list[int]:
    dye_keys = list(DYE_CHANNELS.keys())
    locus_to_dye = {}
    for d_idx, d_key in enumerate(dye_keys):
        for locus in DYE_CHANNELS[d_key]:
            locus_to_dye[locus] = d_idx
    return [locus_to_dye.get(locus, 0) for locus in GLOBALFILER_LOCI]


DYE_INDEX = _build_dye_index()


class PeakEmbedder(nn.Module):
    """89-d → d_model MLP with two GELU layers."""

    def __init__(self, in_features: int = NUM_FEATURES_PER_PEAK,
                 d_model: int = 96, dropout: float = 0.1):
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
        return self.net(x)


class StutterBiasAttention(nn.Module):
    """
    Multi-head self-attention with an additive bias matrix derived from the
    allele-distance between every pair of peaks at the locus.

    The bias is a small MLP over the signed allele difference Δ_{ij} =
    allele_j − allele_i (per head). Padding positions are masked out via
    key_padding_mask so they neither receive nor send signal.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        # Bias network: 1-d signed allele delta → per-head scalar.
        # Two layers, small width, GELU. Output is added to the QK^T
        # attention logits before softmax.
        self.bias_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, n_heads),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, alleles: torch.Tensor,
                key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        """
        x:                 [N, P, d_model]
        alleles:           [N, P]   normalised allele values (feature 24)
        key_padding_mask:  [N, P]   True = ignore (PyTorch convention)
        """
        n, p, d = x.shape
        qkv = self.qkv(x).reshape(n, p, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)                 # [N, P, H, D_h] each
        q = q.transpose(1, 2)                       # [N, H, P, D_h]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)  # [N, H, P, P]

        # Allele-distance bias.
        delta = alleles.unsqueeze(2) - alleles.unsqueeze(1)          # [N, P, P]
        bias = self.bias_mlp(delta.unsqueeze(-1))                    # [N, P, P, H]
        bias = bias.permute(0, 3, 1, 2)                              # [N, H, P, P]
        logits = logits + bias

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]                # [N, 1, 1, P]
            logits = logits.masked_fill(mask, float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, 0.0)                           # all-pad row safety
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(n, p, d)            # [N, P, d_model]
        return self.proj(out)


class LocusEncoderBlock(nn.Module):
    """Pre-LN block wrapping StutterBiasAttention + FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 ffn_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = StutterBiasAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, alleles: torch.Tensor,
                key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), alleles, key_padding_mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class DeepSetsPool(nn.Module):
    """
    Permutation-invariant pooling: phi(peak) → rho([sum, max, count]).

    Critically, the sum AND count statistics are preserved (raw count after
    log1p scaling), giving the head a direct numerical signal for the number
    of "salient" peaks at the locus.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # rho takes [sum, max, count_feature] → d_model
        self.rho = nn.Sequential(
            nn.Linear(2 * d_model + 1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, peaks: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        peaks: [N, P, d_model]
        mask:  [N, P] True = real peak
        returns: [N, d_model]
        """
        h = self.phi(peaks)
        m = mask.unsqueeze(-1).float()
        h_masked = h * m
        h_sum = h_masked.sum(dim=1)
        # For max, set padding to a very negative value so it never wins.
        h_for_max = h.masked_fill(~mask.unsqueeze(-1), float("-1e4"))
        h_max = h_for_max.max(dim=1).values
        # All-padded rows would now be -1e4; clamp safely to 0.
        h_max = torch.where(mask.any(dim=1, keepdim=True),
                            h_max, torch.zeros_like(h_max))
        cnt = mask.sum(dim=1, keepdim=True).float()
        cnt_feat = torch.log1p(cnt) / math.log(MAX_PEAKS_PER_LOCUS + 1)
        agg = torch.cat([h_sum, h_max, cnt_feat], dim=-1)
        return self.rho(agg)


class LocusTransformerBlock(nn.Module):
    """Plain pre-LN Transformer block (no stutter bias) for the 24-token stack."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 ffn_mult: int = 4):
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


class CrossLocusTransformer(nn.Module):
    """Self-attention over 24 locus tokens, with dye + locus pos embeddings."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 dropout: float = 0.1, num_dyes: int = 5):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, NUM_LOCI, d_model))
        self.dye_emb = nn.Embedding(num_dyes, d_model)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([
            LocusTransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.register_buffer("dye_idx",
                             torch.tensor(DYE_INDEX, dtype=torch.long),
                             persistent=False)

    def forward(self, locus_tokens: torch.Tensor,
                locus_active: torch.Tensor) -> torch.Tensor:
        dye = self.dye_emb(self.dye_idx).unsqueeze(0)
        x = locus_tokens + self.pos_emb + dye
        kp_mask = ~locus_active
        for block in self.blocks:
            x = block(x, key_padding_mask=kp_mask)
        return self.ln(x)


class ProfilePool(nn.Module):
    """Pool 24 locus tokens → profile vector via [sum, mean, max]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, loci: torch.Tensor,
                locus_active: torch.Tensor) -> torch.Tensor:
        m = locus_active.unsqueeze(-1).float()
        sum_ = (loci * m).sum(dim=1)
        cnt = m.sum(dim=1).clamp(min=1.0)
        mean_ = sum_ / cnt
        max_ = loci.masked_fill(~locus_active.unsqueeze(-1), float("-1e4"))\
                   .max(dim=1).values
        max_ = torch.where(locus_active.any(dim=1, keepdim=True),
                           max_, torch.zeros_like(max_))
        return self.proj(torch.cat([sum_, mean_, max_], dim=-1))


class CountAwareHead(nn.Module):
    """Three parallel heads: softmax, scalar regression, CORN cumulative."""

    def __init__(self, d_model: int, num_classes: int,
                 mix_dim: int = MAX_NOC, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(d_model, num_classes)
        self.reg = nn.Linear(d_model, 1)
        self.corn = nn.Linear(d_model, num_classes - 1)
        self.mix = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, mix_dim),
        )

    def forward(self, profile: torch.Tensor) -> dict:
        h = self.shared(profile)
        cls = self.cls(h)
        reg = self.reg(h).squeeze(-1)
        corn = self.corn(h)
        mix = F.softmax(self.mix(profile), dim=-1)
        return {"cls_logits": cls, "reg": reg, "corn_logits": corn,
                "mix_props": mix}


def corn_to_class_probs(corn_logits: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(corn_logits)
    cum = torch.cumprod(sig, dim=1)
    b = corn_logits.size(0)
    one = torch.ones(b, 1, device=corn_logits.device, dtype=corn_logits.dtype)
    zero = torch.zeros(b, 1, device=corn_logits.device, dtype=corn_logits.dtype)
    p_gt = torch.cat([one, cum], dim=1)
    p_lt = torch.cat([cum, zero], dim=1)
    return (p_gt - p_lt).clamp(min=0.0)


def scalar_to_class_probs(scalar: torch.Tensor, num_classes: int,
                          sigma: float = 0.5) -> torch.Tensor:
    """
    Gaussian smoothing around the predicted scalar NoC: assigns mass to each
    integer class proportional to the bell-curve density at that integer.
    """
    classes = torch.arange(1, num_classes + 1, device=scalar.device,
                           dtype=scalar.dtype).unsqueeze(0)        # [1, K]
    z = (scalar.unsqueeze(-1) - classes) / sigma                   # [B, K]
    p = torch.exp(-0.5 * z * z)
    return p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)


class NoCNetV2(nn.Module):
    """
    Hierarchical Deep-Sets + Transformer for NoC with count-aware multi-head
    classification.
    """

    def __init__(
        self,
        d_model: int = 96,
        n_heads: int = 4,
        peak_layers: int = 2,
        locus_layers: int = 2,
        num_classes: int = MAX_NOC,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        self.peak_embed = PeakEmbedder(NUM_FEATURES_PER_PEAK,
                                       d_model=d_model, dropout=dropout)
        self.peak_blocks = nn.ModuleList([
            LocusEncoderBlock(d_model, n_heads, dropout=dropout)
            for _ in range(peak_layers)
        ])
        self.peak_pool = DeepSetsPool(d_model)
        self.cross_locus = CrossLocusTransformer(
            d_model, n_heads, n_layers=locus_layers,
            dropout=dropout, num_dyes=len(DYE_CHANNELS),
        )
        self.profile_pool = ProfilePool(d_model)
        self.head = CountAwareHead(d_model, num_classes=num_classes,
                                   dropout=max(0.2, dropout))

        # Auxiliary per-locus n_alleles head (CE over 0..19).
        self.locus_nall_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 20),
        )

    @staticmethod
    def _peak_mask(x: torch.Tensor) -> torch.Tensor:
        return x[..., HEIGHT_FEATURE_IDX] > 0.0

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.size(0)
        mask = self._peak_mask(x)                       # [B, 24, 50]
        locus_active = mask.any(dim=-1)                  # [B, 24]

        alleles = x[..., ALLELE_FEATURE_IDX]             # [B, 24, 50]

        h = self.peak_embed(x)                           # [B, 24, 50, D]
        n = b * NUM_LOCI
        h_flat = h.view(n, MAX_PEAKS_PER_LOCUS, self.d_model)
        a_flat = alleles.view(n, MAX_PEAKS_PER_LOCUS)
        m_flat = mask.view(n, MAX_PEAKS_PER_LOCUS)
        kp = ~m_flat

        for block in self.peak_blocks:
            h_flat = block(h_flat, a_flat, kp)

        locus_tok = self.peak_pool(h_flat, m_flat)        # [B*24, D]
        locus_tok = locus_tok.view(b, NUM_LOCI, self.d_model)

        # Zero out tokens for fully-empty loci so they don't poison the
        # cross-locus stack.
        locus_tok = locus_tok * locus_active.unsqueeze(-1).float()

        nall_logits = self.locus_nall_head(locus_tok)     # [B, 24, 20]

        locus_tok = self.cross_locus(locus_tok, locus_active)
        profile = self.profile_pool(locus_tok, locus_active)
        out = self.head(profile)

        # Ensemble probabilities from the three views of NoC.
        p_cls = F.softmax(out["cls_logits"], dim=-1)
        p_corn = corn_to_class_probs(out["corn_logits"])
        p_reg = scalar_to_class_probs(out["reg"].clamp(1.0,
                                                       float(self.num_classes)),
                                      self.num_classes)
        p_ens = (p_cls + p_corn + p_reg) / 3.0

        return {
            "profile_noc_logits": out["cls_logits"],
            "profile_noc_corn": out["corn_logits"],
            "profile_noc_scalar": out["reg"],
            "profile_noc_probs": p_ens,
            "profile_mix_props": out["mix_props"],
            "locus_n_alleles": nall_logits,
        }

    @torch.no_grad()
    def predict_noc(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        return out["profile_noc_probs"].argmax(dim=-1) + 1


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = NoCNetV2(num_classes=5)
    print(f"NoCNet-v2 parameters: {count_parameters(m):,}")
    x = torch.zeros(2, 24, 50, 89)
    x[0, 5, :4, HEIGHT_FEATURE_IDX] = 0.3
    x[0, 5, :4, ALLELE_FEATURE_IDX] = torch.tensor([0.10, 0.11, 0.12, 0.13])
    x[1, 10, :7, HEIGHT_FEATURE_IDX] = 0.2
    x[1, 10, :7, ALLELE_FEATURE_IDX] = 0.10 + 0.01 * torch.arange(7)
    out = m(x)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    print("predict_noc:", m.predict_noc(x).tolist())

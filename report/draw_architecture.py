"""
Draw the NoCFormer architecture diagram as a PNG to embed in the report.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


# ---- colour palette ---------------------------------------------------------
C_INPUT   = "#FFE9B5"
C_EMB     = "#CDE7FF"
C_PEAK    = "#A6D8FF"
C_LOCUS   = "#9FD49F"
C_PROFILE = "#F4B0B0"
C_HEAD    = "#E2C8FF"
C_OUT     = "#FFD9C2"
C_EDGE    = "#444"


def block(ax, x, y, w, h, text, fc, *, fontsize=9, bold=False,
          edgecolor=C_EDGE):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2, edgecolor=edgecolor, facecolor=fc,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, color=C_EDGE, lw=1.2,
          style="->", connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
        color=color, lw=lw, connectionstyle=connectionstyle,
    )
    ax.add_patch(a)
    if text:
        midx = (x1 + x2) / 2
        midy = (y1 + y2) / 2
        ax.text(midx + 0.05, midy, text, fontsize=7, color="#222",
                va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.85))


def draw():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Input -----------------------------------------------------------
    block(ax, 0.4, 8.2, 4.0, 0.7,
          "Input EPG tensor   [B, 24, 50, 89]\n"
          "(B profile · 24 locus · 50 peak · 89 feature)",
          C_INPUT, bold=True, fontsize=9)

    # ---- PeakEmbedder ----------------------------------------------------
    block(ax, 0.4, 7.0, 4.0, 0.7,
          "PeakEmbedder (MLP 89 → d=128)\n"
          "Linear-GELU-LN-Drop-Linear-GELU-LN", C_EMB, fontsize=8)

    # ---- PeakSetEncoder (per locus) -------------------------------------
    block(ax, 0.4, 5.4, 4.0, 1.3,
          "PeakSetEncoder  (chạy song song trên 24 locus)\n"
          "── CLS token + 50 peak token (key-padding mask)\n"
          "── 2 × Pre-LN Transformer block  (self-attention)\n"
          "→ locus token [B, 24, d] ; peak token [B, 24, 50, d]",
          C_PEAK, fontsize=8)

    # ---- PeakHead --------------------------------------------------------
    block(ax, 5.0, 5.7, 3.6, 1.0,
          "PeakHead   (giải thích mức peak)\n"
          "prop_allelic  [B, 24, 50, 1]   (sigmoid)\n"
          "n_alleles     [B, 24, 50, 21]  (CE 0..20)",
          C_HEAD, fontsize=8)

    # ---- LocusTransformer -----------------------------------------------
    block(ax, 0.4, 3.2, 4.0, 1.7,
          "LocusTransformer  (cross-locus)\n"
          "+ learnable locus position [1, 24, d]\n"
          "+ dye-channel embedding (5 dye)\n"
          "+ profile CLS token\n"
          "── 4 × Pre-LN Transformer block\n"
          "→ profile token [B, d] ; refined locus [B, 24, d]",
          C_LOCUS, fontsize=8)

    # ---- LocusHead -------------------------------------------------------
    block(ax, 5.0, 3.5, 3.6, 1.0,
          "LocusHead   (giải thích mức locus)\n"
          "n_alleles   [B, 24, 20]   (CE 1..20)\n"
          "mix_props   [B, 24, 10]   (softmax)",
          C_HEAD, fontsize=8)

    # ---- ProfileHead -----------------------------------------------------
    block(ax, 0.4, 1.5, 4.0, 1.2,
          "ProfileHead\n"
          "CORN NoC head    [B, K-1]   (sigmoid logits)\n"
          "mix_props head   [B, 10]    (softmax)\n"
          "→ predict NoC = (Π sigmoid > 0.5).sum() + 1",
          C_PROFILE, bold=True, fontsize=8)

    # ---- Outputs box -----------------------------------------------------
    block(ax, 5.0, 1.5, 3.6, 1.2,
          "Profile outputs (chính)\n"
          "profile_noc_logits  [B, K-1]\n"
          "profile_noc_probs   [B, K]\n"
          "profile_mix_props   [B, 10]",
          C_OUT, bold=True, fontsize=8)

    # ---- Inference helpers ----------------------------------------------
    block(ax, 9.2, 4.6, 3.4, 1.7,
          "Inference helpers\n"
          "• predict_noc(x):\n"
          "    cum=Πσ(logits); pred=(cum>0.5).sum()+1\n"
          "• predict_with_uncertainty(x, n_samples):\n"
          "    bật dropout, n lần forward,\n"
          "    trả về (probs trung bình, entropy, pred)",
          "#F8F2C8", fontsize=8)

    # ---- Loss & training -------------------------------------------------
    block(ax, 9.2, 2.4, 3.4, 1.9,
          "Training (NoCFormerLoss)\n"
          "• CORN ordinal + class-balanced (Cui 2019)\n"
          "  + focal γ=1\n"
          "• MSE: profile/locus mix_props\n"
          "• CE   : peak/locus n_alleles (mask -1)\n"
          "• MSE  : peak prop_allelic (peak_mask)",
          "#FFE5F4", fontsize=8)

    # ---- Augment box -----------------------------------------------------
    block(ax, 9.2, 0.4, 3.4, 1.7,
          "Augmentation (TrainingAugmenter)\n"
          "• synthetic_mix(K nguồn 1-người)\n"
          "• peak_height_jitter (LN(0,σ))\n"
          "• random_peak_dropout (p)\n"
          "• shuffle_peak_axis (an toàn vì set)",
          "#E8FFE8", fontsize=8)

    # ---- Arrows main flow ------------------------------------------------
    arrow(ax, 2.4, 8.2, 2.4, 7.7)
    arrow(ax, 2.4, 7.0, 2.4, 6.7)
    arrow(ax, 2.4, 5.4, 2.4, 4.9)
    arrow(ax, 2.4, 3.2, 2.4, 2.7)

    # ---- Side arrows to heads -------------------------------------------
    arrow(ax, 4.4, 6.05, 5.0, 6.2, text="peak tokens")
    arrow(ax, 4.4, 4.05, 5.0, 4.0, text="locus tokens")
    arrow(ax, 4.4, 2.10, 5.0, 2.10, text="profile token")

    # ---- Legend ----------------------------------------------------------
    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="", color=C_INPUT,
               markeredgecolor=C_EDGE, markersize=12, label="Input"),
        Line2D([0], [0], marker="s", linestyle="", color=C_EMB,
               markeredgecolor=C_EDGE, markersize=12, label="Embedding"),
        Line2D([0], [0], marker="s", linestyle="", color=C_PEAK,
               markeredgecolor=C_EDGE, markersize=12, label="Peak Transformer"),
        Line2D([0], [0], marker="s", linestyle="", color=C_LOCUS,
               markeredgecolor=C_EDGE, markersize=12, label="Locus Transformer"),
        Line2D([0], [0], marker="s", linestyle="", color=C_PROFILE,
               markeredgecolor=C_EDGE, markersize=12, label="Profile head (CORN)"),
        Line2D([0], [0], marker="s", linestyle="", color=C_HEAD,
               markeredgecolor=C_EDGE, markersize=12, label="Aux explain head"),
        Line2D([0], [0], marker="s", linestyle="", color=C_OUT,
               markeredgecolor=C_EDGE, markersize=12, label="Output"),
    ]
    ax.legend(handles=legend_handles, loc="lower left",
              bbox_to_anchor=(0.0, -0.04), ncol=4, fontsize=8, frameon=False)

    fig.text(0.5, 0.965, "NoCFormer — kiến trúc tổng thể",
             ha="center", fontsize=14, fontweight="bold", color="#1F2D5C")
    fig.text(0.5, 0.94,
             "Hierarchical set-Transformer cho bài toán xác định số người "
             "đóng góp DNA (NoC)",
             ha="center", fontsize=10, color="#1F2D5C")

    out = os.path.join(os.path.dirname(__file__), "nocformer_arch.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    draw()

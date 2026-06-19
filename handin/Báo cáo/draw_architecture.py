"""
Vẽ sơ đồ kiến trúc tổng thể (top-down) của mô hình đề xuất NoCNet-v2 (Deep Sets).
Chạy: python draw_architecture.py  -> nocnet_v2_arch.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, LBLUE, ORANGE, GREEN, GREY = "#1f4e79", "#dbe7f3", "#fff0d6", "#e3f1e1", "#eeeeee"

fig, ax = plt.subplots(figsize=(10.5, 14))
ax.set_xlim(0, 10); ax.set_ylim(0, 15); ax.axis("off")


def box(x, y, w, h, title, sub, fc=LBLUE, ec=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=BLUE)
    ax.text(x + w / 2, y + 0.24, sub, ha="center", va="bottom", fontsize=8.3, color="#333")


def arrow(x0, y0, x1, y1, label=""):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
                                 lw=1.5, color="#444"))
    if label:
        ax.text((x0 + x1) / 2 + 0.12, (y0 + y1) / 2, label, fontsize=8, color="#b00020",
                ha="left", va="center")


ax.text(5, 14.6, "Kiến trúc tổng thể  NoCNet-v2 (Deep Sets)  ~0,8M tham số",
        ha="center", fontsize=13, fontweight="bold", color=BLUE)
ax.text(5, 14.2, "B = batch · d_model = 96 · 4 đầu chú ý · K = 5 lớp NoC",
        ha="center", fontsize=9, color="#555", style="italic")

cx, w = 2.6, 4.8
box(cx, 13.0, w, 0.95, "Đầu vào: tensor (B, 24, 50, 89)",
    "24 locus × ≤50 đỉnh × 89 đặc trưng/đỉnh", fc=GREY)
box(cx, 11.7, w, 0.95, "(1) PeakEmbedder",
    "MLP 89→96 (2× GELU+LayerNorm+Dropout)\n→ (B,24,50,96)")
box(cx, 10.3, w, 1.0, "(2) 2 × LocusEncoderBlock (per-locus)",
    "StutterBiasAttention (bias theo Δ allele) + FFN, pre-LN\nmã hóa stutter → (B,24,50,96)")
box(cx, 8.9, w, 1.0, "(3) DeepSetsPool  (gộp đỉnh→locus)",
    "ρ([sum, max, log1p(count)]) — bất biến hoán vị,\ngiữ tín hiệu ĐẾM → locus tokens (B,24,96)")
box(cx, 7.5, w, 1.0, "(4) 2 × CrossLocusTransformer",
    "MHA + FFN trên 24 token locus\n+ dye_emb (5 kênh) + pos_emb → (B,24,96)")
box(cx, 6.1, w, 0.95, "(5) ProfilePool",
    "gộp [sum, mean, max] theo 24 locus\n→ profile vector (B,96)", fc=GREEN)
box(cx, 4.4, w, 1.25, "(6) CountAwareHead — 3 góc nhìn NoC",
    "cls softmax (CE)  ·  reg vô hướng (smooth-L1)\ncorn ordinal (BCE)  →  ensemble probs (B,5)\n+ mix_props (B,10)", fc=ORANGE)
box(cx, 2.9, w, 0.95, "Hậu xử lý suy luận",
    "TTA 20× MC-dropout + jitter → trung bình softmax\n+ bias cộng theo lớp [1.6, 0, −0.6, 0, 2.6] → argmax", fc="#f6e1ef", ec="#8e2a73")
box(cx, 1.7, w, 0.7, "Đầu ra: NoC ∈ {1..5}", "số người đóng góp", fc=GREY)

# aux head
box(7.7, 8.95, 2.1, 0.9, "Đầu phụ (train)", "locus_n_alleles\n(B,24,20), CE", fc="#f3f3f3", ec="#999")

for y0, y1 in [(13.0, 12.68), (11.7, 11.32), (10.3, 9.92), (8.9, 8.52),
               (7.5, 7.08), (6.1, 5.68), (4.4, 3.87), (2.9, 2.42)]:
    arrow(5.0, y0, 5.0, y1)
arrow(7.4, 9.4, 7.7, 9.4, "")  # deepsets -> aux

ax.text(5, 1.05, "Mất mát đa nhiệm: CE(cls) + smooth-L1(reg) + BCE(corn) + phụ; trọng số lớp cân bằng (β=0,999)",
        ha="center", fontsize=8.8, color=BLUE,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fbfbe7", ec="#caca7a"))
ax.text(5, 0.55, "Huấn luyện: tiền huấn luyện synthetic (p_synth=0,8) → tinh chỉnh PROVEDIt thật + SWA",
        ha="center", fontsize=8.3, color="#555", style="italic")

plt.tight_layout()
plt.savefig(os.path.join(HERE, "nocnet_v2_arch.png"), dpi=160, bbox_inches="tight")
plt.close()
print("Đã vẽ nocnet_v2_arch.png")

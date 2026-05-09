"""
Draw a dataset-structure diagram for the report.

Two panels:
  - top: tensor [24, 50, 89] anatomy, broken down into the three feature groups.
  - bottom: PROVEDIt sample-name decoder (one row per token, no ASCII tree).
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def block(ax, x, y, w, h, text, fc, fontsize=8, bold=False,
          ha="center", va="center"):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0, edgecolor="#444", facecolor=fc,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    tx = x + (w / 2 if ha == "center" else 0.15)
    ty = y + h / 2 if va == "center" else y + h - 0.15
    ax.text(tx, ty, text, ha=ha, va=va,
            fontsize=fontsize, fontweight=weight)


def draw():
    fig, ax = plt.subplots(figsize=(13, 13.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- title ----------------------------------------------------------
    fig.text(0.5, 0.975, "Cấu trúc tập dữ liệu PROVEDIt sau khi xử lý",
             ha="center", fontsize=14, fontweight="bold", color="#1F2D5C")

    # ====================  PANEL A — TENSOR LAYOUT  ====================
    block(ax, 0.4, 12.0, 12.2, 1.2,
          "Một profile DNA   →   tensor [24, 50, 89]\n"
          "trục 0 = 24 locus    trục 1 = tối đa 50 peak / locus    "
          "trục 2 = 89 đặc trưng / peak",
          "#FFE9B5", fontsize=11, bold=True)

    # Group 1: identity (idx 0..28)
    block(ax, 0.4, 10.0, 4.0, 1.7,
          "Nhóm 1 — định danh peak  (idx 0..28)\n\n"
          "0..23 : one-hot 24 locus\n"
          "24    : allele / 100  (số repeat)\n"
          "25    : size / 100    (bp)\n"
          "26    : height / 33000  (rfu)\n"
          "27    : allele frequency\n"
          "28    : peak label probability (plp)",
          "#CDE7FF", fontsize=8, ha="left", va="top")

    # Group 2: stutter (idx 29..76)
    block(ax, 4.5, 10.0, 4.0, 1.7,
          "Nhóm 2 — quan hệ stutter  (idx 29..76)\n\n"
          "Mỗi loại stutter (back / dbl-back /\n"
          "forward / 0.2-repeat) chiếm 6 giá trị,\n"
          "có cho 2 chiều  (peak là stutter / là cha).\n\n"
          "→ 4 loại × 6 giá trị × 2 chiều = 48 ô",
          "#A6D8FF", fontsize=8, ha="left", va="top")

    # Group 3: locus + profile + mix props (idx 77..88)
    block(ax, 8.6, 10.0, 4.0, 1.7,
          "Nhóm 3 — context  (idx 77..88)\n\n"
          "77 : số peak ở locus / 100\n"
          "78 : số peak toàn profile / 1000\n"
          "79..88 : mixture proportions\n"
          "         (tỷ lệ 10 người đóng góp\n"
          "          lớn nhất, smart-start)",
          "#9FD49F", fontsize=8, ha="left", va="top")

    # ====================  PANEL B — SAMPLE NAME DECODER  ====================
    block(ax, 0.4, 7.6, 12.2, 1.7,
          "Ví dụ tên sample PROVEDIt:\n\n"
          "RD14-0003_GF_25sec_GM_SE33F_2-5P : "
          "A02_RD14-0003-31_32-1;1-M2c-0.03GF-Q2.0_01.25sec.hid\n\n"
          "Phần trước dấu ‘:’ là tên file CSV.   Phần sau là tên sample bên trong file.",
          "#FFF7E6", fontsize=10, bold=True)

    decoder_rows = [
        ("RD14-0003",     "ID dự án PROVEDIt — RD = Rutgers DNA, batch 14, file 0003"),
        ("GF",            "kit khuếch đại GlobalFiler™ (24 locus)"),
        ("25sec",         "thời gian inject 25 giây trên capillary"),
        ("GM",            "GeneMapper run (phần mềm đọc EPG)"),
        ("SE33F",         "đã bao gồm locus SE33"),
        ("2-5P",          "phạm vi NoC trong file: 2 → 5 người"),
        ("A02",           "giếng A02 trên đĩa 96-well"),
        ("31_32",         "ID 2 người đóng góp DNA (donor 31 + donor 32)"),
        ("1;1",           "tỷ lệ đóng góp 1:1 (NoC = 2, hai người đều nhau)"),
        ("M2c",           "ID chuỗi xử lý (mixture protocol M2 variant c)"),
        ("0.03GF",        "tổng template DNA ban đầu (ng) trong PCR"),
        ("Q2.0",          "Q-score chất lượng EPG"),
        ("01",            "số replicate (lần inject thứ 1)"),
        ("25sec.hid",     "file gốc HID (Hi-Density Indexed Data)"),
    ]
    y = 7.30
    for token, meaning in decoder_rows:
        ax.text(0.5, y, token, fontsize=9, fontweight="bold",
                color="#1F2D5C", family="monospace", va="top")
        ax.text(3.0, y, meaning, fontsize=9, color="#222", va="top")
        y -= 0.32

    # ====================  PANEL C — CSV columns  ====================
    block(ax, 0.4, 0.2, 12.2, 1.5,
          "Cột chính trong CSV GeneMapper (wide format)  →  long format dùng cho tensor\n\n"
          "wide:  Sample Name | Marker | Dye | Allele 1 | Size 1 | Height 1 | "
          "Allele 2 | Size 2 | Height 2 | …\n"
          "long:  SampleName  | Marker | Dye |    Allele |   Size |   Height       "
          "(mỗi peak một dòng)\n\n"
          "Loại bỏ: peak height ≤ 0; allele = OL (off-ladder); allele không parse "
          "được; locus không thuộc 24 locus GlobalFiler.\n"
          "Riêng locus AMEL: allele ‘X’→1.0, ‘Y’→2.0  (giới tính).",
          "#F4F4F4", fontsize=9, ha="left", va="top")

    out = os.path.join(os.path.dirname(__file__), "dataset_layout.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    draw()

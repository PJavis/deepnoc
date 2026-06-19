"""
Sinh hình kết quả cho mô hình NoCNet-v2 (Deep Sets) — kết quả TRUNG THỰC
(đã loại rò rỉ dữ liệu): accuracy 0,927 trên 923 profile kiểm thử grouped seed 42.
Số liệu lấy trực tiếp từ `python predict.py --reproduce-test`.

Chạy:  python make_figures.py
Đầu ra: confusion_matrix.png, per_class.png, system_comparison.png,
        metrics_nocnet_v2_honest.json
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREY = "#1f4e79", "#e07b00", "#9e9e9e"

# ── Số liệu trung thực (predict.py --reproduce-test) ──────────────────────
LABELS = [1, 2, 3, 4, 5]
CM = np.array([
    [691,  0,  0,  0,   0],
    [  6, 16, 22,  0,   4],
    [  0, 16, 43,  1,   4],
    [  0,  0,  9,  3,   2],
    [  0,  0,  2,  1, 103],
])
SUPPORT = [691, 48, 64, 14, 106]
ACC = [1.000, 0.333, 0.672, 0.214, 0.972]
F1 = [0.9957, 0.40, 0.6143, 0.3158, 0.9406]
BIAS = [1.6, 0.0, -0.6, 0.0, 2.6]
OVERALL_ACC = 0.9274
MACRO_F1 = 0.6533


def save_metrics_json():
    out = {
        "model": "NoCNet-v2 (Deep Sets)",
        "note": "Trung thực, đã loại rò rỉ dữ liệu (nocnet_v2_ft.pt). "
                "Pipeline: TTA 20x MC-dropout + per-class additive bias + argmax.",
        "split": "grouped pedigree, seed 42, 2455 train / 923 test",
        "overall": {"accuracy": OVERALL_ACC, "macro_f1": MACRO_F1, "total": 923},
        "per_class": {
            str(l): {"accuracy": ACC[i], "f1": F1[i], "support": SUPPORT[i]}
            for i, l in enumerate(LABELS)},
        "per_class_bias": BIAS,
        "confusion_matrix_rows_true_cols_pred": CM.tolist(),
    }
    json.dump(out, open(os.path.join(HERE, "metrics_nocnet_v2_honest.json"), "w"),
              ensure_ascii=False, indent=2)


def fig_confusion():
    cmn = CM / CM.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6.5, 5.6))
    plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Tỉ lệ theo hàng (recall)")
    for i in range(5):
        for j in range(5):
            c = "white" if cmn[i, j] > 0.5 else "#222"
            plt.text(j, i, f"{CM[i, j]}\n{cmn[i, j]:.2f}", ha="center", va="center",
                     fontsize=8.5, color=c)
    plt.xticks(range(5), [f"NoC{l}" for l in LABELS])
    plt.yticks(range(5), [f"NoC{l}" for l in LABELS])
    plt.xlabel("Dự đoán"); plt.ylabel("Thực tế")
    plt.title(f"Ma trận nhầm lẫn NoCNet-v2 (acc={OVERALL_ACC:.3f}, macro-F1={MACRO_F1:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "confusion_matrix.png"), dpi=150)
    plt.close()


def fig_per_class():
    x = np.arange(5); w = 0.38
    plt.figure(figsize=(8, 5))
    b1 = plt.bar(x - w / 2, ACC, w, color=BLUE, label="Accuracy (recall)")
    b2 = plt.bar(x + w / 2, F1, w, color=ORANGE, label="F1")
    for b, v in zip(b1, ACC):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for b, v in zip(b2, F1):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    plt.xticks(x, [f"NoC{l}\n(n={s})" for l, s in zip(LABELS, SUPPORT)])
    plt.ylim(0, 1.1); plt.ylabel("Giá trị")
    plt.title("Hiệu năng theo từng lớp NoC (tập kiểm thử, 923 profile)")
    plt.grid(axis="y", alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "per_class.png"), dpi=150)
    plt.close()


def fig_system_comparison():
    systems = ["MAC+RF\n(đếm allele)", "NoCFormer\n(grouped)", "deepNoC CNN\n(alternating*)",
               "NoCNet-v2\n(đề xuất, trung thực)"]
    vals = [0.66, 0.668, 0.82, OVERALL_ACC]
    colors = [GREY, GREY, GREY, BLUE]
    plt.figure(figsize=(8.5, 5))
    b = plt.bar(systems, vals, color=colors)
    for bar, v in zip(b, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylabel("Độ chính xác đếm NoC"); plt.ylim(0.5, 1.0)
    plt.title("So sánh độ chính xác đếm NoC trên PROVEDIt\n(* alternating split có rò rỉ — không trung thực)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "system_comparison.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    save_metrics_json()
    fig_confusion()
    fig_per_class()
    fig_system_comparison()
    print("Đã sinh metrics + 3 hình kết quả.")

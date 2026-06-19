# Mô hình — NoCNet-v2 (Deep Sets)

Mô hình cuối cùng (tốt nhất) ước lượng **số người đóng góp (NoC)** từ hồ sơ STR DNA
hỗn hợp, huấn luyện và đánh giá trên PROVEDIt (GlobalFiler / ABI-3500 / 25 giây).

## Tệp trong thư mục

| Tệp | Mô tả |
|---|---|
| `nocnet_v2_ft.pt` | **Trọng số cuối — DÙNG TỆP NÀY.** Bản đã tinh chỉnh trên PROVEDIt thật, *đã loại rò rỉ dữ liệu* (bể synthetic chỉ lấy đơn nguồn của tập train). Đây là mô hình trung thực đạt accuracy 0,927. |
| `nocnet_v2_pretrain.pt` | Checkpoint tiền huấn luyện trên dữ liệu synthetic (để tái lập quy trình). |
| `bias_tuned.json` | Vector bias cộng theo lớp `[1,6; 0; −0,6; 0; 2,6]`, tinh chỉnh trên validation để tối đa macro-F1. |

## Kết quả (tập kiểm thử grouped seed 42, 923 profile)

Pipeline suy luận: `nocnet_v2_ft.pt` → TTA (20× MC-dropout + jitter) → bias theo lớp → argmax.

- Accuracy (đếm NoC): **0,927**
- Macro-F1: **0,653**
- Per-class accuracy: NoC1 1,000 · NoC2 0,33 · NoC3 0,67 · NoC4 0,21 · NoC5 0,97
  (NoC4 chỉ có 14 profile kiểm thử → nhiễu)

> Một lần chạy trước đạt 0,943 nhưng bể synthetic vô tình chứa profile đơn nguồn của
> tập kiểm thử (rò rỉ); con số đó *lạc quan* và **không** được đóng gói ở đây.

## Cách nạp + tái lập

Xem **Mã nguồn/HUONG_DAN.md**. Tóm tắt:

```python
import sys; sys.path.insert(0, "Mã nguồn")
from models.nocnet_v2.train import load_nocnet_v2
model = load_nocnet_v2("Mô hình/nocnet_v2_ft.pt", device="cpu")
```

Tái lập đúng accuracy 0,927:

```bash
cd "Mã nguồn"
python predict.py --reproduce-test     # cần data/X_gf25.npy (xem HUONG_DAN)
```

# Hướng dẫn cài đặt và chạy mã nguồn — NoCNet-v2 (Deep Sets)

Mã nguồn của mô hình đề xuất cuối cùng: ước lượng số người đóng góp (NoC) từ hồ sơ
STR DNA hỗn hợp trên cơ sở dữ liệu **PROVEDIt** (GlobalFiler / ABI-3500 / 25 giây).

## 1. Cài đặt môi trường

Yêu cầu Python 3.11+. Cài các gói:

```bash
pip install torch numpy scikit-learn
```

(Khuyến nghị GPU CUDA; vẫn chạy được trên CPU — suy luận TTA 20× cho 923 profile mất ~18 giây trên CPU.)

## 2. Cấu trúc thư mục

```
Mã nguồn/
├── models/nocnet_v2/
│   ├── architecture.py   # kiến trúc NoCNetV2 (PeakEmbedder, StutterBiasAttention,
│   │                     #   DeepSetsPool, CrossLocusTransformer, CountAwareHead)
│   ├── losses.py         # mất mát đa nhiệm (CE + smooth-L1 + CORN, trọng số cân bằng)
│   └── train.py          # huấn luyện + load_nocnet_v2 + predict_nocnet_v2(_tta)
├── src/
│   ├── constants.py      # 24 locus GlobalFiler, kênh màu, hằng chuẩn hóa
│   └── split.py          # grouped_stratified_split (tái lập phép chia)
├── predict.py            # suy luận / tái lập kết quả tập kiểm thử
└── data/
    ├── y_gf25.npy                 # [3378] nhãn NoC (1..5)
    ├── split_grouped_seed42.json  # chỉ số train/test (grouped, seed 42)
    └── sample_names.txt           # 3378 định danh mẫu PROVEDIt
```

> **Lưu ý về dữ liệu lớn:** tensor đầu vào `data/X_gf25.npy` (`[3378, 24, 50, 89]`, ~1,4 GB)
> **không** được đính kèm do dung lượng. Sinh lại từ CSV PROVEDIt qua pipeline tiền xử lý
> (`prepare_data` của dự án), hoặc xin trực tiếp từ tác giả. Các tệp nhỏ (nhãn, split,
> tên mẫu) đã có sẵn để tái lập phép chia.

## 3. Định dạng dữ liệu (input/output)

**Input** mỗi profile: tensor `[24, 50, 89]` (`float32`):
- trục 0 = 24 locus GlobalFiler (thứ tự cố định, xem `src/constants.py: GLOBALFILER_LOCI`);
- trục 1 = ≤ 50 đỉnh/locus (đệm 0);
- trục 2 = 89 đặc trưng/đỉnh: one-hot locus (0–23), allele/100 (24), size/100 (25),
  height/33000 (26), tần số allele (27), peak-label-prob (28), quan hệ stutter (29–76),
  số đỉnh locus/profile (77–78), tỉ lệ hỗn hợp (79–88).
  Mặt nạ đỉnh = `x[..., 26] > 0`.

**Output**: NoC ∈ {1..5} cho mỗi profile (mảng `int`), lưu `predictions.npy`.

## 4. Quy trình chạy

### 4a. Tái lập kết quả tập kiểm thử (accuracy 0,927)

```bash
python predict.py --reproduce-test
```

- **Input**: `data/X_gf25.npy`, `data/y_gf25.npy`, `data/split_grouped_seed42.json`,
  `../Mô hình/nocnet_v2_ft.pt`, `../Mô hình/bias_tuned.json`.
  (Mặc định `predict.py` tìm model/bias trong `model/`; truyền `--model ../Mô hình/nocnet_v2_ft.pt
  --bias ../Mô hình/bias_tuned.json` nếu chạy từ thư mục này.)
- **Xử lý**: tải fold kiểm thử (923 profile) → TTA 20× MC-dropout → bias theo lớp → argmax.
- **Output**: in `accuracy`, `macro-F1`, F1 từng lớp, ma trận nhầm lẫn; lưu `predictions.npy`.

### 4b. Chấm điểm tensor bất kỳ

```bash
python predict.py --x my_X.npy --y my_y.npy --tta     # --y tùy chọn (để in metric)
```

- **Input**: `my_X.npy` `[N,24,50,89]` (+ `my_y.npy` `[N]` nhãn 1-based, tùy chọn).
- **Output**: `predictions.npy` `[N]` NoC dự đoán; nếu có `--y` thì in accuracy/macro-F1/confusion.

### 4c. Huấn luyện lại (tham khảo)

Quy trình: tiền huấn luyện synthetic (`p_synth = 0,8`, loader memmap) + SWA → tinh chỉnh
PROVEDIt thật. Cấu hình mặc định (xem `models/nocnet_v2/train.py: TrainConfig`):
AdamW lr = 3·10⁻⁴, weight_decay = 5·10⁻⁴, warmup 3 epoch, batch 16, ≤ 80 epoch,
dừng sớm 12, dropout 0,15, d_model 96, peak-dropout 0,03.

## 5. Tóm tắt input/output toàn cục

| Bước | Input | Output |
|---|---|---|
| Tiền xử lý (pipeline dự án) | CSV PROVEDIt | `X_gf25.npy [N,24,50,89]`, `y_gf25.npy`, split json |
| Huấn luyện | X/y + synthetic | `nocnet_v2_pretrain.pt` → `nocnet_v2_ft.pt`, `bias_tuned.json` |
| Suy luận (`predict.py`) | tensor `[N,24,50,89]` + model | `predictions.npy` (NoC ∈ {1..5}) + metric |

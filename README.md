# Kế Hoạch Thử Nghiệm Lại Bài Báo deepNoC

## Tổng Quan

**Mục tiêu:** Tái hiện (replicate) bài báo *"deepNoC: A deep learning system to assign the number of contributors to a short tandem repeat DNA profile"* (Taylor & Humphries, 2024) trên cấu hình phần cứng cá nhân.

**Cấu hình máy:**

| Thành phần | Cấu hình gốc (bài báo) | Cấu hình của bạn |
|---|---|---|
| CPU | Intel i9-14900HX, 2.2GHz | AMD Ryzen 5 7500F 6-Core, 3.7GHz (12 threads) |
| RAM | 128GB | 16GB (WSL: 12GB) |
| GPU | NVIDIA RTX 4090 16GB | **Chưa rõ — cần kiểm tra** |
| OS | Windows 11 Pro | Windows 11 Pro + WSL Debian |

> ⚠️ **Lưu ý quan trọng:** Bài báo dùng RTX 4090 để huấn luyện. Nếu máy bạn **không có GPU NVIDIA**, bạn vẫn có thể chạy trên CPU nhưng sẽ rất chậm (x50–x100 lần). Trong trường hợp đó, nên cân nhắc dùng Google Colab (miễn phí, có GPU T4) hoặc giảm quy mô thử nghiệm.

---

## Giai Đoạn 0: Kiểm Tra Môi Trường & Xác Định Phạm Vi

**Thời gian ước tính:** 1–2 ngày

### 0.1 Kiểm tra GPU
```bash
# Trên Windows (PowerShell)
nvidia-smi

# Trên WSL
nvidia-smi  # Cần WSL2 + NVIDIA driver trên Windows
```
Nếu không có GPU → quyết định: (a) chạy trên CPU với dataset nhỏ, (b) dùng Google Colab, hoặc (c) thuê GPU cloud (Vast.ai, Lambda, RunPod).

### 0.2 Xác định phạm vi thử nghiệm

Bài báo có 2 phần chính. Với tài nguyên hạn chế, nên ưu tiên:

| Phần | Mô tả | Độ khó | Khuyến nghị |
|---|---|---|---|
| **Baseline trên PROVEDIt** | Tinh chỉnh model trên 743 hồ sơ thực (1–5 NoC) | ⭐⭐ Trung bình | **Ưu tiên 1** — Bạn đã có dữ liệu |
| **Huấn luyện trên 100K mô phỏng** | Pipeline mô phỏng + huấn luyện từ đầu (1–10 NoC) | ⭐⭐⭐⭐ Rất khó | Ưu tiên 2 — Cần R, GAN, nhiều RAM |

**Khuyến nghị:** Bắt đầu với phần fine-tuning trên PROVEDIt (GF 3500, 25 sec) vì bạn đã có dữ liệu (`X_gf25.npy`, `y_gf25.npy`) và chỉ cần ~743 hồ sơ.

---

## Giai Đoạn 1: Chuẩn Bị Dữ Liệu PROVEDIt

**Thời gian ước tính:** 3–5 ngày

### 1.1 Hiểu cấu trúc dữ liệu PROVEDIt

Bạn đã có dữ liệu tại `data/provedit_processed/`. Cần tập trung vào thư mục:
```
PROVEDIt_1-5-Person CSVs Filtered_3500_GF29cycles/
├── 1-Person/     (15sec, 25sec, 5sec)
├── 2-5-Persons/  (15sec, 25sec, 5sec)
└── Known Genotypes.xlsx
```

Bài báo chỉ sử dụng **GlobalFiler, ABI 3500, 25 sec injection**. Đây tương ứng với các file `*_25sec_*` trong thư mục trên.

### 1.2 Xử lý dữ liệu CSV → Tensor đầu vào

Mỗi hồ sơ cần chuyển thành tensor `[24 × 50 × 89]`:

- **24 loci** của GlobalFiler: D3S1358, vWA, D16S539, CSF1PO, TPOX, Yindel, AMEL, D8S1179, D21S11, D18S51, DYS391, D2S441, D19S433, TH01, FGA, D22S1045, D5S818, D13S317, D7S820, SE33, D10S1248, D1S1656, D12S391, D2S1338
- **50 đỉnh tối đa** mỗi locus (zero-pad nếu ít hơn)
- **89 đặc trưng** mỗi đỉnh (chi tiết ở mục 2.4 bài báo)

**Đặc trưng chính cho mỗi đỉnh (89 features):**

| Index | Mô tả | Chuẩn hóa |
|---|---|---|
| 1–24 | One-hot encoded locus | 0/1 |
| 25 | Allele designation | /100 |
| 26 | Size (bp) | /100 |
| 27 | Height (rfu) | /33000 |
| 28 | Allele frequency | Giá trị gốc |
| 29 | Peak label probability (plp) | 0–1 |
| 30–77 | Thông tin stutter (4 loại × 6 giá trị × 2 hướng) | Đa dạng |
| 78 | Tổng đỉnh tại locus | /100 |
| 79 | Tổng đỉnh trong hồ sơ | /1000 |
| 80–89 | Ước tính mixture proportion (10 contributor) | 0–1 |

**Lưu ý quan trọng:**
- Bài báo **KHÔNG lọc** stutter hay artifact — dùng tín hiệu thô
- Đỉnh có xác suất >0.97 là artifact thì mới bị loại
- Feature 80–89 cần thuật toán "smart start" từ STRmix (cần implement lại hoặc ước lượng)

### 1.3 Tạo nhãn (labels)

Bài báo có **6 đầu ra**. Cho fine-tuning trên PROVEDIt, bạn cần:

1. **Profile NoC** (bắt buộc): one-hot `[10]` — biết từ tên file (1P, 2P, 3P...)
2. **Locus number of alleles**: `[24 × 20]` — tính từ known genotypes
3. **Locus mixture proportions**: `[24 × 10]` — tính từ peak heights
4. **Profile mixture proportions**: `[10]` — tính từ tổng DNA
5. **Peak number of alleles**: `[24 × 50 × 21]` — tính từ known genotypes
6. **Peak proportion allelic**: `[24 × 50 × 1]` — khó tính, bài báo dùng model predict trước

### 1.4 Phân chia dữ liệu

Theo bài báo: 371 train / 372 test (xen kẽ, every other profile).

Phân bố mục tiêu: 68 (1P), 175 (2P), 158 (3P), 186 (4P), 156 (5P) = 743 tổng.

**Task cần làm:**
```
src/data_loader.py  →  Đọc CSV, tạo tensor [24×50×89], tạo labels
src/data_split.py   →  Phân chia train/test
```

---

## Giai Đoạn 2: Xây Dựng Kiến Trúc deepNoC

**Thời gian ước tính:** 5–7 ngày

### 2.1 Kiến trúc tổng quan

```
Input [24 × 50 × 89]
    │
    ├─── Main Branch (16 layers) ──────────────────► Profile NoC [10]
    │         │              │              │
    │    Peak Outputs    Locus Outputs   Profile Outputs
    │    ├── prop_allelic   ├── n_alleles    ├── mix_proportions
    │    └── n_alleles      └── mix_props    └── (NoC ở trên)
    │
    └─── Secondary outputs được feedback vào main branch
```

### 2.2 Chi tiết kỹ thuật

Bài báo không công bố chi tiết kiến trúc đầy đủ (chỉ có hình tổng quan trong supplementary). Cần suy luận:

- **Main branch**: 16 layers, có khả năng kết hợp Conv1D/Conv2D + Dense
- **Peak-level branch**: xử lý từng đỉnh → output per-peak
- **Locus-level branch**: aggregate đỉnh → output per-locus
- **Profile-level branch**: aggregate locus → NoC classification

**Framework khuyến nghị:** PyTorch (linh hoạt hơn TensorFlow cho multi-output)

### 2.3 Loss functions

| Output | Loss Function |
|---|---|
| Peak proportion allelic | MSE |
| Peak number of alleles | Categorical Cross-Entropy |
| Locus mixture proportions | MSE |
| Locus number of alleles | Categorical Cross-Entropy |
| Profile mixture proportions | MSE |
| **Profile NoC** | **Categorical Cross-Entropy** |

### 2.4 Hyperparameters

- Optimizer: Adam (lr=0.00001, β₁=0.5)
- Batch size: 100
- Epochs: 200 (simulated) → 2000 (fine-tuning)

**Task cần làm:**
```
models/deepnoc/architecture.py  →  Định nghĩa model
models/deepnoc/train.py         →  Training loop
models/deepnoc/losses.py        →  Multi-output loss
```

---

## Giai Đoạn 3: Phương Án Đơn Giản Hóa (Baseline)

**Thời gian ước tính:** 3–5 ngày

Trước khi xây dựng deepNoC đầy đủ, nên triển khai baseline đơn giản hơn để:
- Kiểm tra pipeline dữ liệu hoạt động đúng
- Có kết quả so sánh
- Quen thuộc với dữ liệu

### 3.1 Baseline: MAC (Maximum Allele Count)

Đơn giản nhất — không cần ML:
```python
def mac_predict(profile):
    max_alleles = max(count_alleles(locus) for locus in profile)
    return ceil(max_alleles / 2)
```

Kỳ vọng: ~60–80% accuracy (tùy NoC range).

### 3.2 Baseline: Random Forest trên summary features

Giống PACE/Benschop — dùng ~19 features tóm tắt:
- MAC per locus, max MAC
- Tổng số đỉnh, mean/std peak height
- Peak height ratios
- Allele frequency statistics

Kỳ vọng: ~77–83% accuracy.

### 3.3 Baseline: Simple MLP

MLP đơn giản trên flattened features (giống TAWSEEM nhưng đơn giản hơn):
- Input: summary features (~50–100 dims)
- Hidden: 3–5 layers, 64 neurons each
- Output: softmax [5] hoặc [10]

**Task cần làm:**
```
models/baseline/mac.py           →  MAC baseline
models/baseline/random_forest.py →  RF baseline
models/baseline/mlp.py           →  MLP baseline
notebooks/01_baseline.py         →  Chạy & so sánh baselines
```

---

## Giai Đoạn 4: Huấn Luyện & Đánh Giá

**Thời gian ước tính:** 5–10 ngày (tùy GPU)

### 4.1 Fine-tuning deepNoC trên PROVEDIt

1. **Nếu có GPU NVIDIA:** Huấn luyện trực tiếp, ~2000 epochs, ước tính 1–4 giờ
2. **Nếu chỉ có CPU:** Giảm xuống ~500 epochs trước, ước tính 12–24 giờ
3. **Nếu dùng Colab:** Upload data, chạy trên GPU T4, ước tính 2–6 giờ

### 4.2 Metrics đánh giá

- **Confusion matrix** (Bảng 2 trong bài báo)
- **Accuracy, Precision, Recall, F1** per NoC
- **Probability threshold analysis** (Figure 7): accuracy vs % classified

### 4.3 Mục tiêu kết quả

| NoC | Mục tiêu (bài báo) | Chấp nhận được |
|---|---|---|
| 1 | 100% | >95% |
| 2 | 99.2% | >90% |
| 3 | 96.5% | >85% |
| 4 | 91.1% | >80% |
| 5 | 92.7% | >80% |

**Task cần làm:**
```
main.py                →  Entry point: train / evaluate / predict
results/               →  Confusion matrices, plots, metrics
notebooks/02_deepnoc.py →  Phân tích kết quả chi tiết
```

---

## Giai Đoạn 5 (Nâng Cao): Pipeline Mô Phỏng

**Thời gian ước tính:** 2–4 tuần  
**Chỉ thực hiện nếu giai đoạn 1–4 thành công**

### 5.1 Cài đặt simDNAmixtures (R package)

```bash
# Cài R trên WSL
sudo apt install r-base
# Trong R
install.packages("simDNAmixtures")
```

Cần calibration data cho GlobalFiler (từ STRmix parameters).

### 5.2 Mô phỏng EPG bằng GAN

Bài báo tham chiếu GAN từ Taylor & Humphries (arXiv:2408.16169). Cần:
- Implement hoặc tìm code của GAN
- Chuyển peak info → EPG signal
- Phát hiện đỉnh trong EPG (theo Woldegebriel)
- Phân loại đỉnh bằng MHCNN

> ⚠️ Đây là phần phức tạp nhất. GAN và MHCNN chưa được công bố mã nguồn. Có thể cần liên hệ tác giả hoặc ước lượng/đơn giản hóa bước này.

### 5.3 Tạo 100K profiles mô phỏng

Với 16GB RAM, nên tạo theo batch (10K/lần) rồi lưu ra disk:
```
data/simulated/batch_01.npz  (10K profiles)
data/simulated/batch_02.npz
...
data/simulated/batch_10.npz
```

---

## Cấu Trúc Thư Mục Đề Xuất

```
deepNoC/
├── data/
│   ├── provedit_processed/     # Dữ liệu đã có
│   ├── provedit_raw/           # ZIP gốc
│   └── simulated/              # Dữ liệu mô phỏng (giai đoạn 5)
├── src/
│   ├── data_loader.py          # Đọc CSV/XLSX → tensor
│   ├── data_split.py           # Train/test split
│   ├── features.py             # Trích xuất 89 features per peak
│   ├── smart_start.py          # Ước lượng mixture proportions
│   └── evaluation.py           # Metrics, confusion matrix, plots
├── models/
│   ├── baseline/
│   │   ├── mac.py
│   │   ├── random_forest.py
│   │   └── mlp.py
│   └── deepnoc/
│       ├── architecture.py     # Model definition (PyTorch)
│       ├── train.py            # Training loop
│       └── losses.py           # Multi-output losses
├── notebooks/
│   ├── 01_baseline.py
│   ├── 02_data_exploration.py
│   └── 03_deepnoc_analysis.py
├── R/
│   └── simulate_profiles.R     # Giai đoạn 5
├── results/
│   ├── confusion_matrix_*.png
│   └── metrics_*.json
├── main.py
├── pyproject.toml
└── README.md
```

---

## Lộ Trình Tổng Thể

```
Tuần 1:  [GĐ 0] Kiểm tra môi trường + [GĐ 1] Xử lý dữ liệu PROVEDIt
Tuần 2:  [GĐ 1] Hoàn thành data pipeline + [GĐ 3] Chạy baselines
Tuần 3:  [GĐ 2] Xây dựng kiến trúc deepNoC
Tuần 4:  [GĐ 4] Huấn luyện & đánh giá, so sánh với bài báo
Tuần 5+: [GĐ 5] Pipeline mô phỏng (tùy chọn)
```

---

## Rủi Ro & Giải Pháp

| Rủi ro | Ảnh hưởng | Giải pháp |
|---|---|---|
| Không có GPU | Training rất chậm | Dùng Google Colab / giảm model size |
| RAM 12GB không đủ | Không load được toàn bộ data | Dùng DataLoader với batch loading |
| Thiếu chi tiết kiến trúc | Không replicate chính xác | Thử nhiều variant, so sánh kết quả |
| Thiếu smart start algorithm | Feature 80–89 không chính xác | Ước lượng bằng simple heuristic hoặc bỏ qua |
| Thiếu peak label probability | Feature 29 không có | Dùng giá trị mặc định hoặc rule-based estimate |
| GAN/MHCNN không public | Không mô phỏng được EPG | Tập trung vào fine-tuning trên PROVEDIt |

---

## Bước Tiếp Theo Ngay Bây Giờ

1. **Chạy `nvidia-smi`** trên WSL để xác nhận GPU
2. **Khám phá dữ liệu** `X_gf25.npy` và `y_gf25.npy` đã có sẵn
3. **Đọc vài file CSV** trong PROVEDIt để hiểu cấu trúc cột
4. **Implement MAC baseline** — nhanh nhất, không cần ML
5. **Quyết định framework**: PyTorch vs TensorFlow (bài báo dùng TF 2.10)
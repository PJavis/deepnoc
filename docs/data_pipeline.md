# Pipeline dữ liệu deepNoC / NoCNet-v2

Tài liệu này giải thích từng bước biến đổi dữ liệu trong repo: từ file `.zip`
gốc trong `data/provedit_raw/`, qua bước giải nén & filter, ra tensor numpy
`[N, 24, 50, 89]`, đến split train/test, cuối cùng là synthetic pool dùng cho
pretrain.

---

## 1. Nguồn dữ liệu — `data/provedit_raw/`

Repo dùng 3 file ZIP tải từ kho PROVEDIt (Catherine Grgicak, Rutgers).

| File ZIP | Dung lượng | Nội dung |
|---|---|---|
| `PROVEDIt_1-5-Person CSVs Filtered.zip` | 27 MB | CSV GeneMapper đã filter, tổ chức theo kit × cycle × NoC × injection. **Đây là nguồn chính** được pipeline đọc. |
| `PROVEDIt_1-Person Profiles_3500 25sec_GF29cycles.zip` | 1.97 GB | EPG thô (file `.fsa`/`.hid`) cho profile 1 người, GlobalFiler 29 cycle, ABI 3500, 25 giây inject. Chỉ giữ để debug, không dùng trong pipeline hiện tại. |
| `PROVEDIt_2-5-Person Profiles_3500 25sec_GF29cycles.zip` | 505 MB | EPG thô cho mixture 2–5 người, cùng kit/máy. Cũng chỉ để debug. |

Pipeline chính chỉ cần ZIP đầu tiên. Hai ZIP còn lại là EPG nhị phân — repo
chưa dùng đến (paper gốc dùng để chạy MHCNN; ta xài plp heuristic thay thế).

---

## 2. Giải nén

Giải `PROVEDIt_1-5-Person CSVs Filtered.zip` vào thư mục
`data/provedit_processed/PROVEDIt_1-5-Person CSVs Filtered/`. Cấu trúc sau khi
giải:

```
data/provedit_processed/PROVEDIt_1-5-Person CSVs Filtered/
├── PROVEDIt_1-5-Person CSVs Filtered_3130_IDPlus28cycles/
├── PROVEDIt_1-5-Person CSVs Filtered_3130_PP16HS32cycles/
├── PROVEDIt_1-5-Person CSVs Filtered_3500_F6C29cycles_hlfrxn/
├── PROVEDIt_1-5-Person CSVs Filtered_3500_GF29cycles/    ← repo dùng nhánh này
└── PROVEDIt_1-5-Person CSVs Filtered_3500_IDPlus29cycles/
```

Mỗi nhánh là một tổ hợp `{máy điện di}_{kit}{số cycle}`. Lệnh giải nén:

```bash
mkdir -p data/provedit_processed
unzip -q "data/provedit_raw/PROVEDIt_1-5-Person CSVs Filtered.zip" \
    -d data/provedit_processed/
```

Dưới mỗi nhánh kit là cây `NoC × injection`:

```
PROVEDIt_1-5-Person CSVs Filtered_3500_GF29cycles/
├── 1-Person/
│   ├── 5 sec/
│   ├── 15 sec/
│   └── 25 sec/             ← repo dùng đúng thư mục này
│       └── RD14-0003_GF_25sec_GM_SE33F_1P.csv
└── 2-5-Persons/
    ├── 5 sec/
    ├── 15 sec/
    └── 25 sec/             ← repo dùng đúng thư mục này
        └── RD14-0003_GF_25sec_GM_SE33F_2-5P.csv
```

---

## 3. Filter — chỉ lấy GlobalFiler + ABI 3500 + 25 giây inject

Loader `src/data_loader.load_provedit_dataset()` filter 3 chiều, mặc định:

| Cờ filter | Giá trị | Ý nghĩa |
|---|---|---|
| `instrument_filter` | `3500` | Máy điện di ABI 3500 (không lấy 3130). |
| `kit_filter` | `GF` | Kit GlobalFiler™. |
| `injection_filter` | `25sec` | Thời gian inject 25 giây. |

Filter này khớp với paper gốc deepNoC (Taylor & Humphries 2024) để so sánh
fair. Sau filter, loader đọc đúng 2 file CSV:

```
RD14-0003_GF_25sec_GM_SE33F_1P.csv      (single-source, NoC = 1)
RD14-0003_GF_25sec_GM_SE33F_2-5P.csv    (mixture 2..5 người)
```

---

## 4. Định dạng CSV GeneMapper

Mỗi CSV ở dạng **wide format**, một dòng = một (sample, locus). Cột:

```
Sample File , Marker , Dye , Allele 1 , Size 1 , Height 1 , Allele 2 , Size 2 , Height 2 , … (đến Allele 100/Size 100/Height 100)
```

Một row mẫu (đã rút gọn cho dễ đọc):

```
A02_RD14-0003-15d2U60-0.25GF-Q4.5_01.25sec.hid , D3S1358 , B ,
  OL , 100   , 14 ,
  13 , 112.92, 14 ,
  15 , 121.07, 4276 ,
  16 , 125.28, 4213 ,
  17 , 129.38, 18 ,
  17.1 , 130.16, 8 , (còn lại trống)
```

Cột nghĩa là:

| Cột | Ý nghĩa |
|---|---|
| `Sample File` | tên file `.hid` gốc — chứa pedigree đầy đủ (xem báo cáo VI § 3.6) |
| `Marker` | tên locus (D3S1358, vWA, AMEL, …) |
| `Dye` | kênh màu (B, G, Y, R, P) |
| `Allele K` | giá trị allele peak thứ K (số repeat; ký tự `X`/`Y` cho AMEL; `OL` = off-ladder bị loại) |
| `Size K` | kích thước fragment (base pair) |
| `Height K` | chiều cao peak (RFU) |

Tới 100 peak được hỗ trợ theo header, thực tế peak thường chỉ vài chục.

---

## 5. Pipeline biến đổi CSV → tensor (`main.py prepare`)

Lệnh:

```bash
python main.py prepare \
    --data-dir "data/provedit_processed/PROVEDIt_1-5-Person CSVs Filtered" \
    --output-dir data/provedit_processed \
    --kit GF --instrument 3500 --injection 25sec
```

Trình tự xử lý trong `src/data_loader.py`:

1. **Tìm file CSV/XLSX** dưới `--data-dir` có chứa cả 3 substring filter
   (`3500`, `GF`, `25sec`).
2. **Đọc wide table** bằng `pandas.read_csv` (delimiter tự dò `,` hoặc `\t`).
   Strip whitespace tên cột, locate `Sample Name | Marker | Dye`.
3. **Melt wide → long**: với mỗi dòng (sample, locus), duyệt 100 cột peak; emit
   một record `{SampleName, Marker, Dye, Allele, Size, Height}` cho mỗi peak
   có giá trị hợp lệ.
4. **Lọc peak**:
   - bỏ peak Allele rỗng / `NaN`
   - bỏ Allele `OL` (off-ladder)
   - bỏ Allele không parse được thành float (trừ `X`→1.0, `Y`→2.0 ở AMEL)
   - bỏ Height ≤ 0
   - bỏ Locus không thuộc 24 locus GlobalFiler chuẩn
5. **Chuẩn hoá tên locus**: alias `VWA` → `vWA`, `Yindel` → `Y-Indel`, …
6. **Suy ra NoC**:
   - từ tên file: `_1P.csv` → 1; `_2P.csv` → 2; ...; `_2-5P.csv` → multi.
   - từ tên sample (cho file `2-5P`): pattern `-1;1-` → 2 người, `-1;2;1-` →
     3 người, `-1;1;1;1-` → 4, `-1;1;1;1;1-` → 5. Fallback `1to1` /
     `2p,3p,4p,5p`. Nếu không suy được → bỏ sample.
7. **Cap số profile NoC=1**: tham số `max_1person` (mặc định ∞ ở pipeline
   chính, hoặc 70 nếu muốn giảm imbalance). Repo hiện tại không cap → giữ
   2712 profile NoC=1.
8. **Detect stutter** cho từng locus của từng sample bằng
   `detect_stutter_relationships()`: tìm cặp peak có Allele lệch ±1 / ±2 /
   ±0.2 repeat, peak thấp hơn là stutter, peak cao hơn là parent.
9. **Build feature 89 chiều mỗi peak** bằng `build_peak_features()`:
   - idx 0..23: one-hot 24 locus
   - idx 24: allele / 100
   - idx 25: size / 100
   - idx 26: height / 33000
   - idx 27: allele frequency (mặc định 0.01)
   - idx 28: peak label probability (heuristic dựa height tương đối + có là stutter không)
   - idx 29..76: 4 loại stutter × 6 giá trị × 2 chiều = 48 ô
   - idx 77: số peak ở locus / 100
   - idx 78: số peak toàn profile / 1000
   - idx 79..88: mixture proportions của 10 donor lớn nhất (smart-start)
10. **Build tensor profile** `[24, 50, 89]`: gom peak theo locus, zero-pad
    nếu < 50 peak, sort theo Size, truncate nếu > 50.
11. **Stack** tất cả profile thành `X [N, 24, 50, 89]`, label `y [N]`, name `[N]`.

---

## 6. Output sau `prepare`

```
data/provedit_processed/
├── X_gf25.npy           # float32  (3378, 24, 50, 89) ≈ 1.4 GB
├── y_gf25.npy           # int64    (3378,)             ≈ 26 KB
└── sample_names.txt     # text     3378 dòng
```

Phân phối NoC:

| NoC | Số profile | % |
|---|---|---|
| 1 | 2712 | 80.3% |
| 2 |  174 |  5.2% |
| 3 |  160 |  4.7% |
| 4 |  176 |  5.2% |
| 5 |  156 |  4.6% |
| **Tổng** | **3378** | 100% |

Lưu ý mất cân bằng nặng. Sample name dạng:

```
RD14-0003_GF_25sec_GM_SE33F_2-5P:A02_RD14-0003-31_32-1;1-M2c-0.03GF-Q2.0_01.25sec.hid
└─ file stem ──────────────────┘└─ tên sample bên trong file ───────────────────┘
                                         └─ pedigree (donor 31+32, tỷ lệ 1:1)
```

---

## 7. Split train / test — `src/split.py`

Mặc định grouped stratified với seed = 42, test_size = 0.25.

**Tại sao không random hoặc alternating?**

- `train_test_split_alternating` (cũ) lấy mỗi profile thứ 2 → các replicate của
  cùng một mixture sinh học bị chia hai phía → leakage → test acc bị thổi.
- `stratified_split` random theo NoC → vẫn có thể leak vì cùng pedigree rớt
  hai phía.
- `grouped_stratified_split` tạo **pedigree key** từ tên sample
  (vd `31_32-1;1`), dùng `GroupShuffleSplit` để mọi replicate của cùng
  pedigree đều ở một phía split, đồng thời cố giữ phân phối NoC ở cả hai phía.

Trên dataset hiện tại, 3378 profile gom thành **65 pedigree key**. Sau split:

| Chỉ tiêu | Train | Test |
|---|---|---|
| Số profile | 2455 | 923 |
| NoC=1 | 2021 | 691 |
| NoC=2 |  126 | 48 |
| NoC=3 |   96 | 64 |
| NoC=4 |  162 | 14 |
| NoC=5 |   50 | 106 |

`NoC=4` test chỉ có 14 mẫu — do số pedigree chứa NoC=4 rơi vào test rất ít.
Đây là nguyên nhân chính làm metric NoC=4 nhiễu (acc 0.143 với 95% CI rộng).

Code sử dụng:

```python
from src.split import grouped_stratified_split
X_train, X_test, y_train, y_test, names_train, names_test = (
    grouped_stratified_split(X, y, names, test_size=0.25, seed=42)
)
```

---

## 8. Synthetic pool — `src/synth.py`

Pretrain cần nhiều data hơn 2455 profile train thật. Script `src/synth.py`
sinh mixture nhân tạo bằng cách superposition các profile NoC=1 thật theo
trọng số Dirichlet.

Lệnh:

```bash
python -m src.synth \
    --source data/provedit_processed/X_gf25.npy \
    --labels data/provedit_processed/y_gf25.npy \
    --out-dir data/synthetic \
    --n 30000 --max-noc 5 \
    --alpha 1.5 --threshold 50 --jitter 0.08 --seed 0
```

| Cờ | Giá trị | Vai trò |
|---|---|---|
| `--source` / `--labels` | từ bước prepare | Lấy pool NoC=1 làm thành phần. |
| `--n` | 30000 | Số mixture sinh ra. |
| `--max-noc` | 5 | NoC cao nhất được sinh. |
| `--alpha` | 1.5 | Concentration của Dirichlet — alpha lớn → tỷ lệ donor đồng đều, alpha nhỏ → lệch nhiều. |
| `--threshold` | 50 RFU | Peak có height < threshold bị xem là không phát hiện được. |
| `--jitter` | 0.08 | Log-normal σ áp lên height — mô phỏng noise injection. |
| `--seed` | 0 | Random seed cho reproducibility. |

Quy trình mỗi mixture:

1. Sample `k ~ Uniform{1, …, max_noc}`.
2. Sample `k` profile NoC=1 ngẫu nhiên khác nhau từ pool.
3. Sample trọng số `w ~ Dirichlet(alpha, …, alpha)` ∈ Δ^{k−1}.
4. Tại mỗi (locus, allele), cộng dồn `height × w_donor` qua k donor.
5. Áp jitter log-normal lên từng peak.
6. Bỏ peak có height < threshold sau khi cộng dồn.
7. Recompute feature global (số peak/locus, số peak/profile, mix_props).

### Output `data/synthetic/`

| File | Shape | Dung lượng | Ghi chú |
|---|---|---|---|
| `X.npy` | `(30000, 24, 50, 89)` float32 | 12.0 GB | Tensor đầu vào — phải `mmap_mode='r'` (đã fix bug). |
| `y.npy` | `(30000,)` int64 | 235 KB | Label NoC. |
| `mix.npy` | `(30000, 10)` float32 | 1.2 MB | Mixture proportions per profile. |
| `locus_nall.npy` | `(30000, 24)` int8 | 704 KB | Số allele kỳ vọng mỗi locus per profile. |

**Cảnh báo RAM**: `X.npy` 12 GB. Pre-fix, `SynthProfileDataset.__init__`
gọi `np.load(X.npy)` (không mmap) → load toàn bộ vào RAM → OOM trên WSL 14 GB.
Đã fix bằng `np.load(..., mmap_mode='r')` ở commit `23c732b`. Tham khảo
báo cáo VI § 5.1.

---

## 9. Pipeline train end-to-end

Tóm tắt thứ tự lệnh (từ zip → checkpoint cuối acc 0.943):

```bash
# 0. Chuẩn bị môi trường
source .venv/bin/activate

# 1. Giải nén CSV
unzip -q "data/provedit_raw/PROVEDIt_1-5-Person CSVs Filtered.zip" \
    -d data/provedit_processed/

# 2. CSV → tensor numpy (3378 profile)
python main.py prepare \
    --data-dir "data/provedit_processed/PROVEDIt_1-5-Person CSVs Filtered" \
    --output-dir data/provedit_processed

# 3. Sinh synthetic pool 30k mixture
python -m src.synth \
    --source data/provedit_processed/X_gf25.npy \
    --labels data/provedit_processed/y_gf25.npy \
    --out-dir data/synthetic --n 30000 --max-noc 5

# 4. Pretrain NoCNet-v2 trên synthetic (split grouped tự động)
python main.py train --model nocnet_v2 \
    --output-dir data/provedit_processed --synth-dir data/synthetic \
    --epochs 300 --batch-size 12 --samples-per-epoch 3000 \
    --p-synth 0.85 --d-model 96

# 5. Finetune real-only trên PROVEDIt thật
python main.py finetune --checkpoint results/best_nocnet_v2.pt \
    --output-dir data/provedit_processed --results-dir results \
    --epochs 120 --batch-size 12 --samples-per-epoch 1500

# 6. Tune per-class bias trên holdout
python main.py tune --checkpoint results/best_nocnet_v2_ft.pt \
    --val-frac 0.5 --metric macro_f1 --tta --tta-samples 20 \
    --out-name tune_ft_step2.json

# 7. Eval cuối (TTA + bias áp lên full test)
python scripts/eval_nocnet_v2.py \
    --checkpoint results/best_nocnet_v2_ft.pt --tta --tta-samples 20
```

Sau bước 7, kết quả mong đợi: **ALL accuracy = 0.943** trên 923 profile
grouped test (seed = 42).

---

## 10. Tham chiếu nhanh

| Hỏi | Trả lời |
|---|---|
| Dataset gốc lấy từ đâu? | 3 file ZIP trong `data/provedit_raw/`; chỉ ZIP đầu tiên (CSV Filtered) thực sự dùng. |
| Filter gì? | `instrument=3500 + kit=GF + injection=25sec`. |
| Tensor đầu vào shape gì? | `[N, 24, 50, 89]` (locus × peak × feature). |
| Tổng profile sau prepare? | 3378 (2712 NoC=1 + 666 NoC=2..5). |
| Tổng profile train/test sau split? | 2455 / 923 (grouped stratified, seed=42). |
| Synthetic pool ở đâu? | `data/synthetic/X.npy` (12 GB) + 3 file phụ. |
| Bug RAM đã fix? | Có — `mmap_mode='r'` trong `SynthProfileDataset`, commit `23c732b`. |
| Checkpoint cuối? | `results/best_nocnet_v2_ft.pt` + `results/tune_ft_step2.json` (bias). |
| Pipeline inference? | TTA 20× → softmax mean → `log(probs) + bias` → argmax + 1. |
| Mã nguồn liên quan? | `src/data_loader.py`, `src/split.py`, `src/synth.py`, `models/nocnet_v2/train.py`, `scripts/eval_nocnet_v2.py`. |

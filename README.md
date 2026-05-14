# deepNoC

Repo này là bản triển khai thực nghiệm cho bài toán dự đoán số người đóng góp (`NoC`, Number of Contributors) từ hồ sơ STR DNA, lấy cảm hứng từ bài báo *deepNoC* của Taylor & Humphries (2024). Trạng thái hiện tại của codebase là một pipeline gọn để:

- chuẩn bị dữ liệu PROVEDIt từ GeneMapper CSV/XLSX,
- chạy baseline `MAC` và `Random Forest`,
- huấn luyện model `simple` / `full` (deepNoC CNN), `nocformer`, hoặc `nocnet_v2`,
- sinh dữ liệu hỗn hợp tổng hợp (synthetic mixtures) từ pool NoC=1,
- fine-tune trên dữ liệu thật sau khi pretrain trên synthetic,
- chạy 5-fold grouped cross-validation để so sánh các model trên cùng split,
- đánh giá bằng confusion matrix, accuracy, MAE, off-by-one acc, macro-F1.

README này mô tả đúng những gì repo đang chạy được bây giờ, không phải kế hoạch dài hạn ban đầu.

## Model nào nên dùng?

Đang có 4 nhánh model. Khuyến nghị:

- `nocnet_v2` — **mặc định mới**. Deep Sets + stutter-aware attention + count-aware multi-head. Train trên dữ liệu synthetic rồi fine-tune trên PROVEDIt thật.
- `nocformer` — bản hierarchical Transformer cũ. Overfit trên dữ liệu PROVEDIt nhỏ; chỉ giữ để so sánh.
- `simple` / `full` — deepNoC CNN gốc theo paper. Vẫn chạy được nhưng số liệu trong các README/checkpoint cũ tính trên split `alternating` (bị leak replicate), nên không phải số honest. So sánh fair phải dùng split `grouped`.

## Pipeline đề xuất cho NoCNet-v2

Bốn bước, có thể chạy tuần tự bằng `main.py`:

```bash
# 1. Chuẩn bị dữ liệu PROVEDIt -> .npy
python main.py prepare --data-dir "data/provedit_processed/PROVEDIt_1-5-Person CSVs Filtered"

# 2. Sinh pool synthetic mixtures từ NoC=1 (chạy 1 lần, 5-15 phút)
python main.py synth --n 30000 --max-noc 5

# 3. Pretrain NoCNet-v2 (synthetic dominant + real)
python main.py train --model nocnet_v2 --epochs 100 --batch-size 16 \
    --p-synth 0.85 --samples-per-epoch 4000 --split grouped

# 4. Fine-tune trên PROVEDIt thật
python main.py finetune --checkpoint results/best_nocnet_v2.pt \
    --epochs 40 --lr 1e-5 --tag nocnet_v2_ft
```

Cuối cùng có thể chạy 5-fold grouped CV để có bảng so sánh fair với baseline và các model cũ:

```bash
python main.py cv --models mac rf deepnoc_full nocformer nocnet_v2 \
    --folds 5 --epochs 80
```

Trên GPU 16 GB tăng quy mô:

```bash
python main.py synth --n 30000 --max-noc 5
python main.py train --model nocnet_v2 --epochs 150 --batch-size 64 \
    --samples-per-epoch 8000 --p-synth 0.85
python main.py finetune --checkpoint results/best_nocnet_v2.pt --epochs 60
```

Mặc định peak VRAM:
- `--batch-size 16` ≈ 0.7 GiB
- `--batch-size 32` ≈ 1.3 GiB
- `--batch-size 64` ≈ ~2.5 GiB

Train trên CPU vẫn chạy được nhưng sẽ rất chậm.

## Trạng thái hiện tại

Pipeline đã chạy được end-to-end trên bộ PROVEDIt `GlobalFiler + ABI 3500 + 25 sec`.

Sau khi sửa parser tên sample cho file `2-5P`, bước `prepare` hiện tạo ra:

- `3378` profiles
- `NoC=1: 2712`
- `NoC=2: 174`
- `NoC=3: 160`
- `NoC=4: 176`
- `NoC=5: 156`

Output mặc định được lưu tại:

- `data/provedit_processed/X_gf25.npy`
- `data/provedit_processed/y_gf25.npy`
- `data/provedit_processed/sample_names.txt`

## Cấu trúc repo

```text
deepNoC/
├── main.py                  # CLI entrypoint: prepare/synth/train/finetune/cv/...
├── src/
│   ├── constants.py
│   ├── data_loader.py       # GeneMapper CSV -> [N, 24, 50, 89] tensor
│   ├── synth.py             # Sinh synthetic mixtures + physics (stutter/dropout/noise)
│   ├── split.py             # stratified / grouped split
│   ├── cv.py                # 5-fold grouped CV runner
│   └── evaluation.py
├── models/
│   ├── baseline/
│   │   └── baselines.py     # MAC + Random Forest
│   ├── deepnoc/             # CNN gốc theo paper (simple / full)
│   ├── nocformer/           # Transformer cũ
│   └── nocnet_v2/           # MỚI: Deep Sets + stutter-bias attn + count head
│       ├── architecture.py
│       ├── losses.py
│       └── train.py         # gồm cả finetune_nocnet_v2()
├── data/
│   ├── provedit_raw/
│   ├── provedit_processed/  # X_gf25.npy, y_gf25.npy, sample_names.txt
│   └── synthetic/           # X.npy, y.npy, mix.npy, locus_nall.npy
├── results/
├── pyproject.toml
└── README.md
```

## Yêu cầu môi trường

- Python `>= 3.11`
- Nên dùng virtual environment
- GPU không bắt buộc, nhưng train trên CPU sẽ chậm

Dependencies chính đã có trong [pyproject.toml](/home/nguyenquocdung/work/deepNoC/pyproject.toml): `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`, `tqdm`, `jupyter`.

## Cài đặt

Nếu dùng `uv`:

```bash
uv sync
source .venv/bin/activate
```

Nếu dùng `venv` thuần:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Nếu `pip install -e .` không phù hợp với môi trường của bạn, cài tối thiểu các package trong `pyproject.toml`.

## Dữ liệu

Repo hiện đang làm việc với dữ liệu PROVEDIt dạng GeneMapper đã lọc. Mặc định `prepare` tìm dữ liệu ở:

```text
data/provedit_processed/PROVEDIt_1-5-Person CSVs Filtered
```

Loader sẽ tiếp tục lọc theo:

- `instrument = 3500`
- `kit = GF`
- `injection = 25sec`

Mỗi profile được chuyển thành tensor có shape:

```text
[24, 50, 89]
```

Trong đó:

- `24` loci
- `50` peaks tối đa mỗi locus
- `89` features mỗi peak

Nhãn hiện tại dùng cho train/baseline là `profile-level NoC`.

## Mô tả chi tiết định dạng dữ liệu

Phần này mô tả đúng định dạng dữ liệu mà repo đang đọc và cách nó được biến đổi thành tensor đầu vào cho mô hình.

### 1. Dữ liệu đầu vào thô

Loader hiện hỗ trợ:

- `.csv`
- `.xlsx`
- `.xls`

Các file GeneMapper thường ở dạng wide table, nghĩa là một dòng có thể chứa nhiều peak của cùng một marker thông qua các cột lặp lại như:

- `Allele 1`, `Allele 2`, ...
- `Size 1`, `Size 2`, ...
- `Height 1`, `Height 2`, ...

Ngoài ra loader còn tìm các cột định danh chính:

- cột sample: một trong `Sample Name`, `SampleName`, `Sample File`, `Sample`
- cột locus/marker: một trong `Marker`, `Locus`, `marker`, `locus`
- cột dye: một trong `Dye`, `dye`, `Color`, `Dye Color`

Nếu không tìm thấy cột sample hoặc marker, loader sẽ báo lỗi và bỏ qua file đó.

### 2. Chuẩn hóa về long format

Trong [src/data_loader.py](/home/nguyenquocdung/work/deepNoC/src/data_loader.py), dữ liệu wide format được chuyển về long format với mỗi dòng tương ứng một peak hợp lệ. DataFrame trung gian sau chuẩn hóa có các cột:

- `SampleName`
- `Marker`
- `Dye`
- `Allele`
- `Size`
- `Height`

Các peak bị loại ở bước này gồm:

- peak không có allele,
- peak có `Height <= 0`,
- allele không parse được,
- allele `OL` (off-ladder),
- locus không thuộc bộ 24 locus của GlobalFiler.

Riêng `AMEL`, allele ký tự được ánh xạ như sau:

- `X -> 1.0`
- `Y -> 2.0`

### 3. Bộ locus được sử dụng

Repo hiện cố định đúng `24` locus của GlobalFiler theo thứ tự sau:

```text
D3S1358, vWA, D16S539, CSF1PO, TPOX,
Y-Indel, AMEL, D8S1179, D21S11, D18S51,
DYS391, D2S441, D19S433, TH01, FGA,
D22S1045, D5S818, D13S317, D7S820, SE33,
D10S1248, D1S1656, D12S391, D2S1338
```

Thứ tự này quan trọng vì nó được dùng trực tiếp cho:

- one-hot encoding của locus,
- trục đầu tiên của tensor đầu vào `[24, 50, 89]`.

Một số alias được map về tên chuẩn, ví dụ:

- `Yindel` hoặc `Y Indel` -> `Y-Indel`
- `VWA` -> `vWA`
- `Amelogenin` -> `AMEL`

### 4. Cách suy ra nhãn NoC

Nhãn `y` hiện tại là số contributor ở cấp profile.

Loader suy ra `NoC` theo thứ tự ưu tiên:

1. từ tên file, ví dụ `_1P.csv`, `_2P.csv`, `_3P.csv`
2. nếu là file multi-person kiểu `_2-5P.csv`, parse từ sample name

Repo hiện đã hỗ trợ đúng sample name PROVEDIt dạng:

- `...-1;1-...` -> `NoC = 2`
- `...-1;2;1-...` -> `NoC = 3`
- `...-1;1;1;1-...` -> `NoC = 4`
- `...-1;1;1;1;1-...` -> `NoC = 5`

Ngoài ra vẫn giữ fallback cho các pattern cũ như:

- `1to1`
- `1to1to1`
- `2p`, `3p`, `4p`, `5p`

Nếu không suy ra được `NoC`, sample sẽ bị bỏ qua.

### 5. Tensor đầu ra cho mỗi profile

Mỗi profile sau xử lý được biểu diễn thành tensor:

```text
[24, 50, 89]
```

Ý nghĩa từng chiều:

- `24`: số locus cố định của GlobalFiler
- `50`: số peak tối đa mỗi locus
- `89`: số đặc trưng cho mỗi peak

Nếu một locus có ít hơn 50 peak thì phần còn lại được zero-pad.

Nếu một locus có nhiều hơn 50 peak thì chỉ giữ tối đa 50 peak đầu sau khi sắp xếp.

Toàn bộ dataset sau `prepare` có dạng:

- `X`: `[N, 24, 50, 89]`
- `y`: `[N]`

### 6. Cấu trúc 89 đặc trưng mỗi peak

Các feature được xây trong `build_peak_features()` và `build_profile_tensor()`.

#### Nhóm 1: Định danh locus và peak cơ bản

- `1-24`: one-hot locus
- `25`: allele đã chuẩn hóa bằng `ALLELE_NORM = 100`
- `26`: size đã chuẩn hóa bằng `SIZE_NORM = 100`
- `27`: height đã chuẩn hóa bằng `HEIGHT_NORM = 33000`
- `28`: allele frequency
- `29`: peak label probability

Trong code, chỉ số mảng Python tương ứng là:

- `0:24` cho one-hot locus
- `24` cho allele
- `25` cho size
- `26` cho height
- `27` cho allele frequency
- `28` cho peak label probability

#### Nhóm 2: Thông tin stutter

Feature `30-77` mã hóa quan hệ stutter cho 4 loại:

- back stutter
- double-back stutter
- forward stutter
- 0.2-repeat stutter

Mỗi loại có 2 hướng thông tin:

- peak hiện tại là stutter của một peak cha
- peak hiện tại là peak cha của một stutter

Với mỗi hướng, code lưu các giá trị như:

- allele liên quan,
- height liên quan,
- tỉ lệ chiều cao,
- expected stutter ratio,
- allele frequency,
- peak label probability.

Tổng cộng phần này chiếm `48` feature.

#### Nhóm 3: Đặc trưng mức locus và mức profile

- `78`: tổng số peak tại locus, chuẩn hóa theo `LOCUS_PEAK_NORM = 100`
- `79`: tổng số peak trong profile, chuẩn hóa theo `PROFILE_PEAK_NORM = 1000`
- `80-89`: estimated mixture proportions cho tối đa 10 contributor

### 7. Cách ước lượng peak label probability

Repo hiện chưa có MHCNN như paper gốc, nên `peak label probability` đang là heuristic:

- peak càng cao so với peak lớn nhất trong locus thì xác suất càng cao,
- peak nằm ở vị trí có khả năng là stutter thì bị giảm xác suất.

Giá trị này được chặn trong khoảng:

```text
[0.01, 0.99]
```

Nghĩa là đây là feature xấp xỉ để pipeline chạy được, chưa phải bản tái hiện chính xác hoàn toàn từ paper.

### 8. Cách ước lượng mixture proportions

Feature `80-89` hiện được tạo từ hàm `estimate_smart_start()`, là một bản đơn giản hóa ý tưởng `smart start`.

Logic hiện tại:

- gom tất cả peak height trong profile,
- sắp giảm dần,
- chia thành các cụm thô để ước lượng tỷ lệ contributor,
- chuẩn hóa để tổng bằng `1`,
- sắp xếp contributor lớn nhất trước.

Vì vậy phần mixture proportion hiện nên được hiểu là feature hỗ trợ mang tính heuristic, không phải ground-truth contributor proportion.

### 9. Tên sample lưu sau khi prepare

Ngoài `X` và `y`, pipeline còn lưu `sample_names.txt`.

Mỗi dòng có format:

```text
<file_stem>:<sample_name>
```

Ví dụ:

```text
RD14-0003_GF_25sec_GM_SE33F_2-5P:A02_RD14-0003-31_32-1;1-M2c-0.03GF-Q2.0_01.25sec.hid
```

File này hữu ích để:

- truy vết profile gốc,
- debug sample bị dự đoán sai,
- đối chiếu tensor với file nguồn.

### 10. Tóm tắt luồng biến đổi dữ liệu

Luồng xử lý hiện tại có thể tóm tắt như sau:

```text
GeneMapper CSV/XLSX
-> chọn file đúng filter (3500, GF, 25sec)
-> đọc wide table
-> chuyển thành peak-level long table
-> chuẩn hóa tên locus
-> parse NoC từ filename / sample name
-> gom peak theo sample
-> build tensor [24, 50, 89]
-> lưu X_gf25.npy, y_gf25.npy, sample_names.txt
```

### 11. Những giới hạn cần biết khi đọc dữ liệu

Phần dữ liệu hiện tại có vài giới hạn quan trọng:

- allele frequency đang dùng giá trị mặc định xấp xỉ, chưa phải bảng allele frequency đầy đủ,
- peak label probability là heuristic,
- mixture proportion là ước lượng đơn giản hóa,
- label huấn luyện hiện tại tập trung vào `profile-level NoC`,
- chưa có pipeline sinh toàn bộ label phụ ở mức peak/locus như bản đầy đủ của paper.

Vì vậy, phần định dạng dữ liệu hiện tại phù hợp để chạy baseline, train thử và phân tích mô hình `NoC`, nhưng chưa nên coi là bản tái hiện hoàn chỉnh toàn bộ pipeline dữ liệu của bài báo.

## NoCNet-v2: kiến trúc và lý do

NoCNet-v2 là bản kiến trúc mới được thiết kế riêng cho bài toán NoC trên PROVEDIt nhỏ. Mục tiêu là khắc phục 3 vấn đề chính của các model cũ:

1. **Dữ liệu multi-contributor cực ít** (NoC=2..5 chỉ ~160-176 profile mỗi class). → giải bằng synthetic mixture pool sinh từ NoC=1.
2. **Pooling bằng CLS-token làm mất tín hiệu "count"**. → giải bằng Deep Sets `[sum, max, log1p(count)]`.
3. **Self-attention không có inductive bias cho stutter**. → giải bằng allele-distance attention bias.

Sơ đồ chính:

```text
[B, 24, 50, 89]
  -> PeakEmbedder (89 -> 96)
  -> 2x StutterBiasAttention   (attn logits + MLP(allele_i - allele_j))
  -> DeepSetsPool [sum, max, log1p(count)]   <- count-preserving
  -> 2x CrossLocusTransformer  (dye + locus pos)
  -> ProfilePool [sum, mean, max]
  -> CountAwareHead -> {softmax, scalar, CORN}   <- ensemble 3 view
  + aux per-locus n_alleles head (chỉ học khi sample là synthetic)
```

Model ~574k params, peak VRAM ~1.3 GiB ở batch=32.

## Synthetic mixture pool (`src/synth.py`)

Sinh dữ liệu trộn lên đến 5 contributor từ pool NoC=1, có physics đầy đủ:

- siêu vị (superposition) chiều cao peak theo trọng số Dirichlet,
- tái sinh stutter (back / dbl-back / forward / 0.2) quanh từng parent peak với tỉ lệ kỳ vọng ± CV,
- allelic dropout xác suất `p = 0.30 * exp(-h / 250)`,
- baseline noise peak `Poisson(0.5)` mỗi locus,
- gom (rebucket) stutter và peak thật ở cùng allele rồi mới lọc theo LOD,
- ghi lại nhãn ground-truth (NoC, mix proportions sorted, true n_alleles per locus) trước khi thêm artefact.

Lệnh sinh:

```bash
python main.py synth --n 30000 --max-noc 5
# tuỳ chọn: --alpha 1.5 --threshold 50 --jitter 0.08
```

Output `data/synthetic/X.npy`, `y.npy`, `mix.npy`, `locus_nall.npy`. Đọc bằng `np.memmap` lúc train, không tốn RAM.

## Cách chạy

### 1. Chuẩn bị dữ liệu

```bash
python main.py prepare --data-dir "data/provedit_processed/PROVEDIt_1-5-Person CSVs Filtered"
```

Lưu `X_gf25.npy`, `y_gf25.npy`, `sample_names.txt` vào `data/provedit_processed/`.

### 2. Chạy baseline

```bash
python main.py baseline
```

`MAC` + `Random Forest` trên summary features. Lưu metrics/confusion matrix vào `results/`.

### 3. Sinh synthetic pool

```bash
python main.py synth --n 30000 --max-noc 5
```

### 4. Pretrain NoCNet-v2

```bash
python main.py train --model nocnet_v2 --epochs 100 --batch-size 16 \
    --p-synth 0.85 --samples-per-epoch 4000 --split grouped
```

Lưu best checkpoint `results/best_nocnet_v2.pt`. Trong tail 25% epoch sẽ tự kích hoạt SWA (Stochastic Weight Averaging) và pick best giữa live model vs SWA model.

Knob quan trọng:

- `--p-synth` (default 0.8): tỉ lệ sample synthetic mỗi batch
- `--samples-per-epoch`: số lần lấy mẫu mỗi epoch
- `--d-model 96 --peak-layers 2 --locus-layers 2`: kích thước default đã được fit cho 6 GB VRAM
- `--no-synth`: tắt synthetic pool nếu chỉ muốn train trên real

### 5. Fine-tune trên dữ liệu thật

```bash
python main.py finetune --checkpoint results/best_nocnet_v2.pt \
    --epochs 40 --lr 1e-5 --tag nocnet_v2_ft
```

Fine-tune chỉ chạy trên real PROVEDIt, low LR. Trước khi train sẽ eval baseline của checkpoint pretrain để best-ckpt không bao giờ tệ hơn pretrain. SWA cũng được dùng trong tail.

Tuỳ chọn `--freeze-peak` để chỉ fine-tune cross-locus + heads, giữ peak encoder cố định (nên dùng nếu real data quá ít).

### 6. Train model cũ (tham khảo)

```bash
python main.py train --model simple   # deepNoC CNN gốc
python main.py train --model full     # deepNoC full, nhiều aux head
python main.py train --model nocformer
```

Lưu ý: số liệu cũ trong `results/` được tính trên split `alternating`. Split này leak replicate giữa train/test (xem `src/split.py:1-9`), nên accuracy bị thổi phồng. Để so sánh fair phải dùng `--split grouped` hoặc chạy `cv`.

### 7. Đánh giá checkpoint riêng lẻ

```bash
python main.py evaluate --checkpoint results/best_model_simple.pt --model simple
python main.py evaluate --checkpoint results/best_model_full.pt --model full
```

### 8. 5-fold grouped cross-validation

```bash
python main.py cv --models mac rf deepnoc_full nocformer nocnet_v2 \
    --folds 5 --epochs 80 --batch-size 16
```

Group key được trích từ pedigree trong sample name (`src/split.py:_pedigree_key`), đảm bảo replicate cùng một mẫu sinh học không nằm cả ở train và test. Output:

- `results/cv/<model>/fold<k>/metrics.json`
- `results/cv/<model>/summary.json` — mean ± std cho accuracy, MAE, off-by-one acc, macro-F1
- `results/cv/summary_all.json` — tất cả model trong cùng file

### 9. Chạy toàn bộ pipeline cũ

```bash
python main.py all
```

Lệnh này chỉ chạy `prepare → baseline → train` model cũ. Pipeline mới (NoCNet-v2 + synth + finetune) chưa được gộp vào `all` — chạy từng bước theo `### 1` -> `### 5`.

## Cấu hình theo phần cứng

| GPU VRAM | Đề xuất                                                                  |
|---------:|--------------------------------------------------------------------------|
| 6 GB     | `--batch-size 16 --samples-per-epoch 4000 --n 30000 --epochs 80`         |
| 12 GB    | `--batch-size 32 --samples-per-epoch 6000 --n 60000 --epochs 120`        |
| 16+ GB   | `--batch-size 64 --samples-per-epoch 8000 --n 100000 --epochs 150`       |
| CPU      | `--batch-size 8 --samples-per-epoch 1000 --n 10000 --epochs 30` (chậm)   |

## Kết quả đầu ra

Trong `results/`, repo hiện sinh ra các file như:

- `confusion_matrix_*.png`
- `metrics_*.json`
- `threshold_*.png`
- `training_history_*.png`
- `best_model_*.pt`, `best_nocformer.pt`, `best_nocnet_v2.pt`, `best_nocnet_v2_ft.pt`
- `checkpoint_*_ep*.pt`
- `history_*.json` (bao gồm `swa_test_acc` cho NoCNet-v2)

## Report
- `report/build_report.py` tạo báo cáo NoCFormer.
- `report/build_nocnet_v2_report.py` tạo báo cáo cho NoCNet-v2 mới.

CV runner thêm:

- `results/cv/<model>/fold<k>/metrics.json`
- `results/cv/<model>/summary.json`
- `results/cv/summary_all.json`

## Những gì đã đúng với code hiện tại

- Prepare dữ liệu từ PROVEDIt CSV/XLSX, parse sample name `-1;1-`, `-1;2;1-`, `-1;1;1;1-`.
- 3 chiến lược split: `alternating` (leak — chỉ giữ cho tương thích), `stratified`, `grouped` (khuyến nghị).
- Baseline `MAC` + `Random Forest`.
- 4 nhánh model: `simple`, `full`, `nocformer`, `nocnet_v2`.
- Synthetic mixture generator có physics (stutter regen + dropout + noise + LOD).
- SWA tự kích hoạt trong tail 25% epoch khi train `nocnet_v2`.
- Fine-tune real-only với baseline-guard (best ckpt không bao giờ tệ hơn pretrain).
- 5-fold grouped CV runner so sánh các model trên cùng split.
- Module đánh giá: accuracy, MAE, off-by-one acc, macro-F1, confusion matrix.

## Những gì chưa nên nói quá mức

- Allele frequency vẫn là default xấp xỉ, chưa nạp bảng tần suất thực tế.
- Peak label probability đầu vào là heuristic; chưa thay bằng MHCNN như paper.
- Mixture proportion feature 80-89 (heuristic smart-start) chỉ dùng làm input feature; với NoCNet-v2 nhãn `mix_props` dùng để supervise là ground truth THẬT (vì sample là synthetic do mình sinh ra).
- Số liệu cũ `deepnoc_full = 82%` được tính trên split `alternating` bị leak. Số honest chỉ có sau khi chạy `cv` với split `grouped`.

## Roadmap thực tế

### Phase 1: pipeline ổn định + honest baseline

- `prepare` dữ liệu PROVEDIt
- chạy `baseline` (`MAC`, `RF`)
- chạy `cv --models mac rf deepnoc_full nocformer --folds 5` để có honest benchmark trên grouped split
- lưu `results/cv/summary_all.json` làm mốc so sánh

### Phase 2: pretrain + finetune NoCNet-v2

- `synth --n 30000`
- `train --model nocnet_v2 --epochs 100 --split grouped`
- `finetune --checkpoint results/best_nocnet_v2.pt --epochs 40`
- `cv --models nocnet_v2` để có CI của model mới

### Phase 3: tối ưu nếu cần thêm điểm

- tăng synthetic pool: `synth --n 100000`
- ensemble nhiều seed (train lại với `--seed`), trung bình probs
- per-class threshold tuning trên val để tối ưu macro-F1
- pseudo-labeling profile high-confidence để thêm dữ liệu real

## Ghi chú thực dụng

- Train trên CPU vẫn được nhưng synth+train sẽ rất chậm; nên thử với `--n 5000 --epochs 10` để dry-run trước.
- GPU CUDA: `torch` tự detect. NoCNet-v2 mặc định ~700 MiB ở batch 16, ~1.3 GiB ở batch 32.
- Cố định seed: `--seed 42` ở `train` / `finetune` / `cv` để reproduce.
- Synthetic pool sinh 1 lần dùng nhiều lần. Train xong cứ giữ `data/synthetic/` cho lần chạy sau.
- Nếu `cv` báo lỗi 1 model, các model còn lại vẫn chạy tiếp (lỗi được nuốt + log).
- Memmap synthetic pool: file `data/synthetic/X.npy` có thể 5-20 GB tuỳ `--n`; bảo đảm đủ ổ cứng.

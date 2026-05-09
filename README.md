# deepNoC

Repo này là bản triển khai thực nghiệm cho bài toán dự đoán số người đóng góp (`NoC`, Number of Contributors) từ hồ sơ STR DNA, lấy cảm hứng từ bài báo *deepNoC* của Taylor & Humphries (2024). Trạng thái hiện tại của codebase là một pipeline gọn để:

- chuẩn bị dữ liệu PROVEDIt từ GeneMapper CSV/XLSX,
- chạy baseline `MAC` và `Random Forest`,
- huấn luyện model `simple` hoặc `full`,
- đánh giá bằng confusion matrix, accuracy, precision, recall, F1 và threshold analysis.

README này mô tả đúng những gì repo đang chạy được bây giờ, không phải kế hoạch dài hạn ban đầu.

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
├── main.py
├── src/
│   ├── constants.py
│   ├── data_loader.py
│   └── evaluation.py
├── models/
│   ├── baseline/
│   │   └── baselines.py
│   └── deepnoc/
│       ├── architecture.py
│       ├── losses.py
│       └── train.py
├── data/
│   ├── provedit_raw/
│   └── provedit_processed/
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

## Cách chạy

### 1. Chuẩn bị dữ liệu

```bash
python main.py prepare --data-dir "data/provedit_processed/PROVEDIt_1-5-Person CSVs Filtered"
```

Lệnh này sẽ:

- đọc file CSV/XLSX phù hợp,
- parse sample name để suy ra `NoC`,
- build tensor `X`,
- lưu `X_gf25.npy`, `y_gf25.npy`, `sample_names.txt`.

### 2. Chạy baseline

```bash
python main.py baseline
```

Baseline hiện có:

- `MAC` rule-based
- `Random Forest` trên summary features trích từ tensor `[24, 50, 89]`

Kết quả và confusion matrix được lưu trong thư mục `results/`.

### 3. Train model

Model đơn giản:

```bash
python main.py train --model simple
```

Model đầy đủ:

```bash
python main.py train --model full
```

Có thể chỉnh các tham số chính:

```bash
python main.py train --model simple --epochs 500 --batch-size 64 --lr 1e-5
```

Lưu ý:

- `simple` là lựa chọn thực dụng hơn để kiểm tra pipeline
- `full` dùng kiến trúc nhiều head/output hơn, nhưng trong training loop hiện tại trọng tâm vẫn là `profile_noc`

### 4. Đánh giá checkpoint

```bash
python main.py evaluate --checkpoint results/best_model_simple.pt --model simple
```

Hoặc:

```bash
python main.py evaluate --checkpoint results/best_model_full.pt --model full
```

### 5. Chạy toàn bộ pipeline

```bash
python main.py all
```

Lệnh này sẽ:

- `prepare` nếu chưa có `.npy`
- chạy baseline
- train model

## Kết quả đầu ra

Trong `results/`, repo hiện sinh ra các file như:

- `confusion_matrix_*.png`
- `metrics_*.json`
- `threshold_*.png`
- `training_history_*.png`
- `best_model_*.pt`
- `checkpoint_*_ep*.pt`
- `history_*.json`

## Những gì đã đúng với code hiện tại

- Có thể prepare dữ liệu từ PROVEDIt CSV/XLSX
- Đã sửa parsing `NoC` cho sample name kiểu `-1;1-`, `-1;2;1-`, `-1;1;1;1-`
- Có split train/test xen kẽ bằng `train_test_split_alternating()`
- Có baseline `MAC` và `Random Forest`
- Có 2 chế độ train: `simple` và `full`
- Có module đánh giá và lưu hình/metrics

## Những gì chưa nên nói quá mức

Repo này chưa phải bản tái hiện hoàn chỉnh toàn bộ paper theo nghĩa chặt. Cụ thể:

- chưa có pipeline mô phỏng dữ liệu lớn như trong paper,
- chưa có toàn bộ label phụ được sinh đầy đủ từ ground truth thực nghiệm,
- nhánh `full` có nhiều output phụ trong kiến trúc, nhưng training hiện tại chủ yếu tối ưu `profile_noc`,
- chưa có bộ benchmark cố định được chốt lại trong README.

Nói ngắn gọn: repo đang là một bản triển khai thực dụng để chạy dữ liệu PROVEDIt và so sánh baseline với model học sâu cho bài toán `NoC`.

## Roadmap 3 tuần

### Tuần 1: Chốt pipeline dữ liệu và baseline

- xác nhận lại số profile sau `prepare` là ổn định,
- kiểm tra nhanh chất lượng tensor và phân bố `NoC`,
- chạy `MAC` và `Random Forest`,
- lưu lại metrics baseline làm mốc so sánh.

Deliverable:

- `X_gf25.npy`, `y_gf25.npy`, `sample_names.txt`
- confusion matrix và metrics cho baseline

### Tuần 2: Ổn định training cho model `simple`

- train `simple` với vài cấu hình `epochs`, `batch size`, `lr`,
- theo dõi overfitting qua `training_history`,
- chốt một checkpoint `simple` tốt nhất,
- so sánh trực tiếp với baseline trên cùng split.

Deliverable:

- `best_model_simple.pt`
- `history_simple.json`
- bảng so sánh `MAC` vs `RF` vs `simple`

### Tuần 3: Thử `full` và viết báo cáo ngắn

- chạy `full` để kiểm tra xem có cải thiện thực sự không,
- nếu `full` không ổn định hoặc không hơn `simple`, giữ `simple` làm kết quả chính,
- tổng hợp kết quả cuối: dữ liệu, split, baseline, model, confusion matrix, nhận xét lỗi thường gặp.

Deliverable:

- `best_model_full.pt` nếu có cải thiện
- bộ hình và metrics cuối cùng trong `results/`
- bản tóm tắt kết quả ngắn để dùng cho báo cáo hoặc thuyết trình

## Lệnh gợi ý cho 3 tuần này

```bash
python main.py prepare --data-dir "data/provedit_processed/PROVEDIt_1-5-Person CSVs Filtered"
python main.py baseline
python main.py train --model simple --epochs 500
python main.py train --model full --epochs 500
python main.py evaluate --checkpoint results/best_model_simple.pt --model simple
```

## Ghi chú thực dụng

- Nếu train trên CPU, nên giảm `epochs` trước để kiểm tra pipeline.
- Nếu có GPU CUDA, `torch` sẽ tự dùng GPU.
- Nếu cần lặp lại thí nghiệm nhiều lần, nên cố định thêm seed trong training và split.
- Nếu muốn README bám sát kết quả hơn nữa, bước tiếp theo hợp lý là chạy lại `baseline` và `train simple` rồi ghi con số thật vào đây.

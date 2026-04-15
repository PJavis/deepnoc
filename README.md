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

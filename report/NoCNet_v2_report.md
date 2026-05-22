# NoCNet-v2

## Bảng thuật ngữ (Forensic DNA + STR profiling)

Bảng dưới giải thích các thuật ngữ chuyên ngành xuất hiện trong report. Đọc trước phần này sẽ giúp hiểu kiến trúc và pipeline phía sau dễ hơn.

### Khái niệm sinh học cơ bản

| Thuật ngữ | Giải thích tiếng Việt |
| --------- | --------------------- |
| **DNA profile** | Bản "vân tay DNA" — tập hợp các vị trí lặp ngắn (STR) được PCR khuếch đại rồi điện di để định danh cá nhân. |
| **STR** (Short Tandem Repeat) | Đoạn DNA gồm một motif ngắn (2–6 base) lặp lại nhiều lần. Số lần lặp khác nhau giữa người với người → dùng để định danh. |
| **Locus** (số nhiều: *loci*) | "Vị trí gene" — một điểm cụ thể trên nhiễm sắc thể được phân tích. GlobalFiler có **24 locus** (D3S1358, vWA, …). Mỗi locus là một "cột" trong tensor `[24, 50, 89]`. |
| **Allele** | Biến thể tại một locus — chính là *số lần lặp motif STR*. Ví dụ locus TH01 có allele 6, 7, 8, 9, 9.3, 10… Mỗi người có 2 allele / locus (di truyền từ cha + mẹ). |
| **Genotype** | Cặp allele một người có tại một locus, ví dụ TH01: `(6, 9.3)`. |
| **Homozygous / Heterozygous** | Cùng allele 2 bản (`9,9`) / khác allele (`6,9.3`). Homozygous chỉ tạo 1 peak ở locus đó. |
| **Mixture profile** | Profile chứa DNA của ≥ 2 người (NoC ≥ 2). Càng nhiều người càng nhiều peak chồng lấn. |
| **NoC** (Number of Contributors) | Số người đóng góp DNA trong mixture. Đây là *biến cần dự đoán* của NoCNet-v2 (1..10). |

### Khái niệm tín hiệu / instrument

| Thuật ngữ | Giải thích tiếng Việt |
| --------- | --------------------- |
| **EPG** (electropherogram) | Biểu đồ tín hiệu thu được từ máy phân tích DNA. Trục X = fragment size (bp), trục Y = cường độ huỳnh quang (RFU). |
| **Peak** | Đỉnh nhọn trên EPG — tương ứng với một allele được khuếch đại. Mỗi peak có locus, allele, size (bp), height (RFU). Trong tensor mô hình, mỗi locus chứa tối đa **50 peak**. |
| **RFU** (Relative Fluorescence Unit) | Đơn vị cường độ huỳnh quang — chính là "chiều cao peak". `HEIGHT_FEATURE_IDX = 26` trong feature vector. |
| **Fragment size (bp)** | Kích thước đoạn DNA tính bằng base pair, đo bằng máy điện di mao quản. |
| **PCR** | Polymerase Chain Reaction — kỹ thuật khuếch đại DNA trước khi điện di. |
| **Capillary electrophoresis** | Điện di mao quản — phương pháp tách các fragment DNA theo kích thước, đọc tín hiệu bằng laser huỳnh quang. |
| **Injection time** | Thời gian "bơm" mẫu vào mao quản (ví dụ 25s). Dài hơn → tín hiệu mạnh hơn, nhưng dễ saturation. |
| **GlobalFiler** | Bộ kit PCR đa locus của Thermo Fisher, đánh dấu **24 locus** trên **5 kênh huỳnh quang (dye)**: Blue, Green, Yellow, Red, Purple. |
| **ABI 3500** | Mẫu máy phân tích DNA (genetic analyzer) phổ biến trong lab forensic. |
| **Dye channel** | Kênh màu huỳnh quang. GlobalFiler dùng 5 dye → mỗi locus nằm cố định trên 1 dye (xem `DYE_CHANNELS`). Mô hình dùng `dye_emb` để đưa thông tin này vào Transformer. |

### Artefact / nhiễu thường gặp

| Thuật ngữ | Giải thích tiếng Việt |
| --------- | --------------------- |
| **Stutter** | Peak "ma" sinh ra do PCR trượt (slippage). Xuất hiện cạnh peak thật, lệch ±1, ±2, ±0.2 repeat. NoCNet-v2 mã hoá stutter thành **bias attention theo Δ allele**. |
| **Back stutter (−1)** | Stutter lệch về phía trái 1 repeat — phổ biến nhất, ratio ~10%. |
| **Forward stutter (+1)** | Stutter lệch về phía phải 1 repeat — hiếm hơn, ~3%. |
| **Double back stutter (−2)** | Lệch −2 repeat, ratio ~1%. |
| **Point-2 stutter (±0.2)** | Stutter "incomplete repeat", lệch 0.2 repeat. |
| **Stutter ratio** | RFU(stutter) / RFU(parent allele). |
| **Pull-up** | Tín hiệu rò sang kênh dye khác (do filter spectrum không tách sạch). |
| **Baseline noise** | Nhiễu nền của tín hiệu huỳnh quang. |
| **Saturation** | Peak vượt ngưỡng đo của máy → bị "cụt đỉnh", mất thông tin chiều cao. |
| **Drop-out** | Allele thật không xuất hiện (RFU quá thấp). Hay xảy ra với low-template DNA. |
| **Drop-in** | Peak ngoại lai (ô nhiễm môi trường) xuất hiện ngẫu nhiên. |
| **Artefact** | Tổng quát: peak không phải allele thật (stutter, pull-up, drop-in, noise…). |
| **Analytical threshold (AT)** | Ngưỡng RFU cũ dùng để lọc peak nhỏ. NoCNet-v2 (và deepNoC gốc) **không dùng AT** — học trực tiếp từ tín hiệu, kể cả peak thấp. |

### Khái niệm modeling-specific

| Thuật ngữ | Giải thích tiếng Việt |
| --------- | --------------------- |
| **PLP** (Peak Label Probability) | Xác suất một peak là *non-artefact* (peak thật). Output của MHCNN pre-stage, là feature 29 trong vector 89-d. |
| **MAC** (Maximum Allele Count) | Baseline đơn giản: NoC ≈ ⌈max(số allele tại locus) / 2⌉. Không học, dùng so sánh. |
| **Mixture proportion (Mₓ)** | Tỷ lệ đóng góp DNA của từng contributor trong mixture (ví dụ 3-person `[0.6, 0.3, 0.1]`). Output `profile_mix_props` của model. |
| **Low-template DNA** | Lượng DNA đầu vào rất ít → tín hiệu yếu, drop-out nhiều, stochastic effect mạnh. |
| **Stochastic effect** | Biến động ngẫu nhiên trong PCR khi template thấp — peak heights mất cân bằng. |
| **PROVEDIt** | Bộ dataset DNA mixture công khai (Boston University) — chuẩn vàng để benchmark NoC. Branch hiện tại dùng GlobalFiler / ABI 3500 / 25s injection. |
| **simDNAmixtures** | Package R sinh mixture STR profile theo mô hình stutter + allele frequency + Mₓ. |
| **CORN** (Conditional Ordinal Regression for Neural networks) | Head ordinal: học K-1 binary "P(NoC > k)". NoCNet-v2 dùng làm 1 trong 3 head ensemble. |
| **CB weights** (Class-Balanced Cui et al. 2019) | Re-weight class theo `(1−β)/(1−β^n_c)`. NoCNet-v2 dùng β=0.999. |
| **Smooth-L1** | Hàm loss kết hợp L1 + L2 ở vùng quanh 0 — robust với outlier. Dùng cho scalar regression head. |
| **MixUp** | Augmentation: `x_mix = λx_i + (1−λ)x_j`, λ ~ Beta(α,α). Loss blend theo λ. |
| **SWA** (Stochastic Weight Averaging) | Trung bình trọng số mạng ở tail of training → minima rộng, generalise tốt hơn. |
| **TTA** (Test-Time Augmentation) | Augment input lúc inference, chạy nhiều forward pass, avg softmax. NoCNet-v2 dùng shuffle peak + jitter height. |
| **Bias calibration** | Cộng vector bias per-class vào log-prob để tối đa macro-F1 trên validation. Đẩy accuracy 0.876 → 0.943. |
| **Deep Sets pooling** | Pooling permutation-invariant: `ρ(Σ φ(x_i))` — phù hợp với set 50 peak không thứ tự. |
| **Macro-F1** | F1 trung bình *không trọng số* theo class — phản ánh fairness cross-class, không bị NoC=1 áp đảo. |

### Cấu trúc 89 feature / peak (tóm tắt)

| Index | Nhóm | Ý nghĩa |
| ----- | ---- | ------- |
| 1–24 | Identity | One-hot locus (24 GlobalFiler loci) |
| 25 | Allele | Allele designation (số lần lặp, normalised /100) |
| 26 | Size | Fragment size (bp, normalised) |
| 27 | Height | RFU peak height (normalised /33000) |
| 28 | Frequency | Allele frequency trong dân số |
| 29 | PLP | Peak label probability |
| 30–~70 | Stutter | Parent allele/height/ratio cho 4 loại stutter (back, double-back, forward, point-2) |
| ~70–76 | Parent-of-stutter | Nếu peak này là parent sinh stutter |
| 77 | Context | Tổng peak locus / 100 |
| 78 | Context | Tổng peak profile / 1000 |
| 79–88 | Context | Estimate mixture proportion (10-d) từ STRmix smart-start |

---

## Tổng quan

NoCNet-v2 là kiến trúc deep learning thế hệ tiếp theo trong repo `deepnoc`, kế thừa motivation của bài báo *deepNoC* (Taylor & Humphries) nhưng được thiết kế lại hoàn toàn để xử lý NoC (Number of Contributors) trên DNA STR profile theo hướng *set-based + ordinal-aware + ensemble-of-heads*.

Khác với deepNoC gốc (CNN 16 lớp trên tensor `[24 × 50 × 89]`) và NoCFormer (Transformer phẳng + CORN head), NoCNet-v2 chuyển ba góc nhìn:

* 50 peak trong một locus là **unordered set**, không phải sequence → dùng **Deep Sets pooling**, không dùng CLS token.
* Stutter là **quan hệ giữa các peak theo khoảng allele**, không phải feature handcraft → đưa vào dạng **additive attention bias** học được.
* NoC là **ordinal label** 1 → K → ba head đồng huấn luyện (softmax + scalar regression + CORN) rồi **ensemble probability**.

Toàn bộ pipeline data, encoding 89 feature/peak, 24 locus GlobalFiler, và shape input `[B, 24, 50, 89]` vẫn dùng chung với deepNoC.

---

# Pipeline tổng thể

```markdown
PROVEDIt CSV  +  simDNAmixtures (offline synth, memmap)
        ↓
Peak detection + feature extraction (89-d per peak)
        ↓
Tensor [B, 24, 50, 89]
        ↓
NoCNet-v2
        ↓
{cls_logits, corn_logits, scalar, mix_props, locus_n_alleles}
        ↓
Ensemble probs  →  Bias-tuning + TTA  →  NoC ∈ {1..5}
```

Pipeline khác deepNoC ở chỗ:

* **Hybrid dataset**: real PROVEDIt + offline synthetic (`src.synth`) trộn theo `p_synth`.
* **Memmap loader**: pool synth >5 GB load qua `mmap_mode='r'`.
* **Bias calibration + TTA**: bước hậu xử lý, push accuracy thật từ ~0.88 lên ~0.94.

---

# Input của NoCNet-v2

## Tensor shape

```markdown
[B, 24, 50, 89]
```

| Dim | Ý nghĩa |
| --- | ------- |
| 24  | 24 locus GlobalFiler (định nghĩa trong `src/constants.py: GLOBALFILER_LOCI`) |
| 50  | tối đa 50 peak / locus (`MAX_PEAKS_PER_LOCUS`) |
| 89  | feature vector mỗi peak (`NUM_FEATURES_PER_PEAK`) |

89 feature giữ nguyên schema deepNoC: one-hot locus 1–24, allele, fragment size, RFU, allele freq, PLP, stutter / parent-of-stutter, global context (`LOCUS_PEAKS_IDX=77`, `PROFILE_PEAKS_IDX=78`, `MIX_PROPS_SLICE=slice(79, 79+10)`).

Peak mask được lấy từ feature RFU:

```python
HEIGHT_FEATURE_IDX = 26
mask = x[..., HEIGHT_FEATURE_IDX] > 0
```

Padding `0.0` cho cả locus/peak rỗng.

---

# Kiến trúc NoCNet-v2

## Sơ đồ tổng quan

```markdown
Input [B, 24, 50, 89]
        ↓
PeakEmbedder (89 → d_model=96)
        ↓
peak_layers × LocusEncoderBlock
   = StutterBiasAttention + FFN  (pre-LN)
        ↓
DeepSetsPool([sum, max, log1p(count)])
        ↓
locus tokens [B, 24, 96]
        ↓
locus_layers × LocusTransformerBlock
   = MHA + FFN, +dye_emb +pos_emb
        ↓
ProfilePool([sum, mean, max])
        ↓
profile vector [B, 96]
        ↓
CountAwareHead
   ├── cls_logits   [B, K]
   ├── reg          [B]      (scalar NoC ∈ ℝ)
   ├── corn_logits  [B, K-1]
   └── mix_props    [B, 10]  (softmax)

aux: locus_n_alleles [B, 24, 20]
```

## 1. PeakEmbedder

MLP `89 → d_model=96` hai lớp GELU + LayerNorm + Dropout. Embed độc lập từng peak, không trộn cross-peak.

## 2. StutterBiasAttention (within-locus)

Multi-head self-attention trên 50 peak của một locus, với additive bias tính từ allele-distance:

```python
delta = alleles.unsqueeze(2) - alleles.unsqueeze(1)   # [N, P, P]
bias  = self.bias_mlp(delta.unsqueeze(-1))            # [N, P, P, H]
logits = q @ k.transpose(-1, -2) / sqrt(d_head) + bias
```

`bias_mlp`: `Linear(1, 16) → GELU → Linear(16, n_heads)` → mỗi head học một function riêng theo Δ allele.

Lý do: stutter peak nằm ở khoảng cách ±1, ±2, ±0.2 repeat so với parent. Bias này thay thế việc nhồi feature stutter handcraft vào input, đồng thời giữ tính chất **permutation-invariant** (mọi cặp được xử lý đối xứng theo Δ chứ không theo vị trí).

`key_padding_mask = ~mask`: peak padding không gửi/nhận signal; row toàn pad được `nan_to_num` về 0.

## 3. LocusEncoderBlock

Pre-LN block:

```markdown
x = x + attn(LN(x))
x = x + FFN(LN(x))
```

FFN multiplier 4×. Stack `peak_layers=2` block.

## 4. DeepSetsPool (locus aggregator)

Pooling permutation-invariant:

```python
h     = phi(peaks)                     # [N, P, D]
h_sum = (h * mask).sum(dim=1)
h_max = h.masked_fill(~mask, -1e4).max(dim=1).values
cnt   = log1p(mask.sum(-1)) / log(MAX_PEAKS_PER_LOCUS+1)
token = rho([h_sum, h_max, cnt])       # [N, D]
```

Khác với attention-CLS pooling (NoCFormer dùng), DeepSets giữ **cả `sum` và `count`** — hai thống kê này là tín hiệu trực tiếp cho "locus có bao nhiêu allele", chính là biến cần thiết để dự đoán NoC. Softmax-CLS bỏ count vì đã normalise.

## 5. CrossLocusTransformer (between-locus)

Self-attention trên 24 locus token:

* `pos_emb` learnable `[1, 24, d_model]`.
* `dye_emb` 5 dye GlobalFiler (`B/G/Y/R/P`) → embed tra cứu theo `DYE_INDEX` (build từ `DYE_CHANNELS`).
* Plain MHA + FFN, pre-LN, `locus_layers=2`.
* `key_padding_mask = ~locus_active` (locus rỗng bị mask).

Locus token đầu vào được zero-out cho `locus_active=False` trước khi vào stack để không "đầu độc" attention.

## 6. ProfilePool

Pool 24 locus → profile vector `[B, 96]`:

```python
agg = [sum_, mean_, max_]   # masked theo locus_active
profile = proj(agg)
```

## 7. CountAwareHead (3 view của NoC)

```python
shared  = GELU(Linear(96, 96)) + Dropout(0.2)
cls     = Linear(96, K)           # softmax classifier
reg     = Linear(96, 1)           # scalar regression
corn    = Linear(96, K-1)         # CORN ordinal cumulative
mix     = softmax(MLP(profile))   # [10] donor proportion
```

CORN → class probs:

```python
sig = sigmoid(corn_logits)
cum = cumprod(sig, dim=1)
p_gt = cat([1, cum])
p_lt = cat([cum, 0])
probs = (p_gt - p_lt).clamp(min=0)
```

Scalar → class probs (Gaussian smoothing quanh giá trị scalar, σ=0.5):

```python
z = (scalar - classes) / sigma
p = exp(-0.5 * z * z); p /= p.sum()
```

**Ensemble probability** (output cuối, dùng cho argmax inference):

```python
p_ens = (p_cls + p_corn + p_reg) / 3.0
```

## 8. Auxiliary head: locus_n_alleles

`Linear(96, 48) → GELU → Linear(48, 20)` — CE trên 0..19 allele per locus. Chỉ supervise trên sample synthetic.

---

# Multi-task loss

`models/nocnet_v2/losses.py: NoCNetV2Loss`

```markdown
total = w_cls  * CE(cls_logits, y)
      + w_reg  * SmoothL1(reg, y)
      + w_corn * CORN_BCE(corn_logits, y)
      + w_mix  * KL(true_mix || pred_mix)    [synth only]
      + w_nall * CE(locus_nall, true_nall)   [synth only]
```

Weight mặc định: `w_cls=1.0, w_reg=0.3, w_corn=0.5, w_mix=0.1, w_nall=0.1`. Label smoothing `0.1` trên CE.

Class-balanced reweight Cui et al. 2019, β=0.999:

```python
eff = 1 - β ** count
w   = (1 - β) / eff
w   = w * (K / w.sum())
```

Áp lên cả CE và CORN. Focal stacking *cố ý* tắt vì combine focal + CB từng phá NoCFormer (xem CHANGELOG).

Aux loss `mix_props` và `locus_n_alleles` masked theo `is_synth`: real PROVEDIt không có ground truth cho hai biến này.

---

# Training pipeline

`models/nocnet_v2/train.py`

## Hybrid dataset

* `RealProfileDataset` — PROVEDIt, augment online.
* `SynthProfileDataset` — pool synth offline, `np.load(..., mmap_mode='r')`.
* `MixedSampler` — sample synth với xác suất `p_synth=0.8`, real với class-balanced weight `1/count(class)`.
* `HybridLoader` — collate batch từ cả hai dataset.

`samples_per_epoch=4000`, `batch_size=16`. Aux loss chỉ fire trên row `is_synth=True`.

## Augmentation (`augment_profile`)

* **log-normal height jitter** σ=0.10: `h *= exp(N(0, σ))`, clip [0, 1].
* **random peak dropout** p=0.03: zero-out một số peak thật.
* **peak shuffle** trong locus — an toàn vì encoder permutation-invariant.
* **`_recompute_globals`**: refresh `LOCUS_PEAKS_IDX`/`PROFILE_PEAKS_IDX` sau khi heights thay đổi.

## MixUp

`mixup_alpha=0.2, mixup_prob=0.5`. λ ~ Beta(α, α), loss blend `λ L(out, y) + (1-λ) L(out, y[perm])`.

## SWA (Stochastic Weight Averaging)

`swa_frac=0.25` — bật ở 25% epoch cuối. `AveragedModel` eval song song với model live, pick best giữa `(live, swa)`.

## Optimizer + schedule

| Hyperparameter | Value |
| -------------- | ----- |
| Optimizer | AdamW |
| LR | 3e-4 |
| Weight decay | 5e-4 |
| Warmup | 3 epoch |
| Schedule | cosine sau warmup |
| Grad clip | 1.0 |
| Epochs | 80 (early stop patience=12) |
| Label smoothing | 0.1 |
| `d_model / n_heads` | 96 / 4 |
| `peak_layers / locus_layers` | 2 / 2 |
| Dropout | 0.15 (head 0.2) |

---

# Fine-tune trên PROVEDIt real

`finetune_nocnet_v2(...)`

Pretrain xong trên hybrid → fine-tune chỉ real:

| Param | Value |
| ----- | ----- |
| LR | 1e-5 |
| Weight decay | 1e-4 |
| Epochs | 30 (default) |
| Augment | jitter 0.05, dropout 0.01 |
| `p_synth` | 0.0 |
| `swa_frac` | 0.4 |
| `freeze_peak_stages` | optional, freeze `peak_embed` + `peak_blocks` |
| MixUp | tắt |

Trước khi train, eval baseline checkpoint pretrain và lưu nó như best ban đầu — tránh trường hợp FT làm tệ đi mà vẫn pick best của FT only.

---

# Inference: TTA + bias calibration

## predict_nocnet_v2_tta

Test-time augmentation:

* `n_samples` forward pass / profile.
* Mỗi pass: peak shuffle + log-normal jitter (σ=0.08), optional MC-Dropout.
* Avg softmax probability across samples → predictive entropy.

## Bias calibration (`tune_ft_step2.json`)

Tìm vector bias per-class `b ∈ ℝ^K` cộng vào log-prob để maximise macro-F1 trên validation:

```markdown
logp_calibrated = log(p_ens) + b
```

Kết quả tune trên split `grouped`:

| | Before | After |
| - | ------ | ----- |
| val macro-F1 | 0.593 | 0.697 |
| val accuracy | 0.876 | 0.944 |
| test macro-F1 | 0.645 | 0.697 |
| test accuracy | 0.905 | 0.942 |

Bias vector tìm được:

```markdown
b = [+0.591, -0.347, -0.678, -1.002, +1.435]
```

Đẩy mạnh prior cho NoC=1 và NoC=5 (hai class support lớn nhất + ít bị nhầm), ép xuống cho NoC=2..4 (overlap mạnh).

---

# Kết quả

## Dataset PROVEDIt (`split=grouped`, seed=42, test_size=0.25)

| NoC | Support |
| --- | ------- |
| 1 | 691 |
| 2 | 48 |
| 3 | 64 |
| 4 | 14 |
| 5 | 106 |
| **Tổng** | **923** |

## Accuracy progression

| Stage | Test acc |
| ----- | -------- |
| `best_nocnet_v2.pt` (hybrid pretrain) | **0.876** |
| `best_nocnet_v2_ft.pt` (fine-tune real) | **0.891** |
| FT + TTA (20×) | **0.891** |
| FT + TTA + bias calibration | **0.943** |

## Per-class final (TTA + bias tuned)

| NoC | Acc | Precision | Recall | F1 | Support |
| --- | --- | --------- | ------ | -- | ------- |
| 1 | 0.994 | 0.993 | 0.994 | 0.993 | 691 |
| 2 | 0.521 | 0.658 | 0.521 | 0.581 | 48 |
| 3 | 0.844 | 0.659 | 0.844 | 0.740 | 64 |
| 4 | 0.143 | 0.400 | 0.143 | 0.211 | 14 |
| 5 | 0.962 | 0.962 | 0.962 | 0.962 | 106 |
| **Overall** | **0.943** | — | — | — | **923** |

---

# So sánh với các mô hình trong repo

| Model | Test acc (grouped) |
| ----- | ------------------ |
| MAC baseline | 0.66 (CHANGELOG) |
| NoCFormer | 0.668 |
| NoCNet-v2 (hybrid pretrain) | 0.876 |
| NoCNet-v2 (FT real) | 0.891 |
| **NoCNet-v2 (FT + TTA + bias tuning)** | **0.943** |

---

# Điểm đặc biệt của NoCNet-v2

| Điểm | Ý nghĩa |
| ---- | ------- |
| Deep Sets pooling | giữ `sum` và `count` per locus — signal trực tiếp cho NoC |
| Stutter-bias attention | encode stutter qua Δ allele, không cần feature handcraft |
| Cross-locus Transformer 24-token | rẻ (24 token), thêm dye + pos embedding |
| Count-aware multi-head | CE + smooth-L1 + CORN cùng supervise, ensemble probability |
| Hybrid loader memmap | scale pool synth >5 GB không OOM |
| MixUp + SWA + cosine warmup | regularise + flat-minima |
| TTA shuffle/jitter | permutation-invariant → TTA free |
| Bias calibration | đẩy macro-F1 / accuracy lên ~5 điểm |

---

# Hạn chế

| Hạn chế | Ý nghĩa |
| ------- | ------- |
| NoC=4 support 14 sample | accuracy chỉ 0.14, không đủ data để tách khỏi NoC=3 và 5 |
| Bias tune trên validation | nguy cơ leak nhẹ qua split |
| Dataset PROVEDIt khoá GlobalFiler / ABI 3500 / 25s injection | chưa cross-kit |
| Multi-PCR replicate | hiện model đơn replicate / profile |
| Input fixed `[24, 50, 89]` | locus / peak vượt 50 bị truncate |
| Synth domain shift | TTA + FT giảm nhưng không loại hết |

---

# Cấu trúc file repo

```markdown
models/nocnet_v2/
  architecture.py   # NoCNetV2, StutterBiasAttention, DeepSetsPool, CountAwareHead
  losses.py         # NoCNetV2Loss, corn_loss, class_balanced_weights
  train.py          # train_nocnet_v2, finetune_nocnet_v2, predict_*_tta

src/
  constants.py      # GLOBALFILER_LOCI, DYE_CHANNELS, MAX_NOC=10, NUM_FEATURES_PER_PEAK=89
  synth.py          # offline synth pool generator (memmap)
  data_loader.py    # PROVEDIt → [N, 24, 50, 89] tensor
  threshold_tune.py # bias calibration loop
  ensemble.py       # multi-seed ensemble helpers

results/
  best_nocnet_v2.pt              # pretrain hybrid checkpoint
  best_nocnet_v2_ft.pt           # fine-tune real checkpoint
  metrics_nocnet-v2.json         # 0.876
  metrics_nocnet-v2_nocnet_v2_ft.json  # 0.891
  metrics_nocnet-v2_final_tta_tuned.json  # 0.943
  tune_ft_step2.json             # bias vector + per-stage acc
  history_nocnet_v2*.json        # training curves
  training_history_nocnet_v2*.png
  confusion_matrix_nocnet-v2*.png
```

# Bài nộp — Ước lượng số người đóng góp hỗn hợp DNA pháp y bằng NoCNet-v2 (Deep Sets)

**Nguyễn Quốc Dung — MSSV 20241586E**

Mô hình cuối cùng (tốt nhất): **NoCNet-v2** — Deep Sets phân cấp + chú ý lệch theo khoảng
allele (mã hóa stutter) + đầu đếm đa khung nhìn (softmax/hồi quy/CORN). Tập kiểm thử
grouped seed 42 (923 profile): **accuracy 0,927 · macro-F1 0,653** (con số *trung thực*,
đã loại rò rỉ dữ liệu).

## Bốn thư mục

| Thư mục | Nội dung |
|---|---|
| **Mô hình** | `nocnet_v2_ft.pt` (trọng số cuối — dùng tệp này) · `nocnet_v2_pretrain.pt` · `bias_tuned.json` · `README.md`. |
| **Mã nguồn** | `models/nocnet_v2/` (kiến trúc, mất mát, huấn luyện) · `src/` (hằng số, split) · `predict.py` · `data/` (nhãn + split) · `HUONG_DAN.md` (cài đặt + input/output từng bước). |
| **Kết quả** | `metrics_nocnet_v2_honest.json` + hình (ma trận nhầm lẫn, per-class, so sánh hệ, lịch sử huấn luyện) + `make_figures.py`. |
| **Báo cáo** | `BaoCao_NoCNetV2_VI.docx` (báo cáo Tiếng Việt theo mẫu) + bản `.md`, sơ đồ kiến trúc `.png`/`.drawio`, script dựng hình/báo cáo. |

Dữ liệu gốc: PROVEDIt (Alfonse và cộng sự, 2017), GlobalFiler / ABI-3500 / 25 giây.
Tensor đầu vào `X_gf25.npy` (~1,4 GB) không đính kèm do dung lượng — xem `Mã nguồn/HUONG_DAN.md`.

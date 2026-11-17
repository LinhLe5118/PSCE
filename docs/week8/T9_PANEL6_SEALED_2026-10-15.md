# T9 — Panel 6: dữ liệu và dự báo đã niêm phong (15–16/10/2026) — **dừng ở chốt H2**

Thẻ T9 của kế hoạch lên hội nghị (bộ agent kit, lưu ngoài repo).
- **Protocol** đóng băng trước khi có dữ liệu: [`artifacts/oeh_panel6_2026-10-15/protocol.json`](../../artifacts/oeh_panel6_2026-10-15/protocol.json), sha `49c27071…`. Bản nháp T8 được duyệt ở H1 ngày 15/10.
- **Runner:** `scripts/analysis/run_oeh_panel6_20261015.py`. Adapter: `scripts/panel6/export_kernels.py`, `scripts/analysis/panel6_families.py`. TabPFN: `scripts/analysis/panel6_tabpfn.py`.
- **Chưa có kết cục nào.** Không file `outcomes`/`labels` nào tồn tại. Bước nhãn cần Linh cho phép rõ ràng (H3).

## Trình tự đã chạy

1. **Dry run** trên mẫu phát triển, 12 MDP mỗi họ, numpy seed 1. Quy tắc cố định trước: dừng nếu một lớp chiếm quá 80% nhãn. Kết quả **PROCEED**: lớp lớn nhất (">x50") chiếm 51% với `mdpp` và 59% với `labmaze`. Không dự báo nào được chấm.
2. **Đóng băng.** Seed dữ liệu được dời từ 30M/31M (bản nháp) sang 70M/72M, vì 30M trở lên là vùng seed huấn luyện của các panel trước (seed instance + 20 000 000). Bản nháp đã yêu cầu kiểm tra này. Tập seed dự kiến (1 632 seed mỗi họ) không trùng với 24 378 seed cũ, cũng không trùng giữa hai họ.
3. **Amendment A1** (Linh duyệt, trước khi có dữ liệu): bước kiểm tra kernel cũ đếm số file (884) thay cho số chữ ký khác nhau (749; Colosseum 2029 và `v2test` là cùng một bộ kernel). Không ảnh hưởng tới ứng viên, quy tắc chọn, dữ liệu, dự báo hay phân tích. Hồ sơ có hash cũ và mới: [`amendments/A1_2026-10-15_guard_count.json`](../../artifacts/oeh_panel6_2026-10-15/amendments/A1_2026-10-15_guard_count.json).
4. **Chuẩn bị (`prepare`)** cho từng họ: xuất kernel từ thư viện gốc, kèm đối chiếu với `step()` của MDP Playground (p-value nhị thức Bonferroni > 1e-3). Sau đó chọn 96 MDP đầu tiên theo thứ tự danh sách, sinh dữ liệu, tính đặc trưng pilot, rồi niêm phong 15 phương pháp.
5. **TabPFN v2 trên B2**, phương pháp mô tả có điều kiện đã đăng ký: niêm phong riêng **sau** dự báo chính và **trước** bước nhãn (`tabpfn_seal.json`), đúng spec T3.

## Kết quả niêm phong

| | `mdpp` | `labmaze` |
|---|---:|---:|
| Ứng viên đã duyệt / được chọn | 97 / 96 (1 trùng chữ ký) | 96 / 96 |
| Số trạng thái (min / trung vị / max) | 8 / 30 / 64 | 33 / 44 / 61 |
| \|S\|\|A\| lớn nhất | 256 | 244 |
| Buffer | 5 280 | 5 280 |
| Dự báo niêm phong (15 phương pháp) | 21 600 | 21 600 |
| Fit so với cohort 7 (lệch trọng số tối đa) | 0 | 0 |
| Khoảng cách dịch chuyển d: trung vị; tỉ lệ d ≥ 3 | 4,0; 71% | 3,0; 50% |

Cả hai họ phần lớn nằm ngoài vùng meta-training (d ≥ 3 nghĩa là cổng bão hòa). Đây đúng là kiểu dịch chuyển panel 6 nhắm tới. Bảng trên chỉ mô tả đầu vào, không có kết cục.

## Ghi chú vận hành

- Dữ liệu panel nằm ở `datasets/oeh_panel6_2026-10-15/` (bị ignore). Kernel cũng bị ignore, vì tái tạo được từ danh sách ứng viên đã đăng ký; hash nằm trong `dataset_seal.json`.
- Để chạy nhanh hơn, `labmaze` được khởi động lại với 14 worker, thay cho 7. Lúc dừng, nó đã có dữ liệu và đặc trưng pilot, và mới đang dở 7 nhóm OEH-B; 7 file `.claim` đã được xoá. Code, seed và protocol giữ nguyên, nên kết quả không đổi.
  - Log lần đầu: `logs/week8/panel6_prepare_labmaze_run1_7workers.log` (đặc trưng pilot mất khoảng 32 phút).
  - `input_runtime_seconds.json` là của lần chạy thứ hai.
- Mọi tiến trình chạy ở ưu tiên BelowNormal để chừa CPU cho máy.

## Bước tiếp theo (cần Linh)

**H3:** cho phép bước nhãn của panel 6. Sau đó runner chạy learner, gán nhãn và chạy phân tích đã đăng ký, không sửa gì:
- cổng F1 rồi F2;
- ước lượng F3;
- phần mô tả, gồm TabPFN.

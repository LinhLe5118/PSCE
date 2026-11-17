# Chương 3 (Dataset): ghi chú cho người viết

## Các file

| File | Nội dung |
|---|---|
| `DAORL_Chapter3_Dataset_draft.docx` | Bản nháp chương 3 (21 trang, template của trường, 12 bảng, 10 hình, 4 công thức). Là sản phẩm build nên không commit; gửi riêng hoặc build lại |
| `03_dataset.md` | Nguồn có placeholder (`@@TABLE:…@@`, `@@NUM:…@@`); sửa chữ ở đây |
| `03_dataset_filled.md` | Bản đã điền số, đúng cú pháp `build_thesis_v2.py`; dùng để ghép vào thesis |
| `references_additions.md` | 5 tài liệu mới cần thêm vào `07_references.md` |
| `figures/`, `tables/`, `numbers.json`, `manifest.json` | Hình, bảng CSV và số liệu; manifest ghi SHA-256 mọi input và output |
| `scripts/build_ch3_assets.py` | Đọc artifact (chỉ đọc repo), sinh hình, bảng và số |
| `scripts/build_ch3_docx.py` | Điền placeholder, build DOCX bằng chính renderer và template của thesis |

**Build lại** từ gốc repo (sau khi sửa chữ hoặc khi có dữ liệu mới):

```
python docs/thesis/chapter3_dataset_2026-10-16/scripts/build_ch3_assets.py
python docs/thesis/chapter3_dataset_2026-10-16/scripts/build_ch3_docx.py
```

Bước thứ hai cần Windows có Word (file `MML2OMML.XSL`), cùng `python-docx` và `latex2mathml`, như `build_thesis_v2.py`.

**Ghép vào thesis:**
1. Chép `03_dataset_filled.md` thành `docs/thesis/.../03_dataset.md`, và chép `figures/fig3_*.png` vào `figures/`.
2. Thêm nội dung `references_additions.md` vào `07_references.md`.
3. Chạy `build_thesis_v2.py`.

Thư mục này chỉ thêm file mới; không file nào của thesis v2 hay của paper bị sửa.

## Đánh giá plan mục 3 (trước khi viết) và cách đã xử lý

**Điểm tốt:**
- Đủ các phần thầy yêu cầu: lý do tự sinh dữ liệu, đơn vị dữ liệu, nguồn, cách gán nhãn, registry, chống rò rỉ, đặc tính dữ liệu, giới hạn.
- Thứ tự hợp lý: định nghĩa → sinh dữ liệu → registry → phân tích.

**Vấn đề của plan và cách đã xử lý:**

| # | Vấn đề | Đã làm |
|---|---|---|
| 1 | 3.6.4 trùng với 2.7 (quy tắc gán nhãn) | Chương 3 chỉ *áp dụng* luật (Eq. 3.4), ghi rõ định nghĩa ở Chương 2. Cần thống nhất số công thức với 2.7 |
| 2 | 3.9.3 dùng khoảng cách d, vốn là thành phần phương pháp (4.7) | Giữ làm thống kê mô tả, dẫn chiếu sang 4.7.2 để định nghĩa |
| 3 | Plan thiếu khối lượng tính toán | Thêm cột learner jobs: tổng 1.835.280 job trên 2.088 MDP đã mở (gồm panel 6) |
| 4 | Plan thiếu bộ lọc dynamic-range (Eq. 3.2) | Thêm vào 3.3.5; áp cho mọi nguồn |
| 5 | FourRooms không có nhãn requirement nhưng plan để chung với các panel | Tách thành 3.5.6, không đưa vào registry |
| 6 | Plan không nói BRIDGE là ngoại lệ (γ = 0,99, sticky action) | Ghi rõ là ngoại lệ, chỉ là stress test cho các formulation cũ |
| 7 | Panel 6 (Test 3) đã mở nhãn ngày 16/10 | Đã thêm: mục 3.5.5, các dòng trong Bảng 3.6, 3.8, 3.10–3.12, Hình 3.4–3.10, cùng đoạn phân tích. Tên vai trò đổi thành Test 1/2/3 cho khớp paper |
| 8 | Lỗi B13: "10 transitions mỗi cặp state–action" | Eq. 3.1 ghi rõ "$10|S||A|$ transitions **tổng cộng**" |

## Phát hiện mới từ dữ liệu: nhóm trưởng nên biết

1. **FrozenLake (panel 5): cả 540 dòng đều là >x50.** Chỉ 3 dòng đạt ở x200. FrozenLake trong Colosseum cũng bị cắt 98,9%. Nghĩa là panel 5 gần như chỉ kiểm được khả năng *nhận ra ca bị cắt*, không kiểm được phần dự báo mức hữu hạn. Cần nói điều này ở 5.3 khi giải thích NO-GO.
2. **Ca bị cắt rất ổn định.** Trong 3.259 dòng >x50 của bốn panel cuối, 93,1% vẫn không đạt dù có x200. Đây là bằng chứng tốt cho cách định nghĩa lớp >x50.
3. **SPIBB-2 gần meta-training theo d (trung vị 1,88)** dù đổi cả kernel lẫn behavior. Điều này khớp với nhận xét cũ: dịch chuyển do behavior không hiện ra trong thống kê pilot.
4. Panel internal cuối gần như trùng meta-training ở mọi thống kê pilot. Panel forest nằm ngoài hẳn: 98,8% nhóm có d ≥ 3.

## Cần xác nhận với nhóm trưởng trước khi nộp

- [ ] **Tên cohort.** Thesis dùng "cohort 4, 7, 8", nhưng không có cohort 5 và 6. Bản nháp đặt tên E1, Validation-60 (1st/2nd), Cohort 1–4, Cohort 7, Cohort 8 (theo `run_oeh_final_120.py`). Cần chốt một bộ tên và dùng thống nhất ở Chương 3, Chương 5 và Phụ lục D.
- [ ] **Số liệu BRIDGE.** Bản nháp ghi "14 + 5 + 5 MDP" và "125/155 bị loại trong expanded selection". Cần đối chiếu với `BRIDGE_ALL_RUNS_RESULTS_2026-10-13.md`.
- [ ] **Số mục dẫn chiếu.** Các chỗ trỏ tới 4.5, 4.7.2, 4.7.4, 4.8, 4.12 và Phụ lục D lấy theo outline mới. Kiểm lại sau khi Chương 4 xong.
- [ ] **Licence** của Colosseum và code SPIBB: chưa kiểm. Mục 3.10 hiện chỉ ghi "under their own licences".
- [x] **Panel 6:** đã thêm (16/10). Hình 3.2 chưa vẽ MDP Playground và labmaze; chú thích của hình đã ghi rõ điều này.
- [ ] **Hình 3.2 là sơ đồ minh họa**, không vẽ từ kernel thật, trừ bản đồ FrozenLake (lấy đúng map đã đăng ký, `frozenlake_009`).
- [ ] Ví dụ ở Hình 3.3 được chọn sau khi mở nhãn. Chú thích đã ghi rõ điều này.

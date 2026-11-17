# Đánh giá B1 tuần 8 — 11/10/2026

## Kết luận

B1 của Việt Trần đã được bàn giao trong main qua PR #49, commit `22b4e9f6f513b86a475da259722fdc58d53b8b1f`. So với bản main `b4a6d08` ngày 08/10, chỉ có hai file mới: `docs/data/DATA_WEEK8.md` và `scripts/analysis/audit_week8_data.py`. Không có thay đổi thuật toán, dữ liệu thí nghiệm đã khóa hoặc kết quả hiệu quả.

Đánh giá: đủ làm đầu vào cho Quốc Việt tiếp tục phần thuật toán, đối chứng, công thức nhãn, chỉ số và viết Results/Limitations. Chưa đủ để coi mọi mục trong checklist H1 đã được người phụ trách ký nghiệm thu hoặc để tuyên bố tái lập toàn bộ thí nghiệm từ clone.

Bản clone: `C:/Users/Nguyen Viet/Downloads/Devoted-review-20261011-main`.

## Những phần kiểm lại đạt

- Chạy `python scripts/analysis/audit_week8_data.py --strict-available`: exit 0.
- 108.000 dòng kết quả, 108.000 khóa job duy nhất; không trùng job.
- Không sai phép kiểm `success == (normalized_return >= 0.95)`.
- Tái tính đủ 1.800 nhãn: không sai số seed thành công, không sai lớp nhu cầu dữ liệu theo luật full-suffix 8/10.
- 120 fingerprint kernel duy nhất. Việc không trùng cohort trước được báo theo biên bản pre-outcome có sẵn, không phải tái sinh kernel từ raw ở lần review này.
- Đối chiếu riêng CSV pilot: 30 cột đặc trưng; số môi trường theo họ 30 Garnet, 36 layered, 18 bottleneck, 36 lock. Các khoảng S×A khớp bảng B1.
- Clone sạch: 53/61 mục hash khớp, không mismatch. Sau dựng runtime đúng công cụ có sẵn: 57/61 khớp, không mismatch.
- CI của đúng commit mới nhất thành công: full suite/coverage/evidence, bảng bài báo, build LaTeX và giới hạn trang. CI hiện chưa gọi riêng script audit B1 mới; script này được chạy trực tiếp trong lần review.

CI: https://github.com/LinhLe5118/Devoted/actions/runs/38143225366
PR tham khảo: https://github.com/LinhLe5118/Devoted/pull/49

## Ý nghĩa dữ liệu dễ hiểu

Một hàng nhãn ứng với một môi trường, một cách thu thập dữ liệu và một thuật toán. Nhóm thử sáu mức dữ liệu, mỗi mức huấn luyện mười lần. Chỉ coi một mức là đủ nếu ít nhất tám lần đạt yêu cầu ở mức đó và ở tất cả mức lớn hơn đã thử.

Trong panel chính gồm môi trường ngẫu nhiên, 900/1.260 hàng (71,43%) nằm ở lớp >x50. Nghĩa là chưa đạt điều kiện thành công ổn định trong lưới ngân sách đã thử. Không có nghĩa thuật toán chắc chắn không thể học dù được cấp thêm dữ liệu.

Theo thuật toán, tỷ lệ >x50 là BC 90,71%, CFQI 45,95%, IQL 77,62%. CFQI có tỷ lệ chưa đạt thấp hơn trong thiết kế này; đây không phải kết luận nó luôn tốt hơn trong mọi môi trường/cấu hình. Tỷ lệ >x50 cũng khác với tỷ lệ thành công tại riêng x10.

## Các điểm cần chỉnh

### 1. Cảnh báo six/seven đã lỗi thời

`DATA_WEEK8.md:62` yêu cầu sửa paper §3 từ six sang seven. Tuy nhiên `paper/sections/03_difficulty.tex:47` hiện đã ghi seven; §5 và protocol cũng ghi bảy cohort. Cần đổi mục này thành đã đối chiếu khớp. Không cần sửa code/protocol lịch sử chỉ vì còn display string cũ.

### 2. Phân biệt thiếu đường dẫn runtime với mất mã nguồn

`DATA_WEEK8.md:48` gộp tám input thiếu và nói cần khôi phục source generator gốc. Bốn trong tám input là mã nguồn runtime có thể dựng từ các byte đã nằm trong Git bằng `python -m scripts.restore_frozen_runtime`.

Đã chạy lệnh: kiểm và tạo đủ 204 file; audit tăng từ 53 lên 57 mục khớp. Source generator và ba source khác đều khớp hash protocol. Do đó bộ sinh dữ liệu không bị mất khỏi Git; clone ban đầu chỉ chưa dựng vị trí runtime.

Sau phục hồi vẫn thiếu bốn artifact lịch sử, hai target dataset seal và raw buffer. Vì vậy giới hạn chưa tái lập end-to-end vẫn đúng. Không nên viết chỉ cần lấy đủ mười file là đã đủ tái chạy toàn bộ.

### 3. Guard của audit chưa kiểm đủ tính đầy đủ

`audit_week8_data.py:173–186` chặn hash mismatch, job trùng, threshold/count/label mismatch và fingerprint trùng. Nó chưa yêu cầu tổng 108.000 outcome, 1.800 nhãn, 120 kernel; chưa chặn `audit_says_all_new=false`, chưa kiểm đầy đủ khóa nhãn và đủ mười seed riêng cho mỗi nhóm/budget. Không có kiểm thử riêng cho script này trong CI hiện tại.

Đã kiểm tra guard bằng thay báo cáo trong bộ nhớ, không sửa CSV: cho các số lượng kỳ vọng về 0 và all_new=false vẫn exit 0. Đây là hạn chế của điều kiện nghiệm thu khi file đầu vào thay đổi; không chứng minh dữ liệu hiện tại sai. Hash hiện tại và các số lượng đã đọc trực tiếp vẫn khớp.

Nên bổ sung assertions lấy cấu hình từ protocol, kiểm tập khóa/seed/panel và đưa audit vào CI. Nếu dùng gate để xác nhận không trùng lịch sử, phải có bằng chứng giao tập fingerprint thay vì chỉ đọc lại boolean lịch sử.

### 4. Hồ sơ W8 chưa liên kết với B1 mới

`docs/week8/README.md` và checklist H1 ngày 08/10 vẫn nói chờ B1. Cần thêm cập nhật ngày 11/10 và link B1 mới, sau đó ghi kết quả review thực tế. Không tự đổi các dòng chờ thành PASS hoặc ký thay Hưng/Linh/Quốc Việt.

## Giới hạn benchmark và bước tiếp theo

B1 phân biệt đúng 120 MDP tự sinh với benchmark ngoài. Thông số FourRooms-random (10.174 episode, 1.000.070 step, observation ảnh 7×7×3, Minari 0.4.3) khớp tài liệu chính thức: https://minari.farama.org/datasets/D4RL/minigrid/fourrooms-random-v0/

Verdict NO-GO của phép thử thành phần lấy từ artifact của dự án. Chưa có kiểm chứng toàn OEH trên benchmark ngoài; RL Unplugged bsuite mới là ứng viên cần kiểm thêm, không phải benchmark đã chạy. Không có kết quả ablation mới.

Quốc Việt có thể làm ngay: review ba learner và công thức nhãn, đối chứng/metrics/selector, viết Results/Limitations, chuẩn bị danh sách đặc trưng và code chạy ablation. Nếu cần tái huấn luyện hoặc phân tích raw mới, phải có dữ liệu còn thiếu. Nếu chạy ablation để xác nhận khoa học, nhóm cần khóa thiết kế trước; cohort 7 đã mở chỉ cho phân tích khám phá, cần cohort chưa mở cho xác nhận độc lập.

Review này không sửa file versioned trong clone, không thay outcome hoặc đẩy Git. Runtime được dựng trong thư mục tmp bị Git ignore để kiểm checksum.

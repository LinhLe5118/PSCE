# Kế hoạch phát triển OEH v2

*Gửi nhóm trưởng. Viết ngày 14/10/2026, sau khi đọc `REANALYSIS_ALL_PANELS_2026-10-14.md`. Số liệu dẫn ở đây lấy từ tài liệu đó và từ gói Colosseum (`README.md`). Đây là đề xuất để nhóm trưởng quyết định; chưa phần nào được chạy.*

## 0. Mục tiêu

Biến OEH thành phương pháp **vượt trội cả trong lẫn ngoài generator**. Kiểm định bằng một lần chạy niêm phong trên panel mới, sau đó viết bài dựa trên OEH v2.

| Mốc | Thời gian dự kiến |
|---|---|
| Phát triển và chọn mô hình | 14/10 – 10/11 |
| Đóng băng OEH v2 và protocol kiểm định | ~10/11 |
| Chạy panel kiểm định mới | 10/11 – 30/11 |
| Chấm điểm, viết bài | 1/12 – 15/12 |

---

## 1. Điểm xuất phát: OEH current mạnh và yếu ở đâu

| | Bằng chứng (reanalysis) |
|---|---|
| ✅ **Trong generator rất mạnh** | 3 cohort niêm phong, gain 16,8% IBS và 10,1% LL, I² = 0; thắng cả B2 + legacy, LCE, CB |
| ✅ **Giữ hiệu chỉnh khi đổi kernel** | Reliability trên Colosseum: OEH 0,010 so với B2 0,023 |
| ⚠️ **Ngoài generator chỉ chắc về Brier** | Gộp 5 panel: IBS 16,0%, LL 4,9% (chứa 0) |
| ❌ **Đặt mức ngân sách kém B2 ngoài generator** | Hoán đổi: p₆ OEH + phần hữu hạn của B2 → LL 0,856, tốt hơn OEH (0,890) |
| ❌ **Mô phỏng hậu nghiệm chưa chứng minh giá trị ngoài miền** | B2 + legacy (không có posterior) ngang OEH ở mọi panel ngoài |
| ❌ **Behavior khác thì không có lợi thế** | SPIBB: gain IBS 7,8%, chứa 0; dịch chuyển do behavior không hiện ra trong thống kê pilot |
| ⚠️ **Head không thêm giá trị** | XGBoost-CF/108 ngang OEH về Brier |
| ⚠️ **Chi phí và quy mô** | OEH-B chiếm ~26/30 giây mỗi cặp; 125/155 task BRIDGE bị loại vì kích thước |
| ⚠️ **Còn xa trần nhiễu nhãn** | Mới lấp ~20% khoảng cách B2 → trần, còn nhiều chỗ cải thiện |

---

## 2. Tiêu chí thành công (khai báo trước khi phát triển)

OEH v2 thành công nếu, trên panel kiểm định mới niêm phong:

1. **Ngoài generator:** thắng chắc **B2 + legacy**, đối thủ khó nhất hiện nay, ở **cả integrated Brier lẫn log-loss**.
2. **Trong generator:** không kém OEH current quá biên định trước (đề xuất 2% IBS tương đối).
3. **Không tệ hơn có ý nghĩa** ở bất kỳ learner nào trên panel đủ lớn.

---

## 3. Quy trình phát triển: chống overfit vào dữ liệu đã mở

- **Dữ liệu phát triển** (đã mở): cohort 1–8, Colosseum 2027 và 2028, SPIBB, BRIDGE.
- **Chọn mô hình bằng leave-one-panel-out (LOPO):** fit trên mọi panel trừ một, chấm trên panel bị bỏ ra, lấy trung bình qua các panel ngoài generator. Cách này mô phỏng đúng tình huống gặp MDP lạ.
- **Panel ngoài đã mở chỉ dùng để chọn mô hình, không đưa vào meta-training** (đề xuất). Nhờ vậy panel kiểm định ngoài generator vẫn đo đúng khả năng chuyển giao.
- **Khóa trước** mọi lưới tham số và danh sách ứng viên, rồi mới chạy LOPO.
- **Mỗi cải tiến chỉ được giữ nếu cải thiện LOPO ngoài generator** mà không làm kém LOPO trong generator quá biên ở mục 2.

---

## 4. Các cải tiến phương pháp (theo thứ tự ưu tiên)

| Mã | Cải tiến | Ưu tiên | Chi phí | Máy |
|---|---|---|---|---|
| M1 | Head hai tầng | **Cao**, đã có tín hiệu | Thấp (refit) | Phụ |
| M4 | Hiệu chỉnh theo dịch chuyển | **Cao** | Thấp (refit) | Phụ |
| M6 | Đơn giản hóa head | Trung bình | Thấp (refit) | Phụ |
| M2 | Prior học từ dữ liệu | **Cao**, quyết định giá trị cốt lõi | Cao (chạy lại OEH-B) | Chính |
| M3 | Đa dạng hóa behavior | Cao | Cao (cohort mới) | Chính |
| M5 | Khấu hao và mở rộng quy mô | Cho hội nghị lớn | Cao | Cả hai |

### M1. Head hai tầng

**Bằng chứng:** trên Colosseum 2027, ghép p₆ của OEH với phân phối hữu hạn của B2 cho LL 0,856, tốt hơn OEH (0,890). Colosseum 2028 tương tự (0,876 so với 0,895). Trong generator thì phần hữu hạn của OEH lại tốt (67–75% lợi thế ở cohort 7/8), nên cần cơ chế biết **khi nào** tin phần nào.

**Thiết kế:**
- Tầng 1: P(E = 6) = p₆ từ head OEH current.
- Tầng 2: q(k) = P(E = k | E < 6), từ head fit **chỉ trên dòng hữu hạn**. Ứng viên field set: OEH; B2 + legacy; hợp OEH ∪ legacy.
- Co theo dịch chuyển: q'(k) = (1 − λ(d)) · q_OEH(k) + λ(d) · q_robust(k).
  - d: khoảng cách 10-NN chuẩn hóa tới meta-training (cách tính như `shift.csv`).
  - λ(d) = λ_max · min(1, d / d₀).
  - q_robust: phần hữu hạn từ B2 + legacy, hoặc phân phối mức của learner trên meta-training.
- Ghép: P(E = k) = (1 − p₆) · q'(k).
- Chọn bằng LOPO trong tập {OEH current, hybrid-legacy, hybrid-union, hybrid co theo d}.

**Khác lần thử cũ:** `oeh_difficulty_hurdle_development_2026-10-01` dùng cùng một bộ feature cho hai tầng và không được chọn. Lần này mỗi tầng dùng bộ feature mạnh nhất cho nó, cộng cơ chế co theo dịch chuyển.

### M4. Hiệu chỉnh theo dịch chuyển

**Bằng chứng:** OEH đã giữ hiệu chỉnh tốt hơn B2, nhưng log-loss ngoài generator vẫn chưa ý nghĩa.

**Thiết kế:** p'(k) = (1 − ε(d)) · p(k) + ε(d) · π_ℓ(k), với ε(d) tăng theo khoảng cách d như ở M1, và π_ℓ là phân phối lớp của learner ℓ trên meta-training. Chọn tham số bằng LOPO. Có thể gộp với M1 thành một tầng hiệu chỉnh.

**Lưu ý:** temperature scaling trên dữ liệu cùng phân phối từng fit ra T = 0,95 (`oeh_difficulty_temperature_development_2026-10-01`), tức làm sắc hơn. Vì vậy phải chọn bằng LOPO, không bằng OOF ngẫu nhiên.

### M6. Đơn giản hóa head

**Bằng chứng:** XGBoost-CF/108 ngang OEH về Brier, nên head ensemble không thêm giá trị.

**Thiết kế:** so head hiện tại với một head ordinal/survival đơn bằng LOPO. Nếu không kém, chọn head đơn: dễ giải thích, ít bị chất vấn, dễ kết hợp với M1 và M4.

### M2. Prior học từ dữ liệu

**Bằng chứng:**
- B2 + legacy (không có posterior) ngang OEH ngoài generator.
- Thất bại tập trung ở RiverSwim, nơi OEH gán khoảng 0,1% cho kết quả thật.
- Reanalysis đã nghi giả định reward `unseen_zero`.

**Bước 0, chẩn đoán (làm trước):** so xác suất thành công do OEH-B dự báo với thành công thực tế theo từng họ, đặc biệt RiverSwim, dưới từng prior có sẵn (`unseen_zero`, `exchangeable`; các cấu trúc A1–A4). Nếu prior không phải nguyên nhân thì **dừng M2** và ghi rõ nguyên nhân còn mở.

**Thiết kế (nếu bước 0 xác nhận):**
- **(a) Empirical Bayes:** fit siêu tham số prior (nồng độ transition, tham số Beta của tỷ lệ reward) bằng cực đại tổng marginal likelihood của pilot trên meta-training. Dirichlet–multinomial và Beta–Bernoulli có dạng đóng.
- **(b) Trung bình hóa Bayes qua cấu trúc:** trọng số w_j ∝ p(pilot | M_j); phân bổ 64 draw theo w_j; OEH-B lấy mẫu từ hỗn hợp. Cách này bỏ được sự phụ thuộc vào một giả định chọn tay.

**Chi phí:** chạy lại OEH-B cho meta-training, tương đương một cohort trên máy chính. Thử trên tập con (ví dụ 120 MDP) trước.

### M3. Đa dạng hóa behavior trong meta-training

**Bằng chứng:** SPIBB gần meta-training theo thống kê B2 (khoảng cách 1,8) nhưng là panel khó nhất. Dịch chuyển do behavior không hiện ra trong thống kê pilot chung.

**Thiết kế:**
- Thêm behavior vào generator: baseline kiểu SPIBB (softmax trên Q\*, nhiễu để đạt tỷ lệ hiệu năng định trước), ε-greedy quanh π\*, trộn ngẫu nhiên.
- Sinh **một cohort meta-training bổ sung** (khoảng 120 MDP × 7 behavior), không sinh lại toàn bộ.
- Feature mới: **độ phủ có trọng số theo chính sách tối ưu hậu nghiệm**, tức khối lượng d^{π̂\*}(s,a) rơi vào các cặp (s,a) có ít hơn k lần thăm. Tái dùng ước lượng C\* của baseline CB.
- Để đo đúng hiệu quả, LOPO phải giữ SPIBB **ngoài** meta-training.

### M5. Khấu hao và mở rộng quy mô

**Bằng chứng:** OEH-B chiếm ~87% thời gian; 125/155 task BRIDGE bị loại vì kích thước.

**Thiết kế:**
- **(a) OEH khấu hao:** huấn luyện mô hình g(thống kê pilot, learner) dự đoán trực tiếp đầu ra OEH-B (xác suất thành công theo mức, tỷ lệ bị cắt), dùng toàn bộ nhóm OEH-B đã tính qua các cohort và panel. Đo hai thứ: độ trung thực (R² so với OEH-B thật), và mức suy giảm dự báo trên LOPO.
- **(b) Posterior thưa:** Dirichlet chỉ trên support quan sát được, cộng một trạng thái "chưa biết" gộp lại. Chi phí giảm từ S²A xuống khoảng số phần tử khác 0.
- **Mục tiêu:** đưa vào được phần lớn task BRIDGE đang bị loại, mở đường cho hội nghị lớn. Không chặn tiến độ M1–M4; có thể làm song song hoặc sau.

---

## 5. Cải tiến cách đánh giá

| Mã | Việc | Ghi chú |
|---|---|---|
| **E1** | Thêm mức ngân sách x100, x200 trên panel kiểm định | Giữ endpoint chính 7 lớp để so được với meta-training. Mức mở rộng là **endpoint phụ**: dự báo ">x50" tương ứng "cần x100/x200" hay "không bao giờ". Chi phí learner tăng 2–3 lần trên panel kiểm định |
| **E2** | Regret khi ra quyết định | Chọn mức thu thập nhỏ nhất có P(E ≤ k) ≥ τ. Chi phí = dữ liệu thu cộng phạt λ khi thất bại; so với oracle. **Khóa τ ∈ {0,5; 0,8; 0,9} và λ trước.** Tính ngay trên panel đã có để làm mốc |
| **E3** | Ablation factorial 2³ khối (P = OEH-B, H = OEH-H/risk, L = legacy) và giá trị Shapley | Khai báo trong protocol kiểm định, chạy trên panel mới. Trả lời câu "mô phỏng hậu nghiệm có cần không" |

---

## 6. Kiểm định cuối (một lần, niêm phong)

### 6.1 Panel mới

| Panel | Cỡ | Vai trò |
|---|---|---|
| Cohort nội bộ mới | ≥ 120 MDP ngẫu nhiên | Tiêu chí 2 (trong generator) |
| Colosseum 2029 (seed mới, loại kernel trùng) | 72 MDP | Ngoài generator, kernel người khác |
| SPIBB mới | ≥ 72 MDP | Ngoài generator, behavior độc lập |
| Một họ benchmark chưa từng dùng | ≥ 36 MDP | Ngoài generator, chống "quen" Colosseum |

**Cỡ mẫu:** lấy độ lệch chuẩn gain theo MDP ngoài generator từ reanalysis. Với gain LL kỳ vọng 0,05–0,09, cần khoảng 150–200 MDP ngoài generator gộp lại. Ba panel ngoài ở trên cộng lại khoảng 180.

### 6.2 Kiểm định phân cấp (fixed-sequence)

Giữ α toàn họ mà không bị phạt đa phép so ở bước chính. Trên Colosseum, họ 18 phép so đồng thời từng đẩy critical lên 2,25.

1. **Bước 1:** OEH v2 vs B2 + legacy, gộp các panel ngoài generator (bootstrap theo MDP), cả hai chỉ số, max-t.
2. **Bước 2** (nếu bước 1 GO): OEH v2 vs OEH current, gồm không kém trong generator (biên 2%) và tốt hơn ngoài generator.
3. **Bước 3** (nếu bước 2 GO): OEH v2 vs B2, XGBoost-CF/18, LCE, CB.
4. **Mô tả, không gate:** theo panel và learner, phân rã Murphy, regret (E2), ablation (E3), endpoint mở rộng (E1), chi phí.

### 6.3 Niêm phong

Như cohort 7 và 8: protocol, danh sách kernel và hash code được khóa **trước khi sinh dữ liệu**; dự báo của mọi phương pháp và arm ablation được niêm phong **trước khi đọc nhãn**.

---

## 7. Lý thuyết

**Mệnh đề nhất quán:** giả sử
- pilot tăng sao cho mọi (s,a) liên quan được thăm vô hạn lần, và
- prior có support chứa MDP thật, và
- learner là hàm liên tục hầu khắp của dữ liệu.

Khi đó phân phối dự báo của OEH-B hội tụ về phân phối nhu cầu dữ liệu thật.

Mệnh đề này giải thích vì sao OEH-B yếu khi pilot thưa hoặc prior sai, khớp với bằng chứng ở M2, và cho bài một nền tảng lý thuyết ngắn gọn.

---

## 8. Lịch chi tiết

| Thời gian | Việc | Máy |
|---|---|---|
| 14–20/10 | E2 regret trên panel đã có (làm mốc). Dev M1, M4, M6 bằng LOPO. Bước 0 chẩn đoán của M2 | Phụ: E2, M1, M4, M6. Chính: chẩn đoán M2 |
| 21–31/10 | M2 trên tập con. Sinh cohort behavior cho M3. Chốt kết quả M1, M4, M6 | Chính: M2, M3. Phụ: hoàn thiện M1, M4, M6 |
| 1–10/11 | Chốt M2, M3. Ghép OEH v2. **Đóng băng OEH v2 và protocol kiểm định** | — |
| 10–30/11 | Sinh và chạy 4 panel kiểm định | Chính: cohort nội bộ, SPIBB. Phụ: Colosseum 2029, họ mới |
| 1–10/12 | Chấm điểm theo gate, ablation, regret | Cả hai |
| 10–15/12 | Viết bài và cập nhật luận văn | — |

**Lưu ý về mốc 26/12:**
- Theo lịch này, bài OEH v2 nộp khoảng giữa tháng 12. Kết quả phản biện (IEEE Access trung bình khoảng 4 tuần) **nhiều khả năng sau ngày bảo vệ**, nên lúc bảo vệ bài sẽ ở trạng thái đang phản biện.
- Cần hỏi khoa xem bài đang phản biện có được tính không. Cũng cần kiểm tra hạn nộp luận văn, có thể sớm hơn ngày bảo vệ.

---

## 9. Phân việc máy phụ làm được ngay

Dùng code của nhóm, không cần code ngoài:

1. **E2 regret** trên cohort 7/8, Colosseum 2027/2028, SPIBB (mô tả, làm mốc).
2. **M1, M4, M6:** chỉ refit head trên feature đã có, chạy LOPO.
3. **Colosseum 2029** khi protocol kiểm định đã khóa (~4 giờ; pipeline có sẵn).
4. Hỗ trợ **M5(a)** (mô hình khấu hao) vì chỉ cần các file OEH-B đã có.

---

## 10. Câu hỏi cần nhóm trưởng quyết định

1. Panel ngoài đã mở chỉ dùng để chọn mô hình, hay cả để fit? (Đề xuất: chỉ chọn.)
2. Làm cả M2, M3 (tốn máy chính) trong đợt này, hay chỉ M1, M4, M6 (rẻ) rồi kiểm định sớm hơn?
3. Có mở rộng lưới ngân sách (E1) không? (Tăng chi phí learner 2–3 lần trên panel kiểm định.)
4. M5 làm song song hay để sau?
5. Biên không kém trong generator: 2% IBS có hợp lý không?
6. Họ benchmark mới cho panel thứ 4 chọn từ đâu?

---

## 11. Cách đọc kết quả xấu (ghi trước)

| Nếu… | Thì… |
|---|---|
| M1/M4 không qua LOPO | Giữ head hiện tại; phép hoán đổi chỉ còn là phát hiện mô tả |
| Chẩn đoán M2 cho thấy prior không phải nguyên nhân thất bại | Dừng M2; ghi rõ nguyên nhân còn mở |
| M3 không cải thiện LOPO trên SPIBB | Dịch chuyển do behavior là giới hạn cấu trúc; nêu rõ trong bài |
| Bước 1 kiểm định NO-GO | OEH v2 không vượt B2 + legacy ngoài miền. Bài viết thành nghiên cứu về giới hạn của mô phỏng hậu nghiệm khi dịch chuyển; vẫn có giá trị nếu trung thực |
| Ablation cho thấy khối P (OEH-B) không đóng góp ngoài miền | Định vị lại đóng góp quanh chẩn đoán dựa trên mô hình nói chung; khuyến nghị biến thể không OEH-B (rẻ hơn ~87%) |
| OEH v2 kém OEH current trong generator quá 2% | Không thay OEH current; v2 chỉ là biến thể cho bối cảnh dịch chuyển |

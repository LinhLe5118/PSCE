# Đánh giá độc lập dự án Devoted (Difficulty-Aware Offline RL), 12/10/2026

Nguồn đọc: thư mục `Downloads\Devoted` trên máy của Việt (working tree hiện tại, không dựa vào Git HEAD). Mình đọc báo cáo tổng hợp `BAO_CAO_TONG_HOP_CHO_CLAUDE_2026-10-12.md`, tài liệu Week 8, protocol/seal/artifact của cohort 4 (`oeh_final_unseen_120_2026-10-02`) và cohort 7 (`oeh_confirmatory_unseen_120_2026-10-04`), runner nested và script audit B1.

## 0. Tóm tắt

- **Kết quả dự báo chính đứng vững.** Mình tự tính lại từ `predictions_preoutcome.csv` + `labels.csv` và ra đúng từng số trong báo cáo. Phương pháp OEH dùng ở cohort 7 giống hệt cohort 4, chỉ thêm cohort 4 vào meta-training. Ngay cả với luật cũ khắt khe hơn (bắt buộc giảm ≥5%), cohort 7 vẫn GO. Đây là điểm mạnh nhất, nên nói rõ trong paper.
- **Phần chọn learner yếu hơn cách đang trình bày.** Gate bị nới giữa cohort 4 và 7 (bỏ v5 khỏi gate), selector được thiết kế lại sau khi đã xem outcome cohort 4, và selector dùng nhãn họ MDP thật.
- **Quyết định giữ Current là đúng**, nhưng thí nghiệm nested ngày 12/10 không đủ sức trả lời câu hỏi "có cải thiện được không" (lý do ở mục 3).
- **Ba hướng tiếp theo:** (1) vá các lỗ hổng tính toàn vẹn của paper và chạy ablation; (2) kiểm định trên một họ MDP hoàn toàn mới; (3) một head dự báo dùng đủ 60 outcome/task thay vì 1 nhãn.

## 1. Những gì mình đã chạy, và chưa chạy

| Đã chạy (trong container, đọc file copy từ máy) | Kết quả |
|---|---|
| Tính lại IBS/log loss của 14 phương pháp trên 84 MDP chính, cohort 7, theo đúng quy tắc 15 task → MDP → trung bình | Khớp `summary.csv` tới 6 chữ số. OEH 0,045989/0,563079; B2 matched 0,054727/0,627179 |
| Tính lại cohort 4 | Khớp 0,052905/0,642729 (là "Current" của thí nghiệm nested) |
| Bootstrap cụm MDP riêng của mình (max-t studentized, 22 contrast, 10.000 lần) | Cả 22 cận dưới dương. Nhỏ nhất là Brier so với TQ/SACo, +0,00126 (artifact ghi +0,00101; khác do cách bootstrap, kết luận không đổi) |
| So `protocol.json` cohort 4 với cohort 7 theo từng trường | Xem mục 2 |
| Phân rã lỗi theo learner, theo họ MDP, censored với hữu hạn | Xem mục 4 |

**Chưa chạy:** pytest, `export_week8_d1 --check`, `verify_oeh` (cần môi trường `.venv-restore` trên máy), refit bất kỳ model nào, tái tạo từ raw buffers. Script mình dùng: [recompute_scores.py](recompute_scores.py).

## 2. Lỗi và rủi ro, phân loại theo mức chắc chắn

### Đã xác minh

1. **Gate chọn learner bị nới sau khi thấy outcome cohort 4.** Ở cohort 4, `selection_study.decision_rule` yêu cầu thắng *cả* best fixed learner *và* OEH-v5. Ở cohort 7, gate chỉ còn so với fixed CFQI, còn v5 chuyển thành mô tả bổ sung. Quy tắc selector cũng đổi từ argmax posterior-predictive sang ExtraTrees phát triển trên dữ liệu đã mở. Nếu chấm theo luật của cohort 4 thì cohort 7 vẫn **NO-GO** (so v5: return −0,0018, cận dưới âm). Paper phải viết "thắng fixed CFQI, không thắng v5; gate đã đổi giữa hai cohort", không chỉ ghi "selection GO".
2. **Protocol niêm phong chứa câu nói quá.** `protocol.json → tiers.verdicts[0]` viết tier 1 là "superiority over the published state of the art with the same information". B2 do nhóm xây, ERI/BWD là adaptation, nên câu này không được xuất hiện trong paper. Báo cáo của bạn đã nói đúng là không có SOTA; chỉ cần đảm bảo không ai trích câu này từ protocol.
3. **Có hai bộ cận dưới selection trong cùng một thư mục.** `confirmatory_verdict.json` ghi success +0,0426, return +0,0103 (crit 1,946, 2 contrast). `selection_contrasts.csv` ghi +0,0394 và +0,0092 (crit 2,203, 4 contrast có cả v5). Cả hai đều dương; paper cần chọn một bộ và ghi rõ họ contrast của nó.
4. **Guard của audit B1 có lỗ hổng** (`scripts/analysis/audit_week8_data.py:176-185`). Chế độ `--strict-available` không fail khi:
   - `outcome_rows ≠ 108.000` hoặc `label_rows ≠ 1.800`;
   - một (task, budget) thiếu seed, vì đếm success trên số seed có sẵn;
   - một dòng label không có outcome nào. `grouped` là `defaultdict` (dòng 112, 125) nên số success thành 0, event kỳ vọng là 6, và một nhãn `>x50` sai sẽ lọt qua;
   - `audit_says_all_new = false`.
   Dữ liệu hiện tại khớp khi kiểm riêng, nên đây là rủi ro của guard, không phải dữ liệu sai. Việc sửa đơn giản: thêm bốn điều kiện đó vào `failed`.
5. **Tên cohort dễ gây nhầm.** "Cohort 4" là `fourth120` (lần kiểm định 120-MDP thứ tư); "cohort 7" là lần kiểm định thứ bảy tính từ `first60`. Paper nên dùng một cách đặt tên và có một bảng liệt kê tất cả các lần.

### Rủi ro diễn giải (suy luận, chưa phải lỗi)

6. **Kiểm định tuần tự qua nhiều cohort.** Theo `docs/reassessment/TONG_HOP_KET_QUA_2026-10-03.md`, câu hỏi dự báo đã được thử ít nhất 5 lần trên cohort 120-MDP mới: lần 1–3 NO-GO, lần 4 Tier 1 GO / Tier 2 NO-GO, lần 7 GO. Mỗi gate chỉ kiểm soát sai số trong cohort của nó. Điều bảo vệ kết quả là **lần 4 và lần 7 dùng cùng cấu hình OEH và cả hai đều Tier 1 GO**, tức là đã lặp lại được. Paper nên trình bày toàn bộ chuỗi (kể cả NO-GO) và lấy sự lặp lại này làm luận điểm chính, thay vì chỉ báo cohort 7.
7. **Phạm vi tổng quát hóa hẹp.** 84 MDP thuộc 3 họ generator mà meta-training đã thấy. Bootstrap theo MDP không tính phụ thuộc theo họ; với chỉ 3 họ thì không thể tổng quát sang họ mới. Ở bottleneck, OEH kém B2 (IBS 0,02114 so với 0,02047).

## 3. Thí nghiệm nested 12/10: kết luận đúng, nhưng thiết kế không đủ sức trả lời

- **"Current" là model của cohort 4**, tức meta-training chưa có `fourth120` (`protocol.json` cohort 4 → `oeh_final.meta_training_cohorts`). Calibration học trên dự báo này không áp thẳng được lên model đang khóa của cohort 7.
- **Joint 7 và Two-stage học lại từ khoảng 336 dòng mỗi learner mỗi fold**, trong khi Current được meta-train trên 7 cohort. Với BC, khoảng 90% là censored, nên head điều kiện của Two-stage chỉ thấy vài chục ví dụ hữu hạn. Thua Current là điều dễ đoán và không nói lên nhiều về ý tưởng hai bước. Báo cáo đã rào đúng; chỉ cần không viết "hai bước không hiệu quả" trong paper.
- **GroupKFold trong một cohort không mô phỏng đúng cách dùng thật** (gặp một cohort mới). Khung phát triển hợp lý hơn là leave-one-cohort-out trên 7 cohort đã mở.
- Gain dưới 1% với cận âm, và đảo chiều khi giữ một họ ra ngoài, cho thấy calibration không phải điểm nghẽn. **Giữ Current là hợp lý.**

## 4. Lỗi còn lại tập trung ở đâu (số đã tính lại, cohort 7, OEH)

| Phân nhóm | Kết quả |
|---|---|
| Theo learner | CFQI chiếm **56% tổng IBS và 64% tổng log loss**. Log loss từng learner: CFQI 1,077; IQL 0,435; BC 0,177 |
| Nhãn | 900/1.260 là `>x50`; BC có 381/420 là `>x50`; CFQI có 227 nhãn hữu hạn trải đều x1–x50 |
| Độ nhiễu nhãn | **18%** số dòng label có số seed thành công *giảm* khi budget tăng. Nhãn là ngưỡng 8/10 trên 10 seed, nên một phần lỗi là không thể giảm |
| Theo họ | garnet 0,0380 so với B2 0,0543; layered 0,0651 so với 0,0722; bottleneck 0,0211 so với 0,0205 (OEH kém hơn) |

## 5. Trả lời 7 câu hỏi trong mục 11 của báo cáo

1. **D1 đáp ứng gì, còn thiếu gì?** Bảng learner/đối chứng/quyền truy cập, công thức metric và tái tính score đều đủ và đúng (mình tính lại khớp). Còn thiếu: chưa có cost fit riêng và VRAM của 4 learner bổ sung; bảng LaTeX chưa được build trong manuscript; chưa có review của Hưng/Linh. Chưa thấy lỗi nào trong D1.
2. **Số học, metric, aggregation, split:** đúng, đã xác minh ở cohort 4 và 7. Split nested không trùng MDP (`assert_split`, dòng 58–62).
3. **Leakage, tái dùng test, selection bias, nói quá:** không thấy leakage trong forecast cohort 7. Có selection bias ở phần selector (mục 2.1), có tái dùng cohort 4 nhiều lần (đã được ghi nhận), và có câu SOTA trong protocol (mục 2.2).
4. **Đối chứng có công bằng không:** các hàng matched dùng cùng head, nhưng OEH có 108 cột so với 18 của B2. Lợi ích có thể một phần đến từ số chiều hoặc dung lượng, nên cần ablation. SPIBB nhận true β, selector nhận nhãn họ: đã được ghi rõ, giữ như vậy.
5. **Giữ Current:** hợp lý (mục 3).
6. **Claim được và claim phải bỏ:**
   - *Được:* trên 84 MDP stochastic mới thuộc 3 họ đã biết, kiểm định một lần đã đăng ký trước, OEH giảm IBS 16% và log loss 10% so với B2 matched, thắng cả 11 đối chứng đã đăng ký kể cả TQ/SACo đặc quyền; cùng phương pháp cũng Tier 1 GO ở cohort 4. Selector thắng fixed CFQI ở cả hai endpoint.
   - *Phải bỏ:* SOTA; "difficulty" là thuộc tính nội tại của MDP; tổng quát sang họ hoặc benchmark mới; tiết kiệm dữ liệu (pilot đã tốn x10) và tiết kiệm compute; tái lập end-to-end; gán lợi ích cho một nhóm feature; selector thắng v5; "hai bước thất bại".
7. **Ba hướng:** mục 6.

## 6. Ba hướng tiếp theo, theo thứ tự ưu tiên

### Hướng 1. Vá tính toàn vẹn của paper và chạy ablation (ưu tiên cao, chi phí thấp)
- Thêm bảng "lịch sử kiểm định" gồm mọi lần thử (1–4, 7), cấu hình, gate và verdict. Nhấn vào việc lần 4 và lần 7 lặp lại cùng kết quả Tier 1.
- Viết lại phần selector theo mục 2.1. Chọn một bộ cận. Sửa guard B1 (mục 2.4) kèm test âm cho từng điều kiện mới.
- Chạy ablation bỏ từng khối trong 108 cột theo đề xuất trong `docs/week8/H1_ABLATION_REVIEW_2026-10-08.md`, dùng **leave-one-cohort-out trên 7 cohort đã mở** làm khung đánh giá. Cần kiểm tra trước là input meta-training của 7 cohort có đủ trong thư mục không; nếu thiếu thì đây là việc khôi phục dữ liệu phải làm trước.
- Chi phí: chủ yếu là CPU cho XGBoost (runner nested 84 MDP mất khoảng 41 giây, nên đoán mỗi cấu hình LOCO tốn vài phút; đây là suy luận, chưa đo).

### Hướng 2. Kiểm định trên họ MDP hoàn toàn mới (ưu tiên cao, chi phí trung bình)
- Đây là khoảng trống khoa học lớn nhất. FourRooms chỉ kiểm một thành phần và còn partial observation.
- Bước rẻ trước: chạy lại toàn bộ pipeline OEH với meta-training **bỏ hẳn một họ** (không chỉ phần calibration như stress test 12/10), rồi chấm trên họ đó.
- Bước chính: chọn 1–2 họ MDP tabular stochastic công khai có kernel biết trước, ví dụ FrozenLake slippery, RiverSwim hoặc gridworld của paper SPIBB. Giữ nguyên quy trình chấm chính xác, khóa protocol và dự báo trước outcome. Cost core x10 hiện là khoảng 0,1 giây cho bộ BC+CFQI+IQL mỗi job (`core_x10_cost.csv`); budget x20/x50 sẽ tốn hơn, nên cần đo một shard nhỏ trước khi ước tính tổng.

### Hướng 3. Head dự báo có cấu trúc, dùng đủ 60 outcome mỗi task (ưu tiên trung bình, nghiên cứu)
- Hiện mỗi task chỉ đóng góp 1 nhãn 7 lớp, dù đã có số seed thành công ở cả 6 budget (`labels.csv: successes_x1..x50`).
- Đề xuất: dự báo xác suất thành công từng budget `p_b` (binomial, 10 seed), rồi suy ra **chính xác** phân phối 7 lớp qua luật suffix 8/10. Cách này tận dụng nhiều tín hiệu hơn hẳn, đúng chỗ CFQI hữu hạn đang chiếm phần lớn lỗi, và mô hình hóa trực tiếp độ nhiễu nhãn (18% không đơn điệu).
- Trong repo đã có `scripts/analysis/develop_discrete_hazard.py`. Cần đọc kết quả cũ của nó trước để không lặp lại hướng đã rút.
- Phát triển bằng leave-one-cohort-out, khóa, rồi kiểm trên cohort chưa mở (có thể chung đợt với Hướng 2). Không hứa trước mức tăng.

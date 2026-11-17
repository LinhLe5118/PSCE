# Kế hoạch viết paper + thesis — cập nhật 15/10/2026 (sau kiểm định niêm phong OEH v2)

Bản này thay cho [kế hoạch 14/10](KE_HOACH_VIET_PAPER_THESIS_2026-10-14.md). Bản 14/10 được giữ làm lịch sử.

**Lý do cập nhật:** OEH v2 đã qua kiểm định niêm phong trên bốn panel mới, cả ba bước đều GO ([biên bản OEH v2](OEH_V2_DEVELOPMENT_2026-10-14.md), mục 4). Paper, supplement và thesis đã được sửa theo kết quả này.

## 0. Cập nhật sau panel thứ năm (15/10, chiều) — đọc phần này trước

**Khung viết đã đổi** (đã áp dụng vào `paper/` và thesis `DAORL_Thesis_final_2026-10-15`):
- Phương pháp cuối gọi là **OEH**. Bản v1 gọi là **OEH-F** (head đầy đủ, chưa có cổng).
- "B2 + legacy" đổi thành **head vững (B2 + D)**; "legacy diagnostics" đổi thành **chẩn đoán pilot D**.
- Các kiểm định trước là **quá trình phát triển** và được khai báo đầy đủ.
- Bảng ánh xạ tên code ↔ tên bài nằm trong [biên bản OEH](OEH_V2_DEVELOPMENT_2026-10-14.md), mục 6.

**Việc đã làm trong lượt này:**

| Việc | Kết quả |
|---|---|
| Panel thứ năm, khóa trước: FrozenLake (Gymnasium) 36 + gambler 36; một phép so OEH vs head vững | **NO-GO**. FrozenLake: OEH tốt hơn rõ, nhưng LCE tốt nhất (100% nhãn kiểm duyệt). Gambler: head vững tốt hơn (không có ý nghĩa). Tái lập: OEH > head đầy đủ, OEH > B2 về LL |
| Điểm vào duy nhất `scripts/analysis/oeh_pipeline.py` | Tái tạo dự báo niêm phong cả 4 panel (≤ 9,4e-16); test `tests/test_oeh_pipeline.py` |
| Hình gain theo d (`paper/figures/fig_gain_by_distance`) | Gain tăng theo d; ở d > 3 cổng tốt hơn cả hai head |
| Ablation phương pháp cuối (bỏ OEH-H, bỏ OEH-B) | Bỏ khối nào cũng kém hơn ngoài generator. Câu hỏi "có bỏ được horizon không" đã có câu trả lời: không |
| Baseline chung mạnh (k-NN analog, logistic trên B2) | Kém B2 matched, kém OEH xa |
| Viết lại paper (8 trang; abstract 251 từ) và thesis (58 trang) quanh phương pháp cuối | Xong; bằng chứng phát triển chuyển sang 5.4–5.10 (thesis) và supplement (paper) |

**Claim sau panel thứ năm** (thay cho các câu tương ứng ở mục 3):

| Không viết | Viết |
|---|---|
| "OEH beats the strongest pilot-only control outside the generator" | "OEH beats generic statistics and its own full head on the pooled external panels of both final tests; its advantage over the strongest pilot-only control is panel-dependent (GO on the pooled four-panel test, carried by forest management; NO-GO on a fifth panel from two new sources)." |
| "OEH is the best forecaster on every external benchmark" | "Where every requirement is censored (FrozenLake), the learner's own pilot learning curve is the best forecaster." |
| "The shift gate detects distribution shift" | "The gate's distance uses pilot statistics only and does not detect a change of logging behavior (SPIBB)." |

**Hệ quả với venue:**
- Kết quả NO-GO làm claim "vượt bộ mô tả mạnh nhất" không còn đứng được. Đóng góp chính chuyển thành ba điểm:
  - target và benchmark;
  - phương pháp có đảm bảo theo từng dự báo, cùng phân tích cơ chế;
  - một chương trình đánh giá niêm phong trung thực, cả kết quả dương lẫn âm.
- **TMLR** (chấp nhận kết quả âm, coi trọng độ chặt) và **RLC** là phù hợp nhất. NeurIPS/ICML main track vẫn cần quy mô (posterior thưa, BRIDGE lớn).
- Hướng phương pháp có lợi nhất tiếp theo: đưa learning curve của pilot (LCE) vào cổng như head thứ ba. Phát triển trên panel đã mở, kiểm định một lần.

## 1. Trạng thái bằng chứng

| Panel | Vai trò | Kết quả | Nguồn |
|---|---|---|---|
| Cohort 7 (84 MDP) | Xác nhận chính của OEH | 22/22 cận dưới dương; IBS −15,97%, LL −10,22% vs B2 | `artifacts/oeh_confirmatory_unseen_120_2026-10-04` |
| Cohort 8 (168 MDP) | Tái lập trong generator | Bước 1–2 GO | `artifacts/oeh_eighth_cohort_2026-10-14/internal` |
| Colosseum 2027, 2028; SPIBB | Panel ngoài của OEH | GO vs B2 trên Colosseum; NO-GO vs bộ mô tả mạnh; SPIBB NO-GO | biên bản cohort 8 |
| **OEH v2, 4 panel mới** (internal 168, Colosseum 2029 72, SPIBB-2 108, forest 48) | **Kiểm định niêm phong của v2** | **Bước 1, 2, 3 đều GO**: không kém OEH trong generator; tốt hơn OEH, B2 + legacy và mọi đối thủ pilot-only trên 228 MDP ngoài gộp, cả IBS và LL | `artifacts/oeh_v2_sealed_test_2026-10-14` |
| Độ vững của v2 (mô tả) | Phạm vi claim | Lợi thế so với OEH vững khi bỏ bất kỳ panel ngoài nào. Lợi thế so với B2 + legacy chủ yếu nhờ forest; Colosseum dương nhưng chưa có ý nghĩa riêng; SPIBB-2 ngang | `robustness.csv` |
| BRIDGE | Phụ thuộc learner | Không đổi | `BRIDGE_ALL_RUNS_RESULTS_2026-10-13.md` |

## 2. Định vị đóng góp

1. **Target và benchmark:** nhu cầu dữ liệu thực nghiệm theo learner, có kiểm duyệt phải; benchmark có kiểm soát, nhãn exact.
2. **OEH:** tốt nhất trong generator, đã tái lập ba lần. Ngoài generator, OEH là bộ phát hiện "không đủ dữ liệu" vững nhưng đặt mức ngân sách hữu hạn kém.
3. **OEH v2:**
   - Biến phát hiện trên thành một luật pilot-only có đảm bảo theo từng dự báo (Mệnh đề 2).
   - Chọn trên panel đã mở, kiểm định **một lần** trên bốn panel mới.
   - Đây là phương pháp chính khi pilot nằm xa dữ liệu huấn luyện.
4. **Quy trình đánh giá:** protocol commit và push trước dữ liệu, seal từng panel trước learner, gatekeeping tuần tự; báo cáo đầy đủ các kết quả âm (M2, M3, SPIBB, RiverSwim, BRIDGE Atari).

## 3. Câu được viết và câu không được viết (bổ sung cho bản 14/10)

| Không viết | Viết |
|---|---|
| "OEH v2 generalizes to any external benchmark" | "On 228 pooled external MDPs from three benchmark sources, OEH v2 beats OEH and every pilot-only comparator in both scores under a sealed fixed-sequence gate." |
| "OEH v2 beats the strongest control on every external benchmark" | "The margin over B2 + legacy is carried by the forest-management panel; on Colosseum it is positive but not individually significant, and on SPIBB the methods tie." |
| "OEH v2 is always at least as good as OEH" | "OEH v2's loss on every forecast is at most the distance-weighted average of its components' losses (Proposition 2); empirically it is non-inferior inside the generator." |
| "Behavior shift is solved" | "With the SPIBB benchmark's own logging behavior no method wins; neither a reward prior nor new behavior families in meta-training changed this." |
| "Posterior simulation is useless outside the generator" | "Its external contribution depends on the MDP structure: positive on forest management, not on Colosseum." |

## 4. Venue: nhắm hội nghị cao như thế nào

**Hiện trạng:** bản IEEE conference 8 trang cộng supplement 4 trang đã cập nhật và dựng sạch. Bản này dùng được cho đồ án và cho một venue IEEE.

**Vì sao giờ có thể nhắm cao hơn:** so với 14/10, bài đã có ba điểm mà reviewer hội nghị lớn thường hỏi:
- một phương pháp có lập luận rõ (cơ chế đo được → luật kết hợp), kèm đảm bảo không phụ thuộc phân phối;
- một kiểm định niêm phong duy nhất trên các panel do nhóm khác thiết kế;
- ablation factorial và độ vững được báo cáo đầy đủ.

**Ứng viên** (kiểm tra hạn trên trang chính thức của từng hội nghị; không dựa vào ngày nhớ):

| Venue | Phù hợp | Cần thêm |
|---|---|---|
| RLC (Reinforcement Learning Conference) | Cao: đánh giá thực nghiệm RL chặt chẽ, có track cho phương pháp đánh giá | Việc A, B dưới đây |
| ICML / NeurIPS main track | Trung bình–cao nếu có thêm A, B, C | Cả ba việc; viết lại theo template, 8–9 trang + phụ lục |
| TMLR | Cao, không có hạn chót | A |
| IEEE Access / Transactions | Đã đủ | Chỉ cần định dạng |

**Ba việc có lợi nhất để lên venue cao:**
- **A. Một panel ngoài thứ năm, khóa trước, không phải forest** (vài giờ máy). Mục đích là xác lập lợi thế so với B2 + legacy ngoài một họ MDP duy nhất. Đây là điểm yếu lớn nhất còn lại.
  - Đề xuất: họ MDP công bố khác, ví dụ các bài toán tabular ToyText của Gymnasium, hoặc một họ khác trong MDPToolbox.
  - Một phép so duy nhất đã đăng ký: v2 vs B2 + legacy, cả hai chỉ số.
- **B. Viết lại theo template venue.** Hiện bài bị ép 8 trang IEEE.
  - Phần phương pháp v2, Mệnh đề 1–2 và bảng độ vững nên vào bài chính.
  - Phần lịch sử cohort, selector và BRIDGE chuyển sang phụ lục.
- **C. M5: posterior thưa / khấu hao OEH-B** để đưa các task BRIDGE lớn vào (125/155 task đang bị loại vì kích thước). Việc này tốn vài tuần, chỉ làm nếu nhắm ICML/NeurIPS.

## 5. Thesis (`Drive/Thesis/v2_2026-10-13/`, build `DAORL_Thesis_v2_2026-10-15.docx/.pdf`, 55 trang)

| Chương | Đã làm 15/10 |
|---|---|
| Tóm tắt | Thêm cohort 8, cơ chế ngoài generator, OEH v2 và kết quả niêm phong |
| 1 | RQ2 tách thành RQ2a/RQ2b; thêm RQ4; thêm đóng góp 4 |
| 3 | Mục 3.7 mới: các panel do nhóm khác thiết kế (Colosseum, SPIBB, forest, BRIDGE), khoảng cách tới meta-training |
| 4 | 4.3: tính nhất quán của luật OEH-B. 4.9: gatekeeping và kiểm tra fit niêm phong. 4.10: factorial niêm phong. 4.11 mới: OEH v2, Mệnh đề 2, quy trình chọn, các thay đổi bị loại |
| 5 | Sửa đánh số (5.7 tái lập, 5.8 OEH v2, 5.9 chi phí và giới hạn). 5.8 mới: cổng, bảng theo panel, độ vững, cơ chế, endpoint mở rộng, regret, Hình 5.3 |
| 6 | Trả lời RQ2a, RQ2b, RQ4; đóng góp (iv); hướng phát triển viết lại theo kết quả M2/M3 |
| Phụ lục | Bảng C.1 amendment trước outcome; D.1 thêm các panel mới; E.1 thêm artifact |

**Còn lại cho một đồ án xuất sắc** (cần người):
- Giảng viên duyệt claim ở mục 3.
- Điền tác giả và affiliation.
- Slide bảo vệ, khoảng 15 trang:
  - bài toán → target → OEH → kết quả cohort 7;
  - cơ chế ngoài generator → OEH v2 → kiểm định niêm phong;
  - giới hạn (SPIBB, phụ thuộc forest).
- Dùng Hình 5.3 làm hình chính.
- Tập trả lời câu hỏi "vì sao không thắng trên SPIBB?" và "vì sao lợi thế so với B2 + legacy nhờ forest?".

## 6. Việc tiếp theo, xếp theo hiệu quả trên thời gian

| Ưu tiên | Việc | Chi phí |
|---|---|---|
| P0 | Review PR #53; giảng viên duyệt claim | Người |
| P1 | Việc A (panel ngoài thứ năm, một phép so) | Vài giờ máy |
| P1 | Slide bảo vệ dùng Hình 5.3 và Bảng 5.9 | Nửa ngày |
| P2 | Việc B (bản cho venue cao) | 2–3 ngày viết |
| P3 | Việc C (M5) | Vài tuần |

## 7. Kiểm tra trước khi nộp

1. `python -m scripts.testing.run_full_tests` (không có test bị skip), cộng `verify_oeh` và `export_week8_d1 --check`.
2. `paper/build.ps1` ≤ 8 trang; `build_supplement.ps1`.
3. Thesis: `build_thesis_v2.py` rồi `update_fields.ps1`; kiểm mục lục, danh sách bảng/hình, mục 5.8.
4. Mọi số của v2 lấy từ `artifacts/oeh_v2_sealed_test_2026-10-14/{result.json, report.md, robustness.csv}`.
5. Không có dòng ghi công AI trong commit hay PR.

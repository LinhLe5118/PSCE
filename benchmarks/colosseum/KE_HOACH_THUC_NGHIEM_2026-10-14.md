# Kế hoạch thực nghiệm để bài báo OEH mạnh nhất có thể

*Đi kèm `README.md` của gói thử nghiệm Colosseum. Người viết: thành viên nhóm (máy phụ), 14/10/2026. Đây là đề xuất để nhóm trưởng quyết định; chưa thí nghiệm nào dưới đây được chạy, trừ những gì `README.md` đã báo cáo.*

## 0. Nguyên tắc chung

1. **Mọi phương pháp mới được chọn trên meta-training (720 MDP, OOF).** Không chọn trên cohort 7, Colosseum hay BRIDGE, vì các tập này đã mở nhãn.
2. **Kiểm định trên một cohort mới, niêm phong trước kết quả**, giống quy trình cohort 7.
3. **Mỗi panel một bảng riêng.** Không gộp nội bộ, Colosseum và BRIDGE; báo cáo cả dòng bất lợi (đúng "manuscript reporting rule" của `CURRENT_EVIDENCE_AND_RETIREMENT_2026-10-09.md`).
4. **Kết quả xấu vẫn là kết quả.** Mục 8 ghi trước cách đọc từng kịch bản xấu, để không phải quyết định sau khi đã thấy số.

## 1. Claim mạnh nhất có thể nhắm tới, và bằng chứng còn thiếu

> **Claim mục tiêu:** "Từ một pilot nhỏ, OEH dự báo phân phối nhu cầu dữ liệu của từng learner offline RL tốt hơn mọi baseline có cùng thông tin, kể cả cách thực tế nhất là chạy thử learner trên pilot rồi ngoại suy. Ưu thế giữ được trên MDP do người khác thiết kế, đến từ mô phỏng hậu nghiệm, và chuyển thành tiết kiệm dữ liệu khi ra quyết định."

| Mệnh đề trong claim | Bằng chứng hiện có | Còn thiếu | Thí nghiệm |
|---|---|---|---|
| Tốt hơn baseline cùng thông tin | Cohort 7: 22/22 GO | Đối thủ "thực tế" (ngoại suy learning curve), đối thủ lý thuyết (concentrability) | A4, B3 |
| Giữ được trên MDP của người khác | Colosseum: GO vs B2, NO-GO vs B2+ERI; BRIDGE 5+5 lẫn lộn | XGBoost-CF trên Colosseum; panel có behavior từ nguồn ngoài (SPIBB) | A1, C1 |
| Ưu thế đến từ mô phỏng hậu nghiệm | M1 exploratory: bỏ OEH-B gần như không mất gì | Ablation factorial trên dữ liệu chưa mở | A2, B4 |
| Dự báo đáng tin (calibrated) ngoài phân phối | Log-loss "nổ" ở RiverSwim | Hiệu chỉnh có tính đến dịch chuyển phân phối | B1 |
| Ước lượng được mức ngân sách, không chỉ phát hiện thất bại | OEH current không hơn B2 ở case hữu hạn; OEH đời trước tốt hơn | Mô hình hai tầng | B2 |
| Có giá trị khi ra quyết định | Chưa có | Phân tích chi phí/regret | A3, D2 |

---

## 2. Giai đoạn A: không cần dữ liệu mới (khoảng 1–2 ngày, máy phụ làm được)

### A1. Colosseum thành panel chính thức, thêm XGBoost-CF

- Chạy XGBoost-CF/18 (`published_survival_120_exploratory.py`) và XGBoost-CF/108 (`published_survival_120_full_architecture.py`) trên 72 MDP Colosseum, dùng đúng cấu hình của các panel BRIDGE (`bridge_*_published_xgboost_cf_*`).
- Bảng chính theo quy ước so sánh hiện tại: OEH vs B2 vs XGBoost-CF/18, có cột XGBoost-CF/108 làm kiểm soát kiến trúc.
- Supplement: 9 comparator của cohort 7 (đã có), gồm dòng B2+ERI.
- Báo cáo theo learner, theo họ MDP, tỷ lệ >x50, runtime.

### A2. Ablation factorial trên Colosseum

M1 chỉ *bỏ* từng khối. Vì các khối chồng lấp thông tin, bỏ một khối thì khối khác bù lại, nên M1 không trả lời được "khối nào cần thiết". Đề xuất chạy **đủ 2³ = 8 tổ hợp** của 3 khối trên nền B2:

| Khối | Thành phần (theo `feature_lineage.csv` của M1) |
|---|---|
| P (posterior simulation) | OEH-B trực tiếp (G2, G3, G6) |
| H (horizon/risk) | OEH-H và risk profile (G4, G5) |
| L (legacy) | Diagnostics cũ và context (G7, G8) |

8 arm: B2, B2+P, B2+H, B2+L, B2+P+H, B2+P+L, B2+H+L, full.

- Từ đó tính **giá trị Shapley theo khối** (đóng góp trung bình của mỗi khối qua mọi thứ tự thêm vào), kèm khoảng bootstrap theo MDP.
- Refit rẻ: cùng `matched_forecast`, cùng fold và tuning. Mở rộng `run_m1_matched_ablation.py` thêm 4 arm.
- Chạy trên Colosseum: định nghĩa khối đóng băng 13/10 16:20 UTC, trước khi có nhãn Colosseum (14/10 09:05). Phải ghi rõ trình tự thời gian này.
- Đồng thời chạy trên cohort 7 (exploratory) để so sánh.

**Câu hỏi được trả lời:** OEH-B có đóng góp riêng (Shapley > 0) không, kể cả khi bỏ nó đi không làm kết quả kém đi? Khối nào gây ra thất bại ở RiverSwim?

### A3. Giá trị khi ra quyết định (descriptive, trên cohort 7 và Colosseum)

Biến Brier/log-loss thành "tiết kiệm bao nhiêu dữ liệu". Nhãn đã có nên chỉ cần tính lại.

- **Bài toán:** với mỗi dòng (MDP, behavior, learner) và dự báo p trên E ∈ {0..6}, chính sách chọn mức thu thập b\* = mức nhỏ nhất trên lưới (x1, x2, x5, x10, x20, x50) sao cho P(E ≤ b\*) ≥ τ. Nếu không có mức nào thỏa thì "không dùng learner này".
- **Chi phí:**
  - Thu thập b\* mà đạt (E ≤ b\*): chi phí = budget[b\*].
  - Thu thập b\* mà không đạt: chi phí = budget[b\*] + λ.
  - Bỏ qua khi thực ra đạt được: phạt λ_miss.
  - Oracle: chi phí = budget[E], hoặc 0 nếu E = 6 và bỏ qua.
- **Chỉ số:** regret = chi phí − chi phí oracle, trung bình theo MDP, với τ ∈ {0,5; 0,8; 0,9} và λ ∈ {x50, 2·x50}. **Khóa các giá trị này trước khi tính.**
- **Thêm bài toán chọn learner:** chọn learner có kỳ vọng nhu cầu nhỏ nhất, so với learner tốt nhất cố định. Phần này bổ sung cho selector study hiện có.
- **So sánh:** OEH, B2, XGBoost-CF, LCE (A4), "luôn x50", "luôn x10".

### A4. Baseline ngoại suy learning curve (LCE): đối thủ "thực tế" nhất

Reviewer sẽ hỏi: *"Sao không chạy learner trên pilot rồi xem?"*. Pilot có x10, nên các mức x1..x10 có thể "thử" trực tiếp.

- **Dữ liệu sẵn có:** `curve_preoutcome.csv` (cohort 7) và `curve_{cohort}_A2.csv` (meta) đã có `oehc_success_x{1,2,5,10}`. Đây là learner huấn luyện trên prefix của pilot, đánh giá trong các MDP hậu nghiệm.
- **LCE:**
  1. Khớp đường cong đơn điệu cho xác suất thành công theo log(budget). Dạng logistic s(b) = σ(α + β·log b), ràng buộc β ≥ 0, fit trên 4 điểm x1..x10 của chính dòng đó.
  2. Ngoại suy tới x20, x50.
  3. Chuyển sang phân phối 7 lớp: P(E ≤ k) xấp xỉ s(b_k), làm đơn điệu bằng isotonic, cộng làm mịn nhỏ.
- **Hai biến thể:**
  - LCE thô (không học gì từ meta-training): đây là đối thủ "zero-shot".
  - LCE + head hiệu chỉnh: đưa các đại lượng LCE vào `matched_forecast` như một field set.
- **Chi phí:** cohort 7 rất rẻ, vì chỉ cần đọc file curve. Colosseum cần chạy `run_curve`, khoảng 20–30 phút.
- **Lưu ý:** biến thể A5 (`plus_prior_profile_curve`) từng đưa curve *vào* OEH. LCE thì là *đối thủ độc lập*, hai thứ khác nhau. Nếu LCE mạnh, OEH phải chứng minh giá trị ở vùng ngoại suy x20/x50 và >x50.

### A5. Leave-one-family-out (chẩn đoán dịch chuyển họ MDP)

- Trên panel 720 MDP: lần lượt bỏ hẳn một họ khỏi meta-training, fit lại, dự báo họ đó. So OEH với B2 cho từng họ.
- Dữ liệu đã mở nên đây chỉ là chẩn đoán. Nhưng nó trả lời câu reviewer hay hỏi ("có chuyển sang họ chưa thấy không?") và **cung cấp OOF cho B1**.

### A6. Trình bày

- Bảng chi phí theo |S||A|, lấy từ `runtime_seconds` trong file OEH-B và log feature.
- Ghi con số 125/155 BRIDGE bị loại vì kích thước.
- Chỉnh câu chữ claim theo bảng ở mục 2 của `README.md`.

---

## 3. Giai đoạn B: phương pháp mới và cohort niêm phong mới

### B1. OEH-S: hiệu chỉnh có tính đến dịch chuyển (sửa log-loss "nổ")

**Bài học từ lần trước:** `oeh_difficulty_temperature_development_2026-10-01` fit temperature trên dữ liệu cùng phân phối và ra T = 0,95, tức là làm *sắc hơn*. Trong cùng phân phối, OEH không quá tự tin. Lỗi xuất hiện khi **phân phối dịch chuyển** (RiverSwim), nên phải hiệu chỉnh trên OOF *có dịch chuyển*.

**Phương pháp:**

p'(k) ∝ (1 − ε) · p(k)^(1/T) + ε · π_ℓ(k)

trong đó π_ℓ là phân phối lớp của learner ℓ trên meta-training.

- Chọn (T, ε) bằng cách tối thiểu log-loss trên OOF của **leave-one-family-out** (A5) hoặc **leave-one-cohort-out**. Không dùng OOF ngẫu nhiên theo MDP.
- Lưới nhỏ: T ∈ {1; 1,25; 1,5; 2}, ε ∈ {0; 0,01; 0,02; 0,05}. Khóa lưới trước.
- Phương án dự phòng định trước, không fit: sàn xác suất 0,005 cho mọi lớp.
- **Kỳ vọng:** log-loss ngoài phân phối giảm mạnh, Brier gần như giữ nguyên. Có thể kiểm tra post hoc trên Colosseum với nhãn **post hoc**, nhưng không dùng Colosseum để chọn T và ε.

### B2. OEH-2T: mô hình hai tầng, mỗi tầng dùng bộ feature mạnh nhất cho nó

**Bài học từ lần trước:** `oeh_difficulty_hurdle_development_2026-10-01` (thành công trong x50, rồi 6 lớp có điều kiện) không được chọn. Lần đó hai tầng dùng chung một bộ feature.

**Khác biệt lần này:** dữ liệu Colosseum cho thấy hai tầng cần hai bộ feature khác nhau.

| | Tốt nhất ở tầng này |
|---|---|
| Tầng 1: P(E = 6), "không học nổi" | OEH current (AUC 0,915) |
| Tầng 2: P(E = k \| E < 6), mức ngân sách | OEH đời trước, `prior_oeh_matched` (đúng ±1 mức 62% vs 52%; log-loss có điều kiện 1,66 vs 2,01) |

**Phương pháp:**
- P(E = 6) lấy từ head OEH current.
- P(E = k \| E < 6) lấy từ head fit **chỉ trên các dòng hữu hạn**, với field set `prior_oeh_matched` (hoặc hợp của hai bộ feature).
- Ghép: P(E = k) = (1 − P₆) · P(k \| hữu hạn).
- Chọn giữa {OEH current, OEH-2T, convex blend} bằng OOF trên 720 MDP. Nếu OOF không chọn OEH-2T thì **không đưa vào cohort mới**.
- Phần quan sát trên Colosseum chỉ là động lực, không phải bằng chứng.

### B3. Hai đối thủ mới cho cohort mới

1. **LCE** (A4): cả bản thô lẫn bản có head.
2. **Concentrability / sample-complexity baseline (CB):**
   - Ước lượng C\* = max_(s,a) d^{π̂\*}(s,a) / μ̂(s,a).
     - μ̂: phân phối (s,a) thực nghiệm của pilot.
     - d^{π̂\*}: occupancy chiết khấu của chính sách tối ưu trên mô hình hậu nghiệm trung bình.
     - Cặp chưa thấy mà có d > 0 thì C\* = ∞: chặn trên và thêm cờ.
   - Cận sample complexity (Li et al. 2022, *Settling the sample complexity of model-based offline RL*; khớp với cận dưới tới thừa số log): n̂(ε) ∝ S · C\* / ((1 − γ)³ ε²).
   - Field: log n̂, log C\*, cờ C\* vô hạn. Hai comparator: `cb_only` và `b2_plus_cb`, qua cùng `matched_forecast`.
   - Đây là đối thủ có gốc lý thuyết cho đúng đại lượng cần dự báo.

### B4. Thiết kế cohort mới niêm phong (gộp B1, B2, B3, A2)

**Điều đã làm hỏng cận dưới trên Colosseum:** họ 18 phép so đồng thời đẩy critical value lên 2,25. Đề xuất **kiểm định phân cấp (fixed-sequence/gatekeeping)**, khai báo trước. Cách này vẫn kiểm soát sai lầm loại I toàn họ, nhưng không bị phạt đa phép so ở bước chính.

1. **Bước 1 (chính, α đầy đủ):** OEH current vs B2 và vs XGBoost-CF/18. Max-t đồng thời 4 phép so: 2 đối thủ × 2 chỉ số.
2. **Bước 2 (chỉ khi bước 1 GO):** OEH current vs LCE, CB, B2+ERI.
3. **Bước 3 (chỉ khi bước 2 GO):** OEH-S và OEH-2T vs OEH current.
4. **Mô tả, không gate:** ablation factorial và Shapley; regret ra quyết định (A3, với τ, λ đã khóa); các comparator tier 2.

**Cỡ mẫu:**
- Trên Colosseum, độ lệch chuẩn của gain log-loss theo MDP là 0,43 với gain 0,09. Với critical khoảng 1,9 (4 phép so), cần khoảng (1,9 × 0,43 / 0,09)² ≈ 82 MDP.
- Đề xuất **≥ 120 MDP ngẫu nhiên**, để còn biên cho dịch chuyển.

**Thành phần panel:**
- Cohort nội bộ mới, như cohort 7.
- **Thêm một tầng Colosseum mới** (khoảng 72 MDP, sampler seed mới, loại kernel trùng), báo cáo riêng.
- Panel đồng thời đi qua A1–A3, nên chỉ cần **một lần sinh dữ liệu** cho cả kiểm định phương pháp mới lẫn ablation sạch.

**Chi phí:** tương đương cohort 7 cộng một lần Colosseum, trên máy chính. Máy phụ có thể chạy tầng Colosseum song song, pipeline đã có sẵn.

---

## 4. Giai đoạn C: tổng quát hóa ra ngoài generator

### C1. Benchmark SPIBB: behavior policy cũng từ nguồn ngoài

- **Nguồn:** Laroche, Trichelair & Tachet des Combes (ICML 2019), *Safe Policy Improvement with Baseline Bootstrapping*. Nhóm đã có bản pin `external_spibb` trong pipeline learner.
- **MDP:** random MDP của benchmark, khoảng 50 trạng thái, 4 hành động, chuyển trạng thái thưa. Đúng cỡ OEH xử lý được.
- **Behavior:** baseline policy sinh theo quy trình của tác giả (trộn softmax trên Q\* để đạt một tỷ lệ hiệu năng định trước), không dùng 5 behavior của nhóm.
- **Pilot và nhãn:** pilot sinh bằng baseline policy của benchmark; nhãn dùng quy tắc `suffix_event` và các learner của nhóm.
- **Ý nghĩa:** lần đầu tiên cả MDP lẫn cách sinh dữ liệu không đến từ generator của nhóm.
- **Lưu ý thiết kế:** meta-training không có các behavior kiểu SPIBB. Đây chính là phép thử dịch chuyển cần đo, nên báo cáo trung thực dù kết quả xấu.
- **Chi phí:** khoảng 3–4 giờ trên máy chính, vì cần chạy code SPIBB của tác giả.

### C2. Tăng cỡ mẫu BRIDGE

Hai panel đầy đủ hiện mỗi panel chỉ có 5 MDP. Trong 155 bản ghi, 5 task mới đã qua sàng lọc. Nếu D1 nới được giới hạn kích thước, nhiều task BRIDGE hơn sẽ đủ điều kiện, và đây là con đường tự nhiên để có panel BRIDGE đủ lực thống kê.

---

## 5. Giai đoạn D: nâng tầm cho hội nghị lớn (vài tháng)

### D1. Mở rộng quy mô OEH

- **Posterior thưa:** chỉ giữ Dirichlet trên support quan sát được, cộng một trạng thái "chưa biết" gộp lại cho khối lượng prior trên các chuyển trạng thái chưa thấy. Chi phí giảm từ S²A xuống khoảng số phần tử khác 0.
- **Ít draw hơn, có kiểm soát sai số:** dừng tuần tự khi phân phối E ổn định. Chạy OEH-B trên GPU (repo đã có `requirements-gpu.txt`, cupy).
- **Mục tiêu:** đưa vào được phần lớn 125 task BRIDGE đang bị loại. BRIDGE là bản tabular của Atari/MiniGrid/Procgen, cây cầu tự nhiên sang môi trường mà reviewer quan tâm.

### D2. Regret ra quyết định làm kết quả tiêu đề

Khóa trước cách tính A3 (τ, λ, chính sách) trong protocol của cohort mới. Báo cáo dạng "dùng OEH tiết kiệm X% dữ liệu so với B2/LCE ở cùng tỷ lệ thất bại". Reviewer hội nghị lớn hiểu ngay con số này.

### D3. Một kết quả lý thuyết

- **Mệnh đề nhất quán:** khi pilot tăng và mọi (s,a) liên quan được thăm đủ, posterior hội tụ về MDP thật. Nếu learner là hàm liên tục (hầu khắp) của dữ liệu, phân phối dự báo của OEH-B hội tụ về phân phối nhu cầu thật. Điều này cũng giải thích vì sao OEH-B yếu khi pilot thưa.
- **Liên hệ hình thức** giữa OEH-H và effective horizon của Laidlaw et al., vì tên phương pháp gợi liên hệ này.

---

## 6. Chiến lược venue

| Nơi nộp | Hợp hiện tại | Sau giai đoạn A + B | Sau giai đoạn D |
|---|---|---|---|
| IEEE Access/Transactions (định hướng hiện tại) | Cao | Rất cao | — |
| TMLR (chấm tính đúng đắn, claim có được chứng minh) | Cao | Rất cao | — |
| RLC (Reinforcement Learning Conference) | Khá | Cao | Cao |
| Workshop NeurIPS/ICML (offline RL, data-centric) | Cao | Cao | — |
| NeurIPS Datasets & Benchmarks (đóng gói thành benchmark dự báo nhu cầu dữ liệu) | Trung bình | Khá | Khá |
| NeurIPS/ICML/ICLR main track | Thấp | Trung bình thấp | **Trung bình–khá** |

Hạn nộp thay đổi theo năm, cần kiểm tra trang chính thức. Gợi ý: nộp TMLR/RLC hoặc IEEE với kết quả sau A + B, gửi workshop song song. Nếu nhắm main track thì cần D1 + D2, tốt nhất có cả D3.

---

## 7. Lộ trình và phân công máy

| Bước | Việc | Phụ thuộc | Máy | Ước tính |
|---|---|---|---|---|
| 1 | Nhóm trưởng review code `driver/` và quyết định đưa Colosseum vào bài | — | — | 0,5 ngày |
| 2 | A1, A2, A3, A4 (Colosseum + cohort 7), A6 | 1 | Máy phụ | 1–2 ngày |
| 3 | A5 (leave-family-out), dev B1, B2, B3 trên OOF 720 MDP | — | Máy chính hoặc phụ | 2–3 ngày |
| 4 | Viết và khóa protocol cohort mới (B4) | 2, 3 | Nhóm trưởng | 0,5 ngày |
| 5 | Sinh và chạy cohort nội bộ mới | 4 | Máy chính | Như cohort 7 |
| 5' | Tầng Colosseum mới (song song) | 4 | Máy phụ | ~4 giờ |
| 6 | C1 SPIBB | 4 | Máy chính | ~4 giờ |
| 7 | Chấm điểm, cập nhật bài | 5, 5', 6 | — | 1 ngày |
| 8 | D1–D3 (nếu nhắm main track) | 7 | Cả hai | Vài tháng |

---

## 8. Cách đọc các kịch bản xấu (ghi trước khi có kết quả)

| Nếu… | Thì bài báo nói… |
|---|---|
| LCE thắng OEH | Chạy thử trên pilot là baseline đủ mạnh. Giá trị của OEH (nếu có) chỉ nằm ở vùng ngoại suy x20/x50/>x50; thu hẹp claim về đó |
| Shapley của khối P ≈ 0 trên dữ liệu chưa mở | Mô phỏng hậu nghiệm không phải nguồn của ưu thế. Viết lại đóng góp quanh tổ hợp tín hiệu; khuyến nghị biến thể không OEH-B (rẻ hơn 84%) |
| OEH-S không giảm log-loss ngoài phân phối | Quá tự tin là giới hạn cấu trúc; báo cáo kèm phân tích RiverSwim |
| OEH-2T không qua OOF | Không đưa vào cohort mới; báo cáo ngắn trong supplement như lần hurdle trước |
| Bước 1 gatekeeping NO-GO trên cohort mới | Kết quả cohort 7 không lặp lại được. Phải báo cáo như một thất bại tái lập; claim chỉ còn mức mô tả |
| SPIBB cho OEH thua B2 | OEH phụ thuộc vào cách sinh dữ liệu của generator; nêu rõ đây là giới hạn chính |
| CB (concentrability) mạnh | Kết nối tốt với lý thuyết. Định vị OEH như phần bổ sung thực nghiệm cho cận lý thuyết, không phải đối thủ |

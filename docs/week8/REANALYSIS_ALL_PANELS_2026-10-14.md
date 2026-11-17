# Phân tích lại toàn bộ kết quả OEH (14/10/2026)

**Phạm vi:**
- Đây là phân tích **mô tả**. Nó chỉ dùng dự báo đã niêm phong và nhãn đã mở của mọi panel kiểm định OEH đầy đủ; không fit dự báo mới, không có gate mới, và không thay đổi verdict nào đã đăng ký.
- Mục tiêu: kiểm tra lại các con số, đo độ vững, tìm nguồn của lợi thế, và tổng hợp chéo panel.
- Script: [`cross_panel_reanalysis_20261014.py`](../../scripts/analysis/cross_panel_reanalysis_20261014.py). Đầu ra: [`cross_panel_reanalysis_2026-10-14/`](../../artifacts/week8/cross_panel_reanalysis_2026-10-14/).
- Bootstrap 10.000 lần theo MDP, seed 20261016. Log loss dùng sàn 1e-12. Lợi thế = B2 − OEH (dương là OEH tốt hơn).

| Nhóm | Panel | MDP | Ghi chú |
|---|---|---:|---|
| Trong generator | cohort 4, cohort 7, cohort 8 | 84, 84, 168 | Cùng cấu hình OEH. Cohort 4 có meta-training ít hơn một cohort. |
| Kernel người khác | Colosseum 2027, Colosseum 2028 | 72, 72 | Behavior, learner, nhãn của nhóm. |
| MDP và behavior người khác | SPIBB | 108 | Mỗi MDP một behavior (baseline của tác giả). |
| MDP công bố | BRIDGE Atari-5, Expanded-5 | 5, 5 | Sticky action, γ = 0,99. |

## 1. Tái lập số liệu

Tính lại bằng một bộ chấm chung, toàn bộ số đã báo cáo đều khớp (IBS / LL, [`verification.csv`](../../artifacts/week8/cross_panel_reanalysis_2026-10-14/verification.csv)):
- Cohort 4: 0,0529 / 0,643 vs B2 0,0624 / 0,700.
- Cohort 7: 0,0460 / 0,563 vs 0,0547 / 0,627.
- Cohort 8: 0,0466 / 0,548 vs 0,0566 / 0,614.
- Colosseum 2027: 0,1010 / 0,890 vs 0,1342 / 0,982.
- Colosseum 2028: 0,1050 / 0,895 vs 0,1436 / 1,000.
- SPIBB: 0,0192 / 0,366 vs 0,0209 / 0,387.

Một điểm chưa từng được nêu: ở BRIDGE Atari-5, **gộp ba learner thì OEH kém B2** (0,1417 / 1,104 vs 0,1355 / 0,974). Trước đây panel này chỉ được báo theo từng learner.

## 2. Câu hỏi chính: OEH có tốt hơn thống kê chung (B2) không?

| Panel | Gain IBS | KTC 95% (tuyệt đối) | Gain LL | KTC 95% (tuyệt đối) | MDP tốt hơn (IBS) | Bỏ 5 MDP có lợi nhất (IBS / LL) |
|---|---:|---:|---:|---:|---:|---:|
| Cohort 4 | 15,2% | [0,0041; 0,0151] | 8,1% | [0,025; 0,090] | 75% | 0,0054 / 0,033 |
| Cohort 7 | 16,0% | [0,0042; 0,0135] | 10,2% | [0,038; 0,091] | 69% | 0,0054 / 0,046 |
| Cohort 8 | 17,8% | [0,0070; 0,0133] | 10,8% | [0,049; 0,084] | 72% | 0,0081 / 0,057 |
| Colosseum 2027 | 24,8% | [0,0144; 0,0534] | 9,3% | [−0,008; 0,189] | 69% | 0,0163 / **0,015** |
| Colosseum 2028 | 26,9% | [0,0194; 0,0582] | 10,5% | [0,011; 0,197] | 72% | 0,0251 / 0,047 |
| SPIBB | 7,8% | [−0,0003; 0,0035] | 5,2% | [−0,004; 0,043] | 62% | 0,0005 / 0,010 |
| BRIDGE Atari-5 | −4,6% | [−0,022; 0,008] | −13,4% | [−0,293; 0,005] | 40% | — |
| BRIDGE Expanded-5 | 30,7% | [−0,004; 0,095] | 17,5% | [−0,063; 0,364] | 60% | — |

**Tổng hợp random-effects** của log(B2/OEH) ([`meta_analysis.csv`](../../artifacts/week8/cross_panel_reanalysis_2026-10-14/meta_analysis.csv)):

| Nhóm | Gain IBS gộp [95%] | I² | Gain LL gộp [95%] | I² |
|---|---:|---:|---:|---:|
| Trong generator (3 cohort) | **16,8%** [13,2; 20,3] | 0 | **10,1%** [8,2; 12,1] | 0 |
| Ngoài generator (5 panel) | 16,0% [2,6; 27,6] | 0,79 | 4,9% [−3,8; 12,8] | 0,69 |
| Tất cả (8 panel) | 15,3% [9,1; 21,1] | 0,69 | 7,6% [4,0; 11,2] | 0,64 |

**Holm trên 16 phép so** (8 panel × 2 metric, p một phía từ bootstrap; [`holm.csv`](../../artifacts/week8/cross_panel_reanalysis_2026-10-14/holm.csv)):
- Còn ý nghĩa: cả 6 phép so trong generator, cùng IBS của hai panel Colosseum.
- Mất ý nghĩa: LL của Colosseum 2027 và 2028 (p hiệu chỉnh 0,24 và 0,11), SPIBB và BRIDGE.

**Đọc kết quả:**
- Trong generator, kết quả **tái lập gần như y hệt** ba lần (I² = 0). Hiệu ứng vững với việc bỏ các MDP cực trị.
- Ngoài generator, lợi thế Brier vẫn có nhưng dao động mạnh giữa các panel. Lợi thế log loss không được xác lập khi gộp.
- Ở Colosseum 2027, LL dựa vào vài MDP: bỏ 5 MDP có lợi nhất thì gain giảm từ 0,091 xuống 0,015.
- Verdict trong từng panel (gate đã đăng ký) vẫn đúng. Holm chéo panel chỉ là góc nhìn bảo thủ bổ sung, không phải gate.

## 3. Lợi thế đến từ đâu

### 3.1 Dòng bị cắt (`>x50`) so với dòng hữu hạn

Tỉ lệ đóng góp vào lợi thế OEH − B2 ([`decomposition.csv`](../../artifacts/week8/cross_panel_reanalysis_2026-10-14/decomposition.csv)):

| Panel | IBS: cắt / hữu hạn | LL: cắt / hữu hạn |
|---|---:|---:|
| Cohort 4 | 59% / 41% | 53% / 47% |
| Cohort 7 | 33% / 67% | 28% / 72% |
| Cohort 8 | 25% / 75% | 27% / 73% |
| Colosseum 2027 | 98% / 2% | 120% / −20% |
| Colosseum 2028 | 77% / 23% | 90% / 10% |
| SPIBB | 49% / 51% | −6% / 106% |

### 3.2 Hoán đổi thành phần

Ghép P(E = 6) của một phương pháp với phân phối mức hữu hạn của phương pháp kia (LL; [`swap.csv`](../../artifacts/week8/cross_panel_reanalysis_2026-10-14/swap.csv)):

| Panel | B2 | p₆ B2 + hữu hạn OEH | p₆ OEH + hữu hạn B2 | OEH |
|---|---:|---:|---:|---:|
| Cohort 8 | 0,614 | 0,591 | 0,572 | **0,548** |
| Colosseum 2027 | 0,982 | 1,016 | **0,856** | 0,890 |
| Colosseum 2028 | 1,000 | 1,019 | **0,876** | 0,895 |
| SPIBB | 0,387 | **0,370** | 0,382 | 0,366 |

**Kết luận:**
- **Trong generator,** cả hai thành phần của OEH đều có ích. OEH nhận ra ca bị cắt tốt hơn và đặt mức ngân sách tốt hơn; hai cái cộng lại.
- **Trên Colosseum,** toàn bộ lợi thế nằm ở việc nhận ra ca bị cắt. Phân phối mức hữu hạn của OEH **kém hơn của B2**: thay nó bằng của B2 làm LL tốt lên 0,02–0,03. Lần thất bại ở RiverSwim cũng nằm ở đây.
- **Trên SPIBB,** mọi phương pháp đều nhận ra BC và IQL không đạt (AUC ≈ 0,99). Chút lợi thế còn lại của OEH là đặt mức cho CFQI.

### 3.3 Hiệu chỉnh hay phân biệt (phân rã Murphy, trung bình sáu ngưỡng)

| Panel | Brier B2 → OEH | Reliability B2 → OEH | Resolution B2 → OEH |
|---|---:|---:|---:|
| Cohort 8 | 0,0566 → 0,0466 | 0,0012 → 0,0007 | 0,0606 → 0,0705 |
| Colosseum 2027 | 0,1342 → 0,1010 | **0,0228 → 0,0101** | 0,0389 → 0,0594 |
| Colosseum 2028 | 0,1436 → 0,1050 | **0,0247 → 0,0094** | 0,0400 → 0,0626 |
| SPIBB | 0,0209 → 0,0192 | 0,0027 → 0,0031 | 0,0381 → 0,0403 |

- Trong generator, mọi phương pháp đều hiệu chỉnh gần hoàn hảo, nên lợi thế của OEH là **phân biệt**.
- Khi kernel thay đổi, **B2 mất hiệu chỉnh còn OEH giữ được tương đối tốt**. Khoảng 40% lợi thế Brier trên Colosseum là do hiệu chỉnh. Đây là phát hiện mới, có lợi cho OEH, và nên đưa vào bài.

## 4. OEH so với các đối thủ mạnh nhất

Gain tương đối của OEH và KTC 95% của cải thiện tuyệt đối ([`paired.csv`](../../artifacts/week8/cross_panel_reanalysis_2026-10-14/paired.csv)):

| Đối thủ | Cohort 4 | Cohort 7 | Cohort 8 | Col. 2027 | Col. 2028 | SPIBB |
|---|---|---|---|---|---|---|
| B2 + legacy, IBS | 5,8% (chứa 0) | 9,7% | 7,0% | 3,6% (chứa 0) | 13,3% (chứa 0) | −1,7% (chứa 0) |
| B2 + legacy, LL | 3,0% (chứa 0) | 7,4% | 5,5% | **−5,6%** (chứa 0) | 0,3% (chứa 0) | **−4,4%** (chứa 0) |
| B2 + ERI, IBS / LL | 14% / 8% | 19% / 11% | 17% / 10% | 16% / 3% (LL chứa 0) | 25% / 8% (LL chứa 0) | 15% / 8% |
| XGBoost-CF/108, IBS | — | — | 2,1% (chứa 0) | — | 3,0% (chứa 0) | 22% |

- **B2 + legacy** (`prior_oeh_matched`, không cần posterior) là đối thủ sát nhất. OEH vượt nó có ý nghĩa ở cohort 7 và 8; không vượt ở cohort 4 và ở mọi panel ngoài. Ở Colosseum 2027 và SPIBB, B2 + legacy còn có LL tốt hơn OEH.
- **XGBoost-CF/108** dùng cùng 108 cột: Brier ngang OEH trong generator và trên Colosseum. Phần lợi thế nằm ở **tập feature**, không ở head ensemble. Lợi thế LL của OEH trước XGBoost-CF chủ yếu do XGBoost gán xác suất 0 (mục 5).
- Ở Colosseum 2027, **B2 + TQ/SACo gốc** (dùng dữ liệu tham chiếu từ MDP thật) tốt hơn OEH một chút.

## 5. Độ nhạy

**Sàn log loss** ([`floor_sensitivity.csv`](../../artifacts/week8/cross_panel_reanalysis_2026-10-14/floor_sensitivity.csv)):
- LL của OEH và B2 gần như không đổi khi sàn chạy từ 1e-15 tới 1e-3. Kết luận OEH vs B2 không phụ thuộc vào sàn.
- XGBoost-CF/18 thì nhạy: ở cohort 7, LL bằng 1,34 với sàn 1e-15 và 0,68 với sàn 1e-2, so với OEH 0,56. Ở BRIDGE Atari với sàn 1e-2, XGBoost-CF/18 (0,93) còn tốt hơn OEH (1,04).
- Bài nên nói rõ: khoảng cách LL lớn với XGBoost-CF là hệ quả của các xác suất bằng 0.

**Quy tắc bootstrap:** quy tắc của nhóm (SE gốc) và quy tắc studentized của bàn giao chỉ khác nhau ở critical value (1,81 vs 1,72 với 4 phép so). Không verdict trong-panel nào đổi, trừ việc cận dưới LL ở Colosseum 2027 sát 0 (+0,0003).

**Phân phối lệch:**
- Ở Colosseum, trung vị gain IBS (0,009–0,012) nhỏ hơn nhiều so với trung bình (0,033–0,039). Lợi thế tập trung ở các MDP nơi B2 dự báo sai chiều kiểm duyệt.
- Trong generator, trung vị bằng khoảng 55% trung bình, ít lệch hơn.

## 6. Dịch chuyển phân phối và trần có thể đạt

**Khoảng cách tới meta-training** (trung vị khoảng cách 10-NN chuẩn hóa trong 18 thống kê B2; [`shift.csv`](../../artifacts/week8/cross_panel_reanalysis_2026-10-14/shift.csv)):
- Cohort 4/7/8: 0,7–1,1. Colosseum: 2,7–3,1. SPIBB: 1,8. BRIDGE: 6–10.
- Trong cùng nhóm, khoảng cách theo MDP **không tương quan** với gain (Spearman 0,06 cho IBS, cả trong lẫn ngoài).
- SPIBB khá gần meta-training theo B2 nhưng là panel khó nhất cho OEH. Dịch chuyển do behavior **không hiện ra** trong thống kê pilot chung.

**Trần nhiễu nhãn:**
- Cách ước lượng: luật bootstrap theo seed của chính nhãn, tức biết xác suất thành công thật của từng mức (một cận dưới lạc quan).
- IBS trần bằng 0,006–0,008 trong generator và 0,004–0,005 trên Colosseum/SPIBB ([`noise_floor.csv`](../../artifacts/week8/cross_panel_reanalysis_2026-10-14/noise_floor.csv)).
- OEH (0,047 nội bộ) còn xa trần. Ở cohort 8, OEH mới lấp được khoảng 20% khoảng cách B2 → trần. Sai số còn lại chủ yếu là **sai số dự báo**, không phải nhiễu của nhãn 10 seed.

## 7. Theo learner (gain IBS tương đối OEH vs B2)

| Panel | BC | CFQI | IQL |
|---|---:|---:|---:|
| Cohort 7 | 23% | 13% | 18% |
| Cohort 8 | 39% | 9% | 19% |
| Colosseum 2027 | 47% | 11% | 27% |
| Colosseum 2028 | 45% | 19% | 26% |
| SPIBB | 18% | 5% | 49% |
| BRIDGE Atari-5 | 53% | 4% | **−127%** |
| BRIDGE Expanded-5 | **−24%** | 46% | 5% |

- Gain tương đối lớn nhất ở BC và IQL, nhưng đây là các learner bị cắt 75–100%. Gain tuyệt đối của chúng nhỏ và chủ yếu là nhận ra ca bị cắt.
- CFQI, learner có nhiều nhãn hữu hạn nhất (khoảng 50%), có gain tương đối nhỏ nhất (5–19%) ở mọi panel Colosseum và nội bộ.

## 8. Đánh giá tổng thể

**Bằng chứng ủng hộ:**
1. Trong các họ MDP của generator, OEH dự báo phân phối nhu cầu dữ liệu tốt hơn mọi đối thủ cùng thông tin đã thử. Kết quả đã tái lập niêm phong ba lần với hiệu ứng đồng nhất (IBS khoảng 17%, LL khoảng 10%, I² = 0), kể cả trước B2 + legacy (cohort 7 và 8) và trước LCE, CB (cohort 8).
2. Trên kernel do người khác thiết kế, OEH tốt hơn thống kê chung về Brier (hai lần, Holm vẫn ý nghĩa). Lý do là OEH nhận ra ca "không học nổi trong x50" và giữ hiệu chỉnh khi B2 mất hiệu chỉnh.

**Bằng chứng chưa ủng hộ:**
3. Lợi thế log loss ngoài generator (gộp 4,9%, chứa 0).
4. Lợi thế so với bộ chẩn đoán legacy và ERI ngoài generator.
5. Khả năng đặt mức ngân sách cụ thể ngoài generator. Trên Colosseum, phân phối mức hữu hạn của OEH kém B2.
6. Bất kỳ lợi thế nào khi behavior đến từ quy trình khác (SPIBB), hoặc trên BRIDGE Atari.
7. Lợi thế từ head ensemble so với XGBoost trên cùng feature.

**Rủi ro về giá trị nội tại cần nêu trong bài:**
- Cohort 8 dùng lại template recipe của cohort 7; meta-training có cùng họ MDP. "Ba lần tái lập" là ba lần lấy mẫu từ **cùng một generator**.
- Ba cohort đầu thất bại và cấu hình hiện tại được chọn sau đó. Các lần tái lập sau đó dùng cấu hình cố định, nên giảm nhưng không xóa được rủi ro chọn lọc.
- Lợi thế ngoài generator gắn với kiểm duyệt (73% nhãn là `>x50`). Một phương pháp chỉ giỏi phát hiện "không học nổi" sẽ trông tốt trên Brier gộp.
- Trên SPIBB, panel có behavior độc lập, BC và IQL không bao giờ học được, nên chỉ CFQI cho thông tin. Panel này yếu về độ phân biệt giữa các phương pháp.

**Hệ quả cho bài và thesis:**
- Câu "beats generic statistics on Colosseum" chỉ nên gắn với **integrated Brier**.
- Nên nói rõ lợi thế ngoài generator là **phát hiện ca bị cắt và hiệu chỉnh**, còn việc đặt mức thì kém B2.
- Nên thêm kết quả tổng hợp chéo panel: trong generator 16,8% / 10,1% với I² = 0; ngoài generator 16,0% (IBS) và 4,9% (LL, không ý nghĩa).
- Nên ghi chú về sàn log-loss khi so với XGBoost-CF.

**Hướng cải thiện có cơ sở từ phân tích này** (chưa làm):
1. Head lai: P(>x50) từ OEH, phân phối mức hữu hạn từ B2 + legacy. Trên Colosseum, hoán đổi kiểu này đã tốt hơn OEH. Phải chọn trên dữ liệu đã mở và kiểm định trên panel mới.
2. Thay giả định reward `unseen_zero` của khối horizon.
3. Đưa behavior kiểu SPIBB vào meta-training.

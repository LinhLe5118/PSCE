# Panel Colosseum: review code bàn giao, kết quả và phân tích post hoc (14/10/2026)

**Trạng thái:**
- Kết quả đăng ký trước trên 72 MDP Colosseum là **GO** (OEH vs B2 matched, OEH vs B2 + độ khó Colosseum).
- Mọi phân tích khác trong tài liệu này là **post hoc**: nhãn Colosseum đã mở lúc 02:05 UTC ngày 14/10.
- Không sửa artifact niêm phong nào, kể cả của cohort 7.

**Nguồn:**
- Gói bàn giao `oeh_colosseum_handover.zip`, chạy trên máy phụ ngày 13–14/10 với bản copy repo @ `d7134ad`.
- Code driver được giữ nguyên trong [`benchmarks/colosseum/`](../../benchmarks/colosseum/). Thay đổi duy nhất: biến `OEH_WORKSPACE` trong `common.py`.
- README gốc và kế hoạch của người chạy nằm cạnh driver: [`HANDOVER_README_2026-10-14.md`](../../benchmarks/colosseum/HANDOVER_README_2026-10-14.md), [`KE_HOACH_THUC_NGHIEM_2026-10-14.md`](../../benchmarks/colosseum/KE_HOACH_THUC_NGHIEM_2026-10-14.md).

| Thư mục | Nội dung |
|---|---|
| [`colosseum_confirm_2026-10-14/`](../../artifacts/week8/colosseum_confirm_2026-10-14/) | Lần xác nhận 72 MDP: protocol, seal, dự báo, nhãn, outcome 64.800 job, điểm, amendment 1 (bộ comparator cohort 7), 360 nhóm OEH-B, log chạy |
| [`colosseum_exploratory_2026-10-13/`](../../artifacts/week8/colosseum_exploratory_2026-10-13/) | Lần thăm dò 32 MDP (seed sampler 2026) |
| [`colosseum_meta_hardness_2026-10-13/`](../../artifacts/week8/colosseum_meta_hardness_2026-10-13/) | Độ khó Colosseum của 7 cohort meta-training (cho comparator B2 + độ khó) và tái lập cohort 7 trên máy phụ |
| [`colosseum_posthoc_2026-10-14/`](../../artifacts/week8/colosseum_posthoc_2026-10-14/) | Phân tích post hoc của tài liệu này: kế hoạch khóa trước, dự báo niêm phong trước khi đọc nhãn, điểm |

## 1. Thiết kế (tóm tắt)

- **MDP:** 72 MDP lấy bằng sampler gốc của tác giả Colosseum (Conserva & Rauber, NeurIPS 2022).
  - Hàm `sample_mdp_parameters`, seed 2027, repo `MichelangeloConserva/Colosseum` @ `981b344`.
  - 4 họ: RiverSwim, DeepSea, FrozenLake, SimpleGrid, mỗi họ 18 MDP.
  - Bộ lọc: 16 ≤ |S||A| ≤ 256, kernel ngẫu nhiên, bộ lọc dynamic-range của dự án.
  - Đã loại mọi kernel trùng với lần thăm dò.
- **Phần của nhóm, giữ nguyên:** generator log, 5 behavior, pilot x10, BC/CFQI/IQL, quy tắc nhãn `suffix_event` (≥8/10 seed, NR ≥ 0,95).
- **Forecaster:** `oeh_final` (108 cột), fit trên đúng panel meta-training 720 MDP / 10.800 dòng của cohort 7.
- **Thứ tự thời gian (UTC):**
  1. Protocol khóa lúc 18:34 ngày 13/10.
  2. Export MDP bắt đầu ngay sau đó.
  3. Dự báo niêm phong lúc 01:55 ngày 14/10.
  4. Nhãn có lúc 02:05.
  - Bằng chứng là log và hash do chính người chạy ghi; không có commit hay tag ngoài trước outcome.
- **Gate chính:** OEH vs B2 matched, cận dưới đồng thời max-t một phía 95% > 0 cho cả integrated Brier (IBS) lẫn log loss 7 lớp (LL).

## 2. Review code `driver/` (bước 1 của bàn giao)

**Kiểm tra lại độc lập (đều đạt):**
- Hash protocol/seal/amendment khớp.
- Tính lại IBS/LL từ `colosseum_predictions_preoutcome.csv` + `colosseum_labels.csv` ra đúng `score/summary.csv`.
- 1.080/1.080 nhãn khớp `compare_difficulty.suffix_event` chạy lại trên `colosseum_outcomes.csv`.
- Cờ `success` đúng ngưỡng 0,95.
- Phân tích post hoc (mục 4) tái tạo dự báo OEH/B2 niêm phong của Colosseum với sai khác ≤ 1,2e-7.
- Arm B2 + legacy của mục 5 tái tạo `prior_oeh_matched` của amendment 1 với sai khác ≤ 3,2e-8. Hai đường code độc lập cho cùng số.

**Phát hiện:**

| # | Mức | Phát hiện | Hệ quả | Xử lý |
|---|---|---|---|---|
| R1 | Cao | `score_colosseum.py` và `forecast_comparators.py` **tính lại SE trong từng mẫu bootstrap**. Quy tắc của nhóm (`run_oeh_120_unseen.analyze`, dùng cho cohort 7) chia cho **SE gốc**. Báo cáo bàn giao viết "cùng quy tắc với bài báo", điều này không đúng. | Với quy tắc của nhóm, verdict không đổi nhưng cận dưới LL của OEH vs B2 giảm từ +0,0049 xuống **+0,0003** (critical 1,72 → 1,81). GO rất mỏng. | Bảng ở mục 3 báo cáo cả hai quy tắc. Bài báo dùng quy tắc của nhóm. |
| R2 | Cao | Báo cáo ghi "0/360 nhóm thiếu posterior draw". Thực tế **42/360 nhóm OEH-B có 0 draw được chấp nhận**, 1 nhóm chỉ có 9/64 draw. RiverSwim chiếm 29/90 nhóm, DeepSea 10, FrozenLake 3. Lần thăm dò ghi đúng (26/160). | Nhóm 0 draw nhận luật mặc định "bị cắt" (p₆ ≈ 0,96). Xem mục 6: luật này **không** gây ra thất bại ở RiverSwim. | Sửa con số trong tài liệu dự án. Báo cáo cùng kiểu với BRIDGE ("fallback do không có posterior draw"). |
| R3 | Trung bình | Bản sửa expectile được dùng cả khi **tạo nhãn** IQL trên 11/72 MDP SimpleGrid (`expectile_fallbacks_so_far` > 0), không chỉ ở 5 nhóm feature như báo cáo ghi. | Đây là bản sửa nhóm đã ghi nhận, bitwise-identical khi bản gốc chạy được. Nhưng nhãn IQL của 11 MDP phụ thuộc vào nó. | Ghi vào giới hạn. |
| R4 | Trung bình | Colosseum có reward ngẫu nhiên. `colosseum_export.py` chỉ lấy **reward kỳ vọng** `mdp.R` (S×A) rồi broadcast sang R(s,a,s'). γ = 0,95, horizon 100, không có trạng thái kết thúc. | MDP dùng kernel chuyển trạng thái của tác giả, nhưng nhiễu reward bị bỏ. | Ghi vào bài là "Colosseum transition kernels with mean rewards". |
| R5 | Trung bình | Amendment 1 khóa 22 phép so nhưng chỉ chạy 18: thiếu `b2_plus_tq_saco_original` và `published_metrics_only`, vì máy phụ không chạy code `ml-jku/OfflineRL`. Họ phép so vẫn chứa B2 + BWD, dù T1 v1.3.0 đã loại BWD khỏi so sánh đang dùng. | Thêm hai phép so chỉ làm critical tăng, nên verdict tier 1 NO-GO không đổi. | Báo cáo như amendment. Dòng BWD đánh dấu "đã loại". |
| R6 | Thấp | 355/360 nhóm OEH-B chạy với OpenBLAS 16 luồng/worker, 5 nhóm chạy 1 luồng. Không thể kiểm parity vì bundle pilot không được gửi kèm. Thời gian OEH-B trên Colosseum chậm hơn cohort 7 khoảng 5–10 lần ở cùng |S||A| (tranh CPU). | Seed OEH-B cố định nên sai khác được kỳ vọng chỉ ở mức làm tròn. Thời gian tuyệt đối không so được giữa các máy. | Ghi vào giới hạn. Chỉ đọc xu hướng chi phí trong cùng panel. |
| R7 | Thấp | Docstring `colosseum_export.py` đã cũ (seed 2026, 9 MDP/họ, "first-5" cut), trong khi giá trị thật lấy từ biến môi trường. Danh sách họ có 7 họ, 3 họ MiniGrid/Taxi bị lọc vì kích thước. `hardness_arrays.py` tính bằng float32. | Không ảnh hưởng kết quả. Train và test dùng cùng hàm độ khó. | Ghi chú, không sửa code đã chạy. |
| R8 | Thấp | Môi trường máy chính có scikit-learn 1.7.2, trong khi `requirements-difficulty.txt` ghim 1.9.1. | Tái tạo dự báo niêm phong vẫn đạt ≤ 1,2e-7. | Ghi phiên bản trong `prediction_seal.json`. |

**Kết luận review:**
- Pipeline dùng đúng code của nhóm cho sinh dữ liệu, feature, forecaster và nhãn. Kết quả chính tái lập được.
- Hai con số trong báo cáo bàn giao phải sửa: quy tắc max-t (R1) và số nhóm 0 draw (R2).
- Đủ điều kiện đưa Colosseum vào bài như **một panel riêng**, kèm các giới hạn R3–R6.

## 3. Kết quả đăng ký trước (72 MDP, 1.080 dòng)

| Phép so | Metric | OEH / đối chứng | Gain | Cận dưới (quy tắc bàn giao) | Cận dưới (quy tắc nhóm) |
|---|---|---|---:|---:|---:|
| OEH vs B2 matched | IBS | 0,1010 / 0,1342 | 24,8% | +0,0160 | **+0,0151** |
| OEH vs B2 matched | LL | 0,8903 / 0,9815 | 9,3% | +0,0049 | **+0,0003** |
| OEH vs B2 + độ khó Colosseum | IBS | 0,1010 / 0,1422 | 29,0% | +0,0234 | +0,0225 |
| OEH vs B2 + độ khó Colosseum | LL | 0,8903 / 1,0227 | 12,9% | +0,0446 | +0,0399 |

Critical: 1,72 (bàn giao), 1,81 (nhóm). Verdict theo cả hai quy tắc: **GO / GO**. Nguồn: [`maxt_contrasts.csv`](../../artifacts/week8/colosseum_posthoc_2026-10-14/maxt_contrasts.csv), họ `registered_primary_4`.

**Amendment 1 (post hoc, 18 phép so của bộ cohort 7):** tier 1 NO-GO theo cả hai quy tắc. Theo quy tắc nhóm (critical 2,50), các cận dưới âm là:
- B2 matched LL: −0,034.
- B2 + ERI: IBS −0,005, LL −0,092. Gain chỉ 15,7% IBS và 3,2% LL.
- B2 + BWD LL: −0,033. BWD đã bị loại theo T1 v1.3.0.

Tier 2 (EH oracle) GO. Cohort 7 có tier 1 GO, nên trên Colosseum, **B2 + ERI là đối thủ mà OEH không vượt có ý nghĩa**.

## 4. Đối chứng theo hợp đồng so sánh hiện hành: B2 và XGBoost-CF (post hoc)

**Cách chạy:**
- XGBoost-CF/18 và /108 dùng đúng code `published_survival_120_*`: theo learner, tầng primary của meta-training, 100 cây sâu 3, seed 42.
- Cùng đường code tái tạo dự báo cohort 7 đã lưu (sai khác 1e-16).

**Bảng IBS / LL theo learner.** Nguồn: [`summary.csv`](../../artifacts/week8/colosseum_posthoc_2026-10-14/summary.csv). Các dòng có `>x50` lần lượt là BC 91,4%, CFQI 51,4%, IQL 75,6%.

| Learner | B2 matched | XGBoost-CF/18 | XGBoost-CF/108 (kiểm kiến trúc) | B2 + ERI | OEH |
|---|---:|---:|---:|---:|---:|
| BC | 0,0946 / 0,503 | 0,1030 / 0,854 | 0,0573 / 0,599 | 0,0880 / 0,474 | **0,0498 / 0,336** |
| CFQI | 0,1710 / 1,473 | 0,1728 / 1,807 | **0,1451 / 1,342** | 0,1646 / 1,427 | 0,1531 / 1,458 |
| IQL | 0,1371 / 0,968 | 0,1456 / 2,360 | 0,1065 / 1,991 | 0,1070 / **0,858** | **0,1001** / 0,877 |
| Gộp | 0,1342 / 0,982 | 0,1405 / 1,674 | 0,1029 / 1,311 | 0,1198 / 0,920 | **0,1010 / 0,890** |

**Họ post hoc chính** (OEH vs {B2, XGBoost-CF/18} × {IBS, LL}, quy tắc nhóm, critical 2,07):
- IBS dương với cả hai đối chứng: cận dưới +0,0125 vs B2, +0,0162 vs XGBoost-CF/18.
- LL vs XGBoost-CF/18 dương: cận dưới +0,365.
- **LL vs B2 âm: −0,012.**

**Kiểm soát kiến trúc:** XGBoost-CF/108 dùng chính 108 cột của OEH. Brier của nó gần như bằng OEH: chênh 0,0019, cận dưới −0,0021, OEH thắng 30/72 MDP. Ở CFQI nó còn tốt hơn OEH (ΔIBS −0,0081, khoảng loại trừ 0). Giống kết luận ở cohort 7: một phần lớn lợi thế về Brier đến từ feature, không đến từ head của OEH.

Log loss của XGBoost-CF bị chi phối bởi một số dòng có xác suất 0 cho lớp thật (26 dòng ở /18, 17 dòng ở /108):
- Với sàn 1e-15 (sàn của bàn giao): /18 = 1,674, /108 = 1,311.
- Với sàn 1e-12 (sàn dùng ở bài): /18 = 1,507, /108 = 1,202.

## 5. Ablation factorial 2³ và Shapley theo khối (post hoc)

**Định nghĩa khối** theo lineage M1:
- P = mô phỏng hậu nghiệm trực tiếp (G2, G3, G6).
- H = OEH-H / risk (G4, G5).
- L = legacy OEH3/LCOEH4 và context (G7, G8).

**Nguồn gốc các arm:**
- 5 arm của M1 được định nghĩa từ 13/10 16:20 UTC, trước khi có nhãn Colosseum.
- 3 arm mới (B2+P, B2+H, B2+L) được định nghĩa trong [`analysis_plan.json`](../../artifacts/week8/colosseum_posthoc_2026-10-14/analysis_plan.json), sau khi nhãn đã mở.

**Cách tính:**
- Mọi arm dùng cùng `matched_forecast`, cùng fold và tuning.
- Dự báo được hash trước khi đọc nhãn. Audit hook chặn mọi file outcome của Colosseum và cohort 7 trong pha dự báo.
- Shapley tính theo MDP. Khoảng 95% là percentile của 10.000 bootstrap theo MDP, mang tính mô tả.

| Phạm vi | Metric | Gain B2 → full | φ(P) [95%] | φ(H) [95%] | φ(L) [95%] |
|---|---|---:|---:|---:|---:|
| Gộp | IBS | 0,0332 | 0,0103 [−0,0005; 0,0209] | 0,0115 [0,0029; 0,0205] | 0,0114 [0,0056; 0,0174] |
| Gộp | LL | 0,0912 | 0,0405 [−0,0205; 0,0963] | −0,0038 [−0,0569; 0,0475] | **0,0545 [0,0230; 0,0861]** |
| RiverSwim | LL | −0,1357 | −0,1080 [−0,2859; 0,0650] | **−0,1507 [−0,2623; −0,0400]** | +0,1230 [0,0587; 0,1939] |
| CFQI | LL | 0,0149 | 0,0350 [−0,0815; 0,1443] | **−0,0929 [−0,1581; −0,0294]** | 0,0728 [0,0188; 0,1274] |
| BC | LL | 0,1673 | 0,0465 [0,0228; 0,0702] | 0,1032 [0,0590; 0,1504] | 0,0176 [−0,0317; 0,0656] |

**Đọc kết quả** (đầy đủ trong [`block_shapley.csv`](../../artifacts/week8/colosseum_posthoc_2026-10-14/block_shapley.csv)):
- Với Brier, ba khối chia đều lợi ích, mỗi khối khoảng 1/3. Khoảng của P vừa chạm 0.
- Với log loss, **khối legacy L là đóng góp duy nhất có khoảng loại trừ 0**.
  - B2 + L chính là `prior_oeh_matched` ("OEH đời trước").
  - Arm này có LL gộp tốt nhất trong mọi arm (0,8431 so với OEH 0,8903), nhưng Brier kém hơn (0,1048 so với 0,1010).
- Riêng từng khối, mỗi khối đều đã cải thiện B2 rõ về Brier: P −0,0252, H −0,0290, L −0,0295. Điều này khớp với kết luận M1 rằng các khối **chồng lấn thông tin**.
- Mô phỏng hậu nghiệm (P) không có đóng góp riêng có ý nghĩa ở mức gộp. Nó có ý nghĩa ở BC, IQL-Brier, DeepSea và FrozenLake.
- Kịch bản xấu đã ghi trước trong kế hoạch bàn giao mục 8 ("Shapley của P ≈ 0") **xảy ra một phần**. Nhưng đây là dữ liệu đã mở, nên chưa phải kết luận cuối.

## 6. Vì sao RiverSwim thất bại

**Giả thuyết đã ghi trước trong kế hoạch: không được ủng hộ.** Giả thuyết là luật mặc định "bị cắt" của nhóm 0 draw gây thất bại. Số liệu từ [`zero_draw_diagnosis.csv`](../../artifacts/week8/colosseum_posthoc_2026-10-14/zero_draw_diagnosis.csv):
- Ở 87 dòng RiverSwim thuộc nhóm 0 draw, 98,9% nhãn đúng là `>x50`, và OEH dự báo tốt (LL 0,187).
- Thất bại nằm ở 183 dòng **có** draw:
  - 39,9% nhãn là x1.
  - OEH có LL 1,767, so với B2 1,620 và B2 + L 1,169.
  - OEH gán dưới 2% cho lớp thật ở 19,1% số dòng.

**Theo Shapley:**
- Khối OEH-H (G4, G5, tính trên mô hình posterior mean với reward prior `unseen_zero`) làm LL của RiverSwim tệ đi có ý nghĩa.
- Khối P cũng làm tệ đi nhưng khoảng chứa 0.
- Khối legacy giúp.

Giải thích hợp lý nhưng chưa kiểm chứng: ở RiverSwim, pilot ít khi chạm reward lớn ở đầu sông, nên mô hình posterior đánh giá thấp khả năng học. Trong khi đó learner vẫn đạt ngay ở x1 nhờ dữ liệu huấn luyện riêng. Muốn kiểm chứng cần một thí nghiệm với reward prior khác, trên dữ liệu chưa mở.

## 7. Giá trị ra quyết định (A3, post hoc)

**Quy tắc** (khóa trong kế hoạch trước khi tính):
- Chọn mức nhỏ nhất x_k có P(E ≤ k) ≥ τ, không có mức nào thỏa thì bỏ qua.
- Chi phí khi thu thập: x_k, cộng λ nếu E > k. Chi phí khi bỏ qua: λ nếu E < 6, ngược lại 0.
- Regret = chi phí − chi phí oracle (x_E, hoặc 0 khi E = 6).
- Đơn vị là bội số của |S||A|. τ ∈ {0,5; 0,8; 0,9}, λ ∈ {50; 100}.
- Nguồn: [`decision_regret.csv`](../../artifacts/week8/colosseum_posthoc_2026-10-14/decision_regret.csv), [`decision_contrasts.csv`](../../artifacts/week8/colosseum_posthoc_2026-10-14/decision_contrasts.csv).

**Kết quả:**
- **Cohort 7 (84 MDP, mô tả):** OEH có regret thấp hơn B2 và B2 + ERI ở cả 6 tổ hợp (τ, λ), khoảng loại trừ 0. Ví dụ τ = 0,8, λ = 50: OEH 6,58, B2 7,82.
- **Colosseum:** OEH thấp hơn B2 rõ ở τ = 0,5 (λ = 50: 11,15 vs 15,90; khoảng [−6,89; −2,56]). Ở τ = 0,8 và 0,9, khoảng chứa 0.
- **Luật "luôn bỏ qua" là baseline mạnh**, vì 73% nhãn là `>x50`. Với λ = 50, không phương pháp nào thắng nó có ý nghĩa trên Colosseum (OEH 10,99 vs 10,75 ở τ = 0,8). Với λ = 100, OEH thắng nó: −4,11 [−7,67; −0,75].
- XGBoost-CF/108 có regret thấp nhất ở vài tổ hợp, cùng hướng với kết luận kiểm soát kiến trúc ở mục 4.

**Hệ quả cho bài:** con số "tiết kiệm X% dữ liệu" chưa được bằng chứng ủng hộ. Phải khóa (τ, λ) và baseline "bỏ qua" trong protocol của cohort mới trước khi báo cáo.

## 8. Chi phí theo kích thước

Nguồn: [`cost_by_size.csv`](../../artifacts/week8/colosseum_posthoc_2026-10-14/cost_by_size.csv). Đơn vị là trung vị giây OEH-B mỗi cặp MDP–behavior, với 64 draw.

| Panel | |S||A| ≤ 64 | 64–128 | 128–256 | 304–975 | 1.652–3.102 | 6.888–11.061 |
|---|---:|---:|---:|---:|---:|---:|
| Cohort 7 (máy chính) | 5,1 | 16,5 | 33,1 | | | |
| Colosseum (máy phụ, BLAS đa luồng) | 51,8 | 85,7 | 197,9 | | | |
| BRIDGE Atari-5 | | | | 136 | 2.159 | |
| BRIDGE mở rộng-5 | | | | | | 8.822 |

- Chi phí tăng nhanh hơn tuyến tính theo |S||A|: posterior Dirichlet dày trên S×A×S cộng với replay learner.
- Trong 155 MDP của BRIDGE, **125 bị loại** vì vượt giới hạn mô hình dày / horizon, xem [BRIDGE expanded](BRIDGE_EXPANDED_EXECUTION_2026-10-08.md).
- Colosseum chỉ giữ 4/7 họ, vì 3 họ MiniGrid/Taxi vượt |S||A| ≤ 256.

## 9. Claim được phép sau panel này

| Không viết | Viết |
|---|---|
| "OEH dự báo chính xác lượng dữ liệu cần thiết" | "OEH phát hiện tốt khi nào một learner không đạt trong x50 (AUC 0,915 trên 72 MDP Colosseum). Ước lượng mức ngân sách cụ thể khi đạt được không tốt hơn B2." |
| "Vượt mọi comparator khi chuyển giao" | "Trên 72 MDP Colosseum, OEH vượt B2 matched theo protocol khóa trước (cận dưới log loss rất sát 0). So với B2 + ERI, chênh lệch không có ý nghĩa." |
| "Lợi thế đến từ mô phỏng hậu nghiệm" | "Các khối tín hiệu chồng lấn. Trên Colosseum, phần lợi log loss có ý nghĩa đến từ khối chẩn đoán legacy, và khối horizon làm hại RiverSwim (post hoc)." |
| "OEH tiết kiệm dữ liệu" | Không viết, cho tới khi có phân tích quyết định được khóa trước trên cohort mới. |

## 10. Chưa làm và cần quyết định của nhóm trưởng

Các mục giai đoạn A trong kế hoạch bàn giao **chưa chạy**:
- A4, LCE trên Colosseum: cần sinh lại bundle pilot, tức chạy lại export Colosseum (môi trường Python 3.9) và `run_curve`.
- A5, leave-one-family-out trên 720 MDP.

Giai đoạn B–D cần quyết định thiết kế trước khi chạy, nên không tự làm:
- OEH-S (làm mềm xác suất), OEH-2T, baseline concentrability.
- Cohort mới niêm phong với kiểm định phân cấp. Cỡ mẫu khuyến nghị ≥ 120 MDP vì cận dưới log loss vs B2 rất sát 0.
- SPIBB, TQ/SACo gốc trên Colosseum.

Kết quả ở mục 5–6 gợi ý ưu tiên cho cohort mới:
1. Kiểm reward prior `unseen_zero` của OEH-H trên RiverSwim.
2. Đưa B2 + L (`prior_oeh_matched`) vào họ phép so chính như một đối thủ rẻ, vì nó không cần posterior.

## 11. Tái lập

```bash
python -m pytest tests/test_colosseum_posthoc.py -q   # 7 test
python -m scripts.analysis.colosseum_posthoc_20261014 freeze   --output <thư mục mới>
python -m scripts.analysis.colosseum_posthoc_20261014 forecast --output <thư mục mới>   # ~10 phút
python -m scripts.analysis.colosseum_posthoc_20261014 score    --output <thư mục mới>
```

- Pha `forecast` dừng nếu không tái tạo được dự báo niêm phong: Colosseum 1e-6, XGBoost-CF cohort 7 1e-6.
- Muốn chạy lại toàn bộ pipeline Colosseum (khoảng 3,4 giờ), xem [`benchmarks/colosseum/README.md`](../../benchmarks/colosseum/README.md).

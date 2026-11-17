# Báo cáo kiểm định xác nhận OEH trên Colosseum (MDP mới) — 2026-10-14

## Thiết kế (khóa trước khi sinh MDP)
- Protocol: `results/confirm/protocol_colosseum.json`, SHA-256 `dd9771576d50c99ba37ceffbd68f5195164b5b41356b03baf20e5d3677b4d3b8`.
- 72 MDP Colosseum mới (RiverSwim, DeepSea, FrozenLake, SimpleGrid, mỗi họ 18), sampler của tác giả, seed 2027; loại mọi kernel trùng với 32 MDP thăm dò. 114/120 ứng viên qua bộ lọc dynamic-range; lấy 18 đầu tiên mỗi họ theo thứ tự sampler.
- OEH current (`oeh_final`, 108 trường, A2, 64 draws) và forecaster huấn luyện trên panel niêm phong 7 cohort (720 MDP / 10.800 dòng) — không chỉnh gì sau lần thăm dò.
- Dự báo niêm phong lúc 08:55:17 (SHA `200871d8…`), learner chạy sau đó (outcome lúc 09:05:02); hash được kiểm tra lại ở bước nhãn và chấm điểm.
- Gate chính: OEH vs B2 matched — cả integrated Brier lẫn log-loss 7 lớp phải có mean gain > 0 và cận dưới đồng thời (max-t, 10.000 bootstrap toàn-MDP, một phía 95%) > 0.

## Kết quả chính (72 MDP, 1.080 dòng dự báo)

| Phương pháp | Integrated Brier | Log-loss 7 lớp |
|---|---|---|
| **OEH current** | **0.1010** | **0.8903** |
| B2 matched | 0.1342 | 0.9815 |
| B2 + độ khó Colosseum | 0.1422 | 1.0227 |
| (tham chiếu) hằng số = phân phối nhãn thật | 0.1517 | 1.0196 |

| So sánh | Metric | Gain tương đối | Cận dưới đồng thời | MDP tốt hơn / tệ hơn |
|---|---|---|---|---|
| OEH vs B2 | Brier | +24.8% | **+0.0160** | 50 / 22 |
| OEH vs B2 | Log-loss | +9.3% | **+0.0049** | 46 / 26 |
| OEH vs B2+hardness | Brier | +29.0% | **+0.0234** | 53 / 19 |
| OEH vs B2+hardness | Log-loss | +12.9% | **+0.0446** | 49 / 23 |

**Verdict: GO (vs B2) và GO (vs B2 + hardness).** Critical max-t = 1.72.

## Phân tích mô tả (không đổi verdict)
- Theo họ: OEH thắng rõ ở SimpleGrid, DeepSea, FrozenLake; **RiverSwim: Brier gần hòa (0.157 vs 0.161), log-loss OEH tệ hơn (1.258 vs 1.122)**.
- Theo learner: OEH tốt hơn ở cả BC, IQL; CFQI chỉ nhỉnh nhẹ (Brier 0.153 vs 0.171, log-loss 1.458 vs 1.473).
- Case hữu hạn (294 dòng) vs bị cắt >x50 (786 dòng): lợi thế đến gần như hoàn toàn từ case bị cắt (Brier 0.045 vs 0.090). Trên case hữu hạn ba phương pháp ngang nhau (Brier 0.250 vs 0.253; log-loss OEH 2.517 vs 2.449, hơi tệ hơn).
- Xếp hạng 3 learner trong cùng MDP-behavior (160 nhóm có khác biệt): Spearman OEH 0.722 vs B2 0.463.
- Tương quan với yêu cầu thật: OEH expected event 0.66–0.83; độ khó Colosseum |ρ| ≤ 0.50.
- Posterior draws: 0/360 nhóm thiếu draw (lần thăm dò 26/160).
- Tỷ lệ bị cắt: BC 91%, IQL 76%, CFQI 51%.

## So với lần thăm dò (32 MDP)
| | Thăm dò | Xác nhận |
|---|---|---|
| OEH vs B2 Brier: gain / cận dưới | +21.8% / +0.003 | +24.8% / +0.016 |
| OEH vs B2 log-loss: gain / cận dưới | +9.5% / −0.021 | +9.3% / +0.005 |
| Verdict vs B2 | NO-GO | **GO** |
Hiệu ứng ổn định (gain gần như giống hệt); thêm MDP làm cận dưới vượt 0 như ước tính power.

## Sự cố và cách xử lý
- Phiên trước bị ngắt giữa bước OEH-B (355/360) → chạy tiếp (runner tự bỏ qua nhóm đã xong).
- 5 nhóm của `confirm_colosseum_deepsea_002` bị chậm ~4000 lần do OpenBLAS 16 luồng/worker tranh CPU → đặt `OPENBLAS_NUM_THREADS=1`, 5 nhóm xong trong 53 giây. Không đổi phương pháp/MDP/protocol.
- 5 nhóm feature cần expectile fallback (bản sửa đã ghi nhận của nhóm), như lần thăm dò.

## Thời gian tính toán (không tính lúc bị treo)
Xuất MDP 103 phút (bốc 600 mẫu/họ + loại trùng) · độ khó 17 giây · sinh dữ liệu 2 phút · OEH-B ~77 phút · feature 10 phút · niêm phong 2 phút · learner 10 phút (64.800 job) · chấm điểm 3 giây → **~3,4 giờ**.

## Giới hạn
- Vẫn là Colosseum tabular (4 họ, |S||A| ≤ 256), cùng generator dữ liệu/behavior/learner của dự án.
- Lợi thế chủ yếu là nhận biết case "không đạt được trong x50"; dự báo ngân sách cụ thể khi đạt được không tốt hơn B2.
- RiverSwim là họ OEH yếu.

---

## Bổ sung (post hoc): bộ comparator đầy đủ của cohort 7
Amendment: `protocol_colosseum_amendment_1_comparators.json` (SHA `e420e4ae…`), ghi trước khi tính; kết quả learner đã mở nên đây là phân tích **post hoc**. Định nghĩa comparator, field, cách fit lấy nguyên từ cohort 7 (`run_oeh_final_120.all_forecasts`); `all_forecasts` tái tạo dự báo OEH/B2 đã niêm phong (|diff| ≤ 1.1e-16).

Chưa chạy: `b2_plus_tq_saco_original`, `published_metrics_only` (cần code gốc ml-jku/OfflineRL @8cc8eec — bước chạy code ngoài bị hệ thống quyền chặn, chờ người dùng cho phép). Hai comparator này thuộc tier 2 hoặc gộp — **không thay đổi verdict tier 1** (thêm phép so chỉ làm critical value tăng).

18 phép so đồng thời, critical max-t 2.25 (cohort 7 dùng 22):

| Comparator | Tier | Brier gain / cận dưới | Log-loss gain / cận dưới |
|---|---|---|---|
| B2 matched | 1 | +24.7% / +0.011 | +9.3% / **−0.022** |
| **B2 + ERI** | 1 | +15.7% / **−0.003** | +3.2% / **−0.080** |
| B2 + BWD | 1 | +23.9% / +0.010 | +9.4% / **−0.020** |
| train_frequency | 1 | +31% / +0.025 | +18% / +0.075 |
| coverage / quality / cov+qual adaptation | 1 | +38–44% / > 0 | +47–52% / > 0 |
| B2 generic | 1 | +30% / +0.019 | +42% / +0.396 |
| B2 + EH oracle | 2 | +30% / +0.019 | +15% / +0.035 |

**Verdict: tier 1 NO-GO; tier 2 (phần có) GO; tất cả NO-GO.**
So với cohort 7 (tier 1 GO): trên Colosseum, ERI (Swazinna et al. 2021) là đối thủ mạnh — B2+ERI tốt hơn B2 rõ rệt (Brier 0.120 vs 0.134) và OEH không vượt nó có ý nghĩa.

Báo cáo không gate (theo protocol nhóm): `prior_oeh_matched` (OEH đời trước: OEH3/LCOEH4 + context) Brier 0.105 / log-loss **0.843** — log-loss tốt hơn OEH current (0.890); thắng ở RiverSwim (0.829 vs 1.257), CFQI, và case hữu hạn.

## Bổ sung: case hữu hạn (294 nhãn 0–5) vs case bị cắt (786 nhãn 6)

| | AUC phát hiện "không học nổi" | Spearman mức (hữu hạn) | Đúng ±1 mức | Log-loss có điều kiện |
|---|---|---|---|---|
| OEH current | **0.915** | 0.497 | 52% | 2.005 |
| OEH đời trước | 0.903 | **0.506** | **62%** | **1.659** |
| B2 + ERI | 0.881 | 0.413 | 52% | 1.836 |
| B2 | 0.855 | 0.390 | 53% | 1.880 |
| B2 + EH oracle | 0.849 | 0.379 | 52% | 2.022 |

- OEH current tốt nhất ở phát hiện; ở ước lượng mức cụ thể thì OEH đời trước tốt hơn.
- Quá tự tin trên case hữu hạn là chung cho mọi phương pháp (độ tin cậy lớp cao nhất ~0.46 vs độ chính xác ~0.28), không riêng OEH.
- File: `score_comparators/{summary,contrasts,mdp_scores,finite_vs_censored}.csv`, `verdict.json`.

# Kế hoạch viết paper + thesis — cập nhật 14/10/2026

Bản này thay cho [kế hoạch 13/10](KE_HOACH_VIET_PAPER_THESIS_2026-10-13.md). Bản 13/10 được giữ làm lịch sử. Lý do cập nhật:
- Có thêm bốn panel niêm phong: cohort 8, Colosseum 2027 và 2028, SPIBB.
- Có [phân tích lại chéo panel](REANALYSIS_ALL_PANELS_2026-10-14.md).
- Ba quyết định của nhóm trưởng ngày 14/10:
  1. Sửa claim theo phân tích lại (đã áp dụng vào `paper/` và thesis v2).
  2. Mở PR #53.
  3. **Không** viết lại lịch sử git (giữ nguyên các commit cũ do Codex làm tác giả, vì các seal E1 và các file validation đang trích dẫn hash của chúng).

Tài liệu này không đổi phương pháp, protocol hay verdict nào.

## 1. Trạng thái bằng chứng

| Panel | Vai trò trong bài | Kết quả | Nguồn |
|---|---|---|---|
| Cohort 7 (84 MDP) | Kết quả xác nhận chính | 22/22 cận dưới dương; IBS −15,97%, LL −10,22% vs B2 | `artifacts/oeh_confirmatory_unseen_120_2026-10-04` |
| Cohort 8 nội bộ (168 MDP) | Tái lập niêm phong thứ hai, gatekeeping | Bước 1–2 GO, gồm cả XGBoost-CF/18, B2 + legacy, LCE, CB; OEH-S/2T không thay OEH | [biên bản cohort 8](COHORT8_SPIBB_DEVELOPMENT_2026-10-14.md) |
| Cohort 4 (84 MDP) | Tái lập lịch sử, cùng cấu hình | Tier 1 GO | `artifacts/oeh_final_unseen_120_2026-10-02` |
| Colosseum 2027 (72) | Kernel người khác, gate 2 phép so | GO vs B2 (cận dưới LL +0,0003); bộ 22 phép so post hoc NO-GO, TQ/SACo gốc nhỉnh hơn OEH | [biên bản Colosseum](COLOSSEUM_EXTERNAL_PANEL_2026-10-14.md) |
| Colosseum 2028 (72) | Kernel người khác, gatekeeping | Bước 1 GO, bước 2 NO-GO | biên bản cohort 8 |
| SPIBB (108) | MDP và behavior của người khác | Bước 1 NO-GO | biên bản cohort 8 |
| BRIDGE Atari-5, Expanded-5 | MDP công bố, phụ thuộc learner | Atari gộp: OEH kém B2; Expanded: OEH tốt hơn, khoảng chứa 0 | `BRIDGE_ALL_RUNS_RESULTS_2026-10-13.md` |
| Tổng hợp random-effects | Một câu và một hình (nếu đủ chỗ) | Trong generator 16,8% IBS / 10,1% LL, I² = 0; ngoài generator 16,0% IBS / 4,9% LL (chứa 0) | phân tích lại |
| Ablation | Cohort 8: factorial niêm phong; cohort 7: M1 exploratory | Cohort 8: cả ba khối có Shapley > 0. Ngoài generator: chỉ khối legacy có tín hiệu LL; horizon gây hại ở RiverSwim và SPIBB | biên bản cohort 8, M1 |
| Selector | Ứng dụng phụ | GO vs fixed CFQI, NO-GO vs v5 | cohort 7 |

## 2. Định vị đóng góp (thay mục 1 và 2 của bản 13/10)

1. **Target và benchmark:** nhu cầu dữ liệu thực nghiệm theo learner, có kiểm duyệt phải; benchmark có kiểm soát với nhãn exact.
2. **OEH:**
   - Trong các họ MDP của generator, đây là bộ dự báo tốt nhất trong mọi đối thủ cùng thông tin đã thử. Kết quả tái lập niêm phong ba lần, hiệu ứng đồng nhất.
   - Trên kernel người khác, OEH là **bộ phát hiện "không đủ dữ liệu" vững và giữ được hiệu chỉnh**, chứ không phải bộ ước lượng mức ngân sách tốt hơn.
3. **Quy trình đánh giá:** protocol commit trước dữ liệu, dự báo niêm phong, gatekeeping, nhiều panel ngoài, và báo cáo đầy đủ kết quả âm (ba cohort đầu, FourRooms, SPIBB, BRIDGE Atari, RiverSwim).

**Venue phù hợp:** TMLR hoặc IEEE (Access/Transactions), có thể gửi workshop song song. Main track NeurIPS/ICML/ICLR chưa phù hợp nếu chưa có D1 (mở rộng quy mô).

## 3. Câu được phép viết và câu không được viết

| Không viết | Viết (tiếng Anh, đã dùng trong `paper/`) |
|---|---|
| "OEH generalizes to independently designed environments" | "On kernels drawn from the Colosseum sampler the method beats generic statistics in integrated Brier, mainly by recognizing insufficient data and staying calibrated; its log-loss advantage is not established and its finite-budget placement is worse than that of generic statistics." |
| "OEH accurately predicts the required budget" | "Within the studied generator families OEH forecasts the requirement distribution best; outside them it is a robust detector of insufficient data." |
| "The gain comes from posterior simulation" | "Within the generator all three blocks contribute (Shapley intervals exclude zero); outside it only the legacy diagnostics carry log-loss signal." |
| "Large log-loss gain over XGBoost-CF" | Nêu kèm sàn: 50% với sàn 1e-12, khoảng 17% với sàn 0,01, do XGBoost gán xác suất 0. |
| "OEH saves data" | "A decision rule fixed before outcomes lowers regret relative to B2 by 16% on the internal panel, not significantly on Colosseum; OEH's own cost is excluded." |
| "Three independent confirmations" | "Three sealed tests from the same generator and recipe templates." |
| "SOTA" | Không dùng. |

## 4. Paper (`paper/`, IEEE conference, 8/8 trang)

| Mục | Trạng thái 14/10 | Còn lại |
|---|---|---|
| Abstract | Đã cập nhật: cohort 8, Colosseum (chỉ Brier, cơ chế), SPIBB | Đếm lại ≤ 250 từ khi chốt venue |
| I. Introduction | Chưa sửa; đóng góp 3 vẫn ghi "external transfer tests" | Thêm một câu về tái lập niêm phong và vai trò phát hiện "không đủ dữ liệu". Không thêm trang |
| II–IV | Không đổi | — |
| V. Setup | Không đổi; gate cohort 8 mô tả ở VI.E | Nếu thừa chỗ, chuyển một câu mô tả gatekeeping lên V.E |
| VI.A–D | Không đổi | — |
| VI.E Sealed replication and external panels | **Mới**: bảng 4 panel, kết quả từng bước, gộp random-effects, cơ chế (Murphy, hoán đổi), Shapley | Hình forest plot chéo panel nếu thay được một bảng |
| VI.F Generalization, cost, limitations | Đã cập nhật: regret khóa trước, giới hạn cùng generator | — |
| VII. Conclusion | Đã cập nhật | — |
| Supplement | Thêm chi tiết Colosseum 2027 và hàng TQ/SACo | Thêm bảng factorial/Shapley ba panel và bảng cohort 8 đầy đủ |

**Page budget:** bản hiện tại đầy đúng 8 trang. Mỗi thêm mới cần một phần cắt tương ứng:
- Ứng viên cắt 1: Bảng IV (selector), rút còn hai câu, bảng chuyển sang supplement.
- Ứng viên cắt 2: Bảng transfer BRIDGE, gộp vào bảng VI.E dưới dạng một dòng.

## 5. Thesis (`Drive/Thesis/v2_2026-10-13/`, build `DAORL_Thesis_v2_2026-10-14.docx/.pdf`, 48 trang)

| Chương | Trạng thái 14/10 | Còn lại |
|---|---|---|
| 1 Giới thiệu | Chưa sửa | RQ2 nên tách thành RQ2a (subgroup trong generator) và RQ2b (chuyển giao: Colosseum, SPIBB, BRIDGE) |
| 2 Cơ sở | Chưa sửa | Thêm D3: mệnh đề nhất quán của luật OEH-B, kèm giả thiết ([D3](D3_THEORY_NOTE_2026-10-14.md)) |
| 3 Dataset | Chưa sửa | Thêm mục 3.7 "Panel ngoài": sampler Colosseum, reward trung bình, bộ lọc; SPIBB Garnet và baseline của tác giả; khoảng cách tới meta-training |
| 4 Phương pháp, đánh giá | Chưa sửa | 4.9 thêm gatekeeping ba bước và kiểm tra fit niêm phong. 4.10 không còn là "kế hoạch": đã có factorial niêm phong trên cohort 8 |
| 5 Kết quả | **5.8 mới**; 5.7 đã cập nhật | 5.5 tham chiếu Shapley của cohort 8; thêm bảng gộp random-effects và phân rã cắt/hữu hạn vào 5.8 |
| 6 Kết luận | Đã cập nhật 6.1, 6.3 | 6.2 thêm vai trò "detector" ngoài generator |
| Phụ lục | Chưa sửa | Bảng amendment trước outcome (cohort 8, SPIBB), bảng sửa số của bàn giao Colosseum |

## 6. Việc tiếp theo, xếp theo hiệu quả trên thời gian

| Ưu tiên | Việc | Chi phí | Lợi ích |
|---|---|---|---|
| P0 | Review và merge PR #53. Hưng duyệt diễn giải cơ chế (Shapley, hoán đổi); Việt Trần kiểm split/leakage của cohort 8 và SPIBB; Quốc Việt kiểm metric và bảng | Người | Bắt buộc trước khi nộp |
| P0 | Điền tác giả, affiliation, venue; giảng viên duyệt claim (mục 3) | Người | Bắt buộc |
| P1 | Kiểm định niêm phong **head lai**: P(>x50) từ OEH, phân phối mức hữu hạn từ B2 + legacy | Khoảng 2 giờ máy | Có thể biến claim log loss ngoài generator từ "chưa xác lập" thành "xác lập" |
| P2 | Thesis chương 1–4 và phụ lục theo mục 5 | Khoảng 1 ngày viết | Thesis khớp bằng chứng |
| P2 | Forest plot chéo panel (từ `meta_analysis.csv`, `paired.csv`) cho thesis, và cho paper nếu cắt được một bảng | Khoảng 1 giờ | Truyền đạt cả kết quả trong lẫn ngoài generator trong một hình |
| P3 | Reward prior thay `unseen_zero` cho khối horizon | Vài giờ tính lại OEH-H cho 7 cohort meta + một panel mới | Lợi ích chưa chắc; chỉ làm nếu P1 thất bại |
| Không làm | Thêm cohort nội bộ (I² = 0, không thêm thông tin); D1 mở rộng quy mô và thêm BRIDGE (vài tháng); viết lại lịch sử git (đã quyết định không làm) | — | — |

**Chi tiết P1:**
- Ứng viên được lập luận từ OOF leave-one-family-out trên 720 MDP đã mở: IBS 0,0907 vs 0,0962, LL 0,904 vs 0,931 cho OEH.
- Nhưng ý tưởng xuất hiện sau khi đã xem nhãn Colosseum (phân tích hoán đổi). Phải ghi rõ điều này trong protocol.
- Kiểm định trên tầng Colosseum mới (seed 2029, 72 MDP) và một panel SPIBB mới (108 MDP, seed khác). Dùng runner `run_oeh_eighth_cohort_20261014`/`run_spibb_panel_20261014` với namespace mới; trước khi chạy song song, bật `no_ecoqos.ps1`.

## 7. Kiểm tra trước khi bàn giao bản nộp

1. `python -m pytest -q` (hiện 204 passed).
2. `paper/build.sh` ≤ 8 trang; build supplement.
3. Thesis: `build_thesis_v2.py` rồi `update_fields.ps1`; kiểm mục lục, danh sách bảng và trang của mục 5.8.
4. Mọi số trong bài lấy từ CSV của thư mục artifact tương ứng: `cross_panel_reanalysis_2026-10-14`, `oeh_eighth_cohort_2026-10-14/*/result.json`, `colosseum_posthoc_part2_2026-10-14`.
5. Không có dòng ghi công AI trong commit hay PR.

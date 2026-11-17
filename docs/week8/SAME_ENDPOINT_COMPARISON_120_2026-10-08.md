# Cùng phép so sánh bảy lớp trên 120 MDP

**Ngày 08/10/2026.** Mỗi nhánh dự báo phân phối trên `x1, x2, x5, x10, x20, x50, >x50` từ cùng pilot, learner và budget. Brier tích lũy và log loss càng thấp càng tốt. Bảng chia **84 MDP stochastic primary** (420 hàng/learner) và **36 MDP deterministic control** (180 hàng/learner); không dùng điểm gộp 120 để thay endpoint primary. Bốn nhánh có cùng 1.800 khóa `(panel, MDP, behavior, learner)` trong CSV dự báo đã chấm, không thiếu khóa.

Nguồn: [B2 matched/OEH đã niêm phong trước nhãn](../../artifacts/oeh_confirmatory_unseen_120_2026-10-04/scored_predictions.csv), [XGBoost-CF/18](../../artifacts/week8/published_xgboost_cf_120_exploratory_2026-10-08/summary.csv), [XGBoost-CF/108](../../artifacts/week8/published_xgboost_cf_120_full108_architecture_2026-10-08/summary.csv). Mỗi ô là **Brier / log loss**.

| Panel | Learner | B2 matched, nội bộ | OEH cuối, 108 đặc trưng | XGBoost-CF/18, thích nghi | XGBoost-CF/108, kiểm soát kiến trúc |
|---|---|---:|---:|---:|---:|
| Primary 84 | BC | 0,024564 / 0,200285 | 0,018882 / 0,177185 | 0,024695 / 0,270426 | **0,017289** / 0,216623 |
| Primary 84 | CFQI | 0,089056 / 1,176630 | **0,077394 / 1,077472** | 0,093609 / 1,528658 | 0,080730 / 1,206383 |
| Primary 84 | IQL | 0,050560 / 0,504623 | **0,041691 / 0,434579** | 0,053762 / 1,773310 | 0,044793 / 1,234181 |
| Control 36 | BC | 0,034235 / 0,349302 | **0,027789 / 0,286588** | 0,031785 / 0,340507 | 0,029170 / 0,459112 |
| Control 36 | CFQI | 0,022062 / 0,323571 | **0,018478 / 0,306795** | 0,024476 / 0,518303 | 0,019631 / 0,342560 |
| Control 36 | IQL | 0,016371 / **0,252689** | 0,016356 / 0,263820 | **0,015007** / 0,285556 | 0,017345 / 0,300987 |

Tính mô tả trên **cả 120 MDP** bằng trung bình theo số hàng 84:36:

| Learner | B2 matched | OEH | XGBoost-CF/18 | XGBoost-CF/108 |
|---|---:|---:|---:|---:|
| BC | 0,027466 / 0,244990 | 0,021554 / 0,210006 | 0,026822 / 0,291450 | **0,020853** / 0,289369 |
| CFQI | 0,068958 / 0,920712 | **0,059719 / 0,846269** | 0,072869 / 1,225552 | 0,062400 / 0,947236 |
| IQL | 0,040304 / 0,429043 | **0,034091 / 0,383351** | 0,042135 / 1,326984 | 0,036559 / 0,954223 |

Trên primary, [phân tích ghép cặp theo MDP](../../artifacts/week8/matched_ablation_analysis_exploratory_2026-10-07/paired_mdp_delta.csv) cho `OEH − B2` Brier: BC `−0,005682 [−0,013341; +0,001242]`, CFQI `−0,011662 [−0,017993; −0,006106]`, IQL `−0,008869 [−0,015713; −0,002398]`. [OEH − XGBoost-CF/18](../../artifacts/week8/published_xgboost_cf_120_exploratory_2026-10-08/paired_mdp_bootstrap.csv): BC `−0,005813 [−0,012646; +0,001137]`, CFQI `−0,016216 [−0,022398; −0,010644]`, IQL `−0,012071 [−0,019621; −0,004943]`. [OEH − XGBoost-CF/108](../../artifacts/week8/published_xgboost_cf_120_full108_architecture_2026-10-08/paired_mdp_bootstrap.csv): BC `+0,001593 [−0,000344; +0,003654]`, CFQI `−0,003336 [−0,006337; −0,000206]`, IQL `−0,003102 [−0,006027; −0,000386]`. Các khoảng theo learner này **chỉ mô tả**, không là gate đồng thời mới.

Gate gốc [đã đăng ký trước nhãn](../../artifacts/oeh_confirmatory_unseen_120_2026-10-04/contrasts.csv) gộp ba learner trên **84 MDP primary**: OEH Brier `0,045989`, log loss `0,563079`; B2 matched `0,054727 / 0,627179`. Cải thiện tuyệt đối `0,008738 / 0,064100`; cận dưới đồng thời một phía `+0,002749 / +0,029810`. Đây là kết luận đối với **B2 nội bộ** trong họ generator đã dùng phát triển. XGBoost-CF/18 và /108 được chọn/chạy sau khi nhãn 120 MDP đã mở, nên các phép so với chúng là **khám phá** dù dùng cùng hàng và metric. XGBoost-CF/108 dùng đặc trưng OEH nên không là phương pháp độc lập với OEH.

Log loss của XGBoost-CF cần đọc cùng các dự báo quá tự tin: ở primary, XGBoost-CF/18 gán xác suất **0** cho lớp thật ở 1 hàng BC, 5 hàng CFQI và **22 hàng IQL**; XGBoost-CF/108 có lần lượt 1, 1 và 13 hàng. Scorer dùng sàn `1e-12`, tức mỗi hàng như vậy nhận log loss khoảng `27,631`. Vì thế phần lớn chênh log loss IQL với XGBoost-CF/18 đến từ 22 hàng này; Brier và chênh ghép cặp nên được báo song song, không chỉ nêu tỷ lệ cải thiện log loss.

Các score nguyên bản EH/ERI/TQ/SACo không xuất phân phối bảy lớp để đặt vào bảng Brier. [Bảng rank phụ trên 84 MDP](E3_OEH_VS_PUBLISHED_METHODS_2026-10-08.md) dùng endpoint tương quan thứ hạng và ghi riêng quyền truy cập MDP thật của TQ/SACo. BRIDGE [đã kiểm độc lập trên 5 task](BRIDGE_FRESH_FULL_RESULT_2026-10-08.md) cho kết quả khác chiều ở IQL. Không suy kết quả thắng toàn miền hay SOTA từ bảng 120 MDP.

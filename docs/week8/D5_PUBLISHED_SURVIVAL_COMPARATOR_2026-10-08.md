# Đối chứng công bố có đầu ra tương thích cho difficulty-aware offline RL

**Ngày 08/10/2026. Trạng thái:** đối chứng trên 120 MDP đã chạy (khám phá); lượt BRIDGE mới đã hoàn tất với dự báo niêm phong trước nhãn và [kết quả đầy đủ](BRIDGE_FRESH_FULL_RESULT_2026-10-08.md). Không dùng bảng này để tuyên bố “vượt SOTA offline RL”.

## 1. Tìm đúng đối tượng so sánh

Đầu vào bài toán là pilot offline `D0`, learner `l`, MDP/behavior chưa biết nhãn; đầu ra là phân phối bảy lớp của budget đầu tiên `E_l` đạt ngưỡng return, gồm lớp `>x50`. Brier dùng sáu xác suất tích lũy; log loss dùng xác suất lớp quan sát. Không so Brier này với return, sample-complexity upper bound hoặc score EH/TQ/SACo.

Rà soát đến 08/10/2026 **chưa xác minh được** một leaderboard hay phương pháp nguyên bản đã công bố nhận chính `D0,l` và phát trực tiếp phân phối `P(E_l=0…6)` trên stochastic tabular offline RL. Đây là kết luận của phạm vi tìm kiếm hiện tại, không phải chứng minh không tồn tại công trình khác. Các nghiên cứu gần nhất:

| Công trình | Đầu ra của tác giả | Vai trò ở đây |
|---|---|---|
| [Mateos-Melero et al., EWRL 2024](https://plg.uc3m.es/papers/mateos-melero-et-al-ewrl24.pdf), [AAMAS 2025 doctoral consortium](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2950.pdf) | Hiệu năng/return của một learner từ biểu diễn ảnh của dataset, trong domain cố định | Gần về dự báo hiệu năng offline RL, nhưng không có đầu ra budget-law và ảnh theo domain không áp dụng nguyên trạng cho panel này. |
| [Kim, Lai & Zhang, arXiv 2026 v2](https://arxiv.org/html/2601.22259v2), [mã tác giả](https://github.com/dain20000222/TabularFM4Survival) | Phân phối thời điểm xảy ra sự kiện có censoring từ đặc trưng tabular; XGBoost-CF dự báo trực tiếp cumulative failure ở từng mốc | **Đối chứng công bố có cấu trúc đầu ra tương thích nhất có thể chạy trên máy hiện tại**, sau khi thích nghi time grid và đặc trưng pilot. Đây là survival, không phải SOTA của offline RL. |
| [Laidlaw et al., EH stochastic 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/3e18f7152b5e1696e73722dc5121b6f2-Paper-Conference.pdf); [Schweighofer et al., TQ/SACo 2022](https://proceedings.mlr.press/v199/schweighofer22a.html); [Swazinna et al., ERI 2021](https://arxiv.org/abs/2111.13461) | Mô tả độ khó MDP, chất lượng/coverage dataset, tiềm năng cải thiện | Score gốc chấm liên hệ thứ hạng với `E_l`, có quyền truy cập MDP/reference ghi riêng; không đưa vào bảng Brier/log loss. |

Trong bảng 43 dataset survival tĩnh của Kim và cộng sự, MITRA-CF có thứ hạng tổng hợp cao nhất, XGBoost-CF rất gần và có mã/siêu tham số cố định; kết quả đó **không** là thứ hạng trên offline RL. Bài đang ở dạng **arXiv preprint under submission**, chưa gọi là một chuẩn peer-reviewed của offline RL. MITRA/TabPFN cần gói/checkpoint và phép suy luận in-context lớn hơn; `autogluon`, `tabpfn`, `sksurv` hiện không cài trên máy và GPU 4 GB đang dùng cho OEH-B. Vì vậy chọn XGBoost-CF là đối chứng công bố khả thi, không gọi nó là “SOTA chung” hoặc đối chứng hoàn chỉnh cho toàn bộ literature.

## 2. Cấu hình chạy và quyền truy cập

Lấy đúng [mã XGBoost-CF của tác giả](../../benchmarks/bridge/third_party/TabularFM4Survival/static/static_xgboost_final.py) tại commit `b27d69c27a5c39b87822b1aebf838e3dd900000e`, lưu SHA-256 trong manifest. Bản thích nghi dùng đúng cách mở rộng mỗi quan sát thành các bài toán phân loại `1(E_l <= budget)`, thêm một hàng tại thời điểm event/censor của từng quan sát, và phép sửa survival không tăng. Dùng `100` cây, learning rate `0.1`, depth `3`, seed `42` từ mã tác giả; CPU một luồng để không tranh tài nguyên với OEH-B. Giữ sáu budget `{x1,x2,x5,x10,x20,x50}`, time=50 cho censoring hành chính, và hiệu chỉnh thành bảy xác suất bằng hiệu CDF. Không tune theo nhãn BRIDGE.

Môi trường máy chạy là Python 3.12.7, XGBoost 3.3.0, scikit-learn 1.7.2 và NumPy 2.5.1; `requirements.txt` của tác giả pin Python 3.9/XGBoost 2.1.4. Do đó đây là **tái lập công thức và siêu tham số**, không phải chạy nguyên vẹn environment/package của tác giả. `tree_method=hist` và `n_jobs=1` phục vụ giới hạn tài nguyên local.

| Điều kiện | OEH | XGBoost-CF 18 đặc trưng | XGBoost-CF 108 đặc trưng |
|---|---|---|---|
| Meta-training | Cùng 720 MDP đã mở; nhánh primary dùng 504 MDP, 2.520 hàng/learner | Y như OEH | Y như OEH |
| MDP/behavior/learner test, pilot, budget, nhãn | Cùng hàng | Cùng hàng | Cùng hàng |
| Dữ liệu nhập dự báo | 108 descriptor từ pilot, gồm OEH-B/H | 18 thống kê pilot cơ bản `FEATURES_B2` | Chính 108 descriptor của OEH |
| Bộ dự báo | Ensemble AFT + threshold + multiclass của nhóm | XGBoost-CF công bố, thích nghi endpoint | Cùng XGBoost-CF; **architecture control**, vì dùng descriptor OEH |
| Tuning | Trọng số từ grouped OOF đã khóa | Siêu tham số gốc cố định, không tune test | Siêu tham số gốc cố định, không tune test |

Do đó bảng 18 đặc trưng đo **toàn bộ phương pháp và representation** trên cùng thông tin pilot, không cô lập riêng kiến trúc. Bảng 108 đặc trưng cô lập phần kiến trúc, nhưng không là đối chứng độc lập với ý tưởng đặc trưng OEH. Cần trình bày cả hai, không chọn riêng bảng thuận lợi. [Script 18 đặc trưng](../../benchmarks/bridge/fresh_published_survival_forecast.py), [script 108 đặc trưng](../../benchmarks/bridge/fresh_published_survival_full_forecast.py), [test biến đổi survival](../../tests/test_published_survival_baseline.py).

## 3. Kết quả 120 MDP đã mở nhãn: khám phá

[Artifact đầy đủ](../../artifacts/week8/published_xgboost_cf_120_exploratory_2026-10-08/summary.csv) dùng đúng 84 MDP primary và 36 control, 5 behavior, 3 learner; OEH lấy từ dự báo đã niêm phong trước nhãn ở cohort gốc. Đối chứng XGBoost-CF được chọn và chạy **sau khi nhãn 120 MDP đã mở**, nên các CI sau chỉ mô tả; không biến chúng thành lượt confirmatory mới. Chênh lệch là `OEH - XGBoost-CF`, số âm tốt cho OEH.

| Primary (84 MDP) | Brier OEH / CF | Log loss OEH / CF | Δ Brier [bootstrap 95% theo MDP] |
|---|---:|---:|---:|
| BC | 0,018882 / 0,024695 | 0,177185 / 0,270426 | −0,005813 [−0,012646; +0,001137] |
| CFQI | 0,077394 / 0,093609 | 1,077472 / 1,528658 | −0,016216 [−0,022398; −0,010644] |
| IQL | 0,041691 / 0,053762 | 0,434579 / 1,773310 | −0,012071 [−0,019621; −0,004943] |

Đối chứng này thua OEH về điểm Brier/log loss trên cả ba learner primary, nhưng khoảng Brier của BC **vẫn cắt 0**. Log loss của XGBoost-CF có thể nhạy với xác suất gần 0; xem hàng từng ví dụ trong [`scored.csv`](../../artifacts/week8/published_xgboost_cf_120_exploratory_2026-10-08/scored.csv). Bảng control và mọi hàng âm/dương nằm trong [`paired_mdp_bootstrap.csv`](../../artifacts/week8/published_xgboost_cf_120_exploratory_2026-10-08/paired_mdp_bootstrap.csv). Không suy luận “vượt SOTA” từ kết quả này.

**Kiểm soát kiến trúc 108 đặc trưng cũng đã chạy** trên đúng 120 MDP: [summary](../../artifacts/week8/published_xgboost_cf_120_full108_architecture_2026-10-08/summary.csv), [chênh lệch paired](../../artifacts/week8/published_xgboost_cf_120_full108_architecture_2026-10-08/paired_mdp_bootstrap.csv). Đây không phải phương pháp độc lập vì cùng dùng các descriptor OEH.

| Primary (84 MDP) | Brier OEH / CF-108 | Log loss OEH / CF-108 | Δ Brier [bootstrap 95% theo MDP] |
|---|---:|---:|---:|
| BC | 0,018882 / **0,017289** | **0,177185** / 0,216623 | +0,001593 [−0,000344; +0,003654] |
| CFQI | **0,077394** / 0,080730 | **1,077472** / 1,206383 | −0,003336 [−0,006337; −0,000206] |
| IQL | **0,041691** / 0,044793 | **0,434579** / 1,234181 | −0,003102 [−0,006027; −0,000386] |

Hàng BC Brier đảo chiều điểm so với CF-18, còn khoảng paired cắt 0. Điều này cho thấy phần cải thiện so với CF-18 có thể đến từ **đặc trưng OEH**, không thể quy toàn bộ cho kiến trúc ensemble. CFQI/IQL vẫn có chênh lệch Brier âm ở lượt khám phá này, nhưng cần xem kiểm định BRIDGE mới trước mọi claim chuyển môi trường.

## 4. Lượt BRIDGE mới: gate trước nhãn và kết quả

Trên năm MDP BRIDGE Atari chưa từng mở nhãn ở panel mới, [addendum trước nhãn](../../benchmarks/bridge/fresh_panel_2026-10-08/published_survival_comparator_addendum.json) ghi rõ hai arm và vai trò từng arm. XGBoost-CF 18 đặc trưng ghi **75/75** dự báo vào [manifest prelabel](../../artifacts/week8/bridge_fresh_published_xgboost_cf_preoutcome_2026-10-08/manifest.json) cùng hash đầu vào, mã tác giả, code adapter và output. [`fresh_prelabel_seal.py`](../../benchmarks/bridge/fresh_prelabel_seal.py) đã fit thêm phiên bản 108 đặc trưng **sau** full OEH feature assembly và **trước** khi runner đọc nhãn. [Seal](../../benchmarks/bridge/fresh_panel_2026-10-08/prelabel_seal.json) kiểm đủ 75 hàng, đúng key, tổng xác suất và hash cho OEH/B2/XGBoost-CF 18/XGBoost-CF 108/score gốc. [`fresh_run_test_labels.py`](../../benchmarks/bridge/fresh_run_test_labels.py) kiểm lại hash và hoàn thành [75 nhãn](../../artifacts/week8/bridge_fresh_test_labels_2026-10-08/manifest.json). [`fresh_score_test.py`](../../benchmarks/bridge/fresh_score_test.py) chấm Brier, log loss và chênh lệch ghép cặp theo task trên **cùng** 75 hàng. Với chỉ 5 task, khoảng bootstrap chỉ mang tính mô tả.

Kết quả hoàn chỉnh và các hàng bất lợi có trong [báo cáo BRIDGE](BRIDGE_FRESH_FULL_RESULT_2026-10-08.md) và [manifest phân tích](../../artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08/manifest.json). OEH có Brier tốt hơn XGBoost-CF/18 ở BC/CFQI về điểm trung bình nhưng kém ở IQL; khoảng ghép cặp BC/CFQI cắt 0 và khoảng IQL nằm phía bất lợi. XGBoost-CF/108 còn có Brier tốt hơn OEH ở CFQI. Vì vậy không có kết luận thắng đồng loạt. Lượt này dùng **MDP do bên thứ ba công bố, offline pilot và label do nhóm tạo**; không được gọi các log nhóm tạo là dataset offline do tác giả BRIDGE công bố.

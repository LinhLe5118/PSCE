# D1 — Learner, comparator và metric matrix

**Ngày:** 11/10/2026. **Phần việc:** Nguyễn Quốc Việt. **Nguồn kết quả:** cohort 7 ngày 04/10, main `22b4e9f`. Đây là tài liệu và kiểm tra lại kết quả đã khóa; không huấn luyện hoặc tuning phương pháp mới.

## 1. Hai câu hỏi phải tách riêng

**Dự báo nhu cầu dữ liệu:** Với một môi trường, một cách thu thập dữ liệu và một learner cụ thể, cần mức dữ liệu nào để learner đạt yêu cầu ổn định? Ba learner tạo nhãn chính là **BC, CFQI, IQL**. Dự báo có bảy lớp: x1, x2, x5, x10, x20, x50 và >x50.

**Chọn learner:** Khi có dữ liệu ở x10, nên dùng learner nào? Đây là ứng dụng phụ với **bảy learner**, không phải bảy phương pháp dự báo nhu cầu dữ liệu. BCQ, CQL, SPIBB, VI-LCB chỉ có kết quả x10 trong phép thử hiện tại; không gán nhãn sáu-budget cho chúng.

Tất cả policy được chấm bằng cùng exact scorer trên MDP thật, với gamma=0,95. Scorer được dùng sau khi fit; việc scorer biết MDP không có nghĩa mọi learner hoặc mọi forecaster đều được cấp kernel thật.

## 2. Bảy learner và lý do sử dụng

| Learner | Giải thích dễ hiểu / lý do dùng | Implementation hiện tại | Vai trò |
|---|---|---|---|
| BC | Làm giống các hành động trong dữ liệu; mốc đơn giản để biết học giá trị có giúp hơn bắt chước không | Tần suất hành động tabular của project | Requirement và selection |
| CFQI | Ước lượng giá trị từ log rồi chọn hành động tốt trong số hành động đã thấy | Reference tabular của project; không phải neural FQI benchmark | Requirement; fixed CFQI là đối chứng selection đăng ký |
| IQL | Học giá trị theo expectile và tăng trọng số hành động có lợi thế; bổ sung cách học khác CFQI | Tabular implementation của project theo ý tưởng IQL; không phải chạy toàn bộ benchmark tác giả | Requirement và selection |
| BCQ | Tránh chọn hành động ít được dữ liệu hỗ trợ | Discrete learner của tác giả, project chuyển state sang one-hot và chấm policy | Selection x10 |
| CQL | Phạt việc ước lượng quá cao giá trị hành động ngoài dữ liệu | DiscreteCQL của d3rlpy đã pin commit | Selection x10 |
| SPIBB | Giữ hành vi nền ở vùng ít dữ liệu, cho thay đổi ở vùng đủ quan sát | Pi-b-SPIBB core của tác giả, adapter của project | Selection x10; được cấp true beta |
| VI-LCB | Trừ mức phạt bất định trong value iteration | Project port theo paper-spec cho discounted MDP; không gọi là unchanged author learner | Selection x10; thêm metadata terminal |

Nguồn học thuật: [IQL](https://arxiv.org/abs/2110.06169), [Discrete BCQ](https://arxiv.org/abs/1910.01708), [CQL](https://arxiv.org/abs/2006.04779), [SPIBB](https://proceedings.mlr.press/v97/laroche19a.html), [VI-LCB](https://arxiv.org/abs/2204.05275). Ý tưởng của paper không tự chứng minh phiên bản/cấu hình hiện tại đạt mọi bảo đảm trong paper gốc.

### 2.1 Nguồn mã, cấu hình và ngân sách

| Learner | Nguồn/pin | Cấu hình và mức tối ưu đã dùng | Dữ liệu huấn luyện |
|---|---|---|---|
| BC | `code/empirical_requirement.py::behavior_cloning_tabular`, bản batched `code/oeh/tabular.py::behavior_cloning` | Đếm hành động; fallback uniform tại state chưa quan sát/terminal; không gradient update | Sáu budget × 10 stream seed; selection lấy riêng x10 |
| CFQI | `code/empirical_requirement.py::cfqi_tabular`, `code/oeh/tabular.py::cfqi` | Tối đa 100 backup, dừng sớm ở tolerance 1e-10; restricted observed-action support; tie order cố định | Như BC |
| IQL | `code/empirical_requirement.py::iql_tabular`, `code/oeh/tabular.py::iql` | Expectile 0,7; inverse temperature 3; tối đa 100 iteration, tolerance 1e-10 | Như BC |
| BCQ | [sfujim/BCQ](https://github.com/sfujim/BCQ), commit `4876f7e5afa9eb2981feec5daf67202514477518`; `_fit_bcq` | One-hot state; hai hidden layer 256; batch 64; 500 update; threshold 0,3; Adam lr=3e-4; tau=0,005; eval epsilon=0; CUDA | x10, 10 stream seed và lịch neural seed cố định |
| CQL | [takuseno/d3rlpy](https://github.com/takuseno/d3rlpy), commit `38c34b6f99fde82875e2ad81d1ba1f7a7da34137`; `_fit_cql` | One-hot; hidden 256×256; batch 64; 500 update; alpha=1; lr=6,25e-5; target update=50; CUDA | Như BCQ |
| SPIBB | [RomainLaroche/SPIBB](https://github.com/RomainLaroche/SPIBB), commit `833f32b642759111907d2c4d086d1ae1ce9d84cb` | Pi-b; N_wedge=10, mask count >10; policy iteration theo author `fit()`; Q-change norm ≤1e-9, nominal max 99999 và nhánh nonconvergence sau 1000 iteration | x10, 10 stream seed |
| VI-LCB | [Model-based-VI-LCB](https://github.com/Laixishi/Model-based-VI-LCB), pin `37858d392cf11a01ff5af4ef7815672ec0a36ca9`; port `code/vi_lcb.py` | c_b=0,05; delta=0,05; 100 iteration. c_b=144 là cấu hình lịch sử khác, không thêm thành learner thứ tám | x10, 10 stream seed |

Nguồn vận hành: `evaluation/e1_external.py` được phục hồi đúng hash trong frozen runtime; `scripts/analysis/oeh_final_learners.py`; `scripts/external/run_neural_discrete_development.py`; `protocol_config.json`; `code/oeh/tabular.py`. Commit project `8597639` trong bảng provenance W6 là nguồn lịch sử; checksum của protocol cohort 7 mới quyết định source dùng cho evidence hiện tại.

W6 từng ghi BC là sanity-only; **protocol cohort 7 đăng ký BC là một trong ba target requirement**. Không dùng ghi chú W6 để loại BC khỏi bảng hiện tại. SPIBB gọi author fit không cung cấp cùng ngân sách gradient như BCQ/CQL; các lượt chạy hợp lệ không chứng minh mọi learner đã hội tụ hoặc tuning được cân bằng.

### 2.2 Quyền truy cập phải hiển thị

- BC/CFQI/IQL/BCQ/CQL fit trên log, kích thước state/action và các hằng số công khai; không cấp true transition/reward kernel để fit.
- **SPIBB nhận true logging policy beta**, được generator dựng lại trong `evaluation/e1_external.py`. Nó không có cùng information budget với learner chỉ biết log.
- VI-LCB nhận thêm terminal-state metadata. Phải phân biệt metadata với true transition kernel.
- **Selector** nhận pilot-derived features, xác suất OEH cho CFQI và **nhãn họ MDP**. Nó không là phép chọn learner chỉ từ pilot với information budget y hệt forecast chính.
- Oracle selection chọn bằng outcome thật; chỉ là mốc trên, không triển khai được từ pilot.

Ngân sách dữ liệu bằng nhau không có nghĩa ngân sách tối ưu, số phép toán hoặc quyền truy cập bằng nhau. Các cấu hình neural 500 update không thiết lập bảng xếp hạng SOTA hoặc khả năng tốt nhất của thuật toán gốc.

## 3. Matrix các forecaster

`matched` nghĩa là cùng **quy trình** AFT + direct-threshold + multiclass depth-3, 200-round capacity; chọn convex weights bằng OOF log loss với năm fold theo MDP trên bảy cohort đã mở. Mỗi hàng fit riêng, không dùng chung fitted weights. AFT-only chỉ dùng một thành phần survival; frequency không fit head.

| Nhóm | Hàng | Đầu vào | Head / giới hạn phép so sánh |
|---|---|---|---|
| Matched pilot-only | B2 matched | 18 thống kê pilot | Đối chứng chính cùng head với OEH |
| Matched pilot-only | B2 + ERI adaptation | B2 + ERI và eligibility | Cùng head; adaptation từ chỉ số chất lượng, không tái lập task tác giả end-to-end |
| Matched pilot-only | B2 + BWD adaptation | B2 + BWD tabular | Cùng head; one-hot/SARSA/transport adaptation, không benchmark continuous-action gốc |
| Matched pilot-only | OEH hiện tại | 108 cột plus_prior_profile | Cùng head; chưa chứng minh đóng góp riêng từng nhóm feature |
| AFT-only pilot | B2 generic | Cũng 18 cột B2 | Khác head; không diễn giải chênh lệch như tác dụng feature-only |
| AFT-only pilot | Coverage, quality, coverage+quality | Các subset tín hiệu pilot | Báo riêng, không gộp thành matched-head |
| Frequency | Train-frequency | Tần suất lớp trong meta-train theo learner/panel | Không fitted matched head, không nhận feature query |
| Privileged matched | B2 + oracle EH | B2 + descriptor từ MDP thật | Có thông tin không có trong pilot deployment |
| Privileged matched | B2 + TQ/SACo reference | B2 + author-code metrics và true-MDP reference data | Cùng head nhưng khác quyền truy cập |
| Privileged matched | Published-metrics-only | TQ/SACo reference cùng ERI/BWD | Không được gọi toàn bộ block là pilot-only |
| Diagnostic, ngoài gate | Earlier OEH diagnostics | Feature lịch sử và context | Matched; không là ablation chỉ bỏ một cơ chế |
| Diagnostic, ngoài gate | Raw OEH-B | Posterior predictive law có smoothing | Không fit matched head; không gọi là forecaster cuối |

Nguồn quyết định kiến trúc là `run_oeh_final_120.py::FIELD_SETS/all_forecasts`, không phải câu “every fitted comparator” quá rộng trong protocol lịch sử. Giữ nguyên protocol đã seal.

### 3.1 Đủ 18 feature B2

1. `log_pilot_transitions`
2. `log_state_action_count`
3. `observed_sa_fraction`
4. `observed_state_fraction`
5. `normalized_visitation_entropy`
6. `log1p_min_positive_sa_count`
7. `trajectory_return_mean`
8. `trajectory_return_std`
9. `trajectory_return_iqr`
10. `terminal_trajectory_fraction`
11. `weighted_transition_entropy`
12. `observed_cell_transition_entropy`
13. `low_count_cell_fraction`
14. `empirical_stochastic_cell_fraction`
15. `low_coverage_entropy`
16. `high_coverage_entropy`
17. `bootstrap_return_mean_std`
18. `bootstrap_rare_exposure_mean_std`

Đọc dễ hiểu: số lượng dữ liệu/kích thước bài toán; vùng state-action đã quan sát; mức phân tán và thiếu dữ liệu; phần thưởng trajectory; độ ngẫu nhiên chuyển trạng thái; độ bất định bootstrap. B2 do nhóm xây, không phải published SOTA và không phải tên một learner.

## 4. Công thức metric và đơn vị tổng hợp

Với J là discounted expected return của policy, đặt:

$$NR(\pi)=\frac{J(\pi)-J_{random}}{J^*-J_{random}}.$$

Mẫu số phải dương. NR không là xác suất, không clip về [0,1]. Một seed thành công khi NR ≥0,95; không dùng dấu > nghiêm ngặt.

Với budget N_j=b_jSA, b=(1,2,5,10,20,50), j=0,…,5 và 10 seed:

$$C_{l,j}=\sum_{i=1}^{10}\mathbf1\{NR(\pi_{l,j,i})\ge0.95\},\qquad E_l=\min\{j:C_{l,m}\ge8\ \forall m\ge j\}.$$

Nếu không có j hợp lệ thì E_l=6 (>x50). Ví dụ counts [2,5,8,7,9,10] cho E=4 (x20), vì x5 chưa duy trì ở x10. [10,10,10,10,10,7] cho E=6. Không loại lớp 6 hoặc thay bằng một budget hữu hạn tùy ý.

Với p_c cho bảy lớp và F_j=Σ(c≤j)p_c:

$$IBS=\frac16\sum_{j=0}^{5}\left(F_j-\mathbf1\{E_l\le j\}\right)^2,\qquad LL=-\log\max(p_{E_l},10^{-12}).$$

Cả hai càng thấp càng tốt. IBS là trung bình trên sáu ngưỡng đo rời rạc, không phải tích phân theo thời gian liên tục. Log loss phạt mạnh khi dự báo rất tự tin vào lớp sai. Lớp censored vẫn được chấm; nó có sáu CDF target bằng 0.

Mỗi MDP có 5 behavior × 3 learner = 15 hàng forecast. Trung bình 15 hàng trước, rồi trung bình 84 MDP primary. Bootstrap resample cả MDP, không coi 1.260 hàng hoặc 10 seed như 1.260/12.600 môi trường độc lập. 36 control được báo riêng.

Paired gain metric = mean_MDP(score_baseline − score_OEH), nên gain dương có lợi. Relative gain = (baseline_mean − OEH_mean)/baseline_mean. Các cận một phía 95% đã đăng ký được hiệu chỉnh đồng thời trên 22 contrast. D1 đối chiếu nguồn và số học; không tự cấp approval cho claim thống kê.

Trong selection tại x10, success là trung bình tỷ lệ seed đạt NR≥0,95, **khác** nhãn 8/10 full-suffix. Return gain = mean_MDP(J_NR_selector − J_NR_fixed), success gain cũng lấy paired differences. 0,066667 = **6,67 điểm phần trăm**, không phải tăng tương đối 6,67%. Regret có thể mô tả bằng NR_oracle_group − NR_rule ở cùng nhóm; group oracle dùng outcome nên không là lựa chọn triển khai. Không suy tỷ lệ abstention từ lớp >x50: đây là censoring requirement, không phải quyết định từ chối chọn learner.

## 5. Bảng primary và kết quả phụ đã đối chiếu

Bảng primary đủ 14 hàng, giữ riêng kiến trúc và quyền truy cập: [bản đọc](../../artifacts/week8/d1_2026-10-11/primary_forecast.md), [CSV đầy đủ](../../artifacts/week8/d1_2026-10-11/primary_forecast.csv), [table LaTeX cho paper](../../artifacts/week8/d1_2026-10-11/primary_forecast.tex). [Bảng learner LaTeX](../../artifacts/week8/d1_2026-10-11/learner_matrix.tex) và [matrix cấu hình đầy đủ](../../artifacts/week8/d1_2026-10-11/learner_matrix.csv) đi kèm. LaTeX fragment dùng booktabs; nhóm chọn bảng phù hợp trước khi tích hợp và kiểm giới hạn tám trang.

Các con số chính: OEH Brier 0,045989 / LL 0,563079; B2 matched 0,054727 / 0,627179. Giảm lỗi tương đối **15,97% / 10,22%**. Có 22 cận dưới đăng ký dương trong artifact hiện hành. Đây là bằng chứng forecasting trong các họ generator đã nghiên cứu; không chứng minh toàn bộ ngành SOTA hoặc một feature cụ thể tạo ra toàn bộ lợi ích.

Selection: NR 0,811385 và success 47,95%, so với CFQI 0,792729 và 41,29%. Hai cận dưới đăng ký dương. So v5, NR giảm 0,001817 và success tăng 0,62 điểm phần trăm; supplementary gate vẫn NO-GO. Bảng đủ fixed learners, v5, random và oracle ở [selection.csv](../../artifacts/week8/d1_2026-10-11/selection.csv). Không so mean trực tiếp với cohort khác để tuyên bố tăng hiệu quả.

Các kết quả âm phải giữ: cohort 4 Tier 2/combined/selection NO-GO; FourRooms chỉ là component stress test NO-GO, không là full OEH validation; raw OEH-B không hiệu chỉnh có Brier 0,097826 / LL 0,910682; chưa có ablation hoặc full-method benchmark ngoài mới.

### 5.1 Cost và failure semantics

[core_x10_cost.csv](../../artifacts/week8/d1_2026-10-11/core_x10_cost.csv) tổng hợp đúng 4.200 callback runtime record/learner trong primary x10: BC mean 0,011261 s; CFQI 0,030134 s; IQL 0,059505 s. Timer bao callback load/fit/exact scoring, không là isolated fit benchmark; sum là tổng thời gian các job, không là wall-clock makespan khi chạy song song. Không dùng bảng này xếp tốc độ feature-generation hay lợi ích tiết kiệm compute của forecaster.

External execution audit xác nhận 16.800 job (4.200 mỗi learner ngoài). Individual fit/VRAM records và raw buffers không có đầy đủ trong clone; D1 không tự điền cost cho BCQ/CQL/SPIBB/VI-LCB. Cần đo riêng posterior simulation, OEH-H diagnostics, fitted head và downstream data trước claim saving. Pilot đã tốn x10, nên dự báo x1/x2/x5 có ý nghĩa hồi cứu trên stream train độc lập.

Job lỗi phải có failure artifact và được xử lý theo protocol; không loại job âm hoặc coi censored như technical failure. Success thấp là outcome hợp lệ. Audit compact xác nhận score/label consistency, không xác nhận end-to-end raw reproduction hoặc mọi tối ưu đều hội tụ.

## 6. Kiểm tra tái tạo và bàn giao

```powershell
python -m scripts.analysis.export_week8_d1
python -m scripts.analysis.export_week8_d1 --check
python scripts/analysis/audit_week8_data.py --strict-available
```

Exporter mới chỉ đọc compact artifacts và xuất vào namespace D1 riêng. Nó kiểm hash của input seal hiện diện, tập khóa đủ 1.800 nhãn × 14 method, xác suất bảy lớp, tổng hợp theo 120 MDP, 28 summary row, 22 contrast và 420 selection group. IBS/log loss được tái tính độc lập bằng standard library. Published bootstrap bounds được đọc và đối chiếu đúng row, không chạy lại bootstrap hay tự thay verdict.

[validation.json](../../artifacts/week8/d1_2026-10-11/validation.json) ghi checksum nguồn, số hàng và phạm vi kiểm. Human review, scientific approval và quyết định merge do nhóm thực hiện riêng. Hưng review giả định/cách diễn giải; Linh review thống kê/tích hợp. D1 không ký thay hai người này và không sửa các source/artifact đã niêm phong.

**Nghiệm thu D1:** đã có matrix bảy learner, quyền truy cập true-beta/family-label, đủ 18 feature B2, phân loại head, công thức metric, bảng primary không trộn nhóm, nguồn tái tạo và cost/failure limits. Kết quả kiểm thử cuối cùng được ghi tại [D1 handover](D1_HANDOVER_2026-10-11.md).

## 7. Note gợi ý cải thiện — cập nhật 12/10/2026

**Giữ bản hiện tại.** Các gợi ý dưới đây là hướng nghiên cứu tiếp, chưa phải cải thiện được xác nhận. Đã chạy [thử nghiệm nested trên cohort 4 đã mở](NESTED_EXPERIMENT_RESULTS_2026-10-12.md): hiệu chỉnh xác suất giảm Brier khoảng 0,58% và log loss 0,37%, nhưng cận thăm dò chưa đủ chắc; khi giữ một họ môi trường ngoài phần fit mới thì chất lượng giảm. Không dùng các số này để thay verdict của cohort 7.

- **Hiệu chỉnh xác suất ít tham số:** tiếp tục xem như candidate, fit trên dữ liệu phát triển và xác nhận ở cohort chưa mở. Cách này chỉnh mức tự tin, không đổi lớp được dự đoán cao nhất; không khắc phục mọi dự đoán sai.
- **Kết hợp OEH với B2:** thử học trọng số trên nhiều cohort để tận dụng ưu thế khác nhau; không đặt rule theo một họ chỉ vì đã xem kết quả của họ đó. Probe hiện tại giảm lỗi rất nhỏ và chưa chuyển tốt qua họ bị giữ lại.
- **Tập trung phân biệt x1–x50:** các trường hợp đạt trong miền này tạo ra phần lớn lỗi dù là thiểu số. Giữ lớp >x50 và đánh giá cả nhóm hữu hạn lẫn censored. Mô hình hai bước vừa thử kém hơn, nên chưa ưu tiên triển khai cấu hình đó; chưa kết luận mọi thiết kế hai bước đều thất bại.
- **Selector phải kiểm cả hiệu quả và tỷ lệ thành công:** so với cả CFQI và v5, giữ rõ thông tin true beta/family label. Các quy tắc học chênh lệch hoặc chỉ đổi khi dự đoán cả hai chỉ số tăng chưa cải thiện đồng thời trong probe đã chạy.
- **Khôi phục nguồn/dữ liệu và làm ablation:** tái tạo đầy đủ training trước khi thử head trên toàn bộ 108 feature; khóa nhóm feature/contrast trước outcome để biết phần nào đóng góp. Cần kiểm định mới và benchmark ngoài phù hợp trước claim tổng quát.

**Điều kiện nâng thành cải thiện:** chọn cấu hình bên trong MDP-grouped development, khóa model/prediction trước outcome mới, giữ metric/budget/access gốc, báo các phép thử âm và chi phí riêng. Nested CV không xóa lịch sử xem dữ liệu. Thử nghiệm nhẹ vì dùng kết quả đã có: runner khoảng 41 giây, output khoảng 4,5 MB, XGBoost một luồng CPU và không dùng GPU; không phải chi phí tái tạo toàn bộ pipeline hoặc train lại learner RL. RAM cao nhất chưa được đo.

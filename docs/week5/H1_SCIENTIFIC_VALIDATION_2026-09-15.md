# H1 — OEH Core & Scientific Validation

**Người phụ trách:** Đỗ Trọng Hưng

**Ngày chốt audit:** 2026-09-15

**Phạm vi:** development + probe/positive control; evaluation 120 MDP vẫn niêm phong

**Trạng thái:** hoàn thành nội dung kiểm chứng và bản thảo để Linh tích hợp

## Kết luận nghiệm thu

1. **Bão hòa của statistic không đồng nghĩa learner không học được.** Trên expanded development, cả 266 đường cong OEH đo được đều phẳng. Trong cùng gói kết quả, khi multiplier tăng từ 1 lên 50, median normalized return của CFQI tăng từ 0.425 lên 0.933 và success rate mức fit tăng từ 14.79% lên 47.08%. Kết luận hợp lệ là OEH statistic bão hòa trên các pilot này; không được suy ra thêm dữ liệu vô ích với learner.
2. **Định lý bão hòa là định lý có điều kiện.** Trần `L_k(1) <= G A_k(w_k, sigma_k)` đúng cho loss bậc một `psi = sigma g(z; mu)`, `g(.;1)` bị chặn, aggregator đơn điệu và thuần nhất dương. Kết luận tiệm cận còn cần aggregated dispersion tiến về 0. K2 chỉ cho median slope tương thích với heuristic `-1/2`; dung sai theo từng instance không đạt.
3. **Định lý 2 không đặc trưng mọi statistic thỏa A1.** Kết quả đã chứng minh là: hạng thuần nhất bậc 0 khi và chỉ khi viết được dưới dạng `h(Delta/sigma, Delta/Vhat; mu)`; hạng này đủ cho A1 khi occupancy không đổi. Chiều đảo từ requirement-level A1 sang tính thuần nhất từng hạng không suy ra từ R1–R2.
4. **Positive controls có phạm vi hữu hạn.** Ba control cho interior `k_dagger` bằng estimator không sửa, nên loại trừ một universal hard-coded/implementation floor. Chúng không xác nhận Gaussian approximation, scaling theo cỡ mẫu, hay mọi đường thực thi.
5. **Corridor loại đúng hai nhiễu.** Probe loại reward starvation và occupancy dilution. Ground truth cho `k_settle = {0:24, 2:9, 6:3}` trên 36 cell; chỉ 12/36 thật sự cần positive depth. Trên 9 cell scorable có positive depth, `max_state` đúng 1/9, kém constant answer 2 đúng 7/9. Vì vậy probe không đủ để tuyên bố mọi bão hòa là tính chất phổ quát của định nghĩa.
6. **Không có kết luận held-out.** Lưới evaluation 120 MDP chưa mở. Mọi kết quả mới là development-only/candidate.

## Ma trận phát biểu — giả thiết — bằng chứng

| Phát biểu được phép dùng | Giả thiết/phạm vi bắt buộc | Bằng chứng tương ứng | Không được suy ra |
|---|---|---|---|
| Registered loss có trần `G_reg * sum w sigma`, `G_reg = 0.169971...` | Loss đúng dạng đăng ký; weighted-sum aggregator; gap và sigma không âm | `paper/sections/03b_axioms_and_impossibility.tex`, kiểm thử feasibility | Learner không thể học; mọi statistic đều bão hòa |
| Statistic tiệm cận về `m_k = 1` | Tập depth hữu hạn và aggregated dispersion tiến về 0 | Hệ quả trong phần axioms | K2 đã chứng minh convergence đều |
| K2 tương thích theo median với slope `-1/2` | Ba cỡ mẫu, dải 5 lần, 100 bootstrap replicate | `artifacts/week2/scale_invariance_2026-09-05/`: slope pilot -0.472, state -0.626 | Scaling được xác nhận trên từng instance; K2 đạt |
| Original development bão hòa | 15 MDP, 75 pilot; 62 curve được chấm | 12,019 điểm có `m_k=1`; 59/62 verdict infeasible | 59 training job đã được tiết kiệm; RL không học được |
| Expanded OEH vẫn phẳng | 48 MDP, 240 MDP-policy group; development only | `artifacts/week5/pipeline_validation_2026-09-15/`: 266/266 curve phẳng | Đúng trên sealed evaluation |
| CFQI cải thiện khi dữ liệu tăng | Tabular CFQI implementation và fixed budget của gói expanded development | `learning_curves.csv`: median return 0.425→0.933; success 14.79%→47.08% | Quan hệ nhân quả hoặc ưu thế universal của CFQI |
| Calibrated comparison đã sửa 17/09 | Pairing theo MDP–policy, Brier toàn grid; development-only | Không có fair-baseline win; CFQI kém coverage likelihood và visit entropy likelihood/Brier | Không giữ claim IQL Brier thắng từ artifact cũ |
| Pipeline integrity đạt | Oracle-undefined được xử lý đúng frozen contract | 91/150 so sánh, 59 expected undefined, 0 unexpected error | Agreement depth 0 chứng minh positive-depth recovery |
| Estimator có thể cho interior depth | Ba positive control, estimator/config không sửa | `k_dagger`: 2, 1, 1; `max m`: 3.391, 1.160, 1.114 | Mọi lỗi mô hình/cài đặt đã bị loại trừ |
| Corridor có tín hiệu gap ở depth nội | Probe post hoc, không phải benchmark held-out | 31/32 scored cell peak nội; Figure 1 cell có swing 123.17 lần | Mọi cell đều cần positive planning depth |

## Đối chiếu định nghĩa với code

| Thành phần toán | Hợp đồng | Hiện thực nguồn |
|---|---|---|
| Support `A_K(s)` | Tối thiểu `n_min` transition và `min_trajectory_support` trajectory riêng biệt | `code/oeh_estimator.py::_supported_actions`, `_trajectory_counts` |
| `Q^(0)` | `Q^{beta_hat^K}`, tính bằng behavior-policy linear solve; **không phải** chỉ immediate reward | `code/oeh_estimator.py::_q_sequence` |
| Chuỗi backup | `Q^(k+1) = r_hat + gamma P_hat max_{a in A_K} Q^(k)` | `code/oeh_estimator.py::_q_sequence` |
| Top-two gap | Hai action support cao nhất; tie cố định theo action ID | `code/oeh_estimator.py::estimate_oeh` (`top_pairs`) |
| `sigma_Delta` | Bayesian bootstrap ở mức trajectory; support và top pair giữ theo full pilot; `ddof=1` | `code/oeh_estimator.py::_model`, `estimate_oeh` |
| Occupancy weight | Discounted occupancy của greedy policy trên empirical model; chỉ state critical/eligible nhận weight | `code/oeh_estimator.py::_occupancy`, `estimate_oeh` |
| `L_k(mu)` và `m_k` | `sum w Delta Phi(-z sqrt(mu))`; nếu `L_k(1)<=epsilon` thì `m_k=1`, ngược lại root find | `code/oeh_estimator.py::_loss_and_multiplier` |
| `k_dagger` | Argmin multiplier; near-tie `1e-12` chọn depth nhỏ nhất | `code/oeh_estimator.py::_smallest_near_minimum` |
| `H_OEH`, `N_req_hat` | `min_k[k + log_A(max(1,m_k))]`; `N0 * min_k m_k` | `code/oeh_estimator.py::estimate_oeh` |
| Feasibility | Chỉ áp dụng degree-one registered definition; không áp bound này cho alternative degree-zero | `code/feasibility.py`, `paper/sections/04_method.tex` |
| Oracle | Chỉ scoring/validation; không được đi vào estimator | `code/exact_evidence_curve.py`, no-leak tests |

## Hình cơ chế OEH

Figure 1 được chọn theo quy tắc mô tả đã khai báo trong script: scored corridor cell có finite `weighted_gap_swing` lớn nhất. Cell được chọn là `decoy_corridor_L8_f0p7_p0p05`, multiplier 10, `N0=180`.

- weighted gap: 0.001211 tại `k=0`, cực đại 0.149210 tại `k=4`, swing 123.172 lần;
- `max_k L_k(1) = 0.000464`, thấp hơn `epsilon=0.05` khoảng 108 lần;
- tại gap peak, `L_4(1) = 1.543e-9`;
- `max weighted sigma = 0.026148`, nên maximum theorem ceiling là 0.004444, thấp hơn tolerance 11.25 lần;
- `m_k=1` toàn đường cong và tie-break cho `k_dagger=0`.

Tái tạo:

```powershell
python -m scripts.analysis.plot_oeh_mechanism --output-dir <thu-muc-moi>
```

Script chạy lại estimator đã đăng ký, đối chiếu summary với artifact đông cứng rồi mới xuất CSV/JSON/PNG/SVG/PDF. Bản paper dùng `paper/figures/oeh_mechanism.pdf`.

## Nội dung gửi Linh để tích hợp paper

Đoạn tiếng Anh đã được tích hợp vào Results/Discussion:

> The expanded-development audit makes the distinction directly observable. All 43,200 fits completed while all 266 measured OEH curves were flat. Over data multipliers 1 to 50, CFQI median normalized return rose from 0.425 to 0.933 and its fit-level success rate from 14.79% to 47.08%. Thus flat OEH output is evidence about this statistic on these pilots, not evidence that the learner cannot benefit from additional data. These pooled development descriptions do not isolate a causal mechanism or imply monotonic improvement for every learner.

Phạm vi positive control để dùng nguyên văn:

> The controls rule out a universal hard-coded or implementation-level floor; they do not validate every modeling assumption or execution path.

## Các sửa đổi khoa học đã thực hiện

- Biến giả thiết `sigma = O(N^-1/2)` thành giả thiết first-order, không trình bày như kết quả tự động của bootstrap.
- Bỏ phát biểu chưa chứng minh rằng OEH phải có interior minimum hoặc `k_dagger` chính là planning depth.
- Sửa special case một state: đây chỉ là Gaussian plug-in simple-regret consistency check, không phải best-arm sample-complexity theorem.
- Sửa chứng minh Mills ratio tại `z=0`.
- Sửa A3 cho khớp định nghĩa global loss và bảng alternative.
- Rút chiều đảo sai của Định lý 2; đổi “axioms cần và không đủ” thành “design desiderata, không phải necessary-and-sufficient theorem”.
- Bỏ phát biểu có “bootstrap stability certificate” vì dự án chưa có artifact/code chứng minh certificate đó.
- Sửa tài liệu dẫn xuất để `Q^(0)=Q^{beta_hat^K}` khớp code.
- Sửa bảng frozen configuration từ `n_min=5` sai thành primary `n_min=2`, robustness 3/5, đồng thời ghi riêng `min_trajectory_support=2`, khớp `PLAN_Final.md` và `protocol_config.json`.
- Thu hẹp diễn giải positive control và corridor đúng với điều từng test đo.

## Checklist nghiệm thu H1

- [x] Phân biệt statistic saturation với learner learnability bằng số liệu expanded development.
- [x] Mọi định lý/corollary nêu rõ giả thiết.
- [x] Mọi kết luận thực nghiệm chỉ đúng cohort và mẫu số tương ứng.
- [x] Định nghĩa LaTeX đối chiếu với đường code thực thi.
- [x] Positive control và corridor có giới hạn diễn giải.
- [x] Figure 1 có script, dữ liệu số và hash truy nguyên.
- [x] Không dùng hoặc mở evaluation split.
- [x] Không tuyên bố neural SOTA, quan hệ nhân quả hay held-out generalization.

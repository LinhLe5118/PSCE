# H1 — Công thức → giả định → code → giới hạn

08/10/2026, baseline `a0a0d13`, bản đối chiếu nội bộ của Đỗ Trọng Hưng.
[Glossary và chứng minh có điều kiện](H1_THEORY_GLOSSARY_2026-10-08.md) định nghĩa
ký hiệu; [T1 v1.1.0](T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) giữ quyền
quyết định estimand/gate. Link trỏ tới **hàm**, không dùng line number dễ lệch.

## 1. Nguồn implementation

| ID | Đại lượng/công thức | Giả định/điều kiện | File::hàm | Giới hạn cần viết |
|---|---|---|---|---|
| F01 | $J=\rho^\top(I-\gamma P_\pi)^{-1}r_\pi$; NR so với random/optimal | Discounted model hợp lệ, terminal absorbing reward 0, range dương | [tabular.py](../../code/oeh/tabular.py)::policy_values, normalized_returns, reference_returns | Đánh giá exact của mô hình, không phải return quan sát của pilot; collection horizon khác scoring discount |
| F02 | $E_l$ theo suffix 8/10 trên sáu budget | Cùng learner/config, đầy đủ 10 seed ở mỗi budget | [compare_difficulty.py](../../scripts/analysis/compare_difficulty.py)::suffix_event; [predictive.py](../../code/oeh/predictive.py)::suffix_event | Category 6 là right censoring; không suy ra probability quần thể hay vô cực thật |
| F03 | Lắp outcome → labels → forecast scores | Job/seal hợp lệ, không bỏ job lỗi hoặc seed không thành công | [run_oeh_120_unseen.py](../../scripts/analysis/run_oeh_120_unseen.py)::analyze | Helper historical được current runner configure lại; không dùng gate cũ thay gate cohort 7 |
| F04 | Dirichlet posterior theo count và concentration fit | Multinomial row model, prior structure đã chọn, reward factorization nhất quán trên pilot | [posterior.py](../../code/oeh/posterior.py)::pilot_statistics, fit_posterior, sample_mdps | Empirical Bayes/model assumption, không là confidence set; unobserved reward 0 là prior convention |
| F05 | Replay learner trên nested streams trong posterior draws | Budget/seed/terminal/restart đúng protocol, conditioning dynamic range | [predictive.py](../../code/oeh/predictive.py)::posterior_outcomes; [posterior.py](../../code/oeh/posterior.py)::simulate_streams | Mô phỏng không là outcome độc lập của MDP thật; báo requested và returned |
| F06 | $(n_c+1/2)/(B_{eff}+7/2)$ | Số event thực tế của estimator | [predictive.py](../../code/oeh/predictive.py)::requirement_forecast, kt_probabilities | Zero-returned có censored fallback; smoothing không chứng minh calibration |
| F07 | Posterior mean $(\bar P,\bar R,\bar\beta)$ | Single/mixture priors từ frozen structure | [develop_oeh_horizon.py](../../scripts/analysis/develop_oeh_horizon.py)::posterior_mean_model | $\bar\beta>0$ có thể bao gồm action chưa quan sát |
| F08 | $Q_1=Q^{\bar\beta}$, $Q_{k+1}=\mathcal T_{mod}Q_k$, $\pi_k=\arg\max Q_k$ | Fixed model/support, terminal continuation 0; finite scan | [horizon.py](../../code/oeh/horizon.py)::offline_effective_horizon | Current k bắt đầu 1; historical k_step_backups bắt đầu 0; không gọi min depth là EH gốc |
| F09 | $d_\beta^{stream}=$ expected visits / expected episode length | Logging theo stationary behavior, restart và horizon đã cho | [horizon.py](../../code/oeh/horizon.py)::stream_visitation | Không phải discounted occupancy; trên posterior model không là tần suất thật được chứng nhận |
| F10 | $\nu_k=\sum W_k^2\sigma_k^2/d_\beta^{stream}$ | Delta-method, plug-in Bellman target variance | [horizon.py](../../code/oeh/horizon.py)::offline_effective_horizon | Bỏ qua covariance/trajectory dependence trong approximation |
| F11 | Misrank $\Phi(-\Delta\sqrt N/\sqrt{\nu_1+\nu_2})$ | Gaussian approximation, fixed comparison | [horizon.py](../../code/oeh/horizon.py)::_misrank_probability | Không có uniform CLT/PAC certificate; zero/inf variance có branches riêng |
| F12 | $\widetilde\ell_k$, $\widetilde N_k$, $h_H$ | Risk form, NR slack, A>1, range dương | [horizon.py](../../code/oeh/horizon.py)::_risk_requirement, offline_effective_horizon | Model risk inversion, không cận mẫu learner; numerical search có trần |
| F13 | $\widetilde\ell(N_j)$ và exponential missing mass | Converged model policy; independent-draw approximation của missing mass | [horizon.py](../../code/oeh/horizon.py)::risk_profile | Profile trong candidate không đồng nghĩa runner dùng coverage_risk form |
| F14 | Actual OEH-H gọi trên posterior mean và oracle riêng | `form="risk"`, horizon=100, tau=.95 | [run_oeh_final_120.py](../../scripts/analysis/run_oeh_final_120.py)::horizon_group | Observed mask chỉ truyền riêng cho asymptotic_ceilings; không áp nhầm lên mọi horizon feature |
| F15 | B2: 18 generic pilot statistics | Pilot arrays hợp lệ; bootstrap theo trajectory | [stochastic_features.py](../../code/stochastic_features.py)::extract_stochastic_features; [compare_difficulty.py](../../scripts/analysis/compare_difficulty.py)::FEATURES_B2 | Bộ do nhóm xây, không phải published SOTA; bootstrap feature không làm trajectory độc lập |
| F16 | Feature union `plus_prior_profile`, 108 cột | Tên/thứ tự khớp protocol; nonfinite encoding giữ nguyên | [run_oeh_final_120.py](../../scripts/analysis/run_oeh_final_120.py)::use_candidate; [oeh_final_forecast.py](../../scripts/analysis/oeh_final_forecast.py)::oeh_feature_rows | Có diagnostics lịch sử/cross-learner; chưa ablation attribution từng khối |
| F17 | $p_l=\sum_t w_t p_{l,t}$, $w$ fit OOF log loss | Grouped folds theo MDP, seven opened cohorts, frozen capacity | [oeh_final_forecast.py](../../scripts/analysis/oeh_final_forecast.py)::matched_forecast | Cùng procedure, không cùng fitted weights; `candidate_law=false` |
| F18 | Matched vs AFT-only/frequency/privileged | Actual FIELD_SETS/all_forecasts là nguồn kiến trúc | [run_oeh_final_120.py](../../scripts/analysis/run_oeh_final_120.py)::all_forecasts | JSON historical "every fitted comparator" quá rộng; không sửa JSON niêm phong |
| F19 | IBS trên sáu CDF, seven-class LL | Probabilities hợp lệ; log floor 1e-12 | [develop_oeh_difficulty.py](../../scripts/analysis/develop_oeh_difficulty.py)::score_probabilities | Chấm forecast, không chấm policy; không lấy Brier nhóm làm Brier paper EH |
| F20 | GO nếu 22 simultaneous lower bounds >0 | Paired MDP clusters; registered seventh-cohort gate | [analyze_oeh_confirmatory_20261004.py](../../scripts/analysis/analyze_oeh_confirmatory_20261004.py)::main | Không áp điều kiện >=5% của cohort trước; không tự nâng GO thành SOTA |
| F21 | Selector: success + .15 return, switch advantage >.1 | ExtraTrees và family label đã seal | [run_oeh_confirmatory_20261004.py](../../scripts/analysis/run_oeh_confirmatory_20261004.py)::fixed_selector; [develop_fixed_selector_20261004.py](../../scripts/analysis/develop_fixed_selector_20261004.py)::design, model | Secondary, không pilot-only; v5 comparison chưa vượt trội |
| F22 | $\kappa_k=A^{-k}$ dưới uniform behavior | A cố định, k action choices, kernel hợp lệ | [horizon.py](../../code/oeh/horizon.py)::lookahead_cost | Chỉ rollout-cost identity; không chứng minh OEH-H bằng EH hoặc offline theorem |

## 2. Quyền truy cập thông tin

| Hàng | Inputs | Forecasting head | Nhãn đúng |
|---|---|---|---|
| OEH, B2 matched, B2+ERI, B2+BWD | Pilot/public contract | AFT + direct thresholds + depth3 multiclass | Matched pilot-only feature comparison |
| B2 generic, coverage/quality adaptations | Pilot | AFT-only | Supplementary, không same-head attribution |
| Train-frequency | Meta-training labels | Class frequency | No fitted matched head |
| B2+EH oracle | MDP thật | Matched | Privileged diagnostic/adaptation |
| B2+TQ/SACo original | Author-code metrics + true-MDP references | Matched | Privileged feature augmentation, không end-to-end reproduction |
| Selector | Pilot-derived features + probabilities + family metadata | ExtraTrees | Secondary selection task |

## 3. Những annotation lịch sử chưa được phép sửa

1. Docstring của `oeh_final_forecast.py` gọi generic B2 matched quá rộng;
   `all_forecasts()` thực tế gọi AFT riêng cho `b2_generic`.
2. `predictive.py`/`posterior.py` có nhãn "hyperparameter-free". H1 không dùng
   nhãn đó cho whole forecaster hoặc prior structure đã chọn trong development.
3. Một số branch `prepare()` còn in "fourth" nhưng được configure cho cohort 7.
   Không lấy display string làm định nghĩa cohort.
4. `verify_oeh` kiểm toán bundle router lịch sử; trường `confirmation=false`
   của nó không thay thế verdict cohort 7. PASS không nghĩa có đủ raw inputs.
5. `identified` và `coverage_trap` là status descriptor; feature encoding có
   thể biến nonfinite thành fallback số. Người đọc phải đọc cả status và value.

Không sửa source đã pin hoặc historical seal chỉ để đồng bộ wording. Nếu sửa
logic sau H1 thì cần scientific revision/protocol riêng. Các annotation này
được xử lý ở tài liệu Method/glossary, không đổi thuật toán.

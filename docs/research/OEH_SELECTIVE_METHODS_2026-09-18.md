# OEH — thử cơ chế từ phương pháp mới, 18/09/2026

## Phạm vi và nguồn đã đọc sâu

Vòng KL trước chỉ thay confidence intervals và proposal; chưa thử thuật toán
selective propagation. Vòng này đọc phần phương pháp, điều kiện và pseudocode,
triển khai hai cơ chế, rồi kiểm tra trên cùng 225 development pilots.

| Nguồn | Phần đã đối chiếu | Kết luận áp dụng |
|---|---|---|
| [Selective Uncertainty Propagation, AAAI 2025; arXiv v4](https://arxiv.org/html/2302.00284v4) | §2–5, Theorem 4.1, Appendix A/B | Đã triển khai SPVI proposal từ Appendix B. Thêm cận chênh lệch discounted theo suy luận riêng dưới đây. |
| [Don't Be Pessimistic Too Early, AISTATS 2024](https://proceedings.mlr.press/v238/wang24h/wang24h.pdf) | Definitions 1–3, Theorem 1, §3.2, Algorithm 1, limitations | Đã triển khai kiểm tra điều kiện support của lookahead. Chưa huấn luyện LP-MDP. |

SUP gắn phần bất định tương lai với độ lệch transition so với reference. Theorem
4.1 cần các cận đầu vào hợp lệ và holdout độc lập; chỉ thay variance bằng estimated
shift không đủ. LP-MDP yêu cầu true reachable set nằm trong learned lookahead set.
Support quan sát trong pilot ít mẫu không tự thỏa điều kiện này. Đó là các điều
kiện quan trọng khi chuyển phương pháp vào hệ thống hiện tại.

PDF nguồn và SHA256 được lưu cục bộ tại
[source provenance](../../artifacts/research/oeh_new_methods_sources_2026-09-18/).
Đây là reimplementation/adaptation từ bài báo, không phải chạy code gốc tác giả,
không phải full replication benchmark của hai bài.

## 1. SPVI proposal: đã triển khai thuật toán chọn action

Giữ ba value arrays: nominal, lower, upper. Với empirical transition `P_hat`,
reference `beta_hat`, đặt `shift_hat = P_hat(a|s) - P_hat(beta_hat|s)` theo trục
next-state. Điểm chọn action của SPVI trừ local bonus và trừ phần truyền bất định
`gamma * abs(shift_hat) @ (V_upper - V_lower)`. PVI control dùng nominal reward
và transition với lower future value, trừ cùng local bonus.

Adaptation trong repository:

- Dùng discount 0.95; backward recurrence từ leaf 0 với horizon H=1..101.
  Lấy first-step action map ở mỗi H thành stationary receding-horizon candidate.
  Không đánh tráo value của nonstationary finite-horizon policy thành value của
  stationary policy: mọi candidate được chấm lại bằng exact discounted evaluator.
- Khi gamma=1, recurrence khớp finite-horizon pseudocode cho các inputs được cấp.
  Các action chưa gán rõ trong value updates của pseudocode được hiểu là action
  vừa chọn tại state đó.
- `b(s,a)=sqrt(log(S*A*101/delta)/max(n(s,a),1))`, cùng PVI/SPVI và mọi H.
  Đây là **proposal bonus**, không tuyên bố là cận high-probability hợp lệ cho
  benchmark discounted này. Không dùng nó để phát certificate.
- Beta là empirical reference; unknown rows dùng uniform actions. Unknown
  dynamics dùng reward 0/self-loop chỉ ở proposal. Terminal tuân theo adapter.
- Global regret certificate vẫn dùng KL model confidence set chứa unseen mass,
  giữ epsilon=delta=0.05 và public reward bound 1.

H=1 tương ứng index depth 0 của SPVI/PVI; khác leaf empirical-behavior value ở
family empirical cũ. Vì vậy báo cả family, index và các baseline, không diễn giải
chênh lệch depth đơn thuần là cải thiện OEH.

## 2. Cận chênh lệch giá trị với bất định chung được triệt tiêu

Đây là suy luận discounted riêng, lấy cảm hứng từ cơ chế shift-dependent
propagation; **không phải Theorem 4.1 của SUP**. Không cần giả định đã biết behavior
thật vì reference là policy cụ thể, cũng có thể được chọn từ pilot.

Với candidate pi và reference b trong cùng một MDP, reward đã chia public Rmax:

```text
D = V_pi - V_b
g = r_pi - r_b + gamma * (P_pi - P_b) V_pi
D = g + gamma * P_b D
```

Từ model set C và robust policy evaluation, có vector `L <= V_pi <= U` cho mọi
model trong C. Đặt `m=(L+U)/2`, `h=(U-L)/2`, `w=pi-b`. Lấy cận reward difference
bằng cách nhân reward endpoints với dấu của w; các action dùng chung triệt tiêu
trước khi lấy cực trị. Tính cận của `(P_pi-P_b)m` qua box-simplex optimization.

Nếu `[d_lower,d_upper]` bao phủ từng coordinate của true transition difference,
thì phần dư value được chặn bởi:

```text
abs((P_pi-P_b)(V_pi-m))
 <= min(sum_j max(abs(d_lower_j),abs(d_upper_j))*h_j,
        sum_a abs(w_a) * max_j h_j).
```

Hai vế đều là cận hợp lệ: vế thứ nhất dùng coordinate intervals; vế thứ hai dùng
`||sum_a w_a P_a||_1 <= sum_a |w_a|`. Suy ra `g_lower <= g <= g_upper`. Robust
evaluation với transition của reference b và signed auxiliary reward g cho cận
trên/dưới của D, nhờ tính đơn điệu của discounted resolvent. Cuối cùng giao với
interval lấy hiệu hai robust value bounds độc lập; cận mới không rộng hơn cũ.

Sự kiện true model thuộc C đồng thời bao phủ tất cả candidate/reference, kể cả
policy phụ thuộc dữ liệu, nên không cần holdout để áp dụng **suy luận này**. Đây
không phải cách bỏ holdout khỏi theorem gốc: ở đây dùng uniform model confidence
set và một estimand/derivation khác. Không cài true P/R vào estimator.

Khi hai policy giống hệt nhau, trả `[0,0]` theo đồng nhất thức. Trường hợp đó bị
loại khỏi số comparison được báo là thu hẹp không tầm thường. Hai action có
cùng transition đã biết cũng triệt tiêu phần tương lai chung, dù giá trị tương
lai chưa biết. Trái lại, estimated transitions giống nhau nhưng còn sai số không
được coi là true shift bằng 0.

Giá trị được bound là `(1-gamma)*(J_pi-J_b)/Rmax` trong `[-1,1]`. Certificate
improvement cần lower bound >0 (implementation dùng tolerance 1e-9). **Nó không
chứng nhận gần policy tối ưu, không phải dự báo số mẫu và không thay global OEH.**

## 3. Lookahead pessimism: kiểm tra điều kiện có thể áp dụng

Cho mỗi row, lấy upper bound total variation từ KL box:

```text
tv_upper = min(1, 0.5*sum_j max(abs(P_hat_j-P_lower_j),
                              abs(P_hat_j-P_upper_j))).
```

Khóa `xi=.25`, K={1,2,5,10}, policy empirical depth 100. Row là uncertain nếu
tv_upper >= xi. Kiểm tra liệu mọi state có thể đi tới sau k bước đều chọn action
certain, so sánh hai loại support:

1. Nominal support `P_hat > 0`, có thể bỏ sót successor thật.
2. Confidence-possible support `P_upper > 0`, là superset bảo thủ.

Đây là applicability audit, không tự nó là performance bound của LP-MDP. Mốc xi
chỉ để kiểm tra cơ chế; không tuning theo return và không dùng để phát certificate.
Nếu dùng nominal support mới thấy rescue còn confidence support không rescue,
không được báo điều đó là thoát bão hòa có bảo đảm. Tập lookahead từ C có thể rất
rộng; đó là chi phí của việc chưa biết support trong offline stochastic data.

## Protocol và tái lập

225 pilot lịch sử = 15 MDP × 5 behavior × 3 prefix; 165 stochastic và 60 lock
controls. Estimation chạy trước oracle scoring mỗi pilot. Ba proposal families
empirical/PVI/SPVI, mỗi family 101 candidates; model set KL chung với delta .05.
Reference chung là empirical depth 0. Safe selection chọn candidate có coupled
lower bound lớn nhất nếu >1e-9, nếu không trả reference. Global selector giữ quy
tắc chọn min U từ vòng trước. Plan/source/input hashes được ghi trước execution.

Không mở evaluation split; không xem development này là held-out. Các pilot cùng
MDP phụ thuộc nhau. Kết quả cần kiểm tra đồng thời: policy diversity, variation
của global U, global certificates, interval narrowing cho policy khác reference,
certified improvement, true regret và fixed-depth comparisons.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -m scripts.analysis.run_oeh_selective --dataset-dir datasets/week1_development_validation --output-dir artifacts/research/oeh_selective_NEW_RUN
python -m pytest tests/test_oeh_selective.py tests/test_oeh_refinements.py tests/test_oeh_certificate.py tests/test_oeh_certificate_adapter.py tests/test_exact_solver.py tests/test_evaluation_leakage.py tests/test_oeh_estimator.py tests/test_oeh_validation.py -q
```

Mã: [oeh_selective.py](../../code/oeh_selective.py),
[runner](../../scripts/analysis/run_oeh_selective.py),
[tests](../../tests/test_oeh_selective.py).
Targeted suite: 105 tests và 7 subtests đạt. Trong đó 10 tests mới kiểm tra
triệt tiêu shared future, exact stochastic differences, containment trên sampled
models, unseen actions, reward scaling, delayed rewards, zero-shift propagation,
support mismatch và invalid inputs. Đây chưa phải full repository test suite.

## Kết quả execution

**Đã hoàn tất** 225 pilot × 3 family × 101 depth = 68.175 hàng chấm.
[Báo cáo](../../artifacts/research/oeh_selective_2026-09-18/REPORT.md),
[report.json](../../artifacts/research/oeh_selective_2026-09-18/report.json),
[audit.json](../../artifacts/research/oeh_selective_2026-09-18/audit.json) và
[biểu đồ PDF](../../artifacts/research/oeh_selective_2026-09-18/comparison.pdf).

Trên **165 stochastic pilots**:

| Proposal | Nhiều policy theo depth | Global U không phẳng | Range U >0.001 | Mean regret, fixed depth 100 | Global certificate |
|---|---:|---:|---:|---:|---:|
| Empirical | 134 | 66 | 2 | **0.024123** | 0 |
| PVI | 29 | 10 | 0 | 0.055308 | 0 |
| SPVI adaptation | 113 | 30 | 0 | 0.057234 | 0 |

SPVI tăng policy diversity so với PVI, nhưng không thắng về chất lượng: ở fixed
depth 100, tốt hơn PVI trên 47 pilot, kém hơn trên 51, ngang nhau trên 67. Mean
regret cao hơn 0.001926. Nếu chọn min global U, mean regret tương ứng empirical,
PVI, SPVI là 0.028549, 0.057837, 0.058320. Không có bằng chứng trong phép thử này
ủng hộ thay empirical planning bằng bản SPVI chuyển thể.

Coupled interval **không thu hẹp thêm so với independent interval cho bất kỳ
comparison khác reference nào** trong cả 225 pilot. Không có certified
improvement; safe selector luôn fallback về empirical depth 0. Interval hẹp nhất
cho policy khác reference vẫn rộng khoảng 0.972576 trên thang difference [-1,1].
Các case pi=reference được xử lý đúng thành [0,0] nhưng không tính là thành công.
Điều này cho thấy phép triệt tiêu hợp lệ về toán học vẫn chưa đủ với confidence
set hiện tại, không phủ định mọi cách triển khai SUP.

LP applicability audit: trên 16.860 row-cases có 16.046 row uncertain. Nominal
support cho 518 row được rescue ở K=1, 5.144 row ở K=10; confidence-possible
support cho **0 row ở mọi K đã thử**. Đây là thiếu chứng cứ cho điều kiện support,
không phải oracle đã chứng minh mọi nominal rescue là sai. Chưa đủ cơ sở đưa
LP-MDP vào với guarantee nếu chỉ dùng support đã quan sát.

Không có observed global/difference bound violation trên 68.175 hàng; không có
certificate để đánh giá false-positive rate. 22.725 điểm empirical baseline khớp
vòng KL cũ cả upper bound và oracle regret. Hash source/input/CSV đã xác minh.
Wall time vòng chính 967.51 giây; đây là một run prototype, có thời gian cạnh tranh
tài nguyên với probe khác, không phải benchmark runtime độc lập.

### Quyết định kỹ thuật sau vòng mới

Giữ các implementation mới để tái lập và ablation, **chưa bật làm OEH mặc định**.
Hai điểm cần thay đổi thực chất nếu nghiên cứu tiếp:

1. Ước lượng **shift hoặc value contrast trực tiếp**, thay vì dựng hai model
   confidence boxes rộng rồi lấy hiệu. Phải có cận cho sai số shift; equality hay
   pooling giữa action cần giả định public hoặc kiểm soát sai số model selection.
   Một điểm estimate có shift nhỏ không tự chứng minh shift thật nhỏ.
2. Nếu theo LP-MDP, cần support contract được biết trước hoặc inference về missing
   successor mass có bảo đảm, rồi đưa phần sai số ấy vào cận. Không lấy absence
   trong log làm bằng chứng transition không tồn tại.

Phải đăng ký nhánh thử tiếp theo trước khi xem outcome của nó, giữ empirical
planning/PVI/fixed depths làm đối chứng và chấm cả utility. Kết quả hiện tại chỉ
cho phép nói **các bản chuyển thể đã thử chưa giải được bão hòa**, không nói
phương pháp gốc vô dụng, không nói stochastic tabular không thể làm được. Hướng
chính B1/B2 và phép kiểm tra giá trị bổ sung B3 vẫn giữ như tài liệu định hướng.

### Tối ưu tính toán và trạng thái các run

[Vectorized water filling](../../code/oeh_fast_expectation.py) có 9 tests đối chiếu
với hàm gốc. Targeted suite khi gắn kernel đó đạt **114 tests và 7 subtests**.
Lượt full-run song song bằng kernel mới được dừng để ưu tiên vòng chính, giữ
[INTERRUPTED.md](../../artifacts/research/oeh_selective_fast_2026-09-18/INTERRUPTED.md);
không có scored CSV/report hoàn chỉnh, không dùng nó làm bằng chứng speedup.
Các kết quả khoa học ở bảng trên đều từ vòng gốc hoàn chỉnh.

### Probe cơ chế bổ sung: shared stochastic future

[Probe có dữ liệu thật được mô phỏng](../../artifacts/research/oeh_shared_future_probe_2026-09-18/REPORT.md)
dùng một topology 3 state, 2 action, 3 seed và bốn ngân sách cố định
20/200/2.000/20.000 lượt thăm mỗi action tại root. Hai root actions có cùng true
successor distribution nhưng estimator không được biết equality đó. Các future
actions chưa quan sát. True normalized improvement là 0.03; true global regret
là 0.475. Không có certificate improvement trong 12 cases, kể cả budget lớn nhất.

Đây là failure analysis có ích: triệt tiêu shared future thành công ở test với
transition equality biết chính xác, nhưng cận sai số của transition difference
từ hai empirical action samples còn rộng. Không tăng budget sau khi xem kết quả
để chỉ báo một case thành công. Cần nghiên cứu estimator cho shift trực tiếp hoặc
structure sharing có kiểm soát sai số nếu muốn khai thác trường hợp này tốt hơn;
không tự đặt shift bằng 0 khi hai sample means gần nhau.

# Nghiên cứu lại OEH cho stochastic tabular Offline RL

Ngày: 18/09/2026. Trạng thái: **thiết kế nghiên cứu và prototype development**.
Đây là đề xuất sau khi đã biết kết quả OEH-v1; không phải preregistration lịch sử,
không phải kết quả held-out, và chưa chứng minh novelty hay chất lượng A/A*/Q1.
Không sửa định nghĩa hoặc artifacts của OEH-v1.

Tích hợp sau prototype: [smoke test trên development trajectories](OEH_DEVELOPMENT_SMOKE_2026-09-18.md)
đã có adapter terminal/truncation, runner so cùng pilot với OEH-v1 và đường dẫn
artifact. Đây là bước mới sau các synthetic probes bên dưới; không thay đổi
provenance/source hashes của những probe đã lưu.

## 1. Quyết định thiết kế

Chuyển nền tảng từ **xác suất đảo thứ hạng của hai action ước lượng** sang
**regret của policy trong các mô hình còn phù hợp với dữ liệu**. Độ sâu là số
backup của một họ policy được khai báo rõ, còn evidence là số dữ liệu cần để
đạt một mục tiêu regret có xác suất thành công xác định. Không cộng hai đại
lượng bằng `k + log_A(m)` khi chưa có lý do chi phí offline tương ứng.

Đầu ra ưu tiên là đường cong cận regret theo depth và trạng thái chứng nhận.
Tên làm việc: **OEH-R, offline regret-certified depth profile**. Đây là ứng viên
cho B3 trong hướng nghiên cứu chung, chưa thay B2 bằng một phương pháp đã xác nhận.

Một công thức không thể được bảo đảm là predictor tốt chỉ bằng lựa chọn đại số.
Phần có thể bảo đảm ở đây là ngữ nghĩa regret và mệnh đề chứng nhận có điều kiện;
độ chặt, predictive utility và sample-requirement forecasting cần thực nghiệm riêng.

## 2. Những điểm phải sửa trong lời giải thích và đề xuất trước

1. `m=1` khi dữ liệu thật sự đã đủ là đúng. Mọi statistic về *dữ liệu cần thêm*
   nên có thể chạm floor này. Lỗi của OEH-v1 là suy ra planning difficulty hoặc
   learner sufficiency từ một proxy không kiểm soát các đại lượng đó.
2. Chia loss cũ cho value scale chỉ sửa đơn vị. Nó không thêm planning bias,
   không phục hồi unseen transitions, và không biến Gaussian tail thành guarantee.
3. Lọc `Delta/V >= f`, rồi lấy quantile của `Phi(-z sqrt(mu))`, vẫn có thể loại
   một quyết định hiếm nhưng quan trọng. Empirical gap bằng 0 không chứng minh
   true tie; top-two gap không kiểm soát mọi action và error propagation.
4. Bootstrap variance bằng 0 khi pilot chỉ thấy một successor không chứng minh
   dynamics deterministic. Resampling không tạo ra successor chưa xuất hiện.
5. Một policy có ranking hoàn toàn ổn định vẫn có thể sai vì leaf value và depth
   chưa đủ. Cận uncertainty giảm về 0 phải để lại **actual planning regret**.
6. Trong confidence model, thiếu coverage không tự động chứng minh unlearnable.
   State không liên quan hoặc nhiều action gần tối ưu có thể không cần phân biệt.
   Cần giữ uncertainty và đánh giá hậu quả lên return, thay vì bỏ state đó.

Vì vậy đề xuất quantile trong trao đổi trước chỉ là heuristic cần kiểm tra,
không được sử dụng làm công thức chính đã có cơ sở bảo đảm.

## 3. Estimand: độ khó của cái gì?

Giả định MDP hữu hạn, stationary, fully observed; `S,A,gamma,rho0` được biết;
`0 <= reward <= R_max` với bound được biết từ task, không lấy observed maximum.
Prototype dùng discounted continuing representation. Episodic terminal phải được
chuyển thành absorbing zero-reward state theo contract trước khi áp dụng; reset
collector không phải transition của MDP. Truncation không phải termination.

Đặt public value scale và normalized control regret:

\[
 B=\frac{R_{\max}}{1-\gamma},\qquad
 \mathcal R_M(\pi)=\frac{J_M^*-J_M(\pi)}{B}\in[0,1].
\]

`epsilon` là mức regret chấp nhận được; `delta` là xác suất certificate có thể sai.
Đây là hai khái niệm khác nhau. `epsilon=.05` ở đây **không tương đương**
`NR >= .95` của benchmark, vốn chia cho `J* - J(random)`.
Không dùng oracle denominator đó ở phía estimator. Với known zero reward bound,
bài toán có regret bằng 0; prototype yêu cầu bound dương và xử lý bài toán này
ngoài API. Positive reward rescaling phải scale cả `R_max`.

Gọi `A_k` là procedure tạo policy từ dataset ở depth `k`, bao gồm baseline,
backup, pessimism, tie-breaking và preprocessing. Tách ba đại lượng:

**Population planning depth**, nếu đã khai báo policy ở mô hình chính xác:

\[
 h_\epsilon^{\rm oracle}(M,\beta)
 =\min\{k:\mathcal R_M(\pi_k^{M,\beta})\le\epsilon\}.
\]

Đây là first acceptable depth, không phải suffix-settling depth. Hai định nghĩa
không được dùng lẫn nhau khi đường cong không monotone.

**Population evidence requirement cho một procedure cụ thể**:

\[
 N_{\rm req,k}(M,\beta;\epsilon,\delta)
 =\inf\left\{n:\ \forall n'\ge n,\quad
 \Pr_{D_{n'}\sim\mathcal D(M,\beta)}
 [\mathcal R_M(A_k(D_{n'}))\le\epsilon]\ge1-\delta\right\}.
\]

Collection law gồm cách reset, truncate và đếm transitions, phải cố định.
Probability còn bao gồm learner randomness nếu có. Tập rỗng cho `+infinity`.
Đây là target lý thuyết, không phải con số đã tính được từ một pilot.

**Depth-evidence frontier**: các cặp `(k,N_req,k)` không bị cặp khác trội cả hai
chiều. Có thể báo `min_k N_req,k` riêng, nhưng không gọi nó là complexity riêng
của environment và không gán cho mọi learner.

Cùng một MDP nhưng behavior bỏ hẳn action cần thiết có thể không nhận dạng được;
behavior đủ coverage có thể học được. Muốn xếp hạng *environment* giữa các task,
cần cố định một behavior/collector tham chiếu, reward normalization và learner
family; nếu không phải ghi rõ độ khó của `(M,beta,A_k,epsilon,delta)`.

## 4. Công thức chính: robust normalized regret

Từ pilot xây dựng confidence set `C_delta(D)` trên **toàn bộ public action space**
và successor space, sao cho:

\[
 \Pr[M\in\mathcal C_\delta(D)]\ge1-\delta.
\]

Đối với một candidate `pi_hat_k(D)`, target robust là:

\[
 G_k(D)=\sup_{M'\in\mathcal C_\delta(D)}
 \frac{J_{M'}^*-J_{M'}(\widehat\pi_k)}{B}.
 \tag{R1}
\]

Hai return trong hiệu ở (R1) phải thuộc **cùng MDP**. Tối ưu trực tiếp có thể khó.
Prototype dùng relaxation tính được bằng robust dynamic programming:

\[
 U_k(D)=\min\left\{1,
 \frac{\sup_{M'\in\mathcal C}J_{M'}^*
       -\inf_{M'\in\mathcal C}J_{M'}(\widehat\pi_k)}{B}
 \right\}.
 \tag{R2}
\]

Hai extrema ở (R2) có thể đạt ở **hai MDP khác nhau** nên `U_k >= G_k` và có thể
rất bảo thủ. Không trình bày (R2) là exact worst-case regret. Numerical code thêm
contraction-residual allowances và kiểm tra convergence trước khi chứng nhận.

Định nghĩa depth chứng nhận được tại pilot hiện tại:

\[
 \widehat h^{\rm cert}_{\epsilon,\delta}(D)
 =\min\{k\in\{0,\ldots,K_{\max}\}:U_k(D)\le\epsilon\}.
 \tag{R3}
\]

Nếu tập rỗng, trả `uncertified_within_depth_grid`, không trả `h=0`, không suy
`N_req > N` và không tuyên bố planning horizon vô hạn. Mệnh đề đúng trong exact
arithmetic là:

> Với xác suất ít nhất `1-delta`, mọi candidate đạt `U_k<=epsilon` đều có
> true normalized regret không vượt `epsilon`, kể cả k/policy được chọn từ D.

**Chứng minh.** Trên event `M in C`, `J_M* <= sup_C J*` và
`J_M(pi) >= inf_C J(pi)`. Trừ hai bất đẳng thức và chia `B` cho bound. Regret luôn
không vượt 1 nên clipping an toàn. Event này đồng thời cho mọi policy; vì vậy
chọn depth trên cùng data không cần thêm union bound theo k nếu confidence set
ban đầu uniform trên mô hình. Floating-point implementation chỉ là xấp xỉ có
allowance, không phải một proof assistant/interval-arithmetic certificate.

Nếu `C={M}`, (R2) bằng **actual policy regret**. Depth ngắn có planning bias vẫn
thất bại ngay khi model uncertainty bằng 0: đây là khác biệt cần thiết với OEH-v1.
Nếu mọi action thực sự tốt như nhau, (R1) bằng 0. Với một action duy nhất, control
regret luôn bằng 0 dù model chưa biết; prototype xử lý trường hợp này trực tiếp.
Trong các trường hợp tie phức tạp, relaxation (R2) có thể chưa khai thác hết việc
triệt tiêu uncertainty giữa hai return; đây là hạn chế thực tế phải báo.

## 5. Xây dựng confidence set có stochasticity và unseen outcomes

Prototype nhận raw successor counts, reward sums và reward square sums. Không
nhận `P,R,Q*,V*`, manifest family hay oracle optimality làm feature.
Với mỗi `(s,a)`, `n` là số quan sát thật. Có `d=S*A*(S+1)` scalar means gồm
các successor indicators và normalized reward. Cho mỗi coordinate tại count n:

\[
 \alpha_n=\frac{\delta}{d\,n(n+1)}.
\]

Với `n>=2`, empirical Bernstein two-sided radius trên biến trong `[0,1]`:

\[
 b_n=\sqrt{\frac{2\widehat v_n\log(4/\alpha_n)}{n}}
       +\frac{7\log(4/\alpha_n)}{3(n-1)}.
 \tag{R4}
\]

Đây là ứng dụng Theorem 4 của Maurer--Pontil [S5] cho cả `X` và `1-X`.
`v_hat_n` là unbiased sample variance, không phải bootstrap standard error.
Hai phía chia alpha; log(4/alpha) đã tính phần này. Union theo `d` coordinate
và mọi count n hợp lệ vì `sum_{n>=1}1/[n(n+1)]=1`.
Hoeffding two-sided radius `sqrt(log(2/alpha_n)/(2n))` là ablation độc lập;
không chọn min hai cận sau khi nhìn dữ liệu mà quên confidence allocation.

Transition coordinate interval là `[p_hat-b,p_hat+b]` cắt vào `[0,1]`, rồi giao
với simplex. Reward interval tương tự, scale lại bằng `R_max`. Với `n=0`, giữ
full simplex và `[0,R_max]`; với `n=1`, Bernstein giữ full range. Một successor
chưa thấy vẫn có upper probability dương. Phần `1/(n-1)` giữ uncertainty khi
empirical variance bằng 0. Không suy deterministic từ một successor quan sát được.

**Trajectory dependence.** Không xem toàn bộ transitions là iid. Dưới stationary
fully observed Markov law, có thể biểu diễn lần ghé thứ i của mỗi `(s,a)` bằng
phần tử i trong một chuỗi outcome iid riêng cho pair đó. Dataset chỉ quan sát
một prefix với số lần ghé ngẫu nhiên. Cận uniform theo mọi prefix n xử lý số
đếm phụ thuộc trajectory/stopping time. Reward và next state trong cùng outcome
có thể tương quan. Outcome-dependent filtering, bootstrap fractional counts,
latent state/nonstationarity phá giả định này và không được nhận như chứng cứ
đủ điều kiện. Prototype không tự kiểm chứng được nguồn gốc raw counts.

Biết public `S,A` là cần thiết: unseen state phải có chỗ trong simplex. Nếu
`rho0` được ước lượng, cần confidence set cho rho0; prototype yêu cầu rho0 biết.
Known terminal-zero constraints có thể làm cận chặt hơn. Prototype ban đầu chưa
có trajectory adapter; bước tích hợp tiếp theo được mô tả trong tài liệu smoke
ở đầu trang. Nó chỉ dùng terminal flags trong pilot, không dùng oracle terminal list.

## 6. Vì sao phù hợp hơn với stochastic tabular?

Lý thuyết APVI [S2] cho intrinsic bound trong finite-horizon offline RL dựa trên
occupancy của policy tối ưu, coverage của behavior và variance của Bellman target.
Kết quả discounted [S3] cũng dùng pessimism/Bernstein và coverage; các theorem
này có giả định riêng, không tự động chuyển thành guarantee cho OEH-R.

Đại lượng stochastic liên quan đến value là:

\[
 \nu_V(s,a)=\operatorname{Var}[R+\gamma V(S')\mid s,a].
\]

Nó gồm reward variance, `gamma^2 Var(V(S'))` và covariance giữa reward và next
value. Hai kernel có cùng entropy nhưng successor values lần lượt `(1,1)` và
`(0,2)` ở xác suất `(1/2,1/2)` cho variance continuation 0 và 1. Vì vậy không
đặt tiên đề “slip/entropy tăng thì OEH luôn tăng”: stochasticity còn đổi gap,
optimal policy và coverage. Thử nghiệm monotonicity chỉ hợp lý khi các biến
khác được giữ đúng theo phép can thiệp được thiết kế.

R4 thích ứng với variance từng coordinate, rồi robust backups truyền uncertainty
qua toàn chuỗi. Khi tất cả successor có cùng value, transition uncertainty không
đổi expected continuation. Tuy vậy coordinate boxes đánh mất một số covariance,
dùng union bound theo S, và chưa đạt intrinsic/instance-optimal tightness.
Variance-aware confidence set trực tiếp theo Bellman targets là nghiên cứu tiếp;
với V chọn từ data phải có uniform analysis, sample splitting hoặc lập luận
decoupling, không được áp cận iid cho V tùy ý sau khi fit.

## 7. Họ candidate và ý nghĩa depth

Prototype khởi tạo bằng pessimistic evaluation của baseline `beta`, rồi dùng
`T^- V(s)=max_a min_{(r,p) in C_sa}[r+gamma*p^T V]`.
Policy ở k=0 là lần greedy đầu từ baseline leaf value; k tăng thêm một backup.
Khi C là singleton, nó khớp cách đánh chỉ số Q-beta/backups ở OEH-v1 với **đầy đủ
action space**. Khi data hữu hạn, đây là họ candidate mới, không phải các policy
OEH-v1 support-restricted và cũng không phải CFQI/IQL.

Certificate có thể đánh giá policy CFQI/IQL/BC nếu policy được cung cấp, vì event
model coverage uniform cho mọi policy. Nhưng depth của họ candidate này không
tự động dự báo số iteration hay data requirement của các learner đó.
`min k` được chứng nhận cũng không phải ước lượng không chệch của oracle depth:
conservative uncertainty có thể làm nó lớn hơn hoặc chưa xác định.

## 8. Dự báo số dữ liệu cần thêm: không tạo chứng nhận giả

Không thay `n` bằng `mu*n` trong confidence set hiện tại rồi gọi đó là event có
coverage `1-delta` ở tương lai: chưa có observations đó, empirical center sẽ đổi,
rare successors sẽ xuất hiện, và pilot-zero coverage không chứng minh beta-zero.

Nếu cần forecast, bước riêng phải giả định collection law tương lai (cùng beta,
stationary MDP), mô hình hóa count allocation và unseen outcomes, rồi so dự báo
`N_req,k` với independent training streams. Có thể dùng calibration/survival B1,
B2, B3 như kế hoạch hiện hành. Prototype chủ ý trả `sample_requirement=None`.
Không đổi một bound chưa vượt epsilon thành right-censored true requirement;
đó chỉ là right-censored *certification stopping point* nếu thực sự quan sát một
chuỗi ngân sách theo protocol riêng. Conditional bound không phải minimax lower
bound hay statement về learner đang thất bại.

Phản ví dụ nhận dạng: một state, gamma=0, beta chỉ chọn action 0 có reward .5.
Hai MDP có action 1 reward 0 hoặc 1 cho cùng distribution của mọi log dưới beta.
Không một quyết định deterministic nào epsilon-optimal cho cả hai khi epsilon<.5;
với randomized policy, minimax regret là .25. Không công thức chỉ đọc log có thể
bảo đảm giải quyết sự mơ hồ đó cho epsilon<.25 nếu không thêm giả định.

## 9. Kiểm chứng đã làm

Code: [`../../code/oeh_certificate.py`](../../code/oeh_certificate.py).
Tests: [`../../tests/test_oeh_certificate.py`](../../tests/test_oeh_certificate.py).
Runner: [`../../scripts/analysis/run_oeh_certificate_probe.py`](../../scripts/analysis/run_oeh_certificate_probe.py).
Evidence hiện hành: [`../../artifacts/research/oeh_redesign_2026-09-18/verified/report.json`](../../artifacts/research/oeh_redesign_2026-09-18/verified/report.json).

17 unit tests kiểm tra LP water filling với scipy.linprog, singleton regret với
exact linear solve, reward rescaling, unseen action/successor, reward variance,
tie/one-action, malformed inputs, solver convergence và bound trên các MDP
sample trong confidence set.

Lệnh kiểm tra cùng các thành phần hiện có:
`python -m pytest tests/test_oeh_certificate.py tests/test_oeh_estimator.py tests/test_exact_solver.py -q`
đạt **51 tests và 2 subtests**. Đây là targeted verification, chưa phải full suite.

Phản ví dụ stochastic delayed reward dùng gamma=.9: action ngắn nhận .6; action
dài đến reward 1 với xác suất .8 sau một bước, tức optimal value .72. Baseline
leaf bỏ qua reward xa nên k=0 sai: normalized regret (.72-.6)/10=.012. Khi biết
model chính xác, k=1 sửa được và đạt epsilon=.005. Prototype phục hồi depth 1;
uncertainty bằng 0 không ép mọi depth về “đủ tốt”.

Synthetic probe gồm 120 confidence sets (2 confidence methods, 5 visit budgets,
12 seeds), 480 depth rows. Có 0 false-certificate rows, 0 bound-violation rows,
36 rows được chứng nhận. Raw CSV và source hashes được lưu cùng report.
Đây là **một topology tổng hợp**, sampling theo pair có kiểm soát để kiểm tra cơ
chế; không phải offline benchmark trajectory collection, không phải 120 MDP độc
lập và không thay thế validation về difficulty hoặc coverage xác suất rộng hơn.

## 10. Protocol để quyết định giữ hay bỏ thiết kế

1. Viết adapter raw trajectories/terminal để confidence model khớp MDP contract;
   kiểm tra empirical coverage ở nhiều topology, collector và random sample size.
2. Matched factorial: topology, transition noise, behavior coverage và N; giữ
   topology cùng variants trong một fold. Thêm rare bottleneck, observed tie,
   unseen good action, stochastic successors cùng value, và shallow wrong policy.
3. Báo tỷ lệ false certification, tỷ lệ certify, độ rộng U-G, true regret và thời
   gian robust DP. Constant `U=1` có coverage tốt nhưng vô dụng, nên không chỉ
   báo validity; quá ít certificate là failure về informativeness.
4. So OEH-v1, normalized-v1, materiality/max_state, quantile heuristic, count-only,
   Hoeffding certificate và Bernstein certificate trên cùng cohort. B2/B3 so thêm
   predictive utility theo learner, không coi certificate là sample predictor.
5. Tách intrinsic planning depth, current-data policy quality và empirical learner
   requirement thành ba endpoint. Dùng chính normalized regret mới để chấm
   certificate; benchmark NR vẫn báo như endpoint riêng, không tráo normalization.
6. Đăng ký protocol/amendment trước confirmatory run, khóa tolerance/confidence,
   generators, candidate family, baselines và selection criteria. Evaluation cũ
   vẫn sealed; chưa chạy vào đây. Không chọn epsilon để ép hết saturation.

Nếu cận R2 quá rộng, ưu tiên nghiên cứu same-model regret hoặc value-directed
confidence sets; không giảm delta/epsilon hoặc bỏ unseen actions để tạo thành
tích. Nếu B3 không thêm thông tin so với B2, giữ phương pháp tốt hơn trong hệ thống
và báo OEH-R là negative result. Chưa có căn cứ hứa thiết kế này đạt A/A*/Q1.

## 11. Nguồn chính và phạm vi kế thừa

- **[S1]** Laidlaw et al. [Bridging RL Theory and Practice with the Effective Horizon,
  NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8be628bf719550b560de8bec9456e0b-Abstract-Conference.html),
  và [stochastic EH, ICLR 2024](https://arxiv.org/abs/2312.08369): động lực phân tích
  lookahead; không cung cấp dataset-only guarantee cho OEH.
- **[S2]** Yin & Wang. [Towards Instance-Optimal Offline Reinforcement Learning with
  Pessimism, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/212ab20dbdf4191cbcdcf015511783f4-Abstract.html):
  coverage, occupancy và Bellman variance cho instance-dependent analysis.
- **[S3]** Li et al. [Settling the Sample Complexity of Model-Based Offline
  Reinforcement Learning](https://arxiv.org/abs/2204.05275): minimax tabular bounds
  với pessimism/Bernstein, gồm discounted setting.
- **[S4]** Jin, Yang & Wang. [Is Pessimism Provably Efficient for Offline RL?,
  ICML 2021](https://proceedings.mlr.press/v139/jin21e.html): nền tảng uncertainty
  quantification và pessimistic policy learning.
- **[S5]** Maurer & Pontil. [Empirical Bernstein Bounds and Sample Variance
  Penalization, COLT 2009](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf), Theorem 4:
  hằng số confidence radius sử dụng trong prototype.

R1--R3 là đề xuất thiết kế ở tài liệu này; chứng minh coverage implication ở trên
là lập luận elementary từ confidence sets, không được nhận là theorem mới có
novelty đã xác minh. Đây là literature review tập trung cho thiết kế, không phải
systematic review đủ để kết luận chưa có phương pháp tương tự.

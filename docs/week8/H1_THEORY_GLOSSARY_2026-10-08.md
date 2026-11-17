# H1 — Lý thuyết và glossary của OEH hiện hành

**Người phụ trách:** Đỗ Trọng Hưng. **Ngày:** 08/10/2026.
**Phiên bản:** 1.0.0, bản bàn giao nội bộ; không phải đăng ký thí nghiệm mới.
**Baseline đối chiếu:** `a0a0d13`; [T1 v1.1.0](T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md)
và [protocol cohort 7](../../artifacts/oeh_confirmatory_unseen_120_2026-10-04/protocol.json).

Phạm vi H1: chuẩn hóa ký hiệu, phân biệt estimand/descriptor/forecast, đối chiếu
công thức với implementation và nguồn học thuật. Không thay feature, prior,
ngưỡng, learner, seed, gate hoặc artifact đã niêm phong. Việc chuẩn bị tài liệu
không có nghĩa Hưng hoặc các thành viên đã ký duyệt; cần phản biện nội bộ.

## 1. Câu hỏi và luồng thông tin

Từ pilot của behavior policy, dự báo phân phối ngân sách đầu tiên mà một learner
cụ thể đạt tiêu chí return ổn định trên lưới đã đăng ký. Đây là **nhu cầu dữ liệu
thực nghiệm có điều kiện**, không phải độ khó tuyệt đối hay cận mẫu phổ quát.

```text
Môi trường + behavior
  ├─ pilot D0 → posterior / OEH-B / OEH-H / statistics → forecast → seal
  └─ training độc lập → learner × budget × seed → return → nhãn E_l
                                                       │
                   dự báo đã seal ← đối chiếu ──────────┘
```

MDP thật dùng để tạo dữ liệu và chấm outcome. Những feature oracle dùng MDP thật
phải nằm ở tier riêng. Outcome của các cohort đã mở được phép dùng để meta-train;
outcome của cohort đánh giá mới không được chọn feature, trọng số hoặc threshold.
Pilot x10 cũng có chi phí: dự báo các lớp x1/x2/x5 là hồi cứu trên stream training
độc lập, không phải bằng chứng tiết kiệm dữ liệu trước khi thu pilot.

## 2. Glossary dùng chung

| Ký hiệu/thuật ngữ | Định nghĩa hiện hành | Điều không được đánh đồng |
|---|---|---|
| $M=(\mathcal S,\mathcal A,P,R,\gamma,\rho)$ | MDP tabular hữu hạn; $P(s'\mid s,a)$ và $R(s,a,s')$; $\rho$ là phân phối khởi đầu | Tabular không đồng nghĩa deterministic |
| $S,A$ | Số trạng thái và hành động; $S=|\mathcal S|$, $A=|\mathcal A|$ | Không phải số mẫu |
| $\beta(a\mid s)$ | Behavior policy sinh log | Khác policy learner trả về và khác posterior-mean behavior |
| $D_0$ | Pilot log gồm $N_0=10SA$ transition | Không phải training buffer dùng chấm nhãn |
| Transition / trajectory | Một bước $(s,a,r,s')$ / một chuỗi bước kết thúc hoặc bị truncate | Budget đếm transition, không đếm episode |
| $l$ | Learner với implementation/config đã đăng ký; primary là BC, CFQI, IQL | Không phải policy hay thuật toán OEH |
| $b_j,N_j$ | $b=(1,2,5,10,20,50)$, $j=0,\ldots,5$; $N_j=b_jSA$ | Đây là tổng dữ liệu training, không phải số mẫu bổ sung sau pilot |
| $\pi_{l,j,i}$ | Policy của learner $l$ tại budget $j$, training seed $i=1,\ldots,10$ | Không coi các policy cùng MDP là MDP độc lập |
| $J_M(\pi)$ | $\rho^\top V_M^\pi$, return chiết khấu theo mô hình chấm điểm | Collection horizon 100 không mặc định biến scoring thành horizon-100 return |
| $\mathrm{NR}_M(\pi)$ | $(J_M(\pi)-J_{\rm random})/(J_M^*-J_{\rm random})$; mẫu số dương | Không phải xác suất, không clip vào $[0,1]$; policy kém random có thể âm |
| $\tau$ / seed success | $\tau=0.95$; success nếu NR đạt ngưỡng | Không phải xác suất thành công 95% |
| $E_l$ | Chỉ số lớp đầu tiên đạt luật suffix 8/10, hoặc lớp 6 | Là outcome thực nghiệm, không phải OEH-H hay định lý sample complexity |
| Right censoring $>x50$ | Không có suffix thành công trong lưới đã đo | Không khẳng định learner không bao giờ học được; không gán nhãn bằng vô cực |
| $n(s,a,s'),n(s,a)$ | Số đếm trong pilot; $n(s,a)=\sum_{s'}n(s,a,s')$ | Không phải xác suất chuyển thật |
| Posterior / empirical Bayes | Phân phối mô hình điều kiện theo pilot và cấu trúc prior; concentration fit bằng marginal likelihood | Không phải tập tất cả MDP khả dĩ và không phải confidence set đã chứng minh |
| $(\bar P,\bar R,\bar\beta)$ | Mô hình posterior mean dùng cho descriptor data-only | Không phải MDP thật |
| $\mathcal A_{\rm obs}(s)$ | Những action có $n(s,a)>0$ trong pilot | Khác support thật và support mô hình |
| $\mathcal A_\beta(s)$ / $\mathcal A_{\rm mod}(s)$ | Action có $\beta(a\mid s)>0$ / $\bar\beta(a\mid s)>0$ | Smoothing có thể làm $\mathcal A_{\rm mod}$ lớn hơn $\mathcal A_{\rm obs}$ |
| $d_\beta^{\rm stream}(s,a)$ | Tần suất transition dài hạn của stream có restart/terminal/horizon | Không phải discounted occupancy của target policy |
| $d_\rho^\pi(s)$ | $(1-\gamma)\rho^\top(I-\gamma P_\pi)^{-1}$; discounted state occupancy chuẩn hóa | Không phải tần suất logging của behavior |
| OEH-B, $\widehat p_l^{\rm B}$ | Phân phối requirement thô từ mô phỏng posterior; có thể tóm tắt bằng median | Không phải forecast cuối đã fit hoặc xác suất được chứng nhận hiệu chuẩn |
| OEH-H, $h_{\rm H}$ | Descriptor lookahead/planning-risk trên mô hình đã chọn; có requirement proxy $\widetilde N_k$ | Không phải nhãn $E_l$ hay cận hữu hạn mẫu |
| $p_l=f_l(x(D_0))$ | Forecast cuối: 7 xác suất, tổng bằng 1; fit trên cohort đã mở | Khác raw OEH-B và khác selector |
| B2 matched | 18 thống kê pilot của nhóm, fit dưới cùng quy trình forecaster như OEH | Không phải paper gốc, SOTA hoặc OEH phiên bản cũ |
| MDP cluster | Đơn vị suy luận; giữ behavior/learner/seed cùng MDP trong cụm | 108.000 job không phải 108.000 đơn vị độc lập |
| Selector | Ứng dụng phụ x10; ExtraTrees dùng thêm nhãn họ MDP | Không phải hệ dự báo requirement pilot-only; không phải router v5 cũ |

## 3. Estimand và phép chấm điểm

### 3.1 Nhãn thực nghiệm

Với $C_{l,j}=\sum_{i=1}^{10}\mathbf 1\{\mathrm{NR}(\pi_{l,j,i})\ge\tau\}$:

$$
E_l=\begin{cases}
\min\{j\in\{0,\ldots,5\}:\ \min_{m=j,\ldots,5}C_{l,m}\ge8\},
  &\text{nếu tập không rỗng},\\
6,&\text{nếu không có suffix đạt.}
\end{cases}
$$

Ví dụ $C=(2,5,8,7,9,10)$ cho $E_l=4$, tức x20, không phải x5.
Đây là nhãn từ 10 seed và lưới hữu hạn; không suy ra xác suất quần thể >=0.8
hay budget liên tục tối thiểu. Seed namespace khác nhau không làm các prefix
trong cùng stream độc lập; sự phụ thuộc đó là một phần của protocol.

### 3.2 Metric dự báo

Đặt $p_{l,c}\ge0$, $\sum_{c=0}^6p_{l,c}=1$,
$F_{l,j}=\sum_{c=0}^jp_{l,c}$. Code chấm:

$$
\mathrm{IBS}=\frac16\sum_{j=0}^5
   (F_{l,j}-\mathbf 1\{E_l\le j\})^2,\qquad
\mathrm{LL}=-\log\max(p_{l,E_l},10^{-12}).
$$

IBS ở đây là trung bình trên **sáu mốc lưới**, không phải tích phân liên tục
theo số transition. Lớp censor 6 có sáu indicator bằng 0 và vẫn được chấm log
loss ở lớp 6. Điểm được lấy trung bình behavior/learner trong MDP rồi theo MDP;
bootstrap theo cụm MDP. Giảm các loss này không chứng minh policy tốt hơn hoặc
chi phí thu dữ liệu giảm.

## 4. OEH-B: công thức và giả định

Một row transition dưới single Dirichlet prior có dạng
$P(\cdot\mid s,a,D_0)\sim\mathrm{Dir}(n(s,a,\cdot)+\alpha_Pu_S)$.
$\alpha_P$ được fit từ pilot; behavior mixture có concentration và responsibility
riêng. Công thức single-row không được dùng để che việc candidate đã chọn mixture.
Candidate hiện hành: `reward_prior=unseen_zero`, `transition_base=uniform`,
`behavior_prior=mixture`, `plus_prior_profile`, `candidate_law=false`.

Trong mỗi draw, replay learner, 10 seed, nested budget và luật suffix để có
$E_l^{(b)}$. Với $B_{\rm eff}$ event thực sự đưa vào estimator:

$$
\widehat p_l^{\rm B}(c)=
\frac{\sum_{b=1}^{B_{\rm eff}}\mathbf1\{E_l^{(b)}=c\}+1/2}
     {B_{\rm eff}+7/2}.
$$

Code yêu cầu 64 draw; lưu cả requested/returned/proposals. Nếu không trả về
draw nào, `requirement_forecast()` dùng 64 censored pseudo-events; nếu trả về
một số draw, mẫu số dùng số event thực tế, không tự điền mọi shortfall. Không
viết xác suất có bảo đảm với đúng 64 mẫu nếu sampling report không xác nhận.

Giả định cần công khai: reward có factorization tất định phù hợp các reward đã
quan sát; reward chưa quan sát được gán 0 theo quy ước phát triển; terminal được
suy từ cờ quan sát; posterior được condition theo bộ lọc dynamic range công khai.
Khớp dữ liệu hữu hạn không chứng minh factorization, terminal hoặc prior đúng
trên MDP thật. KT smoothing làm tránh xác suất 0, không chứng minh calibration.
Mixture/prior, 64 draw và head vẫn là lựa chọn thiết kế; không gọi toàn OEH là
"parameter-free" hoặc "không có hyperparameter".

## 5. OEH-H: công thức mô hình, không phải cận mẫu

Mọi đại lượng sau tính trên mô hình cố định $(\bar P,\bar R,\bar\beta)$.
Đặt $\bar r=\sum_{s'}\bar P\bar R$, $Q_1=Q^{\bar\beta}$ và

$$
Q_{k+1}(s,a)=\bar r(s,a)+\gamma\sum_{s'}\bar P(s'\mid s,a)
 \max_{a'\in\mathcal A_{\rm mod}(s')}Q_k(s',a'),\quad
\pi_k(s)=\arg\max_{a\in\mathcal A_{\rm mod}(s)}Q_k(s,a).
$$

Terminal continuation bằng 0. Tie trong implementation dùng action index nhỏ
nhất (`argmax`); empty/terminal support có fallback hình thức action 0, không
phải chứng nhận action ấy đã quan sát. `horizon_group()` gọi **`form="risk"`**
dù mặc định của hàm là `critical`; finite scan là $k=1,\ldots,100$.

Đặt $\Delta_k=\max(Q_k(s,\pi_k(s))-Q_k(s,a),0)$ và
$L_k=\max(Q^{\pi_k}(s,\pi_k(s))-Q^{\pi_k}(s,a),0)$.
Code lan truyền variance qua cây lookahead $W_k$:

$$
\nu_k(s,a)=\sum_{x,b} W_k[(s,a),(x,b)]^2
                 \frac{\sigma_k^2(x,b)}{d_\beta^{\rm stream}(x,b)},\qquad
\widetilde\ell_k(N)=
\frac{\sum_{s,a}d_\rho^{\pi_k}(s)L_k(s,a)
 \Phi\!\left(-\frac{\Delta_k(s,a)\sqrt N}
 {\sqrt{\nu_k(s,\pi_k(s))+\nu_k(s,a)}}\right)}
 {(1-\gamma)(J^*-J_{\rm random})}.
$$

Chỉ cộng competitor thuộc support mô hình, nonterminal, occupancy dương và
$L_k>0$. $\sigma_k^2$ là variance của Bellman target; $W_1=(I-\gamma
P_{\bar\beta}^{SA})^{-1}$, và $W_{k+1}=I+\gamma\bar P\,S_{\pi_k}W_k$.
Phần tử có weight dương nhưng logging frequency 0 được xử lý variance vô cực.
Trường hợp variance 0/vô cực được xử lý riêng trong `_misrank_probability()`.

$$
\widetilde N_k=\inf\{N\ge1:\widetilde\ell_k(N)\le
  \mathrm{NR}(\pi_k)-\tau\},\qquad
h_{\rm H}=\min_{k:\mathrm{NR}(\pi_k)\ge\tau}\log_A\widetilde N_k.
$$

Đây là **risk inversion có giả định**, tính bằng numerical search có trần
$\log N=200$; $A>1$, dynamic range dương và terminal conventions là điều kiện
cần. Không đồng nhất $\widetilde N_k$ với budget thật của learner hoặc $E_l$.
Gaussian approximation, delta-method và cách cộng variance bỏ qua covariance
giữa các comparison; plug-in model và phụ thuộc trajectory còn gây sai lệch.
Performance-difference lemma là động cơ của occupancy/advantage weighting,
không biến tổng modeled misranking thành một đẳng thức regret hoặc PAC bound.

`risk_profile()` bổ sung $\widetilde\ell(N_j)$ và
$\sum_s d_\rho^\pi(s)e^{-N_jd_\beta^{\rm stream}(s,\pi(s))}$ trên policy
support-optimal hội tụ. Exponential missing-mass là xấp xỉ independent-draw,
không phải xác suất chính xác cho stream Markov có restart. Các cờ `identified`,
`coverage_trap`, `not_tau_solvable` là trạng thái mô hình, không phán quyết về MDP
thật. Quy tắc `coverage_trap` còn có fallback bảo thủ theo support thiếu.

## 6. Hai lập luận có thể phát biểu với giả thiết

### 6.1 Ổn định greedy không đồng nghĩa learner hết học

**Mệnh đề có điều kiện.** Trên MDP chiết khấu hữu hạn, reward bị chặn,
$0<\gamma<1$, một tập action khả dụng cố định không rỗng ở mỗi state quyết định,
terminal continuation 0, Bellman operator bị hạn chế là contraction theo
sup norm. Nếu mỗi state quyết định có action tối ưu duy nhất với gap tối thiểu
$\delta>0$, greedy của exact Q-iteration ổn định sau hữu hạn backup.

**Chứng minh.** Gọi fixed point $Q^*_{\mathcal A}$ và khởi tạo $Q_1$ hữu hạn.
Với $\epsilon_k=\gamma^{k-1}\|Q_1-Q^*_{\mathcal A}\|_\infty$,
$\|Q_k-Q^*_{\mathcal A}\|_\infty\le\epsilon_k$. Mỗi action gap thật giảm
tối đa $2\epsilon_k$ khi dùng $Q_k$. Chọn $k$ sao cho $2\epsilon_k<\delta$;
action tối ưu vẫn lớn hơn mọi competitor ở mọi state quyết định. Do
$\epsilon_k\to0$, tồn tại $k$ hữu hạn. □

Đây là lập luận exact-model; không có uniform finite-depth guarantee khi gap
bằng 0, và dung sai số học không là giả thiết toán. Nó chỉ chứng minh **greedy
ổn định theo depth trên mô hình cố định**. Không chứng minh toàn OEH-H hay
forecast bất biến: $W_k$, variance, gap, occupancy và model có thể thay đổi;
càng không suy ra return của learner ước lượng từ dữ liệu ngừng tăng theo N.
Không dùng mệnh đề để tuyên bố "đã thoát bão hòa phổ quát".

### 6.2 Action không được logging không thể nhận diện phổ quát

Xét MDP một bước, hai action; behavior luôn chọn action 0. Reward action 0 bằng
0.5 trong cả hai MDP; reward action 1 bằng 0 ở MDP thứ nhất, bằng 1 ở MDP thứ
hai. Pilot có cùng phân phối trong cả hai, nhưng action tối ưu khác nhau.
Mọi quyết định chỉ nhận pilot có cùng phân phối đầu ra ở hai trường hợp nên
không thể chắc chắn chọn tối ưu trong cả hai. Thêm prior có thể chọn một cách
ngoại suy, không tạo thêm dữ liệu nhận diện. □

Mệnh đề nói về **thiếu support thật**, không nói action chưa xuất hiện trong một
pilot hữu hạn chắc chắn có behavior probability 0. Uniform support trên prior
không thay thế bằng chứng coverage. Đây là giới hạn nhận diện, không phải
chứng minh mọi dữ liệu tự sinh vô giá trị.

## 7. Nguồn phát triển và phạm vi kế thừa

Ký hiệu MDP, discounted value và Bellman iteration theo nền tảng trong
[Sutton & Barto, 2018](https://mitpress.mit.edu/9780262039246/reinforcement-learning/).
Lập luận contraction/gap ở mục 6.1 là hệ quả được chứng minh ở đây, không phải
định lý mới về learner offline. Occupancy/advantage weighting có động cơ từ
performance-difference identity của
[Kakade & Langford, ICML 2002](https://www.cs.cmu.edu/~jcl/papers/papers.html);
thay advantage thật bằng modeled misranking không giữ một identity regret.
Half-count categorical smoothing là dạng
[Krichevsky–Trofimov, 1981](https://ieeexplore.ieee.org/document/1056331);
chỉ công thức smoothing được dùng, không chuyển guarantee coding sang calibration.

| Nguồn | Vai trò | Không kế thừa tự động |
|---|---|---|
| Laidlaw et al., NeurIPS 2023, [EH](https://papers.nips.cc/paper_files/paper/2023/hash/b8be628bf719550b560de8bec9456e0b-Abstract-Conference.html) | Lookahead/rollout complexity; BRIDGE và phân tích GORP trong deterministic finite-horizon MDP | Theorem của GORP không là theorem cho CFQI/IQL/BC từ pilot |
| Laidlaw et al., ICLR 2024, [stochastic EH](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3e18f7152b5e1696e73722dc5121b6f2-Abstract-Conference.html) | Stochastic QVI-solvability và SQIRL; nguồn cảm hứng cho planning depth | Interactive sampling, learner và assumptions khác offline log cố định |
| [Li et al., 2024](https://arxiv.org/abs/2204.05275), [Wang et al., 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/5f5f7b6080dcadced61cf5d96f7c6dde-Abstract-Conference.html) | Coverage/concentrability và gap trong offline sample complexity | Thay quantities thật bằng posterior mean không giữ nguyên guarantee |
| [Schweighofer et al., 2022](https://proceedings.mlr.press/v199/schweighofer22a.html), [Swazinna et al., 2021](https://arxiv.org/abs/2111.13461), [Asadulaev et al.](https://arxiv.org/abs/2510.12638) | Dataset quality; TQ/SACo, ERI và BWD làm nguồn đối chứng | Adaptation sang nhãn bảy lớp không là reproduction end-to-end |
| [Barnwal et al., AFT](https://arxiv.org/abs/2006.04920) | Công cụ fit outcome có kiểm duyệt | Không chứng minh độ mới hay tính đúng của OEH |

EH 2023 kết hợp depth và rollout effort; $H=\min_k[k+\log_A m_k]$ không chỉ là
`min_k`. `stationary_eh()` trong repo là diagnostic chiết khấu của nhóm, không
mặc định byte-identical với code tác giả hoặc bản finite-horizon/stochastic.
Ở bảng Brier/log loss, B2+EH oracle là adaptation dùng thêm MDP thật; không ghi
"Brier của paper EH gốc". Phân biệt historical `Q^(0)=Q^beta` trong
`exact_solver.k_step_backups()` với current `Q_1=Q^beta`: lệch chỉ số một, không
phải thêm một backup trong nội dung khoa học.

## 8. Tiêu chí nghiệm thu và review B1

- [x] Glossary phân biệt outcome, descriptor, posterior forecast và selector.
- [x] [Bảng công thức → giả định → code → giới hạn](H1_FORMULA_CODE_MAP_2026-10-08.md).
- [x] Method/Related Work LaTeX cập nhật thuật ngữ và nguồn; không thay kết quả.
- [x] Lập luận greedy stabilization và non-identification có giả thiết/phạm vi.
- [x] [Review cấu trúc phép bỏ khối](H1_ABLATION_REVIEW_2026-10-08.md): đủ 108 cột,
  phát hiện overlap/lineage và đề xuất contrast; chưa tự freeze hay chạy.
- [x] [Pre-review B1](H1_B1_REVIEW_2026-10-08.md) có checklist và điểm chặn.
- [ ] Review bản B1 **tuần 8** thực tế, sau khi Việt Trần bàn giao.
- [ ] Hưng xác nhận bàn giao và nhóm phản biện; Linh duyệt tích hợp/ablation.

B1 tuần 6 có trong repo chỉ được đọc như hồ sơ lịch sử, không ký duyệt thay B1
tuần 8. H1 không cấp quyền chạy ablation hay biến cohort đã mở thành test kín mới.
Các kết quả đã seal tiếp tục hỗ trợ phạm vi within-family; không có kiểm chứng
độc lập toàn phương pháp chỉ nhờ hoàn thiện tài liệu này.

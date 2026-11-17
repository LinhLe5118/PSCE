# OEH cuối cùng: phương pháp, công thức và lý do lựa chọn

> **Nguồn thuật ngữ hiện hành (05/10/2026):** [T1 v1.1.0](docs/week8/T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) và [claim map](docs/week8/T1_CLAIM_MAP_2026-10-05.md). Difficulty trong bài hiện tại là nhãn nhu cầu dữ liệu thực nghiệm có điều kiện theo learner/behavior/ngưỡng; OEH-H là descriptor có giả định, không phải nhãn hay cận mẫu đã chứng minh. Tài liệu dài dưới đây giữ diễn tiến phát triển và các kết quả cohort trước theo đúng ngày, không thay định nghĩa T1 hoặc verdict niêm phong.

> **Cập nhật kiểm định 04/10/2026:** Các mục nói về cohort "cuối" bên dưới mô tả lần thứ tư và giữ nguyên như lịch sử phương pháp. Một lượt xác nhận thứ bảy trên 120 kernel mới đã hoàn tất 108.000 job core: 84 MDP stochastic có OEH Brier 0,045989 và log loss 0,563079, so với B2 cùng kiến trúc 0,054727 và 0,627179. Cả 22 cận dưới đồng thời của các tương phản dự báo đều dương, kể cả TQ/SACo đặc quyền. Đã kiểm toán 16.800 job chọn learner: gate mới so với fixed CFQI GO cho success (+6,67 điểm, cận +4,26 điểm) và return (+0,018656, cận +0,010341), nhưng so sánh bổ sung với v5 NO-GO. [Báo cáo kiểm toán](docs/reassessment/KIEM_DINH_2026-10-04.md) và [artifact](artifacts/oeh_confirmatory_unseen_120_2026-10-04/confirmatory_verdict.json) là nguồn hiện hành. Lượt kiểm tra thành phần trên dataset Farama FourRooms công bố sẵn NO-GO và chưa chứng minh chuyển miền toàn OEH.

Phiên bản: **v2 (sau khi anh/chị duyệt Phase 2, có điều chỉnh).** Bản v1 (đề
xuất Phase 2) được thay thế hoàn toàn. Các điểm của v1 không còn đúng được ghi
ở §9. Mọi lựa chọn trong quá trình phát triển đều nằm trong registry
append-only `artifacts/oeh_final_development_2026-10-01/registry.jsonl`.

## 0. Năm quyết định đã chốt với anh/chị

1. Tự phân tích test kín thứ ba và dùng nó để tìm điểm yếu của OEH. Kết quả: test
   thứ ba **NO-GO**, và chẩn đoán chi tiết ở §1.
2. Phát triển OEH theo hướng toán học, lấy cảm hứng từ EH gốc, rồi chốt một
   ứng viên cuối cùng.
3. Chọn learner là nghiên cứu bổ sung: mọi learner trong thư viện chịu **cùng
   một quy tắc**, và được so sánh với lựa chọn của OEH.
4. Cohort xác nhận cuối có 120 MDP.
5. **Không hardcode hyperparameter nào trong logic học của OEH.** Mọi đại lượng
   phải được suy ra từ dữ liệu (empirical Bayes, BIC, marginal likelihood), từ
   hằng số của protocol (γ, τ, lưới budget, luật 8/10, horizon 100, bộ lọc
   dynamic range), hoặc từ nguyên lý chuẩn tắc (prior Jeffreys/KT, CLT,
   performance-difference lemma).

## 1. Điểm yếu đã đo được của OEH trước (test thứ ba và các cohort đã mở)

| Điểm yếu | Bằng chứng |
|---|---|
| Thang requirement của OEH3 sai khoảng 100 lần và **sai chiều** | log1p N̂ trung bình là 13,5, tức khoảng 700 nghìn transition, so với budget x50 cỡ 5–12 nghìn; ρ Spearman với thời điểm thành công là −0,20 |
| Không phân biệt learner | OEH3 cho giá trị giống hệt nhau với BC, IQL và CFQI, trong khi tỷ lệ censored lần lượt là 90%, 77% và 44% |
| Bẫy support | 88% các hàng OEH3 cho là "đủ trong pilot" thực tế bị censored >x50 |
| LCOEH4 suy biến | Cờ censored bật trên 1254/1260 hàng |
| Ứng viên trước (cross-capacity) | Test thứ ba so với B2 matched: Brier −7,96%, log-loss −4,35% (dưới ngưỡng 5%), cận dưới đồng thời âm, **NO-GO** |
| Lỗi tiềm ẩn trong IQL đã đăng ký | Hàm expectile dùng dung sai tuyệt đối 1e-15, nên ném lỗi khi hai Q-value bằng nhau và có độ lớn > ~4. Chưa từng xảy ra trên dữ liệu thật; nay đã sửa mà vẫn giữ kết quả giống hệt ở mọi trường hợp bản gốc chạy được |

## 2. Ký hiệu

MDP $M=(S,A,P,R,\gamma,\rho)$ với $R(s,a,s')\in[0,1]$ và $\gamma=0{,}95$. Behavior
là $\beta$; pilot $D$ là prefix x10, tức $10\,|S||A|$ transition. Learner là
$l\in\{\text{CFQI},\text{IQL},\text{BC}\}$ cho endpoint độ khó, và thư viện 7
learner cho nghiên cứu chọn learner. Budget là $N_j=j\,|S||A|$ với
$j\in\{1,2,5,10,20,50\}$. Normalized return là
$NR(\pi)=(J(\pi)-J_{\text{rand}})/(J^*-J_{\text{rand}})$, ngưỡng $\tau=0{,}95$.
Nhãn là hạng $E_l\in\{0,\dots,6\}$: budget đầu tiên mà từ đó trở đi mọi budget
lớn hơn đều có ≥ 8/10 seed thành công; nếu không có thì là 6 (>x50).

## 3. Thành phần 1: posterior không hyperparameter

**Mục đích.** Ở thời điểm dự báo, ta chỉ có pilot. Muốn hỏi "learner $l$ cần bao
nhiêu dữ liệu" thì phải mô tả được *mọi MDP và behavior phù hợp với pilot*.
Posterior là cách chuẩn tắc để làm điều đó. Mỗi biến dưới đây có lý do riêng.

| Đại lượng | Định nghĩa | Vì sao chọn như vậy |
|---|---|---|
| $n(s,a,s')$, $n(s,a)$ | Số đếm transition của pilot, gồm cả các đích terminal | Thống kê đủ của mô hình multinomial |
| $P(\cdot\mid s,a)$ | $\mathrm{Dir}\big(n(s,a,\cdot)+\alpha_P\,u_S\big)$, với $u_S$ uniform trên $S$ | Liên hợp với multinomial. $u_S$ không chứa thông tin về họ MDP |
| $\alpha_P$ | $\arg\max_\alpha \prod_{(s,a)}\mathrm{DirMult}(n(s,a,\cdot)\mid\alpha u_S)$ | **Empirical Bayes**: độ thưa của transition được *đo* từ chính pilot, không tự đặt. Log-likelihood tính chính xác bằng $\sum_{j<n}\log(\alpha m+j)$ |
| $\beta(\cdot\mid s)$ | $\mathrm{Dir}(n(s,\cdot)+\alpha_\beta u_A)$, hoặc **hỗn hợp $K$ Dirichlet** (arm A2) | Behavior thật thường là hỗn hợp giữa state gần tất định (expert) và state gần uniform (hòa Q\*). Một $\alpha$ chung làm mờ cả hai, đúng điểm yếu đã đo được với BC/IQL |
| $K$ (số thành phần) | Tăng $K$ khi BIC còn giảm; mỗi $K$ được fit bằng EM | BIC là tiêu chuẩn chọn mô hình chuẩn tắc; không đặt trần $K$ bằng tay |
| Khóa reward | Khóa thô nhất trong {$s'$, $(s,a)$, $(s,a,s')$} còn nhất quán với mọi reward quan sát được | Nguyên lý MDL: mô hình đơn giản nhất giải thích được dữ liệu |
| Reward của khóa chưa thấy | $0$ (`unseen_zero`, D2 thắng vòng 1) | Trùng với cách đọc support-conservative của các learner; prior exchangeable (D1) làm $J^*$ posterior bị thổi phồng |
| Điều kiện bộ lọc | Loại các mẫu có $J^*-J_{\text{rand}}<0{,}01/(1-\gamma)$ | Mọi MDP của benchmark đều qua bộ lọc công khai này, nên conditioning là chính xác |

## 4. Thành phần 2: OEH-B, requirement posterior-predictive

Với mỗi mẫu posterior $b=1..B$: mô phỏng 10 stream trajectory (horizon 100,
khởi động lại từ $\rho$) dài $50\,|S||A|$, với các prefix lồng nhau đúng như
protocol. Huấn luyện **chính learner $l$** trên từng prefix, đánh giá chính xác
trong $M_b$, áp luật 8/10 và luật suffix, thu được $E^{(b)}_l$. Khi đó

$$\hat p_l(c)=\frac{\#\{b:E^{(b)}_l=c\}+\tfrac12}{B+\tfrac72},\qquad c=0..6 .$$

- **Vì sao mô phỏng thay vì công thức.** Đây là *đúng estimand* của nhãn: cùng
  learner, cùng luật, cùng chuẩn hóa. Công thức đóng như OEH3 phải xấp xỉ cả
  learner lẫn luật, và chính phần xấp xỉ đó gây ra sai thang 100 lần.
- **Vì sao dùng $+\tfrac12$ (Krichevsky–Trofimov).** Đây là estimator
  minimax-regret dưới log-loss. Nó không phải số được tinh chỉnh; nó đến từ prior
  tham chiếu Jeffreys.
- **$B=64$** là độ chính xác Monte Carlo (giống số vòng lặp), không phải tham số
  học. Phần đi vào forecaster là các tóm tắt (CDF theo budget, tỷ lệ thành công,
  return trung bình) của từng learner, cộng với ngữ cảnh của hai learner kia.

## 5. Thành phần 3: OEH-H, horizon offline có cấu trúc EH

EH gốc là $H=\min_k[k+\log_A m_k]$, với $Q^1=Q^{\text{rand}}$ và
$Q^{k+1}=\mathcal TQ^k$. OEH-H giữ dãy backup đó nhưng lấy **leaf là behavior**
($Q^1=Q^\beta$), và **giới hạn greedy trong support của mô hình behavior $\beta$**
(`beta > 0` trên MDP rút từ posterior). Posterior có làm trơn nên support mô
hình có thể rộng hơn các action đã thấy trong pilot; đây là giả định mô hình,
không phải chứng nhận an toàn cho action chưa quan sát. Riêng đặc trưng trần
return theo support ở mục 6 dùng support *quan sát được* từ pilot.

**Dạng risk (H2, đã thăng cấp).** Theo performance-difference lemma,

$$\mathcal L_k(N)=\frac{1}{(1-\gamma)(J^*-J_{\text{rand}})}\sum_s d^{\pi_k}(s)\sum_{a'\ne\pi_k(s)}A_k(s,a')\;
\Phi\!\Big(-\frac{\Delta_k(s,a')}{\sqrt{(V_k(s,\pi_k(s))+V_k(s,a'))/N}}\Big),$$

$$N_k=\min\{N:\mathcal L_k(N)\le NR(\pi_k)-\tau\},\qquad \mathrm{OEH\text{-}H}=\min_k\log_A N_k .$$

| Thành phần | Ý nghĩa | Vì sao |
|---|---|---|
| $d^{\pi_k}(s)$ | Occupancy của policy mục tiêu | Lỗi ở state hay được ghé thăm mới làm hỏng return; bản EH-worst-case (H1) thua ở điểm này |
| $A_k(s,a')$ | Advantage thua thiệt nếu chọn $a'$ | Lấy đúng từ performance-difference lemma |
| $\Delta_k$ | Gap của $Q^k$ | Cùng vai trò với $\Delta$ trong cận EH |
| $V_k=\sum W_k^2\sigma^2/d_\beta$ | Phương sai delta-method qua cây lookahead $W_k$, chia cho tần suất ghi log $d_\beta$ | Coverage đi vào *nhân* với phương sai, nên dữ liệu hiếm ở cặp quan trọng làm tăng requirement |
| $\Phi$ | Xấp xỉ CLT của xác suất đảo thứ hạng | Không có hằng số tin cậy $\delta$ (khác OEH3 với các hằng số 2 và 7/3) |
| $NR(\pi_k)-\tau$ | Biên dư so với ngưỡng | Ngưỡng lấy từ protocol (0,95), không chọn $\varepsilon$ riêng |
| $d_\beta$ | Tần suất dài hạn của stream có restart (định lý renewal-reward) | Trùng với cách dữ liệu thật được sinh |

**Tính chất đã chứng minh hoặc kiểm bằng test.**
- (i) Với $\beta$ uniform, chi phí lookahead $\log_A 1/\kappa_k=k$, tức trùng hạng
  $k$ của EH.
- (ii) Requirement tăng đơn điệu khi coverage của action cần thiết giảm.
- (iii) Bẫy coverage cho $\infty$; trên dữ liệu thật, mọi nhóm bị báo $\infty$
  đều censored (độ chính xác 100%).
- (iv) Không có reward thì trả về trạng thái `degenerate_range`, thay vì gây lỗi.

## 6. Thành phần 4: các trần tiệm cận phụ thuộc learner

$$c_{\text{BC}}=NR(\bar\beta),\qquad c_{\text{supp}}=NR\big(\pi^*_{\text{supp}(D)}\big),$$

tính trên mô hình posterior trung bình. Trong đó $\pi^*_{\text{supp}(D)}$ là
policy tối ưu khi chỉ dùng các action đã quan sát.

**Vì sao.** Khi có vô hạn dữ liệu, BC hội tụ về $\beta$, nên nếu $NR(\beta)<\tau$
thì BC không bao giờ thành công. Learner support-conservative (CFQI) hội tụ về
$\pi^*$ trên support. Đây là định nghĩa learner-conditional của OEH (§4.1 của v1)
được biến thành đại lượng đo được. Trên cohort thứ ba, AUC dự báo censored đạt
**0,89 (BC), 0,95 (IQL) và 0,77 (CFQI)**, cao nhất trong mọi đặc trưng.

## 7. Thành phần 5: khối coverage lý thuyết (data-only)

Gồm $\log C^*$, $\log C^*_{\text{clip}}$ (Li et al., 2022) và cận intrinsic
Yin–Wang, tính trên mô hình posterior trung bình. **Vì sao:** đây là các đại
lượng minimax/instance-dependent của lý thuyết offline RL. Thí nghiệm oracle cho
thấy chúng xếp hạng độ khó tốt hơn EH (§10), và chúng bổ sung cho OEH-H (ρ ≈ 0,2).

Khối này còn gồm hai thành phần nữa. **Hồ sơ rủi ro của OEH-H tại đúng các budget
của protocol:** regret do nhiễu $\mathcal L(N_j)$, missing-mass
$M(N_j)=\sum_s d^\pi(s)e^{-N_jd_\beta(s,\pi(s))}$ và biên dư $NR(\pi)-\tau$, với
$N_j=j|S||A|$. Và **các diagnostic OEH thế hệ trước** (OEH3, LCOEH4), giữ lại như
thông tin, không phải như phương pháp so sánh. Ablation "chỉ thành phần mới" được
báo cáo kèm.

## 8. Kết hợp thành dự báo, và quy tắc chọn learner

**Forecaster matched.** Mọi phương pháp có fit dùng *cùng một* quy trình: ba thành
phần (XGBoost-AFT, ngưỡng trực tiếp, multiclass depth-3/200 vòng) fit trên 6
cohort đã mở; trọng số lồi chọn bằng log-loss OOF theo nhóm MDP, 5 fold. Cấu hình
cây là cấu hình đã đóng băng từ trước của nhóm, dùng chung cho mọi comparator,
nên khác biệt chỉ đến từ thông tin đầu vào. Đây là lý do OEH được so sánh với B2,
BWD, ERI và TQ/SACo *trên cùng một forecaster*.

**Quy tắc chọn learner (giống nhau cho cả 7 learner).** Với 16 mẫu posterior, mỗi
mẫu sinh **một** dataset x10 dùng chung cho mọi learner (common random numbers).
Huấn luyện từng learner trên dataset đó và đánh giá chính xác trong MDP của mẫu:

$$a(g)=\arg\max_{l}\;\hat P\big(NR_l\ge\tau\big),\quad\text{hòa thì xét }\hat{\mathbb E}[NR_l],\ \text{rồi đến thứ tự registry}.$$

**Vì sao.** Đây là quy tắc quyết định trong §3.1 của luận văn ("chọn learner có
xác suất thành công dự báo cao nhất"). Quy tắc không có tham số, không có biên an
toàn tự chọn, và áp cùng một cách cho CFQI, IQL, BC, SPIBB, VI-LCB, BCQ và CQL.

## 9. Những điểm của bản đề xuất v1 nay không còn đúng

1. **"OEH-H đơn điệu theo coverage"** sai trong trường hợp tổng quát. Khi
   behavior gần expert, action đối thủ hiếm nên ước lượng nó tốn mẫu. Phát biểu
   đúng là: đơn điệu theo từng $d_\beta(x,b)$ khi giữ các phần khác cố định, và
   đơn điệu khi coverage của action cần thiết cạn dần (đã có test).
2. Các hằng số $\delta=0{,}2$ và lưới $\alpha_P$ bị loại bỏ, theo yêu cầu không
   hardcode.
3. Ở tầng oracle, **OEH-H không vượt được C\***. Tương quan thứ hạng với độ khó
   thực nghiệm là khoảng 0,35 so với 0,50. Điểm mạnh riêng của OEH-H là phát hiện
   bẫy coverage chính xác 100%. Paper phải nêu đúng như vậy.
4. OEH-B dạng không fit bị lệch hiệu chuẩn: báo censored quá mức vì posterior làm
   behavior bị mờ. Nó có giá trị nhờ khả năng xếp hạng (AUC 0,76), không phải nhờ
   xác suất tuyệt đối.

## 10. Lịch sử phát triển (đầy đủ trong registry)

| Vòng | Thử | Kết quả |
|---|---|---|
| 1 | Bốn cấu trúc posterior D1–D4 (reward prior × base measure) | D2 (`unseen_zero`, uniform) có log-loss thô thấp nhất |
| 4–7 | Sáu dạng phiếm hàm OEH-H oracle (H1–H5c) | H2 (dạng risk) được thăng cấp; dạng coverage-risk, độ sâu hội tụ và missing-mass không vượt H2; dừng tìm kiếm |
| 3, 8–13 | Chín arm forecast trên gate leave-one-cohort-out (252 MDP) | Tốt nhất là A4: so với B2 +16,4% Brier và +9,0% log-loss; so với BWD +15,7% và +8,4%; mọi cận dưới đều dương. Log-loss so với TQ/SACo đặc quyền là 4,9%, so với OEH cũ là 4,6%. Từ A2 trở đi bão hòa |

Ứng viên đóng băng là **A4**: posterior D2 có hỗn hợp Dirichlet cho behavior, cùng
bộ feature "plus_prior_profile". Thời điểm đóng băng là 2026-10-02, sau quyết
định của anh/chị.

## 11. Protocol của lượt đánh giá cuối (đã đóng băng, chạy một lần)

- 120 MDP mới (84 stochastic: Garnet, forward-layered, bottleneck; 36 lock
  deterministic làm baseline). Kernel và seed được loại trừ khỏi cả sáu cohort
  trước. Có 2 MDP lock được đổi sang độ dài 7, vì không gian kernel độ dài 5 đã
  cạn; việc này được ghi trong protocol.
- **Tầng 1** (đối chứng cùng thông tin pilot và cùng forecaster): B2 matched, B2 + BWD, B2 + ERI,
  các adaptation coverage/quality, B2 AFT, tần suất fold.
- **Tầng 2** (thông tin đặc quyền từ MDP thật): B2 + EH oracle, B2 + TQ/SACo (code
  tác giả), và "chỉ dùng metric đã công bố".
- **Chỉ báo cáo:** OEH cũ (bằng chứng cho điểm bão hòa), OEH-B không fit.
- Gate: cải thiện ≥ 5% ở cả Brier lẫn log-loss, và cận dưới đồng thời một phía
  95% dương (bootstrap max-t theo cụm MDP, 10.000 lần), trên 84 MDP stochastic.
- Chọn learner: 420 nhóm stochastic ở x10, 10 seed. OEH được so với từng learner
  cố định, router v5 và lựa chọn ngẫu nhiên; oracle chỉ dùng làm cận trên.
- H1 (oracle): Spearman của OEH-H với requirement thực nghiệm, so với EH, C\*,
  C\* clipped và Yin–Wang.
- Bốn amendment trước outcome (đều là sửa lỗi điều phối tiến trình, không đổi
  khoa học) được lưu cùng bản protocol trước đó và lý do.

## 12. Kết quả của lượt đánh giá cuối (một lần, 120 MDP mới, 2026-10-03)

Tất cả forecast và lựa chọn learner được seal lúc 2026-10-02 16:12 UTC, khi
`learner_outcomes_observed = 0`. Sau seal đã chạy đủ 108.000 job learner core và
16.800 job learner ngoài, và mọi job đều được verify với plan đã seal. Số liệu
dưới đây tính trên 84 MDP stochastic; cận dưới là bootstrap max-t đồng thời
(critical 2,53).

**Dự báo độ khó: Tầng 1 GO, Tầng 2 NO-GO, gate "mọi comparator" NO-GO.**

| Comparator | Brier | Log-loss | Cận dưới đồng thời | Tầng |
|---|---:|---:|---|---|
| B2 matched | **+15,2%** | **+8,1%** | dương | 1 |
| B2 + ERI | **+14,4%** | **+8,0%** | dương | 1 |
| B2 + BWD (ICLR 2026) | **+14,2%** | **+7,2%** | dương | 1 |
| B2 AFT / các adaptation / tần suất | +26,7% đến +56,8% | +25,1% đến +47,8% | dương | 1 |
| B2 + EH oracle | +18,7% | +10,2% | dương | 2 |
| Chỉ các metric đã công bố (TQ, SACo, ERI, BWD) | +20,5% | +10,1% | dương | 2 |
| B2 + TQ/SACo (code tác giả, có reference đặc quyền) | +7,8% | **+3,7%** | **âm** | 2 |
| OEH cũ (chỉ báo cáo) | +5,8% | +3,0% | không kiểm định | — |

Panel deterministic (36 MDP, chỉ báo cáo): OEH đạt 0,0192 / 0,269, so với B2
0,0207 / 0,274. OEH-B không fit: 0,118 / 1,061, đúng như đã biết là lệch hiệu
chuẩn.

**Chọn learner (bổ sung): NO-GO.** OEH chọn CFQI ở 388/420 nhóm. So với CFQI (learner
cố định tốt nhất): return +0,0057 (không có ý nghĩa, cận dưới −0,0015),
success **+1,76 điểm % (có ý nghĩa, cận dưới +0,0039)**. So với router v5: return
+0,0045 (không có ý nghĩa), success −7,5 điểm % (kém hơn có ý nghĩa). Oracle theo
nhóm: 0,837 / 55,9%.

**H1, thước đo oracle.** Spearman với requirement của CFQI: OEH-H 0,255; EH
dừng −0,120; EH stochastic đã công bố −0,130; C\* 0,481; C\* clipped 0,494;
Yin–Wang 0,242. OEH-H **vượt EH có ý nghĩa** (chênh lệch +0,38, KTC 95%
[0,22; 0,52]), nhưng **kém C\*** (−0,23, [−0,37; −0,08]) và ngang Yin–Wang.

**Diễn giải trung thực.** Trên cohort cuối có kernel mới từ các họ generator đã
dùng trong phát triển, OEH vượt B2 và các adaptation dữ liệu pilot của ERI/BWD
trong cùng forecaster theo gate đã đăng ký. Đây không phải phép chạy nguyên gốc
của các tác giả trên cùng endpoint, nên không suy ra kết luận SOTA toàn văn
liệu. Gate gồm cả TQ/SACo có dataset tham chiếu từ MDP thật là NO-GO.
Trong phân tích oracle trên endpoint offline này, OEH-H xếp hạng tốt hơn
những descriptor EH đã thử nhưng kém C\*; kết quả không bác bỏ định lý EH cho
online RL và không chứng minh công thức OEH-H là cận mẫu hữu hạn.

# Công thức OEH cuối cho stochastic tabular offline RL

Ngày: **2026-09-19**. Trạng thái: **định nghĩa mục tiêu nghiên cứu và đặc tả
estimator kế tiếp**. Tài liệu này phân biệt đại lượng cần ước lượng với thuật
toán ước lượng nó; code `oeh_eh_aligned.py` hiện là proxy development, chưa phải
finite-sample implementation của toàn bộ định nghĩa dưới đây.

## 1. Không thể yêu cầu một finite scalar cho mọi pilot

Không estimator data-only nào có thể trả một finite difficulty đúng cho mọi
stochastic tabular MDP từ mọi pilot hữu hạn. Nếu behavior không lấy action `a`
tại state `s`, hai MDP có thể sinh cùng dataset nhưng một MDP cho `a` reward thấp
và MDP kia cho `a` reward cao. Optimal policy và data requirement khác nhau nhưng
estimator không phân biệt được chúng.

Do đó “hoạt động với mọi môi trường” phải có nghĩa:

1. đại lượng mục tiêu được định nghĩa cho mọi finite tabular MDP;
2. trả finite value khi target có thể nhận dạng dưới collection policy;
3. trả `+infinity` hoặc abstain khi coverage không nhận dạng được target;
4. không biến absence of evidence thành evidence of absence.

Điều này phù hợp với offline-RL theory: các guarantee cần coverage hoặc
single-policy concentrability; không có coverage thì pessimism phải giữ bất
định thay vì bịa một model chắc chắn. Xem
[Shi et al., ICML 2022](https://proceedings.mlr.press/v162/shi22c.html) và
[Li et al., model-based offline RL](https://arxiv.org/abs/2204.05275).

## 2. Đối tượng cơ sở: limited-lookahead policy family

Cho discounted MDP

```text
M = (S, A, P, R, gamma, rho0),       0 <= R <= Rmax,
```

và collection/behavior policy `beta`. Chuẩn hoá return bằng một range `G > 0`
đã khóa. Có hai contract hợp lệ nhưng không được trộn lẫn: benchmark có thể định
nghĩa target bằng oracle range `J* - J_min`, còn estimator triển khai phải dùng
public reward range (thí dụ `Rmax/(1-gamma)`) hoặc một reference chỉ lấy từ model
fold. Oracle target được dùng để tạo nhãn/evaluation, không được đưa vào feature.

Định nghĩa chuỗi Q đúng theo stochastic EH gốc:

```text
Q_M^[1]     = Q_M^beta,
Q_M^[k+1]   = T_M Q_M^[k],
Gamma_k(M)  = tập mọi policy greedy đối với Q_M^[k].
```

Planning bias ở depth `k` là

```text
b_k(M) = max_{pi in Gamma_k(M)} [J*_M - J_M(pi)] / G.
```

Dùng maximum vì định nghĩa `k`-QVI-solvable gốc yêu cầu mọi greedy policy đều
tốt. Depth `k` là `epsilon`-solvable khi `b_k <= epsilon`. Exact tie trong true
model được phép; empirical tie không tự chứng minh exact tie.

## 3. Target phổ quát: offline identification requirement

Trước hết đặt tập policy đạt target

```text
A_epsilon(M)
  = {pi: [J*_M-J_M(pi)]/G <= epsilon}.
```

Với từng `pi in Gamma_k(M)`, alternative liên quan đến câu trả lời cụ thể đó là

```text
Alt_epsilon(M,pi)
  = {M' trong cùng public model class:
       pi not in A_epsilon(M')}.
```

Không dùng điều kiện “hai answer set không giao nhau”: điều kiện đó quá yếu khi
có nhiều câu trả lời đúng. Một procedure phải bảo đảm policy **cụ thể** nó trả ra
vẫn đạt target trên mọi model chưa phân biệt được.

Đặt

```text
Z = (S, A, R/Rmax, S')
```

là một observation dưới collection protocol. Gọi
`L_M(. | s,a)` là joint conditional law của normalized reward và successor, và
`d_beta^M(s,a)` là expected visit count trong **một sampling unit**. Nếu ngân
sách đo bằng transitions thì chuẩn hoá để tổng `d_beta=1`; nếu đo bằng trajectory
thì giữ expected visit count mỗi trajectory. Sampling unit phải được khóa trước,
vì nó quyết định đơn vị của `N`.

Information rate của một policy và rate bảo thủ cho depth `k` là

```text
I_epsilon(M,beta;pi)
  = inf_{M' in Alt_epsilon(M,pi)}
      sum_{s,a} d_beta^M(s,a)
        KL( L_M(.|s,a) || L_M'(.|s,a) ),

I_{k,epsilon}(M,beta)
  = min_{pi in Gamma_k(M)} I_epsilon(M,beta;pi),
    chỉ khi b_k(M) <= epsilon.
```

Lấy minimum trên mọi greedy policy vì stochastic EH gốc định nghĩa
`k`-QVI-solvable bằng yêu cầu **mọi** greedy policy đều tốt. Một phiên bản chỉ
cần tìm một policy dễ chứng nhận có thể dùng maximum, nhưng đó là mục tiêu khác
và phải mang tên riêng.

Quy ước `inf empty set = +infinity`. Đặt characteristic information time

```text
T_k^ID = 1 / I_{k,epsilon},
```

với `1/0=+infinity` và `1/(+infinity)=0`. Offline identification scale ở mức
tin cậy `1-delta` là

```text
N_k^ID(epsilon,delta)
  = 0,                                      nếu mọi Alt(pi) là empty set,
  = log(c_k/delta) / I,                     nếu 0 < I < +infinity,
  = +infinity,                              nếu I = 0
                                            hoặc b_k > epsilon.
```

`c_k` là multiple-comparison factor phải khóa trong protocol. Đây là
instance-dependent **characteristic scale**. Dấu bằng là định nghĩa độ khó dùng
trong OEH; nó không tự biến thành finite-sample upper bound. Muốn gọi đây là
sample-complexity guarantee còn phải chứng minh một procedure đạt cận trên và
lower bound tương ứng dưới đúng sampling protocol.

### Vì sao định nghĩa này phù hợp

- `d_beta=0` trên một cell có thể làm policy đang xét vượt target làm `I=0`, nên requirement
  bằng `+infinity`: đúng với non-identification offline.
- Stochasticity đi vào full observation law, không qua transition entropy thô.
- Successor randomness giữa các state có cùng hậu quả value không nhất thiết tạo
  alternative gần làm thay đổi quyết định.
- Reward noise và transition noise được xử lý trong cùng law.
- Target return `epsilon` nằm ngay trong alternative set, không bị tráo với một
  loss có đơn vị value.
- Coverage, gap và variance xuất hiện như các thành phần của cùng information
  rate, thay vì ba heuristic cộng rời rạc.

## 4. Quan hệ với EH gốc

Trong local two-action approximation, gọi `Delta_eff` là khoảng cách từ instance
thật tới biên alternative gần nhất làm policy đang xét vượt target. Nếu estimator
của action-value difference có variance `nu` và hai action nhận tỷ lệ mẫu hiệu
dụng `d`, thì

```text
KL gần nhất  ~= d Delta_eff^2 / (2 nu),
N_k^ID       ~= 2 nu log(c/delta) / (d Delta_eff^2).
```

Ở target exact-greedy (`epsilon=0`) và unique greedy action, `Delta_eff` chính là
action gap `Delta_k`. Khi variance và coverage được coi là hằng số, phần phụ
thuộc vào instance còn `1/Delta_k^2`. Vì vậy thành phần stochastic EH gốc

```text
Hbar_k = k + log_A(1/Delta_k^2)
```

được thu lại như trường hợp đơn giản của information requirement này, bỏ các hệ số
variance/coverage/confidence. Đây là lý do đúng để dùng gap bình phương, thay vì
chọn nó chỉ vì đường cong không phẳng. Định nghĩa EH gốc nằm trong
[Laidlaw et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/3e18f7152b5e1696e73722dc5121b6f2-Paper-Conference.pdf).

Trong đề tài, khóa `epsilon=0.05` theo endpoint `NR>=0.95`; khi đó dùng
`Delta_eff` tới biên mất target thay cho exact-ranking gap. Không điều chỉnh
`epsilon` để làm kết quả đẹp hơn trên evaluation split.

### Kiểm tra đóng: stochastic bandit hai action

Với target exact-best (`epsilon=0`), reward Bernoulli có mean `p1 > p2` và
behavior frequencies `d1,d2`, alternative gần nhất làm action 1 không còn tối
ưu nằm ở biên `q1=q2=m`, trong đó

```text
m = (d1*p1 + d2*p2) / (d1+d2),
I = d1*kl(p1||m) + d2*kl(p2||m).
```

Nếu `d1=0` hoặc `d2=0`, `I=0`: action ranking không nhận dạng được. Với gap nhỏ,
Taylor expansion của Bernoulli KL cho `I=Theta(d*Delta^2/nu)`, thu lại
`N=Theta(nu/(d*Delta^2))`. Ở exact tie với `epsilon=0`, alternative có thể gần
tùy ý nên `I=0`: một data-only procedure không thể chứng nhận exact tie với hữu
hạn mẫu. Với `epsilon>0`, alternative phải đẩy regret qua margin `epsilon`, nên
rate có thể dương. Reference implementation và unit tests của reduction này nằm
ở `code/oeh_information.py` và `tests/test_oeh_information.py`; nó là sanity
check cho định nghĩa, không phải solver KL cho MDP tổng quát.

## 5. Không ép profile offline thành một scalar duy nhất

Đầu ra khoa học chính là Pareto profile

```text
P_OEH = {(k, N_k^ID): k = 1,...,Kmax}.
```

Lý do: trong EH tương tác, lookahead `k` đòi hỏi tìm đúng action sequence và có
chi phí exploration gần `A^k`. Trong offline tabular, backup sâu hơn tái sử dụng
cùng buffer; `k` chủ yếu tăng compute/model-propagation error, còn coverage đã
nằm trong `N_k^ID`. Cộng cứng `k` có thể đếm chi phí hai lần.

Ba phép đọc profile được báo riêng:

```text
N_OEH(B) = min_{k <= B} N_k^ID                 data requirement ở compute budget B,
k_OEH(N) = min{k: N_k^ID <= N}                 depth khả thi ở data budget N,
H_OEH,lambda = min_k [lambda*k + log_A N_k^ID] scalar có exchange rate lambda.
```

Trong phép scalar hoá, dùng `log_A max{1,N_k^ID}` để trường hợp trivial
`N_k^ID=0` không tạo `log(0)`; mọi `N_k^ID=+infinity` giữ nguyên vô hạn.

`lambda=1` được báo như **EH-reference** để so với công thức gốc. Với chủ đề đồ
án, primary quantity là `N_OEH(B)`; `k`, slope của profile, coverage và
abstention là feature riêng. Không fit `lambda` trên evaluation split. Nếu
`A=1`, bài toán không có action selection; đặt profile decision difficulty bằng
0 và không dùng logarithm cơ số 1.

## 6. Estimator khả thi từ pilot: CF-EB-OEH

Exact KL optimization trên toàn alternative-MDP set khó tính. Estimator thực tế
dùng cross-fitted empirical Bernstein như một conservative local approximation.
Ký hiệu bắt buộc là `N_hat_k^EB`; không được ghi nó như thể bằng target
`N_k^ID`.

### 6.1 Cross-fitting

Chia trajectory IDs thành folds:

1. model fold khóa `Q^[k]`, candidate policy và action pair;
2. evidence fold độc lập ước lượng gap, Bellman-target variance và coverage;
3. đổi vai trò các folds và hợp nhất bằng rule đã khóa.

Điều này sửa hạn chế hiện tại: cùng dữ liệu đang vừa chọn action pair vừa ước
lượng confidence radius.

### 6.2 Lower-confidence gap

Với Bellman target chuẩn hoá

```text
Y_k = (1-gamma)[R/Rmax + gamma V_leaf(S')],     0 <= Y_k <= 1,
```

tính Bessel-corrected variance và empirical-Bernstein radius `r_k(s,a)`. Với
greedy action `a*` và competitor `a`:

```text
Delta_L,k(s,a)
  = max{0, Delta_hat_k(s,a) - r_k(s,a*) - r_k(s,a)}.
```

Nếu `Delta_L=0`, pair chưa identified và requirement là `+infinity`; không
tie-break thành finite score.

### 6.3 Local requirement

Với `ell` đã Bonferroni/union-correct và `d_L(s,a)` là lower confidence coverage:

```text
N_EB,k(s,a)
 = max{
     128 nu_hat_k(s,a*) ell / [d_L(s,a*) Delta_L^2],
     128 nu_hat_k(s,a ) ell / [d_L(s,a ) Delta_L^2],
     [1 + 56 ell/(3 Delta_L)] / d_L(s,a*),
     [1 + 56 ell/(3 Delta_L)] / d_L(s,a)
   }.
```

Các hằng số trên đến từ yêu cầu mỗi radius không quá `Delta_L/4` sau union bound:
giải phần căn của empirical-Bernstein radius cho `n=dN` sinh term tỷ lệ
`nu*ell/(d*Delta_L^2)`, còn giải phần tuyến tính sinh
`ell/(d*Delta_L)`. Vì vậy cấu trúc và tính đơn điệu được suy ra từ concentration;
các hệ số `128` và `56/3` là lựa chọn đủ bảo thủ của proxy hiện hành, chưa được
phép mô tả là hằng số tối ưu. Linear term ngăn zero empirical variance từ một vài
observation bị hiểu là deterministic certainty. Empirical Bernstein source:
[Maurer và Pontil, COLT 2009](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf).

### 6.4 Aggregate không dùng occupancy threshold tùy ý

Không dùng rule cố định `occupancy >= 1/(10|S|)` trong định nghĩa cuối. Hai lựa
chọn phải báo tách:

1. **EH-faithful:** maximum trên mọi state-action pair reachable có thể thay đổi
   greedy decision; phù hợp exact-action recovery của EH gốc, bảo thủ.
2. **Target-return:** giải alternative-set hoặc error-allocation problem sao cho
   tổng performance loss không vượt `epsilon`; occupancy xuất hiện qua
   performance-difference identity, không qua hard threshold.

Proxy đầu tiên dễ implement và là safety check. Proxy thứ hai phù hợp endpoint
`NR >= 0.95` của đồ án và là phiên bản nên dùng cho prediction sau khi có proof.

## 7. Hành vi trên mọi loại stochastic tabular environment

| Trường hợp | Kết quả đúng |
|---|---|
| Full coverage, positive identifiable gaps | Finite profile |
| Unseen relevant action/successor | `+infinity` hoặc abstain |
| Exact co-maximizer, `epsilon=0` | `+infinity` nếu không có structural tie assumption |
| Exact co-maximizer, `epsilon>0` | Chứng nhận target margin; không cần chọn đúng một co-maximizer |
| Empirical tie | Abstain cho đến khi confidence set chứng nhận target margin |
| Stochastic successors cùng value | Không tăng difficulty chỉ vì entropy cao |
| High Bellman-target variance | Requirement tăng |
| Rare behavior coverage | Requirement tăng theo inverse coverage |
| Depth không đạt target ngay cả với exact model | `N_k=+infinity` |
| Degenerate return range | Báo `degenerate_task`, không chia cho 0 |
| Một action | Decision difficulty bằng 0 |

Đây là nghĩa hợp lệ của “dùng được cho mọi môi trường”; không hứa mọi môi
trường đều có finite answer.

## 8. Vai trò trong đề tài

Câu hỏi chính của đề tài vẫn là dự báo learner-specific empirical requirement và
chọn learner. OEH không nên tự nhận là sample requirement của CFQI/IQL, vì các
learner đó không phải exact `k`-step QVI. OEH profile là feature family B3:

```text
min log N_k, argmin k, N_OEH(B), profile slope/curvature,
finite-depth coverage, infinity/abstention indicators.
```

Đánh giá B3 phải so với B1 coverage/trajectory quality và B2 stochastic
uncertainty trên cùng grouped out-of-fold folds. Chỉ giữ claim đóng góp nếu B3
cải thiện held-out likelihood hoặc learner-selection utility; nếu không, OEH là
failure-analysis/negative-control contribution.

## 9. Thay đổi cần làm so với prototype hiện tại

1. **Đã làm:** đổi tên output hiện tại thành `EB-OEH plug-in proxy`.
2. **Một phần:** target `N_k^ID` và estimator `N_hat_k^EB` đã tách trong tài
   liệu/system contract; paper vẫn mô tả OEH-v1 lịch sử.
3. **Đã làm ở local proxy:** trajectory cross-fitting tách model/evidence folds;
   exact pessimistic target-return solvability bound vẫn chưa có theorem.
4. **Đã làm ở local proxy:** point gap được thay bằng empirical-Bernstein
   lower-confidence gap trong `oeh_crossfit.py`.
5. **Đã làm ở EH-faithful proxy:** bỏ hard occupancy threshold; mọi action ở
   reachable decision state phải có support. Target-return error allocation vẫn
   là hướng lý thuyết chưa triển khai.
6. **Đã làm ở contract layer:** trả profile `(k,N_hat_k,status)` và censoring
   trước khi scalar hóa.
7. **Đã làm:** giữ `lambda=1` như EH-reference; data selector dùng
   `min_{k<=B}N_hat_k`.
8. **Đã làm trên development:** controlled bandit/profile tests, cross-fit run
   và grouped nested B0/B1/B2/B3 validation đã chạy. B3 không cải thiện B2.
9. **Không mở confirmatory:** development predictive/selection gate không đạt;
   evaluation split tiếp tục sealed.

Đây là công thức phù hợp nhất với cả hai mục tiêu: giữ limited-lookahead plus
identification-cost của EH gốc, đồng thời biến coverage, stochasticity, target
return và abstention thành thành phần chính thức của bài toán offline.

## 10. Công thức chốt để dùng trong đề tài

```text
Primary object:
  P_OEH(M,beta;epsilon,delta)
    = {(k, log(c_k/delta) / I_{k,epsilon}(M,beta))}_{k=1..Kmax},
  với depth không epsilon-solvable hoặc I=0 được gán +infinity.

Primary scalar ở compute budget B:
  N_OEH(B) = min_{1 <= k <= B} N_k^ID.

EH-reference, chỉ để đối chiếu bài gốc:
  H_OEH^EH = min_k [k + log_A max{1,N_k^ID}].

Pilot estimator:
  P_hat_OEH^EB = {(k,N_hat_k^EB,status_k)},
  status_k thuộc {identified, unresolved, unsupported, not_epsilon_solvable}.
```

Đây là quyết định công thức cuối ở cấp nghiên cứu. Cross-fitting và kiểm tra
utility out-of-fold đã hoàn tất trên development. Phần chưa hoàn tất là chứng
minh finite-sample cho `N_hat_k^EB`, exact KL/target-return solver và evidence
ngoài cohort development. Kết quả hiện tại cho thấy proxy không còn bão hòa bằng
một scalar finite giả, nhưng không dự báo tốt hơn B2 và B3 không thêm utility.

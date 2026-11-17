# OEH bám sát Effective Horizon cho stochastic tabular

> **Cập nhật 19/09:** đây là `EB-OEH plug-in proxy` đã được kiểm tra về cơ chế
> bão hòa. Định nghĩa mục tiêu cuối, profile `(k,N_k)`, information requirement
> và cross-fitted estimator được chốt tại
> [OEH_FINAL_FORMULATION_2026-09-19.md](OEH_FINAL_FORMULATION_2026-09-19.md).

Ngày chốt prototype: **2026-09-19**. Trạng thái: **development-only,
exploratory**. Công thức này chưa thay estimator đã freeze và chưa được dùng để
mở evaluation split.

## 1. Vấn đề cần sửa

OEH-v1 định nghĩa một loss misranking `L_k(mu)`, giải số nhân dữ liệu nhỏ nhất
`mu >= 1` sao cho `L_k(mu) <= epsilon`, rồi tính
`k + log_A(mu)`. Trên toàn bộ development audit, loss tại `mu=1` đã nhỏ hơn
`epsilon` ở mọi depth, nên `mu_k=1` và tie-break luôn trả depth 0. Đây là một
**floor do phép threshold**, không chứng minh môi trường hoặc bài toán học là dễ.

Thiết kế mới quay lại hai thành phần của stochastic effective horizon trong
[Laidlaw et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/3e18f7152b5e1696e73722dc5121b6f2-Paper-Conference.pdf):

1. số bước planning `k`;
2. số mẫu cần để phân biệt greedy action, tỷ lệ theo `1/Delta_k^2`.

[Mã gốc của tác giả](https://github.com/cassidylaidlaw/effective-horizon)
cũng tạo `Q^1` từ random policy, lặp Q-value iteration, lưu return của greedy
policy và tìm `min_k`. Prototype này giữ cấu trúc đó nhưng dùng empirical
behavior policy vì mục tiêu là offline RL từ một dataset cố định.

## 2. Chuỗi Q và approximate solvability

Với reward công khai trong `[0, R_max]`, discount `gamma` và empirical behavior
policy `beta_hat`, đặt

```text
Q_hat^1       = Q^{beta_hat} trong empirical supported MDP,
Q_hat^{k+1}   = T_hat Q_hat^k,
Pi_hat_k      = toàn bộ policy greedy đối với Q_hat^k.
```

Index của EH gốc bắt đầu từ `k=1`; cột `depth` trong artifact bằng `k-1` để
tương thích code cũ.

Đúng với định nghĩa “mọi greedy policy” của EH, prototype đánh giá policy tệ
nhất trong `Pi_hat_k`, thay vì chọn một action bằng `argmax` rồi bỏ qua tie. Một
depth chỉ admissible nếu

```text
(J_hat^* - min_{pi in Pi_hat_k} J_hat(pi))
------------------------------------------------ <= epsilon = 0.05.
              J_hat^* - J_hat^min
```

Đây là quy ước approximate `k`-QVI-solvability 95% đo từ minimum return mà bài
EH gốc dùng trong phân tích thực nghiệm. Tỷ số này bất biến khi đổi đơn vị
reward.

## 3. Gap bám sát EH gốc

Chuẩn hoá action value về `[0,1]`:

```text
q_tilde_k(s,a) = (1-gamma) Q_hat^k(s,a) / R_max.
```

Tại state `s`, `Delta_hat_k(s)` là khoảng cách giữa greedy value và action có
giá trị lớn nhất trong các action **strictly non-maximizing**. Co-maximizer thật
không cần phân biệt, đúng với định nghĩa `k`-gap của tác giả.

Vì mục tiêu đồ án là expected return của offline learner, prototype chỉ lấy
maximum trên tập critical state

```text
C_k = {s: discounted occupancy của pi_k >= 1/(10|S|),
          s không terminal, có ít nhất hai observed actions}.
```

Đây là thay đổi estimand có chủ đích so với EH gốc, vốn lấy infimum trên mọi
state/time. Nó phải được gọi là **occupancy-scoped, behavior-supported EH**, không
được trình bày như theorem gốc.

Point version là

```text
N_gap,k   = max_{s in C_k} 1 / Delta_hat_k(s)^2,
H_point,k = k + log_A(N_gap,k).
```

Nếu model empirical cho tất cả supported actions bằng nhau, estimator không
được coi đó là exact tie. Pilot hữu hạn chưa phân biệt được hai action không
chứng minh hai true value bằng nhau. Depth đó trả
`unresolved_empirical_tie`; nếu mọi admissible depth đều như vậy, OEH abstain.
Oracle biết model thật mới được phép bỏ qua exact co-maximizer.

## 4. Extension cho stochasticity và coverage

Với leaf value dùng ở backup thứ `k`, định nghĩa Bellman target đã chuẩn hoá

```text
Y_k = (1-gamma) [R/R_max + gamma V_leaf(S')],   0 <= Y_k <= 1.
```

Cho action `a` tại state `s`:

- `nu_hat_k(s,a)` là Bessel-corrected sample variance của `Y_k`;
- `d_hat(s,a)=n(s,a)/N_0` là tỷ lệ transition của cell trong pilot;
- `ell=log(4|S||A|(K+1)/delta)` là union factor đã khóa;
- `Delta=Delta_hat_k(s)` là gap cần phân biệt.

Empirical Bernstein cho biến trong `[0,1]` có radius

```text
r(n) = sqrt(2 nu_hat ell / n) + 7 ell / [3(n-1)].
```

Đây là dạng của [Maurer và Pontil, COLT 2009](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf).
Giả sử dataset tương lai giữ cùng cell frequency, `n(s,a) ~= N d_hat(s,a)`.
Yêu cầu từng số hạng trong radius của cả greedy action và runner-up không quá
`Delta/8` cho ta điều kiện đủ bảo thủ

```text
N_var(s,a) = 128 nu_hat_k(s,a) ell / [d_hat(s,a) Delta^2],
N_lin(s,a) = [1 + 56 ell/(3 Delta)] / d_hat(s,a).
```

Khi đó radius của mỗi action không quá `Delta/4`, tổng hai radius không quá
`Delta/2 < Delta`, nên interval của hai action giữ đúng thứ tự empirical. Đặt

```text
N_EB,k = max_{critical (s, greedy, runner-up)}
         max_{a in {greedy, runner-up}} max(1, N_var(s,a), N_lin(s,a)),

H_EB,k = k + log_A(N_EB,k).
```

Depth được chọn là argmin trong các depth vừa admissible vừa identified, với
tie-break lấy depth sớm nhất.

## 5. Vì sao phải dùng từng biến

| Biến | Vai trò toán học | Nếu bỏ biến |
|---|---|---|
| `k` | Chi phí planning trong EH gốc | Chỉ còn độ khó phân biệt action, không còn effective horizon |
| `Delta_k` | Margin quyết định greedy action | Không biết cần chính xác đến mức nào để giữ action ranking |
| `nu_k` | Độ ngẫu nhiên của **Bellman target có liên quan đến value** | Hai transition kernel cùng gap/coverage nhưng variance khác nhau bị coi như nhau |
| `d(s,a)` | Số mẫu hữu dụng của cell trên mỗi transition tổng | Coverage hiếm và coverage dày bị coi như nhau |
| linear `1/(d Delta)` | Phần hữu hạn mẫu của Bernstein | Một cell quan sát một lần có variance bằng 0 sẽ bị kết luận sai là chắc chắn |
| `rho_0`, occupancy | Giới hạn state theo đóng góp vào expected return | Rare state có thể thống trị score dù gần như không ảnh hưởng endpoint |
| `beta_hat` | Leaf policy quan sát được từ offline data | Dùng random policy sẽ đo một bài toán khác với dataset đang có |

Entropy transition không được dùng trực tiếp. Stochastic successor chỉ khó khi
nó làm Bellman target biến thiên; chuyển ngẫu nhiên giữa các successor có cùng
value không nên tăng độ khó.

## 6. Các tính chất đã kiểm tra

1. **Khớp EH point term.** Khi gap chuẩn hoá là `Delta`, code trả đúng
   `k + log_A(1/Delta^2)`.
2. **Reward-scale invariance.** Nhân reward và `R_max` cùng một hằng số dương
   không đổi score, vì `q_tilde`, `Y`, gap và variance đều không đổi.
3. **Không còn threshold floor.** Không có phép giải `L_k(mu)<=epsilon` với
   `mu>=1`. Gap/variance/coverage thay đổi sẽ đi thẳng vào log sample
   requirement.
4. **Không suy deterministic từ một mẫu.** Khi sample variance bằng 0,
   `N_lin` vẫn dương và có thể lớn.
5. **Tie empirical không giả làm evidence.** Equality chưa xác định tạo
   abstention, không tạo multiplier 1.
6. **Monotonicity cục bộ.** Giữ các biến khác cố định, requirement tăng khi
   variance tăng, coverage giảm hoặc gap giảm.

Các tính chất trên được kiểm bằng `tests/test_oeh_eh_aligned.py`.

## 7. Kết quả development cuối

Artifact chuẩn: `artifacts/research/oeh_eh_aligned_final_v2_2026-09-19/`.

| Metric | Toàn bộ 225 pilot | 165 stochastic pilot |
|---|---:|---:|
| Có ít nhất một depth identified | 142 | 117 |
| Evidence EB non-flat | 120 | 99 |
| Total EB score non-flat | 126 | 101 |
| Selector EB trả lời | 138 | 113 |
| Selector EB abstain | 87 | 52 |
| Success với fallback depth 100 | 82 | 58 |
| Success của fixed depth 100 | 80 | 56 |
| Mean normalized-regret gain so với fixed 100 | 0.002349 | 0.003203 |
| Cluster-bootstrap 95% CI của gain | [0.000027, 0.004706] | [0.000063, 0.006113] |
| Better / worse / tied | 47 / 34 / 144 | 47 / 34 / 84 |
| Mean Spearman với exact-model evidence curve | 0.242 | 0.082 |

Kết luận về saturation phải tách làm hai:

- **Đã sửa floor cơ học:** không còn case nào được gán `mu_k=1` chỉ vì loss
  nằm dưới một threshold. Có evidence thì tính sample requirement; equality
  chưa xác định thì abstain.
- **Không ép mọi curve phải dao động:** trên stochastic pilots, 99/165 evidence
  curve non-flat; 52/165 pilot abstain. Flat do Q/gap thực nghiệm không đổi hoặc
  thiếu nhận dạng là một kết quả dữ liệu, không được “sửa” bằng jitter hay đổi
  threshold.

Utility development của EB tốt hơn point version và tốt hơn fixed depth 100
trung bình 0.0032 normalized regret, nhưng effect rất nhỏ. Success chỉ tăng từ
56 lên 58/165; selector tốt hơn ở 47 case, xấu hơn ở 34 và tied ở 84. Tương
quan với evidence curve của exact model chỉ 0.082. Vì công thức được phát triển
và kiểm tra trên cùng development cohort, cluster CI dương không phải bằng chứng
confirmatory. Kết luận đúng hiện tại là **đã thoát bão hòa công thức**, nhưng
**chưa chứng minh được OEH mới là predictor mạnh hoặc đủ chuẩn để thay endpoint
chính của paper**.

### Audit “thoát bão hòa hoàn toàn”

Audit hậu nghiệm riêng tại
`artifacts/research/oeh_saturation_completion_v2_2026-09-19/` định nghĩa
mechanical saturation là: (i) có ít nhất hai depth identified nhưng finite total
EH score vẫn phẳng, hoặc (ii) positive gap bị gán sample requirement ở floor 1.

- OEH-v1: 131/131 curve assessable nằm ở all-one multiplier floor.
- EH-aligned trên 225 pilot: **0 mechanical-saturation violation**.
- EH-aligned trên 165 stochastic pilot: **0 mechanical-saturation violation**.
- Phân loại stochastic: 99 finite/non-flat selector; 48 không có identified
  depth; 16 chỉ có một identified depth; 2 có evidence nhưng không depth nào
  admissible.
- Delayed-reward positive control làm point và EB cùng chọn depth 1; vì vậy
  selector không bị khóa cứng ở depth 0.

Theo định nghĩa audit này, OEH mới đã thoát hoàn toàn **bão hòa cơ học**. Các
case thiếu nhận dạng được giữ dưới dạng abstention/censoring; biến chúng thành
finite score chỉ để đạt 100% non-flat sẽ tạo evidence giả.

## 8. Giới hạn chứng minh

`H_EB` hiện là **plug-in diagnostic**, chưa phải finite-sample certificate.
Leaf value, action pair và variance đều được chọn/tính từ cùng pilot; empirical
Bernstein cho một target cố định không tự động bao phủ quy trình data-dependent
này. `d_hat` cũng ngoại suy rằng behavior proportions không đổi khi tăng budget.

Muốn nâng thành claim định lý cần một trong hai đường:

1. sample splitting/cross-fitting: một phần pilot khóa Q sequence, critical
   states và action pair; phần độc lập còn lại ước lượng variance/gap;
2. uniform confidence sequence trên toàn lớp depth/state/action, rồi chứng minh
   selection vẫn được bao phủ.

Kết quả development chỉ cho phép kết luận công thức đã loại floor cơ học và đo
được variation ở các case identified. Claim “OEH dự báo learner requirement”
chỉ được giữ nếu validation theo MDP, có fallback cho abstention, cho hiệu quả
ngoài mẫu so với coverage/quality baselines. Không được suy claim đó từ độ
non-flat của đường cong.

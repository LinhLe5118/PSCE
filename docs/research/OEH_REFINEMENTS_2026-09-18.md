# OEH: giảm độ bảo thủ và tách bước tạo policy — 18/09/2026

## Quyết định từ bằng chứng

Đã hoàn thành ablation 2×2 trên toàn bộ 225 pilot development lịch sử. KL kết hợp
policy thực nghiệm giúp một phần đường cong thay đổi theo depth, nhưng **chưa giải
quyết bão hòa theo nghĩa tạo certificate hữu ích hoặc chọn depth tốt hơn baseline
cố định**. Giữ ứng viên này trong nhánh exploratory của B3; chưa thay predictor mặc
định, chưa có kết quả dự báo learner requirement hay learner selection từ nó.

Nghiên cứu này tiếp nối [redesign](OEH_REDESIGN_2026-09-18.md) và
[smoke test âm tính](OEH_DEVELOPMENT_SMOKE_2026-09-18.md).
Số liệu máy đọc và provenance nằm trong
[report.json](../../artifacts/research/oeh_refinements_2026-09-18/report.json),
[summary.csv](../../artifacts/research/oeh_refinements_2026-09-18/summary.csv),
[curves.csv](../../artifacts/research/oeh_refinements_2026-09-18/curves.csv) và
[audit.json](../../artifacts/research/oeh_refinements_2026-09-18/audit.json).

## Tài liệu đã đối chiếu và phạm vi áp dụng

| Nguồn gốc | Điều có thể học | Trạng thái trong hệ thống |
|---|---|---|
| [Garivier & Cappé, COLT 2011: KL-UCB](https://proceedings.mlr.press/v19/garivier11a.html) | KL concentration cho quan sát bị chặn; khoảng bất đối xứng có thể tránh clipping rộng ở biên | Đã triển khai inversion Bernoulli-KL theo từng coordinate với phân bổ xác suất riêng dưới đây. Không phải chạy thuật toán online KL-UCB; không tuyên bố KL luôn chặt hơn Bernstein. |
| [Wang et al., AISTATS 2024: Don't Be Pessimistic Too Early](https://proceedings.mlr.press/v238/wang24h.html) | Lookahead pessimism tránh dừng quá sớm tại vùng bất định; LP-MDP có phân tích riêng về lợi ích lookahead | Đã đối chiếu mô tả phương pháp trên trang proceedings. Ablation tách proposal/evaluation ở đây chỉ kiểm tra cơ chế bảo thủ, chưa tái lập LP-MDP hoặc chuyển guarantee của bài đó. |
| [Krishnamurthy et al., AAAI 2025: Selective Uncertainty Propagation](https://ojs.aaai.org/index.php/AAAI/article/view/33977) | Ước lượng lợi ích khi lệch khỏi behavior, truyền bất định có chọn lọc theo distribution shift | Đã đọc abstract chính thức. Ưu tiên nghiên cứu sâu tiếp; bài xét finite horizon, cần ánh xạ và chứng minh bổ sung cho discounted stationary policies ở đây. Chưa triển khai SUP. |
| [Xie et al., NeurIPS 2021: Bellman-consistent Pessimism](https://proceedings.neurips.cc/paper/2021/hash/34f98c7c5d7063181da890ea8d25265a-Abstract.html) | Đặt pessimism trên tập hàm nhất quán Bellman, thay cho quá bảo thủ từng tọa độ | Hướng đối chiếu cho certificate phụ thuộc value; chưa triển khai. |
| [Koppel et al., ICML 2024: Information-Directed Pessimism](https://proceedings.mlr.press/v235/koppel24a.html) | Penalty dùng Stein discrepancy và cấu trúc mô hình | Chưa đủ cơ sở thay trực tiếp vào tabular confidence box; chưa triển khai hoặc đo. |
| [Panaganti et al., L4DC 2025: Bridging Distributionally Robust Learning and Offline RL](https://proceedings.mlr.press/v283/panaganti25a.html) | Liên hệ robust learning với offline RL dưới điều kiện coverage/concentrability | Đối chiếu giả định trước khi chọn robust baseline; không suy ra pilot hiện tại đủ để chứng nhận. |

Đây là tổng hợp định hướng và ablation tự triển khai, không phải tuyên bố một
thuật toán mới đã vượt các công trình trên. Những mục mới đọc ở mức abstract cần
đọc đầy đủ thuật toán, định lý và code trước khi gọi là replication.

## Công thức và hợp đồng thống kê

Với `d = S*A*(S+1)`, chia ngân sách lỗi cho mỗi scalar mean và mỗi số lượt thăm
`n >= 1` thành `alpha_n = delta / (d*n*(n+1))`. Cho empirical mean `x` của một
successor indicator hoặc reward đã chia cho public `Rmax`, lấy khoảng:

```text
I_n(x) = {q in [0,1]: n*kl(x,q) <= log(2*d*n*(n+1)/delta)}.
kl(x,q) = x*log(x/q) + (1-x)*log((1-x)/(1-q)).
I_0 = [0,1].
```

Hai đuôi Chernoff cho bounded observations cho xác suất bỏ sót ở fixed n không
quá `alpha_n`. Tổng theo mọi coordinate và mọi n không quá delta vì
`sum 1/(n*(n+1)) = 1`. Với stationary Markov sampling, áp dụng cho chuỗi outcomes
của từng `(s,a)` và union theo n; không giả định toàn bộ transition log là iid.
Không được lọc trajectory theo return rồi áp dụng guarantee này. Reward có thể
liên tục trong `[0,Rmax]`; không cần giả định reward là Bernoulli.

Đây là box của các khoảng scalar, giao simplex theo từng transition row, không
phải multinomial KL ball. Khi `x=0`, cận trên `1-exp(-ell/n)` vẫn giữ xác suất
successor chưa thấy; khi `n=0` giữ hoàn toàn bất định. Terminal được xử lý theo
hợp đồng state terminal absorbing với reward sau terminal bằng 0 của adapter;
không suy cấu trúc terminal từ timestep truncation.

Hai họ proposal:

1. **Pessimistic:** greedy backups từ value của empirical behavior trong robust
   model, giữ quy ước OEH-R cũ.
2. **Empirical:** tính value behavior trên point estimate, sau đó nominal greedy
   backups; ở state đã thăm chỉ đề xuất action đã thấy. Unknown row dùng self-loop
   reward 0 **chỉ để định nghĩa proposal**. Confidence set và optimal comparator
   vẫn chứa toàn bộ public actions/successors, không nhận pseudo-count.

Depth 0 ở cả hai là greedy policy từ baseline value, không phải behavior policy
nguyên trạng. Giữ cùng robust evaluator cho mọi candidate:

```text
U_k = min(1, (1-gamma)/Rmax *
              [sup_{M in C} J_M^* - inf_{M in C} J_M(pi_k)]).
certificate <=> U_k <= epsilon.
```

Hai extrema có thể thuộc hai MDP khác nhau: đây là relaxation bảo thủ của
worst-case regret trên cùng MDP. `pi_k` phụ thuộc dữ liệu vẫn được bao phủ bởi
sự kiện model nằm trong C. Tuy nhiên delta=0.05 áp dụng **riêng từng construction**;
muốn chọn hậu nghiệm giữa KL và Bernstein với guarantee chung phải phân bổ lại
delta. Bisection lấy endpoint ngoài; robust iteration dùng residual allowance.
Đây vẫn là tính toán floating point, chưa phải chứng nhận bằng interval arithmetic.

Không quy đổi U thành số mẫu tương lai. Chuẩn hóa bằng public `Rmax/(1-gamma)`
khác với benchmark NR dùng `J* - J_random`.

## Thiết kế phép thử đã chạy

- 15 MDP development lịch sử: 11 stochastic, 4 deterministic lock controls;
  5 behavior families × 3 prefix budget (2, 5, 10) = 225 pilot cases.
- 4 tổ hợp confidence/proposal × 101 depth (`0..100`) = 90.900 hàng kết quả.
- `gamma=.95`, `Rmax=1`, `epsilon=.05`, `delta=.05` mỗi confidence construction.
- Đọc prefix của pilot qua data-only contract và kiểm tra hash. Oracle model chỉ
  dùng chấm regret sau khi hoàn thành cả bốn estimates của pilot.
- Quy tắc exploratory chọn depth: depth nhỏ nhất có U trong `1e-9` của min U.
  Đây không phải một selector đã calibration.
- Phẳng số học nếu range không vượt `1e-9`; báo thêm range `>0.001` như mốc mô tả,
  không gọi mốc đó là mức ý nghĩa thống kê hoặc mức utility được xác nhận.
- Plan và source hashes được ghi trước vòng chạy. Vì đã xem thất bại OEH-R,
  đây là exploratory development, không phải preregistered confirmation.

## Kết quả

| Confidence + proposal | Nhiều policy theo depth /225 | Curve không phẳng /225 | Range >0.001 /225 | Curve ở trần U≈1 /225 | Có certificate /225 |
|---|---:|---:|---:|---:|---:|
| Bernstein + pessimistic (OEH-R cũ) | 0 | 0 | 0 | 223 | 0 |
| Bernstein + empirical | 134 | 0 | 0 | 223 | 0 |
| KL + pessimistic | 51 | 10 | 0 | 27 | 0 |
| KL + empirical | 134 | 66 | 2 | 28 | 0 |

Tách proposal sửa được cơ chế policy trùng nhau ở 134 pilot, nhưng Bernstein
vẫn không phân biệt các policy đó. KL giảm clipping tại trần; **giảm số curve
đúng bằng 1 không đồng nghĩa cận đã hữu ích**. Với KL + empirical, median min U
toàn cohort là 0.999872; riêng 165 stochastic cases là 0.999947, và min U tốt nhất
trong nhóm stochastic vẫn là 0.971524, rất xa ngưỡng 0.05. Range lớn nhất 0.002912.

Riêng stochastic: KL + empirical có 66/165 curve không phẳng, 23/165 curve còn
ở trần, 0/165 certificate. Các mức noise/coverage không phải matched factorial
grid nên không quy kết đây là tác động nhân quả của stochasticity.

### Có chọn policy tốt hơn không?

| Trên 165 stochastic pilots | Mean normalized regret (thấp hơn tốt hơn) |
|---|---:|
| Empirical proposal, cố định depth 0 | 0.034697603 |
| Empirical proposal, chọn min KL upper bound | 0.028549438 |
| Empirical proposal, cố định depth 100 | **0.024122998** |

So với depth 0 cùng họ proposal, KL selector tốt hơn ở 26 pilot, kém hơn ở 13,
còn lại ngang nhau trong tolerance. So với depth 100: tốt hơn 59, kém hơn 59;
regret trung bình **cao hơn 0.00442644**. Không kết luận selector thắng baseline
cố định. 52/134 curve có true regret thay đổi được chọn trúng mức regret tốt nhất
trong candidate grid; oracle này chỉ là reference, không dùng chọn depth.

Lợi ích so với depth 0 tập trung ở Garnet (mean gain 0.016878); layered chỉ
0.00000490 và bottleneck 0.00004691. Pilot prefixes và behavior variants cùng
MDP phụ thuộc nhau; không dùng 225 cases như 225 MDP độc lập để báo significance.
So sánh depth 100 và phân rã family là audit mô tả sau chạy, không là endpoint
confirmatory được đăng ký trước.

Không có observed bound violation ở 90.900 hàng chấm; không có false certificate
vì không có certificate nào. Kết quả đó không xác nhận độ chặt hay finite-sample
coverage tổng quát. Thời gian runner 55.68 giây trên cấu hình lưu trong report,
một lần chạy, không phải đánh giá runtime nhiều lần.

## Hướng tiếp theo và tiêu chí dừng

Ưu tiên cải tiến **cận chênh lệch giá trị giữa candidate và behavior/reference
trên cùng model**, với bất định phụ thuộc visitation/value, thay vì tiếp tục đổi
tên depth hoặc giảm epsilon/delta để có số đẹp. SUP, Bellman-consistent pessimism
và lookahead pessimism là các hướng cần đọc và đối chiếu đầy đủ cho việc này.
Chứng nhận improvement so với behavior là một estimand khác với near-optimality;
không được đổi tên nó thành global regret certificate.

Giới hạn thông tin vẫn tồn tại: ở bandit một state, chỉ thấy action a0 có reward
0.5, action a1 chưa được thấy. Hai môi trường có reward a1 bằng 0 hoặc 1 tạo ra
cùng dữ liệu. Policy luôn chọn a0 có regret 0 hoặc 0.5. Không phương pháp nào bảo
đảm regret ≤0.05 cho policy đó từ dữ liệu này nếu không thêm dữ liệu hoặc giả định
về a1. Discounted self-loop mở rộng ví dụ này giữ normalized regret như cũ.
Vì vậy không đặt tiêu chí mọi pilot bắt buộc thoát bão hòa.

Trước vòng triển khai tiếp, cần khóa estimand, comparator, giả định coverage và
protocol development mới. Đánh giá trên matched stochasticity × coverage × budget,
so với depth 0, depth 100 và empirical planning. Nếu đổi finite horizon từ bài SUP
sang discount, phải nêu truncation bias và phân bổ lỗi. Sau đó mới kiểm tra B3
có thêm giá trị dự báo learner requirement/selection so với B2 trên grouped folds.
Nếu không có utility bổ sung, giữ OEH như diagnostic/negative result và hoàn thiện
hướng chính B1/B2; không trì hoãn toàn đồ án để ép một scalar OEH phải thành công.

## Mã, tái lập và kiểm tra

- [oeh_refinements.py](../../code/oeh_refinements.py): KL box, empirical proposal,
  ablation 2×2; giữ nguyên estimator đã khóa và prototype OEH-R gốc.
- [Runner](../../scripts/analysis/run_oeh_refinements.py) và
  [audit](../../scripts/analysis/audit_oeh_refinements.py).
- [Tests mới](../../tests/test_oeh_refinements.py): endpoint KL, exact binomial
  tail budgets, unseen successors/actions, reward scaling, delayed stochastic
  reward, baseline equivalence, known regret, invalid counts.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -m scripts.analysis.run_oeh_refinements --dataset-dir datasets/week1_development_validation --output-dir artifacts/research/oeh_refinements_NEW_RUN
python -m scripts.analysis.audit_oeh_refinements --run-dir artifacts/research/oeh_refinements_NEW_RUN --dataset-dir datasets/week1_development_validation --baseline-dir artifacts/research/oeh_development_smoke_numpy_2026-09-18
python -m pytest tests/test_oeh_refinements.py tests/test_oeh_certificate.py tests/test_oeh_certificate_adapter.py tests/test_exact_solver.py tests/test_evaluation_leakage.py tests/test_oeh_estimator.py tests/test_oeh_validation.py -q
```

Kết quả kiểm tra: **95 tests và 7 subtests đạt**, trong đó 10 tests mới. Đây là
targeted suite, chưa tuyên bố full repository suite. Audit xác minh source,
artifact, manifest, index và input hashes; 225 baseline cases khớp OEH-R smoke
cũ; 22.725 empirical-policy regrets khớp giữa hai confidence methods. Các output
lịch sử giữ nguyên, evaluation sealed chưa được mở.

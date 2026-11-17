# B1 — Stochastic Dataset & Environment Analysis

Ngày review: **23/09/2026**. Người phụ trách: **Việt Trần**. Trạng thái:
**hoàn thành với giới hạn phạm vi đã ghi rõ**.

Báo cáo máy đọc nằm tại
`artifacts/week6/viettq_b1_b2_2026-09-23/b1_stochastic_dataset_analysis.json`.
Toàn bộ số liệu dưới đây được tổng hợp lại từ artifact development/probe đã có;
không sinh thêm development dataset và không đọc manifest, buffer hay outcome của
evaluation.

## 1. Đơn vị và các trục phải tách riêng

Đơn vị suy luận là **base MDP**. Năm behavior policies, nhiều prefix, mười seed
learner và ba learner không tạo thêm MDP độc lập. Bốn khái niệm không được gộp
thành một biến “độ khó”:

| Trục | Định nghĩa sử dụng | Quan sát hợp lệ |
|---|---|---|
| Transition stochasticity | Một MDP stochastic nếu có ít nhất một hàng `(s,a)` reachable, nonterminal có từ hai successor với xác suất `>1e-12` | Audit trên kernel được dựng lại từ recipe, không phải suy từ buffer hữu hạn |
| Behavior stochasticity | Cơ chế sinh policy ngẫu nhiên đã đăng ký: softmax, epsilon-soft, mixture hoặc shifted-dynamics | `policy_family` và parameter grid là design metadata; visitation entropy chỉ là hệ quả quan sát chung của policy, dynamics và sampling |
| Coverage | Tỷ lệ state-action quan sát được; metric đăng ký là coverage trên reachable decision states | Phải báo kèm count/support; zero entropy ở ô có một quan sát không chứng minh dynamics deterministic |
| Trajectory quality | Return moments và terminal-trajectory fraction của pilot | Mô tả dữ liệu thu được, không cô lập “chất lượng behavior” khỏi reward/dynamics |
| Data budget | Số transition theo bội số `|S||A|` | Prefix lồng nhau và seed lặp là measurements, không phải cluster độc lập |

## 2. Dynamics: kiểm bằng kernel, mô tả bằng pilot

Artifact `development60.json` dựng lại đúng 60 MDP development đã dùng, không
sinh trajectory mới. Kết quả là 44 stochastic MDP và 16 deterministic controls:

| Family | Base MDP | Panel | Hàng kernel stochastic / eligible | Median entropy transition quan sát ở x10 |
|---|---:|---|---:|---:|
| `uniform_garnet` | 16 | stochastic primary | 2,115 / 2,115 | 0.2123 |
| `forward_layered` | 20 | stochastic primary | 1,588 / 1,588 | 0.1425 |
| `bottleneck` | 8 | stochastic primary | 749 / 749 | 0.2496 |
| `lock_structured` | 16 | deterministic control | 0 / 328 | 0.0000 |

Kernel audit xác nhận ranh giới dynamics đúng theo thiết kế. Entropy empirical
không được dùng để đổi nhãn hay lọc cohort. T1 vì vậy tách evaluation thành 84
stochastic MDP (30 Garnet, 36 forward, 18 bottleneck) cho inference chính và 36
lock MDP cho đối chứng mô tả.

Controlled factorial 810 dòng cung cấp kiểm tra cơ chế có ghép cặp. Khi slip
tăng `0.00 -> 0.20`, weighted transition entropy tăng `0.1800`, còn CFQI
normalized return giảm `0.0356` trên 270 cặp. Đây là bằng chứng cơ chế mô tả;
sáu topology/reward blocks không đủ cho kết luận nhân quả phổ quát.

## 3. Behavior, coverage và trajectory quality

Năm họ behavior đều được giữ nguyên, không chọn lại bằng outcome. Trên 300 nhóm
của holdout development, median trajectory return lần lượt là `0.6751`
(`q_star_softmax`), `0.6068` (`perturbed_reward`), `0.5486`
(`shifted_dynamics`), `0.3221` (`structured_random`) và `0.2774`
(`two_policy_mixture`). Đây không phải ranking causal của policy vì family MDP,
reward, coverage và termination cùng tham gia vào return.

Trên artifact 15-MDP dùng đúng metric coverage đăng ký, 75 nhóm có khoảng
`0.1667–1.0000`, median `0.9286`. Các strata theo ngưỡng đã khóa
`[0.25, 0.50, 0.75]` là:

| Low | Medium | High | Very high |
|---:|---:|---:|---:|
| 2 | 4 | 18 | 51 |

Phân bố lệch mạnh về coverage cao là giới hạn thiết kế đã biết. Không re-bin
threshold sau khi xem số liệu. Trên 60-MDP holdout, `observed_sa_fraction` tính
trên toàn `S×A` có median `0.6545`; con số này **không thay thế được** metric
reachable-S×A ở bảng trên.

Terminal fraction và return dispersion biến thiên mạnh giữa MDPs. Ví dụ median
terminal fraction là `0.9412` cho `q_star_softmax`, `0.9524` cho
`shifted_dynamics`, nhưng `0` cho `perturbed_reward` và `structured_random`.
Điều này phản ánh cả horizon/termination của môi trường; không được diễn giải
đơn giản là hai policy sau “kém hơn”.

## 4. Budget và khả năng truy vết

- Pilot prefixes: `{2,5,10}|S||A|`; 900/900 dòng (300 nhóm × 3 prefix) khớp
  chính xác công thức số transition.
- Ở x10, số transition mỗi nhóm nằm trong `120–2,560`, median `720`.
- Training grid: `{1,2,5,10,20,50}|S||A|`, seed indices `0..9`.
- Core panel theo T1 gồm CFQI, IQL và tabular BC: 75,600 fits cho primary
  stochastic panel, 32,400 cho deterministic controls, tổng 108,000 fits.
- Feature/outcome evidence development có 60 base MDP, 300 behavior groups và
  900 learner labels. Chỉ `n=60` là số cluster cho suy luận development.

Các file nguồn, SHA-256 hiện tại và đối chiếu digest ghi trong JSON B1. Ba
artifact có hash generation-time đã được kiểm theo byte exact hoặc tương đương
line ending; nội dung CSV/JSON không bị sửa.

## 5. Kết luận và giới hạn suy luận

1. Transition stochasticity, behavior stochasticity, coverage, trajectory
   quality và budget đã được tách định nghĩa và truy được về artifact.
2. Kernel audit ủng hộ cách chia primary/control của T1; không có lý do khoa học
   để đưa lock MDP vào fit hoặc endpoint chính.
3. Coverage development thiếu cân bằng; giữ nguyên threshold đã khóa và báo hạn
   chế, không tạo thêm dataset hoặc sửa strata để làm đẹp phân bố.
4. Probe factorial cho thấy slip, behavior coverage và budget có tác động khác
   nhau; không có một scalar “difficulty” duy nhất được xác nhận.
5. Tất cả kết quả B1 là development/probe evidence. Không kết luận external
   validity, causal universality hoặc confirmatory utility của v4E.

Phần văn bản có thể đưa vào luận văn nằm tại
`docs/thesis/04a_Stochastic_Dataset_Environment_Analysis_2026-09-23.md`.

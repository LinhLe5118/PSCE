# PROJECT CONTEXT — READ ONLY

> **Mục đích:** file handoff chỉ để đọc, giúp một chat/người mới nắm lại toàn bộ
> bối cảnh mà không cần đọc lịch sử hội thoại. Không coi file này là nguồn định
> nghĩa khoa học tối cao và không sửa các con số trực tiếp tại đây. Khi thay đổi
> protocol, sửa `protocol_config.json`, code, test và tài liệu gốc trước, sau đó
> mới đồng bộ bản handoff này.

## 1. Thông tin nhanh

| Mục | Giá trị |
|---|---|
| Dự án | Difficulty-Aware Offline Reinforcement Learning — DAORL |
| Repository | `https://github.com/LinhLe5118/Devoted` |
| Workspace đã dùng | `C:\Users\Nguyen Viet\Downloads\Devoted` |
| Nhánh phần dataset | `feat/benchmark-dataset-v2` |
| Commit triển khai dataset đầu tiên | `394c5d641c48d7e03946648b22ac3546d53940a3` |
| Người phụ trách phần này | Việt Nguyễn — GitHub `Viet091104` |
| Email commit | `vietnqhe182505@fpt.edu.vn` |
| Ngày cập nhật candidate protocol | 17/08/2026 |

Nguồn ưu tiên khi có mâu thuẫn:

1. Quyết định mới nhất được cả nhóm xác nhận và ghi vào commit/PR.
2. `protocol_config.json` cho hằng số máy đọc của dataset benchmark.
3. Mục định nghĩa khoa học trong `PLAN_Final.md` và `paper/sections/03_definitions.tex`.
4. `DATASET_DESIGN.md` cho giải thích thiết kế dataset.
5. File handoff này chỉ tóm tắt, không được tự động ghi đè các nguồn trên.

## 2. Dự án đang nghiên cứu điều gì?

DAORL nghiên cứu khả năng dự đoán độ khó của một bài toán offline reinforcement
learning từ một pilot dataset nhỏ. Câu hỏi trung tâm:

> Chỉ nhìn vào dữ liệu offline ban đầu, có thể dự báo một thuật toán cần bao nhiêu
> transition để học được policy gần tối ưu trên một MDP chưa thấy hay không?

Trong offline RL, agent không được tiếp tục tương tác để thu thập dữ liệu theo
ý muốn. Khả năng học phụ thuộc mạnh vào:

- chất lượng behavior policy;
- state-action coverage và số đếm trên support;
- reward sparsity;
- độ sâu cần để reward truyền ngược;
- độ bất định của action gap;
- cấu trúc và stochasticity của MDP.

Dự án đề xuất **Offline Effective Horizon — OEH**, giữ ý tưởng “độ sâu + bằng
chứng” của Effective Horizon nhưng ước lượng hoàn toàn từ pilot dataset.

## 3. Các đại lượng chính

### 3.1. Normalized return

```text
NR(pi) = [J(pi) - J(random)] / [J* - J(random)]
```

MDP có dynamic range quá nhỏ bị loại trước:

```text
J* - J(random) < 0.01 / (1-gamma)
```

### 3.2. Nhu cầu dữ liệu thực nghiệm

Với mỗi kích thước trong:

```text
N ∈ {1,2,5,10,20,50} × |S||A|
```

thuật toán chạy trên 10 training stream độc lập. Một run thành công khi:

```text
NR(pi) >= 0.95
```

`N_emp` là mức nhỏ nhất mà ít nhất 8/10 run thành công ở mức đó và mọi mức lớn
hơn. Nếu hết grid vẫn không đạt, quan sát là right-censored; không thay bằng
`N_max` và không xóa khỏi phân tích.

### 3.3. OEH

Từ pilot dataset `D0`, estimator:

1. Đếm `n(s,a)` và số trajectory ID phân biệt; support cần cả hai count đạt 2.
2. Dựng empirical transition/reward model.
3. Đánh giá behavior policy bị giới hạn support; terminal có continuation 0.
4. Chạy Bellman backup theo độ sâu `k`.
5. Tính action gap và Bayesian bootstrap uncertainty theo trọng số trajectory
   `Dirichlet(1,…,1)`.
6. Tạo evidence curve `m_k`.
7. Chọn `k† = min argmin_k m_k`, tie tuyệt đối `10^-12`; action tie lấy ID nhỏ.
8. Dự báo `N_req` và kiểm tra K1–K5.

Terminal destination không phải decision state và không cần action support.
Occupancy chuyển terminal mass về `rho0` để khớp chuỗi episode của collector;
truncation không cắt động lực mô hình.

Exact `P`, `R`, `V*`, `Q*` chỉ dùng tạo benchmark, nhãn và chấm điểm; không được
đưa vào estimator data-only.

## 4. Phân công ban đầu

### Việt Trần — môi trường

Nhiệm vụ:

- 10–14 môi trường bảng;
- Combination Lock có độ dài thay đổi làm trục chính;
- GridWorld và Linear Chain;
- mỗi môi trường có `P`, `R`, initial distribution và terminal semantics.

Trạng thái:

- đã có 14 cấu hình trong `tabular_environments.py`;
- self-test 14/14 pass;
- 14 môi trường này hiện chỉ là path-test/smoke fixtures, không phải benchmark
  chính `120+15+3`.

### Đỗ Trọng Hưng — exact solver

Nhiệm vụ:

- value iteration;
- `V*`, `Q*`, toàn bộ optimal-action ties;
- exact policy evaluation;
- support restriction;
- occupancy, dynamic range và normalized return.

Trạng thái:

- đã có trong `code/exact_solver.py`;
- test exact solver pass;
- solver là oracle infrastructure, không import vào OEH estimator data-only.

### Việt Nguyễn — dataset generator

Nhiệm vụ:

- behavior policy có mức chất lượng điều chỉnh được;
- coverage có thể đo và phân tầng;
- sinh offline trajectory;
- dataset card/manifest đầy đủ và tái lập.

Trạng thái:

- implementation ban đầu đã được push trên `feat/benchmark-dataset-v2`;
- adaptive pilot và OEH data-only đã được tích hợp trên feature branch;
- full protocol generation bị chặn tới khi hoàn tất bốn validation pending;
- chi tiết đầy đủ nằm trong `vietnguyen.md` và `DATASET_DESIGN.md`.

### Linh — định nghĩa và paper

Nhiệm vụ:

- đóng băng định nghĩa `N_emp`, OEH và các ngưỡng khoa học;
- quản lý Plan/Overleaf/paper;
- kiểm tra deadline và authorship;
- review PR, không để thành viên tự merge.

## 5. Những quyết định đã xác nhận trong trao đổi

1. Benchmark chính là `120 + 15 + 3`.
2. Toàn bộ benchmark do dự án tự sinh.
3. 14 môi trường cũ chỉ dùng kiểm tra đường đi.
4. BRIDGE chỉ dùng tham khảo thiết kế/datasheet; không tải hoặc nhập MDP BRIDGE.
5. Coverage theo ba bước **sinh → đo → phân tầng**.
6. Không post-filter transition để ép một target coverage.
7. Phiếu mô tả phải đầy đủ nhất có thể nhưng không dài dòng/lặp nội dung.
8. Người đọc paper phải có thể clone code và tự sinh dataset, không bắt buộc tải
   một bundle dữ liệu có sẵn.
9. Các setting chưa được nhóm cung cấp được Việt Nguyễn chọn dựa trên Plan và
   thiết kế đang có; nhóm có thể yêu cầu sửa sau review.
10. Không tự merge PR; Linh review trước.

## 6. Benchmark đã triển khai

### 6.1. Split

| Split | Số lượng | Mục đích |
|---|---:|---|
| Evaluation | 120 | metric chính và superiority tests |
| Development | 15 | probe, runtime, censoring, hyperparameter, GO/NO-GO |
| OOD | 3 | minh họa ngoài phân phối, không tính superiority |

120 evaluation MDP được phân bổ đúng tỉ lệ trong `PLAN_Final.md`:

- Uniform Garnet: 30 (25%).
- Forward-layered Garnet: 36 (30%).
- Lock-structured: 36 (30%).
- Bottleneck: 18 (15%).

15 development MDP dùng quota `4/5/4/2`. Các level `S`, `A`, branching,
reward sparsity và stochasticity proxy được phân bổ bằng balanced permutation
cycle theo từng họ; dynamic-range regeneration giữ nguyên recipe stratum.

Ba OOD representative:

- Combination Lock.
- Grid Maze.
- RiverSwim.

### 6.2. Setting MDP

| Setting | Giá trị |
|---|---:|
| Master seed | `20260814` |
| Gamma | `0.95` |
| Reward range | `[0,1]` |
| Dynamic-range threshold | `0.01/(1-gamma)` |
| Seed stride khi regenerate | `10.000.000` |
| Max regeneration attempts | `100` |

Dynamic-range rejection chỉ áp dụng development/evaluation. Requested seed,
actual seed và rejection count nằm trong manifest. Full-structure smoke hiện có
5 seed ứng viên bị loại và tái sinh thành công.

## 7. Năm behavior policy

| Family | Định nghĩa | Grid | Dùng `Q*` thật |
|---|---|---|---|
| `q_star_softmax` | softmax theo `(1-gamma)Q*` | `T={0.001,0.003,0.010}` | Có |
| `perturbed_reward` | tối ưu reward nhiễu Gaussian `(s,a)`, clip `[0,1]`, epsilon-soft | `eta={0.05,0.15,0.30}` | Không |
| `structured_random` | epsilon-greedy theo Gaussian low-rank score | `epsilon={0.05,0.20,0.40}` | Không |
| `two_policy_mixture` | trộn argmax/argmin của shared score | `w={0.25,0.50,0.75}` | Không |
| `shifted_dynamics` | tối ưu trên action-permuted mixed kernel | `delta={0.15,0.35,0.55}` | Không |

Mỗi MDP nhận đúng một configuration từ mỗi family. Grid xoay theo ordinal để
không chọn parameter sau khi nhìn kết quả. Manifest ghi exact behavior return và
normalized return để kiểm toán chất lượng; OEH estimator không được đọc các field
oracle này. Development-only exact audit trên 15 MDP cho median normalized
return xấp xỉ `0.982/0.933/0.691` ở ba nhiệt độ mới; grid vẫn là candidate và
chờ Linh review.

## 8. Thu thập dữ liệu và coverage

### 8.1. Collection

| Setting | Giá trị |
|---|---|
| RNG | `numpy.random.PCG64` |
| Python đã kiểm thử | `3.12.10` |
| NumPy khóa | `2.5.1` |
| SciPy khóa | `1.18.0` |
| Horizon | `100` (đồng bộ `K_max=100`) |
| Pilot seed index | `0` |
| Training seed indices | `0–9` |
| Pilot seed offset | `10.000.000` |
| Training seed offset | `20.000.000` |
| Pilot candidate grid | `{2,5,10} × |S||A|` |
| `N0_eff` | mức nhỏ nhất qua data-only certificate/censor/numerical-bootstrap gate |
| Bootstrap OEH | trajectory-level `Dirichlet(1,…,1)`, `B=100` |
| Support OEH | raw count ≥2 và distinct trajectory count ≥2 |
| Mức tham chiếu cũ | `5 × |S||A|` |
| Training size grid | `{1,2,5,10,20,50} × |S||A|` |

Pilot và training stream độc lập bằng seed namespace. Pilot buffer chứa ba
prefix; training buffer chứa sáu prefix. Nếu không candidate nào dùng được,
OEH abstain/censor tại cap `10|S||A|`, không chèn hoặc thu thêm transition.
Audit mới nhất tại cap 10SA trên 15 development MDP `q_star_softmax` cho 2
selected, 11 censor và 2 abstain. Diagnostic 50SA cho 3 usable và 12 censor;
chi phí gấp 5 chỉ thêm một case usable nên cap không được mở để ép computability.
Mọi case tính được đều có `k†=0`, vì vậy exact-range audit là cổng bắt buộc.

Transition schema:

```text
state, action, reward, next_state,
done, terminal, truncated, trajectory_id, timestep
```

Không có audit transition, forced exploring start, lọc hậu kỳ hoặc resampling để
ép support. `done` chỉ nghĩa terminal; `truncated` biểu diễn timeout.

### 8.2. Coverage

Primary metric:

```text
observed_reachable_sa_fraction
```

Ngưỡng candidate, hiện vẫn chờ nhóm review:

- low: `x <= 0.25`;
- medium: `0.25 < x <= 0.50`;
- high: `0.50 < x <= 0.75`;
- very-high: `x > 0.75`.

Các metric phụ:

- observed coverage trên toàn bộ `S×A`;
- reachable state/state-action count;
- policy support observed fraction;
- fraction support có count ít nhất `n_min=2`;
- minimum support count;
- normalized visitation entropy;
- số trajectory.

Terminal reachable chỉ là destination, không phải decision state, nên action
absorbing tại terminal không nằm trong denominator coverage. Stratum dùng
measurement ở maximum prefix rồi gắn cho các prefix con.

## 9. Manifest/dataset card

Manifest ghi:

- identity, split, family và mọi seed;
- MDP shape, gamma, terminal và reward range;
- random/optimal return và dynamic range;
- policy family, parameter, hash, expected return, normalized return;
- pilot/training stream, horizon, size và slice;
- coverage metrics và stratum;
- NPZ SHA-256;
- logical array content SHA-256 không phụ thuộc compression;
- source SHA-256;
- Git commit và trạng thái dirty;
- Python/NumPy/SciPy version;
- registered/effective protocol, run mode, deviation và lý do.

`data_only_index.json` là interface dành cho estimator. Index ghi `S`, `A`,
`rho0`, size multiplier và vai trò ngân sách để không phải suy dimension từ các
state/action tình cờ xuất hiện trong pilot. Index không chứa exact return,
normalized return, coverage label hoặc policy parameter; ID và filename được
hash ổn định để tránh lộ family. Manifest được ghi sau cùng và chứa checksum của
data-only index.

Không commit thư mục `datasets/`. Người đọc có thể tự sinh lại; có thể phân phối
bundle riêng chỉ để tiết kiệm thời gian.

## 10. File quan trọng

### Nguồn khoa học

- `PLAN_Final.md`
- `paper/sections/03_definitions.tex`
- `paper/sections/04_method.tex`
- `paper/sections/05_experiments.tex`

### Môi trường và solver

- `tabular_environments.py`
- `ENVIRONMENTS.md`
- `code/exact_solver.py`
- `code/README.md`

### Dataset benchmark v2

- `protocol_config.json` — nguồn setting máy đọc duy nhất.
- `protocol_config.py` — loader config.
- `benchmark_environments.py` — sinh `120+15+3`.
- `behavior_policies.py` — năm behavior family.
- `generate_offline_datasets.py` — collection, coverage, manifest, CLI.
- `code/adaptive_pilot.py` — chọn tuần tự `N0_eff` chỉ từ data-only prefix.
- `code/oeh_estimator.py` — OEH data-only nối trực tiếp với adaptive selector.
- `DATASET_DESIGN.md` — thiết kế và rationale.
- `vietnguyen.md` — báo cáo phần việc Việt Nguyễn.
- `tests/test_benchmark_environments.py`.
- `tests/test_dataset_generation.py`.
- `tests/test_adaptive_pilot.py`.
- `tests/test_oeh_estimator.py`.
- `tests/test_protocol_config.py`.
- `.github/workflows/tests.yml`.

## 11. Kiểm thử đã chạy

Kết quả gần nhất trước khi tạo file handoff:

```text
63/63 unit + integration tests: PASS
14/14 path-test environments × 5 policies: PASS
138/138 benchmark MDP full-structure smoke: PASS
1.380 physical smoke buffers: PASS
git diff --check: PASS
```

Full-structure smoke dùng buffer rất nhỏ, không phải full experiment dataset.

## 12. Lệnh cần nhớ

Setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Test:

```powershell
python -m unittest discover -s tests -v
python generate_offline_datasets.py --smoke-test
```

Review sample:

```powershell
python generate_offline_datasets.py `
  --output datasets/review_sample `
  --limit-instances 3 `
  --max-size 1000 `
  --training-seeds 0,1 `
  --run-mode review `
  --deviation-reason "1% runtime and schema probe"
```

Full generation sau khi protocol được review, đổi sang `frozen` và tag:

```powershell
python generate_offline_datasets.py --output datasets/tabular_v2
```

Lệnh full hiện chủ động báo lỗi khi `protocol_status` còn là `candidate`.

Full default dự kiến:

```text
138 MDP
× 5 policies
× (1 pilot + 10 training streams)
= 7.590 physical buffers
× 6 nested prefixes
= 43.470 logical datasets
```

## 13. Git workflow

Nhánh dataset đã được push:

```text
origin/feat/benchmark-dataset-v2
```

PR có thể mở tại:

```text
https://github.com/LinhLe5118/Devoted/pull/new/feat/benchmark-dataset-v2
```

Không tự merge. Linh review trước. Trước khi làm tiếp ở chat khác:

```powershell
git status -sb
git branch --show-current
git pull --ff-only
python -m unittest discover -s tests -v
```

Không reset/xóa thay đổi chưa rõ nguồn gốc. Dataset sinh ra nằm trong `datasets/`
và đã được `.gitignore` loại trừ.

## 14. Việc tiếp theo

Ưu tiên gần nhất:

1. Hoàn tất development policy-quality probe và Linh review grid candidate.
2. Linh/nhóm review coverage threshold theo phân phối development.
3. Người phụ trách OEH chạy exact `k†` range audit trên development split.
4. Nếu cần, sửa `protocol_config.json` trước khi nhìn kết quả full.
5. Mở PR, đổi status sang `frozen` sau review và tag protocol commit.
6. Sinh full dataset, lưu manifest/checksum và chuyển data-only interface cho
   người làm OEH, CFQI và IQL.

Các hạng mục lớn của toàn dự án vẫn chưa có đầy đủ implementation trong repo:

- OEH estimator hiện có implementation candidate nhưng weighted bootstrap và
  support certificate còn phải qua review/probe 1%;
- tích hợp evidence curve `m_k` vào full experiment pipeline;
- CFQI/IQL và đo `N_emp`;
- survival model/censor-aware evaluation;
- cross-validation, cluster bootstrap, ablation và figure/table pipeline.

Không tuyên bố dự án hoàn thành chỉ vì dataset generator đã chạy.

## 15. Gợi ý cho chat/agent tiếp theo

Khi nhận task mới:

1. Đọc file này để lấy bối cảnh.
2. Đọc `protocol_config.json` để lấy setting thực thi.
3. Đọc `DATASET_DESIGN.md` và `vietnguyen.md` nếu task liên quan dataset.
4. Kiểm tra branch/status/test trước khi sửa.
5. Phân biệt rõ oracle metadata với feature mà OEH estimator được phép dùng.
6. Không nhập BRIDGE data; BRIDGE chỉ là nguồn tham khảo.
7. Không đổi protocol âm thầm. Mọi thay đổi phải đi cùng config, test, tài liệu và
   deviation record.

---

**Kết luận:** phần dataset generator candidate đã có code, setting máy đọc,
test, manifest và tài liệu tái lập. Protocol chưa frozen vì còn bốn validation
pending; full generation bị chặn có chủ đích. Công việc tiếp theo là review,
exact `k†` audit, đóng băng/tag protocol và tích hợp với OEH/algorithm pipeline;
chưa chạy full experiment.

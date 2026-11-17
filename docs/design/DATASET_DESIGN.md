# Thiết kế benchmark và offline dataset — candidate v2

## Trạng thái quyết định

Tài liệu này ghi lại candidate protocol ngày 16/08/2026. Trạng thái máy đọc hiện
là `candidate`, chưa được tag/freeze và chưa cho phép sinh full dataset. Yêu cầu
nhóm đã xác nhận:

- benchmark chính là **120 MDP đánh giá + 15 MDP phát triển + 3 minh hoạ OOD**;
- 14 môi trường trong `environments/tabular_environments.py` chỉ dùng kiểm tra đường ống;
- coverage đi theo đúng thứ tự **sinh → đo → phân tầng**;
- chưa sinh dataset để ép một mức coverage mục tiêu;
- dataset card/manifest đầy đủ để kiểm toán nhưng không lặp giải thích dài dòng.

Trong lúc chờ Linh/nhóm xác nhận bảng coverage, candidate dùng tạm ba mốc
`0.25`, `0.50`, `0.75`, tương ứng bốn tầng low/medium/high/very-high. Các mốc nằm
trong `protocol_config.json` nhưng chưa phải threshold final. Run review có thể
dùng `--coverage-thresholds` khi kèm `--run-mode review` và
`--deviation-reason`; manifest lưu cả giá trị đăng ký, giá trị hiệu lực và sai
lệch.

## Căn cứ thiết kế

- Random MDP dùng ý tưởng Garnet: kiểm soát số trạng thái, hành động và branching
  factor, bắt nguồn từ kỹ thuật sinh MDP của Archibald, McKinnon và Thomas:
  <https://www.research.ed.ac.uk/en/publications/on-the-generation-of-markov-decision-processes>.
- Trục cấu trúc và ba họ OOD bám giao thức trong `PLAN_Final.md` và tinh thần
  benchmark tabular của bài Effective Horizon/BRIDGE:
  <https://openreview.net/forum?id=Lr2swAfwff>.
- Tách chất lượng policy khỏi coverage vì nghiên cứu dataset-centric cho thấy
  cả hai thuộc tính đều ảnh hưởng offline RL:
  <https://proceedings.mlr.press/v199/schweighofer22a.html>.

Các nguồn trên định hướng thiết kế; các grid số cụ thể dưới đây là candidate chờ
development probe và review nhóm, không phải hằng số lấy nguyên từ một paper.

BRIDGE chỉ là nguồn tham khảo thiết kế. Không MDP, transition, label hay kết quả
thực nghiệm nào của BRIDGE được nhập vào benchmark `120 + 15 + 3`; người dùng
không cần tải BRIDGE để tái lập dataset của dự án.

Setting protocol và recipe strata máy đọc nằm trong `protocol_config.json`;
loader kiểm tra quota, range, family, recipe level, grid, seed namespace và
coverage threshold trước khi chạy. Code generator, exact solver và
`environments/tabular_environments.py` đều nằm trong source hash. Generator dùng PCG64 và
`requirements.txt` khóa NumPy `2.5.1` và SciPy `1.18.0`; manifest ghi Git
commit, trạng thái dirty, hash source và phiên bản runtime.

## Benchmark 120 + 15 + 3

### 120 MDP đánh giá

Phân bổ đúng tỉ lệ trong `PLAN_Final.md`:

| Họ | Tỉ lệ | Số MDP | Trục recipe được phân tầng |
|---|---:|---:|---|
| Uniform Garnet | 25% | 30 | `S`, `A`, branching, Dirichlet concentration, reward sparsity |
| Forward-layered Garnet | 30% | 36 | depth, width, `A`, branching, Dirichlet concentration; action-dependent backbone `p=0.9` |
| Lock-structured | 30% | 36 | chain length; dynamics deterministic |
| Bottleneck | 15% | 18 | community size, `A`, branching, bridge probability |

Family được interleave theo quota, không xếp thành bốn block. Trong mỗi
family/split, mỗi trục dùng permutation cycle xác định trước theo
`family_ordinal`; chênh lệch số lần xuất hiện giữa các level của cùng trục không
quá một. Nếu một seed bị dynamic-range rejection, seed thay đổi nhưng recipe
stratum giữ nguyên.

### 15 MDP phát triển

Tách seed hoàn toàn khỏi tập đánh giá và có quota gần tỉ lệ chính
`4/5/4/2`. Chỉ dùng benchmark pipeline, hyperparameter, censoring và cổng
GO/NO-GO; không đưa metric của chúng vào kết quả chính.

### 3 minh hoạ OOD

Một instance đại diện cho mỗi họ Combination Lock, Grid Maze và RiverSwim.
Chúng chỉ dùng minh hoạ ngoài phân phối, không tham gia phép kiểm superiority.

Mỗi MDP có reward trong `[0,1]`, `gamma=0.95`, seed và recipe tái lập. MDP trong
tập development/evaluation không qua dynamic-range filter được tái sinh bằng
seed mới có quy tắc; số lần loại được ghi vào manifest.

## Năm họ behavior policy

Mỗi base MDP nhận đúng năm policy. Grid candidate được xoay theo ordinal của MDP,
không chọn sau khi xem kết quả. Chất lượng policy chưa được gọi là calibrated cho
tới khi hoàn tất development-only probe.

| Họ | Cách sinh | Grid prototype | Dùng `Q*` thật |
|---|---|---|---|
| `q_star_softmax` | softmax của `(1−γ)Q*` | `T={0.001,0.003,0.010}` | Có |
| `perturbed_reward` | tối ưu reward sau nhiễu Gaussian `(s,a)`, clip `[0,1]`, rồi ε-soft | `η={0.05,0.15,0.30}` | Không |
| `structured_random` | ε-greedy theo score Gaussian low-rank | `ε={0.05,0.20,0.40}` | Không |
| `two_policy_mixture` | trộn argmax/argmin của cùng score low-rank | `w={0.25,0.50,0.75}` | Không |
| `shifted_dynamics` | tối ưu trên kernel trộn với action-permuted kernel, rồi ε-soft | `δ={0.15,0.35,0.55}` | Không |

Chỉ policy đầu được đọc `Q*` của MDP thật. Bốn họ còn lại có thể giải một MDP
phụ trợ do reward/dynamics/score khác tạo ra nhưng không truy cập `Q*` thật.

Grid nhiệt độ cũ `{0.03,0.10,0.30}` tạo reference policy quá yếu. Audit chỉ trên
15 development MDP cho median normalized return lần lượt khoảng `0.982`, `0.933`,
`0.691` ở candidate mới `{0.001,0.003,0.010}` khi đánh giá từng mức. Đây là căn
cứ development cho candidate, chưa phải xác nhận cuối của nhóm.

## Thu thập trajectory

Không có audit transition, exploring start cưỡng bức, post-hoc filtering hay
resampling để ép support. Mỗi buffer là một chuỗi rollout nguyên bản của behavior
policy. Terminal reset về `rho0`; task continuing được cắt episode ở horizon
`100`, đồng bộ với `K_max=100` của OEH.

Pilot và training stream dùng namespace seed độc lập. Pilot thu một buffer tới
`10|S||A|` và cung cấp ba candidate prefix `{2,5,10}|S||A|`. Selector data-only
chọn mức nhỏ nhất không abstain, không right-censor và giữ ít nhất 90% bootstrap
replicate hợp lệ về mặt số học; mức được chọn là `N0_eff`. Nếu hết `10|S||A|` vẫn chưa đạt,
OEH giữ kết quả abstain/censor và không mở rộng hoặc sửa dữ liệu. Training dùng
lưới:

```text
{1, 2, 5, 10, 20, 50} × |S||A|
```

Support yêu cầu đồng thời raw transition count `n_min=2` và xuất hiện trong ít
nhất hai trajectory ID phân biệt. `5|S||A|` được giữ làm mức tham chiếu để đối
chiếu kế hoạch cũ. Config validator buộc candidate grid, support rule, Bayesian
bootstrap scheme, horizon và `K_max` nhất quán.

Entrypoint `code/adaptive_pilot.py::select_effective_pilot_with_oeh` chuyển trực
tiếp buffer NPZ thành transition data-only, chạy `code/oeh_estimator.py` trên
từng prefix và lưu toàn bộ decision trace. Estimator đọc constants từ section
`estimator` của config chung, nhận dimension từ data-only index và dùng Bayesian
bootstrap ở mức trajectory với trọng số `Dirichlet(1,…,1)`. Trọng số dương giữ
support full-pilot trong mọi replicate; gate 90% chỉ còn bắt lỗi số học.

Audit cũ trên đủ 15 development MDP (`q_star_softmax`, một seed) cho
`15/15 abstained_at_cap`: 12 case fail bootstrap gate và 3 case critical support
rỗng. Weighted trajectory bootstrap đưa valid fraction lên `1.0`, nhưng audit
tiếp theo còn phát hiện terminal bị hiểu sai là decision state không support.
Sau khi sửa cả hai lỗi, selector tại cap 10SA cho 2 selected ở 5SA, 11 censor và
2 abstain. Diagnostic 50SA cho 3 usable, 12 censor, 0 abstain; tăng cap gấp 5
chỉ thêm một case usable nên không được dùng để ép kết quả. Coverage/pilot-policy
vẫn là validation pending; không hạ support, chèn transition hoặc đổi
abstain/censor thành số giả. Mọi case tính được trong audit này có `k†=0`, nên
exact-range audit là cổng bắt buộc trước freeze.

Schema NPZ:

```text
state, action, reward, next_state,
done, terminal, truncated, trajectory_id, timestep
```

`done` chỉ có nghĩa terminal môi trường; timeout nằm ở `truncated`. Prefix có thể
kết thúc giữa trajectory và không được sửa cờ để giả thành timeout.

## Coverage: sinh → đo → phân tầng

Sau khi sinh, pipeline đo:

- coverage trên toàn bộ `S×A`;
- coverage trên state có thể đạt dưới dynamics;
- phần support dương của behavior policy đã quan sát được;
- tỷ lệ support có count ít nhất `n_min`;
- count nhỏ nhất trên policy support;
- visitation entropy chuẩn hoá;
- số state, state-action và trajectory quan sát được.

Policy support và empirical coverage là hai khái niệm khác nhau. Softmax có xác
suất dương cho mọi action nhưng dataset hữu hạn vẫn có thể không quan sát state
hiếm. Tầng coverage được gán sau khi đo và không làm thay đổi buffer.

Terminal state được tính là reachable destination nhưng không phải decision
state: collector reset ngay khi đi vào terminal, nên action absorbing tại terminal
không nằm trong denominator state-action coverage hay policy support. Manifest
ghi riêng số reachable state, reachable decision state và reachable terminal.

## Phiếu mô tả trong manifest

Manifest v2 lưu vừa đủ các nhóm:

- identity: dataset ID, split, MDP/policy/stream seed;
- MDP: family, `S`, `A`, `gamma`, terminal, reward range, dynamic range;
- policy: family, tham số, cờ dùng `Q*`, hash ma trận policy, exact return và
  normalized return dùng riêng cho kiểm toán/chấm nhãn;
- collection: pilot/training, horizon, số transition, trajectory và prefix;
- coverage: toàn bộ metric đo và stratum;
- storage: file, slice, schema, SHA-256;
- provenance: benchmark seed, requested seed, seed sau tái sinh, số lần loại,
  Git commit, trạng thái dirty, hash toàn bộ source phụ thuộc và phiên bản
  Python/NumPy/SciPy/platform;
- execution: run mode, effective protocol, mọi deviation và lý do.

Giải thích phương pháp chỉ nằm trong tài liệu này; manifest chỉ chứa dữ kiện để
machine đọc và kiểm toán.

`manifest.json` là artifact kiểm toán và chứa oracle metadata, nên estimator
data-only không được đọc file này. Generator viết thêm `data_only_index.json`
chứa vị trí/slice/checksum, `S`, `A`, `rho0`, size multiplier và vai trò ngân
sách. Các dimension đầy đủ ngăn estimator suy sai số state/action khi một pilot
không quan sát được chỉ số lớn nhất. Coverage label, exact return, policy quality
và policy parameter đều bị loại khỏi index. Manifest ghi SHA-256 của index và
được viết sau cùng để đánh dấu run đã qua bước indexing/provenance. Dataset/group
ID và filename trong data-only index là hash ổn định; chúng không để lộ tên
structural family hoặc behavior family.

## Lệnh chạy

Kiểm tra đường đi trên 14 môi trường cũ × 5 policy:

```powershell
python -m scripts.data.generate_offline_datasets --smoke-test
```

Sinh bundle nhỏ trước khi chạy toàn lưới:

```powershell
python -m scripts.data.generate_offline_datasets `
  --output datasets/prototype `
  --limit-instances 3 `
  --max-size 1000 `
  --training-seeds 0,1 `
  --run-mode review `
  --deviation-reason "1% runtime, storage and coverage probe"
```

`--run-mode review` mặc định lọc split development trước khi áp dụng
`--limit-instances`; generator từ chối evaluation instance trong review run.
Có thể dùng `--split ood` cho kiểm tra minh hoạ riêng.

Khi nhóm giao bảng coverage, truyền ba ngưỡng:

```powershell
python -m scripts.data.generate_offline_datasets `
  --coverage-thresholds 0.25,0.50,0.75 `
  --run-mode review `
  --deviation-reason "candidate threshold table for group review"
```

Full data nằm trong `datasets/` và không commit vào Git. Chỉ commit code, test,
tài liệu và một manifest mẫu nhỏ nếu cần review schema.

Generator từ chối ghi vào thư mục không rỗng để tránh trộn artifact của hai run.
`--run-mode protocol` hiện bị chặn vì `protocol_status=candidate`. Chỉ chuyển sang
`frozen` sau khi hoàn tất bốn cổng trong config: development policy-quality probe,
group review coverage threshold, exact `k†` range audit và review adaptive-pilot
weighted bootstrap. Sau đó phải cập nhật config, test, tài liệu, tag commit rồi mới
sinh full dataset.

## Điểm có thể sửa sau review

- Có tăng mỗi họ OOD từ một instance lên nhiều seed hay không.
- Grid policy có được giữ nguyên sau probe 1% hay không.

Mọi thay đổi các điểm này hoặc bảng coverage phải ghi ngày, lý do và commit áp dụng; không chọn sau
khi xem metric cuối.

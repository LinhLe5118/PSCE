# T1 — Final Research Protocol & Statistical Freeze

Version **1.0.0**, ngày 21/09/2026. Trạng thái: **khóa quyết định nghiên cứu và
thống kê; chưa phải GO thực thi E1**. JSON và manifest SHA-256 kèm theo là một
gói; sửa bất kỳ nội dung đã khóa phải phát hành phiên bản mới trước khi mở dữ liệu.

H1 (formula freeze), D1 (runner preflight) và B2 (review độc lập) vẫn phải đạt;
không tự đánh dấu các task này hoàn thành từ T1. Không tạo thêm development dataset.
T1 không đọc manifest, buffer hoặc outcome evaluation; E1 là task riêng trong phân công tuần 6.

## 1. Câu hỏi, phạm vi và thay đổi so với lịch sử

RQ1: Với pilot độc lập, selector OEH-v4E có cải thiện return của learner ở
budget `10|S||A|` so với best-fixed chọn từ training MDPs không?

RQ2: Block `B2 + OEH3 + LC-PMS` có thêm thông tin dự báo requirement của
CFQI/IQL/BC ngoài B2 không? RQ3: B2 có thêm thông tin ngoài B1 không?

Đối tượng là distribution **tabular benchmark đã đăng ký**, gồm cả cấu hình
deterministic, không phải mọi stochastic MDP, continuous control hoặc real-world.
Unit suy luận là base MDP; behavior, prefix, learner và seed không phải MDP mới.
Estimand của forecast là support-constrained proxy; không phải exact KL
sample complexity, confidence certificate hay định lý instance-optimal.

T1 thay thế các quy định mở evaluation cũ đối với ứng viên v4E; không sửa các
quyết định âm của v1/v2/v3. V3 đã fail primary gate trên cohort 60 MDP. V4E
được chọn thích nghi trên chính cohort đó, kể cả khi trọng số bằng nhau và Holm
15 selector đạt. Không gọi việc chọn v4E là preregistered trên development.

## 2. Cohort và ngân sách cố định

- Toàn bộ 120 evaluation MDP theo recipe `protocol_config.json`: family counts
  30/36/36/18 (uniform_garnet/forward_layered/lock_structured/bottleneck).
- Năm behavior policies đã đăng ký, đủ 600 MDP-policy groups. Không chọn lại
  theo coverage, entropy, return hoặc độ dễ. Recipe/hash được pin; không đọc
  sealed manifest để xác nhận số lượng ở T1. Nếu E1 phát hiện khác thiết kế,
  lưu failure report và dừng, không thay cohort.
- Pilot prefixes `{2,5,10}|S||A|`; x10 là input chính. Prefix x2/x5 chỉ dùng
  tính contraction LC đã định nghĩa, không chọn pilot bằng outcome.
- Training grid `{1,2,5,10,20,50}|S||A|`, 10 training seeds `0..9`;
  pilot seed index `0`, offset pilot `10000000`, training `20000000`, PCG64.
  Công thức seed/buffer của generator được pin trong manifest nguồn; D1/B2
  phải chứng minh namespace pilot/training và development/evaluation tách nhau.
- Core panel: CFQI, tabular IQL, tabular BC; 108,000 fits dự kiến
  (`120*5*6*10*3`). Gamma `.95`, 100 updates, tolerance `1e-10`, IQL expectile
  `.7`, inverse temperature `3`; unknown-state policy uniform, greedy loại
  unsupported actions như implementation đã pin.
- Cùng training prefix và seed cho mọi learner; pilot không nối vào training.
  Thành công từng run là normalized return `>=.95`; không đổi theo learner.
- Normalization dùng exact scorer hiện hành `(J_pi-J_random)/(J_star-J_random)`,
  không clip; oracle chỉ dùng tạo label/scoring sau khi feature đã ghi checksum.
  Dynamic-range filter chỉ dùng lúc generation theo recipe; không loại thêm
  MDP vì return thấp. Invalid/nonfinite score là lỗi, không phải censoring.

SPIBB, VI-LCB, BCQ và CQL là **panel phụ cấu hình cố định** tại x10, cùng 10
seeds và buffers; không thuộc portfolio được selector chọn hoặc Holm chính.
Không thêm learner vào v4E sau mở evaluation. D1 khóa provenance và cấu hình
implementation, neural 500 updates, SPIBB N_wedge=10, VI-LCB cb=.05; không
tuning trên evaluation. Ngân sách update khác loại thuật toán không chứng minh
ngang compute/hội tụ; không gọi panel này là SOTA. Các so sánh phụ chỉ descriptive.

## 3. Phương pháp và grouped nested evaluation

Hai return regressors là `B2_OEH3` và `B2_OEH3_LCOEH4`; trung bình score
với weights `(0.5,0.5)`. Chọn score lớn nhất, tie chính xác theo thứ tự CFQI,
IQL, BC. Không clipping prediction, margin tuning hoặc chọn lại feature.
Survival model dùng `B2_OEH3_LC_PMS`. Danh sách cột được chép chính xác vào JSON
từ source đã pin; H1 phải đối chiếu công thức với những cột đó.

Evaluation là **nested cross-validation của thủ tục meta-learning đã khóa
trên cohort evaluation mới**, không phải deployment của một fitted model chỉ
học trên development. Không trộn development vào meta-training. Outcome của
outer-training được dùng tự động đúng thuật toán dưới đây; outcome của outer-test
không được dùng để chọn mô hình, best-fixed hoặc standardization của fold đó.
Đây là xác nhận thủ tục trên cohort độc lập với quá trình thiết kế, không phải
120 test MDP độc lập với tất cả các fitted models.

- Outer 5 folds: `int(sha256(base_mdp_id UTF-8).hexdigest(),16) % 5`.
  Cùng base MDP (mọi behavior, seed, learner, prefix) luôn cùng fold. Nếu recipe
  dùng nhiều ID cho cùng base topology, B2 phải bắt alias trước execution;
  không được coi seed replicate là cluster độc lập.
- Inner 4 folds deterministic balanced SHA-256 theo implementation đã pin.
  Return salt `return|{outer_fold}|{algorithm}|{feature_set}`; survival salt
  `outer={outer_fold}|algorithm={algorithm}|features={feature_set}`.
- Mean/std học chỉ trên training của từng inner/outer fold; constant-column
  scale=1. Return ridge `{.0001,.001,.01,.1,1,10,100}`, tối thiểu inner MSE;
  tie chọn ridge lớn hơn. Survival ridge `{.0001,.001,.01,.1,1}`, tối đa
  inner log likelihood; tie chọn ridge nhỏ hơn. Không mở rộng grid.
- Outcome return regression là mean của 10 seeds tại x10 cho từng group/learner.
- Best-fixed mỗi outer fold: learner có mean return cao nhất trên outer-training,
  tính trung bình seeds rồi behavior rồi MDP; tie CFQI/IQL/BC. Không dùng winner
  trên toàn cohort. Oracle-best mỗi test group chỉ là descriptive upper reference.
- Không trình bày fold như năm nghiên cứu độc lập. Frozen predictions, fold IDs,
  inner scores và chosen ridge phải được lưu để kiểm leakage.

## 4. Endpoint và family suy luận duy nhất

Trong bảng dưới, H1/H2/H3 là **hypothesis thống kê**, không phải task H1/H2
trong bảng phân công; execution gate ở mục 7 dùng tên task.

Trước hết lấy mean qua seeds, rồi năm behaviors trong mỗi MDP; pooled predictive
lấy thêm mean qua ba learners. Sau đó lấy mean đều qua base MDP. Với cohort
đủ các ô, đây cũng là group mean; không coi 600 groups/1,800 labels là n độc lập.

| ID | Contrast, chiều tốt | Null/margin | Vai trò |
|---|---|---|---|
| H1 | v4E return − fold-best-fixed return | delta <= .005 | Primary decision |
| H2 | pooled log likelihood hybrid − B2 | delta <= 0 | Confirmatory predictive |
| H3 | pooled log likelihood B2 − B1 | delta <= 0 | Confirmatory stochastic increment |

Holm step-down **m=3, family alpha=.05**, giữ cả hypothesis bất lợi/không đạt.
Raw p hai phía được dùng thận trọng cho claim chiều dương; chỉ pass khi effect
đúng chiều, `p_Holm<=.05` và CI lower vượt margin tương ứng. Không chuyển H2/H3
thành primary nếu H1 fail. Không dùng family 15 exploratory cũ làm family mới.

Margin `.005` là lựa chọn vận hành của protocol: nửa điểm phần trăm trên thang
normalized return. Đây **không** phải ngưỡng kinh tế được xác nhận, không lấy
từ lower CI và không chứng minh tiết kiệm compute. H1 >0 nhưng chưa vượt margin
được báo là positive but practical target unconfirmed. Không suy equivalence
hoặc vô ích từ p không đạt.

Integrated Brier trên toàn sáu mốc, success rate, regret so với oracle-best,
abstention/fallback, switch rate, từng learner/family và external panel là
secondary descriptive: báo estimates/CI, không thêm significance stars hoặc
claim confirmatory. Không thay log likelihood bằng Brier khi kết quả thuận hơn.
Success của selector lấy success counts đúng learner được chọn, không threshold
mean-return thay thế. Regret là oracle mean-return trừ selected mean-return.

## 5. Bootstrap và tính bất định

Primary inference dùng **2,000 paired base-MDP cluster bootstrap draws**, RNG
PCG64 seed `20260921`. Resample base MDP với replacement **trong từng outer
fold**, giữ số clusters mỗi fold. Cùng draw áp dụng mọi endpoint và method.
Giữ tất cả behavior/learner/seed/prefix của cluster cùng nhau; clone của một
MDP không được sang fold khác hoặc sang hai inner folds.

Trong từng draw phải **refit meta-models**, standardization, inner ridge choice
và fold-best-fixed với multiplicity weights (hoặc duplicate rows tương đương).
Unique IDs quyết định inner folds; multiplicity quyết định fitting/scoring.
Các frozen pilot features và base-learner outcomes được tái sử dụng; không cần
train lại RL learners hay thu thập dữ liệu. Đây là uncertainty của nested
procedure; bootstrap chỉ resample fixed OOF rows được giữ làm sensitivity,
không thay primary vì bỏ qua shared-training dependence.

Với estimate `d`, draws `d_b`, dùng basic bootstrap CI
`[2*d-q_.975(d_b), 2*d-q_.025(d_b)]`. Với margin `m`, centered two-sided
bootstrap p là `(1 + count(abs(d_b-d)>=abs(d-m)))/(2000+1)`.
Quantile linear interpolation, float64; không rounded trước testing. CI là
marginal 95%, **không** gọi là simultaneous CI; Holm quyết định familywise gate.
Không dùng strict-tail p=0. Bootstrap này là xấp xỉ, không finite-sample guarantee;
ridge/argmax không trơn vẫn là limitation phải nêu trong paper.

Draw có lỗi numerical/schema phải log và ngừng claim confirmatory, không bỏ
draw/đổi seed hoặc tiếp tục đến khi đủ draws thuận lợi. D1 phải dry-run refit
bootstrap trên fixture/development và ước lượng runtime trước GO. Nếu không
đủ compute, ghi NO-GO trước mở cohort; không âm thầm đổi sang fixed-OOF CI.
120 clusters là fixed sample size, không có interim significance stopping,
không bổ sung sample sau kết quả. Không tuyên bố đủ 80% power từ effect được
chọn thích nghi trên development; thiếu lực là một kết quả có thể xảy ra.

## 6. Censoring, abstention và failures

`N_emp` là mốc nhỏ nhất mà toàn suffix của grid đều có >=8/10 seed thành công.
Nếu không có suffix, right-censored sau mốc x50; không thay bằng 50, 51, vô cùng
trong regression hoặc chỉ phân tích finite cases. Event đóng góp survival trước
mốc rồi hazard ở mốc; censored đóng góp survival qua toàn grid. Brier dùng
indicator requirement chưa đạt trên **toàn grid**, hazard clip `[1e-12,1-1e-12]`.
Không hỗ trợ censoring sớm: training thiếu/invalid là failure, không phải nhãn.

Diagnostic không đủ support giữ status/indicator và encoding đã pin; finite
score với uncertainty cao không tự bị loại. Nếu một trong hai component không
thể tạo finite score cho đủ ba learners, selector **abstain toàn group** và
chọn fold-best-fixed, không renormalize weights hay chọn component còn lại.
Giữ cả những group này trong endpoint và denominator; báo reason/count.
Feature computation exception, file thiếu/corrupt và model-fitting error là
execution failure; không ngụy trang thành abstention hợp lệ.

Core grid thiếu bất kỳ ô nào: báo incomplete; không primary pass, không drop
MDP hoặc thay method. External panel failure không xóa core result, nhưng phải
ghi E1 incomplete_external và không claim đã hoàn tất toàn panel. Retry tối đa
một lần cho job crash hạ tầng với input/config/seed/output identity bất biến;
lưu lần lỗi. Numerical/algorithm error không retry bằng cấu hình khác.

## 7. Execution gate, stop rule và provenance

T1 + H1 + D1 + B2 phải có checksum/evidence; B2 review độc lập không được tự
nhận từ cùng người/agent tạo T1. Không còn yêu cầu thư GVHD. B2 chỉ kiểm metadata
niêm phong đã cung cấp và fixture/development, không đọc evaluation manifest.
Lần đọc manifest/buffer/outcome đầu tiên được ghi là **evaluation opened** ở E1.
Nếu bundle chưa được materialize, chỉ được sinh đúng recipe đã khóa ở E1;
không thay recipe hoặc sinh development mới. Check seal trước đọc và ghi access log.

Run ID cố định `week6_oehv4e_confirmatory_v1`; output mới
`artifacts/week6/oehv4e_confirmatory_v1/`. Refuse nonempty output trừ verified
resume: source/config/input hashes và job keys trùng, completed output hashes
khớp; chỉ chạy missing jobs. Checkpoints gồm seed, method, MDP, policy, budget.
Không hiển thị endpoint/ranking cho người dùng trong lúc chưa đủ grid. Seal
features trước join outcomes; sau đủ grid chạy S1 đúng một analysis plan.

Dừng khi hash mismatch, split overlap, schema/fold violation, thiếu core cell
không phục hồi được, hoặc estimator implementation khác freeze. Không stop sớm
vì kết quả đẹp/xấu. Lưu raw outputs/log/runtime/failure và mọi kết quả âm.
Sau mở cohort, sửa lỗi cần deviation report, giữ nguyên bản lỗi; không gọi
rerun đã biết outcome là một confirmatory study mới. Không retune method,
endpoint, margin, Holm family hoặc chọn subgroup cứu claim.

Checksum T1 không tự unlock `protocol_status` cũ. Runner development hiện tại
vẫn từ chối evaluation; D1 cần runner riêng tuân thủ freeze, không bỏ guard
toàn cục. T1 không có claim runner đã implement primary refit-bootstrap.

## 8. Claim map và nghiệm thu

| Bằng chứng | Claim được phép | Không suy ra |
|---|---|---|
| Saturation theorem + positive controls hiện có | Lỗi cơ học trong lớp statistic đã chứng minh | Mọi offline EH đều bất khả thi |
| V4E development + Holm 15 | Adaptive candidate có tín hiệu utility | Independent confirmation / 15 thử đã bao trùm mọi adaptivity |
| H1 pass | Selection utility vượt margin trên distribution/portfolio/budget đã khóa | SOTA, full-MDP guarantee, compute saving |
| H2 pass | Hybrid thêm predictive information ngoài B2 | OEH scalar chính xác bằng sample complexity |
| H3 pass | Block stochastic thêm predictive information ngoài B1 | Entropy gây khó theo quan hệ nhân quả phổ quát |
| H1/H2/H3 fail | Claim tương ứng chưa được xác nhận; báo effect/CI | Chọn lại endpoint hoặc xóa kết quả |
| Panel external x10 | So sánh cấu hình đã đo | Ranking SOTA được tune công bằng |

Nghiệm thu T1: câu hỏi/endpoint/comparator/folds/bootstrap/Holm/censoring/
abstention/margin/stop rule/claim map đều được quyết định; JSON có version,
SHA-256 và source pins; verifier chỉ đọc allowlisted files, không duyệt datasets.
Các deliverable H1/D1/B2 còn thiếu là **execution dependencies**, không phải
quyết định thống kê chưa chốt. Package chưa commit không được gọi là đã chứng
minh chronology bằng Git; E1 phải lưu commit và checksum trước access đầu tiên.

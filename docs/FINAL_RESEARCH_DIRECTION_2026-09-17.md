# Hướng nghiên cứu chốt — stochastic datasets cho Offline RL

Ngày quyết định làm việc: 17/09/2026. Theo yêu cầu của nhóm qua phiên làm việc này; chưa thay cho xác nhận đề cương của GVHD. Tài liệu này thay phần định hướng tương lai trong PHASE_B_PLAN và THESIS_NARRATIVE; giữ nguyên các protocol/artifact lịch sử.

Cập nhật nghiên cứu 18/09: [OEH redesign cho stochastic tabular](research/OEH_REDESIGN_2026-09-18.md)
đề xuất đo normalized policy regret trên confidence set, kèm prototype và phản ví dụ.
Đây là ứng viên exploratory cho B3; chưa có held-out utility hay dự báo số mẫu được
xác nhận, và không thay protocol/định nghĩa OEH-v1 đã khóa.

Smoke test 18/09 đã hoàn tất trên 15 MDP development lịch sử, 225 pilot cases:
OEH-R vẫn phẳng 225/225, 223 curve ở U=1, không có certificate. Riêng stochastic
cases là 165/165 ở U=1. [Kết quả và chẩn đoán](research/OEH_DEVELOPMENT_SMOKE_2026-09-18.md)
chưa ủng hộ thay OEH-v1 bằng OEH-R làm predictor mặc định.

Ablation tiếp theo 18/09 đã hoàn tất: [KL confidence và tách policy proposal](research/OEH_REFINEMENTS_2026-09-18.md).
KL + empirical proposal làm 66/225 curve không phẳng, nhưng chỉ 2 curve có range
trên 0.001 và vẫn không có certificate. Trên 165 stochastic pilots, chọn min U có
mean regret 0.028549, kém fixed depth 100 (0.024123). Đây là cải thiện cơ chế từng
phần, chưa chứng minh utility của OEH. Giữ B1/B2 là hướng chính, B3 là ablation;
bước nghiên cứu OEH tiếp theo nhắm vào cận chênh lệch value và truyền bất định có
chọn lọc, với estimand/protocol riêng trước khi triển khai.

Vòng đọc sâu và thử phương pháp mới 18/09 cũng đã hoàn tất:
[SPVI, coupled difference và LP support audit](research/OEH_SELECTIVE_METHODS_2026-09-18.md).
SPVI chuyển thể từ pseudocode tạo policy đa dạng hơn PVI nhưng mean regret ở
fixed depth 100 trên stochastic pilots là 0.057234, kém PVI (0.055308) và empirical
planning (0.024123). Cận chênh lệch chưa cấp certificate improvement nào; LP-MDP
chưa đủ chứng cứ về reachable support. Giữ mã để ablation, chưa thay OEH mặc định.
Hướng cải tiến cụ thể tiếp theo là ước lượng shift/value contrast trực tiếp hoặc
support inference có bảo đảm; không suy rằng các phương pháp gốc đều thất bại.

## 1. Tên và mục tiêu cuối cùng

Giữ tên đề tài **Difficulty-Aware Offline Reinforcement Learning**. Phạm vi triển khai: **chẩn đoán độ khó dữ liệu và hỗ trợ lựa chọn learner cho Offline RL trong MDP ngẫu nhiên** (Stochastic-Dataset Diagnostics and Learner Selection for Offline RL).

Câu hỏi chính: **Các đặc trưng coverage, chất lượng trajectory và bất định ước lượng từ pilot có dự báo được khả năng đạt target return của từng learner ở một ngân sách dữ liệu cho trước, và hỗ trợ chọn learner tốt hơn các baseline đơn giản không? Thành phần OEH có bổ sung thông tin không?**

Đầu ra cuối cùng là một quy trình pilot → đặc trưng → dự báo theo learner/ngân sách → chọn learner hoặc abstain, cùng đánh giá tính đúng, chi phí và giới hạn. Bão hòa OEH là failure analysis dẫn đến thiết kế này, không phải điểm kết thúc. Không đặt mục tiêu chứng minh OEH thắng bằng mọi giá; không hứa một thuật toán cập nhật Q mới, SOTA hay chuẩn A/A*/Q1 trước khi có kết quả.

## 2. Stochastic phải được định nghĩa đúng

- **Transition stochasticity:** cùng `(s,a)` có nhiều successor khả dĩ dưới một kernel cố định. Đây là trục chính. Một MDP được sinh ngẫu nhiên nhưng sau đó có transition deterministic không phải stochastic dynamics.
- **Behavior stochasticity:** policy thu thập chọn action ngẫu nhiên. Nó ảnh hưởng coverage nhưng không đồng nhất với transition stochasticity.
- **Finite-sample uncertainty:** sai số ước lượng do số trajectory/visit hữu hạn; bootstrap theo trajectory đo một phần bất định này. Không gọi mọi bootstrap variance là intrinsic environment noise.
- Reward noise là mở rộng riêng, chưa bắt buộc. Random reward được lấy một lần khi tạo MDP rồi giữ cố định không phải reward noise trên mỗi transition.

Hiện có stochastic Garnet/layered/bottleneck, deterministic lock control và stochastic corridor probes. Các kết quả gộp 48 MDP không tự tách được ảnh hưởng của stochasticity khỏi topology, coverage và behavior quality. Không giả định tăng Dirichlet concentration đồng nghĩa tăng độ khó hay tăng mọi loại uncertainty.

## 3. Phương pháp được chọn để triển khai tiếp

Giữ mô hình survival rời rạc có regularization, fit riêng từng learner, dự báo `P(N_emp <= N | pilot)` trên lưới dữ liệu. Đây là **dự báo empirical requirement theo định nghĩa suffix-success**, không đồng nhất với xác suất một training seed đơn lẻ thành công. Diagnostic dùng pilot độc lập; learner labels dùng training streams riêng. Không dùng `P,R,V*,Q*`, exact return hoặc oracle stochasticity làm feature.

Thiết kế bốn nhóm đặc trưng lồng nhau, cùng learner label, folds và eligible cohort:

| Nhóm | Vai trò |
|---|---|
| B0: intercept + size/state-action cardinality | Mốc độ khó cơ bản |
| B1: B0 + observed coverage/counts + visitation entropy + pilot trajectory-return summaries | Baseline dataset-only mạnh; không dùng oracle behavior return |
| B2: B1 + empirical conditional transition entropy + coverage-stratified trajectory-bootstrap uncertainty | Phương pháp ứng viên chính, bổ sung thông tin stochastic/estimation uncertainty |
| B3: B2 + OEH gap/dispersion/depth summaries | Ablation kiểm tra phần OEH có thêm thông tin không |

Transition entropy ở low-count cells có bias: báo số mẫu/support, không xem entropy bằng 0 từ một observation là bằng chứng deterministic. Bootstrap không được trình bày như bảo đảm coverage của prediction interval. Feature mới, regularization grid và cách tổng hợp phải ghi trong protocol mới trước execution; chưa có kết quả B2/B3.

Đầu ra hiển thị learner, budget, predicted probability, support/abstention và phạm vi áp dụng. Lựa chọn learner dùng dự báo requirement ở budget người dùng, cùng ngân sách training mỗi learner. Abstain khi pilot không đạt support contract hoặc không có dự báo hợp lệ; ngưỡng xác suất quyết định được chọn trong inner-development validation và khóa trước confirmatory run. Chưa đưa ra lời khuyên “thu thêm X mẫu chắc chắn sẽ giải được”.

## 4. Thiết kế thực nghiệm và điều kiện đối chứng

### Giai đoạn A — đóng tính đúng và phân tích development hiện có

Sửa pairing/Brier; tái tính 15 và 48 MDP; giữ cohort riêng. Các kết quả hiện có là development, không phải confirmatory cho hướng mới. Cần thêm baseline pilot-return thực sự data-only; các dòng behavior_quality/value hiện tại dùng oracle chỉ là reference.

### Giai đoạn B — thí nghiệm stochastic có kiểm soát

Tạo protocol phiên bản mới, development-only trước. Dùng matched topology và reward function, thay mức transition mixing/slip, behavior coverage và dataset budget theo factorial grid. Có deterministic control. Thay transition có thể đổi optimal return/action gap: đo và báo các thay đổi này ở phía scoring, không giả định giữ nguyên difficulty hoặc coverage. Không lọc case theo kết quả thuận lợi.

Khóa danh sách generator/seed/pilot/train/calibration splits trước chạy; pilot tách training streams. Base topology cùng tất cả stochasticity/behavior variants phải nằm cùng fold và cùng bootstrap cluster, tránh leakage qua biến thể gần giống nhau. Dùng outer grouped folds để chấm, inner grouped folds để chọn feature/regularization/threshold. Fit scaling/imputation trong train fold. Báo missingness và abstention trên toàn cohort, cùng common-cohort scores; không chỉ báo nhóm diagnostic trả lời được.

CFQI, tabular IQL và BC là panel lõi đã có. Thêm một replication từ code tác giả trên **benchmark công khai có stochastic dynamics, action/representation tương thích**, cùng protocol dataset; chốt environment/license/adapter trước khi chọn neural CQL/IQL. Không áp neural continuous-action code lên tabular discrete tasks chỉ để có tên baseline. Runtime probes Hopper hiện tại không đáp ứng yêu cầu này. External replication là hạng mục còn thiếu của mục tiêu cuối, không ghi như đã hoàn thành.

Kết quả same-machine ngày 17/09 được lưu tại
`artifacts/week5/same_machine_comparison_2026-09-17/REPORT.md`: panel tabular đã đo
đủ 240 nhóm cho ba seed runtime; CQL/IQL neural, Effective Horizon và OfflineRL
chỉ là probe component có provenance/caveat riêng. Kết quả này đóng khoảng trống
compute accounting, nhưng không đóng hạng mục external replication tương thích.

Evaluation 120 MDP cũ tiếp tục sealed theo NO_GO. Vòng mới dùng confirmatory split mới hoặc một amendment ghi rõ việc tái sử dụng split cũ, khóa trước khi mở; không gọi development đã xem là held-out. Quyết định freeze cần đủ tính đúng và preflight, không chỉ đổi `protocol_status` để runner chạy.

### Giai đoạn C — giá trị quyết định và cùng cấu hình máy

Đo riêng (i) dự báo requirement, (ii) lựa chọn learner, (iii) chi phí. Lựa chọn so với fixed CFQI, fixed IQL, BC và best-fixed learner chọn từ training folds; oracle-best chỉ là reference. Tất cả policy đánh giá cùng oracle/rollout budget, cùng return normalization và seed schedule. Báo realized return, success rate, regret so với oracle-best và abstention coverage; xử lý abstention theo fallback đã đăng ký, không loại khỏi metric tiện ích.

So trên cùng máy bằng manifest CPU/GPU/RAM/OS/driver/library/threads, cùng dataset hash và seed, tuning budget và stop criteria rõ ràng. Không nhất thiết cùng batch size nếu implementation khác, nhưng công bố configuration và budget. Đo load/preprocess, feature/bootstrap, calibration/selection, learner fitting, evaluation và tổng wall time. GPU phải synchronize; tách startup/warmup; ít nhất 3 lần đo runtime đại diện. Không suy speedup end-to-end từ kernel speedup.

Phép đo prospective compute saving phải tính cả meta-training/amortization và chi phí diagnostic; chỉ gọi “tiết kiệm” khi cùng quality target và có accounting đầy đủ. Offline retrospective selection utility có thể đo trước, nhưng không được gọi là compute đã tiết kiệm thực tế.

## 5. Tiêu chí quyết định, không thay sau khi thấy test

Primary predictive endpoint: paired out-of-fold log-likelihood B2 so với B1 trên cùng observations; cluster CI 95% theo base topology và Holm cho panel learner đã khóa. Brier trên **toàn grid** là secondary, báo reliability và censoring. Effect size tối thiểu thực dụng phải được chốt từ development/chi phí trước confirmatory run; chưa có dữ liệu để bịa ngưỡng cải thiện 5% hay 10%.

Primary decision endpoint: cải thiện mean normalized return của selector so với best-fixed learner ở budget cố định, tính trên toàn test cohort với fallback cho abstention; CI theo topology. Báo thêm success, regret, chi phí. Chỉ claim utility nếu hiệu ứng đạt ngưỡng thực dụng đã khóa và uncertainty ủng hộ; nếu không, gọi là chưa xác nhận.

- B2 có ích, B3 không thêm ích: sản phẩm dùng B2; OEH giữ vai trò negative control/failure analysis.
- B3 thêm ích ổn định: giữ thành phần OEH có bằng chứng; không suy rằng scalar OEH ban đầu đã đúng.
- B2/B3 không vượt B1: chốt kết quả giới hạn và dùng baseline tốt nhất trong demo; không tuyên bố difficulty-aware selector mới đã thành công. Đánh giá lại scope với GVHD trước mở rộng compute.
- Thí nghiệm stochastic không tách được confound: sửa thiết kế development, không tiêu confirmatory split.

## 6. Luận văn, paper và deliverable

Luận văn: (1) vấn đề và stochastic-data scope; (2) foundations/prior work; (3) OEH và failure analysis; (4) diagnostic và selection protocol; (5) stochastic benchmark và đánh giá cùng máy; (6) kết quả, ablations, utility, limitations; (7) kết luận. Phân biệt completed findings với planned contributions xuyên suốt abstract/contributions.

Paper hiện tại giữ phạm vi audit của kết quả đã có; cập nhật lỗi, không thêm kết quả B2/B3 chưa chạy. Chưa dùng “đã được đăng ký trước” nếu Git history không chứng minh chronology. Demo cuối chạy offline từ artifact chốt, cho thấy support, dự báo, learner choice và trường hợp abstain; không kết luận dataset không học được từ OEH saturation.

Checklist hoàn tất hướng này: tính đúng → protocol stochastic → B1/B2/B3 + learner panel → replication phù hợp → selection/runtime → luận văn/demo/release. Mục tiêu hạng cao cần novelty và external evidence; không được dùng như lời hứa nghiệm thu.

## Nguồn và giới hạn của quyết định

Đọc `DAORL_thesis.docx` (bản đang dùng), `Drive/Thesis/01_Introduction.md`, paper hiện hành, PHASE_B_PLAN, config expanded và audit 17/09. Bản DOCX mới chỉ có Introduction và references, nên hướng này chốt phần nghiên cứu còn lại, không phải xác nhận đã hoàn tất các chương.

[Schweighofer et al., 2022](https://proceedings.mlr.press/v199/schweighofer22a.html) mô tả TQ/SACo và thực nghiệm trên deterministic MDPs; đó là cơ sở để đánh giá thêm stochastic dynamics, không chứng minh chưa có ai làm bài toán này. [Laidlaw et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3e18f7152b5e1696e73722dc5121b6f2-Abstract-Conference.html) nghiên cứu effective horizon trong stochastic environments; không thể mặc nhiên chuyển guarantee tương tác sang pilot offline. Novelty cần literature review chuyên biệt trước khi viết claim công bố.

## Cập nhật OEH bám sát EH — 19/09/2026

Đã triển khai và audit công thức point `k + log_A(1/Delta_k^2)` bám sát
stochastic EH gốc, cùng extension empirical-Bernstein dùng Bellman-target
variance, behavior coverage và linear finite-sample remainder. Thiết kế, phép
suy ra và lý do dùng từng biến nằm tại
[OEH_EH_ALIGNED_2026-09-19.md](research/OEH_EH_ALIGNED_2026-09-19.md).

Trên 165 stochastic development pilots, EB evidence non-flat ở 99 case, trả
selector ở 113 và abstain ở 52 do empirical tie/không đủ nhận dạng. Với fallback
depth 100, success tăng 56 lên 58 case; mean normalized-regret gain là 0.003203,
nhưng mean within-curve Spearman với exact-model evidence chỉ 0.082. Vì đây là
development-only post-design audit, kết luận được phép là: **đã loại floor
`mu=1` gây bão hòa cơ học; chưa xác nhận OEH mới có predictive utility đủ mạnh**.
Do đó B3 tiếp tục là ablation ứng viên, chưa thay B2 làm hướng chính và chưa
được đưa vào confirmatory protocol trước khi có sample-split/cross-fit cùng
grouped out-of-fold validation.

Audit saturation bổ sung ngày 19/09 phân loại toàn bộ 225 pilot và chạy
delayed-reward positive control. Không còn case nào có nhiều depth identified
nhưng finite total EH score phẳng, cũng không positive gap nào bị gán requirement
ở floor 1. Positive control chọn depth 1 cho cả point và EB. Vì vậy vấn đề
**bão hòa cơ học được xem là đã đóng**; 83/225 pilot không có identified depth
và 16/225 chỉ có một depth được báo là non-identification/censoring, không được
đổi tên thành “đã thoát bão hòa” bằng cách tạo số giả. Báo cáo tái lập nằm tại
`artifacts/research/oeh_saturation_completion_v2_2026-09-19/REPORT.md`.

### Công thức OEH chốt ở cấp nghiên cứu

[OEH_FINAL_FORMULATION_2026-09-19.md](research/OEH_FINAL_FORMULATION_2026-09-19.md)
thay thế việc xem point/EB score hiện tại là định nghĩa OEH cuối. Đối tượng chính
là profile `(k,N_k^ID)`: `k` là limited-lookahead depth theo chuỗi QVI của EH gốc;
`N_k^ID` là information-identification scale dưới behavior coverage, được xác
định bởi KL tới alternative MDP gần nhất làm thay đổi tập policy đạt target.
Không coverage hoặc depth không đạt target thì trả `+infinity`, không ép ra một
số finite.

Với chủ đề dự báo learner/sample budget, scalar chính là
`N_OEH(B)=min_{k<=B}N_k^ID`. Đại lượng
`min_k[k+log_A(max(1,N_k^ID))]` chỉ là EH-reference, vì trong offline RL các
Bellman backup tái sử dụng cùng buffer và cộng cứng `k` có thể đếm trùng chi phí
dữ liệu. Code `oeh_eh_aligned.py` được gọi chính xác là **EB-OEH plug-in proxy**;
chưa phải estimator finite-sample của target mới. Trước confirmatory run phải
thực hiện trajectory cross-fitting, lower-confidence gaps, bỏ occupancy threshold
tùy ý, và đánh giá B3 trên cùng grouped out-of-fold folds với B1/B2.

### Áp dụng hệ thống và audit hoàn thành — 19/09/2026

Contract profile mới đã được nối vào proxy hiện hành và áp dụng lên 225 pilot,
22.725 depth rows tại
`artifacts/research/oeh_final_profile_application_v2_2026-09-19/`. Data objective
chọn depth khác EH-reference ở 82/138 profile selectable. Trên 165 stochastic
pilots, success với fallback là 59 cho data objective, 58 cho EH-reference và 56
cho fixed depth 100; chênh lệch data objective với EH-reference chưa rõ
(`95%` cluster CI của regret difference cắt 0). Đây là development result.

Audit toàn repo tại
[RESEARCH_COMPLETION_AUDIT_2026-09-19.md](research/RESEARCH_COMPLETION_AUDIT_2026-09-19.md)
kết luận engineering/profile integration đạt. Bản audit ban đầu xác định
cross-fitting, B0–B3 và learner-selection utility là các khoảng trống phải đóng.

### Development B0–B3 và learner selection — 19/09/2026

Ba khoảng trống trên đã được implement và chạy sau khi khóa
[development amendment](research/STOCHASTIC_DEVELOPMENT_AMENDMENT_2026-09-19.md).
Artifact chính là
`artifacts/research/stochastic_b0_b3_oeh_v2_development_final_2026-09-20/`:

- cross-fitted EB-OEH chạy 225/225 pilot, 22.725 depth rows, 0 lỗi; 213 case
  `unsupported`, 12 `unresolved`, không ép case nào thành finite requirement;
- B0–B3 tạo 900 nested grouped OOF predictions cho CFQI, IQL và BC;
- B2 không cải thiện B1 ổn định; B3 không cải thiện B2;
- tại budget `10|S||A|`, B2 selector có mean return `0,5899`, best-fixed
  `0,6803`, difference `-0,0904`, 95% MDP-cluster CI
  `[-0,1546; -0,0325]`;
- cả bốn selector B0–B3 đều kém best-fixed với CI nằm dưới 0.

External panel tương thích đã hoàn tất: Pi_b-SPIBB 4.500 runs, VI-LCB paper-spec
9.000 runs, cùng 75 GPU fits cho mỗi discrete BCQ và CQL. Ở paired budget-x10,
seed-0 slice, mean return là CFQI `0,6729`, IQL `0,5084`, BCQ `0,4929`, CQL
`0,4746`, SPIBB `0,4499`, BC `0,3510`, VI-LCB cb=.05 `0,2559` và cb=144
`0,1018`. Đây là panel cấu hình cố định; neural one-seed/500-update run không
phải tuned SOTA benchmark.

Controlled stochastic factorial cũng đã hoàn tất 810 cell trên sáu matched
topology/reward blocks. Slip, behavior coverage và nested budget được thay độc
lập. Các diagnostic phản ứng đúng với factors, và budget 50 so với 2 làm CFQI
return tăng `0,2001`; nhưng OEH vẫn 0 identified, 568 unsupported và 242
unresolved. Artifact tại
`artifacts/research/controlled_stochastic_factorial_2026-09-19/`.

Do development gate không đạt, evaluation 120 MDP tiếp tục sealed. Hướng hiện
tại được chốt là một negative development result: CF-EB-OEH sửa finite-scalar
saturation bằng abstention có nguyên nhân, nhưng B2/B3 và learner selector hiện
không đạt utility claim. Khoảng trống controlled factorial, external panel và
nháp nội dung luận văn I–VII đã được đóng. Mục tiêu nghiên cứu vẫn chưa hoàn
thành theo claim ban đầu vì predictive/decision gate thất bại; deliverable còn
thiếu là ghép template luận văn và biên tập paper theo venue. Demo/release offline
đã có HTML tự chứa, CLI fallback và SHA-256 manifest. Không
retune trên 75 groups hiện tại rồi gọi kết quả mới là confirmatory.

## Cập nhật OEH v2 cuối — 20/09/2026

OEH v2 đã sửa điều kiện vòng tròn của proxy cũ: ước lượng requirement tương lai
không còn yêu cầu pilot hiện tại đã chứng nhận lower gap; chỉ top/runner quan
trọng được xét, và return-impact trimming chỉ được dùng trong phần epsilon còn
lại. Trường hợp không cần phân biệt action trong epsilon được ghi riêng là
`epsilon_trivial` với requirement 0, không còn giả dạng identified tại sàn 1.

Trên 225 pilot, 95 depth rows có requirement không tầm thường và cả 95 giá trị
đều khác nhau; không có giá trị sàn 1. Trên factorial 810 cell cùng random
streams với v1, 254 cell được identified, cả 254 requirement đều khác nhau;
resolved rate tăng từ `0,0037` ở budget 2 lên `0,2185` ở budget 10 và `0,7185`
ở budget 50. Audit cơ học/implementation đạt 13/13 checks. Chi tiết tại
[OEH_V2_FINAL_AUDIT_2026-09-20.md](research/OEH_V2_FINAL_AUDIT_2026-09-20.md).

Kết quả utility vẫn âm. B3 không cải thiện B2 và B3 selector đạt `0,5651`, thấp
hơn best-fixed `0,6803`; difference `-0,1151`, 95% MDP-cluster CI
`[-0,1901; -0,0442]`. Vì vậy hướng cuối giữ OEH v2 như một diagnostic/proxy đã
thoát bão hòa cơ học, nhưng không tuyên bố nó là learner selector đã được xác
nhận. Không tiếp tục chỉnh trên cùng 75 groups để ép gate dương.

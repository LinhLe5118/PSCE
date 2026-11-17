# Audit toàn repo và mức hoàn thành mục tiêu nghiên cứu — 19/09/2026

## Kết luận kiểm toán cập nhật

**Mục tiêu nghiên cứu hiện hành chưa hoàn thành, nhưng ba khoảng trống kỹ thuật
lớn của audit trước đã được đóng:** cross-fitted EB-OEH, B0–B3 grouped nested
out-of-fold evaluation và learner-selection utility đã được implement, test và
chạy đủ trên development. Kết quả development là âm: B2 không cải thiện B1 ổn
định, B3/OEH không bổ sung thông tin có ý nghĩa, và selector kém best-fixed.

Điều này không phải lỗi chạy. Nó là kết quả đúng của protocol đã khóa trước run.
Vì development gate không đạt, evaluation split 120 MDP tiếp tục được giữ niêm
phong. Không được mở evaluation hoặc chỉnh feature/threshold trên cùng cohort rồi
gọi lần chạy sau là confirmatory.

## 1. Những phần đã sửa sau audit

### 1.1 Cross-fitted finite-sample OEH proxy

`code/oeh_crossfit.py` dùng hai trajectory folds. Model fold khóa empirical
model, QVI policy và top/runner action pair; evidence fold độc lập đo
Bellman-target gap/variance và coverage. Gap dùng empirical-Bernstein lower
confidence bound, coverage dùng Clopper–Pearson lower bound. Sau khi đổi vai trò
hai folds, estimator chỉ trả finite requirement khi cả hai hướng được nhận dạng.
Action liên quan không được quan sát trả `unsupported`; lower gap không dương trả
`unresolved`.

Artifact:
`artifacts/research/oeh_crossfit_development_v2_2026-09-19/`.

- 225/225 cases, 22.725/22.725 depth rows, 0 lỗi.
- 213 cases `unsupported`, 12 `unresolved`, 0 finite identified case.
- Không còn cơ chế mọi case cùng nhận một scalar finite ở floor/ceiling. Estimator
  abstain đúng khi pilot không nhận dạng được target.
- Đây vẫn là conservative local EB proxy, chưa phải exact alternative-MDP KL
  optimizer và chưa có theorem instance-optimal.

### 1.2 B0–B3 và nested grouped OOF

Protocol development được khóa tại
`docs/research/STOCHASTIC_DEVELOPMENT_AMENDMENT_2026-09-19.md` trước execution.
Code mới:

- `code/stochastic_features.py`: coverage, trajectory quality, conditional
  transition entropy, low-count indicators và trajectory bootstrap uncertainty;
- `code/stochastic_evaluation.py`: standardization trong fold, deterministic
  grouped inner folds, ridge selection và nested OOF hazards;
- `scripts/analysis/run_stochastic_b0_b3.py`: join 75 pilot groups với 225
  learner labels, scoring và learner selection.

Artifact chính:
`artifacts/research/stochastic_b0_b3_oeh_v2_development_final_2026-09-20/`.
Bundle không có hậu tố `v2` bị supersede do lỗi in đường dẫn Unicode sau khi đã
ghi report; không dùng bundle đó để trích dẫn.

Thiết kế đã chạy:

- 15 development MDP × 5 behavior policies = 75 groups;
- pilot x10 độc lập với training streams;
- CFQI, tabular IQL và tabular BC = 225 learner labels;
- 5 outer folds theo base MDP; 4 inner grouped folds;
- ridge grid `{1e-4,1e-3,1e-2,1e-1,1}`;
- 900 OOF predictions; không đọc evaluation split;
- paired cluster bootstrap 10.000 draws theo base MDP, Holm correction trong
  learner panel.

### 1.3 Kết quả predictive ablation

| Learner | B1 log-likelihood | B2 log-likelihood | B3 log-likelihood | B1 Brier | B2 Brier | B3 Brier |
|---|---:|---:|---:|---:|---:|---:|
| CFQI | -1,7379 | -2,1374 | -2,1284 | 0,1266 | 0,1162 | 0,1159 |
| IQL | -1,8600 | -1,6946 | -1,7540 | 0,1008 | 0,1377 | 0,1419 |
| BC | -1,5790 | -1,2927 | -1,3175 | 0,0859 | 0,0983 | 0,1002 |

Không có B2−B1 log-likelihood contrast nào có CI loại 0 sau correction. IQL B2
có Brier **xấu hơn** B1: difference `+0,0369`, CI `[0,0048; 0,0823]`, Holm
`p=0,0240`. B3−B2 không có improvement đáng tin cậy cho learner nào. Vì 67/75
pilot x10 là `unsupported` và 8/75 là `unresolved`, B3 chủ yếu mang trạng thái
abstention; nó không cung cấp finite OEH shape để dự báo.

### 1.4 Kết quả learner selection tại budget 10|S||A|

| Features | Selector return | Best-fixed | Oracle-best | Selector − best-fixed, 95% cluster CI |
|---|---:|---:|---:|---:|
| B0 | 0,5670 | 0,6803 | 0,6956 | -0,1133 `[-0,1739; -0,0549]` |
| B1 | 0,6047 | 0,6803 | 0,6956 | -0,0756 `[-0,1393; -0,0248]` |
| B2 | 0,5899 | 0,6803 | 0,6956 | -0,0904 `[-0,1546; -0,0325]` |
| B3 | 0,5810 | 0,6803 | 0,6956 | -0,0993 `[-0,1621; -0,0419]` |

Best-fixed được chọn chỉ từ outer-training MDPs. Oracle-best chỉ là reference.
Tất cả selector đều kém best-fixed với CI nằm hoàn toàn dưới 0. B1 là selector
tốt nhất trong bốn feature sets nhưng vẫn chưa đạt utility gate.

### 1.5 External author-code replication tương thích

Repository tác giả `RomainLaroche/SPIBB` được pin tại commit
`833f32b642759111907d2c4d086d1ae1ce9d84cb`. Core `spibb.py` và
`modelTransitions.py` không bị sửa; adapter chỉ cấp training prefixes đã kiểm
hash, behavior policy đã đăng ký, empirical reward và scoring chung. Cấu hình
`Pi_b_SPIBB`, `N_wedge=10` được khóa trước full run.

Artifact:
`artifacts/research/final_spibb_development_2026-09-20/` và paired comparison
tại `artifacts/research/final_all_methods_comparison_2026-09-20/`.

- 4.500/4.500 runs, 0 invalid; 19 finite và 56 censored requirements.
- Tại budget x10, mean return SPIBB là `0,4509`.
- SPIBB trừ CFQI: `-0,2294`, CI `[-0,3163; -0,1536]`.
- SPIBB trừ IQL: `-0,0609`, CI `[-0,1028; -0,0218]`.
- SPIBB trừ BC: `+0,1002`, CI `[0,0631; 0,1381]`.

Đây là comparable fixed-configuration author-code replication. Nó không chứng
minh đây là hyperparameter tốt nhất của SPIBB và không được dùng như held-out
algorithm ranking.

### 1.6 External panel và GPU robustness

Panel tương thích đã được mở rộng bằng VI-LCB, discrete BCQ và discrete CQL.
VI-LCB là paper-spec port được ghi nhãn rõ vì repo tác giả chỉ có notebook
finite-horizon; grid hoàn tất 9.000/9.000 runs. BCQ import learner tác giả không
sửa; CQL dùng implementation d3rlpy đã công bố JMLR. Hai job CUDA hoàn tất
75/75 groups ở budget x10, seed 0, 500 updates trên RTX 3050 Ti.

Ở slice ghép cặp chung, mean return lần lượt là CFQI 0,6729, IQL 0,5084, BCQ
0,4929, CQL 0,4746, SPIBB 0,4499, BC 0,3510, VI-LCB cb=.05 0,2559 và VI-LCB
cb=144 0,1018. BCQ/CQL thấp hơn CFQI, cao hơn BC, và không khác IQL sau Holm.
Đây là robustness panel cấu hình cố định, không phải neural SOTA benchmark.

### 1.7 Controlled stochastic factorial

Protocol mới giữ nguyên topology/reward trong sáu block corridor rồi thay độc
lập slip `{0; 0,05; 0,20}`, behavior coverage `{0,05; 0,20; 0,45}`, budget
`{2; 10; 50}|S||A|` và 5 replicate. Artifact 810 cell hoàn tất, evaluation split
không được đọc. Slip cao làm transition entropy tăng 0,1800; budget 50 so với 2
làm coverage tăng 0,2687 và CFQI return tăng 0,2001. OEH vẫn 0 identified, gồm
568 unsupported và 242 unresolved. Như vậy confound thiết kế đã được tách trên
probe này, nhưng predictive gate không được cứu.

## 2. Ma trận hoàn thành hiện tại

| Gate | Trạng thái | Bằng chứng và khoảng trống |
|---|---|---|
| Công thức mục tiêu OEH | **Đạt ở cấp đặc tả** | Information-identification profile `(k,N_k^ID)`, coverage, KL alternatives và abstention |
| OEH output contract | **Đạt** | Profile/status/data selector/EH-reference có API và artifact |
| Cross-fitted EB-OEH proxy | **Đạt về implementation** | 225 cases, 22.725 rows, model/evidence split, 0 lỗi |
| Exact KL solver/theorem | **Chưa làm** | Hiện chỉ có local EB approximation và two-action sanity check |
| Loại finite-scalar saturation | **Đạt** | Không ép unsupported/unresolved case thành finite score |
| B0–B3 grouped nested OOF | **Đạt trên development** | 900 OOF predictions, scaling/tuning trong fold |
| Predictive utility B2/B3 | **Không đạt gate** | B2 không cải thiện B1 ổn định; B3 không cải thiện B2 |
| Learner panel | **Đạt trên development cũ** | CFQI/IQL/BC và BC sample-complexity labels đầy đủ |
| Learner-selection utility | **Không đạt gate** | Cả B0–B3 selector kém best-fixed với CI dưới 0 |
| Controlled stochastic factorial | **Đạt trên development probe** | 810 cell, 6 matched blocks; slip/coverage/budget tách độc lập, 0 evaluation access |
| External compatible replication | **Đạt cho panel cấu hình cố định** | SPIBB 4.500, VI-LCB 9.000, BCQ/CQL 75 fits mỗi phương pháp; cùng data/scoring ở paired slice |
| Confirmatory evaluation | **Không mở, đúng quyết định** | Development gate âm; 120 MDP vẫn sealed |
| Paper hướng mới | **Đã tích hợp kết quả, còn biên tập** | Cross-fit/B0–B3/selection/SPIBB đã vào LaTeX; 18 trang, cần chốt venue và cắt độ dài |
| Luận văn | **Nháp nội dung đủ I–VII** | II–IV/VI đã đồng bộ; còn ghép template, biên tập và figure/table cuối |
| Demo/release | **Đạt ở bản offline** | HTML tự chứa, CLI fallback và manifest SHA-256; chỉ đọc artifact, không train/network |
| Tests | **Đạt** | `711 passed, 323 subtests passed`; `pytest.ini` giới hạn collection đúng `tests/` |

## 3. Claim hiện được phép dùng

1. OEH-v1 bão hòa cơ học trên development và statistic saturation không đồng
   nghĩa learner không học được.
2. Offline OEH hợp lệ phải kết hợp decision margin với behavior information và
   phải trả infinity/abstain khi target không nhận dạng được.
3. CF-EB-OEH loại finite-scalar saturation: trên cohort hiện tại nó phát hiện
   thiếu support/evidence thay vì tạo một con số độ khó giả.
4. Trên 75 development groups hiện có, feature stochastic B2 không cải thiện
   baseline B1 ổn định; OEH B3 không thêm predictive value; selector dựa trên
   các dự báo này kém best-fixed.
5. Đây là development negative result có protocol và provenance, chưa phải bằng
   chứng generalization hay confirmatory claim.

Không được claim `N_hat_k^EB` bằng exact `N_k^ID`, OEH dự báo tốt sample
complexity, selector cải thiện return, generalization sang unseen benchmark, hay
paper đã đạt chuẩn A/A*/Q1.

## 4. Việc còn lại và quyết định go/no-go

Không mở evaluation với implementation hiện tại. Hai hướng khoa học hợp lệ:

1. **Chốt negative-study scope:** paper tập trung failure mechanism,
   identifiability, cross-fitting và negative predictive/selection result; thêm
   controlled factorial để cô lập nguyên nhân và external replication để kiểm
   tra độ khái quát. Không hứa learner-selection gain.
2. **Mở một method-development cycle mới:** đăng ký estimand/model mới và tạo
   development data mới trước khi quan sát kết quả. Không retune trên 75 groups
   hiện tại rồi dùng lại chúng như test.

Khoảng trống kỹ thuật controlled factorial, external panel, demo/release và nháp
luận văn I–VII đã được đóng. Các hạng mục còn bắt buộc để gọi toàn đồ án hoàn
thiện là ghép template luận văn và quyết định scope paper cuối. Paper LaTeX đã
được bổ sung cross-fit, B0–B3, learner-selection và external panel nhưng vẫn cần
biên tập lại độ dài/câu chuyện. Confirmatory chỉ trở
lại lộ trình nếu một protocol mới có development gate đạt trên dữ liệu phát
triển độc lập.

**Engineering gate:** PASS cho cross-fit, feature/evaluation pipeline và artifact.

**Mechanical-saturation gate:** PASS theo nghĩa không còn scalar finite giả;
pilot hiện tại chủ yếu bị từ chối vì không nhận dạng.

**Predictive/decision gate:** FAIL trên development.

**Research-objective gate:** **NO-GO / INCOMPLETE**. Lý do hiện tại là kết quả
khoa học âm và thiếu controlled/external evidence cùng deliverable cuối, không
phải endpoint chưa được code.

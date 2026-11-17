# Kế hoạch thực nghiệm difficulty-aware: đối chứng có kiểm soát và BRIDGE

> **Cập nhật phạm vi 08/10:** [T1 v1.3.0](T1_COMPARATOR_DECISION_v1.3.0_2026-10-08.md) loại BWD khỏi đối chứng hiện tại. Các bảng BWD trong bản thiết kế ngày 07/10 chỉ còn giá trị lịch sử; kết quả đã niêm phong giữ nguyên.

**Ngày:** 07/10/2026. **Đính chính sau khi đọc bài NeurIPS 2023, ICLR 2024 và datasheet BRIDGE:** BRIDGE **đủ tư cách dataset benchmark độc lập do bên thứ ba công bố**; mỗi instance của dataset là một MDP, không phải một buffer trajectory. **Trạng thái thực thi:** metadata preflight; chưa chạy BRIDGE, chưa có kết quả mới. T1 v1.1.0 và kết quả cohort 7 giữ nguyên. Tài liệu này triển khai B1/B2, không thay giao thức đã niêm phong.

**Bổ sung về đối chứng:** [T1 v1.2.0](T1_COMPARATOR_ADDENDUM_v1.2.0_2026-10-07.md) yêu cầu audit phương pháp của tác giả khác trước khi khóa panel BRIDGE. Các hàng B2/ERI/BWD/OEH dưới đây là bảng **đối chứng đặc trưng**, không phải bảng so sánh với hệ thống nguyên bản của tác giả. BRIDGE là nguồn MDP bên thứ ba, còn các log và nhãn offline vẫn do nhóm tạo. Bảng so sánh trực tiếp với nghiên cứu đã công bố chỉ lập khi pipeline ngoài vượt preflight cùng estimand và quyền truy cập; nếu không, công bố NO-GO thay vì gọi các adaptation là phương pháp gốc.

[D1 audit và kế hoạch lấy kết quả đối chứng](D1_ORIGINAL_METHODS_AUDIT_AND_RUN_PLAN_2026-10-07.md) xác định ERI/TQ/SACo nguyên công thức ở bảng score, EH ở bảng oracle và Mateos-Melero là ứng viên chuyển giao có điều kiện. Bảng `Controlled primary` bên dưới là thiết kế cũ cho phép so sánh **feature**, không thay cho bảng tác giả.

## 1. Bài toán và hai tầng bằng chứng

Estimand chính là phân phối mốc dữ liệu đầu tiên `E_l` mà learner `l ∈ {BC, CFQI, IQL}` đạt ngưỡng return, điều kiện theo MDP, behavior `β`, learner và ngưỡng. Input chỉ là pilot `D0`; OEH-H là descriptor, không phải nhãn difficulty hay cận mẫu đã chứng minh. Đánh giá dự báo bằng integrated Brier và seven-class log loss; selector là ứng dụng phụ.

| Tầng | Dữ liệu | Phép so sánh | Claim hợp lệ |
|---|---|---|---|
| A: kiểm soát | Cohort nội bộ đã mở; giữ MDP split, pilot, nhãn, head và tuning tương đương | B2 matched, B2+ERI adaptation, B2+BWD adaptation, OEH đầy đủ; ablation bỏ OEH-B trực tiếp, bỏ OEH-H/risk profile và bỏ tín hiệu bổ sung được định nghĩa | Giá trị bổ sung của khối đặc trưng **trên cohort này**, exploratory nếu chạy sau khi đã xem cohort. Báo cả chênh lệch âm, runtime và dữ liệu censored. |
| B: benchmark độc lập BRIDGE | **Dataset MDP bên thứ ba đã công bố**: MiniGrid gốc, sticky-action 25% theo bài ICLR 2024 | Cùng bài toán offline do nhóm định nghĩa, các forecaster đã khóa và cùng đầu vào/nhãn/learner trong tầng B | Kiểm chứng chuyển dự báo sang MDP thuộc benchmark công bố. Log offline và nhãn BC/CFQI/IQL do nhóm tạo theo giao thức mới trên MDP ấy; **không** gọi đó là buffer hay kết quả nguyên bản của tác giả. |

**Căn cứ phân loại:** [bài NeurIPS 2023, mục 4 và datasheet E.4](https://papers.nips.cc/paper_files/paper/2023/file/b8be628bf719550b560de8bec9456e0b-Paper-Conference.pdf) định nghĩa BRIDGE là 155 MDP và công bố transition/reward, Gym environment cùng phép benchmark PPO/DQN; [bài ICLR 2024, mục 4](https://proceedings.iclr.cc/paper_files/paper/2024/file/3e18f7152b5e1696e73722dc5121b6f2-Paper-Conference.pdf) benchmark PPO/DQN/SQIRL/GORP trên bản sticky của BRIDGE. Vì vậy BRIDGE đáp ứng yêu cầu **dataset benchmark bên thứ ba đã công bố**. Yêu cầu hẹp hơn “trajectory offline có sẵn do bên thứ ba thu” là một loại dữ liệu khác, không phải điều kiện để BRIDGE được gọi là benchmark ngoài.

Không đặt return của SQIRL cạnh Brier của OEH: bài gốc xét learner online và empirical sample complexity, còn nhóm dự báo xác suất nhãn requirement offline. EH/return/SQIRL là đại lượng hoặc thuật toán khác; nếu dùng EH từ MDP thật như predictor tham khảo, phải xếp ở bảng **privileged oracle**, không đưa vào matched pilot-only. ERI/BWD là adaptation của nhóm.

**Cách ghi trong paper:** “We evaluate on the independently published BRIDGE benchmark (MiniGrid subset, sticky-action variant), using our newly specified offline data-collection and difficulty-label protocol.” Bài NeurIPS 2023 dùng MDP BRIDGE để đối chiếu EH bound với sample complexity PPO/DQN qua Spearman correlation, median ratio, AUROC và accuracy; bài ICLR 2024 chạy PPO/DQN/SQIRL/GORP trên sticky BRIDGE. Các số gốc có thể làm **bối cảnh về benchmark**, nhưng không phải score đối chứng trực tiếp cho Brier/log loss BC/CFQI/IQL của nhóm. Muốn so trực tiếp một descriptor EH, phải xác định biến dự báo và cùng nhãn/metric/quyền truy cập trước khi chạy.

## 2. Nguồn BRIDGE và shortlist đã chuẩn bị

Nguồn đã pin: [repo chính thức](https://github.com/cassidylaidlaw/effective-horizon), [README dataset](https://github.com/cassidylaidlaw/effective-horizon/blob/main/data/bridge_dataset/README.md), **[Zenodo v1.1.1, DOI 10.5281/zenodo.12556298](https://zenodo.org/records/12556298)**. Bản v1.1.1 sửa một số NPZ bị hỏng ở v1.1; vì thế không dùng bản record 10966777/v1.1 cho adapter. Archive đúng tên `bridge_dataset_v1.1.1.zip` (28.708.843.161 byte). Metadata `summary.csv` và `summary_sticky.csv` đã tải vào [`benchmarks/bridge/metadata`](../../benchmarks/bridge/metadata); MD5 của **cả hai tệp giống nhau** giữa record v1.1 và v1.1.1 và khớp Zenodo v1.1.1. [`source_manifest.json`](../../benchmarks/bridge/source_manifest.json) ghi nguồn/hash. Đã lấy **16 NPZ candidate** trực tiếp từ ZIP gốc qua HTTP byte range; manifest trong `datasets/bridge_v1_1_1/candidate_download_manifest.json` ghi ZIP CRC/size và SHA-256 mỗi file. Không tải toàn bộ archive 28,7 GB nên MD5 của **toàn ZIP** chưa được kiểm. Mã tác giả đã clone cục bộ ở commit `1a83eb7fe922b5911d7606728455d65f1953a785` để kiểm adapter.

[`prepare_metadata.py`](../../benchmarks/bridge/prepare_metadata.py) chỉ đọc `mdp_name`, `origin`, `type`, số state/action và horizon; nó không dùng return, EH, SQIRL hay kết quả OEH để chọn task. Loại reward-shaped, giữ 33 MiniGrid gốc, rồi sàng lọc trước theo `sticky_states <= 1200`, vì `TabularMDP` hiện lưu **cả P và R dense float64**. [`candidate_registry.csv`](../../benchmarks/bridge/candidate_registry.csv) ghi toàn bộ 33 hàng: **16 candidate theo metadata**, 17 hàng vượt cap. Đây là cap kỹ thuật cho preflight, **chưa** chứng minh 16 task chạy được. Hàng lớn nhất trong candidate có 1.010 state, 6 action và riêng P+R khoảng 93 MiB; x50 tương ứng 303.000 transition *mỗi seed* nếu lấy `|S||A|` sticky làm đơn vị. Tổng chi phí còn nhân với behavior, learner, budget, seed và fit; cần benchmark runtime trước khi GO. Số 16 task cũng cho khoảng tin cậy rộng, nên tầng B thiên về transfer audit, không mặc định là confirmatory gate.

### Điều kiện adapter phải qua trước khi dùng nhãn

1. Tải đúng `consolidated.npz` của mỗi candidate từ Zenodo, ghi SHA-256 từng file, kiểm schema bằng [`preflight_npz.py`](../../benchmarks/bridge/preflight_npz.py). **Đã qua 16/16 schema checks**. Trong metadata `num_states_consolidated` gồm một terminal state, còn NPZ chỉ lưu `S-1` hàng và mã hóa terminal bằng `-1`. Script mới chỉ xác nhận mảng, reward hữu hạn và terminal sentinel; **không** xác nhận tính tương đương động lực.
2. Tạo sticky transition bằng xác suất 0,25 lặp action trước; state phải chứa **action trước**, và cần kiểm thời gian còn lại với horizon 100. Reset đầu episode không được dùng action trước của episode cũ. So khớp sampling/transition/reward/terminal với Gym environment tác giả trên fixture và seed cố định.
3. Kiểm khả năng biểu diễn Markov, horizon và memory. Nếu cần nhân state theo 100 timestep để bảo toàn finite-horizon MDP, dense `(S,A,S)` hiện tại không dùng được; phải có adapter sparse hoặc báo NO-GO, không âm thầm bỏ thời gian khỏi state.
4. Quyết định riêng `gamma` trước fit: BRIDGE sticky MiniGrid trong repo tác giả dùng 0,99, còn cohort nội bộ dùng 0,95. Chọn 0,99 cho tầng B để bám setting tác giả và khóa nó cho **mọi** phương pháp ở tầng B; không gộp score hoặc nhãn của hai tầng. Reward/return normalization và ngưỡng phải được tính lại và đóng băng trước khi mở outcome.
5. Tách seed và file của pilot khỏi training streams chấm nhãn; không dùng MDP thật, reward-optimal công bố hoặc nhãn learner để xây đặc trưng pilot-only. Đo kiểm leakage, support và coverage. Xác nhận learner BC/CFQI/IQL chấp nhận state/action mới, thời gian train và storage trước full run.

## 3. Thứ tự công việc và người nhận theo tuần 8

| Mốc | Chủ trì | Đầu ra/điều kiện nghiệm thu |
|---|---|---|
| B1 metadata (đã chuẩn bị) | Việt Trần phụ trách data; Linh hỗ trợ registry | Hash Zenodo v1.1.1, 33/16/17 task registry, trích dẫn BRIDGE như **benchmark MDP công bố**; ghi rõ log/nhãn mới do nhóm tạo. Việt Trần kiểm độc lập nguồn; chưa ký xác nhận. |
| M1/E1 | Linh; Quốc Việt kiểm metric; Hưng kiểm diễn giải | Runner matched ablation, cùng fold/head/tuning; bảng Brier, log loss, paired MDP difference, CI mô tả, runtime, censoring; nhãn **exploratory**. Chưa chạy trong tài liệu này. |
| B2a preflight | Việt Trần + Quốc Việt; Hưng kiểm Markov/γ; Linh khóa protocol | MDP files/hash, adapter sticky + time, equivalence tests, ước lượng runtime toàn panel, registry cuối trước outcome. Chỉ task đạt preflight mới GO; công bố đầy đủ fail list theo rule đã khóa. |
| B2b freeze | Linh tích hợp; cả nhóm review chéo | File freeze ghi task IDs, β, pilot x10, sáu budgets, 10 seed hay lý do sửa **trước nhãn**, BC/CFQI/IQL, gamma 0,99, normalization, metric, bootstrap theo MDP, stop rule và compute cap. Forecaster không được fit/tune trên labels tầng B. |
| E2 BRIDGE benchmark | Quốc Việt chạy; Việt Trần kiểm data; Linh phân tích; Hưng kiểm claim | Chỉ một lượt sau B2b GO; lưu cả lỗi/NO-GO, manifest, runtime, predictions và nhãn. Gọi đúng là **benchmark trên dataset MDP công bố**, với offline log/label do nhóm thu; không gọi log đó là dữ liệu công bố. |
| Audit buffer offline công bố (bổ sung, không chặn E2 BRIDGE) | Việt Trần | Registry link/version/license/Markov state/khả năng tạo multi-budget label cho dataset trajectory offline bên thứ ba nếu muốn chứng minh thêm trên loại dữ liệu này. |

**Gates dừng:** không có file MDP khớp hash/phiên bản; sticky/time không Markov; compute vượt cap đăng ký; pilot và nhãn dùng chung transitions/seed; thiếu full outcomes cho một learner; hoặc nhìn outcome rồi sửa shortlist. Tất cả đều ghi failure report; không đổi endpoint để tạo kết quả đẹp.

### Phân biệt nguồn MDP với nguồn trajectory offline

| Nguồn | Điều đã xác nhận | Trạng thái đối với requirement bảy lớp |
|---|---|---|
| BRIDGE [Zenodo v1.1.1](https://zenodo.org/records/12556298) | **Đạt tiêu chí dataset benchmark bên thứ ba đã công bố**: MDP tabular, sticky summary và kết quả thuật toán tác giả. Record không cung cấp pilot/training log hoặc đường cong BC/CFQI/IQL cho bài toán của nhóm. | **GO về nguồn benchmark**, còn GO chạy OEH phụ thuộc adapter, pilot/nhãn và freeze. Log mới phải ghi là do nhóm tạo. |
| [Minari D4RL MiniGrid FourRooms](https://minari.farama.org/datasets/D4RL/minigrid/fourrooms-v0/) | Log offline do bên thứ ba phát hành; nhóm đã có phép thử thành phần posterior NO-GO. | Chưa đủ multi-task, budgets/seeds, Markov mapping và label để đánh giá toàn OEH; **không** nâng phép thử thành phần thành full benchmark. |
| [Minari BabyAI GoToObjS4](https://minari.farama.org/datasets/minigrid/BabyAI-GoToObjS4/optimal-v0/) và [danh mục MiniGrid](https://minari.farama.org/datasets/minigrid/index.html) | Có các trajectory offline công bố trên nhiều task; quan sát/action/behavior có thể khác BRIDGE và protocol tabular. | **Cần audit**, chưa gọi FEASIBLE: xem từng version, full-observation/Markov state, task independence, quyền dựng pilot/budget và random/optimal normalization. |

Hai nguồn Minari ở đây là tùy chọn để kiểm tra thêm trên **buffer offline do người khác thu**, không phải điều kiện cần để gọi BRIDGE là benchmark ngoài. Với bất kỳ nguồn nào, checklist adapter và nhãn vẫn phải qua trước khi chạy full OEH.

## 4. Bảng kết quả sẽ tạo, chưa điền số

| Bảng | Hàng | Cột bắt buộc |
|---|---|---|
| Controlled primary | B2 matched, B2+ERI, B2+BWD, OEH | MDP/row count, Brier, log loss, paired Δ với CI, runtime, censoring |
| Ablation | OEH đầy đủ, bỏ OEH-B trực tiếp, bỏ OEH-H/risk, bỏ nhóm bổ sung xác định rõ | Cùng head/fold; Δ so OEH, CI, runtime; không gọi “bỏ posterior” nếu OEH-H vẫn dùng posterior |
| BRIDGE transfer | Cùng bốn forecaster đã khóa, toàn task GO/FAIL | Brier, log loss, paired Δ theo MDP, runtime, số task/seed, fail list |
| Oracle phụ | EH hoặc feature dùng MDP thật | Nêu quyền truy cập đặc biệt; không tính vào matched ranking |

**Đối chiếu paper:** phần DATA mô tả riêng cohort tự sinh, **BRIDGE benchmark công bố** và giao thức thu log offline mới của nhóm trên BRIDGE; Method nêu quyền truy cập của từng đối chứng; Results tách bảng A/B; Discussion nêu gamma, task count, resource exclusions và mọi failure. Không viết kết luận “vượt SOTA” từ preflight metadata.

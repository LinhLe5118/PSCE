# Runbook trên máy này: thực nghiệm difficulty-aware offline RL

> **Trạng thái vận hành 09/10:** Các bước chuẩn bị ngày 07/10 bên dưới là lịch sử, không còn là trạng thái hiện tại. Cohort nội bộ 120 MDP và BRIDGE Atari 5 MDP đã hoàn tất. BRIDGE mở rộng gồm 5 MDP mới với nhãn chưa mở và 14 MiniGrid đã mở nhãn đang chạy theo checkpoint. Xem [ledger bằng chứng hiện hành](CURRENT_EVIDENCE_AND_RETIREMENT_2026-10-09.md), [nhật ký BRIDGE mở rộng](BRIDGE_EXPANDED_EXECUTION_2026-10-08.md) và chạy `powershell -NoProfile -ExecutionPolicy Bypass -File Devoted/benchmarks/bridge/show_progress.ps1` từ `D:\Đồ án` để xem số nhóm mới nhất. Sau khi máy ngủ/tắt, dùng `Devoted/benchmarks/bridge/resume_checkpointed_runs.ps1 -DryRun -WithSmallMiniGrid` để kiểm tra, rồi bỏ `-DryRun` để tiếp tục. Helper chỉ khởi động các job còn thiếu và từ chối trạng thái `NO_GO`. Chỉ đưa số BRIDGE mới vào paper/thesis khi manifest nhãn, manifest phân tích và audit metric chung đều hoàn tất.

> **Cập nhật phạm vi 08/10:** [T1 v1.3.0](T1_COMPARATOR_DECISION_v1.3.0_2026-10-08.md) loại BWD khỏi đối chứng đang sử dụng. Các hàng BWD còn lại trong runbook ngày 07/10 chỉ là kế hoạch lịch sử; artifact đã niêm phong không thay đổi.

**Lập ngày 07/10/2026. Trạng thái: đã lấy MDP nguồn, chưa bắt đầu lượt chạy khoa học mới.** Tài liệu này dùng cùng định nghĩa difficulty, nhãn và quyền truy cập của [T1 v1.1.0](T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md). [Kế hoạch BRIDGE](B1_B2_DIFFICULTY_AWARE_EXTERNAL_BENCHMARK_2026-10-07.md) quy định tiêu chí benchmark; runbook này quy định thứ tự thao tác và cổng kiểm tra trên máy `D:\Đồ án`. Không sửa hoặc chạy đè artifact cohort 120 MDP đã niêm phong ngày 04/10.

**Đính chính đối chứng trước lượt chạy:** [T1 v1.2.0](T1_COMPARATOR_ADDENDUM_v1.2.0_2026-10-07.md) là gate bổ sung cho benchmark mới. Lượt A dưới đây chỉ là đối chứng đặc trưng/ablation; B2, ERI và BWD không đại diện cho phương pháp nguyên bản của tác giả khác. Lượt B chỉ có hàng so sánh trực tiếp với tác giả khi một pipeline công bố vượt preflight cùng đầu vào, nhãn và metric. Cần tính thêm thời gian audit/tái lập phương pháp ngoại nghiên cứu vào ETA; ước lượng thời gian hiện tại chỉ bao phủ phần OEH/learner đã đo.

## 1. Tài nguyên và giới hạn thực tế

| Hạng mục | Đã đo trên máy này | Quyết định vận hành |
|---|---:|---|
| CPU | Intel i7-12700H, 14 lõi, 20 luồng | Bắt đầu smoke với 1 worker; thử 2 rồi 4 chỉ sau khi đo RAM/thời gian. Không lấy mặc định 16 worker của runner cũ. |
| RAM | 23,7 GiB tổng, 10,4 GiB còn trống lúc đo | Đo đỉnh RAM khi giữ một MDP, posterior và learner cùng lúc; tránh hai job lớn nếu tổng đỉnh vượt RAM khả dụng. |
| C: | 4,9 GiB trống lúc đo | Không đặt archive, giải nén, cache hay `TEMP` của lượt BRIDGE trên C:. |
| D: | 111,1 GiB trống lúc đo | Đặt dữ liệu/output mới trên D:. Đã lấy trực tiếp 16 MDP từ archive 28.708.843.161 byte bằng byte range; không cần tải trọn ZIP để chạy panel đã chọn. |
| Dữ liệu cũ | Cohort 120 MDP đã có 108.000 core jobs và 16.800 selector jobs | Dùng feature/nhãn đã lưu cho ablation khám phá; không lặp lại các job này. Không gọi chúng là kiểm định độc lập mới. |

Các con số trống ổ đĩa/RAM là ảnh chụp tại thời điểm lập runbook; kiểm lại ngay trước khi chạy. Kích thước bộ nhớ và thời gian của sticky BRIDGE **chưa đo**, do đó chưa cam kết chạy xong toàn bộ trong một đêm. Nếu giữ 16 candidate, 5 behavior, 3 learner, 6 budget, 10 seed thì riêng đánh giá learner có **14.400 tổ hợp**; con số này là kế hoạch tối đa, chưa là lệnh GO. MDP lớn nhất trong shortlist có 1.010 sticky state và 303.000 transition ở `x50` cho mỗi seed. Việc thêm thời gian còn lại vào state có thể khiến cấu trúc dense hiện tại không dùng được.

## 2. Hai lượt kết quả cần lấy

| Lượt | Input và phương pháp | Output phải có | Trạng thái hiện tại |
|---|---|---|---|
| A — đối chứng/ablation có kiểm soát | Các cohort đã mở; B2 matched, B2+ERI adaptation, B2+BWD adaptation, OEH đầy đủ, OEH bỏ OEH-B trực tiếp, OEH bỏ OEH-H/risk, OEH bỏ diagnostics bổ sung đã định nghĩa. Cùng `matched_forecast`, fold theo MDP, head và tuning. | Brier tích lũy, seven-class log loss, chênh lệch ghép cặp theo MDP, CI **mô tả**, runtime, tỷ lệ `>x50`, số MDP/hàng và cả kết quả âm. Gắn nhãn **exploratory**. | Có feature/nhãn và hàm dự báo, **chưa có runner ablation M1**. |
| B — chuyển sang BRIDGE | MDP BRIDGE v1.1.1 công bố bởi bên thứ ba; sticky action 25%; offline pilot/log và nhãn BC/CFQI/IQL do nhóm tạo theo giao thức đã khóa. Dùng bốn forecaster matched đã freeze từ A; các ablation chỉ nếu đã freeze trước outcome. | Cùng metric, paired MDP CI, runtime, mọi task FAIL/GO, manifest và hash; bảng BRIDGE riêng. | Đã có 16 NPZ gốc qua schema preflight và mã tác giả; **chưa có adapter, log, nhãn hoặc kết quả**. Không kiểm MD5 toàn ZIP vì chỉ tải member cần thiết. |

BRIDGE là benchmark **MDP công bố sẵn**. Log offline/nhãn của bài toán nhóm chưa được tác giả BRIDGE công bố. Không đặt return SQIRL/GORP của paper gốc cạnh Brier của OEH như cùng endpoint.

## 3. Thứ tự chuẩn bị trước lúc treo máy

**Gate 0 — trước cả bước 1:** thực hiện [D1 audit và kế hoạch đối chứng](D1_ORIGINAL_METHODS_AUDIT_AND_RUN_PLAN_2026-10-07.md): ma trận tác giả/input/output/mã, ERI nguyên bản, reference TQ/SACo, EH oracle và preflight Mateos-Melero trên development task. Chốt GO/NO-GO từng hàng theo [addendum v1.2.0](T1_COMPARATOR_ADDENDUM_v1.2.0_2026-10-07.md). Chỉ hàng GO mới vào manifest/predictions trước outcome BRIDGE; nếu CNN cần BRIDGE task để fit, loại các task đó khỏi test độc lập. NO-GO không cản ablation nội bộ, nhưng cản claim “vượt phương pháp tác giả khác”.

1. **Chốt snapshot.** Ghi commit/source hash, môi trường Python, manifest dữ liệu và tên thư mục output mới. Giữ nguyên `artifacts/oeh_confirmatory_unseen_120_2026-10-04/`. Dự kiến `artifacts/week8/ablation_exploratory_<run_id>/` và `artifacts/week8/bridge_transfer_<run_id>/`; kết quả lớn phải nằm trong `datasets/` hoặc đường dẫn local được ignore trước khi chạy.
2. **Hoàn thành M1 rồi chạy E1.** Lập danh sách 108 cột của `plus_prior_profile`, gán từng cột cho OEH-B trực tiếp, OEH-H/risk hay diagnostics bổ sung; lưu dependency để không gọi nhầm phép bỏ OEH-B là “bỏ toàn bộ posterior”. Runner mới chỉ đọc artifact cũ và `matched_forecast`; kiểm số hàng, 7 xác suất cộng thành 1, cùng fold/fit trên mọi arm và không dùng outcome trong feature. Sau test, chạy từng arm với log riêng, checkpoint sau arm và xuất bảng A. Không thay runner lịch sử để ghi đè verdict.
3. **Kiểm tra nguồn BRIDGE trên D: (đã thực hiện):**

   ```powershell
   cd 'D:\Đồ án\Devoted'
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\benchmarks\bridge\fetch_archive.ps1 -CheckOnly
   & 'D:\anaconda3\python.exe' .\benchmarks\bridge\prepare_metadata.py
   ```

   `prepare_metadata.py` giữ 33 MiniGrid gốc, 16 candidate, 17 loại theo cap, không dùng kết quả OEH. Đã chạy 07/10 và kiểm cả 16 NPZ: metadata đếm thêm một terminal state, NPZ dùng sentinel `-1` nên số hàng thật ít hơn một. Manifest cục bộ: `datasets/bridge_v1_1_1/candidate_download_manifest.json`.
4. **Khi cần tải lại MDP đã chọn:**

   ```powershell
   cd 'D:\Đồ án\Devoted'
   & 'D:\anaconda3\python.exe' .\benchmarks\bridge\fetch_candidates.py
   ```

   Script dùng HTTP 206 lấy chỉ mục ZIP64 và đúng 16 `consolidated.npz`, kiểm độ dài, CRC32 trong ZIP và SHA-256 từng file. Nó không kiểm MD5 toàn archive; nếu cần lưu trọn ZIP để audit riêng, dùng `fetch_archive.ps1` (28,7 GB). Không bung toàn bộ archive ra C:. Trạng thái này mới là kiểm **schema**, chưa phải GO benchmark.
5. **B2a: adapter và smoke trên 1–2 MDP.** So mẫu transition/reward/terminal với Gym BRIDGE tác giả, reset previous action đúng mỗi episode; kiểm sticky 0,25 và horizon 100; quyết định biểu diễn Markov có thời gian hay adapter sparse nếu cần. Khóa gamma 0,99 cho tầng B, kiểm normalization/threshold, BC/CFQI/IQL. Chạy pilot, một budget nhỏ rồi `x50` trên một task để đo RAM, thời gian/job, dung lượng output và độ đúng nhãn. Smoke dùng task phát triển được ghi rõ; không nhìn outcome để chọn panel xác nhận.
6. **B2b: freeze trước khi tính nhãn BRIDGE.** Ghi task GO/FAIL theo tiêu chí kỹ thuật, behavior, seed namespace tách pilot và training, budget `x1,x2,x5,x10,x20,x50`, 10 seed hoặc thay đổi đã quyết định trước outcome, quy tắc `>=0,95` trên `>=8/10` seed ở budget đó và mọi budget lớn hơn, `>x50`, forecaster/weight, metric, bootstrap theo MDP, resource/stop rule, mã và hash. Fit/tune forecaster **chỉ trên meta-training cũ**. Tạo và lưu `predictions_preoutcome` trước khi mở kết quả nhãn BRIDGE.
7. **E2: chạy BRIDGE một lượt sau GO.** Chia checkpoint theo MDP × behavior × learner × budget × seed; resume bằng xác minh input hash, không bỏ qua job lỗi. Chạy 1 worker trước, tăng lên 2/4 nếu đo đỉnh RAM và tốc độ cho phép. Thu tất cả lỗi và runtime. Sau khi hoàn tất nhãn mới chấm Brier/log loss, paired difference và CI; không tuning theo score BRIDGE rồi trình bày cùng cohort là kiểm định độc lập.

## 4. Điều kiện GO/STOP và bàn giao

| Mốc | GO khi | STOP/ghi nhận khi |
|---|---|---|
| E1 ablation | Feature lineage, arm definitions, fold/head và test leakage xong | Thiếu cột, nhãn/feature lẫn nhau, arm đổi kiến trúc mà không ghi rõ |
| Nguồn BRIDGE | 16 file NPZ từ URL v1.1.1 có ZIP CRC/size, SHA-256 và schema pass; mã tác giả pin commit | HTTP range/hash sai, thiếu file hoặc không đọc được schema |
| B2a adapter | State Markov, sticky/reward/terminal/horizon khớp bản tác giả; smoke đủ learner/budget | Dense không chịu được time state; không được âm thầm bỏ time/action trước |
| B2b freeze | Panel và protocol ký nội bộ trước outcome, pilot và stream nhãn độc lập; source hash lưu | Đã xem outcome rồi mới đổi task hoặc endpoint |
| E2 full | Có checkpoint, thời gian và RAM đo từ smoke; dự báo preoutcome đã niêm phong | Không đủ ổ đĩa/RAM, job thiếu hoặc sai hash; lưu failure report nguyên trạng |

**Bàn giao sau lượt chạy:** một manifest máy/source/dataset, log và checkpoint, bảng primary và ablation tầng A, bảng transfer tầng B (nếu GO), file predictions trước outcome, file labels, bảng runtime, danh sách task/job FAIL và phần diễn giải giới hạn. Nếu B chưa qua adapter/freeze, chỉ bàn giao kết quả A và báo B đang chờ; không điền số BRIDGE vào paper. Selector bảy learner là kết quả phụ, không dùng để thay bảng dự báo difficulty.

[Báo cáo dữ liệu đã tải và ETA](LOCAL_EXPERIMENT_ETA_2026-10-07.md) tách thời gian đã đo khỏi phần dự trù trước smoke.

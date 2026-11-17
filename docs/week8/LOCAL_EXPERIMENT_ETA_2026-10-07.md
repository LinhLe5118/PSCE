# Dữ liệu BRIDGE đã lấy và dự toán thời gian thực nghiệm

**Ngày 07/10/2026.** Đây là báo cáo tài nguyên và thời gian, **không phải kết quả OEH trên BRIDGE**. Giao thức khoa học vẫn theo [T1 v1.1.0](T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) và [runbook](RUNBOOK_LOCAL_EXPERIMENTS_2026-10-07.md).

**Phạm vi ETA:** các phép đo và dự toán bên dưới chỉ bao phủ pipeline OEH/learner/ablation. [T1 v1.2.0](T1_COMPARATOR_ADDENDUM_v1.2.0_2026-10-07.md) bổ sung audit, tái lập và kiểm định đối chứng từ tác giả khác. Chưa có runtime smoke hay đủ điều kiện khả thi cho pipeline Mateos-Melero, nên **không cộng một số giờ giả định** cho đối chứng này và không coi ETA bên dưới là thời hạn hoàn thành toàn bộ bảng so sánh với tác giả khác.

[D1 audit/plan](D1_ORIGINAL_METHODS_AUDIT_AND_RUN_PLAN_2026-10-07.md) ước lượng riêng 0,5–1 ngày audit/fixture và 1–3 ngày preflight mô hình ảnh, **chưa gồm** tạo nhãn development task, huấn luyện CNN hoặc online reference TQ/SACo. Các số này là dự trù công việc, không phải runtime đã đo.

## Dữ liệu đã có trên máy

- Nguồn: [BRIDGE Zenodo v1.1.1](https://zenodo.org/records/12556298), archive `bridge_dataset_v1.1.1.zip` dài **28.708.843.161 byte**. HTTP 206 cho phép lấy trực tiếp 16 `consolidated.npz` đã chọn từ chỉ mục ZIP64. Tổng 16 tệp NPZ đã giải nén là **2.554.563 byte**; ZIP đầy đủ không được tải vì tốc độ thử khoảng 0,3 MB/s sẽ mất hơn một ngày và không cần cho panel này.
- Vị trí: `datasets/bridge_v1_1_1/mdps/<MDP>/consolidated.npz`; danh sách, URL, CRC32, kích thước, SHA-256 mỗi file trong `datasets/bridge_v1_1_1/candidate_download_manifest.json`. Kiểm `Content-Range` đúng tổng byte của archive và 16/16 member qua CRC32/kích thước, 16/16 qua preflight schema. **Chưa kiểm MD5 toàn ZIP**; không diễn đạt các kiểm tra thành xác minh MD5 toàn archive.
- Mã tác giả để đối chiếu environment: `datasets/bridge_v1_1_1/official_source/` tại commit `1a83eb7fe922b5911d7606728455d65f1953a785`. Wrapper chính thức reset `last_action=None`, lấy action trước với xác suất 0,25 và cập nhật bằng **action đã thực thi**. Julia evaluator của tác giả không áp dụng sticky ở timestep đầu. Adapter cần khớp các chi tiết đó.
- Phát hiện schema: `summary.csv` báo số state gồm một terminal state; NPZ chỉ có `S-1` hàng transition/reward, terminal successor là `-1`. Preflight/test đã sửa theo 16 tệp gốc (4 test pass). Đây mới là kiểm cấu trúc, chưa là adapter tương đương.

## Khối lượng theo panel đang dự kiến

Với **16 MDP × 5 behavior × 3 learner × 6 budget × 10 seed = 14.400 fit**. Tổng `|S||A|` của 16 sticky candidate là 46.119; nếu mỗi seed/behavior lưu một stream tới `x50`, số transition training tối thiểu là `46.119 × 5 × 10 × 50 = 115.297.500`. Pilot `x10` một seed/behavior cần `46.119 × 5 × 10 = 2.305.950` transition. Nếu tạo riêng cả sáu prefix, lượng đọc/tạo tương đương 202.923.600 transition; nên tái dùng prefix của cùng stream và giữ seed pilot riêng. Các con số này là **khối lượng công việc**, chưa phải bằng chứng learner chạy hợp lệ trên BRIDGE.

| Thành phần | Phép đo trên máy i7-12700H, 23,7 GiB RAM | Cách dùng |
|---|---:|---|
| B2 matched hiện có | 9.000 hàng meta-training + 1.800 hàng query: nạp 60,9 giây, fit/dự báo 24,4 giây, tổng 85,4 giây | 7 arm cùng cỡ vào khoảng vài phút compute sau nạp; OEH nhiều cột hơn nên **không** nhân chính xác 24,4 cho mọi arm. Chưa gồm viết runner M1/phân tích. |
| BC/CFQI/IQL dense, synthetic | Đo 16 kích thước × 6 budget: tổng fit một seed và một behavior trên 16 MDP là 210,8 giây; nhân 50 stream = **2,93 giờ CPU** (BC 0,14; CFQI 0,65; IQL 2,14) | **Mốc chi phí fit của mã hiện tại**, không phải ETA full run. Probe dùng transition tổng hợp; sticky/time representation cuối có thể đổi thuật toán/bộ nhớ và thời gian. |
| Vòng lặp rollout sticky tối giản | 1.000.000 bước trên một NPZ gốc: 3,045 giây, khoảng 328.396 bước/giây; 115,3 triệu bước tương ứng 5,85 phút **nếu** cùng throughput | Chỉ là mốc thấp: random behavior, chưa mã hóa state Markov đầy đủ, chưa ghi đĩa, chưa tạo feature hay kiểm đúng Gym. |
| Scorer dense hiện tại | Policy trên MDP synthetic: 0,001 giây (`S=97`), 0,154 (`S=485`), 0,241 (`S=1010`) | Scorer hiện giải hệ vô hạn horizon; **không dùng nguyên số này** để khẳng định thời gian hay giá trị của đánh giá BRIDGE horizon 100. |
| OEH-B lịch sử | 600 nhóm cohort cũ: 2,51 giờ CPU, median 10,7 giây/nhóm, p90 32,3 giây | Chỉ tham khảo. BRIDGE có kích thước/hình học khác; phải smoke OEH-B trên một MDP nhỏ và một MDP lớn. |

## ETA có điều kiện

| Mốc | Dự trù để xếp lịch | Cơ sở/điều kiện |
|---|---:|---|
| M1/E1 trên cohort đã mở | **0,5–1 ngày làm việc** để viết/kiểm runner và phân tích; phần fit thuần khoảng **5–20 phút** | Runner ablation chưa có. Các arm phải cùng fold/head/tuning, feature lineage và nhãn exploratory. |
| Adapter BRIDGE, scorer horizon 100, test tương đương và smoke | **1–3 ngày làm việc** | Đây là **ước lượng công phát triển**, không suy ra từ stopwatch. Phải quyết định xử lý timeout: state có thời gian, hoặc tách truncation ngoài mô hình và chứng minh scorer finite horizon phù hợp. Dense `(S,A,S)` không thể mở rộng tùy ý theo thời gian. |
| Freeze và full BRIDGE sau smoke GO | Đặt **một đến vài phiên chạy qua đêm (8–36 giờ wall)** để dự phòng; cập nhật ETA bằng `T_smoke` của task nhỏ/lớn trước khi GO | Đã đo **2,93 giờ CPU fit** cho representation cỡ sticky metadata, nhưng chưa đo scorer finite horizon, 80 nhóm OEH, 115 triệu rollout chính thức, I/O/checkpoint và giới hạn RAM. 8–36 giờ là **slot kế hoạch**, không phải thời gian đã chứng minh. Bắt đầu 1 worker, thử 2 sau khi đo đỉnh RAM. |
| Hoàn tất cả hai bảng và audit | **Khoảng 2–5 ngày làm việc + phiên chạy qua đêm**, nếu adapter/smoke đạt | Nếu representation Markov buộc viết learner sparse hoặc task không đủ coverage/nhãn, phải cập nhật phạm vi và ETA trước khi chạy full; không biến fail thành kết quả thắng. |

**Công thức cập nhật ETA sau smoke:** `T_full ≈ 5 behaviors × 10 seeds × Σ_{MDP,budget}(T_collect_prefix + T_fit_BC + T_fit_CFQI + T_fit_IQL + 3 T_score) + 5 × Σ_MDP(T_pilot_features + T_OEH_B + T_OEH_H) + T_forecast + T_validation`. Khi dùng prefix chung, `T_collect_prefix` chỉ tính phần mới của stream. Ghi wall time, CPU time, RAM đỉnh và dung lượng cho MDP nhỏ nhất/lớn nhất; áp hệ số dư địa cho lỗi và thermal khi lập lệnh full. Đến lúc này, **không có cơ sở cho một ETA giờ kết thúc chính xác** của toàn bộ kết quả BRIDGE.

## Cập nhật ETA theo checkpoint thực tế, 11/10/2026 22:00

Dự toán 8–36 giờ ở bảng kế hoạch ngày 07/10 **đã lỗi thời** đối với panel BRIDGE mở rộng. Tại thời điểm này, 25/25 pilot và OEH-H, 7/25 OEH-B đã có checkpoint; nhãn kiểm định mới vẫn 0/25. Sáu nhóm OEH-B đã trả đủ 64 draw mất 2,36–7,13 giờ/nhóm, trung vị 3,90 giờ và tổng 25,72 giờ chạy nhóm. Nhóm thứ bảy trả 0/64 draw trong 17 giây do bộ lọc dynamic range; không đưa trường hợp đó vào tốc độ điển hình. Một worker tính toán tại một thời điểm, với khoảng phục hồi checkpoint sau mỗi lần tiến trình cha thoát.

| Mốc còn lại | Ước lượng để xếp lịch, nếu máy chạy liên tục | Cơ sở và bất định |
|---|---:|---|
| 18 nhóm OEH-B mới còn lại | Khoảng **50–100 giờ**; trung vị sáu nhóm đã đo nhân 18 là **70,2 giờ** | Các MDP còn lại khác kích thước, action count, horizon và khả năng bị lọc draw; ngoại suy chưa phải cam kết. Nhóm đang chạy chưa có checkpoint không được tính là hoàn thành. |
| Seal, 25 nhóm nhãn, score panel mới | **Vài giờ đến khoảng một ngày** sau OEH-B | Chưa có stopwatch nhãn trên năm MDP mới; nhóm nhãn Atari nhỏ hơn không đủ để suy ra chính xác. Không mở nhãn trước seal. |
| 20 nhóm MiniGrid đã mở và phân tích | **Khoảng 1–3 ngày** sau panel mới theo queue hiện hành | Bốn MDP còn lại lớn hơn các nhóm đã hoàn tất; đây là exploratory, không thay kiểm định prospective. |
| Strict common-metric audit và build paper/thesis | **Vài giờ** nếu tất cả gate đạt | Có thể chuyển `REVIEW_REQUIRED` nếu hash, metric, TeX hoặc build DOCX lỗi; không tự bỏ qua lỗi để giữ lịch. |

Tổng dự trù thận trọng là **khoảng 4–8 ngày lịch kể từ tối 11/10**, tức khoảng **15–19/10/2026**, với điều kiện máy không bị tắt/ngủ chủ động và không có NO-GO. Mốc này có thể trượt nếu nhóm còn lại chậm hơn mẫu đã đo hoặc phát sinh lỗi cần review. Bộ giám sát, checkpoint và yêu cầu chống idle sleep giảm thời gian chết; chúng không bảo đảm chạy qua một lần tắt máy chủ động. Cập nhật lại ETA khi OEH-B đạt 12/25 và khi nhãn của MDP đầu tiên có stopwatch thực tế.

**12/10, 00:18 — phép thử tăng song song đang chờ checkpoint:** OEH-B đạt 8/25; supervisor đã được cấu hình hai worker cho lần phục hồi tiếp theo, còn watcher đã kiểm tra được worker cũ sau khi nâng cấp. Nhóm hiện hành vẫn chạy một worker để không mất tính toán dở. Nếu hai worker ổn định và RAM còn đủ, phần OEH-B còn lại có thể giảm thời gian thực tế, nhưng chưa có hai checkpoint chạy đồng thời để đo hệ số tăng tốc. Vì vậy **chưa rút ngắn mốc 4–8 ngày** trên cơ sở cấu hình đơn thuần; cập nhật bằng throughput và RAM quan sát được sau khi hai worker thực sự chạy.

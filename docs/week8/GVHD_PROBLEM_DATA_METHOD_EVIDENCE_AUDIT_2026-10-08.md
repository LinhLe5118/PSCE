# Rà soát góp ý GVHD: từ bài toán đến bằng chứng (08/10/2026)

## Kết luận kiểm tra

Phân tích của trợ lý AI nhìn chung đúng với [T1 v1.1.0](T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) và phần đầu [paper](../../paper/main.tex). Cần làm rõ hơn khi **trình bày**, không cần chạy lại 120 MDP để sửa cách đặt vấn đề. Riêng claim chuyển sang họ môi trường ngoài phải chờ lượt **full OEH trên BRIDGE mới** hoàn tất; kết quả BRIDGE 14 task trước chỉ là ablation giảm đặc trưng, BC còn kém B2. Không dùng lại tập đã mở nhãn để gọi là kiểm định độc lập mới.

| Câu hỏi của thầy | Nguồn đã có | Cách nói và việc còn thiếu |
|---|---|---|
| Bài toán và output | [T1](T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md), [paper §Problem/Data](../../paper/sections/03_data.tex) | Input là pilot `D0`, learner `l`, behavior `β` và ngưỡng `τ=0,95`; output là `P(E_l=c \mid D0,l)` trên `x1,x2,x5,x10,x20,x50,>x50`. `E_l` là **mốc đầu tiên trên lưới đã đo** mà ≥8/10 seed đạt ngưỡng ở mốc đó **và mọi mốc lớn hơn**, không phải số transition tối thiểu liên tục hay cận mẫu lý thuyết. `>x50` là kiểm duyệt phải. |
| Difficulty nằm ở đâu | [method](../../paper/sections/03_difficulty.tex), [nhãn](../../artifacts/oeh_confirmatory_unseen_120_2026-10-04/labels.csv) | Độ phủ action/state, biến động chuyển trạng thái và learner cùng quyết định nhãn. Minh họa cùng MDP và behavior bằng hàng nhãn dưới đây; không gọi ví dụ được chọn sau khi mở nhãn là kiểm định độc lập. |
| Data tự sinh vì sao | [paper §Data](../../paper/sections/03_data.tex), [thesis §4](../../../Drive/Thesis/04_Experimental_Design_2026-09-19.md), [manifest](../../artifacts/oeh_confirmatory_unseen_120_2026-10-04/dataset_seal.json) | Cần pilot tách stream nhãn, sáu budget lồng nhau, 10 seed/learner và `J_random,J*` để chấm đúng nhãn. Có dataset và MDP stochastic bên ngoài; nhóm tự sinh **log và nhãn phù hợp endpoint**, không vì dữ liệu stochastic không tồn tại. 120 kernel mới vẫn thuộc họ generator đã dùng. BRIDGE có MDP do bên thứ ba công bố, nhưng log/nhãn `E_l` do nhóm thu. |
| Vì sao chạy learner offline | [protocol](../../artifacts/oeh_confirmatory_unseen_120_2026-10-04/protocol.json), [outcome](../../artifacts/oeh_confirmatory_unseen_120_2026-10-04/all_core_budget_seed_outcomes.csv) | BC, CFQI, IQL là **hệ tạo nhãn và đối tượng dự báo**. Chạy ở sáu budget × 10 seed để quan sát requirement thực nghiệm; OEH không thay thế việc đo nhãn khi đánh giá. Ba learner là lựa chọn thực nghiệm đã đăng ký, không đại diện toàn bộ offline RL. |
| Tại sao có bảy learner | [paper §Selection](../../paper/sections/04_routing.tex) | Chỉ BC/CFQI/IQL có target requirement chính. BCQ/CQL/SPIBB/VI-LCB được thêm ở **ứng dụng phụ chọn learner tại x10**. Selector có thêm nhãn họ MDP; SPIBB dùng behavior đã biết, nên quyền truy cập khác với dự báo pilot-only. Không dùng bảy hàng return để nói OEH dự báo difficulty cho bảy learner. |
| Metric và đối chứng | [paper §Method](../../paper/sections/03_difficulty.tex), [results](../../paper/sections/06_results.tex), [T1 v1.3](T1_COMPARATOR_DECISION_v1.3.0_2026-10-08.md) | Integrated Brier trên sáu CDF và log loss bảy lớp đo xác suất dự báo; normalized return/success đo policy/selector. B2 là đối chứng **nội bộ**, ERI là tín hiệu tác giả được nhóm thích nghi dưới cùng head; TQ/SACo có reference MDP đặc quyền. Không nói bảng Brier chứng minh vượt SOTA nguyên bản. BWD không ở bảng đối chứng hiện tại. |

### Một ví dụ learner-specific có thể kiểm tra

Trong [`labels.csv`](../../artifacts/oeh_confirmatory_unseen_120_2026-10-04/labels.csv), nhóm `oehfinal120_forward_layered_004` + `q_star_softmax` có **cùng MDP, behavior và giao thức log**:

| Learner | Số seed đạt ngưỡng tại `x1/x2/x5/x10/x20/x50` | Lớp `E_l` |
|---|---|---|
| BC | `2/3/2/2/2/1` | `>x50` |
| CFQI | `6/7/8/10/10/10` | `x5` |
| IQL | `4/6/7/10/10/10` | `x10` |

Như vậy câu “CFQI có thể cần ít log hơn IQL” có chứng cứ cụ thể. Đây là **một nhóm được chọn để giải thích**, không phải ước lượng hiệu ứng trung bình. Trên toàn bộ 420 nhóm stochastic, tỷ lệ `>x50` là BC `381/420 = 90,71%`, CFQI `193/420 = 45,95%`, IQL `326/420 = 77,62%`; luôn đặt các tỷ lệ này cạnh bảng Brier để không diễn giải Brier thấp của nhóm bị censor nhiều là dự báo tốt mọi mức budget. SHA-256 của `labels.csv`: `019D3BBA34A8E14C41382D6523840B97F70F25E834A89F6EBAAA47569ACE760E`.

### Thứ tự đề xuất cho buổi trình bày

1. **Bài toán và ký hiệu:** `D0 + l → P(E_l|D0,l)`; giải thích `β`, sáu budget, `>x50`, quy tắc 8/10 duy trì. Nói rõ x1–x5 là đánh giá hồi cứu vì pilot đã tốn x10.
2. **DATA:** 84 stochastic + 36 deterministic control, 5 behavior/MDP, pilot tách mười stream nhãn, ba learner × sáu budget × mười seed = 108.000 core jobs. Đưa sơ đồ luồng và ví dụ learner-specific trên; công khai true MDP chỉ phục vụ sinh/chấm nhãn và các đối chứng được đánh dấu privileged.
3. **METHOD và đối chứng:** OEH-B, OEH-H, coverage, fitted head; OEH-H là descriptor theo mô hình, không phải cận mẫu đã chứng minh. Bảng quyền truy cập tách OEH/B2/ERI adaptation, oracle EH và TQ/SACo. Đối chứng tác giả khác có endpoint khác phải báo đúng phép chuyển đổi.
4. **METRICS → RESULT:** đặt Brier/log loss cạnh bảng dự báo và bootstrap theo MDP; kèm censoring, calibration, runtime, ablation có cả kết quả bất lợi. Bảng BRIDGE 14 task trước phải ghi **reduced OEH**, BC kém; chỉ thay bằng full OEH trên panel mới khi manifest kiểm định hoàn tất.
5. **Ứng dụng phụ và giới hạn:** bảy learner chỉ ở bảng selector với return/success; nêu family label và true beta; kết thúc bằng tính giới hạn của generator nội bộ, chi phí posterior, dữ liệu ngoài và quyền claim SOTA.

Slide một trang hiện được dựng bởi [`make_slide.py`](../../../OEH_Hop_2026-10-04/make_slide.py) là **slide kết quả**, không phù hợp làm slide mở đầu. Trước nó cần tối thiểu một slide đặt bài toán/nhãn và một slide quy trình sinh data–tạo nhãn/đầu vào dự báo; bảng selector đặt sau bảng Brier/log loss. Demo bảng hiện tại cũng in kết quả ngay, nên khi trình bày phải nói hai slide giải thích đó trước khi chạy demo.

**Không cần chạy lại để sửa lời giải thích:** 120 MDP, 108.000 outcome jobs và bảng selector đã có seal/manifest. Nếu cần một bảng chéo mới từ nhãn đã mở, ghi `exploratory`. **Cần tiếp tục phép chạy độc lập đang khóa:** [BRIDGE fresh protocol](../../benchmarks/bridge/fresh_panel_2026-10-08/protocol.json) và [tiến độ](../../artifacts/week8/bridge_fresh_overnight_progress_2026-10-08.json); không công bố thắng/chuyển miền trước khi đủ 25 nhóm, prelabel seal, 75 nhãn và manifest chấm. Nếu thí nghiệm đó thất bại, báo kết quả âm/NO-GO nguyên trạng. Lượt BRIDGE cũ không thể được “làm mới” bằng cách xóa nhãn.

**Minh chứng về thành phần còn có giới hạn:** [ablation E1](E1_EXECUTION_RECORD_2026-10-07.md) trên 120 MDP đã mở là khám phá. Bỏ OEH-H/risk làm Brier BC tốt hơn về điểm, CFQI/IQL kém hơn; các khoảng tin cậy của thay đổi riêng khối cắt 0. Vì thế chưa được nói từng khối OEH đều cần thiết hoặc cải thiện đồng đều. Nếu muốn claim nhân quả cho từng khối trên môi trường ngoài, phải khóa arm và phân tích trước nhãn của một cohort mới; không chạy lại trên 120 MDP hay 14 BRIDGE đã mở để tạo claim đó.

**Đã sửa mâu thuẫn thuật ngữ:** [thesis §3.3](../../../Drive/Thesis/03_Theoretical_Foundation_2026-09-19.md) và [methodology §5](../../proposed_methodology.md) nay phân biệt support của behavior posterior (OEH-H, có smoothing) với support action quan sát từ pilot (supported-return ceiling); điều này khớp [mã horizon](../../code/oeh/horizon.py) và [runner](../../scripts/analysis/run_oeh_final_120.py).
Khi posterior làm trơn cho mọi action xác suất dương, điều kiện `beta > 0` của OEH-H có thể hầu như không loại action nào. Đây là **giới hạn mô hình** cần nêu khi giải thích thuật ngữ support; không sửa code niêm phong sau khi đã xem outcome rồi giữ nguyên claim kiểm định cũ.

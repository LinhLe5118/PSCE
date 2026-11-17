# E3 — OEH cuối cùng so với tín hiệu tác giả công bố (08/10/2026)

## Câu hỏi và phạm vi

Đầu ra chính của OEH là phân phối bảy lớp nhu cầu dữ liệu `E_l` của từng learner từ pilot offline `D0`. Không gọi một baseline ghép đặc trưng B2, hoặc B2 + ERI, là phương pháp nguyên bản của tác giả ngoài nhóm. Bảng chính Brier/log loss hiện chỉ có đối chứng **nội bộ**. Trong các công trình đã rà, chưa xác minh được hệ tác giả nào nhận cùng `D0`, cùng learner và xuất trực tiếp phân phối `E_l`; vì vậy hiện **không có kết quả OEH vượt SOTA trên cùng endpoint**.

Phép đối chiếu ngoại nghiên cứu khả thi ngay là **bài toán phụ xếp hạng độ dễ của nhu cầu dữ liệu**. Chúng tôi chạy lại `scripts/analysis/week8_original_score_audit.py` trên dự báo, score và nhãn đã niêm phong của 120 MDP. Bản chạy mới ở [`original_score_rank_recheck_2026-10-08`](../../artifacts/week8/original_score_rank_recheck_2026-10-08/manifest.json) có CSV SHA-256 trùng bản cũ [`original_score_rank_exploratory_2026-10-07`](../../artifacts/week8/original_score_rank_exploratory_2026-10-07/manifest.json). Không thay score, cohort, dấu hay nhãn theo kết quả.

## Đối chiếu định lượng: 84 MDP stochastic

Spearman giữa score (cao = dự kiến dễ hơn) và `-E_l` thực nghiệm. OEH dùng `-E[E_l|D0,l]`. Mỗi hàng dùng đúng các MDP–behavior mà cả OEH và score gốc đều có giá trị. Khoảng trong cột cuối là bootstrap ghép cặp 1.000 lượt theo cụm MDP, **mô tả trên cohort đã mở**. `>x50` là lớp right-censored nên có nhiều tie.

| Learner | Score tác giả | Quyền truy cập | Hàng / MDP | ρ gốc | ρ OEH cùng hàng | Δρ OEH−gốc [95%] |
|---|---|---|---:|---:|---:|---:|
| BC | ERI | Pilot | 398 / 84 | −0,340 | 0,502 | +0,842 [0,739; 0,941] |
| BC | TQ | MDP thật để tạo reference | 420 / 84 | 0,404 | 0,491 | +0,087 [0,057; 0,119] |
| BC | SACo | MDP thật để tạo reference | 420 / 84 | −0,142 | 0,491 | +0,633 [0,533; 0,729] |
| CFQI | ERI | Pilot | 398 / 84 | −0,312 | 0,711 | +1,023 [0,885; 1,157] |
| CFQI | TQ | MDP thật để tạo reference | 420 / 84 | 0,343 | 0,727 | +0,384 [0,282; 0,475] |
| CFQI | SACo | MDP thật để tạo reference | 420 / 84 | 0,348 | 0,727 | +0,379 [0,262; 0,491] |
| IQL | ERI | Pilot | 398 / 84 | −0,505 | 0,711 | +1,216 [1,079; 1,338] |
| IQL | TQ | MDP thật để tạo reference | 420 / 84 | 0,603 | 0,699 | +0,096 [0,060; 0,134] |
| IQL | SACo | MDP thật để tạo reference | 420 / 84 | −0,307 | 0,699 | +1,006 [0,883; 1,121] |

Nguồn số đầy đủ: [`rank_audit.csv`](../../artifacts/week8/original_score_rank_recheck_2026-10-08/rank_audit.csv). ERI chỉ xác định trên 398/420 hàng primary do điều kiện mean return dương. ERI nguyên bản đo *cơ hội cải thiện so với behavior*, không cam kết rằng score cao nghĩa là cần ít dữ liệu để đạt ngưỡng return. Do đó **không dùng Δρ ERI để nói OEH thắng ERI ở nhiệm vụ gốc của họ**. TQ/SACo là descriptor chất lượng/độ phủ dữ liệu và dùng tham chiếu có quyền truy cập cao hơn pilot; chỉ nên đọc những hàng đó như stress test chuyển endpoint. Không dùng score đơn vô hướng để tính Brier/log loss bảy lớp.

## Phương pháp khác và trạng thái thực thi

| Nguồn tác giả | Đầu ra nguyên bản | Trạng thái trên giao thức hiện tại |
|---|---|---|
| [Swazinna et al. — ERI](https://arxiv.org/abs/2111.13461) | Tiềm năng cải thiện từ return trajectory | Đã chạy score gốc trên 120 MDP và 14 BRIDGE task; subset undefined được báo. EAS/COI continuous-action không chuyển nguyên bản sang action rời rạc. |
| [Schweighofer et al. — TQ/SACo](https://proceedings.mlr.press/v199/schweighofer22a.html) | Trajectory quality và state–action coverage chuẩn hóa | Đã chạy với code DQN tác giả làm reference trên 120 MDP và 14 BRIDGE task; cần MDP thật. TQ undefined ở 3/14 BRIDGE task do mẫu số chuẩn hóa bằng 0. |
| [Laidlaw et al. — EH/BRIDGE](https://github.com/cassidylaidlaw/effective-horizon) | Effective horizon và độ khó RL online từ MDP thật | Đã chấm metadata tác giả `EPW`, `k_min` trên 14 BRIDGE task; là oracle với endpoint khác. OEH đầy đủ chưa chạy BRIDGE vì gate tài nguyên, nên không có so sánh trực tiếp đầy đủ ở đây. |
| [Mateos-Melero et al. — image-based performance predictor](https://plg.uc3m.es/papers/mateos-melero-et-al-ewrl24.pdf) | Return của policy học từ một dataset tại learner/cấu hình cố định | Gần nhất với dự báo, nhưng không xuất `E_l`. Chuyển ảnh và return sang bảy mức budget cần mô hình/nhãn/fit mới: sẽ là **tái lập thích nghi**, không phải hệ tác giả nguyên bản. Mã/weights chính thức dùng trực tiếp cho panel này chưa được xác minh. |
| [Nie et al. — SSR](https://arxiv.org/abs/2210.08642) và các phương pháp offline policy selection | Chọn policy/siêu tham số sau khi train | Có thể đối chiếu selector ở endpoint riêng, nhưng không phải forecaster nhu cầu dữ liệu trước train. |

BWD bị loại khỏi bảng hiện tại vì thiết kế action continuous không khớp tabular discrete theo quyết định [T1 v1.3.0](T1_COMPARATOR_DECISION_v1.3.0_2026-10-08.md). Các hàng B2 và feature adaptation vẫn hữu ích để kiểm thành phần OEH nhưng không dùng cho claim vượt phương pháp tác giả.

## Kết luận được phép đưa vào paper

OEH bản đầy đủ có tương quan thứ hạng với `E_l` cao hơn ERI/TQ/SACo **trong phép chuyển mục tiêu phụ trên 84 MDP đã mở**; các score tác giả không được thiết kế để dự báo phân phối `E_l`. Bảng này đã được thêm vào `paper/sections/06_results.tex`. Bằng chứng hiện tại **chưa đủ** cho phát biểu “vượt SOTA” về dự báo nhu cầu dữ liệu hay cho BRIDGE với OEH đầy đủ. Để kiểm định claim đó, cần một forecaster ngoại nghiên cứu được chốt trước outcome, cùng input/target/split và một panel test mới chưa dùng để chọn cách chuyển giao.

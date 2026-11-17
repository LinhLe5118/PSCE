# OEH, EH và kết quả đối chiếu trên E1

Ngày rà soát: 01/10/2026. Phạm vi: mã `code/oeh/`, protocol và paper trong
`Devoted`, các artifact E1 120 MDP, hai cohort mới, cùng tài liệu phân công và
báo cáo tuần ở thư mục gốc. Các thay đổi chưa commit có sẵn được giữ nguyên.

## OEH rốt cuộc giải quyết vấn đề gì?

**Đúng: mục tiêu khoa học gốc của Offline Effective Horizon (OEH) là ước
lượng độ khó học offline trên MDP stochastic từ một pilot log.** Cụ thể,
OEH-v3 ước lượng lượng dữ liệu tương lai cần để rủi ro chọn sai action trong
support quan sát cộng với planning regret giảm dưới ngưỡng chất lượng.
Nó kết hợp độ sâu QVI, action gap, phương sai transition, tần suất behavior,
occupancy và tác động lên return; dạng chỉ số là
`OEH(k) = (k+1) + log_A N_hat_k`. Bài toán thực nghiệm đối ứng là dự báo
*empirical data requirement* có điều kiện theo MDP, behavior dataset,
learner và mục tiêu return, thay vì gán một nhãn "dễ/khó" tuyệt đối cho
MDP. OEH-v3 thô là proxy theo MDP–behavior, chưa phải ước lượng riêng cho
từng learner; LC-OEH4 và mô hình dự báo mới thêm thông tin learner. Định
nghĩa và thứ tự câu hỏi nghiên cứu được viết rõ trong
`../../../Drive/Thesis/01_Introduction.md` và
`../../../Drive/Thesis/03_Theoretical_Foundation_2026-09-19.md`.

Nhãn thực nghiệm là ngân sách đầu tiên trong
`{x1,x2,x5,x10,x20,x50}` mà learner duy trì ít nhất 8/10 seed thành công ở
mọi ngân sách lớn hơn đã đo; không đạt thì right-censor ở `>x50`. Nhánh này
dùng 12 đặc trưng OEH thêm vào 18 đặc trưng pilot B2 và một mô hình
XGBoost-AFT. **Proxy requirement thô của OEH3/LCOEH4 không phải bộ dự báo
ngân sách đã được hiệu chuẩn**; bản thân chúng cho điểm Brier kém.

**Chọn learner là bước sử dụng phép đo ở phía sau.** Package OEH-v5 hiện tại
lấy diagnostic từ pilot, học chênh lệch *normalized return* và tỷ lệ seed
đạt ngưỡng 0,95 của từng learner so với CFQI, rồi định tuyến một trong bảy
learner E1 cho mỗi nhóm MDP–behavior. Đây là lý do README và paper bản hiện
tại dễ khiến người đọc tưởng OEH chỉ là router. Router không trực tiếp kiểm
định độ chính xác của `N_hat_k`, không cập nhật Bellman hoặc chứng nhận
sample complexity/no-harm. Chi tiết implementation nằm ở
`code/oeh/README_METHOD.md` và `paper/sections/04_routing.tex`.

## OEH cải thiện gì so với EH gốc?

| | EH của Laidlaw và cộng sự | OEH trong repo |
|---|---|---|
| Câu hỏi | MDP khó đến mức nào đối với RL tương tác/random exploration; độ sâu lookahead, QVI và action gap chi phối sample complexity ra sao? | Một pilot dataset của MDP stochastic cho thấy learner offline cần bao nhiêu dữ liệu để đạt chất lượng mục tiêu? Dự báo đó có giúp chọn learner không? |
| Thông tin | MDP thật, rollout/giá trị tối ưu/action gap trong phép tính oracle | Pilot log và nhãn meta-training từ MDP khác; không dùng transition kernel thật làm feature |
| Đầu ra | Thước đo/giới hạn độ khó của môi trường và thuật toán online GORP/SQIRL | Proxy `N_hat_k` và đặc trưng độ khó trên support quan sát; mô hình dự báo requirement và router downstream |
| Loại cải thiện có thể kiểm chứng | EH giải thích mẫu độ khó online | OEH đưa **behavior coverage, stochasticity và learner** vào bài toán offline, rồi kiểm tra ích lợi dự báo và lựa chọn; chưa suy ra giới hạn sample complexity mạnh hơn EH |

EH 2023 đánh giá 155 MDP deterministic BRIDGE; mở rộng 2024 xét MDP
stochastic và SQIRL. Hai công trình là nguồn cảm hứng cơ chế, không phải
baseline cùng estimand của OEH. Nguồn gốc:
[EH 2023](https://papers.nips.cc/paper_files/paper/2023/file/b8be628bf719550b560de8bec9456e0b-Paper-Conference.pdf),
[stochastic EH 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/3e18f7152b5e1696e73722dc5121b6f2-Paper-Conference.pdf).

## Đối chiếu đúng 120 MDP E1

E1 gồm 84 MDP stochastic (primary) và 36 deterministic (control), 5 behavior,
10 seed/nhóm; đánh giá router ở x10. Với bài toán requirement, mỗi MDP có 15
hàng behavior–core learner. Năm fold giữ nguyên toàn bộ hàng của một MDP.
Mọi kết quả E1 dưới đây **hậu nghiệm** vì OEH đã được phát triển sau khi xem
E1. Không thể dùng khoảng bootstrap nội bộ để xóa hiệu ứng chọn mô hình này.

### Sử dụng downstream: chọn learner ở x10

| Panel | OEH return / success | CFQI return / success | Chênh lệch OEH–CFQI |
|---|---:|---:|---:|
| 84 stochastic | 0,806803 / 45,76% | 0,783373 / 41,19% | +0,023429 / +4,57 điểm % |
| 36 control | 0,428267 / 45,22% | 0,428267 / 45,22% | 0 / 0 |
| 120 gộp, mô tả | 0,693242 / 45,60% | 0,676842 / 42,40% | +0,016400 / +3,20 điểm % |

Nguồn: `artifacts/week7/oeh_v5_e1_reuse_r44_2026-09-29/` và
`artifacts/README.md`. Đây là ưu thế *trong thư viện learner E1* trên chính
cohort đã tái sử dụng, không phải ưu thế so với mọi phương pháp offline RL.
OEH-v4E gốc từng bị NO-GO; kết quả mới không thay verdict đó.

### Dự báo requirement: cùng endpoint, cùng AFT, cùng fold

Integrated Brier qua sáu ngưỡng; thấp hơn là tốt hơn. Bảng chỉ dùng các
predictor có input pilot, không trộn các oracle biết MDP thật.

| Đặc trưng cho AFT | 84 stochastic | 36 control | Tình trạng |
|---|---:|---:|---|
| Tần suất event trong train fold | 0,11343 | 0,14165 | Mốc đơn giản |
| Coverage adaptation | 0,11566 | 0,07086 | Không phải SACo gốc |
| Quality adaptation | 0,10411 | 0,03097 | Không phải TQ gốc |
| Coverage + quality adaptation | 0,09260 | 0,02969 | Adaptation |
| Tổ hợp tín hiệu tabular chất lượng mới | 0,08648 | 0,04241 | Adaptation; nhiều giá trị thiếu |
| B2 generic | 0,08781 | 0,02912 | Cùng kiến trúc AFT |
| B2 + OEH | **0,07646** | **0,02894** | Cùng kiến trúc AFT |
| OEH3 raw proxy, không fit | 0,20873 | 0,71142 | Kém hơn tần suất fold |

Nguồn: `artifacts/difficulty_e1_posthoc_2026-09-30/comparison.csv`,
`artifacts/e1_dataset_signals_posthoc_2026-09-30/comparison.csv`,
`artifacts/tabular_e1_quality_scoring_2026-10-01/summary.csv`. So với B2
generic trên stochastic, thêm OEH giảm Brier 0,01134 (12,92%). So với tổ
hợp adaptation mới 0,08648, chênh lệch mô tả là 0,01002 (11,59%). Tổ hợp
mới có 20,42% giá trị feature thiếu trên panel stochastic và thiếu reference
dataset của công thức TQ/SACo gốc; không được gọi là tái lập nguyên bản.
Control cải thiện rất nhỏ so với B2 (0,00018) và log-loss B2+OEH xấu hơn
(0,4057 so với 0,4010). Kết quả E1 này không thiết lập SOTA.

### EH và các comparator dùng oracle

Đã tái dựng 120 MDP theo source/metadata E1 để tính stochastic EH, effective
planning window, covering-length *upper bound* và mốc theo kích thước. Trên
84 stochastic MDP, Spearman giữa âm độ phức tạp oracle và tỷ lệ nhóm offline
thành công tới x50 lần lượt là **-0,131** (stochastic EH $\bar H$),
**0,283** (độ sâu QVI tối thiểu, một thành phần của EH stochastic),
**0,266** (planning window), **0,233** (covering bound, giữ năm giá trị vô
hạn) và **0,373** (worst-case size reference). Đây chỉ là phép chuyển thước đo *online*
sang outcome *offline*, không phải head-to-head sample-complexity hay bằng
chứng OEH thắng EH. E1 seal không chứa hash transition-kernel gốc, nên khớp
identity chỉ tới mức source và metadata/oracle audit được lưu. Nguồn:
`artifacts/e1_oracle_qvi_posthoc_2026-09-30/rank_comparison.csv` và manifest.

Các công trình gần nhiệm vụ offline hơn cũng khác đích hoặc input:
[TQ/SACo](https://proceedings.mlr.press/v199/schweighofer22a.html) cần
reference datasets; [offline policy selection](https://proceedings.mlr.press/v151/yang22a.html)
chọn policy từ belief về giá trị;
[image predictor](https://openreview.net/pdf?id=AKU4h6BPG7) cần ảnh biểu diễn;
[BWD](https://arxiv.org/abs/2510.12638) dùng critic và khoảng cách action.
Các bản gốc đó chưa được chạy trên cùng endpoint E1. Bảng adaptation không
đủ để tuyên bố vượt SOTA trong văn liệu.

## Kiểm tra trên MDP mới và quyết định cho paper

- Cohort 60 MDP mới đầu tiên: B2+OEH AFT đạt Brier 0,06414 so với B2
  0,07137 trên 42 stochastic; cận dưới đồng thời -0,00895, **NO-GO**.
- Cohort 60 MDP mới thứ hai: OEH blend đạt Brier/log-loss
  0,06133/0,77527 so với B2 AFT 0,07383/0,89886; gate đã đăng ký **GO**.
  Khi đối chiếu B2 cùng kiến trúc dự báo, gain log-loss chỉ 4,18% và cận
  dưới khám phá -0,01827; chưa chứng minh ưu thế riêng của đặc trưng OEH.
- Router trên 60 MDP mới: OEH 0,78182 return, 47,62% success so với CFQI
  0,75908 và 38,62%; hai cận dưới đồng thời so CFQI lần lượt -0,06850 và
  -0,00124. Gate router **NO-GO**.

Nguồn: `artifacts/independent_difficulty_2026-09-30/`,
`artifacts/oeh_difficulty_confirmation_2026-10-01/`,
`artifacts/oeh_difficulty_matched_b2_confirmation_2026-10-01/summary.json`,
và `artifacts/independent_validation_2026-09-30/`.

**Kết luận biên tập:** nêu mục tiêu nghiên cứu OEH là ước lượng độ khó học
offline từ dữ liệu trên MDP stochastic; mô tả router v5 là ứng dụng downstream
của diagnostic đó. Đưa kết quả E1 và hai cohort mới cùng verdict,
nhấn mạnh điểm mạnh có chứng cứ: trên E1 OEH giúp AFT và router theo trung
bình; trên cohort mới có một gate requirement GO so với baseline đăng ký.
Giữ rõ các kết quả NO-GO, matched-control và raw-proxy kém. **Chưa chuyển
paper sang luận điểm “vượt SOTA/EH gốc”.** Muốn khẳng định điều đó cần
comparator nguyên bản hoặc adaptation được khóa trước trên cùng estimand,
cùng input, và một cohort độc lập chưa tham gia phát triển.

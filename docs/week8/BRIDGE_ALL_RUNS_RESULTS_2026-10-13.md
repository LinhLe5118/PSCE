# Tổng hợp các lượt BRIDGE đã chấm điểm — 13/10/2026

## Phạm vi và cách đọc

Đây là kiểm tra **chuyển giao sang các MDP BRIDGE đã công bố**. Nhóm tự tạo log thăm dò offline và nhãn nhu cầu dữ liệu của BC, CFQI, IQL theo giao thức đã khóa; đây **không phải** bộ log offline hoặc kết quả thuật toán gốc do tác giả BRIDGE công bố. Mỗi dự báo là phân phối trên sáu mức budget và lớp `>x50`. Brier tích lũy (trung bình sáu ngưỡng) và log loss càng thấp càng tốt. Mỗi bảng dùng cùng nhãn và metric trong chính panel đó; không lấy mức điểm tuyệt đối giữa các panel làm hiệu quả chuyển giao vì tập MDP và giao thức môi trường khác nhau.

| Panel | MDP | Hành vi/MDP | Hàng/learner | OEH được chấm | Trạng thái |
|---|---:|---:|---:|---|---|
| MiniGrid ban đầu | 14 | 5 | 70 | `oeh_without_direct_oehb` | Đã chấm; đây là **OEH rút gọn**, không phải hệ thống đầy đủ |
| Atari mới | 5 | 5 | 25 | `oeh_full_108` | Đã chấm, dự báo niêm phong trước nhãn |
| Mở rộng: 2 Atari, 2 MiniGrid, 1 Procgen | 5 | 5 | 25 | `oeh_full_108` | Đã chấm, dự báo niêm phong trước nhãn |
| OEH đầy đủ trên chính 14 MiniGrid đã mở | 14 | 5 | 70 | Đang tính OEH-B | **Chưa có bảng điểm**; nhãn của panel này đã mở trước khi bắt đầu phần mở rộng, nên kết quả sau này chỉ là khám phá |

Tổng cộng có **24 MDP được chấm** trong ba panel hoàn tất, nhưng chỉ **10 MDP thuộc hai panel kiểm OEH đầy đủ**. Không được gộp 14 MDP rút gọn vào một kết luận về OEH đầy đủ, cũng không gọi 24 MDP này là 24 phép xác nhận độc lập cùng một cấu hình. Các lượt hoàn tất có [bảng điểm và manifest của 14 MiniGrid](../../artifacts/week8/bridge_reduced_test_analysis_2026-10-07/), [5 Atari](../../artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08/) và [5 MDP mở rộng](../../artifacts/week8/bridge_expanded_full_test_analysis_2026-10-08/). Manifest ghi hash SHA-256 của các đầu ra; các hash này đã được kiểm tra lại khi lập báo cáo.

## 1. Panel 14 MiniGrid: OEH rút gọn với B2 nội bộ

Mỗi dòng là trung bình 70 dự báo của một learner. `>x50` là tỷ lệ nhãn bị kiểm duyệt phải cần hơn budget lớn nhất. Các số lấy từ [summary.csv](../../artifacts/week8/bridge_reduced_test_analysis_2026-10-07/summary.csv); [paired_task_bootstrap.csv](../../artifacts/week8/bridge_reduced_test_analysis_2026-10-07/paired_task_bootstrap.csv) ghép cặp theo MDP, giữ năm behavior cùng nhau.

| Learner | `>x50` | B2 Brier / log loss | OEH rút gọn Brier / log loss | Δ Brier OEH−B2, khoảng 95% mô tả |
|---|---:|---:|---:|---:|
| BC | 90,0% | 0,022425 / 0,169015 | 0,044523 / 0,323741 | +0,022098 [+0,007196; +0,037803] |
| CFQI | 60,0% | 0,155727 / 1,630302 | 0,081931 / 1,037709 | −0,073796 [−0,096982; −0,049428] |
| IQL | 87,1% | 0,025290 / 0,274767 | 0,015152 / 0,205251 | −0,010137 [−0,018050; −0,002443] |

OEH rút gọn tốt hơn B2 ở CFQI và IQL, nhưng **kém hơn ở BC**. B2 là đối chứng nội bộ có cùng dạng đầu ra, không phải kết quả nguyên bản của một tác giả khác. Panel này không trả lời hiệu quả của toàn bộ OEH-B.

## 2. Panel năm Atari mới: OEH đầy đủ

[Protocol](../../benchmarks/bridge/fresh_panel_2026-10-08/protocol.json), [dấu niêm phong trước nhãn](../../benchmarks/bridge/fresh_panel_2026-10-08/prelabel_seal.json), [nhãn](../../artifacts/week8/bridge_fresh_test_labels_2026-10-08/labels.csv) và [manifest phân tích](../../artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08/manifest.json) cho phép đối chiếu. Bảng dưới đây ghi **Brier / log loss** từ [summary.csv](../../artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08/summary.csv).

| Learner | `>x50` | B2 nội bộ | XGBoost-CF/18 thích nghi | XGBoost-CF/108 kiểm kiến trúc | OEH đầy đủ |
|---|---:|---:|---:|---:|---:|
| BC | 92% | 0,059865 / 0,497956 | 0,045579 / 2,277528 | 0,028191 / 1,305974 | **0,028178 / 0,382242** |
| CFQI | 48% | 0,296849 / 2,099780 | 0,292000 / 3,924963 | **0,261571** / 2,920034 | 0,284121 / 2,111864 |
| IQL | 84% | **0,049781 / 0,324704** | 0,071166 / 0,358160 | 0,117773 / 0,634500 | 0,112798 / 0,819092 |

OEH có điểm BC thấp nhất, nhưng khoảng mô tả của Δ Brier OEH−B2 là −0,031687 [−0,083220; +0,009082], vẫn chứa 0. Ở CFQI, Δ OEH−B2 là −0,012728 [−0,071202; +0,035563]; OEH lại kém kiểm soát kiến trúc 108 cột về Brier. Ở IQL, OEH **kém B2**: +0,063018 [+0,004222; +0,138822] Brier và log loss cao hơn cả ba đối chứng. Xem [bảng ghép cặp](../../artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08/paired_task_bootstrap.csv). Không có chiến thắng đồng đều trên panel này.

## 3. Panel năm MDP mở rộng: OEH đầy đủ

Panel gồm hai Atari, hai MiniGrid và một Procgen. [Protocol](../../benchmarks/bridge/expanded_panel_2026-10-08/protocol.json), [dấu niêm phong trước nhãn](../../benchmarks/bridge/expanded_panel_2026-10-08/prelabel_seal.json), [nhãn](../../artifacts/week8/bridge_expanded_test_labels_2026-10-08/labels.csv) và [manifest phân tích](../../artifacts/week8/bridge_expanded_full_test_analysis_2026-10-08/manifest.json) xác định cùng endpoint. Bảng ghi **Brier / log loss** từ [summary.csv](../../artifacts/week8/bridge_expanded_full_test_analysis_2026-10-08/summary.csv).

| Learner | `>x50` | B2 nội bộ | XGBoost-CF/18 thích nghi | XGBoost-CF/108 kiểm kiến trúc | OEH đầy đủ |
|---|---:|---:|---:|---:|---:|
| BC | 92% | 0,045499 / 0,214500 | **0,037074 / 0,157030** | 0,065261 / 0,367247 | 0,056365 / 0,274487 |
| CFQI | 40% | 0,284935 / 2,023458 | 0,266283 / 4,019918 | 0,184938 / 3,855647 | **0,153300 / 1,521154** |
| IQL | 92% | 0,074708 / 0,348738 | 0,074309 / 0,287602 | **0,064962 / 0,286363** | 0,071049 / 0,338863 |

CFQI là điểm mạnh của OEH ở panel này: Δ Brier OEH−XGBoost-CF/18 = −0,112983 [−0,226770; −0,016333]; Δ log loss = −2,498765 [−4,557442; −0,546106]. OEH cũng thấp hơn B2 về CFQI Brier: −0,131635 [−0,269437; −0,018675]. Với BC, OEH kém XGBoost-CF/18 cả hai điểm; Δ log loss = +0,117458 [+0,005975; +0,228940]. Với IQL, OEH Brier hơn XGBoost-CF/18 một ít nhưng khoảng chứa 0, còn log loss **kém**: +0,051261 [+0,011761; +0,112425]. Xem [bảng ghép cặp](../../artifacts/week8/bridge_expanded_full_test_analysis_2026-10-08/paired_task_bootstrap.csv).

## Đối chứng, giới hạn và kết luận được phép nói

- XGBoost-CF là **khung survival đã công bố được nhóm thích nghi** sang nhãn bảy lớp với 18 thống kê pilot; các điểm trên đây không phải điểm offline RL do tác giả công bố. Hàng 108 cột nhận đặc trưng OEH để kiểm soát ảnh hưởng của kiến trúc, nên không phải đối thủ độc lập dùng thông tin riêng. B2 là mốc kiểm soát nội bộ.
- Điểm gốc của ERI, TQ, SACo và EH được giữ trong `original_rank_comparison.csv` và `original_rank_paired_transfer.csv` của hai panel full. Chúng đo **liên hệ thứ hạng của điểm vô hướng** với nhãn khó hơn, không xuất phân phối bảy lớp. TQ/SACo/EH còn có tầng thông tin MDP thật hoặc tham chiếu online. Vì thế không xếp Brier của OEH cạnh điểm gốc, không tuyên bố OEH thắng SOTA chỉ nhờ bảng này.
- Các khoảng bootstrap là **mô tả theo từng phép so** với chỉ năm MDP mỗi panel, không điều chỉnh đồng thời nhiều phép thử. Tỷ lệ `>x50` rất cao ở BC/IQL khiến vài điểm trung bình nhỏ nhưng không chứng minh dự báo hữu dụng cho mọi learner. Một nhóm ở Atari và bốn nhóm ở panel mở rộng dùng fallback do không có posterior draw; dấu niêm phong vẫn được giữ, nhưng đây là giới hạn triển khai cần báo cáo.
- Kết luận trung thực: OEH đầy đủ có lợi thế rõ ở **CFQI trên panel mở rộng**, nhưng kết quả chuyển giao **phụ thuộc learner và MDP**; IQL trên Atari và BC trên panel mở rộng là trường hợp bất lợi. Chưa có chứng cứ cho tuyên bố vượt trội đồng đều trên BRIDGE hay SOTA toàn lĩnh vực.
- Luồng 14 MiniGrid dùng OEH đầy đủ là thí nghiệm **sau khi đã mở nhãn** của panel cũ. File tiến độ tại thời điểm kiểm tra ghi 50/70 nhóm OEH-B, chưa có forecast hay analysis. Khi xong phải báo riêng là khám phá; không trộn vào hai panel full đã niêm phong trước nhãn.

Để tái tính, dùng `scored.csv` trong từng thư mục phân tích: mỗi hàng chứa nhãn, xác suất bảy lớp, Brier và log loss. Các `summary.csv` là trung bình theo learner/phương pháp; manifest chứa số hàng, trạng thái và hash của đầu ra. Bản tổng hợp này lưu **bằng chứng gọn** trên Git; log thô, checkpoint và bộ dữ liệu lớn tiếp tục ở kho dữ liệu riêng.

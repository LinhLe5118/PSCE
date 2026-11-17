# H1 — Review độc lập phép bỏ khối trước khi khóa ablation

**Ngày:** 08/10/2026. **Phạm vi:** review cấu trúc feature hiện hành từ code và
[protocol cohort 7](../../artifacts/oeh_confirmatory_unseen_120_2026-10-04/protocol.json),
không chạy ablation, không dùng kết quả để chọn phép bỏ khối.
**Trạng thái:** đề xuất để Hưng/Linh duyệt, chưa phải protocol đã freeze.
Bản B1 mới không cần có trước khi bàn giao phần review cấu trúc này.

## 1. Phân rã đủ 108 cột, không đếm trùng

Tên dưới đây lấy từ danh sách `oeh_final.fields` đã seal; giữ thứ tự trong
protocol khi dựng mọi subset. Bảng phân nhóm là quy ước review, không thay đổi
candidate. Nguồn tuple:
[oeh_final_forecast.py](../../scripts/analysis/oeh_final_forecast.py) và
[run_oeh_final_120.py](../../scripts/analysis/run_oeh_final_120.py).

| Khối review | Quy tắc lấy cột trong danh sách đã seal | Số cột | Lưu ý |
|---|---|---:|---|
| G1 — B2 | `FEATURES_B2` | 18 | Đã gồm coverage, entropy và return/trajectory statistics |
| G2 — OEH-B của learner hiện tại | `OWN`: `oehb_cdf_*`, `oehb_success_x*`, `oehb_return_x*` | 18 | Không gồm `oehb_p_*`; raw law không là ensemble component khi law=false |
| G3 — concentration posterior | `oehb_log_transition_concentration`, `oehb_log_behavior_concentration` | 2 | Nằm trong tuple `HORIZON` nhưng nguồn là posterior, không phải depth |
| G4 — OEH-H/ceiling hiện hành | Cột `oehh_*` không thuộc `PROFILE` | 11 | Gồm trạng thái, support/behavior ceiling và concentrability |
| G5 — risk/coverage profile | `PROFILE`: `oehh_risk_x*`, `oehh_miss_x*`, `oehh_risk_slack` | 13 | Có cả risk và missing mass, không phải riêng depth |
| G6 — OEH-B context | `CONTEXT`: `other_<learner>_oehb_cdf_*` | 18 | Có cả bản sao CDF của chính learner ở hàng đó |
| G7 — diagnostic lịch sử của learner | `oeh3_*`, `lcoeh4_*` trong candidate | 10 | Không mặc định cùng công thức OEH-H hiện hành |
| G8 — diagnostic lịch sử context | `other_<learner>_lcoeh4_*` | 18 | Có cả learner hiện tại; phụ thuộc cùng pilot |
| Tổng | Hợp rời nhau G1–G8 | 108 | Không thêm cột curve hoặc Bayesian ceiling variant khác |

Đặt tên phép bỏ theo **thông tin bị bỏ**, không theo tên tuple Python.
`HORIZON` trộn G3 và G4; bỏ nguyên tuple này đồng thời bỏ concentration.
`CONTEXT`/`LC_CONTEXT` chứa bản sao cùng-learner: khi bỏ context, bản chính vẫn
còn. Đây không phải 36 nguồn thông tin hoàn toàn độc lập.

## 2. Các phép bỏ khối có thể định nghĩa rõ ngay

| Phép thử đề xuất | Cột bị bỏ | Cột còn | Kết luận được phép nếu có bằng chứng |
|---|---|---:|---|
| Full | Không | 108 | Reference của ablation, không fit lại outcome test để chọn cấu hình |
| Minus current OEH-H/profile | G4 + G5 | 84 | Incremental value của bộ descriptor hiện hành khi giữ legacy; không phải "không còn planning information" |
| Minus learner-simulation summaries | G2 + G3 + G6 | 70 | Giá trị của summary/concentration/context đó; posterior-mean model vẫn cấp đầu vào cho G4/G5 |
| Minus context columns | G6 + G8 | 72 | Giá trị tăng thêm của context đã khai báo; có redundancy với own columns |
| Minus legacy diagnostics | G7 + G8 | 80 | Giá trị tăng thêm của legacy, không gán mọi lợi ích còn lại cho OEH-H |
| B2 matched | Giữ riêng G1 | 18 | Generic control với cùng fitting procedure; không phải phép bỏ đúng một cơ chế |

**Không chấp thuận cách gọi "without coverage" chỉ vì bỏ G5.** Coverage vẫn có
ở G1, G4, G7/G8. Muốn kiểm định cơ chế coverage rộng phải lập danh sách lineage
và quyết định xử lý feature hỗn hợp; đây là lựa chọn khoa học cần Linh khóa,
không tự suy từ prefix. Tương tự, "without posterior" khác với bỏ summaries:
OEH-H hiện hành còn sử dụng posterior mean.

## 3. Điều kiện review trước khi chạy

1. Export danh sách cột đầy đủ cho từng variant; kiểm tra union/disjointness,
   số cột, thứ tự và checksum. Không chọn khối bằng outcome cohort 7.
2. Refit từng variant với cùng AFT/direct/depth3 procedure, grouped folds,
   learner strata, seed schedule, tuning objective/capacity và dữ liệu meta-train.
   Cùng quy trình không có nghĩa copy weights của full sang mọi subset.
   Không chỉ điền 0 vào feature của model full đã fit: đó là masking intervention,
   không phải retrained leave-block-out.
3. Giữ nguyên nhãn suffix, censoring, pilot cost, primary/control panels; so
   sánh paired theo MDP. Freeze contrast family và multiplicity trước outcome.
4. Cohort 7 đã mở: ablation trên đó chỉ là exploratory. Muốn xác nhận độc lập
   phải seal subsets/models/predictions rồi dùng cohort chưa mở; không đổi seal cũ.
5. Chi phí feature generation và fit phải được báo riêng. Mọi variant dùng
   cùng nguồn input; không cấp true MDP cho một variant data-only.
6. Không có ablation đã chạy ở H1 này. Một kết quả giảm accuracy khi bỏ khối
   chỉ chứng tỏ giá trị tăng thêm dưới fitting procedure đã chọn, không chứng
   minh tính nhân quả phổ quát, tính cần thiết hoặc từng công thức đúng.

## 4. Quyết định cần nhóm xác nhận

Đề xuất khóa trước bốn contrast Full so với bốn phép bỏ G ở bảng trên; nếu đưa
Full so với B2 vào family kiểm định mới thì tổng là năm. B2 là control đã có.
Hưng cần xác nhận nhóm cột và Linh quyết định thiết kế, gate,
cohort và compute budget. Không chọn thêm khối theo kết quả của các contrast này.
[B1 checklist](H1_B1_REVIEW_2026-10-08.md) là cross-review tiếp theo khi Việt
Trần gửi bản B1 thực tế; không chặn bàn giao glossary/Method/Related Work.

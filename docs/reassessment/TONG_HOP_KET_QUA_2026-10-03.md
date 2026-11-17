# Tổng hợp kết quả OEH đến 03/10/2026

Cập nhật tài liệu ngày 04/10/2026: phần dưới lưu trạng thái đến cohort thứ tư. Lượt xác nhận thứ bảy đã hoàn tất 108.000 job core trên 120 kernel mới. Trên 84 MDP stochastic, OEH đạt Brier 0,045989 và log loss 0,563079; 22/22 cận dưới đồng thời của các tương phản dự báo dương, kể cả TQ/SACo đặc quyền. Tier 1, Tier 2 và gate tổng của cohort thứ bảy GO. Sau khi kiểm toán 16.800 job ngoài, gate chọn learner đã đăng ký so với fixed CFQI cũng GO trên success và return, nhưng so sánh bổ sung với v5 NO-GO. Benchmark Farama FourRooms công bố sẵn cho kiểm tra thành phần NO-GO. Xem [báo cáo kiểm toán mới](KIEM_DINH_2026-10-04.md). Các NO-GO dưới đây vẫn giữ nguyên lịch sử.

Tài liệu này tổng hợp trạng thái của repo từ mã, bản thảo, protocol và các bảng kết quả đã lưu. Hai nhánh kết quả cần đọc riêng: bộ chọn learner OEH-v5 trên E1 và mô hình OEH cuối dự báo nhu cầu dữ liệu trên các cohort mới. Các con số dưới đây lấy từ artifact tương ứng, không gộp hai estimand thành một kết luận.

## Bài toán và sản phẩm đã có

- Bài toán độ khó: từ pilot log x10, dự báo cho từng learner ngân sách đầu tiên trong `{x1, x2, x5, x10, x20, x50, >x50}` mà normalized return đạt ít nhất 0,95 trên ít nhất 8/10 seed ở ngân sách đó và mọi ngân sách lớn hơn đã đo. `>x50` là nhãn kiểm duyệt phải. Hai thước đo chính là integrated Brier qua sáu ngưỡng và log loss bảy lớp.
- Bài toán chọn learner: chọn một trong bảy learner E1 cho mỗi nhóm MDP–behavior, rồi đánh giá return và tỷ lệ seed vượt ngưỡng 0,95. Đây là endpoint khác với dự báo nhu cầu dữ liệu.
- Repo có package `code/oeh/`, runner trong `scripts/analysis/`, test, protocol/manifest, dữ liệu chứng cứ và bản thảo IEEE trong `paper/`. [Phương pháp OEH cuối](../../proposed_methodology.md) phát triển posterior predictive theo learner, horizon offline và các tín hiệu trần hiệu năng; [package router](../../code/oeh/README_METHOD.md) mô tả nhánh OEH-v5 lịch sử. Bản paper hiện tại đã cập nhật kết quả cohort thứ bảy và giữ kết quả cohort thứ tư như lịch sử.

## Kết quả đã đo

| Đánh giá | Kết quả chính | Diễn giải |
|---|---|---|
| E1 gốc, 84 MDP stochastic, x10 | OEH-v5 đạt return 0,806803 và success 45,76%; hơn CFQI 0,023429 return và 4,57 điểm phần trăm success. | Kết quả phát triển hậu nghiệm trên E1 được tái sử dụng thích nghi; không phải xác nhận độc lập. [Bảng và định nghĩa](../../artifacts/README.md). |
| E1 hậu nghiệm, dự báo độ khó | B2+OEH AFT có Brier 0,0765 so với B2 0,0878 trên 84 MDP stochastic. | Diagnostic OEH giúp mô hình đã fit; proxy requirement thô kém hơn fold-frequency. [So sánh](../../artifacts/difficulty_e1_posthoc_2026-09-30/comparison.csv). |
| Kiểm chứng routing mới, 60 MDP | Return và success trung bình dương so với CFQI nhưng cận dưới đồng thời âm; **NO-GO**. | Không có bằng chứng xác nhận bộ chọn learner vượt CFQI theo gate đã đăng ký. [Kết quả](../../artifacts/independent_validation_2026-09-30/result_manifest.json). |
| Kiểm chứng độ khó 120 MDP lần 1 | Trên 84 stochastic: OEH 0,059205 Brier / 0,671877 log loss; B2 cùng kiến trúc 0,066246 / 0,700669. **NO-GO** vì log loss chỉ cải thiện 4,11% và cận dưới không đạt. | [Protocol và quyết định](../protocol/OEH_UNSEEN_120_DECISION_2026-10-01.md), [manifest](../../artifacts/oeh_difficulty_unseen_120_2026-10-01/result_manifest.json). |
| Kiểm chứng độ khó 120 MDP lần 2 | OEH 0,060143 / 0,669121; B2 0,066086 / 0,697772. **NO-GO** vì log loss cải thiện 4,11% và một số cận dưới âm. | [Manifest](../../artifacts/oeh_pooled_successor_unseen_120_2026-10-01/result_manifest.json). |
| Kiểm chứng độ khó 120 MDP lần 3 | OEH cross-capacity 0,058181 / 0,635605; B2 ba thành phần 0,063216 / 0,664531. **NO-GO**: log loss cải thiện 4,35%; các cận dưới đồng thời so với B2/EH âm. | [Đối chiếu](../../artifacts/oeh_cross_capacity_unseen_120_2026-10-01/contrasts.csv), [manifest](../../artifacts/oeh_cross_capacity_unseen_120_2026-10-01/result_manifest.json). |
| OEH cuối, kiểm chứng 120 MDP lần 4 | Trên 84 stochastic: OEH 0,052905 Brier / 0,642729 log loss; B2 cùng kiến trúc 0,062418 / 0,699556. Giảm 15,24% / 8,12%, với cận dưới đồng thời dương. **Tier 1 GO**. | [Bảng](../../artifacts/oeh_final_unseen_120_2026-10-02/summary.csv), [đối chiếu](../../artifacts/oeh_final_unseen_120_2026-10-02/contrasts.csv). |
| OEH cuối, đối chứng mở rộng | So với B2+oracle EH: giảm Brier 18,65%, log loss 10,24%, cận dưới dương. So với B2+TQ/SACo bản author-code có dữ liệu tham chiếu true-MDP: giảm 7,76% / 3,66%, nhưng hai cận dưới âm. **Tier 2 và gate tổng NO-GO**. | Kết quả không xác nhận ưu thế trên toàn bộ đối chứng đã đăng ký. [Verdict](../../artifacts/oeh_final_unseen_120_2026-10-02/result_manifest.json). |
| Chọn learner bổ sung ở lần 4 | OEH cuối đạt return 0,796323 và success 44,43%; CFQI 0,790621 và 42,67%. Cận dưới return âm, dù cận dưới success dương. OEH-v5 router đạt success 51,95%. **Selection NO-GO**. | [Bảng](../../artifacts/oeh_final_unseen_120_2026-10-02/selection_summary.csv), [đối chiếu](../../artifacts/oeh_final_unseen_120_2026-10-02/selection_contrasts.csv). |

Lần thứ tư có protocol và dự báo/chọn learner được niêm phong trước outcome, 120 kernel mới (84 stochastic, 36 control), 108.000 learner job hoàn tất. [Seal](../../artifacts/oeh_final_unseen_120_2026-10-02/preoutcome_seal.json) và [manifest](../../artifacts/oeh_final_unseen_120_2026-10-02/result_manifest.json) giữ hash kiểm toán. Paper PDF đã được dựng lại ngày 04/10/2026 từ bảng kết quả này.

## Kết luận có thể báo cáo

OEH cuối có bằng chứng độc lập về cải thiện dự báo nhu cầu dữ liệu so với các đối chứng cùng thông tin pilot của tier 1 trên cohort thứ tư. Gate gộp tất cả đối chứng thất bại vì TQ/SACo author-code, và gate chọn learner cũng thất bại. Các thử nghiệm độc lập trước đó có cả GO lẫn NO-GO theo gate riêng; không được gộp chúng thành một chuỗi xác nhận liên tục. Kết quả E1 và development chỉ mô tả quá trình xây dựng phương pháp. Chưa có cơ sở để khẳng định OEH dẫn đầu toàn bộ phương pháp đã công bố hoặc chọn learner tốt hơn một cách ổn định.

## Dọn repo ngày 03/10/2026

Giữ nguyên protocol, seal, manifest, bảng outcome, execution jobs, dữ liệu local và snapshot provenance để kiểm toán/tái chạy. Các phần được dọn là môi trường ảo dùng cho audit/GPU probe, hai checkout tạm cũ, cache CuPy rỗng, log rỗng và file phụ LaTeX có thể tạo lại. Các phần này đều bị `.gitignore` bỏ qua hoặc không được Git theo dõi; checkout tạm không được mã/tài liệu hiện hành tham chiếu.

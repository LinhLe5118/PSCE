# Tìm hướng cải thiện OEH — 12/10/2026

**Kết luận:** có hai thay đổi nhỏ cho tín hiệu tốt hơn trên dữ liệu phát triển: hiệu chỉnh độ tự tin của xác suất và kết hợp dự báo OEH với B2. Bốn cách sửa selector thử lần này chưa thắng đồng thời về return và success. Chưa thay candidate, protocol hoặc verdict đã khóa; chưa có cải thiện xác nhận trên cohort mới.

## 1. Điểm yếu thực tế tìm được

Phân tích sau khi xem outcome của cohort 7, chỉ dùng để chẩn đoán và đặt giả thuyết; không là kiểm định subgroup đã đăng ký.

| Họ môi trường | Brier B2 | Brier OEH | Log loss B2 | Log loss OEH |
|---|---:|---:|---:|---:|
| uniform_garnet | 0,054315 | **0,038023** | 0,615085 | **0,510542** |
| forward_layered | 0,072199 | **0,065051** | 0,802781 | **0,736960** |
| bottleneck | **0,020469** | 0,021142 | **0,296131** | 0,302878 |

OEH tốt hơn khi tính chung nhưng không tốt hơn ở mọi họ. Trong bottleneck (môi trường có nút thắt), Brier OEH cao hơn khoảng 3,29%, log loss cao hơn khoảng 2,28% so với B2. Đây là chênh lệch mô tả, chưa chứng minh khác biệt subgroup có ý nghĩa thống kê.

**Lỗi tập trung ở những trường hợp đạt được trong x1–x50.** 360/1.260 task chính thuộc sáu lớp hữu hạn, chiếm 28,57%, nhưng tạo ra 73,74% tổng lỗi Brier và 81,39% tổng log loss của OEH. Brier trung bình của chúng là 0,118692, so với 0,016908 ở lớp >x50. Nghĩa là OEH còn khó phân biệt mức dữ liệu cụ thể khi bài toán thực sự có thể đạt yêu cầu trong miền đã thử. Không suy từ phân tích có điều kiện theo outcome này rằng xác suất đã bị miscalibration toàn cục.

Theo learner, CFQI có lỗi dự báo lớn nhất: Brier 0,077394 và log loss 1,077472; BC là 0,018882/0,177185 và IQL là 0,041691/0,434579. Các nhóm có độ khó và tỷ lệ censoring khác nhau; đây không là bằng chứng thuật toán CFQI kém hơn các learner khác.

**Selector cũng có một số lần đổi learner gây hại.** Trong 71 nhóm forward_layered chuyển sang BCQ, 12 nhóm có return thấp hơn CFQI và 12 nhóm có success thấp hơn CFQI. Đây là đếm MDP–behavior groups, không phải 12 MDP độc lập; hai tập 12 chưa được khẳng định là trùng nhau. Trong 32 nhóm uniform_garnet chuyển sang BCQ, 3 nhóm giảm return dù không nhóm nào giảm success. Tăng tỷ lệ thành công và tăng return là hai mục tiêu có thể xung đột.

Nguồn tái tính: [forecast diagnostics](../../artifacts/week8/improvement_2026-10-12/forecast_diagnostics.csv), [selection switches](../../artifacts/week8/improvement_2026-10-12/selection_switches.csv).

## 2. Hai thử nghiệm dự báo đã chạy

Dùng **cohort 4 đã mở**, 84 MDP chính/1.260 task. Dự báo gốc được tạo trước outcome của cohort đó. Chia năm fold theo MDP; tất cả behavior và learner của một MDP cùng fold. Chỉ fit tham số trên bốn fold train, áp dụng lên fold còn lại. Không dùng outcome cohort 7 để fit. Không huấn luyện lại learner, không sinh thêm raw data.

- **Kết hợp B2:** p_new = w × p_OEH + (1−w) × p_B2. Fit một w trong [0,1] bằng log loss ở train fold, có xét hai đầu mút. Các w thu được khoảng 0,84–0,91: phần lớn vẫn dùng OEH, bổ sung một phần B2.
- **Hiệu chỉnh xác suất:** p_new(c) ∝ p_OEH(c)^(1/T). Fit một T trong [0,5;2] trên train fold. T khoảng 1,09–1,15 làm dự báo bớt cực đoan, không đổi thứ tự lớp có xác suất cao nhất. Bản này không sửa nội dung nhãn.

| Cách dự báo trên development OOF | Brier | Log loss | Giảm Brier so với gốc | Giảm log loss |
|---|---:|---:|---:|---:|
| Quy trình hiện tại | 0,052905 | 0,642729 | — | — |
| Kết hợp OEH + B2 | 0,052761 | 0,640681 | 0,27% | 0,32% |
| Hiệu chỉnh độ tự tin | **0,052599** | **0,640361** | **0,58%** | **0,37%** |

Đây là cải thiện nhỏ về point estimate trên cùng development panel. Không so 0,052599 của cohort 4 với 0,045989 của cohort 7 để nói phương pháp mới tốt hoặc tệ hơn. Cohort 4 đã được thử nhiều lần trước đây; grouped OOF không xóa lịch sử sử dụng đó. Chưa tính cận đã hiệu chỉnh nhiều phép thử và chưa có fresh confirmation. Cả hai thay đổi là thay quy trình dự báo; không diễn giải thành lợi ích feature-only dưới matched head cũ.

Calibration có cơ sở phương pháp từ [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html). Paper nghiên cứu classifier neural trên dữ liệu khác; áp dụng ở đây là giả thuyết được thử bằng development OOF, không suy ra bảo đảm cải thiện OEH từ paper đó. Mã OEH đã có recalibration của raw law; thử nghiệm này hiệu chỉnh **dự báo cuối**, không phải thêm lại cùng bước raw-law calibration.

Nguồn: [summary](../../artifacts/week8/improvement_forecast_2026-10-12/summary.csv), [MDP scores](../../artifacts/week8/improvement_forecast_2026-10-12/mdp_scores.csv), [folds, parameters, source hashes](../../artifacts/week8/improvement_forecast_2026-10-12/scope.json).

## 3. Bốn thử nghiệm selector: chưa có cách nên thay ngay

Dùng đúng design hiện có: pilot của CFQI, xác suất forecast trước outcome và nhãn họ MDP; ExtraTrees 150 tree, leaf 7, max_features 0,8, seed 42. Năm fold theo MDP trên cohort 4, cùng threshold 0,1 và score = predicted success + 0,15 × predicted NR. Không tuning threshold bằng kết quả lần chạy này.

| Cách chọn, development OOF | Mean NR | Seed success | Số lần đổi khỏi CFQI |
|---|---:|---:|---:|
| Quy tắc hiện tại, tái fit trong từng fold | **0,813229** | 49,07% | 92 |
| Như trên, thêm điều kiện cả return/success dự báo đều tăng | 0,803503 | 46,24% | 57 |
| Học trực tiếp chênh lệch so với CFQI | 0,799939 | **49,21%** | 84 |
| Học chênh lệch và thêm điều kiện cả hai đều tăng | 0,802585 | 46,74% | 49 |

Chỉ đổi sang học chênh lệch tăng success khoảng 0,14 điểm phần trăm nhưng giảm NR khoảng 0,01329 so với quy tắc hiện tại trên cùng OOF. Điều kiện thận trọng đơn giản làm giảm cả hai endpoint so với quy tắc hiện tại. Mọi candidate đều có success thấp hơn v5 trên development panel này. **Không chọn candidate nào thay selector đã khóa.**

Project từng có paired-advantage selector cho cohort/learner set khác và có router-v6 development âm về success. Đây là probe dùng ExtraTrees trên bảy learner và hai endpoint, không tuyên bố ý tưởng paired learning mới. Các rule được ấn định trước lần chạy; chọn rule thắng trong bốn rule vẫn cần nested selection nếu đánh giá nội bộ tiếp.

Nguồn: [development probes](../../artifacts/week8/improvement_2026-10-12/development_probes.csv), [group effects](../../artifacts/week8/improvement_2026-10-12/development_group_effects.csv), [phạm vi, folds và hashes](../../artifacts/week8/improvement_2026-10-12/scope.json). Độ nhạy của offline policy selection với cấu hình cũng được nghiên cứu trong [Paine et al., 2020](https://arxiv.org/abs/2007.09055); không dùng paper đó để khẳng định các rule mới đã tốt hơn.

## 4. Thứ tự cải thiện đề xuất

1. **Thử và kiểm chứng calibration ít tham số trước.** Đây là hướng đã có tín hiệu dương, chi phí nhỏ, không cần tăng ngân sách learner. So baseline/temperature/blend trong nested MDP-grouped development; kiểm cả IBS và LL, subgroup và sensitivity theo split. Chọn và khóa một candidate, sau đó seal prediction trên cohort chưa mở. Không hứa mức tăng lớn từ các con số dưới 1% hiện tại.
2. **Giải quyết bài toán đoán mức cụ thể trong x1–x50.** Candidate hai phần: q=P(E=6), rồi r_j=P(E=j | E<6) cho j=0…5; ghép p_6=q và p_j=(1−q)r_j. Thay đổi cấu trúc học, không thêm thông tin mới; mọi seven-class distribution cũng biểu diễn được theo cách này, nên chưa tự tạo bảo đảm cải thiện. Giữ lớp >x50 và score gốc; không xóa lớp khó hoặc đổi ngưỡng 0,95/8 seed để làm bảng đẹp. Cần regularization vì lớp hữu hạn ít dữ liệu, và kiểm calibration trên phân bố đánh giá gốc nếu dùng sampling/weights lúc train.
3. **Selector cần xử lý trade-off, không chỉ tăng độ thận trọng.** Ước lượng paired advantage và độ bất định bên trong train folds, hoặc học gate giữa CFQI/v5/candidate với information access công khai. Đánh giá đồng thời return và success với cả CFQI lẫn v5. Không đặt rule “nút thắt thì dùng B2” hoặc loại BCQ chỉ vì đã xem subgroup của cohort 7. Các kết quả probe âm phải lưu trong lịch sử tìm kiếm.
4. **Đo khả năng dùng ngoài họ đã biết.** Bổ sung leave-one-family-out ở development và thử selector có/không family label. Khôi phục raw/source và chọn benchmark ngoài phù hợp trước khi chạy; giữ ablation riêng để biết nhóm tín hiệu nào đóng góp. Không gọi một cohort tự sinh mới là benchmark ngoài.

## 5. Điều kiện kết luận cải thiện thật

- Development dùng nested group splits nếu chọn trong nhiều candidate; khóa metric/contrast family, seed, data budget, access và cost trước outcome xác nhận.
- Dự báo: kiểm IBS và LL theo MDP, giữ primary/control riêng. Selection: kiểm NR và success so với cả CFQI và v5, không gộp endpoint bằng score 0,15 rồi gọi cả hai đã thắng.
- Seal candidate/model/prediction trước khi mở outcome cohort mới. Đánh giá cận thống kê phù hợp với nhiều phép thử, ghi cả kết quả âm.
- Đo riêng feature-generation, calibration/fit, downstream learner cost. Không tăng train budget cho một learner rồi trình bày là cải thiện của forecaster.
- Cohort 7 đã mở nên mọi sửa chữa xuất phát từ chẩn đoán này cần được xác nhận ở cohort mới; không sửa lại verdict lịch sử.

## 6. Mã chạy và kiểm tra

```powershell
python -m scripts.analysis.diagnose_improvement_20261012
python -m scripts.analysis.probe_forecast_calibration_20261012
python -m scripts.analysis.export_week8_d1 --check
```

Hai runner thăm dò từ chối ghi đè nếu output directory đã tồn tại. Với checkout này, kết quả đã được lưu ở hai namespace mới phía trên. Không dùng đường dẫn raw thiếu để train lại learner; chỉ fit selector meta-models/calibration từ compact development inputs đã có. Script forecast xác minh hash input trong seal/manifest; script diagnosis kiểm bundle cohort 7 qua D1 audit trước khi đọc. Source hashes và MDP train/validation IDs được lưu.

Đã chạy cả hai runner, đối chiếu tổng hợp với kết quả frozen và kiểm source-hash/fold disjointness. D1 `--check` vẫn PASS. Phạm vi hiện tại là **tìm hướng và chạy probe**, không là claim scientific approval, full ablation, full external benchmark, hay phát hành candidate mới.

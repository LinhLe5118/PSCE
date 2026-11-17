# Thử nghiệm cải thiện OEH — kết quả ngày 12/10/2026

**Kết luận:** đã chạy thử nghiệm có đối chứng và nested validation. Hiệu chỉnh xác suất/kết hợp B2 cải thiện rất nhẹ khi chia theo MDP, nhưng chưa có bằng chứng đủ chắc. Mô hình hai bước kém hơn. Khi loại outcome của một họ môi trường khỏi dữ liệu fit mới, các thay đổi đều giảm chất lượng. **Giữ nguyên phương pháp hiện tại; chưa có candidate mới đủ lý do để thay thế.**

## 1. Đã thử những gì?

| Cấu hình | Cách hiểu |
|---|---|
| Current | Dự báo OEH đã lưu trước outcome, giữ nguyên |
| Temperature | Hiệu chỉnh xác suất để model bớt quá tự tin; fit T trong [0,5;2] trên train fold |
| Blend B2 | Trộn dự báo OEH và B2, học tỷ lệ trộn trong [0;1] trên train fold |
| Joint 7 | Học trực tiếp bảy lớp bằng một multiclass head mới |
| Two-stage | Học “cần >x50 không?”, rồi học x1/x2/x5/x10/x20/x50 ở các trường hợp hữu hạn; ghép thành xác suất bảy lớp |
| Nested selected | Trong từng outer fold, chỉ dùng inner OOF log loss để chọn một trong năm cấu hình trên |

Đã khóa cấu hình, candidate family và 12 contrast metric trước khi fit tại [plan.json](../../artifacts/week8/nested_experiment_2026-10-12/plan.json). Không tìm thêm hyperparameter sau khi thấy bảng này. Thời gian runner thực tế khoảng **41,19 giây**, chỉ gồm các thử nghiệm forecaster từ compact inputs; không là thời gian huấn luyện lại learner offline RL.

Hai head Joint 7/Two-stage dùng **cùng 25 đầu vào**: 18 feature B2 và bảy xác suất OEH đã có trước outcome; fit riêng theo BC/CFQI/IQL, không thêm nhãn họ MDP. XGBoost depth 3, min_child_weight 10, learning_rate 0,05, 200 rounds mỗi submodel, seed cố định. Joint có tối đa 7×200 output trees; Two-stage có 1×200 + 6×200, cùng mức danh nghĩa 1.400 cây. Thời gian, sample count của conditional head và độ phức tạp hiệu dụng vẫn khác nhau.

Các head mới **không** là bản OEH đầy đủ 108 feature được refit trên toàn bộ historical meta-training. So Joint với Two-stage để xem hướng tách hai câu hỏi có giúp dưới thiết lập chung này. So chúng với Current chỉ là so candidate pipeline, không gán mọi chênh lệch cho riêng cấu trúc head. Các input raw/OEH-B group đầy đủ để dựng lại toàn bộ training lịch sử không hiện diện trong clone.

## 2. Thiết kế đánh giá

- Dùng cohort 4 **đã mở**, 84 MDP chính và 1.260 task (5 behavior × 3 learner mỗi MDP). Giữ control ngoài phép thử; không dùng outcome cohort 7 để fit.
- Outer: năm fold theo MDP. Inner: ba fold theo MDP trong outer train. Không MDP nào xuất hiện ở cả train và validation của một split; mọi behavior/learner của nó đi cùng fold.
- Các tham số T/w, head và việc chọn candidate đều chỉ học trong train phù hợp. Scorer vẫn dùng bảy lớp, IBS và log loss gốc; không xóa censored, không đổi NR≥0,95 hoặc nhãn suffix≥8/10.
- Tổng hợp 15 task/MDP rồi trung bình 84 MDP. Báo từng candidate outer-OOF và pipeline được chọn bằng inner validation riêng.
- Bộ stress thứ hai giữ lại cả một họ MDP, fit các sửa đổi bằng outcome của hai họ còn lại, và vẫn chọn candidate bằng inner grouped folds.

**Giới hạn quan trọng:** cohort 4 đã được nghiên cứu nhiều lần trước đây. Nested CV không xóa sự sử dụng lịch sử đó, nên đây là phát triển, không là fresh confirmation. Forecast OEH/B2 gốc cũng đã học các họ generator này ở cohort lịch sử: stress test giữ họ ra ngoài **phần fit/calibration mới**, không là bằng chứng full OEH tổng quát sang một họ hoàn toàn chưa từng học.

## 3. Kết quả khi chia theo MDP

Cả hai chỉ số càng thấp càng tốt. “Giảm” tính tương đối so với Current trên **cùng cohort và split**.

| Phương án | Brier | Log loss | Giảm Brier | Giảm log loss |
|---|---:|---:|---:|---:|
| Current | 0,052905 | 0,642729 | — | — |
| Temperature | 0,052599 | **0,640361** | 0,58% | 0,37% |
| Blend B2 | 0,052761 | 0,640681 | 0,27% | 0,32% |
| Joint 7 | 0,058437 | 0,721906 | **Tệ hơn 10,46%** | **Tệ hơn 12,32%** |
| Two-stage | 0,061925 | 0,734418 | **Tệ hơn 17,05%** | **Tệ hơn 14,27%** |
| Nested selected | **0,052565** | 0,641496 | 0,64% | 0,19% |

Inner selection chọn Temperature ở ba outer folds, Blend ở hai; không chọn Joint/Two-stage. Hai kết quả Temperature/Blend trùng probe trước vì cùng outer split và cùng quy tắc fit tham số: đây không là hai bộ bằng chứng độc lập. Phần mới là kiểm lựa chọn candidate bằng inner folds, hai head mới, family stress và các contrast có cận thăm dò.

Two-stage cũng kém Joint ở cả hai metric. Một giải thích cần kiểm tiếp là ít ví dụ hữu hạn khiến conditional head khó học; bảng này chưa chứng minh đó là nguyên nhân. Không tiếp tục đổi head/threshold cho đến khi bảng trở nên đẹp.

## 4. Cải thiện nhỏ có đủ chắc không?

Đã tính cận một phía 95% max-t đồng thời trên 12 phép so sánh, bootstrap 10.000 lần theo MDP từ **cached OOF scores**. Gain = lỗi baseline − lỗi candidate; cận dương mới là tín hiệu thuận theo phép kiểm này.

| Candidate so với Current | Brier gain | Cận dưới thăm dò | Log loss gain | Cận dưới thăm dò |
|---|---:|---:|---:|---:|
| Temperature | 0,000306 | **−0,000280** | 0,002368 | **−0,003956** |
| Blend B2 | 0,000144 | **−0,000605** | 0,002048 | **−0,004912** |
| Nested selected | 0,000340 | **−0,000384** | 0,001233 | **−0,006074** |

Các cận đều âm: mức giảm lỗi trung bình nhỏ **chưa đủ để gọi là cải thiện chắc chắn**. Temperature giảm lỗi Brier ở 40/84 MDP nhưng tăng lỗi ở 44; log loss giảm ở 33 và tăng ở 51. Lợi ích không đồng đều.

Cận này chỉ mô tả bất định có điều kiện từ loss đã lưu, không refit model trong bootstrap, không bao hết dependence do các CV train folds chồng lấp, và không tính các phép thử lịch sử trước ngày này. Không dùng nó làm gate confirmatory hoặc tuyên bố bảo đảm coverage sau chọn phương pháp.

## 5. Giữ một họ môi trường ra ngoài phần fit mới

| Phương án | Brier | Log loss | Thay đổi so với Current |
|---|---:|---:|---|
| Current | **0,052905** | **0,642729** | Reference |
| Temperature | 0,053323 | 0,650556 | Tệ hơn 0,79% / 1,22% |
| Blend B2 | 0,053644 | 0,649837 | Tệ hơn 1,40% / 1,11% |
| Joint 7 | 0,065352 | 0,799834 | Tệ hơn 23,53% / 24,44% |
| Two-stage | 0,078025 | 0,860566 | Tệ hơn 47,48% / 33,89% |
| Nested selected | 0,053319 | 0,649536 | Tệ hơn 0,78% / 1,06% |

Inner selection lần lượt chọn Blend/Current/Temperature. Các sửa đổi học từ hai họ không chuyển tốt sang họ thứ ba trong cohort này. Kết quả stress làm giảm lý do áp dụng ngay những gain dưới 1% ở bảng grouped CV. Đây là mô tả trên dữ liệu phát triển, không kiểm định xác nhận về mọi distribution shift.

## 6. Quyết định sau thử nghiệm

1. **Giữ Current.** Không thay xác suất đã seal hoặc nâng verdict nghiên cứu từ các thử nghiệm này.
2. **Không chọn Two-stage trong cấu hình vừa thử.** Chưa có bằng chứng hướng này giúp; cả so sánh cùng input với Joint và stress đều âm. Không kết luận mọi mô hình hai bước đều thất bại.
3. **Temperature/Blend chỉ giữ như candidate phát triển.** Trước khi nâng thành phương pháp mới, cần training đa-cohort đầy đủ và cohort chưa mở, khóa model/prediction trước outcome. Không hứa thêm 4% hay một mức tăng bất kỳ.
4. **Ưu tiên tái tạo dữ liệu/training và ablation có kiểm soát.** Hiện chưa biết nhóm feature nào giải thích lợi ích của full OEH. Rà soát nguồn đầy đủ trước khi chạy một refit toàn bộ; giữ thông tin truy cập và compute budget minh bạch.

## 7. Bằng chứng và chạy lại

- [Config đã khóa](../../config/week8_nested_experiment_20261012.json), [runner](../../scripts/analysis/run_nested_improvement_20261012.py).
- [Grouped summary](../../artifacts/week8/nested_experiment_2026-10-12/grouped_nested_summary.csv), [family-held-out summary](../../artifacts/week8/nested_experiment_2026-10-12/family_holdout_nested_summary.csv).
- [Contrasts và cận thăm dò](../../artifacts/week8/nested_experiment_2026-10-12/conditional_contrasts.csv), [subgroup scores](../../artifacts/week8/nested_experiment_2026-10-12/grouped_nested_subgroups.csv).
- [Result: fold IDs, chọn candidate, runtime, versions, artifact hashes](../../artifacts/week8/nested_experiment_2026-10-12/result.json). Plan lưu source/input hashes trước khi fit; runner kiểm chúng còn khớp lúc kết thúc.

```powershell
python -m scripts.analysis.run_nested_improvement_20261012
python -m pytest tests/test_nested_improvement.py -q
python -m scripts.analysis.export_week8_d1 --check
```

Runner từ chối ghi đè output namespace đã tồn tại. Checkout này đã có kết quả hoàn chỉnh; không xóa evidence để chạy lại. Unit checks bao gồm split trùng MDP, all-censored, chỉ một lớp hữu hạn và bootstrap với effect bằng không. Kết quả thí nghiệm này không chỉnh file LaTeX đang mở, manuscript hoặc bộ evidence frozen.

**Kiểm tra sau chạy:** 4 unit checks mới PASS; full suite **166 passed, 22 subtests passed, không skip**, 92,45 giây. Branch-aware coverage của các module OEH core là **95%**; không trình bày đây là coverage toàn bộ runner mới. Kiểm độc lập 9 source/input hashes, tất cả output hashes, 24 inner splits không trùng MDP, 15.120 hàng dự báo và việc chọn candidate đúng inner objective đều PASS. Các score/summary được tái tính từ xác suất đã lưu. D1 `--check` và `git diff --check` PASS; verifier evidence cũ vẫn ghi published_evidence_verified=true và scientific_changes=false. Tests kỹ thuật không thay cho xác nhận hiệu quả khoa học trên cohort chưa mở.

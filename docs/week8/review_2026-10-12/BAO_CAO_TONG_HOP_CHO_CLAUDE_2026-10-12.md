# Báo cáo tổng hợp dự án để Claude đọc và đánh giá

**Ngày tổng hợp:** 12/10/2026.
**Dự án:** Difficulty-Aware Offline Reinforcement Learning.
**Người phụ trách phần D1 tuần 8:** Nguyễn Quốc Việt.
**Thư mục làm việc chính:** `C:\Users\Nguyen Viet\Downloads\Devoted`.

## 1. Kết luận hiện tại, diễn đạt dễ hiểu

Dự án muốn nhìn một lượng dữ liệu ban đầu để dự báo: **một thuật toán học từ dữ liệu có sẵn cần bao nhiêu dữ liệu để đạt kết quả tốt và ổn định?** Một ứng dụng phụ là chọn thuật toán phù hợp khi đã có một mức dữ liệu cố định.

Trong phép thử chính đang lưu, dự báo OEH có lỗi thấp hơn đối chứng B2 cùng quy trình học: giảm khoảng **15,97% ở IBS và 10,22% ở log loss**. Selector hiện tại tốt hơn việc luôn chọn CFQI ở cả return và tỷ lệ thành công. Tuy nhiên, selector chưa vượt phiên bản v5 theo gate phụ; dự án cũng chưa chứng minh full OEH tốt trên benchmark ngoài.

Phần D1 đã có bảng learner/đối chứng/quyền truy cập/cấu hình, công thức metric, bảng kết quả cho paper và công cụ kiểm số liệu. Các cải tiến chạy ngày 12/10 chỉ cho lợi ích trung bình dưới 1% ở một số cấu hình, chưa đủ chắc và kém đi khi giữ một họ môi trường ngoài phần fit mới. **Quyết định hiện tại là giữ phương pháp đã khóa, chưa áp dụng cải tiến mới.**

Không có căn cứ trong gói này để nói “vượt SOTA 4%”. B2 là đối chứng do nhóm xây; giảm lỗi tương đối so với B2 không đồng nghĩa vượt phương pháp tốt nhất toàn lĩnh vực.

## 2. Phạm vi và cách hiểu trạng thái repo

Báo cáo này tổng hợp bằng chứng đang có trong thư mục, trọng tâm là review B1, hoàn thiện D1 và thử nghiệm cải thiện gần nhất. Các công việc lịch sử được nêu theo tài liệu đã lưu; không coi chúng là những phép thử vừa chạy lại.

- Snapshot main được review: `22b4e9f6f513b86a475da259722fdc58d53b8b1f`.
- Git HEAD của thư mục `Devoted` sau chuyển file vẫn là `b87433f8594f7bdce0070d41c176e6daaceb950d`, nhánh `codex/d2-learner-analysis`.
- Nội dung baseline mới và phần D1/thử nghiệm đã được đưa về `Devoted`, nhưng **không reset/merge/commit/push** trong lần chuyển. Vì vậy HEAD cũ không phản ánh đầy đủ nội dung file đang đọc.
- Có thay đổi/file riêng từ trước của người dùng. Claude cần xem working tree; không mặc định mọi diff so với HEAD là phần D1 mới.
- README gốc còn đoạn trạng thái ngày 10/09. Đọc cùng tài liệu Week 8 để tránh coi kết luận của hướng nghiên cứu cũ là verdict hiện tại.

## 3. Bài toán, dữ liệu và môi trường

### 3.1 Offline RL là gì trong dự án này?

Thuật toán học cách chọn hành động từ một bộ ghi chép có sẵn gồm trạng thái, hành động, phần thưởng và trạng thái kế tiếp. Nó không được tự đi thử thêm hành động để thu dữ liệu trong lúc học.

Môi trường nghiên cứu chính là MDP hữu hạn tự sinh: có hữu hạn trạng thái và hành động. Kernel thật được dùng để tạo dữ liệu và chấm policy chính xác; forecaster pilot-only không được nhận kernel thật làm đầu vào. Việc hệ thống đánh giá biết đáp án không có nghĩa mọi thuật toán đều biết đáp án khi học.

### 3.2 Dữ liệu của cohort 7

| Thành phần | Quy mô / ý nghĩa |
|---|---|
| Toàn bộ cohort | 120 môi trường; 120 fingerprint kernel duy nhất |
| Panel chính | 84 môi trường: 30 uniform_garnet, 36 forward_layered, 18 bottleneck |
| Control | 36 môi trường lock; báo riêng |
| Cách thu thập dữ liệu | 5 behavior cho mỗi môi trường |
| Learner tạo nhãn nhu cầu | BC, CFQI, IQL |
| Budget | x1, x2, x5, x10, x20, x50; mức xk là k × số trạng thái × số hành động |
| Lặp lại | 10 stream seed ở mỗi budget |
| Kết quả core | 108.000 job = 120 × 5 × 3 × 6 × 10 |
| Nhãn nhu cầu | 1.800 nhãn toàn cohort; 1.260 nhãn panel chính |

Đọc dễ hiểu: Garnet là môi trường chuyển trạng thái ngẫu nhiên; layered có cấu trúc đi qua các tầng; bottleneck có vùng nút thắt; lock là nhóm đối chứng riêng. Đây là mô tả khái quát, cần đọc generator để biết chính xác cấu trúc và tham số.

Trong panel chính, **900/1.260 nhãn, tức 71,43%, thuộc >x50**. Nghĩa là chưa đạt điều kiện ổn định trong miền budget đã thử; không có nghĩa chắc chắn không thể học nếu tăng dữ liệu tiếp.

Mã runtime đã khôi phục 204 file từ nguồn trong Git và đối chiếu hash. Tuy nhiên vẫn thiếu bốn artifact lịch sử, hai target dataset seal và raw buffers. **Kiểm được nhãn/score từ bảng gọn không đồng nghĩa đã tái tạo toàn bộ quá trình sinh dữ liệu và train từ raw.**

Nguồn: [review B1](outputs/review_week8_2026-10-11/DANH_GIA_B1.md), [hồ sơ dữ liệu](docs/data/DATA_WEEK8.md), [audit B1](scripts/analysis/audit_week8_data.py), [restore runtime](scripts/restore_frozen_runtime.py).

### 3.3 “Đủ dữ liệu” được định nghĩa thế nào?

Return được chuẩn hóa bằng `NR = (J_policy − J_random)/(J_optimal − J_random)`; NR không phải xác suất và không bị ép vào [0,1]. Một seed thành công nếu **NR ≥ 0,95**.

Một budget chỉ được gán là đủ nếu ít nhất **8/10 seed thành công ở budget đó và ở tất cả budget lớn hơn đã thử**. Ví dụ số seed thành công lần lượt là `[2,5,8,7,9,10]`: nhãn là x20, vì mức x5 đạt 8 seed nhưng x10 lại tụt xuống 7. Không có mức thỏa thì gán >x50.

Forecaster xuất bảy xác suất tương ứng bảy lớp này. “>x50” là lớp chưa đạt trong miền thử, không phải technical failure và cũng không phải selector từ chối chọn thuật toán.

## 4. Những phần đã làm và sản phẩm hiện có

| Giai đoạn / hạng mục | Đã có gì | Giới hạn cần giữ |
|---|---|---|
| Hướng OEH lịch sử | README và hồ sơ Week 2 lưu kết quả âm, vấn đề tiêu chí suy biến, positive control và kiểm khả thi | Không xóa kết quả âm; không trộn estimator lịch sử với forecaster hiện tại |
| D1 Week 6 | Provenance/cấu hình 7 learner; 825 development buffer và smoke-fit 7/7 theo hồ sơ bàn giao | Smoke-fit không phải đánh giá hiệu quả confirmatory |
| Review B1 Week 8 | Kiểm số job, nhãn, threshold, fingerprint, hash; phục hồi runtime | Guard audit còn thiếu điều kiện đầy đủ; raw reproduction chưa hoàn tất |
| D1 Week 8 | Matrix learner/comparator/metric/access; đủ B2 18 feature; bảng primary và bảng LaTeX; exporter kiểm số học | Chưa thay review/approval khoa học của Hưng/Linh |
| Chẩn đoán cải thiện | Phân tích lỗi theo họ, learner, lớp; thử selector và calibration | Chẩn đoán cohort 7 đã mở chỉ dùng đặt giả thuyết |
| Thử nghiệm nested | 5 candidate và quy trình chọn bên trong inner folds; stress theo họ; lưu cả kết quả âm | Dữ liệu phát triển đã mở, không phải fresh confirmation |
| Tài liệu trình bày | File LaTeX xem thử kèm note cải thiện, compiler tích hợp đã xác nhận success | Chưa tương đương tích hợp fragment vào manuscript tám trang |
| Chuẩn hóa nơi lưu | 1.253 file được kiểm/copy về Devoted; bảo vệ thay đổi cũ; Python environment chuyển vào Devoted | Git history không cập nhật theo file snapshot |

Phần D1 đã hoàn thành theo yêu cầu được xác nhận. Chưa có bàn giao M1 mới để nghiệm thu bảng E1 mới; không coi B2/E2 hoặc ablation/full benchmark ngoài là đã hoàn thành chỉ vì D1 đã xong.

## 5. Learner, đối chứng và quyền truy cập

Ba learner BC/CFQI/IQL được dùng để tạo nhãn nhu cầu ở cả sáu budget. BCQ/CQL/SPIBB/VI-LCB bổ sung vào ứng dụng chọn learner ở x10; **không có nhãn sáu-budget cho bốn learner này trong phép thử hiện tại**.

| Learner | Cách hiểu | Implementation / lưu ý |
|---|---|---|
| BC | Bắt chước hành động trong dữ liệu | Tabular của project |
| CFQI | Học giá trị, chọn trong hành động đã quan sát | Reference tabular của project; không phải neural FQI benchmark |
| IQL | Học giá trị bằng expectile, ưu tiên hành động có lợi thế | Tabular adaptation; không phải toàn bộ benchmark tác giả |
| BCQ | Hạn chế hành động ít được dữ liệu hỗ trợ | Author discrete BCQ, one-hot state, 500 update |
| CQL | Phạt ước lượng quá lạc quan ngoài dữ liệu | d3rlpy DiscreteCQL đã pin, 500 update |
| SPIBB | Giữ hành vi nền nơi dữ liệu yếu | Author Pi-b core, **nhận true logging policy beta** |
| VI-LCB | Trừ mức phạt bất định khi học giá trị | Project port theo paper-spec, c_b=0,05; có terminal metadata |

Các learner không có cùng số phép tối ưu, cost hoặc information budget. Selector còn dùng **nhãn họ MDP**, pilot features và forecast CFQI. Oracle selection dùng outcome thật, chỉ là mốc trên, không triển khai từ pilot.

Forecaster phải tách bốn nhóm: **matched pilot-only**, **AFT-only**, **frequency**, **privileged**. Matched dùng cùng quy trình AFT + direct-threshold + multiclass depth 3, 200 rounds, nhưng từng hàng có model/weights được fit riêng. OEH có 108 cột; B2 có 18 thống kê pilot. ERI/BWD là adaptation, không được gọi là tái lập benchmark gốc. Privileged có thông tin MDP/reference data không sẵn khi triển khai từ pilot.

Nguồn đủ cấu hình, pin, training budget, 18 feature và công thức: [D1 matrix](docs/week8/D1_LEARNER_COMPARATOR_METRICS_2026-10-11.md), [config matrix](config/week8_d1_matrix.json), [learner CSV](artifacts/week8/d1_2026-10-11/learner_matrix.csv).

## 6. Kết quả đã khóa của phương pháp hiện tại

### 6.1 Dự báo nhu cầu dữ liệu — cohort 7, panel chính

IBS đo sai lệch xác suất “đủ dữ liệu trước từng ngưỡng”; log loss phạt mạnh khi tự tin vào lớp sai. **Cả hai càng thấp càng tốt.** Trung bình 15 task trong mỗi MDP rồi trung bình 84 MDP; không coi mọi hàng là môi trường độc lập.

| Forecaster matched pilot-only | IBS | Log loss |
|---|---:|---:|
| B2 matched | 0,054727 | 0,627179 |
| B2 + ERI adaptation | 0,056672 | 0,632970 |
| B2 + BWD adaptation | 0,053954 | 0,621165 |
| OEH hiện tại | **0,045989** | **0,563079** |

So B2 matched, giảm lỗi tương đối **15,97% / 10,22%**. Artifact ghi 22 cận dưới đăng ký dương; exporter đọc/đối chiếu những cận đã công bố, không bootstrap lại để cấp approval mới.

OEH chưa thắng mọi subgroup: ở bottleneck, lỗi cao hơn B2 khoảng 3,29% IBS và 2,28% log loss. Đây là chẩn đoán sau outcome, chưa phải kiểm định subgroup đã đăng ký. Những task đạt trong x1–x50 chiếm 28,57% nhưng tạo 73,74% tổng IBS và 81,39% tổng log loss: khó khăn còn lại tập trung vào đoán mức hữu hạn cụ thể.

### 6.2 Chọn learner tại x10

| Quy tắc | Mean NR | Seed success |
|---|---:|---:|
| Luôn dùng CFQI | 0,792729 | 41,29% |
| Selector OEH hiện tại | 0,811385 | 47,95% |
| Router v5 | 0,813202 | 47,33% |
| Oracle chọn theo outcome | 0,837881 | 53,50% |

So CFQI, selector tăng NR khoảng **0,018656** và success **6,67 điểm phần trăm**; hai cận đăng ký dương. So v5, NR giảm khoảng **0,001817**, success tăng **0,62 điểm phần trăm**; gate phụ vẫn **NO-GO**. Không viết là selector đã thắng mọi baseline.

Nguồn: [primary table](artifacts/week8/d1_2026-10-11/primary_forecast.md), [selection CSV](artifacts/week8/d1_2026-10-11/selection.csv), [validation](artifacts/week8/d1_2026-10-11/validation.json).

## 7. Đã thử cải thiện gì, và có hiệu quả không?

Thử trên **cohort 4 đã mở**, cùng 84 MDP chính/1.260 task. Không dùng outcome cohort 7 để fit. Outer 5 folds theo MDP; inner 3 folds để chọn candidate. Toàn bộ behavior/learner của cùng MDP phải đi cùng fold. Khóa candidate/config/12 contrast trước lần fit này.

| Phương án | Ý tưởng | IBS | Log loss | So Current trên cùng cohort |
|---|---|---:|---:|---|
| Current | Giữ forecast đã có | 0,052905 | 0,642729 | Mốc so sánh |
| Temperature | Giảm mức tự tin quá mạnh | 0,052599 | 0,640361 | Giảm lỗi 0,58% / 0,37% |
| Blend B2 | Trộn OEH với B2 | 0,052761 | 0,640681 | Giảm lỗi 0,27% / 0,32% |
| Joint 7 | Học lại trực tiếp 7 lớp | 0,058437 | 0,721906 | Tệ hơn 10,46% / 12,32% |
| Two-stage | Học >x50 trước, rồi mức hữu hạn | 0,061925 | 0,734418 | Tệ hơn 17,05% / 14,27% |
| Nested selected | Inner folds chọn candidate | 0,052565 | 0,641496 | Giảm lỗi 0,64% / 0,19% |

Cận dưới thăm dò của gain Temperature/Blend/Nested selected đều âm. Vì vậy giảm lỗi trung bình nhỏ chưa đủ để gọi là cải thiện chắc chắn. Khi giữ một họ MDP ngoài phần fit mới, mọi candidate đều tệ hơn Current; Nested selected tệ hơn khoảng 0,78% IBS và 1,06% log loss.

Hai head Joint/Two-stage dùng cùng **25 đầu vào = 18 feature B2 + 7 xác suất OEH**. Chúng không phải full OEH 108 feature được refit trên toàn bộ historical meta-training. So với Current không cô lập riêng tác dụng đổi head. Two-stage kém Joint trong cấu hình đã thử; chưa kết luận mọi thiết kế hai bước đều thất bại.

Các probe selector học paired advantage/thêm điều kiện chỉ đổi khi cả return và success dự báo tăng cũng chưa thắng đồng thời hai endpoint. Một rule tăng success khoảng 0,14 điểm phần trăm nhưng giảm NR khoảng 0,01329 trên development OOF. Chưa chọn rule thay selector đã khóa.

**Giới hạn:** cohort 4 đã dùng nhiều lần; nested CV không xóa lịch sử xem dữ liệu. Stress chỉ giữ họ ngoài phần fit/calibration mới; forecast gốc đã học các họ này ở cohort lịch sử. Bootstrap 10.000 lần từ cached OOF losses không refit model và không bao hết phụ thuộc giữa train folds hoặc mọi phép thử lịch sử. Hai probe calibration trước và bảng nested dùng cùng outer split, không phải hai xác nhận độc lập.

Nguồn: [kết quả nested](docs/week8/NESTED_EXPERIMENT_RESULTS_2026-10-12.md), [config](config/week8_nested_experiment_20261012.json), [runner](scripts/analysis/run_nested_improvement_20261012.py), [bằng chứng](artifacts/week8/nested_experiment_2026-10-12/), [probe và chẩn đoán](docs/week8/IMPROVEMENT_REVIEW_2026-10-12.md).

## 8. Chi phí và kiểm thử

Runner nested mất khoảng **41,19 giây**, XGBoost một luồng CPU, không GPU; 11 output khoảng 4,5 MB. Không train lại learner offline RL hoặc sinh raw data; RAM cao nhất chưa đo. Đây không phải cost tái tạo toàn dự án.

Cost core x10 đo callback load/fit/exact scoring, không riêng fit: BC khoảng 0,011261 s/job; CFQI 0,030134; IQL 0,059505. Có 4.200 record/learner trong primary. Chưa có đầy đủ individual fit/VRAM records cho bốn learner bổ sung. Chưa đủ để claim tiết kiệm compute end-to-end. Pilot đã tốn x10 nên dự báo x1/x2/x5 mang ý nghĩa hồi cứu trên stream train độc lập, cần giải thích rõ khi nói về tiết kiệm thu thập dữ liệu.

| Kiểm tra | Kết quả / phạm vi |
|---|---|
| D1 exporter | PASS: 25.200 forecast row, 1.680 MDP score row, 28 summary row, 22 contrast, 420 selection group |
| B1 audit sau restore | 57/61 mục hash hiện diện khớp, không mismatch; không tái lập raw end-to-end |
| Full suite sau nested, trước chuyển thư mục | 166 passed, 22 subtests passed, không skip; 92,45 giây |
| Branch-aware core coverage | 95%; không phải coverage toàn runner mới |
| Chạy lại trực tiếp tại Devoted sau chuyển | 11 test D1/nested PASS; D1 check và git diff --check PASS |
| Evidence verifier | published_evidence_verified=true, scientific_changes=false; verifier bundle lịch sử không thay verdict cohort 7 |
| LaTeX preview | Compiler tích hợp success; không thay xác nhận build manuscript đầy đủ |

Tests chứng minh các kiểm tra kỹ thuật hoạt động, không chứng minh hiệu quả khoa học. CI mới có bước D1 check được cấu hình local; chưa nói remote CI đã chạy thay đổi chưa push.

## 9. Các điểm chưa giải quyết và việc nên làm tiếp

1. Khôi phục raw buffers, dataset seal và artifact lịch sử còn thiếu; tái lập full training trước claim reproduction đầy đủ.
2. Bổ sung guard B1 về tổng số hàng, khóa, đủ 10 seed và bằng chứng không trùng cohort; script hiện có thể PASS một số báo cáo thiếu đầy đủ. Dữ liệu hiện tại đã kiểm riêng khớp, nên đây là rủi ro guard chứ không chứng minh dữ liệu hiện tại sai.
3. Làm ablation có kiểm soát cho 108 feature để biết phần nào đóng góp; chưa gán lợi ích cho một nhóm feature riêng.
4. Kiểm công bằng về input/access/capacity/tuning/cost; true beta của SPIBB và family label của selector phải hiện rõ.
5. Temperature/Blend chỉ giữ làm candidate. Nếu tiếp tục: phát triển đa-cohort, khóa model/prediction trước outcome cohort chưa mở, giữ metric/threshold/lớp >x50 và báo cả kết quả âm.
6. Kiểm selector đồng thời return và success so cả CFQI lẫn v5; không chọn rule bằng subgroup đã xem rồi gọi là xác nhận độc lập.
7. Tìm benchmark ngoài phù hợp. FourRooms hiện chỉ có component stress **NO-GO**, chưa có full-method external validation. Không đổi cách gọi cohort tự sinh thành benchmark ngoài.
8. Hưng/Linh review giả định, thống kê và tích hợp manuscript; D1 không ký thay approval của họ.

## 10. Những file Claude nên đọc trước

Đường dẫn dưới đây tương đối với thư mục `Devoted`; chỉ cần mở thư mục này, không cần dùng clone review khác.

1. Báo cáo này, sau đó `docs/week8/D1_HANDOVER_2026-10-11.md`.
2. `docs/week8/D1_LEARNER_COMPARATOR_METRICS_2026-10-11.md` và `config/week8_d1_matrix.json`.
3. `scripts/analysis/export_week8_d1.py`, `tests/test_week8_d1.py` và `artifacts/week8/d1_2026-10-11/validation.json`.
4. `docs/week8/NESTED_EXPERIMENT_RESULTS_2026-10-12.md`, config/runner nested và `tests/test_nested_improvement.py`.
5. Các CSV summary/contrast, `plan.json` và `result.json` trong `artifacts/week8/nested_experiment_2026-10-12/`.
6. `docs/week8/IMPROVEMENT_REVIEW_2026-10-12.md`, hai runner diagnose/probe và artifact của chúng.
7. `docs/data/DATA_WEEK8.md`, `scripts/analysis/audit_week8_data.py` và review B1 trong `outputs/review_week8_2026-10-11/`.
8. Khi đánh giá toàn bộ lịch sử: `README.md`, hồ sơ Week 2/Week 6 và `paper/`; phân biệt ngày, protocol và phiên bản phương pháp.

Chạy kiểm tra an toàn, không ghi đè evidence:

```powershell
Set-Location -LiteralPath 'C:\Users\Nguyen Viet\Downloads\Devoted'
& '.\.venv-restore\Scripts\python.exe' -m scripts.analysis.export_week8_d1 --check
& '.\.venv-restore\Scripts\python.exe' -m pytest tests/test_week8_d1.py tests/test_nested_improvement.py -q
```

Không chạy runner tạo kết quả vào namespace đã có để “làm mới bảng”: runner từ chối ghi đè. Nếu cần thử tiếp, phải xác định thiết kế/output mới và lưu cả lịch sử tìm kiếm.

## 11. Đề nghị gửi cho Claude

> Hãy đọc báo cáo này và kiểm chứng bằng code/config/artifact trong Devoted. Đánh giá độc lập, không mặc định kết luận báo cáo đúng. Trước hết xem working tree vì Git HEAD cũ và có file riêng từ trước. Hãy chỉ ra: (1) phần D1 đã đáp ứng yêu cầu nào, còn thiếu gì; (2) số học/metric/aggregation và split có đúng không; (3) có leakage, test-set reuse, selection bias hoặc diễn giải quá mức không; (4) đối chứng có công bằng về head, feature, quyền truy cập và train budget không; (5) kết luận giữ Current có hợp lý không; (6) có thể claim gì trong paper và claim nào phải bỏ; (7) ba hướng cải thiện tiếp theo theo mức ưu tiên và chi phí. Mỗi nhận xét cần file/dòng hoặc artifact cụ thể, phân biệt lỗi đã xác minh với nghi ngờ. Giữ cả kết quả âm, không sửa frozen evidence hoặc tự chọn cấu hình đẹp sau outcome. Chỉ review và đề xuất trước; nêu rõ kiểm nào bạn đã chạy, kiểm nào chưa chạy.

## 12. Tình trạng thư mục và sao lưu

Mọi sản phẩm hiện tại ở Devoted; Python environment ở `.venv-restore`. Bản trước của các file được cập nhật nằm trong `outputs/workspace_migration_2026-10-12/before/`; file khác của các clone cũ được giữ tại `outputs/review_cleanup_2026-10-12/preserved/`.

Lệnh xóa các thư mục review đã bị kiểm duyệt tự động chặn với `blocked by policy`; log ghi **0 thư mục đã xóa**. Các thư mục ngoài không phải nguồn làm việc hiện hành. Không cần đọc backup như một tập evidence độc lập hoặc trộn kết quả của nhiều snapshot.

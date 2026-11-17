# Chẩn đoán kết quả BRIDGE full OEH, sau khi mở nhãn

**Trạng thái:** phân tích khám phá sau khi có kết quả. Không dùng tài liệu này để đổi phương pháp rồi tuyên bố xác nhận lại trên cùng 5 MDP. Nguồn gốc là [giao thức khóa trước nhãn](../../benchmarks/bridge/fresh_panel_2026-10-08/protocol.json), [prelabel seal](../../benchmarks/bridge/fresh_panel_2026-10-08/prelabel_seal.json), [75 nhãn](../../artifacts/week8/bridge_fresh_test_labels_2026-10-08/labels.csv), [300 hàng đã chấm](../../artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08/scored.csv) và [khoảng bootstrap theo 5 MDP](../../artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08/paired_task_bootstrap.csv). BRIDGE cung cấp MDP công bố; nhóm tự sinh pilot và nhãn.

## 1. Kết quả chính, không gộp learner

| Learner | B2 Brier / log loss | OEH Brier / log loss | XGBoost-CF/18 Brier / log loss | OEH − B2 Brier, khoảng mô tả 95% |
|---|---:|---:|---:|---:|
| BC | 0.059865 / 0.497956 | 0.028178 / 0.382242 | 0.045579 / 2.277528 | −0.031687 [−0.083220,+0.009082] |
| CFQI | 0.296849 / 2.099780 | 0.284121 / 2.111864 | 0.292000 / 3.924963 | −0.012728 [−0.071202,+0.035563] |
| IQL | 0.049781 / 0.324704 | 0.112798 / 0.819092 | 0.071166 / 0.358160 | +0.063018 [+0.004222,+0.138822] |

XGBoost-CF/108 dùng chính 108 đầu vào OEH, nên là đối chứng kiến trúc, không phải phương pháp tác giả độc lập. Nó đạt CFQI Brier 0.261571, thấp hơn OEH 0.284121; chênh OEH trừ đối chứng là +0.022550 [+0.003642,+0.052617]. CFQI log loss lại tốt hơn với OEH (2.111864 so với 2.920034). BC bị censored 23/25, CFQI 12/25, IQL 21/25. Khoảng 95% resample 5 MDP, giữ chung 5 behavior, là mô tả riêng từng phép so sánh, không phải cận đồng thời hay bằng chứng SOTA.

## 2. IQL: bốn nhãn thành công sớm giải thích phần lớn mức thua

Trong bốn hàng IQL có nhãn trước `>x50`, OEH gán trung bình 0.660 cho lớp censored, so với 0.377 của B2. Brier trung bình trên bốn hàng này là 0.622 của OEH so với 0.253 của B2; log loss là 4.530 so với 1.603. Bốn hàng này đóng góp **+0.059098** trong tổng chênh Brier **+0.063018** OEH−B2, tức 93.8% mức thua. Trên 21 hàng censored còn lại, chênh chỉ +0.003919. Như vậy lỗi nằm ở phân biệt những ngoại lệ học nhanh, không chủ yếu ở tỷ lệ censored chung.

| MDP IQL | OEH Brier | B2 Brier | Chênh OEH−B2 | Nhãn sớm |
|---|---:|---:|---:|---:|
| Atlantis horizon 10 | 0.21799 | 0.12869 | +0.08930 | 2/5 |
| Atlantis horizon 20 | 0.00391 | 0.00749 | −0.00358 | 0/5 |
| Breakout horizon 10 | 0.00042 | 0.00004 | +0.00038 | 0/5 |
| Freeway horizon 10 | 0.30626 | 0.10479 | +0.20147 | 2/5 |
| Qbert horizon 10 | 0.03541 | 0.00790 | +0.02751 | 0/5 |

Freeway và Atlantis horizon 10 đóng góp khoảng 92% chênh Brier trung bình. Bốn nhãn sớm có chuỗi thành công `7/9/10/10/10/10`, `9/9/10/10/10/10`, `9/10/10/10/10/10` và `9/10/10/10/10/10`: chúng vượt quy tắc sustained 8/10 trên phần suffix thích hợp, nên đây không phải một thành công đơn lẻ ở một budget. Ví dụ Freeway/`shifted_dynamics` có nhãn `x1`; OEH cho lớp đúng xác suất 0.0022 và `>x50` 0.8338, trong khi B2 cho lớp đúng 0.3182 và `>x50` 0.2949. Log loss hàng đó là 6.128 so với 1.145.

## 3. Mô phỏng posterior và chuyển phân phối đầu vào

Các [checkpoint OEH-B](../../artifacts/week8/bridge_fresh_full_oehb_preoutcome_2026-10-08/manifest.json) có đủ 64 draw trên 25 pilot. **Cả bốn hàng IQL học sớm nói trên đều có 64/64 draw dự báo censored**; ước lượng KT cho `>x50` là 0.9556 ở mỗi hàng. Trong 3.600 nhóm pilot của 7 cohort meta-training, mean raw OEH-B `p(>x50)` của IQL là 0.934, trung vị 0.956 và 89.0% nhóm có `p≥0.95`; nghĩa là độ lệch này đã hiện diện trong thành phần raw, nhưng mô hình fitted trong họ cũ vẫn học được một phần hiệu chỉnh. Trên BRIDGE, cả OEH lẫn XGBoost-CF/108 có Brier IQL tệ hơn B2/XGBoost-CF/18, phù hợp với giả thuyết rằng **khối đặc trưng OEH chuyển kém cho các trường hợp IQL học nhanh**. Đây là tương quan hậu nghiệm; chưa cô lập được OEH-B, OEH-H, mô hình posterior, hay head là nguyên nhân duy nhất.

Các đầu vào cũng lệch phạm vi rõ rệt: từ 10.800 hàng meta-training, `log(S×A)` nằm trong **[2.4849, 5.5452]**, còn cả 75 hàng BRIDGE nằm trong **[5.7170, 8.0398]**. Tức mọi task BRIDGE lớn hơn mức kích thước state-action lớn nhất từng thấy khi fit. Trung vị phần state-action quan sát trong pilot là 0.3257 trên BRIDGE so với 0.6562 ở training. Lượt BRIDGE còn đổi từ horizon 100 và `gamma=0.95` của panel nội bộ sang horizon 10/20, sticky action 25%, reward scaling và `gamma=0.99`. Đây là bằng chứng trực tiếp về chuyển phân phối và là cơ chế hợp lý khiến tín hiệu posterior/hoạch định ngoại suy sai; không thể quy mức thua cho một thay đổi riêng lẻ từ dữ liệu hiện có. Reward scaling giữ nguyên normalized return của **một policy cố định**, nhưng IQL có inverse temperature cố định 3.0 nên policy sau huấn luyện có thể đổi; giao thức BRIDGE đã định nghĩa nhãn trên reward đã scale và cả dự báo dùng cùng adapter.

## 4. BC, CFQI và log loss

BC có 23/25 nhãn censored. OEH cải thiện Brier so với B2 nhiều nhất ở Freeway (chênh theo task −0.1279) và Atlantis horizon 10 (−0.0533), nhưng kém ở Qbert (+0.0209). Với hai nhãn BC học sớm, xác suất censored trung bình của OEH là 0.633, B2 là 0.908; với 23 nhãn censored, cả hai khoảng 0.93. Điểm trung bình có lợi, nhưng khoảng ghép cặp theo 5 task vẫn cắt 0.

CFQI có Brier OEH thấp hơn B2 4.3% về điểm trung bình, chủ yếu nhờ Freeway (chênh theo task −0.1202); Atlantis horizon 10 làm ngược lại (+0.0648). Log loss OEH lại nhỉnh hơn B2 0.6%. Hai hàng Atlantis horizon 10 có nhãn `x1`: OEH chỉ gán xác suất 0.00366/0.00442 cho lớp đúng, B2 gán 0.09373/0.08085, tạo thêm 3.24/2.91 đơn vị log loss ở hai hàng. Mọi forecaster đều quá nghiêng về censored cho CFQI: tỷ lệ nhãn censored thật là 0.48, xác suất censored trung bình từ bốn forecaster là khoảng 0.824–0.850. Đây là sai lệch hiệu chuẩn trên tập chuyển môi trường, không phải bằng chứng riêng OEH sai.

XGBoost-CF/18 có hai hàng BC và hai hàng CFQI gán đúng lớp quan sát xác suất bằng 0; scorer khóa giá trị ở `1e-12`, nên mỗi hàng chịu log loss 27.631. Vì vậy chênh log loss lớn so với nhánh này phản ánh cả dự báo quá tự tin của đối chứng; không nên trình bày tỷ lệ phần trăm cải thiện log loss mà không nêu bốn hàng đó. Không sửa smoothing trên test đã mở rồi gọi đó là kiểm định độc lập mới.

## 5. Đối chiếu score nguyên bản là câu hỏi khác

Ở [bảng rank ghép cặp](../../artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08/original_rank_paired_transfer.csv), tương quan Spearman với thứ tự nhu cầu dữ liệu dễ hơn của TQ nguyên bản là 0.470/0.635/0.632 cho BC/CFQI/IQL, còn kỳ vọng sự kiện từ OEH là 0.429/0.596/0.501 trên cùng 25 hàng mỗi learner. Đây là điểm rank có lợi cho TQ, nhất là IQL; các khoảng theo chỉ năm task không chứng minh một thứ tự tổng quát. TQ dùng thông tin MDP thật và DQN reference, còn OEH chỉ dùng pilot. TQ/ERI/EH/SACo cho score vô hướng, không cho phân phối bảy lớp, nên không lấy chênh rank làm kết luận Brier hay “vượt SOTA” cho bài toán dự báo chính.

Trên [84 MDP stochastic đã mở](../../artifacts/week8/published_xgboost_cf_120_exploratory_2026-10-08/summary.csv), OEH từng có Brier thấp hơn XGBoost-CF/18 ở cả BC/CFQI/IQL, nhưng phép gắn đối chứng công bố vào panel đó được làm sau khi nhãn đã mở nên là phân tích khám phá. Kết quả BRIDGE khác chiều ở IQL và không cho phép suy rộng lợi thế của panel sinh nội bộ sang họ Atari mới.

## 6. Điều gì đã kiểm và điều gì chưa xác định

Không thấy dấu hiệu lỗi join/missing rows: scorer kiểm đủ 300 dự báo, 75 nhãn, tổng xác suất, key duy nhất và các SHA-256 trước khi chấm. Mã OEH-B dùng `gamma`/horizon từ protocol BRIDGE và cùng ba learner với hằng số IQL expectile 0.7, inverse temperature 3.0, 100 vòng lặp như runner nhãn; evaluator đều dùng normalized finite-horizon return. Điều này loại bớt lỗi giao thức hiển nhiên nhưng không chứng minh mọi chi tiết mô phỏng và posterior đúng trên mọi MDP.

Cần một phân tích development tách OEH-B, OEH-H, size/coverage và head trên dữ liệu được phép phát triển, cùng kiểm tra feature lineage và parity theo task. Nếu điều chỉnh phương pháp hoặc hiệu chuẩn để xử lý IQL, phải khóa trước **một panel môi trường khác chưa mở nhãn**. Không tái dùng năm task BRIDGE hiện tại để chứng nhận cải thiện. Kết quả hiện tại không cho phép tuyên bố thắng phương pháp tác giả hay SOTA trên difficulty-aware offline RL.

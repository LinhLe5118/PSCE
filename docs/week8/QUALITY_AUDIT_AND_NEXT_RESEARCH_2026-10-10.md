# Kiểm tra chất lượng và hướng nghiên cứu tiếp — 10/10/2026

## Trạng thái quan sát được

- Cohort nội bộ 120 MDP đã hoàn tất; 84 MDP stochastic là panel chính. OEH đạt integrated Brier 0.045989 và seven-class log loss 0.563079, so với B2 matched nội bộ 0.054727 và 0.627179; hai cận dưới đồng thời của cải thiện đều dương. Kết luận hợp lệ: hơn đối chứng đặc trưng **nội bộ** trên các họ generator đã nghiên cứu.
- BRIDGE Atari 5 MDP đã hoàn tất với full OEH và ba đối chứng cùng đầu ra. BC/CFQI có Brier point estimate thuận lợi so với XGBoost-CF/18 thích nghi nhưng khoảng paired MDP mô tả cắt 0; IQL có Brier kém hơn với khoảng không cắt 0. Không có bằng chứng thắng đồng đều ở môi trường ngoài.
- BRIDGE prospective 5 MDP: OEH-H 25/25, OEH-B 0/25, pilot 25/25, original scalar scores 5/5, labels 0/25; chưa có seal hay kết quả. 14 MiniGrid trước đó đã mở outcome: full OEH-B 35/70 checkpoint, phân tích sau này bắt buộc ghi exploratory. Không tính 24 BRIDGE MDP như 24 kiểm định mới độc lập.
- Lúc 01:51, Python child OEH-B PID 31260 vẫn tăng CPU và dùng khoảng 4.6 GiB RAM, nhưng Python parent PID 12444 đã mất. Health watcher đã được sửa để báo orphan; một watcher phục hồi chỉ tác động nếu child đứng CPU ít nhất sáu lần kiểm tra 60 giây, nhằm giữ lại phép tính và checkpoint hiện tại. Kiểm tra trạng thái phục hồi trước khi diễn giải chữ `running` là tiến độ thật.
- Paper hiện build 8 trang với một figure pipeline và 25 mục BibTeX; thesis DOCX được Word phân trang thực tế **34 trang**, khoảng 11,845 từ. Chúng vẫn chưa chứa hai phân tích BRIDGE đang chạy. Tên tác giả, cơ quan và thông tin nộp vẫn là placeholder. Các artifact tuần 8 và phần lớn thay đổi paper chưa được commit/push ở working tree hiện tại.

## Đối chiếu với góp ý GVHD

| Yêu cầu | Đánh giá | Việc cần khóa |
|---|---|---|
| Vùng bài toán, thuật ngữ, biến, difficulty, đầu ra | Đã rõ trong Problem/Data và T1 | Khi trình bày, nói `P(E_l \mid D_0,l)` và bảy lớp trước tên OEH; `>x50` là right censoring. |
| Data tự sinh như thế nào, tại sao, thiên lệch | Đã có protocol, seed/stream và hình pipeline; còn giới hạn ngoại suy | Đặt bảng họ MDP, hành vi, lọc kernel và điều kiện tiếp nhận cạnh figure; nêu 5/155 BRIDGE mới phù hợp theo screen, không gọi đại diện cho cả bộ. |
| Vì sao dùng BC/CFQI/IQL và tại sao có bảy learner | Đã tách requirement chính và selector phụ | Giữ return/success của selector sau bảng Brier/log loss, không dùng nó làm chứng cứ dự báo độ khó. |
| Cơ sở đối chứng tác giả khác | Chưa đủ cho claim vượt SOTA | Original ERI/TQ/SACo/EH chỉ là scalar rank diagnostics; XGBoost-CF/18 là cách nhóm thích nghi **công thức CDF survival**, không phải nguyên hệ thống tác giả chạy trên offline RL. B2 là đối chứng nội bộ. |
| Ablation và môi trường ngoài | Còn yếu | Ablation trên nhãn 120 đã mở là exploratory; Atari 5 cho kết quả pha trộn; chờ prospective 5 và 14 opened trước khi chốt bản thảo. |
| Metrics và minh chứng | Có cùng bảy lớp, paired MDP CI, censoring và manifest | Thêm calibration theo budget, early-success discrimination, chi phí từ pilot đến quyết định, và thất bại/NO-GO nguyên trạng. |

## Đánh giá mức mục tiêu

**Đồ án xuất sắc: khả thi nhưng chưa tự động đạt.** Bản thảo 34 trang và hệ thống thử nghiệm có giá trị kỹ thuật; trình bày trung thực kết quả âm là điểm mạnh. Hội đồng vẫn có thể hỏi về tính hữu ích sau khi đã trả chi phí pilot x10 và 64-draw posterior, tính khái quát ngoài generator, và vì sao không dùng một mô hình dự báo đơn giản hơn. Cần demo tái lập, audit thuật ngữ/nguồn, chi phí tổng, và kết quả BRIDGE hoàn chỉnh. Điểm thực tế phụ thuộc rubric của trường và phản biện.

**Paper ở venue AI/RL chọn lọc hạng A: chưa đủ bằng chứng để hứa.** Đóng góp mới có thể là dự báo phân phối nhu cầu dữ liệu *theo learner*, nhưng chưa thể tuyên bố `first` hay SOTA. So sánh với đối chứng nội bộ trên họ sinh quen thuộc không đủ để xác lập vị thế; 5 MDP Atari cho kết quả hỗn hợp, 5 prospective còn chờ, 14 MiniGrid là post-outcome. 25 references và bản PDF build được không thay thế được đối chứng mạnh cùng endpoint, kiểm định ngoài nhiều task độc lập, ablation cơ chế, utility/compute và đường tái lập công khai. NeurIPS nêu rõ tính chất lượng, rõ ràng, ý nghĩa, nguyên bản, minh bạch và tái lập là tiêu chí đánh giá; không suy ra xác suất nhận bài từ bảng điểm hiện có.

## Thứ tự nghiên cứu tiếp theo

1. **Đóng băng và hoàn tất thí nghiệm đang chạy.** Không sửa OEH/đối chứng hoặc chọn lại BRIDGE theo outcome. Kiểm tra đủ checkpoint, seal, 75 nhãn, hai analysis manifest và strict four-panel audit. Xuất mọi learner, Brier, log loss, paired MDP CI, censoring, rank diagnostics có quyền truy cập, cùng failure report nếu có. Không pool 84 + 5 + 5 + 14 để tạo một cận dương giả.
2. **Đo giá trị thực tế của dự báo trên development data.** Trên cohort đã mở, chọn trước một bài toán quyết định: sau pilot x10, chọn giữa x20/x50/abstain theo chi phí log bổ sung và phạt không đạt NR 0.95. So với always-x20, always-x50, B2 và survival CDF thích nghi; báo regret/target attainment **bao gồm chi phí pilot, posterior và learner training**. Nếu không có tiết kiệm ròng, paper nên trình bày OEH như mô hình dự báo khoa học, không gắn claim tiết kiệm.
3. **Kiểm tra độ tin cậy của nhãn và xác suất.** Báo calibration theo sáu mốc, phân biệt early success với lớp `>x50`, tỷ lệ censoring theo learner/họ/behavior và độ ổn định nhãn 8/10 ở các lần lặp seed độc lập trên **development cohort**. BC có 90.71% và IQL 77.62% `>x50` ở panel stochastic, nên Brier trung bình dễ che vùng sớm. Sensitivity của ngưỡng/seed nếu phát triển sau outcome phải mang nhãn exploratory, rồi khóa trước cohort mới.
4. **Làm đối chứng cùng endpoint mạnh hơn ở development stage.** Dùng cùng split theo MDP, quyền truy cập và ngân sách tuning cho các mô hình trực tiếp trên pilot: constant/learner prior, CDF/ordinal survival, boosted trees trên 18 đặc trưng, full OEH và kiến trúc kiểm soát 108 đầu vào. Kiểm tra khả thi của việc thích nghi **kiến trúc** image-CNN dự đoán performance của Mateos-Melero et al. và tabular-foundation-model CDF của Kim et al.; nếu representation hoặc tài nguyên không khớp, ghi lý do loại thay vì gọi XGBoost-CF là bản nguyên gốc. Giữ ERI/TQ/SACo/EH trong bảng rank thứ cấp đúng output của tác giả.
5. **Nếu muốn paper mạnh, thêm cohort ngoài hoàn toàn mới sau khi khóa v2.** Mở rộng năng lực xử lý task lớn bằng tính toán thưa/chia batch ở development, chọn task từ metadata và tài nguyên trước outcome; chạy cùng forecast target trên môi trường do bên khác thiết kế. Nếu phương pháp được sửa theo kết quả BRIDGE hiện tại, cả BRIDGE 24 không còn là xác nhận cho v2. Khóa plan, code/hash, đối chứng, endpoint và stop rule trước khi tạo nhãn cohort mới. Không xóa nhãn BRIDGE cũ rồi gọi lại là chưa thấy.
6. **Hoàn thiện đường nộp.** Chốt claims sau strict audit, soát từng con số với artifact, build paper/thesis, cập nhật phần Discussion và giới hạn, điền metadata, xuất gói code/protocol/manifest/log và chỉ dẫn tái lập thực tế. Kiểm PDF có bảng/figure đọc được và thesis đủ 30 trang; bản hiện tại có 34 trang thật. Công khai rõ raw buffers nào chưa tải được từ repo.

## Hai nhánh kết luận sau BRIDGE

- **Nếu prospective 5 cho kết quả thuận lợi:** chỉ nói lợi ích trên những learner, metric và panel có paired interval phù hợp; 10 MDP fresh qua hai panel vẫn là mẫu nhỏ được screen theo khả năng tính toán. Ưu tiên xác nhận mới và utility thay vì viết `vượt SOTA`.
- **Nếu prospective 5 không thuận lợi:** giữ bảng âm, phân tích shift về họ môi trường, horizon, trạng thái, censoring và giả định posterior trên cohort đã mở. Có thể chuyển đóng góp paper sang benchmark/negative transfer và điều kiện áp dụng; phiên bản OEH cải tiến chỉ được kiểm định trên cohort khác được khóa trước outcome.

## Nguồn học thuật cho quyết định này

- [Mateos-Melero et al., EWRL 2024](https://plg.uc3m.es/papers/mateos-melero-et-al-ewrl24.pdf): dự đoán return của offline learner theo dataset/algorithm; endpoint và biểu diễn khác bài này, nhưng là đối chiếu gần về câu hỏi khoa học.
- [Schweighofer et al., CoLLAs 2022](https://proceedings.mlr.press/v199/schweighofer22a.html): TQ/SACo mô tả chất lượng/coverage dataset, không xuất phân phối nhu cầu bảy lớp.
- [Kim et al., arXiv 2026 v2](https://arxiv.org/abs/2601.22259): CDF survival từ các bài phân loại nhị phân với tabular foundation models; XGBoost-CF trong repo chỉ thích nghi công thức đó.
- [Kurenkov & Kolesnikov, ICML 2022](https://proceedings.mlr.press/v162/kurenkov22a.html): kết luận so sánh offline RL phụ thuộc ngân sách đánh giá; công bố utility/cost phải chỉ rõ ngân sách.
- [Cawley & Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html): chọn/tune nhiều lần trên cùng tập đánh giá có thể tạo selection bias.
- [NeurIPS 2026 Evaluations & Datasets reviewer guidelines](https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines) và [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist): tiêu chí chất lượng, tính tái lập, chi tiết split/hyperparameter, error bars và compute.

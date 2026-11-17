# T1 v1.0.0 — Chốt nội bộ câu hỏi nghiên cứu và phạm vi kết luận

> Bản giải thích chi tiết này đã được thay bằng [T1 v1.1.0 một trang](T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) để dùng làm nguồn thuật ngữ hiện hành. Giữ v1.0.0 để truy vết.

Ngày 05/10/2026. **Phạm vi:** chốt thuật ngữ và câu chuyện học thuật cho paper/đồ án; không thay protocol, nhãn, dự báo, gate hoặc verdict của cohort đã niêm phong. Tên đề tài làm việc giữ theo kế hoạch tuần 8: **Difficulty-Aware Offline Reinforcement Learning**. Phụ đề giải nghĩa khi cần: *Forecasting Learner-Specific Data Requirements from Pilot Logs*.

## Câu hỏi và định nghĩa dùng chung

**Câu hỏi chính:** Với log pilot thu bởi behavior policy β trong một MDP tabular stochastic và learner l đã định, OEH dự báo được phân phối của *mốc dữ liệu huấn luyện đầu tiên* mà l đạt chất lượng mục tiêu tốt đến đâu? **Difficulty** ở đây là *nhu cầu dữ liệu thực nghiệm có điều kiện theo (MDP, β, l, ngưỡng)*; không là thuộc tính vô hướng cố hữu của MDP, không là định lý sample complexity và không đồng nhất với descriptor OEH-H.

**Đầu vào khi dự báo:** log pilot D₀ gồm x10 = 10|S||A| transition, danh tính learner l và các đại lượng rút được từ D₀. MDP thật, policy tối ưu, oracle EH, nhãn outcome và dữ liệu tham chiếu TQ/SACo không là input của nhánh OEH pilot-only. Pilot và stream dùng để huấn luyện/chấm nhãn có seed namespace tách biệt. Các mốc x1/x2/x5 được dự báo hồi cứu sau pilot x10; không gọi đây là tiết kiệm dữ liệu trước khi lấy pilot.

**Đầu ra/nhãn:** E_l thuộc bảy lớp x1, x2, x5, x10, x20, x50 và >x50. E_l là mốc đầu tiên có normalized return ≥0,95 trên ≥8/10 seed **tại mốc đó và mọi mốc lớn hơn đã đo**; >x50 là kiểm duyệt phải, không phải quan sát learner cần vô hạn dữ liệu. Đây là nhãn trên lưới budget và mười seed của giao thức, không phải “độ khó thật” độc lập với learner. Ba learner có nhãn requirement chính: BC tabular, CFQI tabular, IQL tabular.

## Phép so chính và thước đo

So OEH đầy đủ với **B2 matched** (18 đặc trưng pilot), **B2+ERI adaptation** và **B2+BWD adaptation** trên cùng log, nhãn, tập meta-training, chia fold theo MDP, kiến trúc AFT/direct-threshold/depth-3 multiclass và quy tắc tuning/trọng số. ERI/BWD là tín hiệu nhóm chuyển sang endpoint này, không phải bản phương pháp tác giả chạy end-to-end. B2 generic và coverage/quality adaptations dùng AFT-only; train-frequency không fit cùng head; oracle EH và TQ/SACo có thông tin MDP/dataset tham chiếu đặc quyền. Báo các nhóm đó riêng, không gộp vào kết luận “same architecture”.

Primary panel là **84 MDP stochastic** của cohort thứ bảy; 36 deterministic lock là control mô tả. Metric chính: **integrated Brier** trên sáu xác suất tích lũy và **seven-class log loss**; hiệu số paired theo MDP, cận dưới một phía 95% đồng thời theo gate đã niêm phong. Không áp ngưỡng cải thiện 5% lên cohort thứ bảy. Calibration, tỷ lệ >x50 theo learner/behavior và chi phí là phân tích bổ sung. Bảy learner BC/CFQI/IQL/BCQ/CQL/SPIBB/VI-LCB chỉ thuộc nghiên cứu **selection tại x10**, với success rate và normalized return riêng; selector không là đóng góp chính của paper.

## Phạm vi claim đã khóa

Có thể nói: OEH đầy đủ cải thiện cả hai proper score so với B2 và các đối chứng đã đăng ký trên 84 kernel stochastic mới **trong những họ generator đã dùng khi phát triển**; các cận dưới của 22 contrast dự báo đều dương theo verdict cohort thứ bảy. Nghiên cứu selection riêng vượt fixed CFQI theo gate đã đăng ký, nhưng **không chứng minh vượt router v5**. Chỉ số FourRooms trên dữ liệu công bố là test **một thành phần** và kết quả NO-GO. Không nói OEH là learner RL mới, vượt SOTA toàn lĩnh vực, OEH-H là cận mẫu đã chứng minh, đã được kiểm định toàn phương pháp trên benchmark độc lập, hay từng khối OEH đã được ablation chứng thực. Các phép ablation trên cohort đã mở là khám phá; kết quả benchmark mới chỉ được đưa vào claim sau khi thực hiện đúng giao thức mới.

**Thứ tự kể chuyện cố định:** Problem/Terms → DATA (sinh thế nào và vì sao, giới hạn) → METHOD (OEH và đối chứng có quyền truy cập tương đương) → RESULT + METRICS → ablation/ngoại kiểm nếu đã chạy → selector/giới hạn. Bảng claim và vị trí mã, artifact, paper nằm ở [T1 claim map](T1_CLAIM_MAP_2026-10-05.md). Hưng, Việt Trần và Quốc Việt dùng tài liệu này làm nguồn thuật ngữ khi phản biện nội bộ; trạng thái phản biện của từng người được ghi riêng, không suy diễn là đã ký.

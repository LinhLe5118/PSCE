# OEH: chẩn đoán gốc và phạm vi nghiên cứu hiện hành

Theo yêu cầu của Linh, các vòng phát triển chưa đạt bị loại khỏi phạm vi:
không giữ prototype, artifact, bảng kết quả hoặc bản lưu ngoài repo của các
hướng đó. Không dùng chúng làm kết quả paper/thesis hoặc phiên bản OEH v5.
Sổ seed và số lần thử chỉ phục vụ chống tái sử dụng dữ liệu và minh bạch quá
trình chọn phương pháp. Dọn dữ liệu không biến một cohort đã đọc thành holdout.

## Vấn đề cần sửa

OEH-v4E gốc dự đoán mean normalized return của learner rồi chọn một policy
cố định. Mục tiêu này khác tỷ lệ training seed đạt normalized return ≥0,95.
E1/S1 niêm phong vẫn giữ nguyên: chênh lệch mean return nhỏ chưa được S1 xác
nhận, còn success giảm so CFQI. Không thay kết quả gốc hoặc chạy lại lock E1.

Một khác biệt estimand đã kiểm bằng fixture giải tích đầy đủ support:
`oeh_risk_v3` và `oeh_crossfit_v2` chia planning loss cho J_best−J_worst
trong observed support; `exact_solver.normalized_return` dùng J_star−J_random.
Epsilon=0,05 ở feature vì vậy không đồng nghĩa success ≥0,95. Fixture cho thấy
một policy admissible theo thang thứ nhất vẫn trượt thang thứ hai, ngay cả khi
model deterministic đã được quan sát đủ. Đây là khác biệt có trong code,
không chứng minh riêng nó gây toàn bộ chênh lệch success của E1. Giữ nguyên
source frozen; bản nâng cấp phải khai báo đúng thang target và support scope.

Lỗi quyết định có hai tầng: chọn sai learner/policy, và nhầm dấu lợi ích action
do empirical transition/reward chưa đủ chính xác. Chia buffer để đánh giá chéo
làm mất support; lấy trung bình các estimate nhiễu có thể bị chi phối bởi một
số model có value lớn. Bound hữu hạn mẫu có thể không đủ thông tin để đổi policy,
nhưng nới ngưỡng sau scoring không sửa lỗi nhận dạng.

CFQI đã tối ưu gần hết trong empirical model ở nhiều task. Một nâng cấp cần
giảm sai số quyết định giữa các model khả dĩ, không chỉ tối ưu sâu hơn trên
cùng model. Missing successor vẫn là giới hạn chung của phương pháp data-only;
không giả định bootstrap nhận dạng được edge chưa thấy.

Kiểm soát giải tích thứ hai trong `tests/test_oeh_estimand_alignment.py`:
cùng một buffer có xác suất dương dưới hai MDP có successor khác nhau và
action tối ưu khác nhau. Đổi sang action tốt ở MDP thứ hai làm mất success
ở MDP thứ nhất. Kiểm soát này không phủ nhận khả năng cải thiện trung bình
trên một population; nó cho thấy surrogate chỉ dựa trên observed edges
không chứng nhận full-MDP success hoặc no-harm bằng cách nới calibration.

## Quy trình hiện hành

Theo yêu cầu mới nhất, chỉ dùng bộ E1 đã chạy: 120 base MDP, 84 stochastic
primary và 36 deterministic control. Không sinh cohort hoặc training buffer
mới. Source/config/tests khóa trước đối chiếu; giữ x10, gamma 0,95, horizon
100, năm behavior, mười seed và baseline learner config. Hash manifest,
buffer identity và artifact phải khớp E1 gốc. Không gọi lại one-shot runner.
API policy learner không nhận family, oracle hoặc score evaluation thật.
Public error target epsilon=0,05 được dùng tường minh để đồng nhất mục tiêu
ε-optimal với protocol, không fit hoặc sweep theo dữ liệu. Công bố compute tăng.

Tái dùng đầy đủ method/seed và policy outcome E1 đã có; chỉ fit lại meta router
theo outer/inner fold gốc, không train lại core learner. Hash lựa chọn learner
và dự báo trước chấm outer test; thiếu grid hoặc lỗi thực thi phải báo lỗi.
Gate joint return/success được khóa trước chạy,
không đổi sau xem kết quả. Thử thất bại được xóa ngay theo yêu cầu; chỉ giữ
attempt ID, seed đã tiêu thụ, hash nguồn và lý do không đạt, không đưa bảng
kết quả âm vào phạm vi paper/thesis đang viết. Không nhận xét chỉ trên các
nhóm thắng hoặc gom các lần thử thích nghi thành confirmation.

Khi có ứng viên đạt screen, phải đối chiếu CFQI và OEH-v4E cùng cohort/config.
Kết quả trên E1 đã đọc là hậu nghiệm; cleanup không biến nó thành holdout.
Không claim confirmation hoặc khái quát ngoài cohort chỉ nhờ thắng E1. Review
độc lập vẫn cần cho phát hành; hiện không tự tạo bộ confirmation mới. Chỉ ghi
nhận kết quả thành công khi đủ điều kiện; không có bảo đảm rằng một thuật
toán sẽ strict-improve trên mọi MDP khi dữ liệu chưa nhận dạng được dynamics.

R34 đã qua screen hậu nghiệm bốn contrast trên E1 gốc và full CI local;
kết quả/giới hạn ở [báo cáo tuần 7](../week7/OEH_V5_POSTHOC_REPORT_2026-09-29.md).
Review độc lập R1/R2 vẫn chưa hoàn tất; không tự ký thay người review hoặc
gọi kết quả này là xác nhận khái quát. Các prototype không qua gate đã
bị xóa cùng bundle; tài liệu này giữ chẩn đoán và quy tắc, không giữ hướng
đã rút hoặc các bảng kết quả thất bại. Resample cùng buffer không phải dữ
liệu độc lập và không chứng nhận true no-harm. Thiếu successor chưa quan sát
vẫn là giới hạn nhận dạng. Thử đã rút không được tái diễn giải thành confirmation.
Trước đối chiếu ngày 29/09, screen dùng joint IUT: H0 là
ít nhất một endpoint với một comparator không tăng; chỉ qua nếu cả bốn cận dưới one-sided theo
MDP ở alpha=0,05 dương. Không tự áp biên thực dụng NR 0,005 mà người dùng
không yêu cầu. IUT không cần chia alpha giữa hai endpoint khi phải thắng cả
hai; đây không phải simultaneous confidence-region coverage. Cluster
bootstrap là xấp xỉ, không phải chứng minh level chính xác ở hữu hạn MDP.
Tham khảo [nghiên cứu IUT](https://pmc.ncbi.nlm.nih.gov/articles/PMC2752611/).
Dấu point estimate từng subgroup dùng chẩn đoán, không veto riêng. Không
bảo đảm không hại ở mọi nhóm; mọi subgroup phải báo đầy đủ khi phát hành.
Các vòng đã rút không được đổi thành thành công; screen thích nghi không
thành confirmation. So CFQI và OEH-v4E trên đúng E1 gốc vẫn bắt buộc.

<!-- retained-reference:r34:start -->
Ứng viên R34: hypothesis dự báo paired return/success cần tương tác phi tuyến
giữa feature quan sát, thay vì chỉ thay tầng chọn learner. Dùng polynomial
kernel bậc hai (offset 1, gamma 1/d), bao gồm toàn bộ interaction; không chọn
feature theo family hay sweep degree/bandwidth bằng evaluation. Standardizer
và kernel centering chỉ fit training, có trọng số whole-MDP multiplicity.
Intercept không bị penalty; grid ridge và grouped inner 4-fold giữ T1.
NR dùng inner MSE; success dùng win/loss/tie simplex và frequency loss tương
đương categorical Brier cho chọn ridge (khác hằng số label, không báo nó là
absolute per-seed Brier). Hai block feature gốc ensemble đều trọng số.
Router tối ưu tích ΔNR × ΔSR dự báo trên các route deterministic thuộc
supported frontier, không fixed endpoint weight hoặc true-score rule.
Không claim optimum nguyên toàn cục, true no-harm hoặc xác nhận khái quát.
Chỉ reuse E1 x10, cùng gate bốn contrast và 2.000 MDP-cluster draw đã khóa;
không tạo MDP/buffer mới hay gọi lại one-shot runner.
<!-- retained-reference:r34:end -->


<!-- active-candidate:r44:start -->
Ứng viên nâng cấp hiện hành R44 đã qua sáu contrast so CFQI, OEH-v4E và R34,
đủ 2.000 MDP-cluster refit trên chính 120 MDP E1. Stochastic x10: return
0,806803, success 45,76%; so R34 tăng 0,013322 và 2,43 điểm phần trăm.
Giữ nguyên dự báo core, paired target, kernel, ridge grid và joint router R34;
cho phép chọn cả bốn learner E1 bổ sung từ toàn bộ registry có sẵn, không
chọn library subset bằng score. Không tạo MDP/buffer hoặc train baseline mới.
Gate khóa trước chạy yêu cầu cả sáu point effect và sáu cận dưới one-sided
95% dương, không practical margin. Giữ mọi MDP âm của lần đạt. Các prototype
và bundle không đạt đã xóa; chỉ còn attempt/hash ledger. R34 là tham chiếu
thành công giữ nguyên, không bị ghi đè. Báo cáo và giới hạn:
[báo cáo R44](../week7/OEH_V5_R44_POSTHOC_REPORT_2026-09-29.md).
R1/R2 chưa review; kết quả hậu nghiệm không xác nhận khái quát ngoài E1 hoặc
thay verdict S1. Không sửa paper H1-frozen trước review.
<!-- active-candidate:r44:end -->

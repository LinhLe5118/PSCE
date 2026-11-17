# Đối chiếu trên artifact E1 đã chạy

Theo yêu cầu của Linh, đối chiếu chỉ tái dùng bộ E1 gốc gồm 120 base MDP,
600 nhóm environment × behavior và mười training seed mỗi nhóm. Không tạo
MDP, seed namespace, pilot hoặc training buffer mới; không gọi lại one-shot
runner hoặc thay lock E1. Kiểm manifest theo input seal, các CSV theo execution
summary và feature pilot theo feature seal. Mỗi kết quả core phải khớp hash
buffer, method, budget và seed đã đăng ký.

Endpoint chính vẫn là x10 trên 84 stochastic MDP; 36 deterministic MDP là
control mô tả. Các baseline đã có toàn bộ grid x1, x2, x5, x10, x20, x50.
Meta model được fit riêng trong từng panel như procedure OEH-v4E gốc;
control không phải phép chuyển một model đã fit trên stochastic sang
population mới. Panel lấy từ phân chia E1 đã đăng ký, không được tự nhóm lại
sau xem score. Không suy ra khả năng transfer ngoài population từ control này.
Đối chiếu routing hiện tại dùng kết quả x10, không tuyên bố đã train lại
baseline hoặc đã chạy một policy mới trên toàn bộ sáu budget.

Training budget của policy được chọn giữ x10 như các core learner. Router
tái dùng feature từ pilot độc lập mà OEH-v4E đã có; các meta label đến từ
outer training MDP. Pilot và chi phí meta training là phần thông tin/compute
bổ sung của hệ thống chọn learner, không được diễn giải thành cùng tổng
information budget với một CFQI cố định không dùng pilot. Wall time của
bootstrap là chi phí phân tích, không phải runtime inference của policy.

## Metrics và đơn vị so sánh

Normalized return giữ nguyên định nghĩa E1:

\[
\mathrm{NR}(\pi)=\frac{J(\pi)-J(\pi_{\mathrm{random}})}
{J(\pi_\star)-J(\pi_{\mathrm{random}})},\qquad \gamma=0{,}95.
\]

Không clip giá trị âm hoặc lọc seed. Success của một seed là NR ≥ 0,95;
success rate là trung bình indicator đó. Return trung bình và tỷ lệ đạt
ngưỡng là hai endpoint khác nhau; tăng endpoint thứ nhất không chứng minh
tăng endpoint thứ hai.

Các paired contrast dùng cùng base MDP, behavior, budget và training seed.
Một router chọn cùng learner cho cả mười seed của một nhóm, như OEH-v4E gốc.
Không chọn learner riêng theo true score từng seed. Khi tổng hợp, lấy trung
bình trong base MDP trước; không coi năm behavior hay mười seed là các MDP
độc lập.

## Fit, seal và inference

Feature pilot và outer-fold assignment E1 được giữ nguyên. Meta model được
phép học return/success label của **outer training MDP**, đúng cấu trúc nested
evaluation của E1. Label của outer test MDP không được đưa vào fit, chọn ridge
hoặc quyết định learner cho chính MDP đó. Học hệ số từ training khác với đặt
rule/threshold bằng score outer test; không gọi meta training này là data-only
policy learning không có label hiệu quả.

Quyết định và dự đoán phải được lưu/hash trước bước chấm outer test. Adapter
OEH-v4E phải tái lập đúng các lựa chọn gốc trước khi dùng đối chiếu. Cả hai
endpoint phải tăng so với cả CFQI và OEH-v4E; cấu hình screen phải được khóa
trước chạy, không đổi sau xem score.

Bootstrap giữ 2.000 draw và seed 20260921 của T1, lấy whole base-MDP cluster
trong strata outer fold cố định như S1. Mỗi draw refit standardization, chọn
ridge inner và router cho cả ứng viên lẫn OEH-v4E. Không reroll, bỏ draw hoặc
chỉ bootstrap các quyết định đã fit cố định. Control không tham gia gate chính.

Screen mới không thay verdict S1 gốc. E1 đã được đọc và quá trình chọn phương
pháp là thích nghi; khoảng bootstrap không biến đối chiếu này thành independent
confirmation. Chỉ báo bằng chứng đúng cohort/config này, cùng giới hạn và số
lần phát triển; không bảo đảm strict superiority trên mọi MDP hoặc ngoài
distribution meta training.

Nếu không đạt, xóa bundle, prototype, test/protocol riêng và mô tả active của
hướng đó sau khi ghi sổ attempt/hash tối thiểu. Sổ không chứa bảng metrics
thất bại và không làm một dataset đã đọc trở lại thành holdout. Chỉ giữ kết
quả nâng cấp để tổng hợp khi đáp ứng đủ gate đã khóa.

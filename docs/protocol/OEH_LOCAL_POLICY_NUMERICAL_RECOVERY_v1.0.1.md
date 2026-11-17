# Phục hồi lỗi số học trước khi mở outcome

Attempt v1.0.0 được giữ nguyên ở `artifacts/reassessment/oeh_local_prospective_r1_2026-09-28/`.
Đủ 300 nhóm đã được xử lý: 299 hoàn tất, nhóm 119 lỗi IQL expectile; chưa có
selection seal, chưa oracle scoring, chưa `rows.csv`/`report.json`. Không bỏ nhóm lỗi.
Tổng runtime trước dừng 171,7811675 giây. Cohort/buffers không được sinh lại.

Lỗi được tái lập **không dùng benchmark** bằng vector constant
`[16.022382365201043,16.022382365201043]`, weights
`[0.07254101021159007,0.9274589897884099]`, expectile 0,7. Trung bình weighted
uncentered lệch bởi roundoff ngoài interval có tolerance tuyệt đối 1e-15.
Expectile thỏa translation equivariance: E_tau(v+c)=E_tau(v)+c. Trừ min(v)
trước solve, cộng lại sau solve; giữ nguyên objective và mọi hyperparameter.
Tests đối chiếu với root solver độc lập, scale/translation, original bytecode và
defaults. Không sửa nguồn learner niêm phong, không monkeypatch global chung.

**Ngoại lệ được đăng ký công khai trước scoring** với quy tắc no-retry của v1.0.0:
chỉ phục hồi nhóm lỗi, vì là lỗi tính toán và chưa đọc outcome. 299 nhóm hoàn tất
được copy nguyên byte sang attempt phục hồi và kiểm hash; không refit hoặc chọn
lại. Nhóm lỗi chưa có checkpoint policy hoàn chỉnh nên phải refit cả nhóm; số
fitting một phần ở attempt lỗi không biết chính xác, không gọi toàn pipeline
training là one-shot. Lưu tất cả log, failure, plan và source/input hash gốc.
Không mở rộng ngoại lệ sang negative result, failure do thiếu support hay retune.

Config, alpha, bootstrap seed/replicates, x10, learner iteration/tolerance,
expectile 0,7/inverse temperature 3, comparator, endpoint, cohort và gate không đổi.
Attempt mới lưu source repair hash/commit trước refit nhóm lỗi; niêm phong tất cả
policy/decision trước mở scoring một lần. Cohort còn unopened về outcome nhưng
buffer đã tồn tại trước numerical repair; không claim final repair freeze trước
data generation. Đây là prospective development có deviation công khai,
**không E1, không confirmation và không GO 120 MDP**.

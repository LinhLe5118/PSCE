# Bàn giao sửa benchmark và T1 v1.1.0

Theo yêu cầu chủ dự án, câu hỏi chính là utility trên **stochastic dynamics**.
Đã sửa ranh giới phân tích trước mở evaluation, không thay môi trường để tạo
kết quả thuận lợi và không sinh thêm development dataset.

## Thiết kế cuối

| Vai trò | Họ môi trường | Evaluation MDP | Groups | Core fits |
|---|---|---:|---:|---:|
| Primary stochastic | Garnet / forward / bottleneck | 30 + 36 + 18 = **84** | 420 | 75,600 |
| Deterministic controls | Lock | **36** | 180 | 32,400 |
| Tổng thu thập | Recipe gốc, seed và buffers giữ nguyên | 120 | 600 | 108,000 |

Không biến lock thành stochastic bằng slip mới: làm vậy sẽ thay benchmark
distribution/recipe và đòi hỏi validation riêng. Controls vẫn hữu ích để khảo
sát giới hạn và khác biệt dynamics, nhưng không đóng góp vào claim stochastic.
Năm behavior families vẫn hiện diện trong cả hai panel.

**Primary chỉ dùng 84 MDP** từ đầu tới cuối: nested folds, scaling, tuning,
best-fixed, fallback, endpoints và bootstrap. Controls chạy pipeline riêng,
descriptive; không cứu gate chính. Holm vẫn m=3 và margin chính vẫn .005,
không thay sau khi phân nhóm. Ngưỡng pass không được hạ vì chỉ còn 84 clusters;
power có thể thấp hơn thiết kế gộp 120, chưa có cam kết 80% power.

## Code đã hoàn thành

- `environments/benchmark_panels.py`: family-to-panel cố định; xác minh
  stochastic dynamics từ kernel ở reachable nonterminal states; tách rows
  trước fitting; lỗi nếu family thiếu/không đăng ký hoặc kernel không khớp.
- `scripts/data/generate_offline_datasets.py`: gọi kiểm tra cho từng base MDP
  sau dynamic-range resolution, trước thu thập; thêm `dynamics_audit` vào
  oracle manifest và pin module mới trong generator provenance. Không thêm
  panel label, kernel hay entropy thật vào estimator data-only index.
- `scripts/validation/preflight_stochastic_panels.py`: đọc quota từ config,
  reconstruct kernel theo development recipe đã có và kiểm mọi MDP; không
  sinh trajectories, không mở sealed bundle. CLI từ chối ghi đè output.
- `scripts/validation/verify_week6_t1_freeze.py`: mặc định kiểm manifest v1.1.0.

Kiểm định dynamics: dùng support tolerance `1e-12`, union-over-actions
reachability từ initial distribution, bỏ terminal; có ít nhất một reachable
row với hai successor vượt tolerance. Đây là tiêu chí thuộc lớp stochastic,
không phải mọi row đều stochastic hoặc đạt entropy tối thiểu. Kernel mismatch
phải dừng, không regenerate, relabel hay loại bỏ MDP để đạt quota.

Bundle development/evaluation đã có không được viết lại metadata. E1 sẽ tạo
sidecar bằng cùng checker sau khi được mở theo technical gate; không sinh lại
buffer chỉ để thêm field. Guard evaluation hiện hành vẫn giữ nguyên.

## Bằng chứng kiểm tra

Các báo cáo nằm ở `artifacts/week6/stochastic_benchmark_2026-09-21/`:

- `development15.json`: 11 stochastic / 4 deterministic; mọi kernel khớp recipe.
- `development48.json`: 36 / 12; mọi kernel khớp recipe.
- `development60.json`: 44 / 16; mọi kernel khớp recipe.
- Quota evaluation chỉ đọc từ cấu hình: 84 / 36; preflight không materialize
  kernel evaluation và không mở manifest/buffer/outcome niêm phong.
- `checks.json`: kết quả suite/coverage, verifier và checksum bằng chứng.

Các kiểm tra trên chỉ tái dựng MDP development đã đăng ký, không tạo dataset
mới hoặc retrain/tune v4E. Không trích gain mới từ outcome subgroup để chọn
thiết kế. Gain v4E `.009574` cũ vẫn là **mixed-development evidence**, chưa phải
gain của stochastic-only nested evaluation. Không sửa artifact/kết quả lịch sử.

## Phiên bản và phạm vi nghiệm thu

[T1 v1.1.0](../protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0_2026-09-21.md)
và [JSON](../protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0.json) thay v1.0.0.
Phiên bản cũ giữ lại làm lịch sử, không dùng để cấp GO mới. Manifest mới pin
protocol, sources và tests. Kiểm tra từ gốc repo:

```text
python -m scripts.validation.verify_week6_t1_freeze
python -m scripts.validation.preflight_stochastic_panels --help
```

**Phần T1 và sửa benchmark đã hoàn thành; E1 chưa chạy.** Không còn yêu cầu
xin phép GVHD. H1 formula review, D1 unified runner/refit-bootstrap và B2 review
độc lập vẫn là các task riêng. D1 phải gọi phân nhóm trước fitting và kiểm tra
full quota, panel completeness, seeds, resume, sidecar hashes. Không có claim
rằng helper phân nhóm tự biến runner development cũ thành sealed runner mới.

Các thay đổi này xử lý mismatch giữa mục tiêu stochastic và endpoint gộp;
chúng không tự nâng kết quả development thành confirmatory hoặc giải quyết
novelty/external validity nêu trong audit dự án.

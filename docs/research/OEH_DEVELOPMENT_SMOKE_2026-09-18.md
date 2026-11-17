# OEH-R: tích hợp thử vào pipeline development

Ngày 18/09/2026. Đây là smoke test sau thiết kế OEH-R, không phải confirmatory
evaluation hoặc một protocol thay thế OEH-v1 đã đăng ký.

**Kết quả hoàn tất:** 225/225 cases không lỗi; OEH-v1 phẳng m=1 ở 131/131 curve
có thể tính (94 abstentions riêng). OEH-R phẳng ở cả 225 curves, 223 tại U=1,
0 certificates. Riêng 165 cases stochastic đều U=1. **Chưa cải thiện saturation
theo nghĩa có thông tin về độ khó/depth.** Xem [báo cáo đầy đủ](../../artifacts/research/oeh_development_smoke_numpy_2026-09-18/REPORT.md).

## Điểm nối với hệ thống

- API mới: `code/oeh_certificate_adapter.py::estimate_oeh_certificate` nhận cùng
  raw trajectory arrays, public dimensions và initial distribution đang dùng.
- Runner mới: `scripts.analysis.run_oeh_development_smoke` đọc data-only index,
  xác minh SHA-256 của index và từng NPZ, rồi cắt đúng `[0,size]` trước suy luận.
- OEH-v1 được chạy lại trên **cùng prefix** với config hiện tại: 100 bootstrap,
  depth 0..100, epsilon .05; không lấy số cũ từ một manifest khác rồi ghép.
- OEH-R dùng Bernstein, delta .05, epsilon .05, gamma .95 và public reward bound
  1. Epsilon của OEH-R chuẩn hóa theo `1/(1-gamma)`; không tương đương tolerance
  của loss cũ hoặc benchmark normalized return. So sánh ở đây về tình trạng phẳng,
  tính khả dụng và độ biến thiên, không phải cùng một scalar/quality target.
- Họ policy OEH-R là pessimistic backups từ empirical behavior; uniform ở state
  chưa có observation chỉ là lựa chọn baseline, không phải bằng chứng về dynamics.
- Runner đo toàn bộ ba prefix, không dừng ở prefix đầu được adaptive selector
  của OEH-v1 chấp nhận. Do đó tổng số curve không phải số của báo cáo lịch sử.

## Terminal và coverage

Những destination có `terminal=True` trong **pilot hiện tại** được gán absorbing
self-loop và zero continuation theo contract state-based termination của dự án.
Không tạo transition/count giả ở terminal. Reward trên bước đi vào terminal
vẫn được giữ. `truncated=True` chỉ đánh dấu ranh giới collector; next state thật
vẫn vào transition counts. State/action chưa được thấy giữ uncertainty.

Adapter từ chối flag mâu thuẫn, act-from-terminal, index ngoài public dimensions,
reward ngoài public bound, trajectory bị đứt, và reset không đúng ranh giới.
Metadata về terminal đầy đủ của oracle không được dùng để giúp pilot.

## Phạm vi smoke

Input: `datasets/week1_development_validation` — 15 MDP thuộc bốn structural
families, gồm stochastic dynamics và deterministic lock controls; 5 behavior
families; pilot budgets `2,5,10 |S||A|` = 225 prefix cases. Mỗi case có 101 depth.
Đây là bộ development lịch sử; không gọi là toàn bộ bundle expanded 48 MDP.

Split gate kiểm tra toàn manifest trước khi áp limit; evaluation record xuất
hiện bất kỳ đâu cũng bị từ chối. Estimator chỉ được nhận raw arrays và public
parameters. Oracle chỉ tái dựng sau khi cả hai estimator đã trả kết quả, kiểm tra
dimensions và recorded optimal return, rồi chấm true regret của policy OEH-R.
Không chạy learner training mới và không mở evaluation split.

Kết quả đầy đủ được lưu tại:
[`../../artifacts/research/oeh_development_smoke_numpy_2026-09-18/`](../../artifacts/research/oeh_development_smoke_numpy_2026-09-18/).
`report.json` là marker hoàn thành; `progress.json` riêng không chứng minh run đã
xong. `summary.csv` lưu từng pilot, `curves.csv` lưu từng depth cùng oracle regret.

## Backend và kiểm tra

Dùng backend NumPy đã có của OEH-v1 để tránh lặp Python cho mỗi state/action/depth.
Không giảm bootstrap, depth hoặc đổi ngưỡng. Trước khi chuyển backend đã so một
pilot đầy đủ giữa reference/NumPy bằng recursive comparator của repo; các field
rời rạc khớp và floating fields nằm trong `rtol=1e-9, atol=1e-10`.
Run reference đầu bị dừng, có `INTERRUPTED.md`, không dùng như kết quả hoàn chỉnh.

Các nhóm test đã đạt:

- Certificate, adapter, acceleration, estimator cũ và exact solver: 66 tests,
  7 subtests.
- Adaptive pilot, artifact IO và leakage: 20 tests, 3 subtests.

Tổng targeted verification: **86 tests, 10 subtests**; không phải full suite.
Adapter tests kiểm tra terminal/truncation, prefix isolation, unseen action,
checksum, split gating và cache policy trùng nhau. Caching chỉ tránh đánh giá
lặp cùng policy; vẫn xuất đủ mọi depth.

Chạy lại từ repo root, chọn output directory mới:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -u -m scripts.analysis.run_oeh_development_smoke --dataset-dir datasets/week1_development_validation --output-dir artifacts/research/oeh_smoke_reproduction --v1-backend numpy
```

Không đưa backend thử nghiệm thành mặc định của estimator hoặc adaptive selector:
API OEH-R không trả `n_req_hat`, bootstrap-valid fraction hay censored sample
requirement, nên không được giả lập các field cũ để qua validation contract.

## Cách đọc kết quả

- `v1_all_floor`: tất cả multiplier bằng 1 ở một curve có thể tính; báo riêng
  observed/censored/abstained, không gộp abstention vào bằng chứng bão hòa.
- `v2_all_ceiling`: mọi upper regret bound nằm ở 1; có thể an toàn nhưng vô dụng.
- `v2_flat`: range theo depth không quá `1e-9`; kể cả flat dưới 1 cũng không mang
  tín hiệu về depth. `v2_unique_policies` giúp xác định cả họ candidate có collapse.
- `positive_reward_lower_fraction` và `full_transition_box_fraction` chỉ ra
  confidence intervals quá rộng ở sample sizes hiện có.
- `false_certificate_count=0` khi `v2_certified_curves=0` là kết quả rỗng về
  calibration, không chứng minh predictor tốt hoặc confidence calibration đã đủ.
- `U>epsilon` không suy true regret lớn hay cần thêm một số mẫu xác định.

Hướng xử lý tiếp nếu cận vẫn phẳng: phân tích độ rộng confidence set và tách
candidate-policy collapse khỏi comparator uncertainty; nghiên cứu cận theo value
và same-model regret. Không khẳng định các thay đổi này chắc chắn giải quyết được
identifiability khi behavior thiếu coverage. Không sửa tolerance để ép có chứng nhận.

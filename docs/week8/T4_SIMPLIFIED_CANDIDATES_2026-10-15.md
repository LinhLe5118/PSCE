# T4 — Ứng viên OEH đơn giản hóa (15/10/2026, hậu kiểm) — **dừng ở chốt H1**

Thẻ T4 của kế hoạch lên hội nghị (bộ agent kit, lưu ngoài repo); trả lời W3.
- Spec và quy tắc chọn (commit **trước** khi chạy): [`T4_SIMPLIFIED_CANDIDATES_SPEC_2026-10-15.md`](T4_SIMPLIFIED_CANDIDATES_SPEC_2026-10-15.md).
- Script: `scripts/analysis/simplified_candidates_t4.py`.
- Kết quả: [`artifacts/week8/simplified_candidates_t4_2026-10-15/`](../../artifacts/week8/simplified_candidates_t4_2026-10-15/).

## Ứng viên

| | Thành phần |
|---|---|
| S1 | B2 + P (mô phỏng hậu nghiệm), **một head** multiclass |
| S2 | S1 + H (rủi ro lập kế hoạch) |
| S3 | S1 + cổng dịch chuyển (head vững = B2, một head) |
| S4 | B2 + P + D, một head, có cổng (head vững = B2 + D, một head) |

- Mọi head fit trên 720 MDP meta-training. Cổng dùng đúng tham số đã khóa.
- Bản fit lại của OEH trong lần chạy này tái tạo dự báo niêm phong (sai số ≤ 9,4e-16).

## Kết quả: **không ứng viên nào đạt quy tắc → khuyến nghị giữ OEH**

| Gộp | OEH-F | S1 | S2 | S3 | S4 |
|---|---:|---:|---:|---:|---:|
| Trong generator, IBS so với OEH | −0,2% | +5,7% | +2,3% | +6,6% | +1,7% |
| Trong generator, LL so với OEH | −0,6% | +3,3% | +0,6% | +4,2% | +2,1% |
| Điểm ngoài generator (TB của (rel IBS + rel LL)/2) | +5,7% | +14,5% | +16,7% | +16,5% | +3,3% |

Ngưỡng chấp nhận là ≤ +1% cho cả ba dòng.

- **Nested LOPO** (11 fold): quy tắc chọn **OEH ở cả 11 fold**.
- **S4 đứng gần nhất.** Nó còn tốt hơn OEH trên SPIBB (−2,5%), SPIBB-2 (−3,7%) và forest (−1,3%), nhưng kém trên mọi panel khác: Colosseum +4 đến +10%, FrozenLake +10%.

## Đọc kết quả

1. **Câu hỏi W3** ("đâu là phương pháp, đâu là phần dán thêm"): trên các panel đã mở, bỏ ensemble head, bỏ OEH-H hay bỏ chẩn đoán D đều làm mất độ chính xác, nhất là ngoài generator.
   - Kết hợp với ablation của T1/hậu kiểm (bỏ OEH-H hoặc OEH-B khỏi OEH đều kém hơn): **mọi thành phần đều đóng góp**.
   - Cách viết đúng cho bài là trình bày OEH như một tổ hợp có lý do cho từng thành phần, kèm bảng ablation. Không thay bằng phiên bản rút gọn.
2. **Tín hiệu đáng ghi lại:** bỏ OEH-H (S4) giúp trên SPIBB, là nơi dịch chuyển do behavior. Điều này khớp với Shapley âm của khối H ngoài generator. Đây là giả thuyết cho panel 6, chưa phải claim.
3. **Không đăng ký ứng viên nào.** Theo quy tắc của spec, nếu panel 6 có kiểm định ứng viên thì chỉ nên là kiểm định *mô tả* của S4. Việc đăng ký thêm hay không là quyết định ở chốt H1.

## Điểm vào và ví dụ một pilot

`scripts/examples/forecast_one_pilot.py` dựng một bundle chỉ gồm các stream pilot của một nhóm MDP–behavior, chạy `scripts/analysis/oeh_pipeline.py` và in luật 7 lớp của 3 learner (xem phần kết quả chạy ở dưới).

**Kết quả chạy ví dụ** (pilot `c8_uniform_garnet_000` / `q_star_softmax` của cohort 8; khớp head trên meta-training mất vài phút):

```
classes x1, x2, x5, x10, x20, x50, >x50
behavior_cloning_tabular   0.000 0.001 0.000 0.000 0.001 0.001 0.996   d = 1.19
cfqi_tabular               0.002 0.005 0.040 0.039 0.465 0.374 0.075   d = 1.19
iql_tabular                0.001 0.001 0.001 0.000 0.000 0.000 0.996   d = 1.19
```

Lệnh: `python -m scripts.examples.forecast_one_pilot --dataset datasets/oeh_eighth_cohort_2026-10-14`.

# H4 — chấm lại độc lập panel 6 (16/10/2026)

**Kết luận: KHỚP. H4 đạt.** F1 GO và F2 GO được tái tạo chính xác trên máy thứ hai.

## Cách làm

- **Máy:** máy phụ của thành viên nhóm, không phải máy chạy chính.
- **Môi trường:** Python 3.12.15, numpy 2.5.1, pandas 2.3.3, scipy 1.18.0.
- **Mã nguồn:** clone sạch của `main` tại `990b2559`, nên repo làm việc không bị ghi đè `result.json`.
- **Lệnh chạy:**
  1. `python -m scripts.analysis.run_oeh_panel6_20261015 analyze`, đúng như T10 yêu cầu.
  2. [`scripts/analysis/verify_panel6_h4.py`](../../scripts/analysis/verify_panel6_h4.py). Script này kiểm tra bằng code viết riêng, không dùng code của nhóm.
- **Output:** [`artifacts/week8/h4_panel6_rescore_2026-10-16/`](../../artifacts/week8/h4_panel6_rescore_2026-10-16/) (`verify_output.txt`, `manifest.json`).

## Kết quả

| Kiểm tra | Kết quả |
|---|---|
| `result.json` so với bản đã commit (bỏ qua timestamp) | Sai khác tối đa **0**; F1 GO, F2 GO |
| `scored.csv.gz` (43.200 dòng) | Nội dung trùng hoàn toàn. Byte khác chỉ do gzip |
| `summary.csv`, `by_learner.csv`, `murphy.csv`, `gain_by_distance.csv`, `label_variant_tau099.csv` | Sai khác **0** |
| Hash protocol | `49c27071…` khớp `protocol.sha256` |
| Seal của mdpp và labmaze | `outcome_run.json` trỏ đúng seal; hash `predictions_preoutcome.csv` khớp seal; `learner_outcomes_observed = 0` |
| Thứ tự commit | Seal commit lúc 00:55, nhãn lúc 02:24 ngày 16/10 |
| Nhãn tính lại từ `outcomes.csv.gz` (115.200 job mỗi họ, đủ 10 seed ở mọi ngân sách) | `event` và `event_extended` khớp **1.440/1.440** dòng mỗi họ. Cờ `success` đúng bằng NR ≥ 0,95 |
| IBS và log loss của endpoint đầy đủ, tính bằng code riêng | Khớp `summary.csv` (sai khác ≤ 9e-16) |

**IBS / LL gộp hai họ:**

| Phương pháp | IBS | LL |
|---|---:|---:|
| OEH | 0,0971 | 1,039 |
| OEH-F | 0,0940 | 1,003 |
| Head vững | 0,1140 | 1,122 |
| B2 + prefix/FQE | 0,1186 | 1,090 |
| B2 | 0,1533 | 1,297 |

## Ghi chú cho H6, không ảnh hưởng H4

1. **Bản ghi T10 dễ gây hiểu nhầm.** Dòng "OEH: IBS 0,0964 / LL 0,989; B2 0,1554 / 1,233; B2 + prefix/FQE 0,1254 / 1,102" là số của **biến thể nhãn τ = 0,99** (`label_variant_tau099.csv`), không phải endpoint đầy đủ. Số của endpoint đầy đủ là 0,0971 / 1,039 (OEH), như bảng trên. Nên ghi rõ.
2. **Dấu của regret trong `06_results.tex`, Test 3, chưa nhất quán.** Câu "−2,61 budget units …, LB 1,29" đặt cạnh nhau một mức thay đổi regret (âm) và một cận dưới của mức cải thiện (dương). Nên viết cùng một chiều, ví dụ "regret giảm 2,61 (cận dưới 1,29)".
3. **Các số panel 6 trong paper đã đối chiếu và khớp `result.json`:** F1 (0,0315 / 0,0247; 0,1034 / 0,0808), F2 (43,8% / 32,8%; 0,0616 / 0,2025), regret (2,61 / 1,29; 7,91), 230.400 job. Abstract cũng khớp: 660 MDP = 396 + 72 + 192, và 688.320 job = 371.520 + 86.400 + 230.400.

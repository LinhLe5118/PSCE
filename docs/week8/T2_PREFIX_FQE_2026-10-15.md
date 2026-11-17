# T2 — Baseline "prefix pilot + FQE + ngoại suy" (15/10/2026, hậu kiểm, mô tả)

Thẻ T2 của kế hoạch lên hội nghị (bộ agent kit, lưu ngoài repo); trả lời W1 và W7 ("chạy thử learner trên pilot rồi ngoại suy là đủ").
- Spec (commit **trước** khi chạy): [`T2_PREFIX_FQE_SPEC_2026-10-15.md`](T2_PREFIX_FQE_SPEC_2026-10-15.md).
- Script: `scripts/analysis/baseline_prefix_fqe.py` (hai pha `features`, `evaluate`).
- Kết quả: [`artifacts/week8/baseline_prefix_fqe_2026-10-15/`](../../artifacts/week8/baseline_prefix_fqe_2026-10-15/).

## Đã chạy gì

- **Đặc trưng** trên mọi pilot: 7 cohort meta-training (720 MDP) và 11 panel đã mở.
- Hai biến thể **cố định trong spec, không chọn tham số**:
  - `prefix_fqe_raw`: luật zero-shot từ đường cong;
  - `b2_plus_prefix_fqe`: 10 đặc trưng prefix + FQE cộng B2, đưa qua head matched fit trên meta-training.
- **Độ phủ meta-training:** 100%.
  - Cohort fourth120 mang namespace `opened_fourth120/` trong meta panel (`run_oeh_confirmatory_20261004.configure`). Lần chạy `evaluate` đầu dừng ở bước kiểm tra độ phủ (83%) vì thiếu namespace này.
  - Sửa phép join rồi chạy lại; không đổi gì khác.
- Endpoint đầy đủ và endpoint ngoại suy dùng bộ chấm T1.

## Kết quả

### 1. Bản thô không cạnh tranh, bản hiệu chỉnh là baseline mạnh

Bản thô có LL 1,7–2,3 trên 9 trong 11 panel (B2 trên cùng các panel: 0,39–1,0). Bản hiệu chỉnh tốt hơn B2 rõ rệt:

| B2 + prefix/FQE so với B2 | IBS | LL | `ibs_ge` | `ll_ext` |
|---|---:|---:|---:|---:|
| Kiểm định cuối, ngoài gộp (228 MDP) | +18,4% | +12,0% | +17,5% | +11,1% |
| Phát triển, ngoài gộp (252) | +9,3% | +5,0% | +5,8% | +3,6% (KTC chứa 0) |
| Panel 5 gộp (72), mọi KTC chứa 0 | −2,8% | −1,4% | −2,0% | −5,1% |

### 2. OEH vẫn hơn baseline hiệu chỉnh, kể cả ở vùng ngoại suy

| OEH so với B2 + prefix/FQE | IBS [KTC 95%, tuyệt đối] | LL | `ibs_ge` | `ll_ext` |
|---|---:|---:|---:|---:|
| Kiểm định cuối, internal (168) | +13,0% [0,004; 0,010] | +7,0% | +15,0% | +9,6% |
| Kiểm định cuối, ngoài gộp (228) | +23,7% [0,014; 0,028] | +13,0% [0,054; 0,124] | +27,0% | +17,0% [0,057; 0,111] |
| Panel 5 gộp (72) | +12,5% [0,001; 0,013] | +13,5% | +15,1% | +13,9% |
| OEH-F, phát triển, ngoài gộp (252) | +16,3% | +4,0% (KTC chứa 0) | +19,7% | +11,8% |

Ngoại lệ:
- **SPIBB (phát triển):** OEH-F và baseline hiệu chỉnh hòa (IBS −3,7%, LL −2,5%; KTC chứa 0). Đây là panel duy nhất baseline nhỉnh hơn về điểm.
- **FrozenLake:** **bản thô** tốt nhất (IBS 0,0018 so với 0,0040 của OEH). 100% nhãn của FrozenLake là ">x50". Đường cong không thấy thành công nào thì đặt gần hết khối xác suất vào ">x50", nên thắng một cách tầm thường. Trên panel này OEH vẫn hơn bản hiệu chỉnh 49% IBS.
- **Colosseum 2029, LL:** OEH-F hòa bản hiệu chỉnh (−2,4%, KTC chứa 0); OEH (có cổng) hơn 10,3%.

### 3. So với LCE

`b2_plus_prefix_fqe` và `b2_plus_lce` (đã có sẵn trong các panel) ở cùng mức:
- prefix/FQE tốt hơn trên Colosseum 2029, SPIBB và SPIBB-2;
- LCE tốt hơn trên forest, FrozenLake và gambler.

Thông tin "chạy learner trên pilot" là có ích. Nhưng cả hai cách khai thác nó, một khi đã hiệu chỉnh, đều kém OEH.

## Đọc kết quả

- **Trả lời W1/W7:** reviewer hỏi "chạy learner trên pilot rồi ngoại suy" có đủ không. Câu trả lời là không: bản thô kém xa, bản hiệu chỉnh kém OEH 12–24% IBS trên các panel kiểm định. Lợi thế của OEH ở vùng ngoại suy (`ibs_ge`, `ll_ext`) lớn hơn trên endpoint đầy đủ, khớp với T1.
- **Cần nói trong bài:**
  - Baseline hiệu chỉnh là comparator mạnh nhất ngoài OEH trên SPIBB.
  - Trên panel thoái hóa (FrozenLake, mọi nhãn ">x50"), quy tắc ngoại suy thô là đủ.
- Đây là **hậu kiểm** trên panel đã mở, không có kiểm định đăng ký trước. Trên panel 6, `b2_plus_prefix_fqe` là comparator đăng ký trước (thẻ T8).

## Tái lập

```
python -m scripts.analysis.baseline_prefix_fqe features --workers 10   # khoảng 1 giờ trên 12 nhân
python -m scripts.analysis.baseline_prefix_fqe evaluate               # khoảng 12 phút
```

`manifest.json` ghi hash của spec, mọi file đặc trưng đầu vào và mọi output. Test: `tests/test_baseline_prefix_fqe.py`.

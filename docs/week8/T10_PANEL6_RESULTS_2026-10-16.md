# T10 — Panel 6: nhãn và phân tích đã đăng ký (16/10/2026)

Thẻ T10 của kế hoạch lên hội nghị (bộ agent kit, lưu ngoài repo).
- **H3:** Linh cho phép bước nhãn của panel 6 ngày 16/10 với protocol gần nhất đã xác nhận, `49c27071…` (kèm amendment A1).
  - Trước khi chạy đã kiểm tra lại: mọi seal trỏ về protocol này; file đã niêm phong còn nguyên; chưa có kết cục nào.
- **Đã chạy, không sửa gì:**
  - `run_oeh_panel6_20261015 evaluate --panel mdpp|labmaze`: 115 200 job learner mỗi họ.
  - `run_oeh_panel6_20261015 analyze`.
  - Phần mô tả của TabPFN: `scripts/analysis/panel6_tabpfn_describe.py`. Dự báo TabPFN đã niêm phong trước nhãn.
- **Kết quả:** [`artifacts/oeh_panel6_2026-10-15/`](../../artifacts/oeh_panel6_2026-10-15/): `result.json`, `summary.csv`, `by_learner.csv`, `murphy.csv`, `gain_by_distance.csv`, `label_variant_tau099.csv`, `tabpfn_descriptive.csv`.

## Cổng đã đăng ký (192 MDP, max-t một phía với SE gốc, 10 000 bootstrap)

| Họ cổng | Đối chứng | Chỉ số | Lợi thế TB của OEH | Cận dưới đồng thời 95% | Kết luận |
|---|---|---|---:|---:|---|
| **F1** | B2 + prefix/FQE | `ibs_ge` | +0,0315 | +0,0247 | **GO** |
| | | `ll_ext` | +0,1034 | +0,0808 | |
| **F2** | B2 | `ibs_ge` | +0,0714 | +0,0616 | **GO** |
| | | `ll_ext` | +0,2322 | +0,2025 | |
| F3 (ước lượng, không có kết luận) | B2 + prefix/FQE | regret | +2,61 | +1,29 | — |
| | B2 | regret | +7,91 | +6,13 | — |

Tương phản của panel 5 (OEH so với head vững) **không** được kiểm định lại, đúng như protocol.

## Nhãn

Nhãn không suy biến. Tỉ lệ ">x50" theo learner:

| Họ | BC | CFQI | IQL |
|---|---:|---:|---:|
| `mdpp` | 84% | 22% | 46% |
| `labmaze` | 82% | 32% | 66% |

## Mô tả (không phải kiểm định; lợi thế tương đối của OEH, gộp hai họ)

| So với | IBS | LL | `ibs_ge` | `ll_ext` |
|---|---:|---:|---:|---:|
| B2 | +36,7% | +19,9% | +43,8% | +32,8% |
| B2 + prefix/FQE | +18,1% | +4,6% | +25,6% | +17,9% |
| TabPFN v2 trên B2 | +31,4% | +13,5% | +37,5% | +25,7% |
| Head vững (B2 + D) | +14,8% | +7,4% | +20,2% | +14,5% |
| S4 (T4) | +8,0% | +5,0% | +10,4% | +5,9% |
| **OEH-F (head đầy đủ, không cổng)** | **−3,3%** | **−3,6%** | **−6,6%** | **−5,5%** |
| LCE thô | +10,9% (KTC chứa 0) | +22,1% | +29,0% | +38,0% |
| XGBoost-CF/108 | −4,5% | +24,8% | −6,5% | +27,0% |

## Đọc kết quả, cả điểm tốt lẫn điểm xấu

1. **Câu hỏi mới của panel 6 được trả lời là có.** Trên hai họ có bộ sinh instance thật, chưa dùng ở đâu, OEH dự báo tốt hơn baseline "pilot + FQE + ngoại suy" mạnh nhất. Lợi thế nằm ở đúng vùng ngân sách mà pilot không với tới (F1 GO), và OEH cũng hơn B2 (F2 GO). Đây là bằng chứng **đăng ký trước** đầu tiên cho endpoint ngoại suy.
2. **Cổng dịch chuyển gây hại ở panel 6.**
   - OEH-F (không cổng) tốt hơn OEH: −3,3% IBS và −3,6% LL, KTC không chứa 0.
   - Gần như toàn bộ đến từ `mdpp`, nơi 71% nhóm có d ≥ 3, nên cổng chuyển sang head vững (−6,5% IBS, −5,2% LL). Trên `labmaze`, hai bản hòa nhau.
   - Ngay cả head vững cũng kém OEH ở đây. Vậy cổng an toàn ở panel 5, nhưng quá thận trọng ở panel 6. **Bài phải báo kết quả này.**
3. **Head vững thua OEH (+14,8% IBS) trên cả hai họ.** Ở panel 5, OEH so với head vững là NO-GO. Kết luận đúng là: so sánh với head vững phụ thuộc panel, không phải lợi thế chung.
4. **LCE thô (đường cong học trên prefix của pilot) rất mạnh về IBS trên `mdpp`** (OEH kém 32,7%), dù LL của nó kém. Ngoại suy đường cong đơn giản đủ tốt khi MDP có thưởng dày; nên ghi điều này như một giới hạn.
5. **Endpoint đầy đủ cho cùng một bức tranh** (OEH: IBS 0,0971 / LL 1,039; B2 + prefix/FQE: 0,1186 / 1,089; B2: 0,1533 / 1,297). Biến thể nhãn τ = 0,99 cũng vậy:
   - OEH: IBS 0,0964 / LL 0,989.
   - B2: IBS 0,1554 / LL 1,233.
   - B2 + prefix/FQE: IBS 0,1254 / LL 1,102.

## Việc còn lại

**H4:** một người khác chấm lại độc lập trên máy khác, từ các file đã commit: `predictions_preoutcome.csv`, `labels.csv`, `protocol.json`. Lệnh: `python -m scripts.analysis.run_oeh_panel6_20261015 analyze`. Kết quả phải trùng `result.json`. Test `tests/test_oeh_panel6_results.py` đã tái tạo các cổng từ `scored.csv.gz` ở máy này.

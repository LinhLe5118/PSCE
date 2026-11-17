# T6 — Độ nhạy theo định nghĩa nhãn (15/10/2026, hậu kiểm, mô tả)

Thẻ T6 của kế hoạch lên hội nghị (bộ agent kit, lưu ngoài repo), trả lời W8. Nguồn: [`artifacts/week8/threshold_sensitivity_2026-10-15/`](../../artifacts/week8/threshold_sensitivity_2026-10-15/), script `scripts/analysis/threshold_sensitivity.py`, hình `fig_threshold_sensitivity`.

## Cách làm

- **Dữ liệu:** dùng `normalized_return` theo từng seed của 11 panel đã mở (cohort 7, 8; Colosseum 2027, 2028; SPIBB; 4 panel kiểm định cuối; 2 họ panel 5).
- **Gán lại nhãn:** ngưỡng thành công τ ∈ {0,90; 0,95; 0,99} × số seed k ∈ {7, 8, 9}/10, giữ nguyên quy tắc hậu tố.
- **Kiểm tra trước:** ở (0,95; 8) tái tạo **đúng 11 988/11 988** nhãn chính thức và đúng cờ `success` đã lưu.
- **Chấm lại:** dự báo đã niêm phong được chấm lại cho từng định nghĩa nhãn.

**Giới hạn:**
- Dự báo niêm phong nhắm (0,95; 8/10). Các ô khác chỉ đo **độ vững của phép so**, không phải dự báo được fit cho target đó.
- Không fit lại được cho từng target: meta-training thiếu outcome theo seed của E1 (120/720 MDP).

**Tỉ lệ ">x50"** (trung bình các panel): 66–68% ở τ = 0,90; 72–76% ở 0,95; 85–89% ở 0,99.

## Kết quả (cải thiện tương đối so với B2; ° = KTC 95% chứa 0)

| Nhóm | τ = 0,90 | τ = 0,95 | τ = 0,99 |
|---|---|---|---|
| Kiểm định cuối, ngoài gộp (OEH) | IBS +23 → +28%, LL +11 → +17% | IBS +37 → +38%, LL +21 → +24% | IBS +30 → +32%, LL +13 → +14% |
| Kiểm định cuối, internal (OEH) | IBS +5 → +8%, LL +4 → +5% | IBS +14 → +16%, LL +8% | **IBS −8 → −10%, LL −5 → −8% (kém có ý nghĩa)** |
| Panel 5 gộp (OEH) | IBS −4° → +2°, LL −0° → +7% | IBS +4° → +18%, LL +9 → +14% | IBS +41 → +50%, LL +26 → +33% |
| Phát triển, internal (OEH-F) | IBS +7 → +12%, LL +2° → +5% | IBS +14 → +17%, LL +8 → +11% | **IBS −12 → −14%, LL −9 → −11% (kém có ý nghĩa)** |
| Phát triển, ngoài (OEH-F) | IBS +11 → +17%, LL +0° → +6% | IBS +23 → +24%, LL +9% | IBS +19%, LL +2° → +3° |

Mỗi ô ghi khoảng giá trị qua k = 7, 8, 9.

## Đọc kết quả

1. **Ngoài generator, phép so vững:** OEH hơn B2 ở cả 9 tổ hợp, khoảng tin cậy không chứa 0, cả IBS lẫn LL.
2. **Trong generator, phép so đảo chiều ở τ = 0,99:** dự báo niêm phong của OEH (và OEH-F) kém B2 có ý nghĩa.
   - Giải thích hợp lý nhất: OEH sắc nét hơn, nên khi target lệch khỏi target nó được fit cho, nó chịu phạt nặng hơn B2 (B2 phẳng hơn).
   - Muốn khẳng định cần fit lại cho τ = 0,99, nhưng chưa làm được (thiếu outcome theo seed của E1).
   - **Kết quả âm này phải nằm trong bài**, kèm giải thích và giới hạn.
3. **Ở τ = 0,90 lợi thế nhỏ đi** vì nhiều ca thành công sớm hơn và mọi phương pháp đều dễ đoán hơn.
4. **Theo tiêu chí dừng ở mục 7 của hướng dẫn** ("kết quả thay đổi mạnh theo ngưỡng thì đổi endpoint chính sang tổ hợp ổn định nhất đã đăng ký trước"): ngoài generator ổn định; trong generator nhạy ở τ = 0,99. Đề xuất cho panel 6 là đăng ký trước một endpoint phụ ở τ = 0,90 và τ = 0,99, với forecaster được fit cho đúng target đó (cần outcome của E1 hoặc một meta-training không có E1).

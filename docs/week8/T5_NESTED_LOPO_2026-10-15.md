# T5 — Nested LOPO của quy trình chọn OEH (15/10/2026, hậu kiểm, mô tả)

Thẻ T5 của kế hoạch lên hội nghị (bộ agent kit, lưu ngoài repo); trả lời W6 ("có phải chọn đến khi qua?"). Nguồn: [`artifacts/week8/nested_lopo_v2_selection_2026-10-15/`](../../artifacts/week8/nested_lopo_v2_selection_2026-10-15/), script `scripts/analysis/nested_lopo_v2_selection.py`.

## Cách làm

**Dựng lại quy trình chọn đã ghi** (M6 rồi lưới M1 × M4: 2 × 2 lựa chọn head × λ_max × ε_max × d0, cùng ràng buộc và quy tắc ưu tiên phương án đơn giản) thành một hàm của tập panel được phép nhìn.
- Khi dùng đủ 5 panel, hàm tái tạo **đúng** lựa chọn đã ghi (head đầy đủ fit trên dòng hữu hạn, phần hữu hạn của head vững, λ_max = 1, ε_max = 0,02, d0 = 3), và đúng điểm mục tiêu tới 1e-12.
- Kiểm tra này là một test chạy trong CI.

**Vòng ngoài:** lần lượt giữ lại từng panel trong 5 panel chọn, chạy lại **toàn bộ** quy trình trên 4 panel còn lại, rồi chấm cấu hình được chọn trên panel giữ lại.
- M2 và M3 đã bị loại theo quy tắc riêng, nên không chạy lại.
- File dự báo của bước phát triển (`heads/predictions.csv.gz`, khớp hash trong `fit_manifest.json`) nay được commit để kết quả tái lập được.

## Kết quả (điểm = trung bình (rel IBS + rel LL)/2 so với OEH-F; âm = tốt hơn)

| Panel giữ lại | Cấu hình chọn ở vòng trong | Điểm vòng ngoài | Điểm khi fit một lần trên cả 5 panel |
|---|---|---:|---:|
| Colosseum 2027 | λ 0,75; ε 0,05; d0 3 | −8,75% | −8,81% |
| Colosseum 2028 | λ 1; ε 0; d0 4 | −7,79% | −8,36% |
| SPIBB | λ 1; ε 0,1; d0 3 | −0,10% | −2,49% |
| Cohort 7 (trong generator) | trùng cấu hình cuối | +1,15% | +1,15% |
| Cohort 8 (trong generator) | λ 0,75; ε 0,05; d0 3 | +0,43% | +0,36% |

**Tóm tắt ba panel ngoài:**
- Vòng ngoài: **−5,55%**.
- Fit một lần trên cả 5 panel: −6,55%.
- Bên trong vòng chọn: −6,91%.

**Độ lạc quan của bước chọn:** khoảng 1,0 điểm phần trăm (so với fit một lần) và khoảng 1,4 điểm (so với điểm bên trong).

## Đọc kết quả

1. **Bước chọn không tạo ra lợi thế giả.** Ước lượng trung thực của cả quy trình (−5,6%) gần ước lượng một lần fit (−6,6%), và cùng chiều với kiểm định niêm phong sau đó.
2. **Lựa chọn không ổn định ở mức tham số** (λ, ε, d0 thay đổi theo panel giữ lại), nhưng **ổn định ở mức cấu trúc**: luôn là head đầy đủ fit trên dòng hữu hạn cộng phần hữu hạn của head vững, λ_max ≥ 0,75.
3. **Giữ lại SPIBB thì lợi ích gần như bằng 0.** Điều này khớp với việc cổng không phát hiện được dịch chuyển do behavior (kiểm định cuối và panel 5).
4. Đây là câu trả lời có số cho câu hỏi của reviewer thống kê (W6). Cận lý thuyết kiểu T2 vô nghĩa với 3 panel, còn ước lượng thực nghiệm này thì không.

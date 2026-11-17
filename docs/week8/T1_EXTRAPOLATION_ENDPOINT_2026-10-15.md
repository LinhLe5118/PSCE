# T1 — Endpoint ngoại suy (15/10/2026, hậu kiểm, mô tả)

Thẻ T1 của kế hoạch lên hội nghị (bộ agent kit, lưu ngoài repo); trả lời W1 (mức tối thiểu) của hướng dẫn lên hội nghị. Nguồn: [`artifacts/week8/extrapolation_endpoint_2026-10-15/`](../../artifacts/week8/extrapolation_endpoint_2026-10-15/), script `scripts/analysis/extrapolation_endpoint.py`.

## Định nghĩa (ghi trước khi chạy)

Pilot có x10 chuyển trạng thái, nên phần dự báo cho x1–x5 một phần là hồi cứu. Vùng ngoại suy gồm:

| Điểm | Định nghĩa |
|---|---|
| `ibs_ge` | Brier tích phân trên các ngưỡng **bằng hoặc lớn hơn** pilot: P(E ≤ x10), P(E ≤ x20), P(E ≤ x50). Ngưỡng cuối mang lớp ">x50" (đúng chữ của thẻ) |
| `ibs_gt` | Biến thể chặt: chỉ x20 và x50 |
| `ll_ext` | Log loss của luật gộp {≤ x10, x20, x50, > x50}, sàn 1e-12 |

**Tương đương:** với đủ 6 ngưỡng và 7 lớp, bộ chấm tái tạo mọi `summary.csv` đã niêm phong, sai lệch tối đa 2,2e-16 trên 159 cặp panel × phương pháp (`equivalence.csv`).

## Kết quả: lợi thế của OEH so với B2 không đến từ phần hồi cứu

Cải thiện tương đối so với B2 matched (trung bình theo MDP):

| Phạm vi | IBS đầy đủ | `ibs_ge` | `ibs_gt` | LL đầy đủ | `ll_ext` [KTC 95%, tuyệt đối] |
|---|---:|---:|---:|---:|---:|
| Kiểm định cuối, internal (168) | 15,7% | 17,4% | 16,8% | 8,3% | 11,2% [0,037; 0,063] |
| Kiểm định cuối, ngoài gộp (228) | 37,7% | 39,8% | 38,4% | 23,4% | 26,2% [0,108; 0,182] |
| Panel 5 gộp (72) | 10,0% | 13,4% | 17,0% | 12,3% | 9,5% [0,000; 0,071] |
| Cohort 7, OEH-F (84) | 16,0% | 17,1% | 16,0% | 10,2% | 11,5% [0,031; 0,072] |

**So với head vững (B2 + D), vùng ngoại suy:**
- Ngoài gộp của kiểm định cuối: `ibs_ge` +23,7%, `ll_ext` +12,8% [0,032; 0,088].
- Panel 5 gộp: `ibs_ge` +0,8%, `ll_ext` +3,0% [−0,026; 0,047]. Hòa, đúng với NO-GO đã đăng ký.
- Gambler: head vững tốt hơn (`ibs_ge` −10,3%).

## Đọc kết quả

- **Câu hỏi của reviewer** "chạy thử learner trên pilot là biết": phần lợi thế nằm ở vùng pilot không với tới (x20, x50, >x50). Ở mọi phạm vi, lợi thế trên vùng này **bằng hoặc lớn hơn** trên endpoint đầy đủ.
- Đây là **hậu kiểm** trên panel đã mở. Muốn thành endpoint chính thì phải đăng ký trước trên panel 6.
- Vùng ngoại suy vẫn bị lớp ">x50" chi phối (71–100% nhãn). Bảng theo panel trong `scores_by_panel.csv` và `gains.csv` cho đủ số liệu.

# T7 — Lý thuyết (15/10/2026) — **đã qua chốt H5**

Thẻ T7 của kế hoạch lên hội nghị (bộ agent kit, lưu ngoài repo); trả lời W4.
- Phụ lục: [`paper/appendix_theory.tex`](../../paper/appendix_theory.tex), header `status: checked`; biên dịch được (4 trang).
- Kiểm tra số: `scripts/analysis/theory_checks_t7.py` → [`artifacts/week8/theory_checks_2026-10-15/`](../../artifacts/week8/theory_checks_2026-10-15/).

**H5 đã được Linh duyệt ngày 15/10/2026.** Từ nay T3, T3-τ, T3-LL và T1 (dạng "hằng cục bộ + cổng bão hòa") được dùng trong bài, kèm giả thiết và phạm vi ghi trong phụ lục. Bài **không** được viết "OEH nhất quán".

## Cập nhật 15/10, tối: đã lấp mọi chỗ `TODO(proof)` trước H5

Theo quyết định H5 ("để agent lấp trước rồi mới duyệt"):
- Những gì chứng minh được thì đã chứng minh.
- Phần còn lại được nêu thành giả thiết hoặc ghi rõ là nằm ngoài phát biểu.
- Không còn chỗ `TODO(proof)` nào; test kiểm tra điều này.

| # | Phát biểu | Trạng thái |
|---|---|---|
| **Định lý T3** | Regret của luật Bayes plug-in ≤ 2λ·max_j \|F_j − F\*_j\| trên từng dòng; theo trung bình ≤ 2λ√(6Δ_IBS) | Chứng minh đầy đủ (từ trước) |
| **Mệnh đề T3-τ** (mới) | Regret của **luật τ đã đăng ký** ≤ (b₅ + λ)·[P(‖F − F\*‖∞ ≥ m) + P(min_j \|F\*_j − τ\| < m)] ≤ (b₅ + λ)·[6Δ/m² + P(lề < m)] với mọi m | Chứng minh đầy đủ. Khi F và F\* cùng phía τ ở mọi ngân sách thì quyết định trùng nhau, và chi phí nằm trong [0, b₅ + λ] |
| **Hệ quả T3-LL** (mới) | Log loss thừa = E KL(p\*‖p). Regret Bayes plug-in ≤ λ√(2Δ_LL) | Chứng minh đầy đủ (Pinsker + Jensen) |
| **Mệnh đề T1** (**viết lại, yếu hơn nhưng đúng**) | Head cây là **hằng cục bộ**: nếu đặc trưng hội tụ trong tập số thực mở rộng và giới hạn không nằm trên ngưỡng tách, mọi head rốt cuộc **bằng hẳn** giá trị tại giới hạn. Luật OEH hội tụ về cổng tại d\* | Chứng minh đầy đủ, đã đối chiếu code của ba thành phần head |
| T2 | Cận chọn ứng viên LOPO: vô nghĩa ở quy mô này | Như cũ |
| Mệnh đề 2 | Đảm bảo an toàn của cổng (Jensen) | Như cũ |

### Điều quan trọng phát hiện khi lấp T1

Bản nháp cũ ("forecaster hội tụ về head tại đặc trưng oracle") **phát biểu quá mạnh**:
1. **Cổng bão hòa.** Hai thống kê B2 (`log_pilot_transitions`, `log1p_min_positive_sa_count`) tăng không bị chặn theo n, nên d → ∞. Khi đó OEH dồn hẳn về head vững, với xác suất kiểm duyệt của head đầy đủ. Đây là đúng thiết kế ngoài vùng huấn luyện, nhưng nghĩa là "nhất quán" là cách nói sai.
2. **OEH-B (G2, G6)** là trung bình Monte Carlo của K = 64 lần rút posterior. Số uniform tiêu thụ phụ thuộc posterior (Dirichlet và bộ lọc dynamic range), nên không có giới hạn hầu chắc chắn. Chỉ bản lý tưởng K → ∞ hội tụ (theo Mệnh đề 1 của phụ lục).
3. **Chẩn đoán D (G7, G8)** trộn prefix cố định 2SA, 5SA với pilot đang tăng (độ dốc log-log tiến về 0 hoặc phân kỳ). Head vững lại dùng các cột này.
4. **G4–G5 (OEH-H)** hội tụ nếu ở θ₀ không có hòa action tối ưu và không đại lượng nào nằm đúng ngưỡng. **G3** (độ tập trung empirical Bayes) chỉ là giả thiết.

**Hệ quả cho bài:**
- **Được** viết: head hằng cục bộ; cổng bão hòa với pilot lớn.
- **Không được** viết: OEH nhất quán.

## Số trên 4 panel cuối (λ = 100, τ = 0,8; F\* = luật plug-in của 10 seed huấn luyện; đơn vị ngân sách)

| Cận | Giá trị thực | Cận | Ghi chú |
|---|---:|---:|---|
| T3, regret Bayes plug-in | 4,3–35,9 | dạng L∞ 24–101; dạng IBS 51–234 | đúng ở mọi panel |
| T3-τ, regret của luật τ | 4,4–34,2 | dạng biến cố 55–127; dạng Markov ≥ 86 | dạng Markov gần cận tầm thường b₅ + λ = 150 |
| T3-LL | 4,3–35,9 | 62–155 | đúng ở mọi panel |

Mọi cận đều đúng trên dữ liệu tổng hợp và trên 4 panel, nhưng lỏng. Giá trị của chúng là nối **thứ tự** của điểm xác suất với thứ tự regret của quyết định, không dự đoán độ lớn của regret.

## Checklist cho người kiểm (H5)

1. **T3:** bước phân rã Brier cần F đo được theo 𝒢. Với 𝒢 = (MDP, behavior, pilot), F\* là luật theo seed, vì pilot độc lập với các seed huấn luyện khi đã biết MDP.
2. **T3-τ:** câu "cùng phía τ ở mọi j ⇒ cùng quyết định". Luật τ chỉ phụ thuộc vectơ chỉ báo 1{F_j ≥ τ}.
3. **T3-LL:** điều kiện p_k > 0. Head XGBoost softprob luôn dương; luật có cổng có thêm ε(d)·π. Log loss khi chấm có sàn 1e-12.
4. **T1:** đọc phần chứng minh "hằng cục bộ" cạnh code của ba thành phần head (các đường dẫn ghi trong chứng minh), và bảng các khối cột trong Remark.
5. Nếu đồng ý thì đổi dòng 1 của file thành `% status: checked`. Khi đó T3, T3-τ, T3-LL và T1 (dạng mới) được phép dùng trong bài.

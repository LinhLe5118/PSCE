# D3 — Mệnh đề nhất quán của OEH-B và giới hạn của nó (14/10/2026)

**Phạm vi:** ghi chú lý thuyết cho mục D3 trong kế hoạch bàn giao. Nó áp dụng cho **luật thô OEH-B** (`oeh/predictive.posterior_outcomes` + `requirement_forecast`), không áp dụng cho head đã fit (AFT / ngưỡng / multiclass) và không áp dụng cho các cột OEH-H. Không có số liệu mới trong tài liệu này.

## 1. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| θ = (P, r, β) | MDP hữu hạn (S trạng thái, A hành động), reward kỳ vọng và behavior. θ₀ là bộ thật. |
| D_n | Pilot gồm n chuyển trạng thái, sinh bởi β₀ trên θ₀. Trong dự án n = 10·SA. |
| ℓ | Learner đã đăng ký (BC, CFQI, IQL): một ánh xạ đo được từ dataset sang policy. |
| E_ℓ(θ) | Biến cố yêu cầu dữ liệu: mức nhỏ nhất k ∈ {x1, …, x50} mà ≥ 8/10 seed đạt NR ≥ 0,95 ở k và mọi mức lớn hơn, hoặc `>x50`. Ngẫu nhiên theo các dataset huấn luyện sinh từ θ. |
| q_θ(k) = Pr_θ(E_ℓ = k) | Luật thật của biến cố khi dữ liệu sinh từ θ. Nhãn quan sát được là **một** lần rút từ q_{θ₀}. |
| Π_n | Posterior của θ cho trước D_n: Dirichlet trên từng hàng P(s,a,·) và β(s,·), reward theo `reward_prior`. |
| F_n(k) | Luật OEH-B: ∫ q_θ(k) dΠ_n(θ), ước lượng bằng M = 64 draw (KT-smoothing) sau bộ lọc dynamic-range. |

## 2. Mệnh đề (tính nhất quán của luật thô)

Giả sử:

- **(A1) Phủ:** mọi cặp (s, a) có xác suất dương dưới β₀ và reachable từ ρ được thăm vô hạn lần khi n → ∞. Với β₀ có full support trên các trạng thái reachable, điều này đúng hầu chắc chắn (MDP hữu hạn, pilot dạng episode khởi động lại).
- **(A2) Prior không áp đảo:** độ tập trung Dirichlet α_n(s, a) (ước lượng theo kiểu empirical Bayes trong `fit_posterior`) thỏa α_n(s, a) / N_n(s, a) → 0 hầu chắc chắn.
- **(A3) Reward:** reward của mỗi khóa (s, a, s') là tất định, như generator của dự án và như kernel Colosseum dùng reward trung bình. Do đó khóa nào đã quan sát thì biết đúng giá trị.
- **(A4) Không có điểm biên:** Pr_{θ₀}(NR_ℓ = 0,95 tại một mức nào đó) = 0. Khi đó θ ↦ q_θ liên tục tại θ₀. Theo định lý ánh xạ liên tục, điều này xảy ra khi NR là hàm liên tục của (P, r, dataset) và tập biên có độ đo 0.
- **(A5) Bộ lọc:** θ₀ có dynamic range lớn hơn ngưỡng 0,01/(1−γ), nên một lân cận của θ₀ không bị bộ lọc loại.

Khi đó, với mỗi k và với M → ∞ draw: F_n(k) → q_{θ₀}(k) hầu chắc chắn khi n → ∞.

**Phác thảo chứng minh:**

1. Với mỗi hàng có số đếm N_n(s, a) → ∞, posterior Dir(α_n·base + counts) có trung bình (α_n·base + counts) / (α_n + N_n). Theo (A2), giá trị này hội tụ về tần suất thực nghiệm, mà tần suất thực nghiệm → P₀(s, a, ·) theo luật số lớn. Phương sai là O(1 / N_n).
2. Do đó Π_n hội tụ yếu về δ_{θ₀} trên các thành phần mà q_θ phụ thuộc, tức các hàng reachable dưới β₀; các hàng không reachable không ảnh hưởng tới phân phối dataset. Theo (A5), bộ lọc không cắt mất khối lượng gần θ₀.
3. q_θ là kỳ vọng của một hàm bị chặn (chỉ báo của biến cố) dưới phân phối dữ liệu, mà phân phối này liên tục theo θ. Theo (A4), θ ↦ q_θ liên tục tại θ₀.
4. Hội tụ yếu kết hợp tính liên tục và bị chặn cho ∫ q_θ dΠ_n → q_{θ₀}. Sai số Monte Carlo với M draw là O_p(M^{−1/2}). KT-smoothing thêm độ lệch O(1/M).

**Hệ quả trực tiếp:**

- Mệnh đề nói về **luật** q_{θ₀}, không nói về nhãn đơn lẻ. Brier/log loss tốt nhất có thể đạt vẫn còn phần nhiễu nội tại của một lần rút 10 seed. Ở cohort 7, 21% dòng primary có số seed thành công giảm giữa hai mức liên tiếp.
- Sai số còn lại của head đã fit trong dự án đến từ hai nguồn: pilot hữu hạn (n = 10·SA, xa tiệm cận), và việc head học ánh xạ từ nhiều tín hiệu chứ không chỉ F_n.

## 3. Vì sao luật thô yếu khi pilot thưa (giải thích, không phải định lý)

1. **Vi phạm (A1)/(A3) ở pilot hữu hạn.** Với `reward_prior = "unseen_zero"`, mọi khóa reward chưa quan sát được gán 0.
   - Nếu pilot chưa chạm reward lớn (RiverSwim: đầu sông xa; DeepSea: góc đích), mọi MDP hậu nghiệm có dynamic range gần 0.
   - Bộ lọc (A5) khi đó loại toàn bộ proposal. Kết quả là `draws_returned = 0`, và luật mặc định đặt toàn bộ khối lượng vào `>x50`.
   - Trên Colosseum xác nhận có 42/360 nhóm như vậy. Phân tích post hoc cho thấy trong các nhóm này nhãn thật gần như luôn là `>x50` (98,9% ở RiverSwim), tức luật mặc định đúng.
2. **Sai lệch có hệ thống khi pilot thấy một phần reward.**
   - Trên các nhóm RiverSwim *có* draw, learner thường đạt ngay ở x1, trong khi các cột OEH-H trên mô hình posterior mean đẩy dự báo về `>x50`. Mệnh đề không mâu thuẫn với điều này, vì nó chỉ là kết quả tiệm cận.
   - Kiểm chứng: thay reward prior (ví dụ prior lạc quan có ràng buộc) trên dữ liệu chưa mở.
3. **Liên hệ với effective horizon.** OEH-H dùng độ sâu lập kế hoạch trên mô hình posterior mean. Mối liên hệ hình thức với EH của Laidlaw et al. chỉ đúng có điều kiện (xem [H1 glossary](H1_THEORY_GLOSSARY_2026-10-08.md), mục giả thiết): EH được định nghĩa cho học tương tác và GORP/SQIRL, không phải cho learner offline cố định. Ghi chú này **không** chứng minh được rằng OEH-H xấp xỉ sample complexity của CFQI/IQL/BC.

## 4. Câu được phép viết trong bài

> "Under full support of the logging behavior, deterministic rewards and a non-dominating prior, the raw posterior-predictive law of OEH-B is consistent for the learner's requirement law as the pilot grows (sketch in the supplement). With the pilot sizes used here the law is far from this regime, which is why a fitted head is needed and why unobserved high rewards (RiverSwim) remain a failure mode."

Không viết: "OEH has convergence guarantees" (head đã fit không có), hay "OEH approximates the effective horizon".

## 5. Mệnh đề 2: đảm bảo theo từng dòng của OEH v2 (bổ sung 14/10, sau khi khóa spec)

**Ký hiệu.**
- p = (p₀, …, p₆) là OEH current. q_O là head chỉ fit trên dòng hữu hạn (108 trường). q_R là phần hữu hạn của B2 + legacy. π là phân phối lớp của learner trên meta-training.
- d là khoảng cách 10-NN tới meta-training. λ = λ(d) và ε = ε(d) là các trọng số trong `oeh_v2_forecast.py`.
- Đặt h_O = [(1 − p₆)·q_O, p₆] và h_R = [(1 − p₆)·q_R, p₆], tức hai bộ dự báo đầy đủ có chung P(>x50).

**Đồng nhất thức.** Vì h_O và h_R có chung p₆:

  OEH v2 = (1 − ε)(1 − λ)·h_O + (1 − ε)λ·h_R + ε·π.

Như vậy OEH v2 là **hỗn hợp ba thành phần, với trọng số chỉ phụ thuộc vào d**: không phụ thuộc vào nhãn, cũng không phụ thuộc vào panel.

**Mệnh đề 2.** Với mọi dòng (mọi nhãn y, mọi pilot, dưới **bất kỳ** dịch chuyển phân phối nào):

1. **Lồi.** Gọi L là log loss −log p(y) hoặc Brier tích phân Σ_j (F_j − 1[y ≤ j])²/6. Cả hai đều lồi theo vector xác suất, vì F_j tuyến tính theo p. Do đó

   L(v2) ≤ (1 − ε)(1 − λ)·L(h_O) + (1 − ε)λ·L(h_R) + ε·L(π).

2. **Chặn trên tuyệt đối của log loss.**
   - −log v2(y) ≤ −log(ε·π(y)), luôn hữu hạn khi d > 0 (khác với XGBoost-CF, vốn có thể gán xác suất 0).
   - Với y = >x50: −log v2(y) ≤ −log p₆ − log(1 − ε) ≤ −log p₆ + 0,0203 nat (vì ε ≤ 0,02).
3. **Không tệ hơn thành phần tốt hơn quá một hằng số.** Với y hữu hạn:

   −log v2(y) ≤ −log(1 − ε) + min{ −log h_O(y) − log(1 − λ), −log h_R(y) − log λ }.

**Chứng minh.**
- (1) là bất đẳng thức Jensen cho hàm lồi.
- (2) và (3) suy ra từ v2(y) ≥ ε·π(y), v2(y) ≥ (1 − ε)(1 − λ)·h_O(y) và v2(y) ≥ (1 − ε)λ·h_R(y).

**Ý nghĩa và giới hạn.**
- Lấy kỳ vọng theo bất kỳ phân phối kiểm định nào, (1) cho: rủi ro của v2 ≤ trung bình có trọng số d của rủi ro ba thành phần.
  - Khi d ≥ 3 (λ = 1, ε = 0,02), v2 gần như là head lai "P(>x50) của OEH + mức hữu hạn của legacy". Đây là head mà phân tích hoán đổi cho thấy tốt nhất ngoài generator.
  - Khi d → 0, v2 trở về head OEH theo hai giai đoạn.
- Mệnh đề **không** nói v2 tốt hơn OEH current. Nó chỉ nói việc chọn trọng số theo d không bao giờ tệ hơn trộn cố định, và loss bị chặn.
  - Việc v2 thực sự tốt hơn là câu hỏi thực nghiệm của kiểm định niêm phong (bước 1–2).
- Tính lồi cho từng dòng là lý do có thể chọn (λ_max, ε_max, d0) trên panel đã mở mà không sợ "gãy" trên panel mới: rủi ro trên panel mới bị kẹp giữa rủi ro của các thành phần.

**Câu được phép viết:**

> "OEH v2 is a three-component mixture whose weights depend only on the pilot's distance to the training panel; by convexity of the log and Brier losses its loss on every forecast is at most the distance-weighted average of its components' losses, and its log loss is bounded by −log(ε(d)π(y)), under any distribution shift."

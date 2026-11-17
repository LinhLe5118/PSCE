# T2 — Spec của baseline "prefix pilot + FQE + ngoại suy" (viết trước khi chạy, 15/10/2026)

Thẻ T2 trong [`docs/agent_kit/TASKS.md`](../agent_kit/TASKS.md). Spec này cố định **mọi** bậc tự do trước lần chạy đầu tiên. Mọi thay đổi sau đó là biến thể mới, phải ghi vào `development_log.csv`.

## 1. Đầu vào

- Pilot x10 của mỗi nhóm MDP–behavior (`run_oeh_final_120.pilot_rows`). Mảng: `state, action, reward, next_state, terminal, trajectory_id, timestep`.
- Không đọc stream huấn luyện, kernel thật hay nhãn.

## 2. Cross-fitting và seed giả lập

- **Fold:** K = 5 fold theo **quỹ đạo**. Hoán vị các `trajectory_id` bằng `numpy.random.default_rng` với seed = 8 byte đầu của `sha256("prefix-fqe-v1|<mdp>|<behavior>")`; fold = thứ hạng sau hoán vị mod 5.
- **Pool huấn luyện của fold k:** các chuyển trạng thái của quỹ đạo ngoài fold k, theo thứ tự gốc của pilot (khoảng 8·|S||A|).
- **Seed giả lập r ∈ {0, 1}:**
  - r = 0: pool theo thứ tự gốc;
  - r = 1: bootstrap các quỹ đạo của pool (lấy có hoàn lại, cùng số quỹ đạo, cùng rng), ghép theo thứ tự đã lấy.
  - Tổng cộng 5 × 2 = **10 seed giả lập** cho mỗi mức ngân sách, mỗi seed có fold đánh giá riêng.
- **Prefix:** b ∈ {1, 2, 4, 8} × |S||A| chuyển trạng thái đầu của pool. Nếu pool ngắn hơn thì dùng cả pool.

## 3. Learner và FQE

- **Learner:** đúng các learner đã đăng ký trong `oeh.tabular.CORE_LEARNERS` (BC, CFQI, IQL), γ = 0,95, huấn luyện trên `count_model` của prefix.
- **FQE dạng bảng** trên fold đánh giá:
  - mô hình thực nghiệm của fold (p̂ trên các chuyển trạng thái không kết thúc, r̂);
  - cặp (s, a) chưa thấy trong fold có Q = 0 (reward ≥ 0, nên đây là bi quan);
  - lặp Q ← r̂ + γ p̂ V^π với V^π(s) = Σ_a π(a|s) Q(s, a), tới khi sai khác < 1e-10 hoặc tối đa 10 000 vòng;
  - phân phối đầu ρ̂ = trạng thái ở `timestep == 0` trong fold; Ĵ(π) = ρ̂ · V^π.
- **Ước lượng J\* và J_random** (không biết từ pilot):
  - π̂\* là chính sách tối ưu của mô hình thực nghiệm trên **toàn pilot**;
  - Ĵ\* = max(Ĵ(π̂\*), max Ĵ của mọi policy learner ở mọi prefix của cùng fold);
  - Ĵ_random = FQE của chính sách đều trên fold.
- **Normalized return ước lượng:** NR̂ = (Ĵ(π) − Ĵ_random) / (Ĵ\* − Ĵ_random). Nếu mẫu ≤ 1e-9 thì NR̂ = 0.
- **Thành công:** NR̂ ≥ 0,95.

## 4. Đường cong và luật 7 lớp

- **Đường cong:** với mỗi learner có 40 kết quả Bernoulli (4 mức × 10 seed). Khớp p(b) = σ(α + β·log b), ràng buộc β ≥ 0, bằng MLE có phạt Gauss yếu (α, β ~ N(0, 3²)). Nếu nghiệm β < 0 thì khớp lại với β = 0.
- **Luật thô (zero-shot):**
  - S_j = P(Binomial(10, p(b_j)) ≥ 8) với b_j ∈ {1, 2, 5, 10, 20, 50};
  - F_j = S_j (không giảm vì β ≥ 0);
  - p_c = F_c − F_{c−1}, p_6 = 1 − F_5;
  - sàn 1e-4 rồi chuẩn hóa lại.
- **Biến thể hiệu chỉnh:** đặc trưng (tỉ lệ thành công ở 4 mức, NR̂ trung bình ở 4 mức, α, β) cộng 18 thống kê B2, đưa qua **cùng** head matched (`oeh_final_forecast.matched_forecast`) fit trên 720 MDP meta-training. Tên: `b2_plus_prefix_fqe`.

## 5. Khác LCE ở đâu

| | LCE (`curve_group`) | Prefix + FQE (spec này) |
|---|---|---|
| Dữ liệu huấn luyện | Prefix x1, x2, x5, x10 của **toàn** pilot | Prefix của 4/5 pilot (cross-fit) |
| Đánh giá policy | Trong các MDP rút từ **posterior**, J\*/J_random chính xác của MDP rút | **FQE** trên phần pilot giữ lại; J\*/J_random **ước lượng** |
| Biến thiên giữa seed | Không có (một prefix mỗi mức) | 10 seed giả lập (fold × bootstrap) |
| Vai trò | Thành phần đặc trưng của OEH (LCE) | Baseline "người làm thực tế" độc lập với mô hình posterior |

## 6. Đánh giá

- **Panel:** chỉ panel đã mở — cohort 7, 8; Colosseum 2027, 2028; SPIBB; 4 panel kiểm định cuối; FrozenLake, gambler.
- **Điểm:** endpoint đầy đủ và endpoint ngoại suy (bộ chấm T1).
- **So với:** B2, LCE thô, B2 + LCE, OEH-F, OEH.
- **Không có tham số nào được chọn** (hai biến thể cố định trong spec), nên mỗi panel là một kiểm tra ngoài mẫu của head fit trên meta-training. `development_log.csv` ghi cả hai biến thể trên mọi panel.
- **Chấp nhận:** test FQE so với đánh giá chính xác trên mô hình thực nghiệm (MDP đồ chơi); xác suất hợp lệ (tổng bằng 1, không NaN); log đầy đủ.

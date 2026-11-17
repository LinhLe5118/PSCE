> Đính chính audit: phát biểu đảo của Định lý 2 từ A1 đến tính thuần nhất từng hạng không suy ra dưới R1–R2 đơn thuần (aggregator bằng 0 là phản ví dụ). Tài liệu này và bản LaTeX hiện chỉ giữ biểu diễn của hạng thuần nhất bậc 0 và chiều đủ cho A1; kết luận tiệm cận của Định lý 1 cần giả thiết dispersion tổng hợp tiến về 0. Không thay đổi định nghĩa thực nghiệm hay artifact.

# Chứng minh và dẫn xuất các công thức

**H1 22/09/2026:** phần dưới là dẫn xuất lịch sử OEH-v1. Định nghĩa đang dùng,
dictionary v3/v4E và các ca biên được khóa tại
[H1 v1.0.0](week6/H1_FORMULA_THEORY_FREEZE_v1.0.0.md).
Với loss Gaussian, phép thế `u=z*sqrt(mu)` còn cho cận chặt hơn
`L_k(mu)<=G/sqrt(mu)*A_k(w_k,sigma_k)`. Vì ngưỡng chấp nhận `<=epsilon`,
thoát sàn `m_k>1` đòi hỏi **`A_k(w_k,sigma_k)>epsilon/G`**; tại equality vẫn
chấp nhận m=1. Các bất đẳng thức không chặt lịch sử dưới đây vẫn là điều kiện
cần yếu hơn, không phải điều kiện đủ. Không thay artifact thực nghiệm.

Tài liệu này giữ **toàn bộ quá trình tính** cho mọi đại lượng được báo cáo
trong paper và đồ án. Nguyên tắc: không công thức nào xuất hiện trong bài mà
không có một mục ở đây dẫn ra nó, kèm giả thiết và điều kiện áp dụng.

Ký hiệu theo `PLAN_Final.md` mục 3. Trạng thái: dự thảo, chờ một giảng viên
đọc và xác nhận (mục tiêu tuần 2–3 của `PHASE_B_PLAN.md`).

---

## 0. Ký hiệu và giả thiết nền

Với mỗi độ sâu `k ∈ {0, …, K_max}` và mỗi state `s`:

| Ký hiệu | Nghĩa | Miền |
|---|---|---|
| `Δ_k(s)` | khoảng cách giữa hai hành động tốt nhất của `Q̂^(k)` | `≥ 0` |
| `σ_k(s)` | độ lệch chuẩn bootstrap của chính khoảng cách đó | `≥ 0` |
| `z_k(s)` | `Δ_k(s) / σ_k(s)` | `≥ 0`, quy ước `+∞` khi `σ = 0, Δ > 0` |
| `w_k(s)` | trọng số occupancy đã chuẩn hoá trên state tới hạn | `≥ 0`, `Σ_s w_k(s) ≤ 1` |
| `w_gap(k)` | occupancy của state tới hạn bị loại khỏi tổng | `∈ [0, 1]` |
| `μ` | hệ số nhân bằng chứng | `≥ 1` |
| `ε` | ngưỡng đăng ký | `0,05` |

**Giả thiết A (chuẩn hoá trọng số).** Khi khối lượng tới hạn dương,

> `Σ_s w_k(s) + w_gap(k) = 1`

Chứng minh: theo cách dựng trong `code/oeh_estimator.py`, với `M_k` là tổng
occupancy trên các state tới hạn, `w_k(s) = d_k(s)/M_k` cho state tới hạn **và**
đủ điều kiện (có ≥2 hành động được support), còn `w_gap(k)` là tổng
`d_k(s)/M_k` trên các state tới hạn **không** đủ điều kiện. Hai tập rời nhau và
hợp lại thành tập tới hạn, nên tổng bằng `M_k/M_k = 1`. ∎

Khi khối lượng tới hạn bằng 0 thì cả hai bằng 0; đây là ca biên duy nhất và
được kiểm riêng (`weight_invariant_ok` trong `scripts/analysis/audit_saturation_sweep.py`).

**Giả thiết B (dạng hàm mất mát đăng ký).**

> `L_k(μ) = Σ_s w_k(s) · Δ_k(s) · Φ(−z_k(s)·√μ)`

**Giả thiết C.** `m_k = min{ μ ≥ 1 : L_k(μ) ≤ ε }`, và `k† = argmin_k m_k` lấy
chỉ số nhỏ nhất khi có đồng hạng trong dung sai `1e-12`.

---

## 1. Bổ đề 1 — cực đại của `z·Φ(−z)`

**Phát biểu.** Hàm `g(z) = z·Φ(−z)` trên `[0, ∞)` đạt cực đại duy nhất tại
nghiệm `z*` của `Φ(−z) = z·φ(z)`, với

> `z* = 0,751791524693564`  và  `g(z*) = 0,169971207479904`

**Chứng minh.** `g(0) = 0` và `g(z) → 0` khi `z → ∞` (vì `Φ(−z) ≤ φ(z)/z`), còn
`g > 0` trên `(0, ∞)`, nên cực đại đạt tại điểm trong. Đạo hàm:

```
g'(z) = Φ(−z) − z·φ(z)
```

`g'(0) = Φ(0) = 1/2 > 0`. Đạo hàm bậc hai:

```
g''(z) = −φ(z) − φ(z) + z²·φ(z) = φ(z)·(z² − 2)
```

nên `g'` giảm nghiêm ngặt trên `[0, √2)` rồi tăng. Vì `g'(0) > 0` và
`g'(√2) = Φ(−√2) − √2·φ(√2) ≈ 0,0786 − 0,2160 < 0`, `g'` có đúng một nghiệm
trên `(0, √2)`, và trên `[√2, ∞)` thì `g'` tăng nhưng vẫn âm (`g' → 0⁻`), nên
không có nghiệm thứ hai. Vậy `z*` duy nhất và là cực đại toàn cục. ∎

Giá trị số lấy bằng `scipy.optimize.brentq` trên `g'`, `xtol = rtol = 1e-15`.
Kiểm tra hai bên: `g(0,7) = 0,169395`, `g(0,8) = 0,169528`, đều nhỏ hơn `g(z*)`.

**Hệ quả 1.1.** `Φ(−z*) = 0,226088219801607` và `1/g(z*) = 5,883349390915`.

---

## 2. Định lý 1 — chặn cấu trúc trên `L_k`

**Phát biểu.** Với mọi `μ ≥ 1` và mọi `k`,

> `L_k(μ) ≤ g(z*) · Σ_s w_k(s)·σ_k(s) = 0,169971 · Σ_s w_k(s)·σ_k(s)`

Đặc biệt, **`Δ` không xuất hiện ở vế phải**.

**Chứng minh.** Xét từng số hạng `T_s(μ) = w_k(s)·Δ_k(s)·Φ(−z_k(s)√μ)`.

*Trường hợp 1: `σ_k(s) = 0`.* Nếu `Δ_k(s) > 0` thì `z = +∞` theo quy ước, nên
`Φ(−∞) = 0` và `T_s = 0`. Nếu `Δ_k(s) = 0` thì `T_s = 0` trực tiếp. Cả hai đều
thoả `T_s ≤ g(z*)·w_k(s)·σ_k(s) = 0`.

*Trường hợp 2: `σ_k(s) > 0`.* Khi đó `Δ_k(s) = z_k(s)·σ_k(s)`, nên

```
T_s(μ) = w_k(s)·σ_k(s)·[ z_k(s)·Φ(−z_k(s)√μ) ]
```

Vì `Φ` là hàm tăng và `−z√μ ≤ −z` với `z_k(s) ≥ 0`, ta có
`Φ(−z√μ) ≤ Φ(−z)`, do đó

```
z·Φ(−z√μ) ≤ z·Φ(−z) = g(z) ≤ g(z*)
```

theo Bổ đề 1. Vậy `T_s(μ) ≤ g(z*)·w_k(s)·σ_k(s)`.

Cộng theo `s` cho kết quả. ∎

**Nhận xét — đây mới là nội dung.** Phép thế `Δ = z·σ` làm khoảng cách action
triệt tiêu: trần chỉ còn phụ thuộc `σ`. Mọi nỗ lực nới trần bằng cách tìm MDP
có khoảng cách action lớn hơn đều vô ích, vì `Δ` lớn kéo `z` lớn theo, và
`z·Φ(−z)` bị chặn.

**Hệ quả 2.1 (điều kiện cần để thoát).** `L_k(1) ≥ ε` đòi hỏi

> `Σ_s w_k(s)·σ_k(s) ≥ ε / g(z*) = 0,294167469545756`

**Hệ quả 2.2 (suy biến theo cỡ mẫu).** Nếu tập support và tập độ sâu hữu hạn
được giữ cố định, và `σ_k(s) = O(n^{−1/2})` đều theo `s`, thì
`L_k(1) = O(n^{−1/2})`. Đây là giả thiết first-order thường dùng cho bootstrap,
không phải định lý được bootstrap bảo đảm trong pipeline có support và depth phụ
thuộc dữ liệu. Do đó
tồn tại `n₀` sao cho với mọi `n > n₀`: `L_k(1) < ε` với mọi `k`, kéo theo
`m_k = 1` với mọi `k`, và `k† = 0` theo quy tắc đồng hạng. Thống kê **hội tụ về
hằng số khi dữ liệu tăng**.

Đây là giới hạn phân biệt của statistic dưới giả thiết dispersion đã nêu:
sàn m=1 chỉ biểu thị pilot hiện tại đủ theo loss này, không chứng minh learner
đã học xong hoặc sample complexity không thay đổi giữa các bài toán. Support
thiếu cũng có thể làm tổng nhỏ do state bị loại; cần đọc cùng support-gap flag.

**Kiểm chứng thực nghiệm.** 62/62 pilot có trần, 0 vi phạm
(`artifacts/week2/alternative_definitions_2026-08-25/`). `Σ w σ` trung vị đo
được `0,0123`, so với ngưỡng cần `0,2942` — thiếu 23,9 lần.

---

## 3. Ba tiên đề cho một thống kê yêu-cầu-bằng-chứng

**A1 — bất biến thang đo.** Với mọi `c > 0`, thay reward `r → c·r` không được
làm đổi `m_k`.

**A2 — vô cảm với hoà.** State có `Δ_k(s) = 0` phải đóng góp 0 vào `L_k`.

**A3 — đơn điệu theo bằng chứng.** `L_k(μ) → 0` khi `μ → ∞`.

**Vì sao ba tiên đề này.** A1: `ε` là hằng số tuyệt đối, nên nếu `L` mang thứ
nguyên giá trị thì cùng một bài toán đo bằng đơn vị khác cho kết luận khác. A2:
hai hành động ngang nhau thì chọn hành động nào cũng không mất gì, nên gọi đó
là "lỗi" là sai ngữ nghĩa. A3: nếu không, `m_k` có thể bằng `+∞`, tức "không
lượng dữ liệu nào đủ", điều vô nghĩa với một state có khoảng cách hữu hạn.

### 3.1 Tác động của phép đổi thang đo

Dưới `r → c·r` với `c > 0`: `Q̂ → c·Q̂`, do đó `Δ → c·Δ` và `σ → c·σ` (bootstrap
tuyến tính theo reward), nên **`z = Δ/σ` bất biến**. Trọng số `w` bất biến vì
occupancy không phụ thuộc thang reward.

### 3.2 Kiểm từng định nghĩa

| Định nghĩa | `L` dưới `r → c·r` | A1 | A2 | A3 |
|---|---|---|---|---|
| `registered` `Σ w Δ Φ(−z√μ)` | `→ c·L` | ✗ | ✓ | ✓ |
| `probability` `Σ w Φ(−z√μ)` | bất biến | ✓ | ✗ | ✗ |
| `materiality` `Σ w 1[Δ ≥ δ] Φ(−z√μ)` | bất biến | ✓ | ✓ | ✓ |
| `max_state` `max_{s: Δ ≥ δ} Φ(−z√μ)` | bất biến | ✓ | ✓ | ✓ |

**A1 cho `registered`:** `L → c·L` nhưng `ε` giữ nguyên, nên
`m_k = min{μ : c·L(μ) ≤ ε}` đổi theo `c`. Vi phạm. ∎

**A1 cho `materiality`/`max_state`:** ngưỡng đáng kể được định nghĩa **tương
đối** — `δ = f · V̂` với `V̂` là thang giá trị dữ liệu-thuần (mục 5). Dưới đổi
thang, `Δ → cΔ` và `V̂ → cV̂`, nên chỉ báo `1[Δ ≥ f·V̂]` bất biến. Cùng với `z`
bất biến, `L` bất biến. ∎

Đây chính là lý do `δ` **phải** tương đối. Nếu đặt `δ` tuyệt đối thì A1 hỏng
đúng như `registered`.

**A2 cho `probability`:** state có `Δ = 0` cho `z = 0`, nên `Φ(0) = 1/2` và số
hạng bằng `w/2 > 0`. Vi phạm. ∎

**A3 cho `probability`:** cùng state đó cho `Φ(−0·√μ) = 1/2` với **mọi** `μ`,
nên `L_k(μ) ≥ w/2` không dần về 0. Nếu `w/2 > ε` thì `m_k = +∞`. Vi phạm. ∎

Đây không phải suy đoán: `m = ∞` **đã quan sát được** trên pilot
`development_forward_layered_000 / perturbed_reward`.

**A3 cho `materiality`/`max_state`:** mọi state còn trong tổng có `Δ ≥ δ > 0`.
Nếu `σ > 0` thì `z > 0` và `Φ(−z√μ) → 0`. Nếu `σ = 0` thì `z = +∞` và số hạng
bằng 0 sẵn. Vậy `L_k(μ) → 0`. ∎

**Kết luận.** `registered` hỏng A1; `probability` hỏng A2 và A3;
`materiality` và `max_state` thoả cả ba. Định lý 1 giải thích **vì sao** A1
hỏng lại gây suy biến: `L` mang thứ nguyên giá trị nên thừa hưởng thang `σ`,
mà `σ → 0` theo cỡ mẫu.

### 3.3 Bổ đề 2 — thang giá trị đã chọn thuần nhất bậc 1

**Phát biểu.** Dưới `r → c·r` với `c > 0`: `Δ_k(s) → c·Δ_k(s)`,
`σ_k(s) → c·σ_k(s)`, `V̂ → c·V̂`, và `w_k(s)` bất biến.

**Chứng minh.** Code khởi tạo bằng giá trị chính sách hành vi trên mô hình hỗ
trợ: `v_β = (I − γP_β)^{-1}r_β` và
`Q̂^(0)(s,a) = r̂(s,a) + γP̂(s,a)^T v_β`, tức
`Q̂^(0)=Q^{β̂^K}`, không phải chỉ `r̂`. Khi `r → c·r`, `P̂` và `β̂^K` cố định,
nên `r_β → c·r_β`, `v_β → c·v_β`, và `Q̂^(0) → c·Q̂^(0)`. Backup Bellman
`Q̂^(k+1) = r̂ + γ P̂ max_a Q̂^(k)` cũng thuần nhất bậc 1; quy nạp cho
`Q̂^(k) → c·Q̂^(k)` với mọi `k`. Do đó `Δ`, là hiệu hai giá trị `Q̂^(k)`, nhân
`c`. Bootstrap Bayes gán trọng số Dirichlet cho **trajectory**,
không cho reward, nên mỗi replicate cũng nhân `c`, kéo theo `σ → c·σ`. `V̂` là
biên độ của `Q̂^(k)` trên support (mục 5) nên cũng nhân `c`. `w_k` dẫn xuất từ
occupancy `d_k`, tính từ `P̂` và `ρ0`, không đụng tới reward — bất biến. ∎

**Hệ quả.** Ba đại lượng `(Δ, σ, V̂)` cùng nhân `c`, nên **mọi tỉ số giữa
chúng đều bất biến**. Hai tỉ số độc lập:

```
z = Δ/σ        (bằng chứng: khoảng cách đo bằng đơn vị sai số)
ρ = Δ/V̂       (đáng kể: khoảng cách đo bằng đơn vị thang giá trị)
```

### 3.4 Định lý 2 — biểu diễn hạng thuần nhất bậc 0 và chiều đủ cho A1

Phần này cho một cách dựng thống kê bất biến thang đo. Nó **không** đặc trưng
mọi thống kê có requirement `m_k` bất biến.

**Lớp hàm xét.** Gọi một *thống kê yêu-cầu-bằng-chứng* là hàm dạng

```
L_k(μ) = A_k( { w_k(s) }, { ψ(Δ_k(s), σ_k(s), V̂; μ) } )
```

với hai điều kiện chính quy, cả hai đều thoả bởi mọi định nghĩa đã đăng ký:

- **(R1) Bộ gộp thuần nhất bậc 1 và đơn điệu.** `A_k(w, λ·a) = λ·A_k(w, a)` với
  mọi `λ > 0`, và `A_k` không giảm theo từng thành phần của `a`. Cả
  `Σ_s w_s a_s` lẫn `max_{s ∈ S'} a_s` đều thoả.
- **(R2) Số hạng không âm.** `ψ ≥ 0`.

Xét trên miền `σ > 0`, `V̂ > 0` (các ca biên `σ = 0` xử lý riêng ở mục 6).

**Phát biểu.** Trên miền `Δ, σ, V̂ > 0`, hạng `ψ` thuần nhất bậc 0 khi và chỉ
khi tồn tại `h` sao cho

> `ψ(Δ, σ, V̂; μ) = h(z, ρ; μ)`, với `z = Δ/σ`, `ρ = Δ/V̂`.

Nếu các occupancy weight không đổi khi đổi thang reward, hạng dạng này là
**điều kiện đủ** để aggregate `L_k(μ)` và requirement `m_k` thoả A1.

**Chứng minh.**

Nếu `ψ = h(z, ρ; μ)` thì theo Bổ đề 2 cả `z` và `ρ` bất biến dưới
`r → c·r`, còn `w` bất biến, nên `L_k(μ)` bất biến với mọi `μ`. Tập
`{μ : L_k(μ) ≤ ε}` không đổi, do đó `m_k` không đổi. A1 thoả.

Ngược lại, giả sử trực tiếp rằng `ψ` thuần nhất bậc 0. Chọn `c = 1/σ`:

```
ψ(Δ, σ, V̂; μ) = ψ(Δ/σ, 1, V̂/σ; μ)
```

Vế phải là hàm của đúng hai đối số `(Δ/σ, V̂/σ)`. Vì `V̂/σ = z/ρ` trên miền
dương, cặp `(z, ρ)` sinh ra cùng thông tin, nên tồn tại `h` với
`ψ = h(z, ρ; μ)`. ∎

**Vì sao không có chiều đảo A1 ⇒ hạng bậc 0.** R1 cho phép aggregator bằng 0,
và requirement chỉ giữ lại thời điểm đường cong cắt một ngưỡng nên không nhận
dạng được mọi giá trị của một loss không đơn điệu. Do đó A1 ở mức `m_k` không
suy ra tính thuần nhất từng hạng dưới R1–R2 đơn thuần.

**Hệ quả 2.1 — vì sao định nghĩa đăng ký hỏng, viết bằng một dòng.** Số hạng
của `registered` là

```
ψ = Δ·Φ(−z√μ) = σ · [ z·Φ(−z√μ) ]
```

Ngoặc vuông thuần nhất bậc 0; thừa số `σ` đứng ngoài thì không. Vậy `ψ` thuần
nhất **bậc 1**. Trên instance weighted-sum một state không tầm thường, đổi
thang reward có thể đẩy `L(1)` qua ngưỡng cố định và làm `m` đổi, nên định nghĩa
đăng ký vi phạm A1. Định lý 1 nói tiếp điều gì xảy ra nếu aggregated dispersion
tiến về 0: `L → 0` và `m ≡ 1`. Đây là kết luận có điều kiện, không phải phát biểu
"bất kể bài toán khó đến đâu".

Đây là toàn bộ cơ chế suy biến, phát biểu bằng phân tích thứ nguyên: *một
thống kê đo bằng đơn vị giá trị, so với một ngưỡng không có đơn vị.*

**Hệ quả 2.2 — điều kiện cần đo được, cho toàn bộ lớp bậc 1.** Cho bất kỳ
thống kê nào dạng `ψ = σ·g(z; μ)` với `g` bị chặn, `G = sup_{z ≥ 0} g(z; 1)`.
Theo (R1) và mục 4,

```
L_k(1) ≤ G · Σ_s w_k(s)·σ_k(s) = G · σ̄_k
```

nên **thoát bão hoà đòi hỏi**

```
σ̄ ≥ ε / G
```

Với dạng đăng ký `g(z) = z·Φ(−z)`, Bổ đề 1 cho `G = 0,169971`, tức
`σ̄ ≥ 0,29417`. Vế trái tính được **chỉ từ pilot**, trước khi chạy estimator.
Đây là cơ sở của phép kiểm khả thi ở `scripts/analysis/run_feasibility_check.py`.

**Đính chính giả thiết bộ gộp — Hưng, 2026-09-06.** Phát biểu trên đúng cho bộ
gộp **tổng có trọng số**. (R1) cho phép cả `max`, và với `max` thì
`max_s σ_s·g(z_s) ≤ G·max_s σ_s`, **không** phải `G·Σ_s w_s σ_s`; vì `Σw ≤ 1`
nên `Σ_s w_s σ_s ≤ max_s σ_s`, và bất đẳng thức thường chặt (`w = (½,½)`,
`σ = (1,0)` cho ½ so với 1). Dạng đúng cho cả lớp, suy trực tiếp từ tính đơn
điệu và thuần nhất bậc 1 của `A_k`:

```
L_k(1) ≤ G · A_k(w_k, σ_k)
```

`A_k = Σ w` cho lại `σ̄_k`; `A_k = max` cho `max_s σ_s`. Dạng này **tổng quát
hơn và gọn hơn** dạng cũ, không phải một ngoại lệ dán thêm.

Không kết quả nào hiện có đổi: `scripts/analysis/run_feasibility_check.py` chỉ chấm `registered`,
vốn gộp bằng tổng; còn `max_state` gộp bằng `max` **nhưng thuần nhất bậc 0**,
nên đã nằm ngoài phạm vi Hệ quả 2.2 vì lý do độc lập (Hệ quả 2.3), và
`tests/test_feasibility.py::test_a_degree_zero_definition_is_not_blocked_by_the_degree_one_bound`
đã chặn việc chuyển verdict sang nó. Đính chính này là để bản LaTeX
(`paper/sections/03b_axioms_and_impossibility.tex`, `\Cref{rem:aggregator}`)
không phát biểu cho "toàn bộ lớp bậc 1" một cận chỉ đúng cho một bộ gộp —
reviewer sẽ bắt được.

**Hệ quả 2.3 — lớp hợp lệ không rỗng và có ít nhất hai phần tử dựng được.**
Đặt `f > 0` cố định và `δ = f·V̂` (tức ngưỡng đáng kể theo `ρ`):

```
materiality:  h(z, ρ; μ) = 1[ρ ≥ f] · Φ(−z√μ)          gộp bằng Σ w
max_state:    h(z, ρ; μ) = Φ(−z√μ) trên {ρ ≥ f}         gộp bằng max
```

Cả hai chỉ phụ thuộc `(z, ρ)` nên thuần nhất bậc 0 và đủ cho A1 theo Định lý 2.
`ρ = 0`
cho chỉ báo bằng 0 nên đóng góp 0 (A2 ✓). `ρ ≥ f > 0` với `σ > 0` cho `z > 0`
và `Φ(−z√μ) → 0` (A3 ✓). ∎

**Ghi chú về tính duy nhất.** Định lý 2 cho một lớp dựng đủ cho **A1**, không
đặc trưng toàn bộ lớp requirement-level. Ngay trong lớp dựng này, thêm A2 và A3
vẫn **không** cho một phần tử duy nhất: mọi
`h(z, ρ; μ) = q(ρ)·Φ(−z√μ)` với `q(0) = 0`, `q ≥ 0` đều thoả cả ba. Chọn `q`
nào là quyết định thiết kế, không phải hệ quả của tiên đề, và phải đăng ký
trước theo `EXPERIMENT_RULE.md` quy tắc 8. Xem mục 9.2.

---

## 4. Ba trần trên `L_k`, từ lỏng đến chặt

Cả ba dùng chung một bất đẳng thức: `Σ_s w_s a_s ≤ max_s a_s` khi `w ≥ 0`,
`Σ w ≤ 1`, `a ≥ 0`.

**4.1 `gap_bound = ½ · max_s Δ_k(s)`.** Vì `Φ(−z√μ) ≤ Φ(0) = ½` với mọi
`z ≥ 0, μ ≥ 1`:

```
L_k(μ) = Σ w Δ Φ(−z√μ) ≤ ½ Σ w Δ ≤ ½ max_s Δ_k(s)
```

Dưới `ε` thì **không lượng ngẫu nhiên nào** thoát được trên instance đó.

**4.2 `reweighting_bound = max_{k,s} Δ_k(s)·Φ(−z_k(s))`.** Giữ `σ` đo được,
chỉ tối ưu trọng số:

```
L_k(1) = Σ_s w_s [Δ_s Φ(−z_s)] ≤ max_s [Δ_s Φ(−z_s)]
```

Dưới `ε` thì cơ chế loại trừ `w_gap` **không** phải nguyên nhân chặn: dồn toàn
bộ occupancy vào state nhiều thông tin nhất vẫn không tới.

**4.3 `exact_ceiling = max_k ½ · Σ_s w_k(s)·Δ_k(s)`.** Khi estimator báo cáo
`w_k(s)`, không cần chặn lỏng nữa. Cho `σ → ∞` với `Δ` cố định thì `z → 0` và
`Φ(−0) = ½`, nên

```
sup_σ L_k(1) = ½ Σ_s w_k(s)·Δ_k(s)
```

Đây là giá trị `L_k` hội tụ tới nếu **mọi** state đủ điều kiện trở nên bất định
tối đa. Dưới `ε` là **chứng minh** bất khả, không còn là chặn trên.

Thứ tự: `exact_ceiling ≤ gap_bound` (vì `Σ w ≤ 1`), và cả hai đều chặn trên
`L_k(1)` thực đo. Kiểm chứng: `bound_consistent_with_measured_loss` trong
`scripts/analysis/audit_saturation_sweep.py`, đúng trên 62/62 pilot.

---

## 5. Thang giá trị dữ liệu-thuần `V̂`

`δ = f · V̂` cần một `V̂` **không** dùng oracle (`PLAN_Final.md` §11 quy tắc 5
cấm `V*` vào estimator).

**Định nghĩa dùng.** `V̂ = max_k [ max_{(s,a) ∈ support} Q̂^(k)(s,a) − min_{(s,a) ∈ support} Q̂^(k)(s,a) ]`

**Vì sao không dùng `r_max/(1−γ)`.** Với `γ = 0,95` nó cho `20`, trong khi
`V̂` đo được nằm trong khoảng `0,137–1,0` — lệch 20 đến 150 lần. Đặt
`δ = 0,01 × 20 = 0,2` đẩy ngưỡng lên trên **mọi** khoảng cách của instance và
làm rỗng tổng một cách âm thầm. Đây là lỗi đã thực sự xảy ra trong lần chạy
thử đầu tiên và được phát hiện nhờ `material_states_at_k_dagger = 0`.

**Vì sao không dùng `dynamic_range` của benchmark.** Nó dẫn xuất từ `V*`.

---

## 6. Hệ số nhân `σ` cần thiết

**Định nghĩa.** `c* = min{ c ≥ 1 : max_k Σ_s w_k(s)·Δ_k(s)·Φ(−z_k(s)/c) ≥ ε }`

**Quan hệ với `μ`.** Thay `σ → c·σ` cho `z → z/c`, tương đương `√μ = 1/c`, tức
`μ = 1/c²`. Vậy `c > 1` chính là phần `μ < 1` mà định nghĩa đăng ký không cho
phép hỏi (vì `m_k` chỉ tìm trên `μ ≥ 1`). `c*` trả lời: *dữ liệu phải bất định
gấp bao nhiêu lần thì mới chạm ngưỡng*.

**Tính chất quan trọng.** Phép nhân **không** cứu được state có `σ = 0` đúng
bằng 0: `z = +∞` thì `z/c = +∞` với mọi `c` hữu hạn. Vì vậy `c*` phải đọc kèm
`exact_ceiling`, không thay thế nó. Được pin bằng test
`test_zero_sigma_cannot_be_rescued_by_inflation`.

**Kết quả đo (đính chính 17/09).** Trong 19 pilot `blocked_by_theorem_1`,
16 có hệ số inflation hữu hạn được ghi trong CSV: nhỏ nhất `2,32`, lớn nhất
`207,90`; ba pilot không có estimate hữu hạn. Không dùng 19 làm mẫu số của
thống kê trên 16 nghiệm hữu hạn.

---

## 7. Tiêu chí thoát bão hoà

Chốt trước lần chạy đầu (`EXPERIMENT_RULE.md` mục 2). Một instance thoát khi
thoả **cả ba**:

```
(1)  max_k L_k(1) ≥ ε
(2)  max_k m_k    > 1 + 1e−9
(3)  0 < k† < K_max
```

(1) là điều kiện cần. (2) loại trường hợp `m` chỉ lệch 1 do nhiễu số học. (3)
đòi cực tiểu nằm trong: `k†` ở biên nghĩa là trục độ sâu không mang thông tin,
dù `m` có lớn.

Ba điều kiện được báo cáo **riêng** để thấy được thoát một phần.

---

## 8. Các đại lượng dẫn xuất

**8.1 `n_req_hat = |D₀| · m_{k†}`.** Dùng độ dài thực tế của pilot, không dùng
`N₀` danh nghĩa. Khi `m_{k†} = 1` thì `n_req_hat = |D₀|` — không mang thông tin
ngoài kích thước pilot, đúng như `WEEK2_DECISION_GATE.md` §10b đo được.

**8.2 Effective horizon.** `H = min_k [ k + log(max(1, m_k)) / log(A) ]`. Khi
`m_k ≡ 1` thì `H = 0`, lại là hệ quả của Định lý 1.

**8.3 Censoring.** `censored_right = (w_gap(k†) > ε)`. Lưu ý đại lượng này
so sánh một **tỉ lệ occupancy** với `ε` vốn là ngưỡng cho một **regret có thứ
nguyên**. Đây là một chỗ dùng chung hằng số cho hai đại lượng khác thứ nguyên,
cần ghi nhận là hạn chế thiết kế trong phần thảo luận.

---

## 9. Những gì tài liệu này **chưa** chứng minh

Ghi ra để không ai nhầm mức độ chắc chắn:

1. **Tính đủ của A1–A3.** A1–A3 là ba yêu cầu thiết kế có động cơ rõ ràng;
   tài liệu này không chứng minh chúng là điều kiện cần cho mọi metric hữu ích.
   Thoả cả ba **không đủ** để thống kê mang thông tin, và điều này nay đã **đo được chứ
   không còn là câu hỏi mở**: `materiality` và `max_state` thoả cả ba tiên đề
   nhưng không vượt baseline cấu trúc `S·A` ở bất kỳ phạm vi nào trong bảy phạm
   vi họ hành vi (`artifacts/week2/c2_correlation_2026-09-02/`). Chúng là
   desiderata, không phải định lý cần-và-đủ cho tính hữu ích.

2. **Tính duy nhất của họ hợp lệ.** Định lý 2 (mục 3.4) biểu diễn các hạng
   `ψ` thuần nhất bậc 0 dưới dạng `ψ = h(z, ρ; μ)` và chứng minh lớp này đủ cho
   A1. Nó không đặc trưng toàn bộ các requirement-level statistic thoả A1.
   Thêm A2 và A3 vẫn để lại một họ vô hạn
   (`h = q(ρ)·Φ(−z√μ)` với `q(0) = 0`), nên **không có phần tử duy nhất** và
   không nên tuyên bố có. Chọn `q` là quyết định thiết kế phải đăng ký trước.
   Điều còn **chưa** chứng minh: liệu điều kiện chính quy (R1) — bộ gộp thuần
   nhất bậc 1 và đơn điệu — có thể nới lỏng tới đâu.

3. **`σ = O(n^{−1/2})`.** Hệ quả 2.2 giả thiết điều này.
   **Đã kiểm bằng số, 2026-09-05** (`artifacts/week2/scale_invariance_2026-09-05/`,
   phép kiểm K2 đăng ký ở §8.1, chạy lần đầu): quét `N₀ ∈ {2,5,10}·|S||A|` trên
   cả 75 pilot, **số mũ tái lập được** — trung vị độ dốc log-log của `σ̄` là
   **−0,472** trên 27 pilot, và **−0,626** trên 305 state ghép cặp ở độ sâu 0,
   so với −0,5 tham chiếu. Các median này tương thích với heuristic first-order,
   nhưng không chứng minh scaling đều trên support/depth phụ thuộc dữ liệu.

   **Nhưng dung sai 2× từng instance của K2 không đạt** (27,3% trên 33 pilot
   kiểm được; 47,9% trên 305 state), nên **tiêu chí thành công số 2 KHÔNG đạt**
   và không được trích như thể đạt. Ba cỡ, dải `N` 5 lần, `B = 100` — đòn bẩy
   quá ngắn để ghim phương sai quanh quy luật, dù trung vị rơi đúng chỗ. Không
   hằng số nào bị nới để chiều kết quả này.

4. ~~**Positive control.**~~ **Đã đóng, 2026-09-01.** Estimator đăng ký, không
   sửa gì, cho `max_k m_k = 3,391` và `k† = 2` nằm trong trên instance dựng theo
   thiết kế với `σ̄ = 0,568` (`artifacts/week2/positive_control_2026-09-01/`).
   Cùng đường code cho `m_k = 1,0` ở 12.019 điểm development. Đối chứng này loại
   trừ một sàn phổ quát do hard-code/cài đặt; nó không xác nhận mọi giả thiết mô
   hình hay mọi đường thực thi.
   **Cập nhật 2026-09-03: đã đóng hoàn toàn.** Workstream 1 bàn giao hai MDP
   dựng độc lập (`benchmark_pc_short_chain`, `benchmark_pc_long_chain`), và cả
   hai PASS trên estimator đăng ký không sửa: `m = 1,160` / `k† = 1` /
   `σ̄ = 1,539` và `m = 1,114` / `k† = 1` / `σ̄ = 0,796`
   (`artifacts/week2/benchmark_positive_control_validation_2026-09-03/`).
   Đối chứng dương nay gồm ba instance, hai do người không sở hữu estimator
   dựng.

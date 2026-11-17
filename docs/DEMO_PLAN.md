> Hướng làm việc hiện hành: [FINAL_RESEARCH_DIRECTION_2026-09-17.md](FINAL_RESEARCH_DIRECTION_2026-09-17.md). Chẩn đoán dữ liệu stochastic và lựa chọn learner là mục tiêu tiếp theo; OEH saturation là kết quả trung gian. Calibration đã sửa tại `artifacts/week5/statistical_correction_2026-09-17/`; không dùng kết luận thắng IQL từ bảng cũ.

# Kế hoạch demo bảo vệ đồ án

Trạng thái: dự thảo. Triển khai ở tuần 10–11 của `PHASE_B_PLAN.md`, chạy thử ở
tuần 15–16.

**Ràng buộc bắt buộc** (`PLAN_Final.md` §Giai đoạn D): demo **chỉ đọc từ kho
kết quả đã có**, không huấn luyện tại chỗ, không gọi mạng.

---

## 1. Demo phải trả lời được cái gì

Hội đồng không quan tâm code chạy được. Họ quan tâm ba câu:

| Câu hỏi của hội đồng | Module trả lời |
|---|---|
| "Hệ thống này *làm* gì?" | D1 — đường đi từ pilot tới kết luận |
| "Sao biết nó không hỏng mà thật sự là metric không chạy?" | D2 — positive control |
| "Phát hiện của các em là gì, và nó *nhìn* ra sao?" | D3 — định lý chặn; D4 — bốn định nghĩa |

Thứ tự trên cũng là thứ tự trình bày. Câu 2 phải đứng **trước** câu 3, vì
không có nó thì mọi kết luận âm đều bị nghi ngờ.

---

## 2. Bốn module

### D1 — Từ pilot tới kết luận (2 phút)

Chọn một `(MDP, policy family)` từ 75 pilot đã chạy, đi hết đường ống bằng dữ
liệu đã lưu:

```
buffer .npz  →  độ phủ, support  →  Δ_k(s), σ_k(s), z_k(s)
             →  đường cong L_k(1)  →  m_k  →  k†, n_req_hat
```

Mỗi bước hiện số thật của instance đó. Điểm nhấn: `m_k = 1` ở **mọi** độ sâu,
`k† = 0`.

**Nguồn:** `artifacts/week2/saturation_sweep_2026-08-24/sweep_results.csv`,
`artifacts/week2/oeh_validation_2026-08-23/evidence_curves.csv`.

**Tiêu chí đạt:** chọn được ≥3 instance khác nhau, mỗi lần render < 1 giây.

### D2 — Positive control (2 phút) — **module quan trọng nhất**

Cùng một estimator **đăng ký**, chạy trên MDP dựng thủ công có `σ̄ ≥ 0,294`.
Kết quả phải là `m > 1` và `k†` nằm trong.

Đặt hai màn cạnh nhau:

| MDP dựng thủ công | Pilot thật của benchmark |
|---|---|
| `m_k > 1`, `k†` nằm trong | `m_k ≡ 1`, `k† = 0` |

Thông điệp một câu: *"Cùng một dòng code. Nó phát hiện được tín hiệu khi tín
hiệu tồn tại. Trên các pilot benchmark đã xét, registered statistic bão hòa; điều đó không chứng minh learner không học được."*

**Nguồn:** `artifacts/week2/positive_control_2026-09-01/` (fixture tham chiếu)
và `artifacts/week2/benchmark_positive_control_validation_2026-09-03/` (hai MDP
của Workstream 1). **Đã có từ 03/09.** Chạy lại trước mặt hội đồng bằng
`python -m scripts.validation.run_positive_control --candidate-dir artifacts/week2/benchmark_positive_control_handover_2026-09-03 --output-dir <thư mục rỗng>`.

**Tiêu chí đạt:** ≥2 MDP dựng thủ công, có unit test, chạy được từ dòng lệnh
trước mặt hội đồng nếu bị hỏi.

### D3 — Định lý chặn, nhìn thấy được (2 phút)

Hai đồ thị cạnh nhau:

1. **`g(z) = z·Φ(−z)`** với cực đại đánh dấu tại `z* = 0,7518`, `g = 0,16997`.
   Đây là toàn bộ nội dung Định lý 1 trong một hình.
2. **Thanh trượt `σ̄`** từ `0` tới `0,5`. Đường trần `0,17·σ̄` chạy theo, đường
   `ε = 0,05` nằm ngang. Chúng cắt nhau tại `σ̄ = 0,2942`. Chấm đỏ đánh dấu
   `σ̄` trung vị đo được `0,0123` — nằm xa bên trái điểm cắt.

Người xem tự kéo thanh trượt và tự thấy cần bất định gấp 24 lần. Không cần
giải thích thêm.

**Nguồn:** `docs/DERIVATIONS.md` mục 1–2, `alternative_definitions.csv`
(cột `sigma_weighted_mean`, `structural_bound`).

**Tiêu chí đạt:** hai hình render offline, thanh trượt phản hồi tức thì.

### D4 — Bốn định nghĩa, cùng một dữ liệu (2 phút)

Cùng một instance, chuyển giữa `registered / probability / materiality /
max_state`, xem `k†` nhảy từ `0` sang giá trị nằm trong. Kèm bảng tiên đề
A1–A3 và cột dao động theo `δ` (3,20× so với 1,14×).

Phải nói rõ tại chỗ: **bốn định nghĩa thay thế viết sau khi thấy bản gốc bão
hoà, nên là giả thuyết chưa kiểm định.** Nói thẳng điều này trước hội đồng
cộng điểm liêm chính; giấu nó thì mất nhiều hơn được nếu bị hỏi.

**Nguồn:** `artifacts/week2/alternative_definitions_2026-08-25/`.

### D5 — Chỉ nếu C2 dương

Scatter `n_req_hat` so với `N_emp`, ba đường: baseline cấu trúc `S·A`,
`registered`, `max_state`. Nếu C2 âm thì bỏ module này, không cố cứu.

---

## 3. Lựa chọn kỹ thuật

**Đề xuất: trang HTML tĩnh, tự chứa, sinh sẵn từ artifact.**

Lý do: phòng bảo vệ có thể không có mạng, không cài được gì, máy không phải máy
của nhóm. Một file `.html` mở bằng trình duyệt bất kỳ là phương án ít điểm hỏng
nhất. Dữ liệu nhúng thẳng dạng JSON, không fetch.

```
demo/
  build_demo.py        # đọc artifacts/, sinh demo/dist/index.html
  templates/           # HTML + CSS + JS thuần, không CDN
  dist/index.html      # sản phẩm, commit vào repo
```

**Về dependency.** `requirements.txt` cố ý chỉ khoá `numpy`/`scipy`
(xem docstring `evaluation/io.py`). Hai phương án:

- **(a) Không thêm dependency** — vẽ SVG trực tiếp bằng Python thuần. Nhiều
  việc hơn khoảng nửa ngày, nhưng không đụng gì tới môi trường đã khoá.
- **(b) `requirements-demo.txt` riêng** với `matplotlib`, chỉ dùng lúc build,
  không ảnh hưởng đường ống khoa học.

Khuyến nghị **(b)**: tách file riêng thì môi trường chạy thí nghiệm vẫn nguyên
vẹn, mà không tốn nửa ngày vẽ SVG tay. Cần nhóm chốt.

**Phương án dự phòng bắt buộc:** một CLI `python demo/cli.py --module d1` in ra
bảng số, không cần trình duyệt. Nếu HTML hỏng trước hội đồng thì vẫn demo
được. Không coi là tuỳ chọn.

---

## 4. Kịch bản 8 phút

| Phút | Nội dung | Module |
|---|---|---|
| 0–1 | Bài toán: cần bao nhiêu dữ liệu offline thì đủ? Vì sao khó | — |
| 1–3 | Đường ống chạy trên một instance thật | D1 |
| 3–5 | Positive control: nó phát hiện được tín hiệu khi tín hiệu có | D2 |
| 5–7 | Vì sao trên benchmark thì không: định lý chặn, kéo thanh trượt | D3 |
| 7–8 | Lớp định nghĩa sửa được, kèm cảnh báo giả thuyết | D4 (+D5) |

Dự phòng câu hỏi: mở thẳng `ARM_REGISTRY.md` để trả lời *"các em đã thử bao
nhiêu cấu hình"* — 16, có ngày và seed từng dòng.

---

## 5. Lịch và tiêu chí nghiệm thu

| Mốc | Việc | Tiêu chí |
|---|---|---|
| Tuần 1 | Positive control (đầu vào cho D2) | ≥2 MDP, có unit test |
| Tuần 10 | D1, D3 | Render offline, < 1s mỗi lần |
| Tuần 11 | D2, D4, CLI dự phòng | Chạy trên máy sạch, không mạng |
| Tuần 11 | D5 nếu C2 dương | — |
| Tuần 15–16 | Chạy thử trong bảo vệ thử | ≥1 người ngoài nhóm bấm được mà không cần hướng dẫn |

**Kiểm tra trên máy sạch là bắt buộc**, không phải tuỳ chọn: máy của người
viết code luôn chạy được.

---

## 6. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Không có D2 kịp | Ưu tiên tuần 1; nếu trượt, cắt D4 chứ không cắt D2 |
| Phòng bảo vệ không có mạng/không cài được | HTML tĩnh tự chứa + CLI dự phòng |
| Hội đồng hỏi số không có trong demo | Mở artifact CSV trực tiếp; registry trả lời được câu "bao nhiêu cấu hình" |
| Demo trông như "chỉ vẽ lại CSV" | D2 và D3 là phần có nội dung; D1 chỉ là dẫn nhập, không kéo dài |
| Bị hỏi "sao không dùng bản định nghĩa sửa" | Đã có sẵn câu trả lời ở D4, nói trước khi bị hỏi |

> **Lưu trữ bản kế hoạch 02/09:** nội dung bên dưới phản ánh quyết định của giai đoạn đó, không còn là hướng hiện hành. Nguồn thuật ngữ và claim hiện tại là [T1 v1.1.0](week8/T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) và [claim map](week8/T1_CLAIM_MAP_2026-10-05.md). Các nhận định cũ về OEH saturation/IQL chỉ dùng để truy vết.

> Cập nhật sau audit: phạm vi chứng minh hiện hành ở `paper/sections/03b_axioms_and_impossibility.tex`. Chiều đảo từ A1 đến tính thuần nhất từng hạng đã được rút vì R1–R2 chưa đủ; Định lý 1 cần loss chuẩn hóa bị chặn và kết luận tiệm cận cần dispersion tổng hợp tiến về 0. Các mục dưới đây là kế hoạch/tường thuật trước đính chính; không dùng chúng để khẳng định đã chứng minh phân loại đầy đủ. Xác nhận của cố vấn còn mở.

# Khung tường thuật đồ án — bốn đóng góp, C1 làm trục chính

Trạng thái: **dự thảo, chờ nhóm duyệt.** Đây là tài liệu định hướng cho tuần
4–5 của `PHASE_B_PLAN.md` (viết lại `paper/sections/`) và tuần 12–14 (viết
chương). Nó **không** thay `PLAN_Final.md` §12 — bảng ánh xạ chương ở đó giữ
nguyên; tài liệu này chỉ định nghĩa lại *nội dung* mỗi chương nói gì sau khi C2
ra âm.

Người soạn: Linh (Workstream 4), 02/09/2026.

Nguồn số liệu: `artifacts/week2/c2_correlation_2026-09-02/`,
`positive_control_2026-09-01/`, `saturation_audit_2026-08-25/`,
`saturation_sweep_2026-08-24/`, `docs/DERIVATIONS.md`.

---

## 1. Vì sao phải viết lại, không phải sửa vặt

Paper hiện tại được viết cho **giả thuyết dương**. Hai chỗ neo của nó nay bị
chính dữ liệu của nhóm bác:

| Neo hiện tại | Ở đâu | Trạng thái đo được |
|---|---|---|
| Luận điểm trung tâm là **early identifiability** | `01_intro.tex`, khối `\paragraph{Claim.}` | **Trượt.** K1a trung vị 0,50 (CFQI) và 5,00 (IQL), ngưỡng đăng ký 0,25 |
| Đóng góp 2: **depth–evidence tradeoff** là hiện tượng có thật | `01_intro.tex`, mục 2 danh sách Contributions | **Không quan sát được** dưới định nghĩa đăng ký: `m_k ≡ 1` ở toàn bộ 12.019 điểm đường cong, `k† = 0` ở 62/62 pilot |

Sửa vặt hai câu này sẽ để lại một bài không có luận điểm. Cần một trục mới, và
trục đó **đã có sẵn** — chỉ là nó đang nằm ở Phụ lục chứ không ở Chương 3.

---

## 2. Luận điểm trung tâm mới

> **Một thống kê "yêu cầu bằng chứng" tính hoàn toàn từ dữ liệu offline phải
> thoả ba tiên đề. Lớp tự nhiên nhất — lớp mà chính chúng tôi đăng ký trước —
> vi phạm bất biến thang đo và suy biến về hằng số, và điều đó chứng minh được
> chứ không phải quan sát được. Lớp hợp lệ thì dựng được. Trên benchmark này,
> ngay cả lớp hợp lệ cũng không vượt một baseline chỉ dùng `|S||A|`, và nguyên
> nhân đo được chứ không phải phỏng đoán.**

Ba tính chất của câu này, và vì sao chọn nó:

1. **Nó là một đặc trưng hoá, không phải một thất bại.** Đối tượng của bài
   chuyển từ *"chúng tôi đề xuất OEH"* sang *"chúng tôi xác định khi nào một
   thống kê loại này có thể hoạt động"*. Kết quả âm trở thành **một trong bốn
   ô** của bảng đặc trưng hoá, không phải toàn bộ bài.
2. **Phần lõi là định lý, không phải thí nghiệm.** Định lý 1 đúng cho mọi
   benchmark, mọi bundle, mọi seed. Không phản biện nào bác được nó bằng cách
   nói benchmark tự dựng quá dễ hoặc quá khó.
3. **Nó sinh ra một công cụ dùng được.** Xem §4 dưới đây — đây là chỗ đồ án có
   phần "chạy được" mà một kết quả âm thuần tuý không có.

### Tiêu đề đề xuất

Tiêu đề hiện tại `Difficulty-Aware Offline Reinforcement Learning` hứa một hệ
thống nhận biết độ khó. Bài không giao được lời hứa đó nữa. Đề xuất:

> **Khi nào một thống kê độ khó tính từ dữ liệu offline có thể hoạt động: tiên
> đề, một định lý bất khả, và một phép kiểm khả thi**
>
> *(When Can a Data-Only Difficulty Statistic Work? Axioms, an Impossibility
> Theorem, and a Feasibility Test for Offline RL)*

Tiêu đề mới hứa đúng cái bài giao được. Cần nhóm chốt trước khi sửa
`paper/main.tex`.

---

## 3. Bốn đóng góp — phát biểu, bằng chứng, chương chứa

| | Đóng góp | Phát biểu một câu | Artifact chống lưng | Chương |
|---|---|---|---|---|
| **C1** | Yêu cầu thiết kế + trần bão hoà có điều kiện + lớp dựng | A1–A3 là desiderata; registered loss vi phạm A1 trên instance không tầm thường, và loss bậc một có normalized factor bị chặn sẽ bão hoà nếu aggregated dispersion tiến về 0; lớp phi thứ nguyên thoả cả ba thì dựng được | `docs/DERIVATIONS.md` §1–3; kiểm chứng 62/62 pilot, 0 vi phạm trần | **3** (trục chính) |
| **C2** | Bản sửa không vượt baseline cấu trúc | **Metric phụ** (tương quan hạng): không định nghĩa nào trong bốn định nghĩa vượt `S·A` ở bất kỳ phạm vi nào trong bảy phạm vi họ hành vi, hai định nghĩa kém hơn có ý nghĩa sau Holm. **Metric chính đã đăng ký** (likelihood sinh tồn rời rạc + integrated Brier): không vượt baseline nào **và cũng không kém** — ở `n = 62` đây là thiếu lực, và phải viết đúng như vậy | `c2_correlation_2026-09-02/`, `calibrated_comparison_2026-09-06/` | **5** |
| **C3** | Bộ công cụ chẩn đoán tái sử dụng được | Phép kiểm khả thi tính được từ pilot: trần chính xác, hệ số nhân `σ` còn thiếu, positive control harness, registry append-only | `scripts/analysis/audit_saturation_sweep.py`, `scripts/validation/run_positive_control.py`, `scripts/analysis/run_c2_correlation.py` | **6** |
| **C4** | Chẩn đoán bão hoà sau khi loại hai nhiễu benchmark | Đói reward và pha loãng occupancy là hai phản bác có cơ sở; hành lang có mồi nhử loại được cả hai mà `k† = 0` vẫn giữ trên 32/32 ô. Probe chỉ loại hai nhiễu này; không chứng minh tính chất phổ quát của định nghĩa hay learner không học được | `saturation_audit_2026-08-25/`, `depth_probe_2026-09-06/`, `docs/week5/H1_SCIENTIFIC_VALIDATION_2026-09-15.md` | **5**, mục cơ chế |

**Ba trên bốn là dương.** Đây là câu phải nói được trong 30 giây đầu buổi bảo
vệ, và là lý do cách đóng khung này không phải là tô hồng một thất bại.

---

## 4. C3 là chỗ đồ án có phần chạy được — cần nâng cấp cách trình bày

Định lý 1 cho một **điều kiện cần tính được từ dữ liệu, trước khi tiêu bất kỳ
compute nào**:

```
L_k(1) ≤ 0,16997 · Σ_s w_k(s)·σ_k(s)     ⇒     cần  σ̄ ≥ ε / 0,16997 = 0,2942
```

Nghĩa là: đưa vào một pilot buffer, trả lời được **trước khi chạy** rằng lớp
metric này có cơ hội hoạt động trên dữ liệu đó hay không, và nếu không thì còn
thiếu bao nhiêu lần. Máy móc đã có sẵn — `scripts/analysis/audit_saturation_sweep.py` tính trần
chính xác và hệ số nhân cần thiết; `code/positive_control.py` đã có
`sigma_headroom`.

Kết quả khi áp lên chính benchmark của nhóm: **40/62 pilot bất khả thi ở mọi
`σ`**; 19/62 còn lại cần nhân `σ̂` lên tối thiểu 2,32×, trung vị 6,29×.

Điều này đổi câu chuyện từ *"chúng em thử và không được"* thành *"chúng em xây
phép kiểm khả thi mà trước đây chưa ai có, và đây là kết quả khi áp nó lên 62
pilot"*. Chương 6 nên mở bằng công cụ này chứ không phải bằng danh sách mở rộng.

Việc cần làm: đóng gói thành một entry point có tên riêng (đề xuất
`scripts/analysis/run_feasibility_check.py`) đọc một buffer và in ra verdict + headroom. Chưa
làm; xem `DEMO_PLAN.md` module D3.

---

## 5. Những câu phải bỏ hoặc viết lại — danh sách cụ thể

Đây là phần quan trọng nhất của tài liệu này. Mỗi dòng là một chỗ trong
`paper/sections/` hiện tại nay **sai hoặc không còn bằng chứng**. Tuần 4–5 đi
theo bảng này.

**Trạng thái 06/09/2026:** 9/10 dòng đã xử lý trong lần viết lại gói `paper/`
cùng ngày (xem `paper/README.md`). Cột **Đã làm** ghi kết quả thật, kể cả hai
chỗ làm **khác** với chỉ dẫn ban đầu — ghi ra thay vì để lệch âm thầm.

| File | Chỗ | Vấn đề | Xử lý | Đã làm 06/09 |
|---|---|---|---|---|
| `01_intro.tex` | Khối `\paragraph{Claim.}` — *"can be identified from a pilot dataset several times smaller… and used to predict per-algorithm data requirements"* | K1a trượt (0,50 / 5,00 so với ngưỡng 0,25); phần "predict" chưa bao giờ được thiết lập | **Thay hẳn** bằng luận điểm §2 | ✅ |
| `01_intro.tex` | Contribution 2 — *"The depth–evidence tradeoff, an offline-specific phenomenon: required evidence is non-monotone in lookahead depth"* | Dưới định nghĩa đăng ký, `m_k ≡ 1` ở mọi độ sâu, `k† = 0` ở 62/62 | **Viết lại có điều kiện**: hiện tượng *tồn tại* (positive control, `k† = 2`) nhưng *không xuất hiện* trên benchmark, và Định lý 1 giải thích vì sao | ✅ |
| `01_intro.tex` | Contribution 1 — *"a dataset-only measure that keeps the depth-plus-evidence form"* | Vẫn đúng như mô tả, nhưng không còn là đóng góp số 1 | **Hạ xuống** sau C1; đóng góp số 1 thành hệ tiên đề + định lý | ✅ |
| `07_discussion.tex` | *"that it predicts requirements on unseen instances"* | Chưa từng đo trên instance chưa thấy; evaluation split còn niêm phong | **Bỏ** | ✅ |
| `07_discussion.tex` | `\paragraph{A negative result would be informative.}` — toàn khối ở thể giả định | Kết quả âm **đã là kết quả**, không còn là khả năng | **Chuyển lên Chương 1**, viết ở thể khẳng định | ✅ |
| `06_results.tex` | Figure 1 — *"the decomposition is what turns the interior minimum from an observation into an explanation"* | Không có cực tiểu trong nào trên benchmark; chỉ có ở positive control | **Đổi trục hình**: đặt cạnh nhau positive control (`k† = 2`) và pilot thật (`k† = 0`) | ⚠️ **Làm khác.** Positive control thành **Bảng 1** (có xuất xứ từng instance) chứ không phải một panel của hình; Hình 1 thành ba panel cơ chế trên hành lang mồi nhử (`gap`, `σ̄`, `L_k(1)` cùng `ε` và trần Định lý 1). Lý do: gói chỉ còn **một** hình bắt buộc, và cái duy nhất không thể thay bằng bảng là chỗ `L_k` sụp xuống đúng tại đỉnh khoảng cách |
| `06_results.tex` | Figure 2 — phân phối `N_k/N_emp` cho K1a | Vẫn vẽ được, nhưng là hình của một tiêu chí đã trượt | **Giữ**, đổi caption thành "tiêu chí trượt, đây là mức độ trượt" | ⚠️ **Bỏ.** K1a không còn *trượt* mà **không đo được**: `k† ≡ 0` nên không có ngân sách nào để "ổn định". Vẽ phân phối `N_k/N_emp` của một đại lượng hằng là vẽ nhiễu. Thay bằng Hình 2 (phép kiểm khả thi, cắt được nếu thiếu chỗ), và `06_results.tex` xếp K1a/K1b vào nhóm **không đạt được dưới định nghĩa này** |
| `05_experiments.tex` | RQ2 — *"do the features predict solvability and data requirement better than calibrated dataset baselines?"* | Nay đã trả lời: **không** | **Giữ nguyên câu hỏi**, thêm câu trả lời. Không sửa câu hỏi sau khi thấy kết quả (quy tắc 2) | ✅ đoạn `\paragraph{Answers.}` mới, chữ RQ giữ nguyên |
| `05_experiments.tex` | RQ3 — *"Do the conclusions hold on behavior policies not generated from `Q*`"* | Nay đã trả lời bằng tiêu chí 4: kết luận âm **giữ nguyên** trên bốn họ không-`Q*` | **Giữ**, thêm bảng bỏ-một-họ | ◻️ câu trả lời đã có; **bảng bỏ-một-họ chưa có** |
| `main.tex` | `\title{Difficulty-Aware Offline Reinforcement Learning}` | Hứa một hệ thống bài không giao được | **Đổi** theo §2 | ✅ |

**Nguyên tắc khi sửa:** không sửa câu hỏi đã đăng ký sau khi thấy kết quả
(`EXPERIMENT_RULE.md` quy tắc 2). RQ2 và RQ3 giữ nguyên chữ, chỉ thêm câu trả
lời. Cái được viết lại là **luận điểm** và **danh sách đóng góp**, vì hai thứ
đó là diễn giải chứ không phải giả thuyết đã đăng ký.

---

## 6. Ánh xạ chương — nội dung mới của từng chương

Bảng chương của `PLAN_Final.md` §12 giữ nguyên. Đây là nội dung sau khi đổi trục.

| Chương | Nói gì (mới) | Nguồn `paper/sections/` | Mức viết lại |
|---|---|---|---|
| **1. Mở đầu** | Bài toán; vì sao khó; **luận điểm §2**; bốn đóng góp; nói thẳng kết quả C2 âm ngay ở mở đầu | `01_intro.tex` | **Viết lại phần Claim + Contributions**; phần bối cảnh giữ |
| **2. Tổng quan** | 15–20 trang. Thêm nhánh mới: các nỗ lực đo độ khó bằng thống kê tính từ dữ liệu, và vì sao chúng không nêu tiên đề | `02_related.tex` nới ra | **Phần lớn viết mới** (đã lên lịch tuần 12–14) |
| **3. Cơ sở lý thuyết** | **Trục chính.** A1–A3; Bổ đề 1 (`max z·Φ(-z)`); Định lý 1 (bất khả); Định lý 2 (dựng được); bảng phân loại bốn định nghĩa theo tiên đề; **điều kiện cần đo được `σ̄ ≥ 0,2942`** | `03_definitions.tex` + `docs/DERIVATIONS.md` | **Mở rộng lớn.** Hiện chỉ có định nghĩa; phải có định lý gốc và chứng minh đầy đủ |
| **4. Thiết kế thực nghiệm** | Estimator data-only; benchmark; giao thức; **positive control là một mục riêng, không phải phụ lục** | `04_method.tex`, `05_experiments.tex` | Giữ phần lớn; **nâng positive control lên** |
| **5. Kết quả** | Thứ tự bắt buộc: (a) positive control PASS → pipeline không hỏng; (b) bão hoà 0/600 ô; (c) audit cơ chế; (d) C2 và tiêu chí 3/4 | `06_results.tex` | **Đổi thứ tự và đổi trục hình**; số thì đã có |
| **6. Mở rộng và ứng dụng** | Mở bằng **phép kiểm khả thi (C3)**; rồi xấp xỉ hàm (tuần 6–7); miền liên tục (tuần 8–9); adaptive conservatism (tuần 10–11) | Không có trong paper | **Toàn bộ mới** |
| **7. Kết luận và hạn chế** | Phạm vi §7 dưới đây; phụ thuộc còn mở; hướng phát triển | `07_discussion.tex` | Giữ khung, **bỏ thể giả định** |

---

## 7. Thứ tự tường thuật — bắt buộc, không phải gợi ý

`DEMO_PLAN.md` §1 đã chốt thứ tự này cho demo. Chương 5 và phần trình bày phải
theo cùng thứ tự, vì cùng một lý do:

```
1. Định lý      →  kết quả âm là bắt buộc về mặt toán, không phải rủi thôi
2. Positive control →  cùng dòng code phát hiện được tín hiệu khi tín hiệu có
3. Kết quả âm   →  giờ mới đọc được như một phát hiện
4. Cơ chế       →  và đây là vì sao
```

Đảo thứ tự này — trình bày kết quả âm trước — thì mọi thứ sau nó bị nghe như
biện minh. `PHASE_B_PLAN.md` §4 câu 3 (*"kết quả âm do benchmark tự dựng quá
dễ/khó?"*) chỉ trả lời được nếu audit đứng **trước** kết luận.

---

## 8. Phạm vi phải nói rõ, và phụ thuộc còn mở

Ba câu này phải xuất hiện nguyên văn trong Chương 7, không được làm mềm:

1. **Kết luận C2 là kết luận trên benchmark này** — một generator, 69% nhãn bị
   kiểm duyệt phải, 13/15 MDP còn dùng được, 37 cặp phân tích. Không phải một
   phát biểu về offline RL nói chung.
2. **Xuất xứ của từng positive control phải nói rõ** — một fixture tham chiếu
   dựng trong repo bởi workstream sở hữu estimator, và **hai MDP do Workstream 1
   dựng độc lập** (03/09), cả ba PASS trên estimator đăng ký không sửa. Câu
   *"positive control đã PASS"* nay đứng được, nhưng vẫn phải kèm xuất xứ; nói
   trước khi bị hỏi thì nó là bằng chứng, để hội đồng tự phát hiện thì nó là
   một lỗ hổng.
3. **Bốn định nghĩa thay thế được viết sau khi thấy bản đăng ký bão hoà**, nên
   là giả thuyết chưa kiểm định chứ không phải kết quả. Nói trước khi bị hỏi;
   `PHASE_B_PLAN.md` §4 câu 4 ghi rõ điều này cộng điểm liêm chính.

Ngược lại, một câu **không** được để lửng vì nó đã đóng: `σ̄ ≥ 0,294` **không**
bất khả với reward trong `[0,1]` và `γ = 0,95` — reference control đạt 0,568,
thăm dò cùng họ đạt 5,33.

---

## 9. Việc tuần 4–5 chỉ việc bám theo

| Thứ tự | Việc | Bám vào mục nào | Trạng thái 06/09 |
|---|---|---|---|
| 1 | Chốt tiêu đề mới và luận điểm trung tâm | §2 | ✅ |
| 2 | Viết lại `01_intro.tex`: Claim + Contributions | §2, §3, và bảng §5 | ✅ |
| 3 | Mở rộng `03_definitions.tex` thành chương lý thuyết, kéo `DERIVATIONS.md` §1–2 vào | §6 dòng Chương 3 | ✅ §*Axioms, an Impossibility Theorem, and the Admissible Class* |
| 4 | Đổi thứ tự và trục hình `06_results.tex` | §7 | ✅ Viết lại theo đúng thứ tự §7: positive control → kết quả âm → cơ chế → C2 → khả thi. Ba bảng có số thật |
| 5 | Bỏ thể giả định khỏi `07_discussion.tex`, thêm ba câu phạm vi | §8 | ⚠️ **Một nửa.** Thể giả định đã bỏ. Ba câu phạm vi §8: câu 1 (kết luận trên benchmark này) và câu 3 (bốn định nghĩa viết sau) nay có trong `06_results.tex`; **câu 2 — xuất xứ từng positive control — đã có ở Bảng 1**. Nhưng §8 yêu cầu ba câu xuất hiện **nguyên văn trong Chương 7**, và `07_discussion.tex` chưa có |
| 6 | Bảng "phản biện hỏi gì → artifact nào trả lời", ≥10 câu | `PHASE_B_PLAN.md` tuần 4–5 | ◻️ **Chưa làm** |

**Hai việc còn mở sau 06/09:** mục 5 nửa sau (ba câu phạm vi vào Chương 7
nguyên văn) và mục 6. Cả hai là việc viết, không chờ số nào.

**Hạng mục chặn của gói `paper/`:** Hình 1 là **bắt buộc** và chưa tồn tại —
kho không có file ảnh nào, không code vẽ, `matplotlib` không nằm trong
`requirements.txt`.

**Chốt chặn thật không nằm trong bảng này:** giảng viên phải đọc và xác nhận
Định lý 1 bằng văn bản (`PHASE_B_PLAN.md` tuần 2–3). Nếu định lý được xác nhận
là đóng góp gốc, toàn bộ khung trên đứng vững. Nếu giảng viên đánh giá nó là hệ
quả tầm thường của một bất đẳng thức một biến, thì C1 không gánh nổi vai trò
trục chính và phải dồn trọng lượng sang Chương 6 (xấp xỉ hàm, miền liên tục) —
đó là một khung khác, và phải biết điều đó **trước** tuần 4, không phải sau.

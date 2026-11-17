# Quyết định giao thức — 16/08/2026

**Ghi chú bắt buộc đọc trước.** Mọi số đo định lượng nhắc tới trong tài liệu này (tỉ lệ occupancy do `|𝒜_K(s)|=1` chiếm, phân bố `N_emp` trên lưới, v.v.) được đo trên nhánh `feat/benchmark-dataset-v2`, nơi đã có benchmark 120+15+3 và năm họ behavior policy. Nhánh `linh` hiện tại **chưa tái lập được** các số đo này vì merge request của Việt Nguyễn chưa chạy, nên chỉ có 14 fixture deterministic trong `environments/tabular_environments.py`. Phải chạy lại toàn bộ số đo ở tài liệu này sau khi merge xong trước khi coi chúng là căn cứ thật cho quyết định cuối.

Còn trong cửa sổ cho phép sửa định nghĩa (đóng băng 21/08, mục 3 PLAN_Final.md, quy tắc 1 mục 11). Sáu quyết định dưới đây đã áp vào `PLAN_Final.md` mục 3.2, Phụ lục A, mục 13, và đồng bộ vào `paper/sections/03_definitions.tex`.

---

## D1 — Phạm vi chứng chỉ tính được: `S_crit` thay vì bao đóng khả đạt

**Trước:** mục 3.2 kiểm `𝒜_K(s) = ∅` trên "bao đóng khả đạt" của `𝒜_K` — không có định nghĩa hình thức về ranh giới của bao đóng đó, và nó có thể bao gồm cả những trạng thái không bao giờ được `π̂_k` ghé thăm với xác suất đáng kể.

**Sau:** kiểm trên `S_crit = {s : d̂(s) ≥ τ}`, `τ = 1/(10|S|)` — đúng ngưỡng đã đóng băng ở Phụ lục A cho mục 3.4, không thêm hằng số mới.

**Lý lẽ.** Mục 3.4 đặt `w_k(s) = 0` với mọi `s ∉ S_crit`, nên trạng thái ngoài `S_crit` không đóng góp một chút nào vào `L_k(μ)`, do đó không ảnh hưởng `m_k`, `k†`, `H_OEH`, hay `N̂_req`. Từ chối trả lời vì thiếu độ phủ ở một trạng thái không đóng góp gì vào câu trả lời là từ chối vì một khiếm khuyết không liên quan — làm tỉ lệ từ chối trả lời (tiêu chí thành công 6, mục 8.2) cao giả tạo mà không phản ánh chất lượng ước lượng.

**Rủi ro đã cân nhắc:** thu hẹp phạm vi kiểm có thể che giấu một `𝒜_K(s)=∅` ở trạng thái *sắp* vào `S_crit` khi occupancy ước lượng lệch. Chấp nhận rủi ro này vì `d̂` chính nó ước lượng từ cùng mô hình thực nghiệm — nếu occupancy ước lượng sai đủ để đẩy một trạng thái quan trọng ra ngoài `S_crit`, đó là lỗi ở `d̂`, không phải lỗi của việc thu hẹp phạm vi, và nó sẽ lộ ra qua K2/K5 chứ không lộ qua chứng chỉ tính được.

---

## D2 — Quy tắc cho `|𝒜_K(s)| = 1`

**Trước:** mục 3.2 chỉ có quy tắc cho `𝒜_K(s) = ∅`. Trường hợp còn đúng một hành động có support (rất phổ biến với `n_min` nhỏ) không được xử lý — cách cài "tự nhiên" nhất là gán `Δ̂_k(s) = 0` vì không có hành động thứ hai để so sánh, giống hệt cách xử lý trạng thái đồng hạng thật ở mục 3.4.

**Sau:** state có `|𝒜_K(s)| = 1` bị loại khỏi tổng trong `L_k`; occupancy `w_k(s)` của nó cộng dồn vào một đại lượng riêng `w_gap(k) = Σ_{s: |𝒜_K(s)|=1} w_k(s)`. Nếu `w_gap(k†) > ε`, kết quả tại `k†` được báo cáo là **BỊ KIỂM DUYỆT PHẢI**, không gán `Δ̂ = 0`.

**Lý lẽ.** Đồng hạng thật (`Δ̂_k(s) = 0` với `|𝒜_K(s)| ≥ 2`) và thiếu support (`|𝒜_K(s)| = 1`) là hai tình huống khác nhau về nhận thức luận: đồng hạng nghĩa là "đã biết, không quan trọng chọn gì"; thiếu support nghĩa là "chưa biết, không có dữ liệu để so sánh". Gán `Δ̂ = 0` cho trường hợp thứ hai làm giảm giả tạo `L_k(μ)` (đóng góp 0 vào tổng), khiến `m_k` nhỏ giả tạo — đúng kiểu sai số hệ thống mà quy tắc 4 mục 11 (không tham số tự do, "nếu thấy mình đang cân trọng số, dừng lại") muốn ngăn.

**Vì sao không phải một dung sai `η`.** Đây không phải vấn đề chia cho khoảng cách nhỏ (mục 3.4 đã xử lý bằng cách đặt khoảng cách ở tử số) — vấn đề là *không có* khoảng cách để tính, khác về loại chứ không phải về độ lớn. Không cần hằng số mới; `w_gap` chỉ là tổng của các `w_k(s)` đã có sẵn.

---

## D3 — Phụ lục A: `n_min` 5 → 2, thêm `horizon = 100`

**`n_min` 5 → 2 (robustness 3, 5).** Với `n_min = 5` và `N₀ = 5|S||A|` (mức thí điểm nhỏ nhất đóng băng), số lượng trạng thái đạt support đầy đủ trên mọi hành động thường quá ít để tính được `S_crit` có ý nghĩa ở đúng mức ngân sách mà K1a cần kiểm sớm nhất. `n_min = 2` là ngưỡng đếm thô tối thiểu; D5 sau đó siết chứng chỉ bằng điều kiện bổ sung rằng cặp `(s,a)` phải xuất hiện trong ít nhất hai trajectory khác nhau. Vì vậy các lần lặp trong cùng một trajectory không bị coi là hai đơn vị bootstrap độc lập. Dải robustness dịch theo, 3 và 5, giữ đúng khoảng cách tương đối với giá trị chính cũ.

**Thêm `horizon = 100`.** Trước đây không có dòng đóng băng cho độ dài episode thu thập dữ liệu; `pilot_collector.py` (đầu mục 3) và các môi trường thăm dò (đầu mục 2) cần một giá trị cố định trước khi chạy. 100 khớp với `K_max = 100` đã đóng băng — backup không thể có ý nghĩa sâu hơn số bước một episode có thể thực sự đi qua.

---

## D4 — Rủi ro mới: ràng buộc giữa `N₀`, lưới `𝒩` và ngưỡng 0,25 của K1a

`N₀ = 5·|S||A|` đã lớn hơn hai mức đầu của lưới `𝒩 = {1, 2, 5, 10, 20, 50}·|S||A|`. Ngưỡng K1a đòi `median(N_k/N_emp) ≤ 0.25`. Nếu `N_emp` — nhu cầu dữ liệu thực nghiệm đo ở mục 3.1 — thường rơi vào các mức thấp nhất của lưới (tức bài toán "dễ", giải được với rất ít dữ liệu), thì `0.25 · N_emp` có thể nhỏ hơn cả `N₀`, nghĩa là **không có ngân sách thí điểm nào hợp lệ** (`N_k ≥ N₀` theo định nghĩa) thoả điều kiện K1a — bất kể chất lượng ước lượng.

Đây là ràng buộc giữa các hằng số đã đóng băng với nhau, không phải giữa hằng số và kết quả thực nghiệm — nên **phải kiểm trước 21/08**, không phải là điều phát hiện được sau và sửa ngưỡng cho khớp (cấm bởi quy tắc 2 mục 11). Cách kiểm: đo phân bố `N_emp` trên `𝒩` ở tập phát triển (15 MDP), xem tỉ lệ MDP có `N_emp` nằm ở hai mức đầu lưới. Nếu tỉ lệ đó cao, bàn lại `N₀` hoặc lưới `𝒩` — cả hai vẫn trong cửa sổ đổi được tới 21/08.

Đã ghi vào `PLAN_Final.md` mục 13 dưới dạng một hàng rủi ro riêng.

---

## D5 — 17/08/2026: Bayesian trajectory bootstrap và support theo trajectory độc lập

**Phạm vi cho phép.** Linh cho phép sửa OEH để giải quyết lỗi computability
trước phase sau, với điều kiện mọi thay đổi được ghi rõ. Thay đổi này không dùng
evaluation split, true `P/R/Q*`, family label hay kết quả thuật toán downstream.

**Trước.** Mỗi replicate lấy mẫu trajectory có hoàn lại. Support dùng duy nhất
raw transition count `n(s,a) ≥ 2`. Replicate nào không còn đủ mọi cặp support
thì bị loại; dưới 90% replicate hợp lệ thì abstain.

**Vấn đề quan sát được.** Development audit cố định trên 15 MDP,
`q_star_softmax`, một pilot seed và các prefix `{2,5,10}|S||A|` cho
`15/15 abstained_at_cap`: tại cap, 12 case dưới gate 90% và 3 case critical
support rỗng. Đây là xung đột cấu trúc: một pair có đúng hai đơn vị trajectory
support có xác suất khoảng `e^-2 ≈ 13,5%` biến mất trong resampling, nên ngay
một pair đã khó đạt gate 90%; nhiều pair làm xác suất toàn mô hình hợp lệ giảm
nhanh. Raw count 2 còn có thể đến từ cùng một trajectory và không phải hai đơn
vị bootstrap độc lập.

**Sau.** Với `T` trajectory, replicate `b` lấy
`ω^(b) ~ Dirichlet(1,…,1)` rồi tính mô hình từ các tổng có trọng số trajectory.
Đây là Bayesian bootstrap chuẩn của
[Rubin (1981)](https://doi.org/10.1214/aos/1176345338). Trọng số dương gần như
chắc chắn nên full-pilot support được giữ trong mọi replicate. Support mới yêu
cầu đồng thời `n(s,a) ≥ n_min=2` và cặp xuất hiện trong ít nhất
`c_min^τ=2` trajectory ID phân biệt. Gate 90% chỉ còn bắt lỗi giải số học hoặc
giá trị không hữu hạn, không còn thay cho coverage certificate.

**Tie-break số học.** Nếu nhiều `m_k` bằng nhau trong tolerance cố định
`10^-12`, chọn `k` nhỏ nhất. Quy tắc này phục hồi trường hợp bandit một trạng
thái `k†=0` và không dùng dữ liệu evaluation để chọn tolerance.

**Audit sau thay đổi, cùng dữ liệu/seed.** Tất cả bootstrap-valid fraction qua
15 MDP và ba prefix đều thành `1.0`; kết quả tại cap là 13 abstain vì critical
support rỗng và 2 censor vì critical occupancy có đúng một supported action,
không có case selected. Diagnostic không đăng ký tại `50|S||A|` cho 11 abstain,
4 censor, 0 usable. Vì vậy chỉ tăng cap bị loại: nó tăng chi phí nhưng không
chữa policy coverage. Không thêm transition, không hạ support và không đổi
abstain thành số giả. Coverage/pilot-policy review vẫn là cổng pending riêng.

**Chi phí.** Weighted bootstrap phải chạy đủ replicate thay vì loại sớm nên
development audit chậm hơn. CI vẫn dùng `B=100`; runtime phải được đo trong
probe 1% trước full experiment.

---

## D6 — 17/08/2026: terminal occupancy, exact-reference parity và audit hằng số chết

**Phạm vi review.** Sau khi Linh cho phép chỉnh OEH nhưng yêu cầu ghi rõ mọi
thay đổi, toàn bộ đường data-only, exact reference, config và paper được rà lại
như một pre-merge audit. Không dùng evaluation split hoặc kết quả CFQI/IQL.

**Lỗi terminal trước sửa.** Collector kết thúc trajectory khi transition đi
vào terminal và không bao giờ ghi action từ terminal. Tuy nhiên occupancy của
OEH lại gán self-loop cho mọi state không support. Terminal vì thế tích luỹ
occupancy vô hạn, lọt vào `S_crit`, rồi bị báo sai là
`empty_supported_action_set_in_critical_state`. Đây là lỗi implementation, không
phải thiếu coverage của behavior policy.

**Terminal sau sửa.** Destination có `terminal=True` được suy ra hoàn toàn từ
pilot, không cần oracle. Nó không phải decision state, không đòi action support
và bị loại khỏi `S_crit`. Bellman value dùng continuation bằng 0 sau terminal;
occupancy chuyển terminal mass về `ρ0`, đúng chuỗi episode tự nhiên của
collector. `truncated=True` chỉ là ranh giới thu thập, không cắt động lực mô
hình. Cờ terminal mâu thuẫn, terminal đồng thời truncated, hoặc action xuất
phát từ terminal đều bị từ chối như dữ liệu lỗi.

**Ba lệch định nghĩa khác đã sửa.** (1) Exact curve trước đây tính
`Q^(0)=Q^β` dù estimator dùng `Q^(0)=Q^{β^K}`; exact reference giờ chuẩn hoá
`β` trên cùng support và cho phép terminal có support rỗng. (2) Tie depth ghi là
tolerance tuyệt đối `10^-12` nhưng code dùng cả relative tolerance; giờ dùng
`rtol=0`, đồng bộ data-only và exact curve, còn action-value tie chọn action ID
nhỏ nhất. (3) `confidence_level=0.95`/`z*` là hằng số chết, không xuất hiện
trong định nghĩa `L_k(μ)` hay phép tìm nghiệm; đã xoá khỏi config và appendix
thay vì để một knob giả. Giá trị Q/bootstrap không hữu hạn cũng bị loại rõ ràng
thay vì âm thầm đổi `NaN` thành độ bất định 0.

**Audit chốt sau sửa.** Trên đúng 15 development MDP, policy
`q_star_softmax` đã đăng ký theo `variant_index`, pilot seed index 0,
bootstrap seed `20260816`, `B=100`: selector `{2,5,10}|S||A|` cho 2 selected
(đều tại `5|S||A|`), 11 censored-at-cap và 2 abstained-at-cap; mọi phép tính
được đều có bootstrap-valid fraction `1.0`. Diagnostic không đăng ký tại
`50|S||A|` cho 3 usable và 12 censored, không còn abstain. Vì tăng dữ liệu gấp
5 chỉ thêm một case usable và vẫn censor 80%, cap không được tự mở rộng để ép
kết quả; đây là coverage limitation thật của policy tập trung.

**Cổng còn mở, không được che.** Cả 13 case tính được ở cap 10 và cả 15 case ở
diagnostic 50 đều cho `k†=0`. Hai case kiểm chi tiết có `L_k(1)` tối đa chỉ
`0.00112` và `0.000157`, thấp xa `ε=0.05`, nên mọi `m_k` bị chặn ở cận dưới 1
và tie-break trả depth 0. Đây có thể là saturation của định nghĩa chứ không còn
là lỗi terminal/bootstrap. Không đổi `ε`, bỏ cận `μ≥1` hoặc thêm tie-break theo
kết quả sau khi đã thấy audit. `exact_k_dagger_range_audit` vì vậy là cổng
load-bearing: phải chứng minh có dải `k†` khác 0 trên exact/probe trước freeze;
nếu không, giả thuyết trung tâm chưa có biến thiên để kiểm định.

**Đính chính D5.** Diagnostic 50SA ghi ở D5 chạy trước khi sửa terminal nên số
`11 abstain, 4 censor, 0 usable` đã bị supersede; chỉ số D6 ở trên được dùng cho
quyết định hiện tại.

---

## Việc bị chặn tới khi merge xong

- Đo lại tỉ lệ occupancy do `|𝒜_K(s)|=1` chiếm ở `N₀` trên benchmark 120+15+3 thật (số liệu ~26% trích trong yêu cầu ban đầu là số đo trên `feat/benchmark-dataset-v2`, chưa tái lập trên nhánh `linh`).
- Đo phân bố `N_emp` trên lưới `𝒩` cho D4 — cần benchmark thật và năm họ behavior policy, cả hai đều chưa có trên nhánh này.
- Phép đo dải `k†` ở mục 5.1 PLAN_Final.md — điều kiện cần của cả nghiên cứu, đòi 120 thể hiện `mdp_suite.py` chưa tồn tại trên nhánh này.

## Chạy probe oracle — 16/08/2026 (không phải kết quả benchmark)

Đường cong bằng chứng oracle chạy với `R = 200`, `N0 = 5|S||A|`, policy hành vi ngẫu nhiên đều và seed Monte-Carlo `20260816`. Đây chỉ là kiểm định nghĩa trên ba MDP probe có ngẫu nhiên, không phải K2/K5 benchmark.

| Probe | Dataset support-valid / 200 | Kết quả oracle |
|---|---:|---|
| RiverSwim 6 state | 101 | `k† = 0`, `m0 = 1.0` |
| Combination Lock (slip 0.1) | < 2 | Không xác định: không đủ hai dataset giữ support cố định |
| GridWorld 5×5 (slip 0.1) | < 2 | Không xác định: không đủ hai dataset giữ support cố định |

Đây là kết quả âm về coverage tự nhiên ở `N0`, được giữ nguyên: không tăng `N0`, không đổi policy, không thêm exploring starts và không lọc/chèn transition để làm phép kiểm đạt. Chưa có phân bố `k†` đủ ba probe; K2/K5 sơ bộ bị chặn cho tới khi có protocol coverage hợp lệ được chốt trước hoặc benchmark sau merge.

---

## D7 — 24/08/2026: audit tổng hợp Tuần 2 trên dữ liệu development thật, cơ chế bão hoà `ε`, và mốc 21/08 đã trễ

**Phạm vi.** Không sửa `ε`, `n_min`, `K_max`, ngưỡng K1a hay bất kỳ hằng số đã đóng băng nào ở Phụ lục A. Mục này chỉ ghi lại một phép đo mới trên artifact thật đã có (`artifacts/week2/oeh_validation_2026-08-23/evidence_curves.csv`, 12.019 dòng) và một sự kiện lịch trình, để cả hai không bị bỏ sót khi nhóm ra quyết định cổng.

**D7.1 — Vì sao `m_k` phẳng tuyệt đối: cơ chế, không phải lỗi.** Toàn bộ 12.019 giá trị `L_k(1)` trên 15 MDP phát triển nằm trong khoảng `[0, 0.01523]` — **tối đa quan sát được nhỏ hơn `ε = 0.05` gần 3.3 lần**, và 94% số điểm dưới `0.01`. Vì `m_k = min{μ ≥ 1 : L_k(μ) ≤ ε}` và `L_k` không tăng theo `μ`, hễ `L_k(1) ≤ ε` (đúng ở mọi điểm đã đo) thì phép tìm nghiệm không bao giờ chạy — `m_k` bị trả về đúng ở cận dưới `μ=1` một cách máy móc, không phải vì không có cấu trúc độ sâu. Đây là hiện tượng **bão hoà ở cận dưới của phép đo**, khác về bản chất với "đường cong đơn điệu không có cực tiểu" (điều `PLAN_Final.md` §13 rủi ro #1 dự đoán) — đường cong ở đây thậm chí không có đủ độ phân giải để lộ hình dạng, vì thước đo (`ε=0.05`) quá thô so với thang giá trị `L_k(1)` thực tế xuất hiện ở quy mô 15 MDP phát triển (24–64 trạng thái).

**D7.2 — Vì sao điều này KHÔNG được sửa bây giờ, dù đã tìm ra nguyên nhân cơ học.** `ε` không phải một tham số tự do được đoán mò — nó được kế thừa có chủ đích từ ngưỡng thành công của `N_emp` ở §3.1 (`PLAN_Final.md` §3.4: "`ε` thừa hưởng từ ngưỡng thành công ở 3.1"), và cửa sổ được phép sửa định nghĩa đã đóng lúc 21/08 (xem D7.3). Đổi `ε` **bây giờ, sau khi đã thấy nó gây bão hoà `m_k=1` ở mọi trường hợp**, là đúng loại thao tác mà `PLAN_Final.md` §11 quy tắc 2 ("không bao giờ chọn ngưỡng sau khi đã nhìn kết quả") viết ra để ngăn — bất kể động cơ là muốn cứu đóng góp khoa học hay không. Ghi nhận D7.1 ở đây chính là để **không** làm việc đó trong im lặng: nếu nhóm quyết định thử lại với `ε` khác, đó phải là một protocol deviation công khai, có lý do độc lập với kết quả mong muốn, tăng phiên bản, chạy lại toàn bộ theo `docs/week2/WEEK2_BACKLOG.md`, và báo cáo cả hai lần chạy (cũ và mới) — không âm thầm thay số cũ.

**D7.3 — Mốc đóng băng 21/08 đã trễ, chưa được ghi nhận ở đâu.** Tính đến 24/08/2026, `protocol_config.json:protocol_status` vẫn là `"candidate"` với đủ 4 `pending_validations` gốc. `PLAN_Final.md` §9 quy định: "Trượt 21/08 thì không lùi tiếp — chuyển thẳng nhánh NO-GO ở cổng." Đây là **sự kiện lịch trình**, không phải phát hiện khoa học, nhưng theo đúng quy tắc 8 (§11, "mọi mốc thời gian phải kiểm") nó phải được ghi lại thay vì lặng lẽ trôi qua. Không có hằng số nào bị đổi vì việc này; đây thuần tuý là ghi nhận để cổng quyết định 31/08 không bị tính nhầm thời gian còn lại.

**D7.4 — Cổng chạy bằng `evaluation/decision_gate/gate.py::decide()` trên số liệu development thật.** Không phải đánh giá chủ quan — chạy trực tiếp bộ quy tắc đã đăng ký (`evaluation/decision_gate/gate.py`, `thresholds.py`) với input đo được từ `artifacts/week2/oeh_validation_2026-08-23/` và `artifacts/week2/development_offline_rl_2026-08-22/`. Kết quả (đầy đủ ở `artifacts/week2/decision_gate_input.json`, provenance commit `042d0b9`, config hash `39bfc530…`):

| Nhánh kiểm | Giá trị đo được | Ngưỡng | Qua? |
|---|---:|---:|---|
| Censoring rate (pilot, 10\|S\|\|A\| cap) | 38/75 = 0.51 | ≤ 0.40 | Không |
| Grid expansion khả thi? | Không (D5/D6: 50\|S\|\|A\| chỉ thêm 1 case usable, vẫn ~80% censor) | — | Không |
| K1a, `cfqi_tabular` | median 0.50 | ≤ 0.25 | Không (vùng yếu) |
| K1a, `iql_tabular` | median 5.00 | ≤ 0.25 | Không (đảo ngược hẳn) |
| K5 (`m_k` có cực tiểu trong trên exact curve) | Không — phẳng tuyệt đối, xem D7.1 | — | Không |

`decide()` trả về **NO_GO** cho cả hai thuật toán, qua **hai nhánh độc lập** (censoring/coverage thất bại trước cả khi xét tới K1a), nên kết luận không phụ thuộc một số đo duy nhất. Đây không phải quyết định cuối cùng của nhóm — `WEEK2_DECISION_GATE.md` cập nhật ngày 24/08 diễn giải đầy đủ và đề xuất hướng đi theo đúng bảng ở `PLAN_Final.md` §10.

---

## D8 — 24/08/2026: kiểm chứng lại trên dữ liệu thật theo họ cấu trúc, kết luận chưa thoát bão hoà, và xử lý pending validations

**Phạm vi.** Không chạy lại pipeline, không đổi hằng số. Toàn bộ số dưới đây tính trực tiếp từ artifact đã có (`evidence_curves.csv` 12.019 dòng, `benchmark_metadata.csv` 15 dòng), join theo `mdp_id`, để trả lời câu hỏi D7.1 để ngỏ: liệu `lock_structured`/`bottleneck` — hai họ được thiết kế riêng để có biên hành động nhỏ/chân trời dài theo §5.1/§5.4 — có thoát khỏi bão hoà `ε` hay không.

**D8.1 — Kết quả: KHÔNG thoát bão hoà, và hai họ được kỳ vọng nhất lại có biên nhỏ nhất, không phải lớn nhất.**

| Họ | Số pilot có curve | max `L_k(1)` | median `L_k(1)` | max/`ε` |
|---|---:|---:|---:|---:|
| `uniform_garnet` | 15 | 0.01523 | 0.00577 | 0.305 |
| `forward_layered` | 24 | 0.01242 | 0.00299 | 0.248 |
| `bottleneck` | 8 | 0.00194 | 0.00000 | 0.039 |
| `lock_structured` | 18 | 0.00015 | 0.00000 | 0.003 |

Không pilot nào, ở bất kỳ họ nào, vượt 31% ngưỡng `ε`. `lock_structured` — họ chính xác được §5.4 dự đoán là "góc cực trị k nhỏ, m khổng lồ" — lại có `L_k(1)` **nhỏ nhất trong bốn họ, nhỏ hơn `uniform_garnet` tới ~100 lần**. Đây không phải điều PLAN dự đoán, và cần được nêu rõ ràng thay vì bỏ qua.

**D8.2 — Vì sao: cấu trúc toán học của `L_k`, không phải lỗi.** `L_k(μ) = Σ w_k(s)·Δ̂_k(s)·Φ(−z_k(s)√μ)` và `Φ(·) ≤ 1`, nên `L_k(1) ≤ Σ w_k(s)·Δ̂_k(s) ≤ max_s Δ̂_k(s)` — **giá trị trần của `L_k(1)` bị chặn bởi chính biên hành động `Δ̂`, không phải bởi độ bất định `σ̂_Δ` (qua `z_k`)**. Ở các trạng thái mà `Δ̂` đã nhỏ (đúng như thiết kế của `lock_structured`: hành động đúng đã thắng ở mọi độ sâu, biên nhỏ theo đúng mô tả §5.4), `L_k(1)` nhỏ theo bất kể `σ̂_Δ` lớn cỡ nào — nên trục "cần thêm bằng chứng vì bất định cao" không bao giờ được kích hoạt ở `ε=0.05`. Có tương quan dương yếu giữa `num_states` và `max L_k(1)` trên toàn bộ 65 pilot đo được (Pearson r = 0.373) — hướng đúng nhưng yếu, và giá trị lớn nhất từng đo (`uniform_garnet`, 24 trạng thái, 0.01523) vẫn cách `ε` gần 3.3 lần.

**D8.3 — Kết luận cho câu hỏi "đã thoát bão hoà chưa": CHƯA, và bằng chứng hiện có không ủng hộ giả thuyết rằng chạy full 120-MDP sẽ tự nhiên thoát.** Hai họ được kỳ vọng đóng góp biên lớn nhất (`lock_structured` 36/120, `bottleneck` 18/120 ở tập đánh giá) lại là hai họ có `L_k(1)` nhỏ nhất trên tập phát triển — cùng recipe cấu trúc (`protocol_config.json:benchmark.environment_recipes`) được dùng cho cả development lẫn evaluation, nên không có lý do để kỳ vọng các thể hiện evaluation của đúng hai họ này đột nhiên có biên lớn hơn nhiều. Tương quan kích thước dương nhưng yếu (r=0.373) không đủ để dự đoán vượt ngưỡng `ε` ở quy mô `num_states` tương tự.

**D8.4 — Hướng nên chốt (khuyến nghị, cần nhóm phê duyệt chính thức).** Kết hợp D8.3 với D7.4 (hai nhánh `decide()` độc lập đã NO_GO — censoring/coverage đã thử mở rộng và thất bại theo D5/D6, K1a thất bại cho cả hai thuật toán, IQL đảo ngược hẳn luận điểm) và mốc 21/08 đã trễ (D7.3): **không có nhánh nào trong ba nhánh thất bại hiện tại (censoring, K1a, K5) có bằng chứng sẽ tự sửa khi chạy full grid với cùng định nghĩa.** Khuyến nghị:

1. **Không chạy lưới đầy đủ 120 MDP để "chờ" phần độ sâu tự thoát bão hoà** — dữ liệu ở D8.1–D8.3 cho thấy đây không phải nhánh có khả năng cao dựa trên bằng chứng hiện tại, và chi phí (thời gian tính toán + thời gian còn lại tới 25/09) không tương xứng với xác suất thành công.
2. Theo đúng bảng `PLAN_Final.md` §10: áp dụng đồng thời hàng **"K5 hỏng — đổi khẳng định, bỏ phần độ sâu, báo cáo như kết quả phủ định"** và hàng **"K1a hỏng — dừng hướng nộp bài, dồn về đồ án"** (rõ nhất với `iql_tabular`, ngân sách pilot gấp 5 lần dữ liệu cần). Đây là kết quả âm có phân tích cơ học đầy đủ (D7.1, D8.2) — đúng tinh thần quy tắc 7 §11 ("kết quả âm được báo cáo cùng mức chi tiết như kết quả dương"), không phải một thất bại không giải thích được.
3. Giai đoạn B (26/09 → 04/11) trở thành đường chính của đồ án, đúng nhánh dự phòng đã có sẵn trong `PLAN_Final.md` §10.
4. Nếu nhóm vẫn muốn thử một bài nộp dạng khác (ví dụ workshop, hoặc phần lõi rút gọn), cơ chế D8.2 chính là phần "vì sao" cần đưa vào mục Giới hạn — nó là một phát hiện toán học sạch (biên `L_k(1)` bị chặn bởi `Δ̂` chứ không phải `σ̂_Δ`), có thể tự nó là một đóng góp nhỏ về giới hạn của định nghĩa `m_k` dạng ngưỡng cố định.

**D8.5 — Xử lý pending validations theo hướng D8.4.**

- `exact_k_dagger_range_audit`: **ĐÓNG, kết luận âm.** Câu hỏi gate này đặt ra ("có dải `k†` biến thiên để kiểm định giả thuyết độ sâu không") nay đã có câu trả lời đầy đủ, có cơ chế giải thích (D7.1, D8.2), kiểm tra theo họ cấu trúc (D8.1) và theo kích thước (D8.2) — không phải "chưa biết", mà là "đã biết, và câu trả lời là không, ở quy mô/tham số hiện tại". Gỡ khỏi `pending_validations`.
- `adaptive_pilot_weighted_bootstrap_review`: **ĐÓNG.** Bằng chứng đã đủ dày qua D5/D6 và audit 24/08 (check 1: agreement tuyệt đối 0.0 với independent linear solve trên 61 pilot; check 2: tỷ lệ 0.35 so với independent-inflation, tương quan 0.921; valid bootstrap fraction = 1.0 trên mọi lần chạy đã ghi nhận). `docs/week1/WEEK1_AUDIT.md` đã ghi nhận điều này "arguably satisfied", giờ chính thức đóng. Gỡ khỏi `pending_validations`.
- `development_policy_quality_probe`, `coverage_threshold_group_review`: **giữ nguyên, hạ ưu tiên.** Đây là hai quyết định cần con người cụ thể duyệt (Linh review grid nhiệt độ; nhóm review ngưỡng coverage), không phải phép kiểm tự động hoá được. Theo hướng D8.4, nếu nhóm đồng ý dồn về đồ án, hai mục này không còn chặn đường nộp ICLR nữa nên không cần gấp — nhưng vẫn cần chốt trước khi dùng benchmark này làm số liệu chính thức cho đồ án.

**Điều KHÔNG làm.** Không đổi `ε`, `n_min`, `K_max`, ngưỡng K1a hay bất kỳ hằng số Phụ lục A nào để "thử xem nếu ngưỡng khác thì có qua không". Nếu nhóm sau này muốn thử ngưỡng `ε` khác cho Giai đoạn B (không còn ràng buộc bởi lịch ICLR), đó phải là một protocol deviation mới, có lý do độc lập với mong muốn kết quả, và báo cáo song song với con số D8 ở trên — không thay thế nó.

---

## D9 — 24/08/2026: chạy thật `pilot/aggregate.py`, sửa lỗi crash, và bằng chứng thứ ba độc lập cho hướng D8.4

**Phạm vi.** Thực hiện đúng gợi ý của GVHD ("các em nên ướm thử chạy được cơ bản trước mới quyết định") — chạy toàn bộ pipeline tổng hợp Tuần 2 thật lần đầu tiên trên 7 artifact thật đã có (trước đây `pilot/aggregate.py` chỉ được unit-test trên fixture giả). Không đổi hằng số Phụ lục A.

**D9.1 — Lỗi crash thật, đã sửa.** `compare_k_dagger` (`evaluation/aggregation/comparisons.py`) raise `ValueError` ngay khi gặp một cặp OEH đã `selected` nhưng không có exact reference — đúng trường hợp "oracle không xác định vì thiếu ít nhất hai bộ dữ liệu oracle hợp lệ" đã biết từ D7/D8 (23/39 cặp). Đây là trạng thái dự kiến, thường gặp, không phải lỗi toàn vẹn dữ liệu. Sửa: ghi nhận `comparison_status="oracle_undefined"` thay vì crash. Test cũ `test_missing_exact_reference_raises` đổi thành `test_missing_exact_reference_is_recorded_as_undefined_not_raised`, thêm `test_present_exact_reference_marks_compared`. Chạy lại xác nhận đúng 62 = 39 `compared` + 23 `oracle_undefined`, khớp chính xác con số đếm tay ở D8. Toàn bộ suite: 186/186 pass.

**D9.2 — Phát hiện chính: `N̂_req` chưa vượt qua baseline chỉ-cấu-trúc.** Trên các case `N_emp` hữu hạn (không censor), tương quan hạng (Spearman) giữa `n_req_hat` (= `N₀`, theo D7/D8) và `N_emp`:

| Thuật toán | n (có cả N_emp hữu hạn và n_req_hat) | Spearman `n_req_hat` vs `N_emp` | Spearman `S·A` (baseline chỉ-cấu-trúc) vs `N_emp` |
|---|---:|---:|---:|
| `cfqi_tabular` | 26/33 | 0.857 | 0.849 |
| `iql_tabular` | 11/14 | 0.900 | 0.900 |

Hai số gần như bằng nhau. Kiểm sâu hơn: tách phần thật sự phụ thuộc dữ liệu pilot ra khỏi `n_req_hat` (chia cho `S·A`, còn lại chính là "mức lưới pilot được chọn" — `{2,5,10}`) và tương quan phần đó với `N_emp` cũng chia cho `S·A` — tương quan **sụp xuống** (`cfqi_tabular`: Spearman 0.206, Pearson **-0.169**, đổi dấu). Đây đúng phép kiểm `PLAN_Final.md` §6.1 đã đăng ký trước cho baseline chỉ-cấu-trúc: *"Nếu nó cạnh tranh được thì phần độ sâu không mang thông tin dữ liệu, và phải nói thẳng."* Kết quả: nó cạnh tranh được. Thêm nữa, `n_req_hat` không có giá trị ở 21% case `N_emp` hữu hạn (baseline chỉ-cấu-trúc thì luôn có).

**D9.3 — Đây là bằng chứng độc lập thứ ba, cùng hướng D7/D8.** (1) Đường cong bằng chứng bão hoà (D7). (2) Hai họ được kỳ vọng có biên lớn nhất lại có biên nhỏ nhất (D8). (3) Số dự báo cuối cùng thống kê không phân biệt được với một baseline chỉ-cấu-trúc tầm thường. Ba phép kiểm độc lập, đo trên ba khía cạnh khác nhau của cùng pipeline, cùng chỉ về một kết luận — đây chính là loại bằng chứng khách quan GVHD nhắc tới ("cần... baseline để so sánh"), và câu trả lời hiện tại là khách quan âm tính.

**D9.4 — Không đổi khuyến nghị D8.4, chỉ củng cố thêm bằng chứng.** Không có thay đổi hằng số nào dựa trên D9.1–D9.3. Cập nhật `artifacts/week2/WEEK2_DECISION_GATE.md` và `artifacts/week2/aggregation_2026-08-24/README.md` với chi tiết đầy đủ.

**Việc còn để ngỏ, không sửa trong lần này (ngoài phạm vi sửa lỗi crash):** `pilot/aggregate.py::run()` gọi `build_decision_gate_input` chỉ với `tests_passing`/`no_leakage_violations`/`reproducible`, chưa nối `k1a_median_ratio`/`k5_interior_minimum_found`/`censoring_rate` từ chính bảng nó vừa tạo ra — nên `decision_gate_input.json` do chính script này ghi ra hiện đọc nhầm thành "NO_GO vì thiếu artifact" dù artifact đã có. Quyết định thật đang dùng bản tính tay đầy đủ ở `artifacts/week2/decision_gate_input.json` (D7.4). Nối dây việc này là việc tiếp theo, không phải bug ảnh hưởng tới D7–D9.

# Cohort 8, panel SPIBB và phát triển ứng viên (14/10/2026)

Tài liệu này hoàn tất các việc còn lại của kế hoạch bàn giao Colosseum ([`KE_HOACH_THUC_NGHIEM`](../../benchmarks/colosseum/KE_HOACH_THUC_NGHIEM_2026-10-14.md)) sau [biên bản Colosseum](COLOSSEUM_EXTERNAL_PANEL_2026-10-14.md).

| Việc | Trạng thái | Nguồn |
|---|---|---|
| A4 LCE, A5 leave-one-family-out, B1 OEH-S, B2 OEH-2T, B3 CB (phát triển trên 720 MDP đã mở) | Xong | [`cohort8_development_2026-10-14/`](../../artifacts/week8/cohort8_development_2026-10-14/) |
| B4 cohort 8 nội bộ, 168 MDP mới, niêm phong, gatekeeping | Xem mục 4 | [`oeh_eighth_cohort_2026-10-14/internal/`](../../artifacts/oeh_eighth_cohort_2026-10-14/) |
| B4 tầng Colosseum mới, 72 kernel, seed 2028 | Xem mục 4 | [`oeh_eighth_cohort_2026-10-14/colosseum/`](../../artifacts/oeh_eighth_cohort_2026-10-14/) |
| C1 panel SPIBB, MDP và behavior của tác giả khác | Xem mục 4 | [`oeh_spibb_panel_2026-10-14/`](../../artifacts/oeh_spibb_panel_2026-10-14/) |
| 2e TQ/SACo gốc trên Colosseum cũ, A4 LCE trên Colosseum cũ | Xem mục 5 | [`colosseum_posthoc_part2_2026-10-14/`](../../artifacts/week8/colosseum_posthoc_part2_2026-10-14/) |
| D2 regret ra quyết định khóa trước | Trong protocol cohort 8 | mục 3 |
| D3 mệnh đề nhất quán | Xong | [`D3_THEORY_NOTE_2026-10-14.md`](D3_THEORY_NOTE_2026-10-14.md) |
| D1 mở rộng quy mô, C2 tăng cỡ BRIDGE | Không làm | mục 7 |

## 1. Thứ tự thời gian (UTC, kiểm chứng bằng commit)

1. Kế hoạch phát triển `rules.json` được ghi trước khi fit OOF. Kết quả OOF và `selection.json` có trước protocol cohort 8.
2. Protocol cohort 8 khóa lúc 11:4x ngày 14/10 (SHA `9d60e262…`) và được commit (`8c526e5c`) **trước** mọi buffer dữ liệu và kernel Colosseum mới.
3. Protocol SPIBB khóa (SHA `d9d2d58f…`) và commit trước khi sinh MDP SPIBB.
4. Hai amendment trước outcome, không thay đổi khoa học (mục 6).
5. Mỗi panel: dataset seal → dự báo niêm phong (`preoutcome_seal.json`) → mới chạy learner (`outcome_run.json` ghi hash seal lúc bắt đầu).

## 2. Phát triển trên 720 MDP đã mở (OOF, panel primary)

Mỗi dòng là một OOF ngoài mẫu. `mdp5` dùng 5 fold băm theo MDP. `lofo` bỏ hẳn một họ primary (bottleneck, forward-layered, Garnet), fit trên phần còn lại, rồi dự báo họ bị bỏ. Nguồn: [`oof_summary.csv`](../../artifacts/week8/cohort8_development_2026-10-14/oof_summary.csv).

| Phương pháp | IBS mdp5 | LL mdp5 | IBS lofo | LL lofo |
|---|---:|---:|---:|---:|
| **OEH current** | **0,0498** | **0,5891** | **0,0962** | **0,9309** |
| B2 + legacy (`prior_oeh_matched`) | 0,0538 | 0,6204 | 0,1047 | 0,9783 |
| B2 + LCE | 0,0585 | 0,6442 | 0,1165 | 1,0807 |
| B2 + CB (concentrability) | 0,0590 | 0,6454 | 0,1210 | 1,1271 |
| B2 + ERI | 0,0602 | 0,6540 | 0,1224 | 1,0664 |
| B2 matched | 0,0611 | 0,6589 | 0,1185 | 1,1067 |
| XGBoost-CF/18 | 0,0622 | 1,2127 | 0,1320 | 1,6319 |
| CB riêng | 0,0946 | 0,8386 | 0,1343 | 1,2400 |
| LCE thô (không học) | 0,1150 | 1,2696 | 0,1150 | 1,2696 |

**A5, chuyển sang họ chưa thấy:**
- Lỗi của mọi phương pháp tăng gần gấp đôi khi họ bị bỏ khỏi meta-training. Ví dụ OEH IBS 0,050 → 0,096.
- OEH vẫn tốt nhất khi gộp. Nhưng ở forward-layered, B2 + legacy thắng OEH (IBS 0,1305 vs 0,1400; LL 1,228 vs 1,273).
- Lợi thế lớn nhất của OEH khi bỏ họ là ở Garnet (IBS 0,077 vs B2 0,130).

**A4/B3, đối thủ mới:**
- LCE thô (ngoại suy học thử trên prefix của pilot, đánh giá trong các MDP hậu nghiệm) yếu hơn nhiều so với mọi head đã fit.
- Đưa LCE hoặc CB vào B2 chỉ cải thiện B2 rất ít.

**B1, OEH-S:**
- Lưới khóa trước T ∈ {1; 1,25; 1,5; 2}, ε ∈ {0; 0,01; 0,02; 0,05}. Tiêu chí là LL OOF-lofo.
- Chọn T = 1,25, ε = 0,05: LL lofo 0,9309 → 0,8819, IBS 0,0962 → 0,0930.
- Đánh đổi: trong cùng phân phối (mdp5) OEH-S **tệ hơn** (LL 0,5891 → 0,6120; IBS 0,0498 → 0,0513).

**B2, OEH-2T:**
- P(E = 6) lấy từ OEH current. P(k | hữu hạn) lấy từ head 108 cột fit trên các dòng hữu hạn. Blend w = 0,5 với OEH current.
- Đây là lựa chọn duy nhất thỏa quy tắc "LL mdp5 thấp hơn và IBS không cao hơn": LL 0,5847 vs 0,5891, IBS 0,0496 vs 0,0498. Trên lofo: LL 0,9077, IBS 0,0949.
- Lợi ích nhỏ. Head có điều kiện dùng bộ legacy làm tệ hơn.

## 3. Thiết kế cohort 8 (protocol `9d60e262…`)

**Panel nội bộ:** 168 MDP stochastic mới.
- Mỗi template primary của cohort 7 cho hai kernel: 60 Garnet, 72 forward-layered, 36 bottleneck.
- Kernel khác mọi cohort đã mở và cohort 7. Không có control tất định.
- 5 behavior, pilot x10, 10 seed, BC/CFQI/IQL, tổng 151.200 job.

**Tầng Colosseum:** 72 kernel mới.
- Seed 2028, 18 kernel/họ, loại mọi kernel của hai lần chạy trước.
- Export bằng `colosseum_export_fast.py`. Bản này chỉ bỏ việc tính độ khó trong lúc export; luật chọn MDP giữ nguyên.

**Forecaster:** đúng bộ fit 720 MDP đã niêm phong của cohort 7. Mọi field set fit bằng `matched_forecast`.

**Gatekeeping** (max-t với SE gốc, 10.000 bootstrap theo MDP, seed 20261015). Áp dụng riêng cho từng panel:
1. **Bước 1:** OEH vs {B2 matched, XGBoost-CF/18} × {IBS, LL}. 4 phép so.
2. **Bước 2** (chỉ khi bước 1 GO): OEH vs {B2 + ERI, B2 + legacy, LCE thô, B2 + LCE, B2 + CB, CB riêng}. 12 phép so.
3. **Bước 3** (chỉ khi bước 2 GO): OEH-S và OEH-2T vs OEH current. Một ứng viên chỉ thay OEH current nếu cả hai metric của nó qua.

**Mô tả, không gate:**
- Regret ra quyết định: τ ∈ {0,5; 0,8; 0,9}, λ ∈ {50; 100}, tiêu đề là τ = 0,8, λ = 100; có các baseline "luôn x10 / x50 / bỏ qua".
- Factorial 8 arm và Shapley.
- AUC phát hiện `>x50`, độ chính xác ±1 mức, số nhóm 0 draw.

**Panel SPIBB (protocol `d9d2d58f…`):**
- 108 MDP Garnet của Laroche et al. (2019) sinh bằng code tác giả (`tmp/external_spibb` @ 833f32b): 50 trạng thái, 4 hành động, 4 trạng thái kế tiếp, đích là trạng thái "xa nhất".
- Baseline policy của tác giả với tỉ lệ hiệu năng ρ ∈ {0,1; …; 0,9}, 12 MDP mỗi ρ. Đây là **behavior duy nhất** của mỗi MDP.
- Cùng forecaster, comparator và gate. Bước 1 là phép thử chuyển giao C1.

## 4. Kết quả các panel niêm phong

**Thứ tự đã kiểm tra:**
- Mỗi panel có `preoutcome_seal.json` trước `outcome_run.json`, và seal được commit trước khi nhãn chạy xong: nội bộ `08dcfc04`, Colosseum `31a4b002`, SPIBB `9b09ed39`.
- `sealed_fit_check.json` của cả ba panel xác nhận trọng số ensemble và OOF log loss của OEH/B2/B2 + ERI/B2 + legacy **trùng tuyệt đối** (sai khác 0,0) với tuning niêm phong của cohort 7. Đây chính là forecaster đã niêm phong.

### 4.1 Verdict gatekeeping (max-t, SE gốc, 10.000 bootstrap)

| Panel | MDP / dòng | `>x50` | Bước 1 | Bước 2 | Bước 3 |
|---|---:|---:|---|---|---|
| **Nội bộ cohort 8** | 168 / 2.520 | 72,6% | **GO** (critical 2,19) | **GO**, 12/12 (2,45) | OEH-S thua; OEH-2T không khác. **OEH current giữ nguyên** |
| **Colosseum mới** (seed 2028) | 72 / 1.080 | 72,6% | **GO** (2,09) | NO-GO (2,46) | Không kiểm định (cổng đóng) |
| **SPIBB** | 108 / 324 | 69,1% | NO-GO (2,20) | Không kiểm định | Không kiểm định |

### 4.2 Các phép so chính (gain tương đối / cận dưới đồng thời của cải thiện tuyệt đối)

| Phép so | Nội bộ IBS | Nội bộ LL | Colosseum IBS | Colosseum LL | SPIBB IBS | SPIBB LL |
|---|---:|---:|---:|---:|---:|---:|
| vs B2 matched | +17,8% / **+0,0066** | +10,8% / **+0,0468** | +26,8% / **+0,0176** | +10,5% / **+0,0060** | +7,8% / −0,0005 | +5,2% / −0,0060 |
| vs XGBoost-CF/18 | +19,6% / **+0,0079** | +50,4% / **+0,3786** | +30,6% / **+0,0206** | +42,1% / **+0,2919** | +20,7% / **+0,0026** | +27,6% / −0,0336 |
| vs B2 + ERI | +17,0% / **+0,0057** | +10,0% / **+0,0385** | +24,8% / **+0,0108** | +7,7% / −0,0415 | (+15,3% / +0,0003) | (+7,9% / +0,0016) |
| vs B2 + legacy | +7,0% / **+0,0007** | +5,5% / **+0,0142** | +13,3% / −0,0062 | +0,3% / −0,1194 | (−1,7%) | (−4,4%) |
| vs B2 + LCE | +14,1% / **+0,0039** | +8,7% / **+0,0312** | +18,3% / **+0,0098** | +5,8% / −0,0115 | (+10,1%) | (+3,6%) |
| vs B2 + CB | +16,4% / **+0,0055** | +10,5% / **+0,0449** | +20,8% / **+0,0102** | +7,0% / −0,0160 | (+2,2%) | (+0,6%) |
| vs LCE thô / CB riêng | ≥ 30% / > 0 | ≥ 30% / > 0 | > 0 | > 0 | (> 0) | (> 0) |
| OEH-S vs OEH | −4,2% / −0,0027 | −5,4% / −0,0359 | (+2,2% / −0,0007) | (+3,6% / +0,0048) | (−12,3%) | (−12,5%) |
| OEH-2T vs OEH | +0,0% / −0,0003 | +0,4% / −0,0016 | (+0,5%) | (+3,2% / +0,0124) | (+1,3%) | (+1,6%) |

Giá trị trong ngoặc thuộc các bước không được kiểm định vì cổng đóng; chỉ mang tính mô tả. Nguồn: `result.json` của từng panel.

### 4.3 Mô tả

**Điểm gộp** (IBS / LL):

| Phương pháp | Nội bộ | Colosseum mới | SPIBB |
|---|---:|---:|---:|
| OEH current | 0,0466 / 0,548 | 0,1050 / 0,895 | 0,0192 / 0,366 |
| B2 matched | 0,0566 / 0,614 | 0,1436 / 1,000 | 0,0209 / 0,387 |
| B2 + legacy | 0,0501 / 0,580 | 0,1211 / 0,897 | **0,0189 / 0,351** |
| XGBoost-CF/108 (kiểm kiến trúc) | 0,0476 / 0,874 | 0,1083 / 1,213 | 0,0247 / 0,432 |
| OEH-S | 0,0485 / 0,577 | **0,1027 / 0,863** | 0,0216 / 0,412 |
| OEH-2T | **0,0466 / 0,546** | 0,1045 / 0,866 | 0,0190 / 0,360 |

**Theo learner và họ:**
- Nội bộ: OEH tốt hơn B2 ở cả ba learner và cả ba họ. Ở bottleneck, B2 + legacy có Brier tốt hơn OEH (0,0185 vs 0,0198).
- Colosseum mới: **RiverSwim lại là họ OEH thua** về LL (1,303 vs B2 1,238 vs B2 + legacy 0,997), như ở lần xác nhận đầu.
- SPIBB: BC và IQL bị cắt 100%; CFQI chỉ cắt 7% với nhãn x10–x50. Trên thực tế panel này đo khả năng ước lượng mức ngân sách của CFQI dưới dịch chuyển behavior; AUC phát hiện `>x50` ≈ 0,99 với mọi phương pháp.

**Phát hiện `>x50` và mức hữu hạn:**
- Nội bộ: AUC OEH 0,963, B2 0,949. Đúng ±1 mức (hữu hạn): OEH 82,8%, B2 79,7%, B2 + legacy 84,2%.
- Colosseum mới: AUC 0,916 vs 0,855. Đúng ±1 mức: 56,8% vs 52,4%, B2 + legacy 65,5%.
- Mẫu hình "OEH phát hiện tốt hơn, B2 + legacy đặt mức tốt hơn" lặp lại lần thứ hai trên kernel mới.

**Regret ra quyết định** (khóa trước; tiêu đề τ = 0,8, λ = 100):

| Panel | OEH | B2 | Luôn bỏ qua | OEH − B2 [95%] |
|---|---:|---:|---:|---:|
| Nội bộ | 13,82 | 16,52 | 22,15 | −2,69 [−3,90; −1,49] |
| Colosseum mới | 20,56 | 23,22 | 25,06 | −2,66 [−6,46; +1,18] |
| SPIBB | 9,07 | 9,78 | 19,51 | — |

- Trên cohort nội bộ, quy tắc dùng OEH giảm regret 16% so với B2 và 38% so với "luôn bỏ qua". Đơn vị là bội số của |S||A| chuyển trạng thái, gồm cả phạt thất bại.
- Trên SPIBB, B2 + legacy có regret thấp nhất (7,25).

**Shapley theo khối** (gộp, khoảng 95%):

| Panel | Metric | φ(P) mô phỏng hậu nghiệm | φ(H) horizon | φ(L) legacy |
|---|---|---:|---:|---:|
| Nội bộ | IBS | 0,0030 [0,0018; 0,0043] | 0,0038 [0,0017; 0,0058] | 0,0033 [0,0017; 0,0049] |
| Nội bộ | LL | 0,0193 [0,0109; 0,0276] | 0,0291 [0,0179; 0,0405] | 0,0178 [0,0092; 0,0266] |
| Colosseum mới | IBS | 0,0103 [−0,0026; 0,0229] | 0,0185 [0,0106; 0,0268] | 0,0098 [0,0037; 0,0163] |
| Colosseum mới | LL | 0,0319 [−0,0344; 0,0958] | 0,0251 [−0,0213; 0,0705] | 0,0483 [0,0186; 0,0797] |
| SPIBB | LL | −0,0014 [−0,0132; 0,0098] | −0,0149 [−0,0267; −0,0030] | 0,0365 [0,0162; 0,0557] |

**Diễn giải được phép:**
1. **Trong generator** (cohort 8 là kiểm định niêm phong thứ hai sau cohort 7): OEH vượt mọi đối thủ cùng thông tin, kể cả LCE, CB và B2 + legacy, và cả ba khối tín hiệu đều đóng góp. Mô phỏng hậu nghiệm có đóng góp riêng dương.
2. **Kernel của người khác** (Colosseum, hai lần chạy độc lập): OEH vượt B2 matched và XGBoost-CF/18, nhưng không vượt có ý nghĩa B2 + ERI, B2 + legacy, B2 + LCE/CB về log loss. Ở log loss, chỉ khối legacy có đóng góp riêng chắc chắn.
3. **Khi cả MDP lẫn behavior đến từ quy trình khác** (SPIBB): OEH không vượt B2 có ý nghĩa, và B2 + legacy tốt nhất. Đây là giới hạn chính của claim chuyển giao.
4. **OEH-S và OEH-2T** không thay OEH current. OEH-S cải thiện khi có dịch chuyển kiểu Colosseum nhưng làm tệ đi trong cùng phân phối và trên SPIBB.

## 5. Post hoc phần 2 trên Colosseum cũ (72 MDP, nhãn đã mở)

**Sinh lại dữ liệu:**
- Kernel được sinh lại từ sampler của tác giả bằng `colosseum_export_fast.py`. Tập ID trùng 120/120 ứng viên và 36/36 ứng viên thăm dò.
- Bundle dựng lại bằng `build_colosseum.py`: **360/360 hash pilot x10 trùng** với `pilot_sha256` trong các nhóm OEH-B đã niêm phong.
- Nhờ vậy tính được hai input còn thiếu: TQ/SACo bằng code gốc ml-jku/OfflineRL @ 8cc8eec, và curve prefix cho LCE.
- Dự báo mới được hash trước khi đọc nhãn. `all_forecasts` tái tạo đúng OEH/B2 niêm phong (≤ 4,6e-8).
- Nguồn: [`colosseum_posthoc_part2_2026-10-14/`](../../artifacts/week8/colosseum_posthoc_part2_2026-10-14/).

**Họ 22 phép so đầy đủ của amendment 1** (quy tắc nhóm, critical khoảng 2,6): tier 1 NO-GO, tier 2 NO-GO, toàn bộ NO-GO.
- Dòng mới quan trọng nhất: **B2 + TQ/SACo gốc (đặc quyền) tốt hơn OEH một chút**: IBS 0,1002 vs 0,1010, LL 0,8751 vs 0,8903; cận dưới −0,022 / −0,124.
- Chỉ dùng điểm đã công bố (`published_metrics_only`) thì OEH thắng IBS 12% (cận dưới −0,007) và LL 11% (cận dưới +0,001).
- So với cohort 7, nơi TQ/SACo là phép so chặt nhất nhưng vẫn GO, trên Colosseum đối thủ đặc quyền này **không bị vượt**.

**Phương pháp cohort 8 trên Colosseum cũ** (mô tả; Δ = phương pháp − OEH, khoảng 95% bootstrap theo MDP; nguồn [`cohort8_methods_vs_oeh.csv`](../../artifacts/week8/colosseum_posthoc_part2_2026-10-14/cohort8_methods_vs_oeh.csv)):

| Phương pháp | IBS | LL | Δ IBS [95%] | Δ LL [95%] |
|---|---:|---:|---:|---:|
| OEH current | 0,1010 | 0,8903 | — | — |
| **OEH-S** (T 1,25, ε 0,05) | **0,0979** | 0,8525 | −0,0031 [−0,0056; −0,0007] | −0,0378 [−0,0643; −0,0140] |
| OEH-2T (blend 0,5) | 0,1006 | 0,8615 | −0,0004 [−0,0010; +0,0003] | −0,0288 [−0,0474; −0,0129] |
| B2 + legacy | 0,1048 | **0,8431** | +0,0038 [−0,0134; +0,0206] | −0,0472 [−0,1498; +0,0466] |
| B2 + LCE | 0,1251 | 0,9531 | +0,0241 [+0,0144; +0,0344] | +0,0628 [+0,0069; +0,1187] |
| B2 + CB | 0,1276 | 0,9619 | +0,0266 [+0,0130; +0,0407] | +0,0716 [+0,0004; +0,1401] |
| CB riêng | 0,1358 | 1,1361 | +0,0348 | +0,2458 |
| LCE thô | 0,1732 | 1,3642 | +0,0722 | +0,4739 |

- OEH-S, được chọn trên dịch chuyển họ (lofo) chứ không trên Colosseum, cải thiện cả hai metric ở đây. Ở RiverSwim, LL giảm từ 1,257 xuống 1,155.
- Đây là kiểm tra post hoc đúng như kế hoạch B1 cho phép: không dùng Colosseum để chọn tham số. Bằng chứng xác nhận nằm ở bước 3 của cohort 8 (mục 4).

## 6. Amendment trước outcome

| File | Nội dung | Trạng thái lúc ghi |
|---|---|---|
| [`oeh_eighth_cohort…/preoutcome_code_amendment_1.json`](../../artifacts/oeh_eighth_cohort_2026-10-14/preoutcome_code_amendment_1.json) | Gọi `_repo_first()` sau `archive_imports()` để worker import đúng runner | Đang sinh dataset nội bộ; chưa có seal hay outcome |
| [`oeh_spibb_panel…/preoutcome_code_amendment_1.json`](../../artifacts/oeh_spibb_panel_2026-10-14/preoutcome_code_amendment_1.json) | Như trên, cho runner SPIBB | Lần chạy đầu lỗi trước khi ghi MDP nào |
| [`oeh_spibb_panel…/preoutcome_code_amendment_2.json`](../../artifacts/oeh_spibb_panel_2026-10-14/preoutcome_code_amendment_2.json) | Seed numpy 20261014000 + i vượt 2³²; đổi thành 2026101400 + i | Chưa sinh MDP nào |

## 7. Không làm, và lý do

- **D1 (posterior thưa, GPU, mở rộng sang 125 task BRIDGE bị loại):** đây là thay đổi phương pháp cần thiết kế, kiểm định và một cohort riêng; thời gian ước tính là vài tháng. Thiết kế gợi ý vẫn là bản trong kế hoạch bàn giao.
- **C2 (tăng cỡ BRIDGE):** phụ thuộc D1.
- **Chọn learner theo chi phí kỳ vọng:** đã có selector study riêng, nên không thêm.

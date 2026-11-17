# OEH v2 — quyết định, phát triển và kiểm định niêm phong (14/10/2026)

Thực hiện theo [kế hoạch OEH v2](../../benchmarks/colosseum/KE_HOACH_OEH_V2_2026-10-14.md) của gói bàn giao cập nhật.
- Mọi lựa chọn mô hình dưới đây dựa trên **dữ liệu đã mở**.
- Kiểm định cuối dùng panel mới, protocol khóa trước khi sinh dữ liệu.

## 1. Quyết định cho các câu hỏi ở mục 10 của kế hoạch

| # | Câu hỏi | Quyết định | Lý do |
|---|---|---|---|
| 1 | Panel ngoài đã mở dùng để fit hay chỉ để chọn? | **Chỉ để chọn.** Fit trên 720 MDP meta, cộng cohort M3 nếu M3 được chấp nhận | Giữ được ý nghĩa "chuyển giao" của panel kiểm định |
| 2 | Làm M2, M3 trong đợt này? | **Có, theo cổng:** M3 chạy ngay (máy làm được trong vài giờ); M2 chỉ làm tiếp nếu chẩn đoán bước 0 đạt ngưỡng khóa trước | M1/M4 không sửa được SPIBB (LOPO); M3 là ứng viên duy nhất cho dịch chuyển do behavior |
| 3 | Mở rộng lưới ngân sách (E1)? | **Có**, x100 và x200 trên mọi panel kiểm định, làm endpoint phụ | Đã kiểm thực nghiệm: thêm x100/x200 không đổi một byte nào của prefix x1–x50 (75/75 bản ghi), nên endpoint chính không bị ảnh hưởng; chi phí learner chấp nhận được |
| 4 | M5 làm song song hay để sau? | **Để sau** | Không cần cho bài; là việc cho hội nghị lớn |
| 5 | Biên không kém trong generator 2% IBS? | **2% cho cả IBS và log loss**, kiểm bằng cận dưới đồng thời của (1,02 · OEH current − v2) | Một biên cho cả hai metric; dùng cận dưới chứ không dùng ước lượng điểm |
| 6 | Họ benchmark mới cho panel thứ 4? | **Forest management của MDPToolbox** (Chadès và cộng sự, 2014; Possingham & Tuck), 48 MDP với (S, p, r1, r2) lấy theo seed | Do người khác thiết kế, ngẫu nhiên, khác hẳn Colosseum, SPIBB và generator của nhóm; vừa kích thước |

## 2. Phát triển rẻ: M6, M1, M4

Nguồn: [`oeh_v2_development_2026-10-14/heads/`](../../artifacts/week8/oeh_v2_development_2026-10-14/heads/). `rules.json` được ghi trước khi fit.

**Thiết lập:**
- Fit trên 720 MDP meta. Chọn trên 5 panel đã mở: cohort 7, 8 (trong generator); Colosseum 2027, 2028, SPIBB (ngoài).
- Phép fit lại tái tạo đúng OEH niêm phong trên cả 5 panel (≤ 4,6e-8).

**M6 (đơn giản hóa head): không chấp nhận.**
- Multiclass đơn kém ensemble +1,4% IBS và +0,8% LL trong generator, vượt biên 1%.
- AFT riêng: +12% IBS. Ngưỡng riêng: +41% LL. XGBoost-CF/108: +59% LL.
- Kết luận: head ensemble có giá trị về log loss. Phát hiện "XGBoost-CF/108 ngang về Brier" không chuyển thành ngang về log loss.

**M1 + M4 được chọn**, theo quy tắc khóa trước gồm điểm ngoài, ràng buộc trong và ưu tiên phương án đơn giản:

```
P(E=6)    = p6 của OEH current
q(k|E<6)  = (1-λ(d))·head hữu hạn OEH(108) + λ(d)·phần hữu hạn của B2+legacy,  λ(d) = min(1, d/3)
OEH v2    = (1-ε(d))·[(1-P6)·q, P6] + ε(d)·π_learner,                       ε(d) = 0,02·min(1, d/3)
```

Ở đây d là khoảng cách 10-NN chuẩn hóa tới meta-training, tính trong 18 thống kê B2. Trung vị của d là khoảng 1,1 trong generator, 2,7–3,1 trên Colosseum, 1,8 trên SPIBB.

| Panel | OEH current IBS / LL | OEH v2 IBS / LL | B2 + legacy IBS / LL |
|---|---:|---:|---:|
| Cohort 7 | 0,0460 / 0,563 | 0,0462 / 0,573 | 0,0509 / 0,608 |
| Cohort 8 | 0,0466 / 0,548 | 0,0466 / 0,552 | 0,0501 / 0,580 |
| Colosseum 2027 | 0,1010 / 0,890 | **0,0945 / 0,791** | 0,1048 / 0,843 |
| Colosseum 2028 | 0,1050 / 0,895 | **0,0977 / 0,808** | 0,1211 / 0,897 |
| SPIBB | 0,0192 / 0,366 | **0,0186** / 0,360 | 0,0189 / **0,351** |

**Leave-one-external-panel-out** (chọn trên hai panel, chấm trên panel thứ ba):

| Held-out | IBS / LL so với OEH current |
|---|---:|
| Colosseum 2027 | −6,0% / −11,5% |
| Colosseum 2028 | −6,5% / −9,1% |
| SPIBB | −1,9% / **+1,7%** |

**Đọc kết quả:** M1/M4 sửa được lỗi khi đổi kernel, và việc chọn tham số chuyển được giữa hai panel Colosseum. Lỗi khi đổi behavior (SPIBB, log loss) thì vẫn còn, nên cần M3.

## 3. M2 (prior) và M3 (behavior)

Quy tắc chấp nhận được khóa trước ở [`rules_m2_m3.json`](../../artifacts/week8/oeh_v2_development_2026-10-14/rules_m2_m3.json).

### 3.1 M2, bước 0: chẩn đoán prior → **dừng M2**

**Cách làm:** chạy lại OEH-B với `reward_prior = exchangeable`, giữ nguyên mọi thứ khác, kể cả seed của từng nhóm đã niêm phong. Chạy trên 214 nhóm cohort 8, 108 nhóm SPIBB và 80 nhóm Colosseum 2027. Phần còn lại dừng sớm, vì quy tắc đã quyết định được với dữ liệu này. Nguồn: [`prior_diagnosis/`](../../artifacts/week8/oeh_v2_development_2026-10-14/prior_diagnosis/).

| Panel | LL thô `unseen_zero` → `exchangeable` | Thiên lệch xác suất thành công (dự báo − thực tế) |
|---|---:|---:|
| Cohort 8 (trong generator) | 0,899 → 0,977 (**+8,8%**) | −0,155 → −0,173 |
| Colosseum 2027 | 1,038 → 1,074 (+3,4%) | −0,082 → −0,107 |
| SPIBB | 0,505 → 0,505 (0%) | −0,027 → −0,027 |

- **Quy tắc** (≥5% tốt hơn trên ≥2 panel ngoài, trong generator không kém quá 5%): **không đạt**.
- RiverSwim là họ duy nhất cải thiện rõ: 1,125 → 1,034. Nghĩa là prior giải thích **một phần** lỗi ở RiverSwim, nhưng không giải thích lỗi chung ngoài generator.
- SPIBB không đổi, vì reward ở đó luôn được quan sát.
- **Phát hiện phụ:** luật OEH-B thô đánh giá thấp xác suất thành công một cách có hệ thống, ở mọi panel và dưới cả hai prior. Head đã fit đang bù phần lệch này.

### 3.2 M3: cohort meta-training với behavior mới → **không chấp nhận**

**Cohort:** 168 MDP mới của generator (2 bản sao × 84 template) với ba behavior chưa có trong meta-training:
- `ratio_baseline`: tỉ lệ ρ xoay vòng giữa π* và chính sách đều;
- `epsilon_star`: ε-greedy quanh π*;
- `random_mixture`: hỗn hợp ngẫu nhiên.

**Quy mô:** 1512 dòng, 90 720 job learner. Protocol khóa trước khi sinh dữ liệu: [`oeh_v2_behavior_cohort_2026-10-14/`](../../artifacts/oeh_v2_behavior_cohort_2026-10-14/).

**Nhãn:** 71% là ">x50". BC bị kiểm duyệt 503/504 dòng, IQL 474/504; chỉ CFQI trải đều trên các ngân sách. Ba behavior này khó hơn hẳn năm behavior đã đăng ký.

**Cách làm:** fit lại cả 11 head trên 720 MDP meta cộng cohort M3, rồi chạy lại đúng lưới và quy tắc chọn của mục 2 ([`heads_m3/`](../../artifacts/week8/oeh_v2_development_2026-10-14/heads_m3/)).
- Quy tắc M6 lúc này chọn base **multiclass**: kém ensemble đã fit lại +0,9% IBS và +0,5% LL, vẫn trong biên 1%.
- Lưới chọn: phần hữu hạn của base, phần hữu hạn của legacy, λ_max = 0,75, ε_max = 0,1, d0 = 4.

So với OEH current đã niêm phong (tham chiếu chung của mọi quy tắc):

| | Điểm ngoài | SPIBB | Cohort 7 / 8 (IBS) | Colosseum 2027 / 2028 (IBS / LL) | SPIBB (IBS / LL) |
|---|---:|---:|---:|---:|---:|
| OEH v2, không M3 | **−6,55%** | **−2,49%** | +0,5% / +0,1% | −6,5/−11,2% · −6,9/−9,8% | −3,1% / −1,8% |
| OEH v2 chọn lại với M3 | −3,97% | +1,17% | −1,5% / +1,3% | −2,8/−8,0% · −5,9/−9,4% | +1,0% / +1,3% |
| OEH current fit lại với M3 | +0,78% | +2,18% | −2,5% / +0,1% | +0,2/+0,7% · −0,8/+0,2% | +3,3% / +1,1% |

- **Quy tắc** (điểm ngoài thấp hơn, SPIBB thấp hơn, trong generator không kém quá 2%): hai điều kiện đầu **không đạt**.
- **Kết luận:** M3 không được chấp nhận. `oeh_v2_spec.json` giữ tham số của mục 2 và huấn luyện trên 720 MDP meta.
- **Phát hiện:** thêm behavior mới vào generator giúp một chút **trong** generator (cohort 7: −2,5% IBS), nhưng làm SPIBB **xấu đi** (+3,3% IBS).
  - Nghĩa là lỗi SPIBB không đến từ việc meta-training thiếu đa dạng behavior.
  - Nhiều khả năng nó đến từ cấu trúc MDP Garnet của SPIBB và cách behavior baseline phủ không gian ở đó. Đây là câu hỏi mở cho M5.
- **LOPO với M3** (để đối chiếu): SPIBB −1,3% IBS / +1,8% LL, tức vẫn không sửa được log loss trên SPIBB.

### 3.3 Đặc tả OEH v2 đã khóa

[`oeh_v2_spec.json`](../../artifacts/week8/oeh_v2_development_2026-10-14/oeh_v2_spec.json) được sinh **tất định** bằng `develop_oeh_v2_heads_20261014 spec`, từ `heads/selection.json`, `heads_m3/selection.json` và `rules_m2_m3.json`, không có lựa chọn thủ công.

- **Tham số:** head hữu hạn riêng của OEH (108 trường), phần hữu hạn của B2 + legacy, λ_max = 1, ε_max = 0,02, d0 = 3.
- **Huấn luyện:** 720 MDP meta.
- File ghi hash của mọi nguồn (quy tắc, lưới, định nghĩa `oeh_v2_forecast.py`, script).
- Test `test_frozen_spec_follows_the_selection_and_the_m3_rule` kiểm lại tính nhất quán.

## 4. Kiểm định niêm phong

**Protocol:** [`oeh_v2_sealed_test_2026-10-14/protocol.json`](../../artifacts/oeh_v2_sealed_test_2026-10-14/protocol.json), sha256 `17019f87…`.
- Khóa và push lên GitHub (commit `6340a490`) **trước khi** có bất kỳ kernel, buffer hay nhãn nào của bốn panel.
- Protocol chứa toàn văn spec OEH v2 và hash của spec.

| Panel | MDP | Behavior | Nguồn |
|---|---:|---:|---|
| Internal | 168 | 5 | Kernel mới của generator, khác mọi kernel trước (kể cả M3) |
| Colosseum 2029 | 72 | 5 | Sampler của tác giả Colosseum, seed 2029, loại mọi kernel của các lần 2026–2028 |
| SPIBB-2 | 108 | 1 (baseline của tác giả) | Code tác giả Laroche và cộng sự (2019), numpy seed mới |
| Forest | 48 | 5 | `mdptoolbox.example.forest` (Chadès và cộng sự, 2014), 80 ứng viên rút theo seed, lấy 48 ứng viên đầu qua bộ lọc |

**Cổng tuần tự** (max-t một phía, SE gốc cố định, 10 000 lần bootstrap; mỗi tập MDP dùng chung một ma trận chỉ số lấy mẫu):
1. OEH v2 tốt hơn B2 + legacy trên panel ngoài gộp (Colosseum 2029, SPIBB-2, forest; mỗi MDP trọng số bằng nhau), cả IBS và LL.
2. Chỉ xét nếu bước 1 GO. Trong generator, OEH v2 không kém OEH current quá biên 2%, tức cận dưới của 1,02·OEH current − v2 dương. Trên panel ngoài gộp, OEH v2 tốt hơn OEH current.
3. Chỉ xét nếu bước 2 GO. OEH v2 tốt hơn B2, XGBoost-CF/18, LCE, B2 + LCE, CB và B2 + CB, cả trong generator và trên panel ngoài gộp.

**Quy tắc chấp nhận:** OEH v2 được chấp nhận nếu bước 1 và bước 2 đều GO.

**Mô tả thêm** (không phải cổng):
- Theo panel và theo learner; phân rã Murphy; Shapley P/H/L.
- Regret của quyết định (τ ∈ {0,5; 0,8; 0,9}, λ ∈ {50; 100}).
- Endpoint mở rộng x100/x200; chi phí tính toán.
- `v2_no_direct_oehb`, cùng `b2_shift` và `legacy_shift` (chỉ co rút ε(d), để tách phần lợi do co rút khỏi phần lợi do trường OEH).

**Khai báo trước khi khóa** (ghi trong protocol):
- Đã chạy thử toàn bộ đường đi trên 3 MDP forest phát triển, rút từ một bộ ứng viên khác (seed 1), và đã thấy 45 nhãn của chúng.
- Một lần rút 80 ứng viên với seed 1 qua bộ lọc 80/80.

**Amendment trước outcome:** một amendment ([`preoutcome_amendment_1.json`](../../artifacts/oeh_v2_sealed_test_2026-10-14/preoutcome_amendment_1.json)).
- Reward của forest lên tới 16, vượt cận công khai [0, 1] của pilot. Sửa bằng cách chia reward cho max.
- Dữ liệu forest chưa chuẩn hóa bị xóa rồi sinh lại. Lúc đó chưa có pilot feature, dự báo hay nhãn nào.

**Niêm phong:** seal của từng panel được commit và push trước learner đầu tiên: forest `b07572ed`, SPIBB-2 `f41018d2`, Colosseum 2029 `53bf1855`, internal `c58074bb`.
- Ở cả bốn panel, trọng số ensemble khớp đúng forecaster niêm phong của cohort 7 (chênh 0).
- Có 108/108 MDP SPIBB-2 qua bộ lọc; Colosseum 2029 lấy đủ 18 kernel mỗi họ.
- Tổng cộng 371 520 job learner.

### 4.1 Cổng: **bước 1, 2, 3 đều GO → OEH v2 được chấp nhận**

| Bước | Tương phản | IBS: cải thiện (cận dưới) | LL: cải thiện (cận dưới) | Critical |
|---|---|---:|---:|---:|
| 1 | v2 vs B2 + legacy, panel ngoài gộp (228 MDP) | +0,0171 (+0,0107) | +0,0597 (+0,0344) | 1,77 |
| 2 | Không kém OEH current trong generator (biên 2%) | +0,0009 (+0,0003) | +0,0115 (+0,0045) | 2,18 |
| 2 | v2 vs OEH current, panel ngoài gộp | +0,0047 (+0,0028) | +0,0662 (+0,0398) | (chung) |
| 3 | v2 vs B2, XGBoost-CF/18, LCE, B2 + LCE, CB, B2 + CB; trong và ngoài | 24/24 cận dưới dương | | 2,79 |

Nguồn: [`result.json`](../../artifacts/oeh_v2_sealed_test_2026-10-14/result.json), [`report.md`](../../artifacts/oeh_v2_sealed_test_2026-10-14/report.md).

### 4.2 Theo panel (IBS / LL, trung bình theo MDP)

| Phương pháp | Internal (168) | Colosseum 2029 (72) | SPIBB-2 (108) | Forest (48) | Ngoài gộp (228) |
|---|---:|---:|---:|---:|---:|
| **OEH v2** | 0,0476 / 0,562 | **0,1080 / 0,833** | 0,0189 / 0,358 | **0,1097 / 0,790** | **0,0662 / 0,599** |
| OEH current | **0,0475** / 0,563 | 0,1153 / 0,951 | **0,0188 / 0,352** | 0,1216 / 0,941 | 0,0709 / 0,665 |
| B2 + legacy | 0,0497 / 0,579 | 0,1154 / 0,873 | 0,0190 / 0,354 | 0,1794 / 1,022 | 0,0832 / 0,659 |
| B2 | 0,0565 / 0,613 | 0,1407 / 0,989 | 0,0252 / 0,420 | 0,2370 / 1,285 | 0,1063 / 0,782 |
| XGBoost-CF/18 | 0,0596 / 1,122 | 0,1436 / 1,256 | 0,0277 / 0,457 | 0,2535 / 1,447 | 0,1118 / 0,918 |
| B2 + LCE | 0,0544 / 0,597 | 0,1396 / 0,995 | 0,0249 / 0,400 | 0,1299 / 0,865 | 0,0832 / 0,686 |
| B2 + legacy, chỉ co rút ε(d) | 0,0498 / 0,580 | 0,1145 / 0,866 | 0,0190 / 0,357 | 0,1790 / 1,018 | 0,0829 / 0,657 |

- **So với B2:** v2 tốt hơn trên **từng** panel ở cả hai chỉ số, với khoảng t 95% theo panel không chứa 0. Ví dụ Colosseum 2029: IBS +23%, LL +16%.
- **Chỉ co rút** (`legacy_shift`, `b2_shift`) gần như không đổi gì. Phần lợi của v2 đến từ việc ghép P(>x50) của OEH với mức hữu hạn của legacy, không đến từ ε(d).

### 4.3 Độ vững (mô tả, không phải cổng; [`robustness.csv`](../../artifacts/oeh_v2_sealed_test_2026-10-14/robustness.csv))

**v2 vs B2 + legacy (bước 1) phụ thuộc vào forest:**

| | IBS [KTC 95%] | LL [KTC 95%] |
|---|---:|---:|
| Colosseum 2029 | +0,0074 [−0,0055; 0,0204] | +0,040 [−0,013; 0,094] |
| SPIBB-2 | +0,0001 [−0,0007; 0,0009] | −0,004 [−0,013; 0,006] |
| Forest | +0,070 [0,048; 0,092] | +0,232 [0,143; 0,321] |
| Ngoài gộp, bỏ forest | +0,0030 [−0,0022; 0,0082] | +0,014 [−0,009; 0,036] |
| Bootstrap phân tầng theo panel | +0,0171 [0,0108; 0,0231] | +0,060 [0,034; 0,085] |

**v2 vs OEH current (bước 2) vững với mọi cách bỏ panel:**
- Bỏ Colosseum: IBS +0,0035 [0,0018; 0,0053], LL +0,042 [0,020; 0,065].
- Bỏ SPIBB-2: IBS +0,0091 [0,0062; 0,0121], LL +0,131 [0,091; 0,172].
- Bỏ forest: IBS +0,0028 [0,0012; 0,0045], LL +0,044 [0,019; 0,069].
- Riêng Colosseum 2029: LL +0,118 [0,063; 0,174]. Riêng forest: +0,150 [0,092; 0,208]. SPIBB-2: −0,006 [−0,018; 0,006].
- Trong generator, v2 và OEH gần như trùng nhau (IBS −0,0001, LL +0,0002).

**Sàn log loss 0,01** (thay cho 1e-12) không đổi kết luận nào.

**Theo learner** (tiêu chí 3 của kế hoạch): v2 kém có ý nghĩa chỉ ở SPIBB-2 cho IQL (LL −0,013 vs B2 + legacy; −0,003 vs OEH) và BC (−0,001 vs OEH).
- Ở các ô này, learner luôn ">x50", và ε(d) lấy bớt xác suất của lớp đó. Mức thiệt bị chặn bởi −log(1 − ε) ≤ 0,0203 nat (Mệnh đề 2).

### 4.4 Cơ chế, ablation, quyết định, endpoint mở rộng, chi phí

**Murphy** (trung bình qua 6 ngưỡng):
- Colosseum 2029: reliability của v2 là 0,0082, so với 0,0123 (OEH) và 0,0110 (legacy); resolution 0,0575, cao nhất.
- Forest: resolution của OEH/v2 là 0,146, so với 0,069 của legacy. v2 sửa reliability của OEH: 0,021 so với 0,032.

**Shapley P/H/L** (cải thiện loss so với B2, ngoài gộp):
- P (mô phỏng hậu nghiệm): +0,0165 IBS, +0,078 LL.
- H (horizon): +0,0078 IBS, −0,029 LL.
- L (legacy): +0,011 IBS, +0,067 LL.
- So với cohort 8 và Colosseum 2028 (chỉ L có tín hiệu LL ngoài generator), giờ **khối P có đóng góp dương rõ ngoài generator**, chủ yếu nhờ forest (P: +0,31 LL).
- Khối H vẫn hại LL ngoài generator (Colosseum −0,033; forest −0,087).

**v2 không có OEH-B trực tiếp** (`v2_no_direct_oehb`): ngoài gộp 0,0708 / 0,621, kém v2 đầy đủ (0,0662 / 0,599), nhưng vẫn hơn B2 + legacy.

**Regret** (τ = 0,8, λ = 100; đơn vị |S||A|):

| Phương pháp | Internal | Colosseum 2029 | SPIBB-2 | Forest |
|---|---:|---:|---:|---:|
| v2 | 13,12 | 21,06 | 7,99 | 20,03 |
| OEH | 13,22 | 20,46 | 7,99 | 19,45 |
| B2 + legacy | 15,07 | 22,88 | **7,04** | 28,15 |
| B2 | 16,47 | 24,86 | 9,07 | 35,81 |
| Luôn bỏ qua | 22,42 | 24,59 | 19,72 | 39,63 |

v2 ngang OEH về quyết định. Cải thiện xác suất của v2 nằm chủ yếu ở mức hữu hạn, nơi quy tắc τ = 0,8 ít nhạy.

**Endpoint mở rộng x100/x200:** trong các dòng có nhãn chính ">x50", tỉ lệ vẫn không đạt ở x200 là 90,6% (internal), 95,2% (Colosseum 2029), 96,0% (SPIBB-2), 98,1% (forest). Nghĩa là ">x50" gần như luôn là "không đạt trong x200".

**Chi phí** (wall-clock, 10 worker):
- Đầu vào pilot-only: internal 2 612 s, trong đó OEH-B 1 761 s; Colosseum 1 788 s; SPIBB-2 906 s; forest 471 s.
- Learner: 1 089, 661, 331 và 201 s.
- Phần co rút và kết hợp của v2 tốn không đáng kể.

### 4.5 Đọc kết quả

1. **OEH v2 thay OEH current làm forecaster chính.** Cổng niêm phong đạt cả ba bước. Trong generator, v2 trùng OEH (không kém). Ngoài generator, v2 tốt hơn OEH ở cả IBS và LL, vững với việc bỏ từng panel.
2. **Claim so với B2 + legacy phải nói rõ phạm vi.** Cổng gộp đạt, nhưng phần thắng chủ yếu từ forest. Trên Colosseum 2029, chênh lệch dương mà chưa có ý nghĩa riêng; trên SPIBB-2 thì ngang.
   - Câu được viết: *"OEH v2 beats B2 + legacy on the pooled external panels (sealed, both metrics); the margin is carried by the forest-management panel and is not individually significant on Colosseum or SPIBB."*
3. **Lỗi khi đổi behavior (SPIBB) vẫn còn.** Không phương pháp nào thắng rõ trên SPIBB-2. Điều này khớp với việc M3 không được chấp nhận.
4. **Mô phỏng hậu nghiệm có giá trị ngoài generator khi cấu trúc MDP phù hợp.** Forest là chuỗi tuổi rừng với reset ngẫu nhiên; ở đó khối P cho resolution gấp đôi legacy. Trên Colosseum, phần lợi chủ yếu đến từ việc ghép head.

## 5. Sau kiểm định 4 panel (15/10): điểm vào duy nhất, hậu kiểm, panel thứ năm

### 5.1 Điểm vào duy nhất của phương pháp cuối

`python -m scripts.analysis.oeh_pipeline --dataset <bundle> --out <dir> [--kernels <dir> --kind colosseum|spibb]` chạy trọn đường dự báo:
- pilot input (B2, OEH-B, OEH-H, curve, chẩn đoán D);
- ba head (head đầy đủ, head đầy đủ fit trên dòng hữu hạn, head vững B2 + D);
- cổng dịch chuyển.

Điểm vào này tái tạo dự báo niêm phong của cả 4 panel với sai số tối đa 9,4e-16 ([`reproduction.json`](../../artifacts/week8/oeh_pipeline_reproduction_2026-10-15/reproduction.json)).

### 5.2 Hậu kiểm trên 4 panel đã mở nhãn (mô tả)

Nguồn: [`oeh_final_posthoc_2026-10-15/`](../../artifacts/week8/oeh_final_posthoc_2026-10-15/).

**Gain theo khoảng cách d:**
- Gần dữ liệu huấn luyện (d ≤ 1,5), ba dự báo gain như nhau so với B2.
- Ở d > 3, cổng tốt hơn cả hai head. Ví dụ ở d > 4, gain LL: OEH 0,32, head vững 0,23, head đầy đủ 0,11.
- Cổng **ghép** P(>x50) của head đầy đủ với phân phối hữu hạn của head vững, chứ không chỉ chọn một trong hai.

**Ablation của phương pháp cuối** (pooled external, so với OEH):

| Biến thể | IBS kém thêm [KTC 95%] | LL kém thêm [KTC 95%] |
|---|---:|---:|
| Bỏ OEH-H | +0,0028 [0,0012; 0,0044] | +0,0085 [0,0009; 0,0161] |
| Bỏ OEH-B trực tiếp | +0,0046 [0,0026; 0,0066] | +0,022 [0,011; 0,033] |

Trong generator, cả hai biến thể đều nằm trong nhiễu. Kết luận: không bỏ được khối nào.

**Baseline chung mới:** k-NN analog và logistic đa lớp trên B2 đều kém B2 matched. So với OEH ngoài generator, chúng kém 0,21 và 0,33 nat.

### 5.3 Panel thứ năm: **NO-GO** so với head vững

**Thiết kế:** protocol [`oeh_fifth_panel_2026-10-15/protocol.json`](../../artifacts/oeh_fifth_panel_2026-10-15/protocol.json), sha `ba8f1bab…`.
- Khóa và push sau phân tích 4 panel, trước mọi dữ liệu.
- Seal: gambler `aea5269d`, FrozenLake `421dad82`.
- Hai nguồn mới:
  - **FrozenLake** (Gymnasium 1.0.0, bộ sinh bản đồ của chính Gymnasium, slippery): 36 MDP, lấy từ 81 ứng viên;
  - **Gambler's problem** (Sutton & Barto, Example 4.3): 36 MDP.
- Cả hai chuyển sang setting continuing: trạng thái kết thúc restart từ phân phối đầu, giữ reward lúc vào.
- Chỉ **một** họ phép so được đăng ký: OEH vs head vững, IBS và LL, max-t trên 72 MDP.

**Kết quả cổng:** IBS −0,0017 (cận dưới −0,0067); LL +0,0117 (cận dưới −0,0196). **NO-GO.**

| Phương pháp | FrozenLake | Gambler | Gộp |
|---|---:|---:|---:|
| OEH | 0,0040 / 0,082 | 0,0931 / 0,752 | 0,0486 / **0,417** |
| Head đầy đủ | 0,0038 / 0,079 | 0,0951 / 0,795 | 0,0495 / 0,437 |
| Head vững (B2 + D) | 0,0105 / 0,154 | 0,0831 / **0,703** | **0,0468** / 0,428 |
| B2 | 0,0122 / 0,153 | 0,0958 / 0,797 | 0,0540 / 0,475 |
| LCE | **0,0007 / 0,025** | 0,1132 / 1,174 | 0,0569 / 0,599 |
| k-NN analog | 0,0176 / 0,243 | **0,0795** / 0,728 | 0,0486 / 0,485 |

- **FrozenLake:** 100% nhãn là ">x50". OEH nhận ra tốt hơn hẳn head vững, nhưng LCE (learning curve của chính learner trên pilot) tốt nhất.
- **Gambler:** head vững tốt hơn OEH (không có ý nghĩa), chủ yếu ở BC.
- **Tái lập được:** OEH tốt hơn head đầy đủ trên panel gộp (IBS +0,0009 [0,0002; 0,0015], LL +0,020 [0,003; 0,038]). OEH tốt hơn B2 về LL (+0,058 [0,031; 0,086]).

### 5.4 Claim cuối

1. **Vững:**
   - OEH > B2 trên mọi panel của kiểm định 4 panel, và về LL gộp trên panel thứ năm.
   - OEH > head đầy đủ trên các panel ngoài gộp của cả hai kiểm định.
2. **Phụ thuộc panel:** OEH > head vững.
   - Đạt ở cổng 4 panel nhưng nhờ forest; NO-GO ở panel thứ năm.
   - Mạnh trên forest và FrozenLake; không có trên SPIBB-2 và gambler.
3. **Không phát hiện được dịch chuyển do behavior** (SPIBB).

## 6. Quy ước khi viết

| Trong bài và luận văn | Trong code và artifact |
|---|---|
| OEH (phương pháp cuối) | OEH v2, `v2`, `oeh_v2_spec.json` |
| OEH-F, head đầy đủ | OEH current, `oeh_final` |
| Head vững (B2 + D) | B2 + legacy, `prior_oeh_matched` |
| Chẩn đoán pilot D | legacy, OEH3/LCOEH4 (G7, G8) |

- Các kiểm định cũ (cohort 1–8, Colosseum 2027/2028, SPIBB) là **quá trình phát triển** và phải được khai báo.
- Chỉ hai protocol cuối đánh giá phương pháp cuối: kiểm định 4 panel và panel thứ năm.

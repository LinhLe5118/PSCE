# Thử nghiệm OEH trên benchmark ngoài Colosseum: kết quả, điểm yếu và kế hoạch khắc phục

*Người chạy: thành viên nhóm, chạy trên máy phụ. Thời gian: 13–14/10/2026. Repo `Devoted` không bị sửa: mọi việc chạy trên một bản copy (`workspace/`) đồng bộ tới commit `d7134ad`.*

> **Kế hoạch hiện hành: [`KE_HOACH_OEH_V2.md`](KE_HOACH_OEH_V2.md)** (phát triển OEH v2, viết sau khi đọc `REANALYSIS_ALL_PANELS_2026-10-14.md`).

> **Kết quả OEH v2 (15/10):** kiểm định niêm phong trên 4 panel mới, bước 1–3 đều GO, OEH v2 được chấp nhận. Xem [`docs/week8/OEH_V2_DEVELOPMENT_2026-10-14.md`](../../docs/week8/OEH_V2_DEVELOPMENT_2026-10-14.md) mục 4 và [`artifacts/oeh_v2_sealed_test_2026-10-14/`](../../artifacts/oeh_v2_sealed_test_2026-10-14/).

> Kế hoạch cũ [`KE_HOACH_THUC_NGHIEM.md`](KE_HOACH_THUC_NGHIEM.md) giữ lại làm tham khảo; phần lớn đã được nhóm trưởng thực hiện trong reanalysis.

Tài liệu gồm 3 phần:

1. **Kết quả**: OEH current trên 72 MDP Colosseum mới, có khóa protocol.
2. **Điểm yếu của dự án** mà thử nghiệm này và việc đọc lại bài báo làm lộ ra, kèm cách khắc phục đề xuất.
3. **Nội dung gói và cách chạy lại** trên máy chính.

---

## 1. Tóm tắt kết quả

### 1.1 Thiết lập

- **MDP:** 72 MDP từ Colosseum (Conserva & Rauber, NeurIPS 2022).
  - Sinh bằng sampler gốc của tác giả, `sample_mdp_parameters`, seed 2027, repo `MichelangeloConserva/Colosseum` @ `981b344`.
  - 4 họ: RiverSwim, DeepSea, FrozenLake, SimpleGrid, mỗi họ 18 MDP.
  - Chỉ lấy các MDP có |S||A| ≤ 256 và qua bộ lọc dynamic-range của dự án.
  - Đã loại mọi kernel trùng với lần thăm dò trước.
- **Giữ nguyên của dự án:** generator dữ liệu, 5 behavior policy, pilot x10, 10 seed, BC/CFQI/IQL, quy tắc gán nhãn 7 lớp.
- **OEH:** `oeh_final` (108 trường, A2, 64 draws), forecaster huấn luyện trên panel 7 cohort (720 MDP / 10.800 dòng) qua `expanded_meta`. Không chỉnh gì.
- **Thứ tự:**
  1. Protocol được khóa trước khi sinh MDP: `results/confirm/protocol_colosseum.json`, SHA `dd977157…`.
  2. Dự báo được niêm phong lúc 08:55, SHA `200871d8…`.
  3. Learner chạy sau đó; outcome có lúc 09:05.
- **Gate chính:** OEH vs B2 matched phải có cận dưới đồng thời > 0 cho cả integrated Brier lẫn log-loss 7 lớp. Cận dưới tính bằng max-t một phía 95%, 10.000 bootstrap theo MDP.

### 1.2 Kết quả đăng ký trước (4 phép so)

| Phương pháp | Brier | Log-loss |
|---|---|---|
| **OEH current** | **0,1010** | **0,8903** |
| B2 matched | 0,1342 | 0,9815 |
| B2 + độ khó Colosseum (diameter, value norm, gaps) | 0,1422 | 1,0227 |

- OEH vs B2: Brier +24,8% (cận dưới +0,016), log-loss +9,3% (cận dưới +0,005) → **GO**.
- OEH vs B2 + độ khó Colosseum → **GO**.
- Lần thăm dò trước đó (32 MDP) cho gain gần như y hệt (+21,8% / +9,5%), nhưng log-loss NO-GO vì thiếu MDP. Hiệu ứng ổn định qua hai lần chạy.

### 1.3 Bộ comparator đầy đủ của cohort 7 (post hoc, 18 phép so)

Amendment `protocol_colosseum_amendment_1_comparators.json` được ghi trước khi tính. Comparator dùng nguyên `run_oeh_final_120.all_forecasts`, và hàm này tái tạo dự báo đã niêm phong với |diff| ≤ 1e-16.

| Comparator | Brier: gain / cận dưới | Log-loss: gain / cận dưới |
|---|---|---|
| B2 matched | +24,7% / +0,011 | +9,3% / **−0,022** |
| **B2 + ERI** | +15,7% / **−0,003** | +3,2% / **−0,080** |
| B2 + BWD | +23,9% / +0,010 | +9,4% / **−0,020** |
| 5 baseline tier 1 còn lại | đều dương | đều dương |
| B2 + EH oracle (tier 2) | +30% / +0,019 | +15% / +0,035 |

**Tier 1 NO-GO, tier 2 (phần đã chạy) GO.** Chưa chạy `b2_plus_tq_saco_original` và `published_metrics_only`, vì máy phụ không được phép chạy code `ml-jku/OfflineRL`. Hai comparator này không làm đổi verdict tier 1.

### 1.4 Phân tích mô tả chính

**Số MDP thắng/thua** (OEH thắng cả 2 chỉ số / thua cả 2 chỉ số, 18 MDP mỗi họ):

| Họ | vs B2 | vs B2 + ERI |
|---|---|---|
| FrozenLake | 16 / 1 | 15 / 2 |
| SimpleGrid | 12 / 1 | 9 / 5 |
| DeepSea | 11 / 5 | 11 / 6 |
| **RiverSwim** | **4 / 12** | **4 / 13** |

**Vì sao cận dưới âm dù gain dương:**
- Gain dao động lớn giữa các MDP: với log-loss vs B2, độ lệch chuẩn 0,43 so với gain trung bình 0,09.
- 5 MDP tệ nhất chiếm 50% tổng phần thua, trong đó 4 là RiverSwim. Ở các MDP này OEH gán xác suất 0,1–2% cho kết quả thật, nên log-loss "nổ".
- Họ phép so lớn hơn đẩy critical value từ 1,72 lên 2,25.
- Riêng với ERI, gain log-loss thực sự gần 0 (t = 0,61).

**Phát hiện thất bại so với ước lượng mức ngân sách:**

| | AUC phát hiện >x50 | Spearman mức (case hữu hạn) | Đúng ±1 mức | Log-loss có điều kiện |
|---|---|---|---|---|
| OEH current | **0,915** | 0,497 | 52% | 2,005 |
| OEH đời trước (`prior_oeh_matched`) | 0,903 | **0,506** | **62%** | **1,659** |
| B2 + ERI | 0,881 | 0,413 | 52% | 1,836 |
| B2 | 0,855 | 0,390 | 53% | 1,880 |

Gain chủ yếu đến từ việc nhận ra case "không học nổi trong x50", chiếm 73% số nhãn. Ở việc ước lượng mức cụ thể, OEH current không hơn B2 và thua OEH đời trước. Mọi phương pháp đều quá tự tin trên case hữu hạn: độ tin cậy vào lớp chọn khoảng 0,46, trong khi độ chính xác khoảng 0,28.

**Khác:**
- Xếp hạng 3 learner trong cùng MDP-behavior: Spearman OEH 0,72 so với B2 0,46.
- Độ khó lý thuyết Colosseum tương quan yếu với nhãn (|ρ| ≤ 0,50) và không giúp forecaster.
- 0/360 nhóm thiếu posterior draw.

### 1.5 Tái lập kết quả chính trên máy khác

Từ code và dữ liệu đã đẩy lên git, `driver/forecast.py reproduce` tái tạo **3.600/3.600** dự báo đã niêm phong của cohort 7 (|diff| tối đa 7,5e-8) và đúng điểm Brier/log-loss trong `summary.csv`. Như vậy kết quả chính của bài tái lập được trên một máy độc lập.

Báo cáo chi tiết: `results/confirm/BAO_CAO_XAC_NHAN_2026-10-14.md` (xác nhận) và `results/BAO_CAO_COLOSSEUM_2026-10-14.md` (thăm dò).

---

## 2. Điểm yếu và kế hoạch khắc phục

Bản thảo hiện tại đã giới hạn claim đúng mức ("within the studied environment families, not general superiority"). Với claim đó thì bài nộp được. Các điểm dưới đây là chỗ reviewer sẽ tấn công, xếp theo mức độ.

### Điểm yếu 1: Tổng quát hóa ra ngoài generator của nhóm (nặng nhất)

**Vấn đề.** Chuỗi MDP → behavior policy → pilot → nhãn ở cả meta-training (720 MDP) và cohort 7 đều do cùng bộ code sinh ra. Forecaster có thể học đặc điểm riêng của generator. BRIDGE thay MDP nhưng mỗi panel chỉ 5 MDP, kết quả lẫn lộn, khoảng tin cậy chứa 0; panel MiniGrid 14 đã mở nhãn. Colosseum có 72 MDP nhưng vẫn dùng behavior policy và learner của nhóm.

**Khắc phục:**

| # | Việc | Ghi chú | Công sức |
|---|---|---|---|
| 1a | **Đưa Colosseum vào bài thành một panel riêng** | Bảng theo learner; báo cáo cả RiverSwim và kết quả B2+ERI (ít nhất trong supplement), vì abstract đang nói "all 22 registered comparisons" | ~1 giờ viết, sau khi nhóm review code |
| 1b | Benchmark SPIBB (Laroche et al. 2019) | MDP **và** behavior policy đều do quy trình của tác giả khác sinh ra; nhóm đã có bản SPIBB pin | ~3–4 giờ máy |
| 1c | Leave-one-family-out trên dữ liệu nội bộ | Huấn luyện bỏ một họ, kiểm định trên họ đó; rẻ, trả lời "chuyển sang họ chưa thấy được không" | ~1–2 giờ |

### Điểm yếu 2: Thiếu đối thủ đã công bố cho đúng bài toán

**Vấn đề.**
- Đối thủ dự báo được phân phối 7 lớp đều do nhóm dựng (B2) hoặc thích nghi (XGBoost-CF).
- ERI/TQ/SACO/EH chỉ cho một con số nên chỉ so được bằng xếp hạng.
- Reviewer có thể nói đây là đối thủ yếu (straw man).

**Khắc phục:**

| # | Việc | Ghi chú |
|---|---|---|
| 2a | **Thêm XGBoost-CF/18 vào Colosseum** | Quy ước so sánh hiện tại (`CURRENT_EVIDENCE_AND_RETIREMENT_2026-10-09.md`) dùng B2 + XGBoost-CF làm bảng chính; Colosseum còn thiếu. Dùng code nhóm, ~30 phút |
| 2b | Baseline ngoại suy learning curve | Huấn luyện learner trên 20/50/100% pilot, khớp đường cong, ngoại suy mức đạt 95%. Đây là cách người làm thực tế hay dùng (Domhan et al. 2015; Mohr & van Rijn 2022). Nhóm đã có feature `curve`, chỉ cần dựng thành baseline độc lập |
| 2c | Baseline từ cận sample complexity offline RL | Ví dụ VI-LCB (Rashidinejad et al. 2021): ước lượng concentrability từ pilot, suy ra số mẫu, hiệu chỉnh thành 7 lớp. Đối thủ có gốc lý thuyết cho đúng đại lượng cần dự báo |
| 2d | Mô tả kỹ cách thích nghi XGBoost-CF | Trong supplement, cho thấy nó được tune công bằng như OEH |
| 2e | TQ/SACO gốc trên Colosseum | Cần chạy `ml-jku/OfflineRL` @ `8cc8eec` trên máy chính; chỉ để bảng đủ |

### Điểm yếu 3: Ablation chưa xác nhận, và kết quả ablation làm yếu câu chuyện chính

**Vấn đề.**
- M1 (`artifacts/week8/m1_matched_ablation_2026-10-13`) chạy trên cohort 7, mà nhãn cohort này đã mở từ 04/10, nên chỉ là exploratory.
- Bỏ bất kỳ khối nào thì Brier chỉ đổi tối đa 1,4%, trong khi khoảng cách B2 → OEH là 19%.
- Bỏ OEH-B (mô phỏng hậu nghiệm, ý tưởng cốt lõi) gần như không mất gì mà nhanh hơn 84%.
- Reviewer sẽ hỏi: "Cái gì làm OEH thắng B2? Có cần mô phỏng tốn kém không?"

**Khắc phục:**

| # | Việc | Ghi chú |
|---|---|---|
| 3a | Chạy lại đúng runner M1 trên **72 MDP Colosseum** | Định nghĩa block đóng băng 13/10 16:20 UTC, trước khi có nhãn Colosseum (14/10 09:05). Ghi rõ thứ tự thời gian. Kết quả còn cho biết khối nào gây thất bại ở RiverSwim. ~1–2 giờ |
| 3b | M1 trên một cohort mới chưa mở nhãn | Sạch nhất, đúng "sealed component test" mà bài tự đề xuất. ~3–4 giờ |
| 3c | Viết lại phần đóng góp nếu OEH-B vẫn không thiết yếu | Ví dụ nhấn mạnh tổ hợp tín hiệu thay vì riêng mô phỏng hậu nghiệm, và nêu biến thể rẻ hơn 84% như một lựa chọn thực tế |

### Điểm yếu 4: Phạm vi ứng dụng chỉ là MDP tabular nhỏ

**Vấn đề.**
- Posterior Dirichlet trên toàn bộ S×A×S, 64 MDP lấy mẫu, learner tabular → chi phí tăng theo S²A.
- Kích thước hiện tại: nội bộ trung bình khoảng 80 cặp (s, a) mỗi MDP; Colosseum giới hạn |S||A| ≤ 256; 125/155 môi trường BRIDGE bị loại vì vượt giới hạn.
- Reviewer sẽ hỏi phương pháp giúp gì cho người dùng D4RL hay dữ liệu ảnh.

**Khắc phục:** không giải được trong bài này, cần trình bày minh bạch.

| # | Việc |
|---|---|
| 4a | Bảng chi phí theo kích thước (|S||A| so với thời gian OEH-B/feature), lấy từ log đã có |
| 4b | Nêu rõ con số 125/155 BRIDGE bị loại vì kích thước |
| 4c | Hướng mở rộng cụ thể ở phần kết luận: ensemble mô hình học được thay posterior Dirichlet, MDP tuyến tính hoặc factored |

### Điểm yếu 5: OEH quá dứt khoát, log-loss bị "nổ" (phát hiện mới từ Colosseum)

**Vấn đề.** OEH đôi khi gán xác suất gần 0 cho kết quả thật, rõ nhất ở RiverSwim, khi learner học được ngay ở x1. Brier bị chặn trên nên ít ảnh hưởng, nhưng log-loss bị phạt rất nặng. Đây là nguyên nhân chính khiến cận dưới log-loss âm.

**Khắc phục:**

| # | Việc | Ghi chú |
|---|---|---|
| 5a | Làm mềm xác suất | Temperature hoặc trộn với phân phối nền, đặt sàn xác suất. **Chọn tham số chỉ trên meta-training (OOF)**, không trên 72 MDP Colosseum |
| 5b | Kiểm định lại trên MDP mới | Là bản OEH mới nên cần kiểm định riêng. Có thể kết hợp với 3b trong cùng một cohort mới |
| 5c | Nghiên cứu vì sao `prior_oeh_matched` ước lượng mức ngân sách tốt hơn | Cân nhắc kết hợp điểm mạnh của hai bản |

### Điểm yếu 6: Claim cần chỉnh cho khớp bằng chứng

| Nên tránh | Nên viết |
|---|---|
| "OEH dự báo chính xác lượng dữ liệu cần thiết" | "OEH phát hiện tốt khi nào dữ liệu không đủ cho một learner (AUC 0,915 trên MDP ngoài) và xếp hạng nhu cầu dữ liệu tốt hơn baseline; ước lượng mức ngân sách cụ thể vẫn là bài toán mở" |
| "Vượt mọi comparator" (khi nói về transfer) | "Trên 72 MDP Colosseum, OEH vượt B2 có ý nghĩa; so với B2+ERI chênh lệch không có ý nghĩa thống kê" |

### Thứ tự đề xuất

| Ưu tiên | Việc | Vá điểm yếu | Cần máy chính? |
|---|---|---|---|
| 1 | Review code `driver/` và đưa Colosseum vào bài (1a) | 1 | Không bắt buộc |
| 2 | XGBoost-CF trên Colosseum (2a) | 2 | Không |
| 3 | M1 trên Colosseum (3a) | 3 | Không |
| 4 | Bảng chi phí (4a, 4b) và chỉnh claim (6) | 4, 6 | Không |
| 5 | Baseline learning curve (2b) | 2 | Nên |
| 6 | Làm mềm xác suất (5a), rồi cohort mới cho cả M1 và OEH mới (3b + 5b) | 3, 5 | Có |
| 7 | SPIBB (1b), TQ/SACO gốc (2e) | 1, 2 | Có (code ngoài) |

Mục 2–4 máy phụ làm được ngay nếu nhóm trưởng đồng ý. Thiết kế chi tiết từng thí nghiệm, gồm công thức, file cần sửa, cách khóa protocol và cách đọc kết quả xấu, nằm trong [`KE_HOACH_THUC_NGHIEM.md`](KE_HOACH_THUC_NGHIEM.md).

---

## 3. Nội dung gói và cách chạy lại

### 3.1 Cấu trúc

| Thư mục / file | Nội dung | Gửi kèm? |
|---|---|---|
| `README.md` | Tài liệu này | ✅ |
| `KE_HOACH_THUC_NGHIEM.md` | Kế hoạch thực nghiệm chi tiết | ✅ |
| `driver/` | Toàn bộ script pipeline (mô tả bên dưới) | ✅ |
| `driver_tools/` | Script khóa protocol/amendment, script chẩn đoán lỗi treo | ✅ |
| `results/confirm/` | Protocol, amendment, seal, dự báo, nhãn, điểm, báo cáo của lần xác nhận | ✅ |
| `results/` (gốc) | Lần thăm dò 32 MDP, tái lập cohort 7, hardness các cohort | ✅ |
| `logs/` | Log từng bước | ✅ (nhỏ) |
| `data/` | npz kernel Colosseum, bundle pilot/training | ❌ lớn, sinh lại được |
| `workspace/` | Bản copy repo `Devoted` @ `d7134ad` (trừ .git/.venv/datasets) | ❌ đã có trong repo |
| `py39/`, `py312/` | Môi trường conda | ❌ |
| `colosseum/src` | Repo Colosseum @ `981b344` | ❌ clone lại từ GitHub |
| `external/public_offlinerl` | Repo OfflineRL @ `8cc8eec`, chưa chạy | ❌ |
| `driver_backup_run1/` | Bản driver trước khi tham số hóa | ❌ |

### 3.2 Các script trong `driver/`

| Script | Môi trường | Việc |
|---|---|---|
| `common.py` | — | Đường dẫn, biến `COLOSSEUM_RUN` (thăm dò = `colosseum`, xác nhận = `confirm`) |
| `colosseum_export.py` | py39 | Gọi sampler của tác giả và xuất kernel T/R/ρ; loại kernel trùng |
| `hardness_arrays.py` | py39 | Diameter, value norm, sub-optimality gaps (code Colosseum) |
| `export_cohort_kernels.py` | py39/py312 | Xuất kernel các cohort nội bộ để tính hardness |
| `build_colosseum.py` | py312 | Kernel → `TabularMDP`, lọc dynamic-range, sinh bundle bằng `generate_benchmark_bundle` của dự án |
| `colosseum_mdp.py` | py312 | Hàm `to_mdp` dùng chung |
| `run_oehb.py` | py312 | OEH-B theo nhóm (seed = sha256(namespace\|mdp\|behavior)), có thể chạy tiếp từ chỗ dừng |
| `features_colosseum.py` | py312 | Pilot features (`e1_pipeline._group_features`) và OEH-H (`horizon_group`) |
| `expectile_fix.py` | py312 | Bản sửa IQL expectile **đã được nhóm ghi nhận**, chỉ dùng khi bản gốc báo lỗi |
| `forecast.py` | py312 | `reproduce` (tái lập cohort 7) hoặc `colosseum` (dự báo và niêm phong) |
| `labels_colosseum.py` | py312 | Kiểm tra seal → chạy BC/CFQI/IQL → nhãn bằng `suffix_event` |
| `score_colosseum.py` | py312 | Chấm điểm và bootstrap max-t cho 4 phép so đăng ký trước |
| `comparators_colosseum.py` | py312 | ERI/BWD/TQ-SACO tham chiếu (`oeh_final_published`) và EH oracle |
| `forecast_comparators.py` | py312 | `all_forecasts` của cohort 7 trên Colosseum, kiểm tra trùng seal, chấm 18 phép so |
| `regen_cohort.py`, `verify_features.py`, `feature_diff.py`, `run_horizon.py` | py312 | Kiểm tra độ trung thực (tái sinh pilot, so feature) |
| `pipeline_confirm.sh` | bash | Toàn bộ lần xác nhận: export → hardness → bundle → oehb → features → seal → labels → score |

### 3.3 Chạy lại trên máy chính

1. **Môi trường:**
   - py39 cho Colosseum (`pip install -e colosseum/src`). Colosseum không chạy trên Python 3.11.
   - py312 với các pin của repo: numpy 2.5.1, scipy 1.18.0, pandas 2.3.3, scikit-learn 1.9.1, xgboost 3.3.0.
2. Đặt `workspace/` trỏ tới repo `Devoted` (sửa `WS` trong `driver/common.py`), cùng dữ liệu cohort đã đẩy lên git.
3. **Quan trọng:** đặt `OPENBLAS_NUM_THREADS=1` (và `OMP_NUM_THREADS=1`). Nếu không, mỗi worker mở 16 luồng OpenBLAS, các worker tranh CPU và một số nhóm OEH-B chậm khoảng 4.000 lần. Lần chạy này đã treo hơn 1,5 giờ ở `deepsea_002`; sau khi đặt biến, xong trong 53 giây. **Máy chính chạy BRIDGE nhiều worker cũng nên kiểm tra điều này.**
4. Tái lập cohort 7 trước: `python forecast.py reproduce` phải ra 3.600/3.600 dòng khớp.
5. Chạy `bash driver/pipeline_confirm.sh`. Protocol đã tồn tại nên script không khóa lại.
6. Sau đó tùy chọn: `comparators_colosseum.py 14`, rồi `forecast_comparators.py`.

**Thời gian thực đo** (14 worker, laptop): xuất MDP 103 phút, hardness 17 giây, bundle 2 phút, OEH-B khoảng 77 phút, feature 10 phút, seal 2 phút, learner 10 phút (64.800 job), chấm điểm 3 giây. **Tổng khoảng 3,4 giờ.**

### 3.4 Giới hạn của gói

- Chạy ngoài repo bằng script riêng. Cần review code trước khi đưa vào bài.
- Phần 9 comparator (mục 1.3) là **post hoc**. Kết quả đăng ký trước (mục 1.2) không bị ảnh hưởng.
- Benchmark vẫn tabular, |S||A| ≤ 256, chỉ 4 họ Colosseum vừa kích thước (MG-Empty/Rooms/Taxi bị loại vì quá lớn).
- 355/360 nhóm OEH-B chạy với OpenBLAS đa luồng, 5 nhóm chạy với 1 luồng. Khác biệt số học chỉ ở mức làm tròn.

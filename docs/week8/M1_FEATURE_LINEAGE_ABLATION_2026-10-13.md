# M1 — Feature lineage và runner ablation (13/10/2026)

**Trạng thái:** hoàn tất cả bốn pha. Nhãn kết quả: **exploratory** — cohort 7 đã mở nhãn từ 04/10. Đây không phải kiểm định xác nhận mới. Không sửa runner hay artifact niêm phong nào.

## 1. Sản phẩm

| Thành phần | Đường dẫn |
|---|---|
| Lineage 108 cột (nguồn, producer, dependency, phạm vi learner, cách mã hóa) | [`m1_feature_lineage.py`](../../scripts/analysis/m1_feature_lineage.py) |
| Runner bốn pha `lineage → cost → forecast → score` | [`run_m1_matched_ablation.py`](../../scripts/analysis/run_m1_matched_ablation.py) |
| Test nghiệm thu | [`tests/test_m1_matched_ablation.py`](../../tests/test_m1_matched_ablation.py) — **11 passed** |
| Output (namespace mới) | [`artifacts/week8/m1_matched_ablation_2026-10-13/`](../../artifacts/week8/m1_matched_ablation_2026-10-13/) |
| Log chạy | `logs/week8/m1_matched_ablation_2026-10-13.log` |

Chạy lại: `python -m scripts.analysis.run_m1_matched_ablation {lineage|cost|forecast|score} --output <thư mục mới>`. Mỗi pha từ chối ghi đè output của chính nó, và từ chối ghi vào thư mục cohort 7 hoặc vào namespace ablation cũ 07/10.

## 2. Lineage: 108 cột, 8 khối, 6 thành phần tính toán

Thứ tự cột lấy từ `protocol.json → oeh_final.fields` và phải trùng `run_oeh_final_120.CANDIDATE_VARIANTS["plus_prior_profile"]`; lệch là runner dừng. Cách chia khối theo [H1 review](H1_ABLATION_REVIEW_2026-10-08.md).

| Khối | Cột | Producer (thành phần tính) | Phạm vi learner | Dùng posterior | Replay learner |
|---|---:|---|---|---|---|
| G1 B2 | 18 | `stochastic_features.extract_stochastic_features` (`b2_statistics`) | chung | không | không |
| G2 OEH-B của learner | 18 | `oeh/predictive.posterior_outcomes` + `requirement_forecast` (`oehb_simulation`) | riêng | có | có |
| G3 concentration | 2 | `oeh/posterior.fit_posterior` (`posterior_fit`) | chung | có | không |
| G4 OEH-H/ceiling | 11 | `oeh/horizon` trên posterior mean (`oehh_horizon`) | chung | có | không |
| G5 risk/missing profile | 13 | `oeh/horizon.risk_profile` (`oehh_horizon`) | chung | có | không |
| G6 OEH-B context | 18 | như G2, sao chép CDF của 3 learner | chéo learner | có | có |
| G7 legacy | 10 | 4 cột `oeh_risk_v3` (`oeh3_risk`, chung) + 6 cột `lcoeh_v4` (`lcoeh4_crossfit`, riêng) | chung/riêng | không | LCOEH4 có |
| G8 legacy context | 18 | `lcoeh_v4`, sao chép của 3 learner | chéo learner | không | có |

Mọi cột đều sinh từ pilot x10 (LCOEH4 dùng thêm prefix x2/x5 của chính pilot) và metadata công khai; không cột nào đọc outcome hay MDP thật. Phụ thuộc tính toán: `oehb_simulation` và `oehh_horizon` cần `posterior_fit`. Vì vậy arm "bỏ OEH-B trực tiếp" vẫn cần posterior (OEH-H dùng posterior mean), nhưng không cần mô phỏng learner.

**Năm arm** (giữ thứ tự cột niêm phong, khớp tuyệt đối định nghĩa ablation 07/10):

| Arm | Bỏ khối | Số cột |
|---|---|---:|
| `b2_matched` | G2–G8 | 18 |
| `oeh_full_108` | — | 108 |
| `oeh_without_direct_oehb` | G2, G3, G6 | 70 |
| `oeh_without_oehh_risk` | G4, G5 | 84 |
| `oeh_without_legacy_diagnostics` | G7, G8 (OEH3/LCOEH4 + context) | 80 |

## 3. Nghiệm thu theo yêu cầu

| Yêu cầu | Cách cưỡng chế trong code | Kết quả |
|---|---|---|
| Test kiểm danh sách cột | lineage = protocol; đếm khối; arm ⊇ B2; giữ thứ tự; khớp lineage 07/10; dependency | PASS |
| Không outcome leakage | (1) audit hook `sys.addaudithook` chặn mở 15 file outcome và 2 thư mục execution của cohort 7 trong các pha lineage/cost/forecast; (2) query từ chối mọi cột `event`/`successes_*`; (3) feature set từ chối tên cột outcome/đặc quyền (`oracle_`, `eh_`, `tq_`, `saco_`, `policy_`, `true_`); (4) inventory input không chứa `labels.csv` của cohort 7 | PASS (test subprocess chứng minh hook chặn `pd.read_csv(labels.csv)`) |
| Dự báo đủ bảy lớp | `check_seven_class`: đủ `p_0..p_6`, hữu hạn, ≥0, tổng 1 (sai số 1e-9), mỗi khóa đúng một lần | 5 arm × 1.800 hàng PASS |
| Namespace output mới | `prepare_output` từ chối thư mục cohort 7 và namespace 07/10; mỗi pha từ chối ghi đè | PASS |
| Hash input/source | `input_inventory.csv` 4.262 file thực sự được mở (ghi lại qua audit hook), `source_inventory.csv` 28 module; seal ghi SHA-256 của predictions, tuning, lineage, hai inventory, và các file niêm phong cohort 7 | PASS |
| Cùng meta-training/fold/tuning | 720 MDP/10.800 hàng, 7 cohort đã mở, fold hash theo MDP, convex weights OOF; **arm full và B2 tái tạo dự báo niêm phong cohort 7 với sai khác tối đa 8,9e-8 / 3,6e-8** (ngưỡng 1e-6) | PASS |

**Ghi chú provenance.** `feature_lineage.json` ghi hash runner tại pha lineage (`8bdc7144…`). Trước pha cost/forecast, runner có đúng một sửa đổi: `check_seven_class` từ chối khung thiếu cột `p_0..p_6` bằng `RuntimeError` thay vì `KeyError` (lỗi do test phát hiện). Sửa đổi này không đổi dự báo. Pha forecast ghi hash runner hiện hành (`e5f4194a…`) trong `source_inventory.csv`, trùng với file đã commit. Định nghĩa lineage và arm được kiểm lại bằng digest trong cả hai pha.

Thêm: pha `cost` tính lại từng thành phần trên 40 nhóm (2 MDP/họ × 4 họ × 5 behavior) và so với giá trị niêm phong — sai khác tối đa 7,3e-12 (`cost_parity.json` = PASS). Dự báo mới trùng ablation 07/10 tới 1,1e-16.

## 4. Kết quả (84 MDP stochastic, exploratory)

IBS / log loss, thấp hơn là tốt. Δ = arm − full, khoảng 95% bootstrap theo MDP (2.000 lần, mô tả, không hiệu chỉnh đồng thời).

| Arm | Gộp IBS | Gộp LL | Δ IBS gộp [95%] | Δ LL gộp [95%] |
|---|---:|---:|---:|---:|
| B2 matched | 0,054727 | 0,627179 | +0,00874 [+0,00440; +0,01340] | +0,06410 [+0,03864; +0,09044] |
| OEH đầy đủ | 0,045989 | 0,563079 | — | — |
| Bỏ OEH-B trực tiếp | 0,046638 | 0,572843 | +0,00065 [−0,00050; +0,00184] | +0,00976 [+0,00161; +0,01848] |
| Bỏ OEH-H/risk | 0,046297 | 0,573348 | +0,00031 [−0,00119; +0,00179] | +0,01027 [−0,00189; +0,02236] |
| Bỏ legacy + context | 0,046204 | 0,561277 | +0,00022 [−0,00133; +0,00183] | −0,00180 [−0,01091; +0,00791] |

Theo learner, các khoảng loại trừ 0: bỏ OEH-B trực tiếp làm **CFQI** kém hơn (IBS +0,00275; LL +0,02711); bỏ OEH-H/risk làm **IQL** log loss kém hơn (+0,02513). Bỏ OEH-H/risk cải thiện điểm trung bình IBS của BC (−0,00150) nhưng khoảng chứa 0. Bỏ legacy không làm kém ở learner nào. Đầy đủ: `summary.csv`, `paired_mdp_delta.csv`, `calibration_by_budget.csv`.

**Diễn giải được phép.** Khoảng cách B2 → OEH đầy đủ (≈19% IBS, ≈11% LL gộp) lớn hơn nhiều so với tác động khi bỏ bất kỳ một khối nào (≤1,8% tương đối). Các khối OEH **chồng lấn thông tin**: không khối đơn lẻ nào là cần thiết cho phần lớn lợi ích, và kết quả không phân bổ lợi ích cho một cơ chế duy nhất. Tín hiệu riêng có khoảng loại trừ 0 chỉ ở log loss (OEH-B trực tiếp gộp và với CFQI; OEH-H/risk với IQL). Legacy OEH3/LCOEH4 không có đóng góp phát hiện được. Không được dùng bảng này để chọn lại bộ feature trên cohort đã mở; muốn kết luận cần cohort chưa mở và khóa contrast trước.

## 5. Chi phí

| Arm | Thành phần cần tính | Trung vị giây/nhóm (40 nhóm) | Ước tính tuần tự 600 nhóm | Fit head (giây) |
|---|---|---:|---:|---:|
| B2 | B2 | 0,003 | 2 s | 25,5 |
| OEH đầy đủ | tất cả | 29,8 | 5,2 giờ | 86,5 |
| Bỏ OEH-B trực tiếp | không mô phỏng learner | 4,7 | 1,4 giờ | 67,4 |
| Bỏ OEH-H/risk | không horizon | 27,5 | 4,4 giờ | 66,5 |
| Bỏ legacy | không OEH3/LCOEH4 | 28,5 | 4,6 giờ | 62,8 |

Mô phỏng OEH-B chiếm phần lớn chi phí (trung vị 26,0 s/nhóm trong probe; log niêm phong 600 nhóm: trung vị 10,7 s, tổng 9.037 s). **Lưu ý đo đạc:** probe chạy tuần tự trên laptop trong lúc job BRIDGE MiniGrid khác đang dùng CPU, nên thời gian tuyệt đối bị phóng đại; chỉ nên đọc tỉ lệ giữa các thành phần. Arm bỏ OEH-B trực tiếp giảm ≈84% thời gian sinh feature đổi lấy +1,4% IBS / +1,7% LL gộp — đây là đánh đổi mô tả, không phải lựa chọn phương pháp mới.

## 6. Việc tiếp theo (không tự làm trong M1)

- Việt Trần kiểm split/leakage, Quốc Việt kiểm bảng/metric, Hưng kiểm diễn giải cơ chế (phân công E1 tuần 8).
- Clean ablation xác nhận cần cohort chưa mở, khóa trước họ contrast và multiplicity.

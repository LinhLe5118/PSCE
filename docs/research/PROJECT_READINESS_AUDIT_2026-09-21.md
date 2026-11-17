# Audit khả năng đạt đồ án xuất sắc và paper A/A*/Q1 — 21/09/2026

> **Cập nhật benchmark sau audit:** [T1 v1.1.0 và bàn giao](STOCHASTIC_BENCHMARK_HANDOVER_2026-09-21.md)
> đã tách 84 stochastic MDP làm cohort chính, 36 deterministic controls riêng
> trước fitting. Các số liệu v4E bên dưới vẫn là mixed development lịch sử;
> không diễn giải chúng thành stochastic-only utility. T1 v1.0.0 bị supersede.

**Kết luận: có nền tảng tốt để hoàn thiện một đồ án xuất sắc; chưa đủ bằng
chứng để đánh giá bản hiện tại là paper cạnh tranh ở A/A*/Q1.** Một sprint
đóng gói tốt có thể đưa đồ án tới trạng thái bảo vệ vững. Chỉ hoàn thành danh
sách tuần 6 không tự đóng khoảng trống novelty và external validity của paper.
Đây là nhận định dựa trên repository và nguồn công khai, không phải điểm của
hội đồng, xác suất accept hoặc cam kết xếp loại; rubric của trường chưa có.

Audit tại source HEAD `2c055152550de85814f6b7890606d7d849cfb57d`. Đã đọc code,
protocol, báo cáo và CSV development, bản thảo paper, các chương thesis, CI;
chạy tests, build paper và tái tính v4E. **Không mở evaluation** và không tạo
development dataset. Không kiểm chứng toàn bộ chứng minh bằng formal methods,
không rerun 54,000 learner fits, không kiểm tra trạng thái remote CI/OpenReview.

## 1. Bằng chứng đo lại trong audit

| Kiểm tra | Kết quả | Artifact |
|---|---|---|
| Full test suite hiện có | 734 passed, 336 subtests passed, không skip | `artifacts/week6/t1_audit_2026-09-21/full_tests.log` |
| Coverage theo cấu hình CI có branches | 86%, vượt gate 82% | `coverage.log` cùng thư mục |
| Fixture generation smoke | 14/14 environments × 5 behaviors pass | `smoke.log` |
| Tests mới của T1 verifier | 4 passed: integrity, tamper, từ chối datasets, không cấp execution GO | `freeze_tests.log` |
| TeX structure/citation keys | Không dangling ref, không thiếu BibTeX key | `texcheck.log` |
| Paper build riêng | BibTeX và hai pass LaTeX cuối exit 0 | `paper_build_status.json`, `main.pdf` |
| Tái tính v4E từ OOF CSV development | `selection_summary` khớp chính xác report gốc | `development_reproduction/report.json`, `evidence_summary.json` |

Môi trường `.venv-audit` thiếu pytest; audit dùng `D:/anaconda3/python.exe`
với NumPy 2.5.1, SciPy 1.18.0, pytest 9.1.1. Không sửa dependency hệ thống.
MiKTeX có cảnh báo chưa kiểm updates; các pass cuối đã ghi returncode 0,
không cài package. Build được không có nghĩa đã đúng format của venue.

## 2. Kết quả khoa học đang có

V4E: return `0.695515` so với best-fixed `0.685941`, gain `0.009574`, CI
`[0.002438,0.018245]`, Holm `p=.0442` trên 15 selectors. Audit tái tạo chính
xác từ 300 groups thuộc **60 MDP**, không phải 300 mẫu độc lập. Gain là khoảng
**0.96 điểm phần trăm normalized return**; chiếm khoảng 39.7% headroom tới
oracle-best `0.710073`. Đây là effect có tiềm năng, nhưng chưa đo lợi ích triển
khai/compute và đang chịu selection optimism do phương pháp được chọn thích nghi.

Không thể diễn giải “weights không học từ labels” thành “method chưa từng được
chọn theo labels”. Việc chọn hai component, chuyển mục tiêu threshold sang
return, mở rộng ridge grid và chọn candidate đều dùng development. Holm 15
không tự điều chỉnh toàn bộ chuỗi nghiên cứu đó. Báo cáo gốc đã ghi đúng
`confirmatory_claim_allowed=false`; cần giữ ranh giới này trong abstract.

V3 frozen có primary pooled log likelihood `+0.03697`, CI cắt 0, p=.1118;
selector `−0.04197` so với best-fixed. Kết quả âm này vẫn có giá trị vì phân
biệt **sửa bão hòa cơ học** với **dự báo/ra quyết định có ích**. Hybrid
`B2+OEH3+LC-PMS` có predictive screen dương trên development, chưa độc lập.
Nguồn: `FINAL_DEVELOPMENT_RESULTS_2026-09-20.md` và artifact được dẫn ở đó.

## 3. Findings theo mức ảnh hưởng

| Mức | Finding và vị trí | Hệ quả / điều kiện đóng |
|---|---|---|
| Critical — scientific claim | V4E còn adaptive development; `oeh_equal_ensemble_selector.../report.json` | Chưa claim confirmatory/SOTA. Đóng bằng E1/S1 đã khóa, giữ cả kết quả âm. |
| Critical — execution | `run_oeh_ensemble_selector.py` hard-code 300 groups, đọc OOF outcomes; `run_oeh_development_smoke.py` từ chối evaluation | Không có bằng chứng unified sealed runner sẵn sàng chỉ từ các runner này. D1 cần runner 600 groups, schema/fold checks, resume và negative tests. Không sửa guard để chạy vội. |
| High — uncertainty | `cluster_bootstrap_difference` resample fixed OOF scores, không refit; p là tail frequency có thể bằng 0 | CI development không gồm toàn bộ bất định meta-training; T1 mới yêu cầu paired cluster refit-bootstrap. D1/S1 phải implement/benchmark, không đổi CI mà vẫn gọi cùng protocol. |
| High — novelty | `paper/sections/02_related.tex` có EH, pessimism, dataset quality, bandit nhưng thiếu đối chiếu trực tiếp OPS/OPE/model selection | Cần chứng minh khác biệt đối với FQE/Bellman-error selection, không chỉ thắng best-fixed. Đây là khoảng trống paper, không được thêm candidate vào evaluation đã khóa sau xem outcome. |
| High — baseline fairness | BCQ/CQL lịch sử chỉ x10, seed 0, 500 updates; panel khác ngân sách tuning | Thắng panel này không là SOTA. D1 khóa provenance, budget, convergence/failure; so sánh selector với selector mới giải thích phần đóng góp lựa chọn. |
| High — external validity | Benchmark tabular tự thiết kế; factorial chỉ sáu base blocks | 54,000 fits không thay thế diversity độc lập. Không suy generalization continuous/real-world. Với ràng buộc không tạo development mới, phải thu hẹp claim hoặc lên một nghiên cứu external riêng sau sprint. |
| High — publication delivery | `RESEARCH_ADMIN.md` nhắm ICLR 2027; `paper/main.tex` dùng IEEEtran; chưa có AI use statement ở main | Chưa sẵn sàng nộp venue đó. P1 phải dùng template đúng, kiểm page budget, disclosure chính xác và submission eligibility. |
| Medium — cross-fit assumption | `code/lcoeh_v4.py`: một trajectory được chia time blocks | Không gọi hai block là trajectory độc lập; field independence_fraction và limitation phải xuất hiện trong H1/H2. |
| Medium — reporting | Abstract nói single-use grouped evaluation và ngay sau đó nói các predictive gains | Dễ trộn v3 frozen với v4 adaptive dù câu cuối có disclaimer. P1 nên tách rõ hai giai đoạn ngay tại claim dương. |
| Medium — discoverability | README vẫn có trạng thái 10/09 và “chưa full training”; docs cũ nói final negative | Người review dễ kết luận sai hiện trạng. Đã thêm đường dẫn entrypoint cập nhật; tài liệu lịch sử không bị viết lại. |
| Medium — delivery/release | Thesis có bảy chương Markdown, demo và release manifest; chưa chứng minh bản thesis theo template trường hoàn tất; README ghi chưa chọn LICENSE | Cần template, hình/bảng/references, demo rehearsal và quyết định license trước public release; không tự chọn license thay nhóm. |

## 4. Khoảng cách tới từng mục tiêu

| Mục tiêu | Nhận định hiện tại | Cần thêm gì để đủ sức thuyết phục |
|---|---|---|
| Đồ án xuất sắc | **Khả thi**, có chất liệu nghiên cứu vượt một implementation demo | Chứng minh có giới hạn rõ, protocol đúng, tái lập, giải thích kết quả âm, thesis hoàn chỉnh, demo tốt và bảo vệ được lựa chọn thiết kế; hội đồng quyết định theo rubric. |
| Paper conference A | **Có hướng phát triển, chưa submission-ready** | Đóng claim độc lập, định vị novelty trước OPS/OPE, ablation đúng câu hỏi và evidence phù hợp venue. Không chỉ tăng số runs. |
| Paper A* | **Chưa đủ cơ sở ở bản hiện tại** | Một đóng góp có sức nặng hơn cho cộng đồng: insight/theory mới đủ tổng quát hoặc kết quả empirical thuyết phục với strong selector baselines và phạm vi ngoại suy được chứng minh. |
| Journal Q1 | **Chưa thể kết luận nếu chưa chọn journal/category/year** | Manuscript sâu và hoàn chỉnh, phù hợp scope, fair comparisons, validity và reproducibility; Q1 không phải cùng thang đánh giá với A*. |

Xếp hạng conference và quartile journal phân loại venue, không trực tiếp chấm
chất lượng từng bài. Quartile phụ thuộc category và năm; tham chiếu
[ICORE](https://www.core.edu.au/icore-portal) và
[Clarivate về quartile](https://clarivate.com/academia-government/blog/a-primer-on-ties-in-the-jcr/).
Không có quy tắc “p<.05 ⇒ paper A*”. Review cần correctness, rigor, novelty,
significance; SOTA không bắt buộc nếu insight đủ giá trị theo
[ICLR reviewer guide](https://iclr.cc/Conferences/2026/ReviewerGuide).

Các prior work cần đối chiếu trực tiếp:

- [A Dataset Perspective on Offline RL](https://proceedings.mlr.press/v199/schweighofer22a.html): vị trí của dataset characterization, coverage/quality.
- [Hyperparameter Selection for Offline RL](https://arxiv.org/abs/2007.09055): offline ranking/selection, ảnh hưởng Q estimation và tuning.
- [When is Offline Policy Selection Sample Efficient?](https://arxiv.org/abs/2312.02355): liên hệ OPS với OPE/Bellman error và phương pháp IBES. Bản v2 năm 2026, không được bỏ qua khi nói novelty hiện hành.

Inference của audit: OEH-v4E có thể khác các phương pháp trên ở **forecast từ
pilot cho future-budget learner**, nhưng repo chưa chứng minh sự khác biệt đó
đủ mới/đủ hữu ích so với các lựa chọn trực tiếp. Audit này là rà soát trọng điểm,
không phải systematic literature review và không xác nhận “chưa ai làm”.

## 5. Lịch tuần 6 và khả năng kịp công bố

[ICLR 2027 CFP](https://iclr.cc/Conferences/2027/CallForPapers) xác nhận abstract
18/09 và full paper 25/09/2026, 23:59 AoE. Full deadline tương ứng
**26/09/2026 18:59 giờ Việt Nam**. Ngày audit là 21/09, nên abstract đã qua hạn;
repository không chứng minh đã nộp abstract. Không suy ra chưa nộp chỉ vì repo
không có receipt. Nếu chưa nộp, không coi ICLR 2027 là đích nộp mới còn khả dụng.

E1 ngày 26/09, S1/P1/H2 kết thúc 27/09 không cho đủ thời gian đưa kết quả đã
review vào full submission. Kể cả E1 chạy sáng 26/09, lịch thống kê/tích hợp
vẫn chưa phù hợp. Không nên rút ngắn các gate kỹ thuật để chạy theo deadline.

[Author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) yêu cầu
template ICLR và tối đa 9 trang main lúc nộp; IEEE 11 trang hiện build không
chuyển tương đương sang page budget đó. AI use statement là required,
reproducibility statement là recommended; văn bản admin cũ ghi bắt buộc kèm
commit hash là quá mức quy định chính thức. T1 freeze hash là yêu cầu nội bộ.

Thứ tự làm tiếp: H1/D1 khóa đúng công thức và runner → B2 review độc lập → E1
một execution → S1 phân tích đúng freeze → P1/H2 tích hợp, review → G1 demo và
bản thesis nộp. D1 phải đo chi phí 2,000 bootstrap refits của meta-models;
không lấy runtime kernel/tabular cũ làm tổng runtime thống kê mới.

Nếu E1 dương: claim benchmark-specific utility và predictor tương ứng gate
đạt; vẫn phải đóng novelty/fairness để nâng paper. Nếu E1 âm hoặc CI rộng:
đồ án vẫn có thể rất tốt với mechanism/negative result nghiêm túc, nhưng không
được kể như một selector đã thành công. Cả hai nhánh đều cần bản thesis hoàn
chỉnh; việc xin GVHD cho mở evaluation đã được bỏ theo chỉ thị chủ dự án.

## 6. T1 đã bàn giao

[Protocol v1.0.0](../protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.0.0_2026-09-21.md)
và JSON khóa primary decision margin .005, ba hypotheses trong Holm family,
grouped nested evaluation, refit-bootstrap base MDP, censoring/fallback,
stop/resume và claim map. Manifest pin nguồn và verifier đi kèm.
T1 hoàn thành không đồng nghĩa H1/D1/B2 đã GO hoặc E1 đã chạy.

Audit không commit/push và không sửa số liệu lịch sử. Chứng cứ chronology của
freeze cần commit/version archive trước E1; SHA-256 tự nó chỉ chứng minh
integrity khi có trusted copy, không chứng minh thời điểm niêm phong.

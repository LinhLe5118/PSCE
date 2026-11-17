# Đánh giá dự án và kế hoạch viết paper + thesis — 13/10/2026

> **Đã được thay bằng [kế hoạch 14/10](KE_HOACH_VIET_PAPER_THESIS_2026-10-14.md)** (cohort 8, Colosseum, SPIBB, phân tích lại chéo panel, định vị đóng góp và câu được phép viết mới). Bản dưới đây giữ làm lịch sử.


Nguồn: toàn bộ working tree `Devoted/` (code, protocol, artifact, docs Week 8), `PAPER_OUTLINE_IEEE_2026-10-13.md`, `Thesis structure.md`, `paper/main.tex`, `DAORL_thesis.docx` và bảy chương Markdown đang duy trì ở `Drive/Thesis/`. Tài liệu này không đổi phương pháp, protocol hay verdict nào.

## 1. Đánh giá ngắn

**Điểm mạnh đã kiểm chứng.**

1. Bài toán đích rõ và đo được: `E_l` là budget đầu tiên có ≥8/10 seed đạt NR≥0,95 ở budget đó và mọi budget lớn hơn đã đo; lớp `>x50` là censoring phải. Code (`compare_difficulty.suffix_event`, `oeh/predictive.suffix_event`) và test (`test_h1_theory_glossary`) khớp định nghĩa.
2. Kết quả chính niêm phong trước outcome và tái tính được: cohort 7, 84 MDP stochastic, OEH 0,045989 IBS / 0,563079 LL so với B2 matched 0,054727 / 0,627179 (giảm 15,97% / 10,22%); 22/22 cận dưới đồng thời dương (`contrasts.csv`). Cùng cấu hình OEH đã Tier 1 GO ở cohort 4 → có lặp lại.
3. Kỷ luật bằng chứng tốt: seal/hash, kết quả âm được giữ (cohort 1–3, cohort 4 Tier 2/selection, FourRooms, BRIDGE bất lợi).
4. M1 (13/10, đã chạy xong) xác nhận runner ablation tái tạo fit đã niêm phong (sai khác ≤8,9e-8 ở hàng B2 và OEH đầy đủ) và tính lại từng khối feature khớp giá trị niêm phong (≤7,3e-12). Xem [biên bản M1](M1_FEATURE_LINEAGE_ABLATION_2026-10-13.md).

**Điểm yếu / rủi ro claim.**

1. Tổng quát hóa hẹp: 84 MDP thuộc 3 họ generator đã thấy khi phát triển. BRIDGE (MDP công bố, log/nhãn do nhóm tạo) cho kết quả **lẫn lộn theo learner**: Atari 5 MDP IQL kém; panel mở rộng 5 MDP CFQI tốt rõ nhưng BC/IQL bất lợi. Không có claim chuyển giao đồng đều.
2. Attribution: chưa có clean ablation trên dữ liệu chưa mở. Ablation M1/E1 trên cohort 7 đã mở chỉ là **exploratory**; nó không chứng minh từng khối cần thiết.
3. Selector: gate đổi giữa cohort 4 và 7 (bỏ v5 khỏi gate), selector thiết kế lại sau khi xem cohort 4, dùng nhãn họ MDP. Phải viết "thắng fixed CFQI, không thắng v5".
4. Chi phí: pilot tốn x10; dự báo x1–x5 là hồi cứu. Chi phí OEH-B (≈10 s/nhóm trung vị trên cohort 7, 9,31 giờ-nhóm trên 5 Atari BRIDGE) chưa được đối chiếu với lợi ích dữ liệu → không claim tiết kiệm.
5. Protocol niêm phong chứa câu "superiority over the published state of the art" — không được trích vào paper/thesis.
6. Hai bộ cận dưới selection trong cùng thư mục: dùng bộ của gate đăng ký 2 contrast (`confirmatory_verdict.json`: +4,26 điểm % / +0,010341) và nêu họ contrast.

## 2. Quyết định viết (có lý do)

| Quyết định | Lý do |
|---|---|
| Paper tiếng Anh, IEEEtran conference, ≤8 trang, cấu trúc I–VII theo đề cương IEEE | Đề cương 13/10 và `build.ps1` (giới hạn 8 trang) |
| Thesis tiếng Anh, giữ template FPT của `DAORL_thesis.docx`, cấu trúc 6 chương theo `Thesis structure.md` | Bản thesis và bảy chương duy trì đều tiếng Anh; cover/style trường nằm trong template. Nếu trường yêu cầu tiếng Việt, nguồn Markdown dịch được mà không đổi số liệu |
| **Thêm** kết quả BRIDGE (Atari 5 + mở rộng 5 OEH đầy đủ; MiniGrid 14 OEH rút gọn) vào paper VI.E và thesis 5.6 | Đề cương chỉ nêu FourRooms; nhưng hai panel OEH đầy đủ đã niêm phong trước nhãn và có hàng bất lợi. Bỏ qua sẽ là báo cáo chọn lọc |
| Giữ hàng B2+BWD trong bảng gate niêm phong, kèm chú thích hình học continuous-action không khớp | Đề cương paper liệt kê BWD; ledger 09/10 loại BWD khỏi narrative. Không được xóa hàng của gate đã niêm phong |
| Ablation M1 vào paper VI.C và thesis 5.5, nhãn **exploratory (cohort đã mở)** | Đề cương cho phép báo ablation nếu có, kèm giới hạn attribution |
| XGBoost-CF/18 trên 120 MDP: chỉ ở thesis 5.5 (exploratory); trên BRIDGE: bảng transfer | Gần endpoint nhất trong literature nhưng là adaptation; chạy trên nhãn đã mở |
| Thí nghiệm nested/target-head 12/10 chỉ ở thesis 5.5 dưới nhãn exploratory, paper bỏ | Đề cương: "chỉ giữ trong phần chính nếu phục vụ luận điểm cần thiết" |
| Mọi số trong bảng/hình lấy từ script đọc CSV, không gõ tay | Kế hoạch thesis 08/10 mục D |

## 3. Phân bổ nội dung

**Paper (≤8 trang).** I Introduction (~0,8 tr) · II Related Work (~0,7) · III Problem Formulation (~1) · IV Proposed Method (~1,5, hình pipeline) · V Experimental Setup (~1,3, bảng dataset/access) · VI Results and Discussion (~2,2: bảng forecast chính + cận; bảng learner/censoring; bảng ablation M1; bảng selection; đoạn BRIDGE/FourRooms) · VII Conclusion (~0,3).

**Thesis (thân ≈30–35 trang).** Ch.1 Giới thiệu 4 · Ch.2 Cơ sở lý thuyết & định nghĩa 5 · Ch.3 Dataset 5 · Ch.4 Phương pháp & đánh giá 7 · Ch.5 Kết quả & thảo luận 9 · Ch.6 Kết luận 2. Phụ lục: 108 cột + lineage (M1), cấu hình learner, access matrix, lịch sử cohort/gate, hash.

**Hình/bảng sinh từ CSV (namespace mới `artifacts/week8/manuscript_tables_2026-10-13/`).**
Hình phân bố lớp nhu cầu theo learner; hình calibration theo budget (OEH vs B2); hình ablation paired delta; bảng subgroup theo learner/họ/behavior/finite-censored; bảng selection; bảng BRIDGE theo panel.

## 4. Kiểm tra trước khi bàn giao

1. `pytest` toàn bộ; test M1 mới.
2. Script xuất bảng tự kiểm: số trong bảng = số tái tính từ `scored_predictions.csv`/`contrasts.csv`.
3. Paper: `paper/build.ps1` (pdflatex + bibtex, ≤8 trang).
4. Thesis: build DOCX từ template, cập nhật mục lục bằng Word, xuất PDF để kiểm số trang và hình.

## 5. Việc chỉ con người làm được

Điền tác giả/affiliation và venue cho paper; xác nhận ngôn ngữ thesis theo quy định khoa; Hưng duyệt diễn giải cơ chế (ablation), Việt Trần kiểm split/leakage của M1 (kế hoạch tuần 8 phân công); giảng viên hướng dẫn duyệt claim. Không ai trong nhóm tự review artifact của chính mình thay cho review độc lập.

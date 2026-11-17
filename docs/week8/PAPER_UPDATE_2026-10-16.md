# Cập nhật paper IEEE theo kết quả cuối (16/10/2026)

Viết lại `paper/main.tex` và các section theo mọi kết quả đã có trên `main` (tới commit `244b6e52`). Dạng: IEEE conference, hai cột, đúng giới hạn 8 trang của `paper/build.sh`.

## Main paper

- **Ba bài kiểm định cuối:**
  - Test 1: bốn panel; Test 2: panel 5; Test 3: panel 6.
  - Gọi theo đúng trạng thái: cổng đăng ký là kết luận chính, phần còn lại là mô tả.
  - Panel 6 có bảng mới `tab:panel6` và hình mới `fig:panel6`.
  - Cổng F1/F2 được nêu cùng ba kết quả ngược hướng thiết kế: OEH-F tốt hơn OEH; so sánh với head vững phụ thuộc panel; LCE thô mạnh về IBS trên MDP Playground.
- **Đảm bảo lý thuyết (mục `sec:theory`), dùng đúng phạm vi đã duyệt ở H5:**
  - Mệnh đề cận an toàn của cổng.
  - Định lý regret theo Brier (T3).
  - Phần "không khẳng định": head hằng cục bộ, cổng bão hòa, OEH **không** nhất quán.
  - Ký hiệu của luật quyết định đổi thành $q$ và $\kappa$, để không trùng $\tau$ (ngưỡng thành công) và $\lambda(d)$ (cổng).
- **Baseline và phân tích hậu kiểm** đưa vào phần kết quả: endpoint ngoại suy (T1), prefix + FQE (T2), TabPFN v2 (T3), S1–S4 (T4), nested LOPO (T5), độ nhạy theo nhãn (T6).
- **Related work** thêm các hướng gần nhất:
  - ngoại suy learning curve (Domhan 2015, LC-PFN 2023, Viering & Loog 2023);
  - FQE/OPE (Le 2019, Voloshin 2021);
  - TabPFN (Hollmann 2025);
  - proper scoring (Gneiting & Raftery 2007);
  - Kumar 2022, Rashidinejad 2021.
- **14 mục bib mới** trong `paper/references.bib`.

## Supplement

Ba section mới, đặt sau `supplement_theory_v2`:
- `supplement_theory_checked.tex`: phát biểu và chứng minh Theorem S1, Proposition S2 (luật ngưỡng), Corollary S3 (log loss) và Proposition S4 (hằng cục bộ, bão hòa).
  - Đây là bản viết lại của `paper/appendix_theory.tex` theo ký hiệu của bài; file gốc đã duyệt **không** bị sửa.
- `supplement_panel6.tex`: thiết kế, amendment A1, bảng theo learner, các so sánh mô tả.
- `supplement_posthoc.tex`: T1–T6 và hình `fig_threshold_sensitivity`.

## Hình panel 6

- Script: `scripts/analysis/paper_fig_panel6.py`. Chỉ đọc `summary.csv` và `tabpfn_descriptive.csv` đã commit của panel 6, không fit lại gì.
- Output: `paper/figures/fig_panel6.{pdf,png}` và `artifacts/week8/paper_fig_panel6_2026-10-16/`, gồm `relative_gains.csv` và `manifest.json` ghi hash của nguồn và output.
- Test: `tests/test_paper_fig_panel6.py`. Kiểm tra số F1/F2 khớp kết quả đã đăng ký và hash nguồn khớp manifest.

## Kiểm tra

- `bash paper/build.sh`: 8 trang, không có overfull hay undefined.
- `paper/build_supplement.ps1`: 7 trang.
- `python -m scripts.analysis.export_paper_tables --check` và hai lệnh `--check` của bảng difficulty: khớp.
- `scripts/testing/run_full_tests.py`: xem PR.

Không động tới protocol, seal hay file đã niêm phong nào.

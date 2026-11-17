# Chốt đồng bộ tên gọi cho luận văn, paper, supplement và hình (16/10/2026)

Linh yêu cầu chốt một bộ tên gọi duy nhất. Bộ tên đầy đủ nằm trong agent kit viết bài (ngoài repo): `TERMINOLOGY.md`, phiên bản 2026-10-16b. Bản ghi này tóm tắt quyết định và những gì đã sửa trong repo.

## Quyết định

| Đối tượng | Tên chuẩn | Không dùng |
|---|---|---|
| Ba bài kiểm định cuối | **Test 1** (internal 168, Colosseum 2029 72, SPIBB-2 108, forest 48), **Test 2** (FrozenLake 36, gambler 36), **Test 3** (MDP Playground 96, labmaze 96) | four-panel test, fifth/sixth panel, "panel 5/6" trong văn xuôi (chỉ còn trong tên artifact và code) |
| Cohort nội bộ | E1; cohort 1–2 (60 MDP); cohort 3–6 (120 MDP, theo thứ tự trong `run_oeh_final_120.py`); cohort 7 (`oeh_confirmatory_unseen_120_2026-10-04`); cohort 8 | seventh/eighth cohort; "cohort 4" cho bài kiểm định 120 MDP thứ tư (đúng là **cohort 6**) |
| Meta-training | E1 và cohort 1–6 (720 MDP) | — |
| Class prior | $\bar p_l$ | $\pi_l$ ($\pi$ là policy) |
| Nhãn dòng bảng | `OEH-F (no gate)`, `Robust head (B2 + D)`, `LCE raw`, `S4 (simplified)`… | `LCE (pilot learning curve)`, `Full head alone`, `B2 + diagnostics head`, `legacy` |
| Phạm vi so với B2 | "in every final test (registered in Tests 1 and 3, descriptive in Test 2)" | "in every test" không kèm phạm vi |
| Số nguồn ngoài | "eight external sources, seven of which enter the final tests" | lúc seven lúc eight không giải thích |

## Đã sửa trong repo (PR này)

**Paper và supplement**
- $\bar p_l$ thay cho $\pi_l$ ở Eq. (8), Proposition 1, `supplement_theory_v2.tex`, `supplement_theory_checked.tex`.
- "Cohort 7" thay cho "seventh cohort"; "cohort 6" thay cho "fourth cohort".
- Bỏ "legacy" ở `supplement_ablation.tex` và `supplement_colosseum.tex`.
- Nhãn `LCE raw`.
- Bỏ câu chú thích tên cũ dưới hình gain-by-distance, vì hình đã được vẽ lại.
- Hai câu phủ định có chữ "SOTA" được viết lại.
- `paper/appendix_theory.tex` (đã qua H5) **không** sửa.

**Bốn góp ý của review H6** (`REVIEW_PR68_BIB_H6_2026-10-16.md`):
- câu về số nguồn ngoài;
- chiều của regret ở Test 3 ("lowers regret by 2.61 … and by 7.91");
- phạm vi của "in every test" (abstract, Introduction, Conclusion);
- câu về τ = 0,99 trong `T10_PANEL6_RESULTS_2026-10-16.md`.

Thêm sửa bib `kumar2022prefer` thành tiêu đề ICLR 2022: "Should I Run Offline Reinforcement Learning or Behavioral Cloning?".

**Hình.** Script `scripts/analysis/figures_final_naming_20261016.py` vẽ lại năm hình có tên cũ, chỉ từ dữ liệu đã commit. Không fit lại hay chấm lại gì.
- Các hình: `fig_cross_panel`, `fig_gain_by_distance`, `fig_ablation`, `fig_threshold_brier` (nhãn cũ ghi sai "OEH", đúng là OEH-F), `fig_threshold_sensitivity`.
- Output nằm trong `artifacts/week8/figures_final_naming_2026-10-16/`, kèm manifest ghi hash của nguồn, script và output.
- Các bản dùng trong paper được làm mới trong `paper/figures/`.
- Thư mục artifact cũ (có hash trong manifest) **không** bị sửa.
- Hình `fig_panel6` giữ nguyên: nhãn của nó đã khớp bộ tên.

**`scripts/analysis/check_paper_tables.py`:** cập nhật tên dòng `LCE raw` cho `tab:fifth`. Script vẫn đối chiếu 144/144 số.

## Kiểm tra

- **Paper:** `bash paper/build.sh` cho 8 trang, không overfull, không undefined. Supplement 7 trang.
- **Lint tên gọi của kit:**
  - luận văn và paper: 0 ERROR (`lint_thesis.py <nguồn luận văn> --tex paper`);
  - `check_paper_tables`: khớp hết.
- **Test:** `scripts/testing/run_full_tests.py`, xem PR.

## Việc cho các thành viên

Bản nháp Chương 3 trong `docs/thesis/chapter3_dataset_2026-10-16/` chưa theo bộ tên và khung mục:
- 40 heading cấp 3;
- 13 chỗ "panel 5/6";
- 5 citation chưa có mục tham khảo;
- 2 câu thuộc B13.

Người phụ trách sửa theo thẻ CH3 và lint của kit. PR này không sửa bản nháp đó.

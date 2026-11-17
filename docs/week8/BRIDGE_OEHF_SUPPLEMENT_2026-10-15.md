# BRIDGE follow-up trong supplement, gọi đúng là OEH-F (15/10/2026)

Quyết định của Linh (15/10, phương án a): đưa kết quả BRIDGE vào supplement như **kết quả mô tả**, gọi head 108 cột là **OEH-F**.

## Bối cảnh

- Chuỗi build BRIDGE của nhóm (`benchmarks/bridge/final_manuscript_watch.ps1` → `scripts/analysis/finalize_verified_manuscripts.py`, khởi động 9/10) chạy xong lúc 18:23 ngày 15/10.
- Nó sinh các đoạn bài báo và luận văn **trước khi có phương pháp cuối**, nên gọi `oeh_full_108` là "Full OEH".
- Theo quy ước hiện tại, mô hình đó là OEH-F. OEH (có cổng dịch chuyển) chưa từng chạy trên BRIDGE.

## Kiểm tra trước khi dùng

Mọi hash trong chuỗi bằng chứng đều khớp với file:
- 4 panel nguồn: manifest, file điểm, chuỗi protocol, seal và nhãn (`verify_common_difficulty_metrics.verify_source_chain`);
- `summary.csv` và `paired.csv` của audit;
- các đoạn đã sinh, hai PDF và docx.

Số của panel 5 MDP trùng với [`BRIDGE_ALL_RUNS_RESULTS_2026-10-13.md`](BRIDGE_ALL_RUNS_RESULTS_2026-10-13.md).

## Đã làm

- **Commit các artifact chưa được commit,** nguyên byte:
  - `bridge_opened_minigrid_full_{oehb,forecast,analysis}_exploratory_2026-10-09`;
  - `difficulty_common_metric_final_2026-10-09`;
  - `difficulty_manuscript_build_manifest_2026-10-09.json`.
- **Sinh lại mục supplement** bằng `scripts/analysis/bridge_supplement_oehf_20261015.py`:
  - dùng đúng code sinh bảng của chuỗi trên cùng audit đã kiểm hash, rồi đổi tên;
  - kết quả: `paper/sections/supplement_bridge_oehf.tex`, manifest trong `artifacts/week8/bridge_supplement_oehf_2026-10-15/`.
- **`supplement.tex`** `\input` file đã commit. Dòng `\IfFileExists{sections/supplement_bridge_verified.tex}` bị bỏ, vì nó khiến bản build ở máy tự chèn file chưa commit, nên PDF không tái lập được từ git.
- **`supplement_reduced_bridge.tex`** dùng cùng tên gọi: OEH-F, và "OEH-F without direct OEH-B" cho bản rút gọn. Mục này cũng ghi rằng OEH-F đã được chạy trên 14 task MiniGrid sau khi nhãn đã mở, và đó là kết quả thăm dò.
- **Không sửa** các output gốc của chuỗi 9/10, gồm các file `paper/sections/bridge_verified_*.tex` chưa commit và các bản nháp luận văn đời 9/10 trong `Drive/Thesis`.

## Kết quả, chỉ mô tả

- **Panel 5 MDP** (niêm phong trước nhãn), OEH-F trừ XGBoost-CF/18 về Brier:
  - BC +0,019 [−0,008; 0,051]
  - CFQI −0,113 [−0,227; −0,017]
  - IQL −0,003 [−0,032; 0,024]
- **14 task MiniGrid** (nhãn đã mở trước, nên chỉ là thăm dò), OEH-F trừ B2 về Brier:
  - BC +0,027
  - CFQI −0,086
  - IQL −0,008

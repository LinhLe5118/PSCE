# Review PR #68 (paper bản cuối), kiểm tra bib và H6 — 16/10/2026

**Người kiểm tra:** thành viên nhóm, trên máy phụ. **Repo:** `main` tại `990b2559`.

**Phạm vi:**
- diff `244b6e52..990b2559` trong `paper/`;
- 14 mục bib mới;
- mọi con số trong các bảng mới và trong đoạn văn nói về Test 1–3.

**Kết luận chung: PR đạt.** Không có con số nào lệch artifact. Còn **4 sửa nhỏ nên làm** (mục 1) và **1 sửa bib** (mục 2).

## 1. Review nội dung PR #68

### Đã đối chiếu và khớp

**Bảng.** Script [`scripts/analysis/check_paper_tables.py`](../../scripts/analysis/check_paper_tables.py) (test: `tests/test_check_paper_tables.py`) đọc số từ `.tex` rồi so với artifact: **144/144 số khớp**.

| Bảng | Số đã kiểm | Nguồn |
|---|---:|---|
| `tab:v2test` (Test 1) | 60 | `artifacts/oeh_v2_sealed_test_2026-10-14/summary.csv` |
| `tab:fifth` (Test 2) | 36 | `artifacts/oeh_fifth_panel_2026-10-15/summary.csv` |
| `tab:panel6` (Test 3) | 48 | `artifacts/oeh_panel6_2026-10-15/summary.csv`, `tabpfn_descriptive.csv` |

**Văn bản (đối chiếu tay):**
- *Abstract:* 660 MDP = 396 + 72 + 192; 688.320 job = 371.520 + 86.400 + 230.400; 37,7% / 23,4%; 25,6% / 17,9%.
- *Test 1:*
  - các bước gate: 0,0171 (0,0107); 0,0597 (0,0344); 0,0009 / 0,0115; 0,0047 / 0,0662;
  - gain so với B2: 15,7% / 8,3% trong generator;
  - regret: 13,1 / 15,1 / 16,5 trong generator; 20,0 / 28,1 / 35,8 ở forest.
- *Test 2:*
  - FrozenLake: +0,0065 / +0,072;
  - OEH so với OEH-F: +0,0009 / +0,020;
  - OEH so với B2 (LL): +0,058 [0,031; 0,086].
- *Test 3:*
  - F1/F2 và regret khớp `result.json`;
  - OEH-F hơn OEH: −3,3% / −3,6%;
  - OEH hơn head vững 14,8%, có ý nghĩa chỉ ở labmaze; ở MDP Playground KTC chứa 0;
  - LCE thô trên MDP Playground: OEH kém 32,7%;
  - TabPFN: 31,4% / 13,5%;
  - bỏ OEH-H: 0,0048 [0,0023; 0,0074];
  - Murphy, resolution: 0,120–0,128 so với 0,056–0,063.
- *Giới hạn:*
  - 71% dòng meta-training bị cắt (tính lại: 71,1%);
  - 91–98% dòng bị cắt ở Test 1 vẫn không đạt ở x200 (tính lại: 90,6 / 95,2 / 96,0 / 98,1%).
- *Định dạng:* `main.pdf` 8 trang, `supplement.pdf` 7 trang, không có tham chiếu lỗi (`??`). `export_paper_tables --check`: "Paper tables match sealed evidence".

### Nên sửa

1. **Số nguồn ngoài không thống nhất.** Contribution 1 (Introduction) ghi "*eight* external sources"; abstract và conclusion ghi "*seven*". Bảy là số nguồn trong ba kiểm định cuối: Colosseum, SPIBB, forest, FrozenLake, gambler, MDP Playground, labmaze. Tám là khi tính thêm BRIDGE. Đề xuất: dùng "seven" ở cả ba chỗ, hoặc ghi "eight, seven of them in the final tests".
2. **Chiều của regret ở Test 3 (`06_results.tex`).** Câu "($-2.61$ budget units against B2 + prefix/FQE, LB $1.29$; $-7.91$ …)" đặt cạnh nhau một mức *thay đổi* âm và một cận dưới *cải thiện* dương. Đề xuất: "lowers regret by 2.61 budget units against B2 + prefix/FQE (LB 1.29) and by 7.91 against B2".
3. **"In every test" trong abstract và mục Results.** Ở Test 2, so sánh với B2 chỉ là mô tả, không phải tương phản đăng ký (Test 2 chỉ đăng ký so với head vững). Reviewer có thể bắt lỗi. Đề xuất: "…in every test (registered in Tests 1 and 3, descriptive in Test 2)".
4. **Bản ghi T10 (`docs/week8/T10_PANEL6_RESULTS_2026-10-16.md`), mục 5.** Bộ số "OEH 0,0964 / 0,989 …" là của biến thể nhãn τ = 0,99, không phải endpoint đầy đủ. Số của endpoint đầy đủ là 0,0971 / 1,039. Paper ghi đúng; chỉ bản ghi nội bộ cần sửa câu chữ.

### Ghi chú quy trình

- PR #68 đã merge sau khi CI xanh, chưa có người review. Bản review này có thể dùng làm review bổ sung: gắn vào PR như comment.
- Ba bảng mới (`tab:v2test`, `tab:fifth`, `tab:panel6`) **vẫn gõ tay**, không có script sinh ra. Đến giờ chúng khớp artifact. Nhánh này đã thêm `scripts/analysis/check_paper_tables.py` và `tests/test_check_paper_tables.py`, để mỗi lần sửa paper đều tự kiểm. Bảng `tab:replication` và `tab:setup` chưa được script này kiểm.

## 2. Kiểm tra 14 mục bib mới

Tra từng mục trên trang chính thức hoặc chỉ mục (NeurIPS, PMLR, IJCAI, IEEE, JAIR, Nature/PubMed, arXiv, mlanthology).

| Key | Kết quả |
|---|---|
| `levine2020offline` | Đúng (arXiv:2005.01643) |
| `kumar2022prefer` | **Cần sửa tiêu đề.** Tiêu đề trong bib ("When Should We Prefer Offline RL over BC?") là của bản arXiv 2204.05618. Bản ICLR 2022 có tên "**Should I Run Offline Reinforcement Learning or Behavioral Cloning?**". Hoặc đổi title cho khớp venue ICLR 2022, hoặc giữ title và đổi venue thành arXiv |
| `rashidinejad2021bridging` | Đúng: NeurIPS 34, tr. 11702–11716 (theo BibTeX chính thức của NeurIPS) |
| `domhan2015speeding` | Đúng: IJCAI 2015, tr. 3460–3468 |
| `adriaensen2023efficient` | Đúng: NeurIPS 36 (2023), 4 tác giả |
| `viering2023shape` | Đúng: IEEE TPAMI 45(6), 7799–7819, 2023 |
| `le2019batch` | Đúng: ICML 2019, PMLR 97:3703–3712 |
| `voloshin2021empirical` | Đúng: NeurIPS 2021, Datasets and Benchmarks track |
| `gneiting2007strictly` | Đúng: JASA 102(477), 359–378 |
| `hollmann2025accurate` | Đúng: Nature 637, 319–326, DOI khớp, 8 tác giả khớp |
| `maurer2009empirical` | Đúng: COLT 2009 |
| `towers2024gymnasium` | Đúng (arXiv:2407.17032) |
| `rajan2023mdp` | Đúng: JAIR 77, 821–890, 2023 |
| `beattie2016deepmind` | Đúng (arXiv:1612.03801). Lưu ý: labmaze là thư viện riêng của DeepMind (bản 1.0.6), nên có thể thêm một chú thích phần mềm cho labmaze bên cạnh DeepMind Lab |

**Nguồn tra cứu:**
- [NeurIPS BibTeX Rashidinejad 2021](https://papers.nips.cc/paper_files/paper/2021/file/60ce36723c17bbac504f2ef4c8a46995-Bibtex.bib)
- [ICLR 2022 poster Kumar et al.](https://iclr.cc/virtual/2022/poster/6654)
- [arXiv 2204.05618](https://arxiv.org/abs/2204.05618)
- [mlanthology Domhan 2015](https://mlanthology.org/ijcai/2015/domhan2015ijcai-speeding)
- [NeurIPS 2023 LC-PFN](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3f1a5e8bfcc3005724d246abe454c1e5-Abstract.html)
- [IEEE Xplore Viering & Loog](https://ieeexplore.ieee.org/document/9944190/references)
- [PMLR Le et al. 2019](https://proceedings.mlr.press/v97/le19a.html)
- [NeurIPS D&B Voloshin et al.](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/a5e00132373a7031000fd987a3c9f87b-Abstract-round1.html)
- [PubMed Hollmann et al. 2025](https://pubmed.ncbi.nlm.nih.gov/39780007/)
- [mlanthology Maurer & Pontil](https://mlanthology.org/colt/2009/maurer2009colt-empirical)
- [JAIR MDP Playground](https://jair.org/index.php/jair/article/view/14314)

## 3. H6: còn lại cho người

- Đã kiểm: mọi số trong ba bảng mới; abstract; các đoạn Test 1–3; phần Limitations.
- Chưa kiểm: `tab:replication` (đã đối chiếu tay một phần với tài liệu nội bộ, khớp), `tab:setup`, Related work, các đoạn của supplement ngoài `supplement_panel6.tex`.
- **Lý thuyết.** Phần này đã qua H5 (Linh, 15/10); PR #68 chỉ viết lại theo ký hiệu mới. Đã đọc `supplement_theory_checked.tex` và kiểm từng bước:
  - **Theorem S1:** cận theo hàng $2\kappa\|F-F^\star\|_\infty$; $\mathbb E\|F-F^\star\|_2^2 = 6\Delta$ vì IBS lấy trung bình trên 6 ngưỡng.
  - **Proposition S2:** quyết định trùng khi $\|F-F^\star\|_\infty < m \le \min_j|F^\star_j-q|$; dùng Markov.
  - **Corollary S3:** TV kết hợp Pinsker, ra $\kappa\sqrt{2\Delta_{LL}}$.
  - **Proposition S4:** trạng thái cục bộ hằng và gate bão hòa.
  - **Mệnh đề gate ở thân bài:** $-\log(1-\varepsilon_{\max})$ trên dòng bị cắt.

  Các bước đúng và nhất quán với phát biểu trong thân bài (Theorem 1, Proposition 1). Phạm vi ("not consistent") được giữ đúng như bản đã duyệt.

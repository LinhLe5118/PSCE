# D1 — Baseline provenance và unified-runner preflight

## Trạng thái

**Hoàn tất phạm vi baseline-preflight của W6.** Điểm vào thực thi là
`scripts/validation/run_baseline_preflight.py`; bằng chứng đã commit nằm tại
`artifacts/week6/d1_unified_preflight_2026-09-21_final/`.

Đây là preflight trên development, không phải lần chạy kết quả. Lệnh cố ý
không fit learner, không chấm policy và từ chối manifest có record ngoài
development trước khi mở buffer. Vì vậy trạng thái `pass` chỉ xác nhận tính
toàn vẹn, provenance và khả năng gọi chung các runner; nó không phải bằng
chứng chất lượng phương pháp hoặc quyền mở evaluation.

## Kết quả nghiệm thu

| Tiêu chí W6 | Bằng chứng | Kết quả |
|---|---|---|
| Đủ bảy họ baseline | CFQI, IQL, BC, Pi_b-SPIBB, VI-LCB, BCQ và CQL; VI-LCB có hai cấu hình đã khóa nên bảng có tám dòng candidate | Pass |
| Nguồn và phiên bản | Mỗi candidate ghi loại implementation, repository, commit và runner; checkout ngoài được kiểm commit và trạng thái file tracked | Pass |
| Cấu hình và giới hạn tuning | Bảng provenance khóa iteration/expectile, `N_wedge`, `c_b`, network/update budget, BCQ threshold và CQL alpha | Pass |
| Cùng data contract | Kiểm toàn bộ 4.725 record development, 4.500 training prefix và 825 physical buffer bằng SHA-256 | Pass |
| Cùng panel preflight | Budget `10|S||A|`, training seed `0`, 75 instance-policy groups; normalization và success threshold lấy từ protocol đã đăng ký | Pass |
| CLI/adapter sẵn sàng | Tabular runner, SPIBB, VI-LCB và neural BCQ/CQL đều qua `--help` | Pass |
| Evaluation vẫn niêm phong | Guard báo `0` evaluation metadata record và `0` evaluation buffer được mở | Pass |
| Hồ sơ kiểm toán | Có report JSON, bảng provenance CSV và báo cáo Markdown | Pass |

Artifact đã commit ghi nhận chính xác:

- 4.725 manifest records được kiểm;
- 825 physical buffers được xác minh checksum;
- 75 prefixes trong panel dry-run, tương ứng 150 registered tabular runs và
  75 BC sanity runs nếu thực thi;
- bốn checkout ngoài đúng commit và không có tracked-file modification;
- bốn CLI probes đều trả exit code `0`;
- `fit_called=false` và `sealed_evaluation_accessed=false`.

## Baseline và mức độ tái sử dụng mã nguồn

| Họ baseline | Candidate | Implementation dùng cho panel |
|---|---|---|
| CFQI | `cfqi_tabular` | Reference implementation trong project |
| IQL | `iql_tabular` | Tabular implementation trong project |
| BC | `behavior_cloning_tabular` | Logged-action frequency policy trong project |
| Pi_b-SPIBB | `pib_spibb_n10_author` | Core từ author repository, adapter dữ liệu/scorer của project |
| VI-LCB | `vi_lcb_cb005_author_empirical`, `vi_lcb_cb144_theory` | Discounted paper-spec port, hai cấu hình đã khóa |
| BCQ | `discrete_bcq_author` | Author learner, adapter one-hot discrete |
| CQL | `discrete_cql_d3rlpy` | Pinned d3rlpy implementation |

Danh sách commit đầy đủ và checksum nguồn nằm trong
`baseline_provenance.csv` và `preflight_report.json` của artifact. Tên
`author`, `port` và `pinned implementation` phải được giữ nguyên trong bảng
kết quả sau này; không gộp chúng thành một tuyên bố “tái lập paper/SOTA”.

## Cách chạy lại

Chạy bằng Python environment của project:

```powershell
python scripts/validation/run_baseline_preflight.py `
  --dataset-dir <development-bundle> `
  --output-dir artifacts/week6/d1_unified_preflight_YYYY-MM-DD
```

Để kiểm luôn các checkout ngoài, đặt bốn repository con `spibb`, `vi_lcb`,
`bcq`, `d3rlpy` dưới cùng một thư mục rồi thêm:

```powershell
  --external-root <external-checkout-root>
```

Mỗi lần chạy phải dùng output directory mới. Bundle thành công gồm:

- `preflight_report.json`: kết quả máy đọc được, contract, checksum và guard;
- `baseline_provenance.csv`: một dòng cho mỗi candidate đã khóa;
- `REPORT.md`: bản đọc nhanh cho reviewer.

Nếu không truyền checkout ngoài, commit vẫn được ghi trong provenance nhưng
trạng thái checkout là recorded-only; không được diễn giải như đã xác minh mã
nguồn local. Bất kỳ CLI failure, hash mismatch, split sai hoặc checkout bẩn
đều làm preflight thất bại.

## Phụ thuộc và ranh giới trách nhiệm

D1 baseline-preflight tự nó không còn phụ thuộc deliverable khác để tái lập.
Các bước downstream vẫn có gate riêng:

1. **B2** phải review độc lập package T1 + H1 + D1 trên fixture/development.
2. **E1** chỉ được chạy sau khi H1/D1/B2 đạt GO theo protocol hiện hành.
3. T1 v1.1 còn yêu cầu dry-run paired refit-bootstrap, ước lượng runtime và
   sealed runner tổng thể trước E1. Những mục này thuộc cổng thực thi
   confirmatory, không được suy ra từ baseline preflight không-fit này.
4. Preflight không chứng minh SOTA, không chứng minh reproduction của paper,
   không cho phép retune và không mở sealed evaluation.

Do đó, câu chính xác để bàn giao là: **“D1 baseline provenance/preflight đã
pass; evaluation vẫn sealed và E1 chưa được mở.”**

## Execution contract đã bổ sung

Các yêu cầu vận hành trước E1 hiện được preflight riêng tại
[`D1_CONFIRMATORY_PREFLIGHT.md`](D1_CONFIRMATORY_PREFLIGHT.md). Entrypoint
[`run_confirmatory_preflight.py`](../../scripts/validation/run_confirmatory_preflight.py)
khóa run ID, kiểm full seed `0..9`/budget/method job plan, chạy paired
refit-bootstrap 2.000 draw trên fixture, ghi runtime/memory và kiểm verified
resume. Những artifact này vẫn là development/fixture evidence; chúng không
mở evaluation và không thay thế B2 review hoặc E1 execution.

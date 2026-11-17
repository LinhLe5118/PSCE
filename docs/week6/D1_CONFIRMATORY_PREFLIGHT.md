# D1 — Confirmatory execution contract preflight

## Trạng thái

Đã implement và kiểm trên fixture; đường development thật cũng đã được nối.
Đây là preflight an toàn trước E1; nó không mở evaluation và không đưa kết quả
fixture/development vào claim nghiên cứu.

- Run ID bị khóa: `week6_oehv4e_confirmatory_v1_1`.
- Output chuẩn: `artifacts/week6/oehv4e_confirmatory_v1_1/`.
- Bootstrap mặc định: 2.000 paired base-MDP draws, PCG64 seed `20260921`.
- Paired contrast lưu basic 95% CI và centered two-sided p theo margin đã khóa;
  fixture kiểm H1 margin `0.005` nhưng không dùng làm kết quả nghiên cứu.
- Mỗi job/draw đo wall-clock và peak Python heap bằng `tracemalloc`.
- Lỗi job hoặc draw dừng ngay; partial checkpoint được giữ nguyên.
- Development train/score gọi trực tiếp `cfqi_tabular`, `iql_tabular` và
  `behavior_cloning_tabular`, sau đó chấm bằng exact policy scorer; không còn
  dùng identity callback khi truyền `--execute-jobs`.
- Development refit-bootstrap dùng panel đã hash trong repo, refit hai return
  component `B2_OEH3` và `B2_OEH3_LCOEH4`, chọn ridge trong inner 4-fold và
  chọn learner trong outer 5-fold ở từng draw. Mặc định dry-run dùng 15 base
  MDP để ước lượng capacity, vẫn chạy đủ 2.000 draw.
- Source pin của run là T1 manifest `v1.1.0-r1`; verifier từ chối manifest khác.

Entrypoint là
`scripts/validation/run_confirmatory_preflight.py`; control logic tái sử dụng
tại `evaluation/confirmatory.py`.

## Đã đóng các finding

1. `build_job_plan` tạo toàn bộ cell `(instance, behavior, budget, seed)` và
   mở rộng đồng thời cho CFQI, IQL, BC. Với protocol hiện hành, mỗi cell phải
   có seed `0..9` và budget `{1,2,5,10,20,50}`; ba method bắt buộc cùng
   `buffer_file`, `buffer_sha256` và `size_multiplier`.
2. `run_missing_jobs_only` ghi `execution_plan.json`, `run_metadata.json`,
   checkpoint ledger và một output riêng cho từng job. Resume bị từ chối nếu
   run ID, plan hash, source/input hash, job key, output path hoặc output hash
   không khớp. Ledger được checkpoint theo batch; nếu process chết sau khi
   output job đã durable nhưng trước lần ghi ledger kế tiếp, resume tự
   reconcile job file bằng `job_sha256` và content hash rồi chỉ chạy phần còn
   thiếu.
3. `run_paired_refit_bootstrap` resample cluster trong từng stratum, gọi
   callback refit đúng một lần cho mỗi draw và trả tất cả endpoint từ cùng draw.
   Checkpoint có hash cho từng draw; lỗi schema/numerical/callback là
   `failed_stop_on_error`.
4. Runtime và memory evidence được ghi trong `execution_summary.json`,
   `bootstrap_summary.json` và `preflight_report.json`; mỗi bundle còn có
   `artifact_manifest.json` để kiểm hash các output chính.

## Chạy fixture

```powershell
python scripts/validation/run_confirmatory_preflight.py --fixture
python scripts/validation/run_confirmatory_preflight.py --fixture --resume
```

Fixture tạo 2 nhóm, 6 budget, 10 seed và 3 method = 360 job keys; bootstrap
chạy đủ 2.000 draw. `--resume` chỉ kiểm tra lại artifact đã hoàn tất và không
fit lại job/draw nào.

## Chạy contract-only trên development

```powershell
python scripts/validation/run_confirmatory_preflight.py `
  --dataset-dir <development-bundle> `
  --output-dir artifacts/week6/oehv4e_confirmatory_v1_1
```

Lệnh này xác minh manifest/buffer development và ghi full job plan; mặc định
không chạy learner hay oracle.

## Probe train/score thật và refit-bootstrap thật

Probe dưới đây chạy một logical cell với đủ ba method (3 job), dùng learner và
buffer development thật; `--real-bootstrap` chạy đủ 2.000 paired draw và ghi
runtime/memory. `--train-score-job-limit` chỉ giới hạn probe, không thay đổi
full job plan đã ghi trong report.

```powershell
python scripts/validation/run_confirmatory_preflight.py `
  --dataset-dir datasets/offline_rl_development_2026-08-22 `
  --output-dir tmp/confirmatory_development_real/oehv4e_confirmatory_v1_1 `
  --execute-jobs `
  --train-score-job-limit 1 `
  --real-bootstrap
```

Resume phải dùng đúng output, đúng limit và đúng cờ; mọi thay đổi source/input
hash hoặc checkpoint output sẽ bị từ chối. E1 vẫn chỉ được mở sau H1/D1/B2 GO
và execution gate của T1 được bật.

## Negative tests

```powershell
python -m pytest -q tests/test_confirmatory_execution.py
```

Các test kiểm: thiếu seed/cell, dừng ở lỗi đầu tiên, resume chỉ chạy job còn
thiếu, từ chối output hash bị sửa, từ chối bootstrap hash bị sửa, callback
development refit thật và paired refit đủ 2.000 draw trên fixture.

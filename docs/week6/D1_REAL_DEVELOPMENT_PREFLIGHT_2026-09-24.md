# D1 — Real development train/score and refit-bootstrap dry-run

Run ID: `week6_oehv4e_confirmatory_v1_1`
Source pin: T1 manifest `v1.1.0-r1`
T1 manifest SHA-256: `28fe4fd5f155c5c2717f0ea9d69b0ad713f1ed5fb752946e4190181aba7f613c`
Execution source commit: `7165836d68c3ad52a9119489f9fd672f4c0aacb0`
Completed at: `2026-09-24T16:07:57Z` (report timestamp after verified resume)

This is a development capacity run. It did not open the sealed evaluation
cohort and it is not a confirmatory estimate.

## Train/score probe

The probe used one complete logical cell from the development bundle, with the
same buffer, budget, and seed sent to all three registered methods:

| Method | Jobs | Status | Normalized return |
|---|---:|---|---:|
| CFQI | 1 | observed | 0.574837 |
| IQL | 1 | observed | 0.574837 |
| BC | 1 | observed | 0.574837 |

The callback is the registered learner implementation plus the exact policy
scorer; it is not an identity/oracle callback. The full development plan was
also validated: 75 groups, 4,500 logical prefixes, 13,500 jobs, seeds `0..9`,
budgets `{1,2,5,10,20,50}`, and shared buffer/budget identity across CFQI,
IQL, and BC.

Probe runtime was `0.2057 s` for three jobs, with peak Python heap
`1,782,511 bytes`. The runtime is a probe measurement, not a claim that all
13,500 jobs have been executed.

## Paired refit-bootstrap

The real callback refit both frozen return components (`B2_OEH3` and
`B2_OEH3_LCOEH4`) with inner 4-fold ridge selection, outer 5-fold learner
selection, and paired base-MDP resampling. The capacity slice used the
registered 15 development MDPs (225 learner rows) and completed the full
2,000 draws with PCG64 seed `20260921`.

| Quantity | Measured value |
|---|---:|
| Draws completed | 2,000 / 2,000 |
| Wall runtime | 218.3692 s |
| Mean draw runtime | 0.10918 s |
| Peak Python heap | 265,720 bytes |
| Observed v4E endpoint | 0.729490 |
| Observed fold-best-fixed endpoint | 0.722818 |
| Observed paired difference | +0.006672 |
| Basic bootstrap 95% CI | [-0.019707, +0.099774] |
| Margin-centred p | 0.9470 |

The CI and p-value above are development/capacity evidence only. Memory is the
peak Python heap measured by `tracemalloc`, not process RSS. The output was
rerun with `--resume` and passed hash verification without re-fitting
completed jobs or draws. The output's source map contains 160 entries with
zero mismatches against the current checkout; all seven files in its output
manifest match their recorded SHA-256 hashes. The three historical source
mismatches were resolved by this fresh run. Attempting `--resume` on the
previous, pre-rebase output now raises `ConfirmatoryIntegrityError` with
`Run/source/config/input hash mismatch on resume.`

Selected local artifact hashes from the verified new run:

| File | SHA-256 |
|---|---|
| `run_metadata.json` | `03f274528430f544fc394b3c40708146b90f4f50fedb262cdd2ad764bb1b365b` |
| `paired_refit_bootstrap.json` | `e1ab328872145361be954682b9df82811c6ce6723d72688a12db3137f49cfec2` |
| `preflight_report.json` | `05e0af89d51cbd05d29d8a194a778f209c9bc2147e7035f4d0330dfc55090a86` |
| `artifact_manifest.json` | `7c59743a8be9f606220c6f06c2dd4f47975ff94086a4c37107af0141a1a15d32` |

The full checkpoint and job outputs remain under the ignored local `tmp/`
directory. The table is a portable integrity record; reproducing the run
requires the separately verified development bundle.

## Reproduction

```powershell
python scripts/validation/run_confirmatory_preflight.py `
  --dataset-dir ../Devoted/datasets/offline_rl_development_2026-08-22 `
  --output-dir tmp/confirmatory_development_real_v1_1_r1_current/oehv4e_confirmatory_v1_1 `
  --execute-jobs `
  --train-score-job-limit 1 `
  --real-bootstrap

python scripts/validation/run_confirmatory_preflight.py `
  --dataset-dir ../Devoted/datasets/offline_rl_development_2026-08-22 `
  --output-dir tmp/confirmatory_development_real_v1_1_r1_current/oehv4e_confirmatory_v1_1 `
  --execute-jobs `
  --train-score-job-limit 1 `
  --real-bootstrap `
  --resume
```

Input hashes recorded by the run include development manifest
`311c8b12f84c21519f3c0ef11b4606cb596e610cc368c2797e0b2268250e5ac2`, base
features `834a2a57b4e20ec21f3d5d84b0fa0cd024fe72b4ca46a3ec44cd1c451f22e53b`,
LC-OEH4 features
`d69d7ae38f936936c84e528a9ae553fcc47162bbabc580824e0f6f0732ca343b`, and
return labels
`084d7519946b6bc82f7c8697cb9cdb3852c2e34bc6eb546d7fd88cce1e1f9c02`.

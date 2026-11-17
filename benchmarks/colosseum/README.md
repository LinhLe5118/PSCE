# Colosseum external panel: driver code

Pipeline that ran the exploratory (32 MDPs, 13/10/2026) and confirmatory (72 MDPs, 14/10/2026) Colosseum panels on a secondary machine. The scripts are archived **as run**. The only edit is `OEH_WORKSPACE` in `driver/common.py`; its default keeps the original behavior.

- Results: [`artifacts/week8/colosseum_confirm_2026-10-14/`](../../artifacts/week8/colosseum_confirm_2026-10-14/) and [`artifacts/week8/colosseum_exploratory_2026-10-13/`](../../artifacts/week8/colosseum_exploratory_2026-10-13/).
- Code review, corrected numbers and post-hoc analyses: [`docs/week8/COLOSSEUM_EXTERNAL_PANEL_2026-10-14.md`](../../docs/week8/COLOSSEUM_EXTERNAL_PANEL_2026-10-14.md).
- The runner's own handover notes: [`HANDOVER_README_2026-10-14.md`](HANDOVER_README_2026-10-14.md) and [`KE_HOACH_THUC_NGHIEM_2026-10-14.md`](KE_HOACH_THUC_NGHIEM_2026-10-14.md). Where they disagree with the review document, the review document is current. The disagreements are the max-t rule and the count of zero-draw OEH-B groups.

## Layout expected by the scripts

```
<trial>/
  driver/  driver_tools/          # this folder
  colosseum/src                   # MichelangeloConserva/Colosseum @ 981b344 (Python 3.9 only)
  py39/  py312/                   # conda environments
  data/                           # kernels and bundles (regenerated, not archived)
  results/  logs/                 # outputs; results/confirm for COLOSSEUM_RUN=confirm
```

Set `OEH_WORKSPACE` to a Devoted checkout that has the published cohort artifacts (LFS hydrated). Otherwise the scripts use `<trial>/workspace`.

## Rerun

1. Environments:
   - Python 3.9 with `pip install -e colosseum/src`.
   - Python 3.12 with the repository pins: numpy 2.5.1, scipy 1.18.0, pandas 2.3.3, scikit-learn, xgboost 3.3.0.
2. Thread settings: `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Without these, parallel OEH-B workers oversubscribe BLAS. On the original run one group stalled for more than 1.5 hours.
3. Check parity first: `python driver/forecast.py reproduce` must reproduce 3,600/3,600 sealed cohort-7 predictions within 1e-6.
4. Run `bash driver/pipeline_confirm.sh`. It refuses to start without `results/confirm/protocol_colosseum.json`.
   - Order: export → hardness → bundle → OEH-B → features → seal → labels → score.
   - Measured time: about 3.4 hours on 14 workers.
5. Optional (amendment 1, post hoc): `python driver/comparators_colosseum.py 14`, then `python driver/forecast_comparators.py`.

## Known deviations

- `score_colosseum.py` and `forecast_comparators.py` restudentize every bootstrap draw. The team rule divides by the original standard error. `scripts/analysis/colosseum_posthoc_20261014.py` reports both.
- Colosseum rewards are stochastic. The export keeps only the mean reward `R(s, a)`, with γ = 0.95, horizon 100 and no terminal states.
- 42/360 confirmatory OEH-B groups accepted no posterior draw and use the "censored" fallback. One more group returned 9/64 draws.
- The IQL expectile fallback (`driver/expectile_fix.py`) was used in feature groups and in label jobs on 11 SimpleGrid MDPs.

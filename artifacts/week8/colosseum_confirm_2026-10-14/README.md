# Colosseum confirmatory panel (72 MDPs), as delivered on 14/10/2026

The files are the secondary machine's `results/confirm/` directory, unchanged.

| Content | Location |
|---|---|
| Pre-outcome OEH-B groups | `oeh_b_groups/` (formerly `results/oehb/confirm_bundle/`) |
| Logs of both Colosseum runs | `run_logs/` |
| Runner's report (Vietnamese) | `BAO_CAO_XAC_NHAN_2026-10-14.md` |

Two statements in `BAO_CAO_XAC_NHAN_2026-10-14.md` are corrected by the review in [`docs/week8/COLOSSEUM_EXTERNAL_PANEL_2026-10-14.md`](../../../docs/week8/COLOSSEUM_EXTERNAL_PANEL_2026-10-14.md):

1. **Max-t rule.** The bounds in `score/` and `score_comparators/` restudentize each bootstrap draw. They do not use the team rule from cohort 7. Under the team rule the primary verdict is still GO. The log-loss lower bound against B2 matched falls from +0.0049 to +0.0003.
2. **Zero-draw groups.** The report states that 0/360 groups lack posterior draws. In fact 42/360 groups accepted no draw (`sampling.draws_returned == 0`), and one group returned 9/64.

Post-hoc analyses of this panel are in [`../colosseum_posthoc_2026-10-14/`](../colosseum_posthoc_2026-10-14/).

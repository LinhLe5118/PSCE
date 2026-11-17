"""Post hoc MDP-bootstrap intervals for the ablation contrasts already in the sealed scores.

Uses the sealed forecasts and recorded labels only; no head is refitted. Each contrast is
(score of the variant without the block) minus (score of the reference), averaged per MDP
first, so a positive value means that removing the block makes the forecast worse.

    python scripts/analysis/posthoc_20261028/ablation_ci.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts"
OUT = Path(__file__).resolve().parent / "out"
DRAWS, SEED = 10_000, 20261028
GROUPS = {
    "Test 1 internal": ("oeh_v2_sealed_test_2026-10-14", ["internal"], "brier", "log_loss"),
    "Test 1 external": ("oeh_v2_sealed_test_2026-10-14", ["colosseum2029", "spibb2", "forest"], "brier", "log_loss"),
    "Test 2": ("oeh_fifth_panel_2026-10-15", ["frozenlake", "gambler"], "brier", "log_loss"),
    "Test 3": ("oeh_panel6_2026-10-15", ["mdpp", "labmaze"], "ibs", "ll"),
}
# (label, variant, reference) per test family
CONTRASTS = {
    "Test 1": [
        ("OEH: remove learner simulation", "v2_no_direct_oehb", "v2"),
        ("OEH: remove shift gate", "oeh_final", "v2"),
        ("Full head: remove learner simulation", "oeh_without_direct_oehb", "oeh_final"),
        ("Full head: remove planning risk (EH)", "oeh_without_oehh_risk", "oeh_final"),
        ("Full head: remove pilot diagnostics", "oeh_without_legacy_diagnostics", "oeh_final"),
        ("B2: add planning risk (EH) alone", "b2_matched", "b2_plus_H"),
        ("B2: add learner simulation alone", "b2_matched", "b2_plus_P"),
    ],
    "Test 2/3": [
        ("OEH: remove planning risk (EH)", "oeh_without_horizon", "oeh"),
        ("OEH: remove shift gate", "oeh_full_head", "oeh"),
    ],
}


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for group, (folder, panels, ibs, ll) in GROUPS.items():
        d = pd.read_csv(ART / folder / "scored.csv.gz")
        if "panel" in d:
            d = d[d.panel == "primary"]
        d = d[d.source_panel.isin(panels)]
        per_mdp = d.groupby(["method", "mdp_id"])[[ibs, ll]].mean()
        mdps = per_mdp.index.get_level_values("mdp_id").unique()
        index = rng.integers(0, len(mdps), size=(DRAWS, len(mdps)))
        for label, variant, reference in CONTRASTS["Test 1" if group.startswith("Test 1") else "Test 2/3"]:
            for metric, column in (("IBS", ibs), ("LL", ll)):
                a = per_mdp.loc[variant, column].reindex(mdps).to_numpy()
                b = per_mdp.loc[reference, column].reindex(mdps).to_numpy()
                diff = a - b
                lo, hi = np.percentile(diff[index].mean(1), [2.5, 97.5])
                verdict = "block helps" if lo > 0 else ("block hurts" if hi < 0 else "no clear difference")
                rows.append(dict(group=group, contrast=label, metric=metric, reference=b.mean(), diff=diff.mean(),
                                 rel=100 * diff.mean() / b.mean(), lo=lo, hi=hi, verdict=verdict, mdps=len(mdps)))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(out.round({"reference": 4, "diff": 4, "rel": 1, "lo": 4, "hi": 4}).to_string(index=False))
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "ablation_ci.csv", index=False)


if __name__ == "__main__":
    main()

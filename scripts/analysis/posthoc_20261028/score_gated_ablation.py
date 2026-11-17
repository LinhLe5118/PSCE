"""Score the gated ablation of gated_ablation.py and give MDP-bootstrap intervals.

Checks that the refitted "without planning risk" law equals the sealed oeh_without_horizon
forecasts of Tests 2 and 3, scores every variant with the recorded labels (IBS over the six
thresholds, log loss floored at 1e-15) and compares each removal with OEH. A positive
difference means that removing the block makes the forecast worse.

    python scripts/analysis/posthoc_20261028/score_gated_ablation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "out" / "gated_ablation"
P = [f"p_{i}" for i in range(7)]
KEYS = ["mdp_id", "policy_family", "algorithm"]
TESTS = {"Test 1": ("oeh_v2_sealed_test_2026-10-14", "v2"), "Test 2": ("oeh_fifth_panel_2026-10-15", "oeh"),
         "Test 3": ("oeh_panel6_2026-10-15", "oeh")}
NAMES = {"no_P": "remove learner simulation", "no_H": "remove planning risk (EH)", "no_L": "remove pilot diagnostics"}
DRAWS, SEED = 10_000, 20261028


def main() -> None:
    pred = pd.read_csv(OUT / "predictions.csv.gz")
    labels = []
    for test, (folder, method) in TESTS.items():
        s = pd.read_csv(ROOT / "artifacts" / folder / "scored.csv.gz")
        if "panel" in s:
            s = s[s.panel == "primary"]
        labels.append(s[s.method == method][KEYS + ["event", "source_panel"]].assign(test=test))
    d = pred.merge(pd.concat(labels), on=KEYS + ["test", "source_panel"], validate="many_to_one")

    for folder, panel in (("oeh_fifth_panel_2026-10-15", "gambler"), ("oeh_panel6_2026-10-15", "mdpp")):
        sealed = pd.read_csv(ROOT / "artifacts" / folder / panel / "predictions_preoutcome.csv")
        sealed = sealed[sealed.method == "oeh_without_horizon"]
        both = d[(d.method == "no_H") & (d.source_panel == panel)].merge(sealed[KEYS + P], on=KEYS, suffixes=("", "_s"))
        print(f"no_H vs sealed oeh_without_horizon, {panel}: max |diff| = "
              f"{np.abs(both[P].to_numpy() - both[[p + '_s' for p in P]].to_numpy()).max():.2e}")

    p, y = d[P].to_numpy(), d.event.to_numpy()
    d["ibs"] = ((np.cumsum(p, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)
    d["ll"] = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))
    d["group"] = np.where(d.test != "Test 1", d.test,
                          np.where(d.source_panel == "internal", "Test 1 internal", "Test 1 external"))
    per = d.groupby(["group", "method", "mdp_id"])[["ibs", "ll"]].mean()

    rng = np.random.default_rng(SEED)
    rows = []
    for group in ["Test 1 internal", "Test 1 external", "Test 2", "Test 3"]:
        x = per.loc[group]
        mdps = x.loc["OEH"].index
        index = rng.integers(0, len(mdps), (DRAWS, len(mdps)))
        for method, label in NAMES.items():
            row = {"group": group, "contrast": label}
            for metric in ("ibs", "ll"):
                a = x.loc[method, metric].reindex(mdps).to_numpy()
                b = x.loc["OEH", metric].reindex(mdps).to_numpy()
                lo, hi = np.percentile((a - b)[index].mean(1), [2.5, 97.5])
                verdict = "helps" if lo > 0 else ("hurts" if hi < 0 else "no clear difference")
                row.update({f"{metric}_without": a.mean(), f"{metric}_oeh": b.mean(), f"{metric}_diff": (a - b).mean(),
                            f"{metric}_lo": lo, f"{metric}_hi": hi, f"{metric}_verdict": verdict})
            rows.append(row)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))
    out.to_csv(OUT / "gated_ablation_ci.csv", index=False)
    d.to_csv(OUT / "scored.csv.gz", index=False)


if __name__ == "__main__":
    main()

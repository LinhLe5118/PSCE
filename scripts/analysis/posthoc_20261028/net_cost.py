"""Post hoc net-cost study of data-collection strategies (no new learner runs).

    python scripts/analysis/posthoc_20261028/net_cost.py

Uses the sealed forecasts and recorded labels of the three final tests. Costs are in
budget units (|S||A| transitions) and include the x10 pilot whenever a strategy uses it.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts"
OUT = Path(__file__).resolve().parent / "out"
BUDGETS = np.array([1, 2, 5, 10, 20, 50])
TAU = 0.8                      # registered decision threshold
PILOT = 10
GROUPS = {
    "Test 1 internal": ("oeh_v2_sealed_test_2026-10-14", ["internal"], {"OEH": "v2", "Robust head": "legacy", "B2 matched": "b2_matched"}),
    "Test 1 external": ("oeh_v2_sealed_test_2026-10-14", ["colosseum2029", "spibb2", "forest"], {"OEH": "v2", "Robust head": "legacy", "B2 matched": "b2_matched"}),
    "Test 2": ("oeh_fifth_panel_2026-10-15", ["frozenlake", "gambler"], {"OEH": "oeh", "Robust head": "robust", "B2 matched": "b2_matched"}),
    "Test 3": ("oeh_panel6_2026-10-15", ["mdpp", "labmaze"], {"OEH": "oeh", "Robust head": "robust", "B2 matched": "b2_matched"}),
}


def strategy_costs(E, F=None, kind=None):
    """Per-row data collected D, training rounds R, failures and misses."""
    n = len(E)
    finite = E <= 5
    bE = np.where(finite, BUDGETS[np.minimum(E, 5)], 50)
    if kind == "collect_x50":                       # collect x50, train once, no pilot
        return dict(D=np.full(n, 50.0), R=np.ones(n), fail=(~finite).astype(float), miss=np.zeros(n))
    if kind == "sequential":                        # x1, x2, ... with a perfect check after each step
        rounds = np.where(finite, E + 1, 6)
        return dict(D=bE.astype(float), R=rounds.astype(float), fail=np.zeros(n), miss=np.zeros(n))
    if kind == "pilot_then_check":                  # train on the pilot, then x20, x50 with a perfect check
        D = np.where(E <= 3, 10, np.where(E == 4, 20, 50)).astype(float)
        R = np.where(E <= 3, 1, np.where(E == 4, 2, 3)).astype(float)
        return dict(D=D, R=R, fail=np.zeros(n), miss=np.zeros(n))
    if kind == "oracle":                            # knows the label; the pilot is already collected
        return dict(D=np.where(finite, np.maximum(PILOT, bE), PILOT).astype(float),
                    R=finite.astype(float), fail=np.zeros(n), miss=np.zeros(n))
    # forecast rule: smallest budget with F >= TAU, otherwise skip
    ok = F >= TAU
    select = ok.any(1)
    j = np.where(select, ok.argmax(1), -1)
    D = np.where(select, np.maximum(PILOT, BUDGETS[np.maximum(j, 0)]), PILOT).astype(float)
    fail = (select & (E > j)).astype(float)
    miss = (~select & finite).astype(float)
    return dict(D=D, R=select.astype(float), fail=fail, miss=miss)


rows = []
for group, (folder, panels, names) in GROUPS.items():
    d = pd.read_csv(ART / folder / "scored.csv.gz")
    if "panel" in d:
        d = d[d.panel == "primary"]
    d = d[d.source_panel.isin(panels)]
    base = d[d.method == names["OEH"]].sort_values(["mdp_id", "policy_family", "algorithm"]).reset_index(drop=True)
    E = base.event.to_numpy()
    plans = {"Collect x50, train once": strategy_costs(E, kind="collect_x50"),
             "Step-by-step, perfect check": strategy_costs(E, kind="sequential"),
             "Pilot, then step-by-step check": strategy_costs(E, kind="pilot_then_check"),
             "Oracle (knows the label)": strategy_costs(E, kind="oracle")}
    for label, method in names.items():
        m = d[d.method == method].sort_values(["mdp_id", "policy_family", "algorithm"]).reset_index(drop=True)
        assert (m[["mdp_id", "policy_family", "algorithm"]].values == base[["mdp_id", "policy_family", "algorithm"]].values).all()
        P = m[[f"p_{i}" for i in range(7)]].to_numpy()
        P = P / P.sum(1, keepdims=True)
        plans[f"Pilot + {label} rule"] = strategy_costs(E, F=np.cumsum(P, 1)[:, :6])
    for name, c in plans.items():
        frame = pd.DataFrame({"mdp": base.mdp_id, **c})
        rows.append(frame.assign(group=group, strategy=name))

df = pd.concat(rows)
per_mdp = df.groupby(["group", "strategy", "mdp"])[["D", "R", "fail", "miss"]].mean().reset_index()

def summary(lam):
    t = per_mdp.assign(cost=per_mdp.D + lam * (per_mdp.fail + per_mdp.miss))
    return t.groupby(["group", "strategy"])[["D", "R", "fail", "miss", "cost"]].mean()

pd.set_option("display.width", 200)
s = summary(100)
s[["fail", "miss"]] *= 100
print(s.round(2).to_string())
for lam in (50, 200):
    print(f"\nTotal cost with failure penalty {lam}:")
    print(summary(lam).cost.unstack(0).round(2).to_string())

# MDP-cluster bootstrap of total-cost differences at penalty 100
rng = np.random.default_rng(20261027)
t = per_mdp.assign(cost=per_mdp.D + 100 * (per_mdp.fail + per_mdp.miss))
print("\nDifference in mean total cost (comparator minus OEH), 95% MDP-bootstrap interval:")
for group in GROUPS:
    w = t[t.group == group].pivot(index="mdp", columns="strategy", values="cost")
    for comp in ["Collect x50, train once", "Pilot, then step-by-step check", "Pilot + B2 matched rule", "Pilot + Robust head rule"]:
        diff = (w[comp] - w["Pilot + OEH rule"]).to_numpy()
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(4000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {group:16s} {comp:32s} {diff.mean():8.2f}  [{lo:7.2f}, {hi:7.2f}]  n={len(diff)}")
OUT.mkdir(exist_ok=True)
df.to_csv(sys.argv[1] if len(sys.argv) > 1 else OUT / "net_cost_rows.csv.gz", index=False)

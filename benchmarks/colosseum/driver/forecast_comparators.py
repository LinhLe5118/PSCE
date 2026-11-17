"""Cohort-7 comparator set on the confirmatory Colosseum MDPs (post-hoc amendment).

Runs the team's ``all_forecasts`` (same fields, fitting and meta-training panel
as the sealed cohort-7 run) on the Colosseum query, checks that its OEH and B2
forecasts equal the sealed ones, then scores every comparator against OEH with
the cohort-7 rule (simultaneous max-t one-sided 95% lower bounds, 10,000
whole-MDP bootstrap draws). Comparators whose inputs are missing are listed as
pending and left out of the joint family.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import forecast as FC
from common import OUT, RES, RUN_BUNDLE_NAME, load_json, sha256, write_json

R, C = FC.R, FC.C
KEYS = R.KEYS
P = [f"p_{i}" for i in range(7)]
seal = load_json(RES / "colosseum_preoutcome_seal.json")
if seal["predictions_sha256"] != sha256(RES / "colosseum_predictions_preoutcome.csv"):
    raise SystemExit("sealed predictions changed")

query = FC.colosseum_query()
query = query.merge(pd.read_csv(RES / "oracle_descriptors_preoutcome.csv"), on="mdp_id", validate="many_to_one")
query = query.merge(pd.read_csv(RES / "published_preoutcome.csv"), on=["mdp_id", "policy_family"],
                    validate="many_to_one")
original = RES / "tqsaco_original_preoutcome.csv"
if original.exists():
    query = query.merge(pd.read_csv(original)[["mdp_id", "policy_family", "tq_original", "saco_original"]],
                        on=["mdp_id", "policy_family"], validate="many_to_one")
pending = [name for name, fields in R.FIELD_SETS.items() if not set(fields) <= set(query.columns)]
for name in pending:
    del R.FIELD_SETS[name]
print("pending comparators (inputs missing):", pending, flush=True)

train = FC.training_panel()
predictions, _ = R.all_forecasts(train, query, R.raw_oeh_b(OUT / "oehb" / RUN_BUNDLE_NAME))
predictions.to_csv(RES / "comparator_predictions.csv", index=False)

sealed = pd.read_csv(RES / "colosseum_predictions_preoutcome.csv")
check = {}
for method in ("oeh_final", "b2_matched"):
    j = predictions[predictions.method == method].merge(sealed[sealed.method == method], on=KEYS,
                                                         suffixes=("", "_s"), validate="one_to_one")
    check[method] = float(np.abs(j[P].to_numpy() - j[[f"{p}_s" for p in P]].to_numpy()).max())
print("max |diff| vs sealed:", check, flush=True)
if max(check.values()) > 1e-9:
    raise SystemExit("all_forecasts does not reproduce the sealed forecasts")

lab = pd.read_csv(RES / "colosseum_labels.csv")
m = predictions.merge(lab[KEYS + ["event"]], on=KEYS, validate="many_to_one")
Pm, e = m[P].to_numpy(float), m.event.to_numpy(int)
m["ibs"] = ((np.cumsum(Pm, 1)[:, :6] - (e[:, None] <= np.arange(6)).astype(float)) ** 2).mean(1)
m["ll"] = -np.log(np.clip(Pm[np.arange(len(e)), e], 1e-15, 1))
mdp = m.groupby(["method", "mdp_id"])[["ibs", "ll"]].mean().reset_index()
wide = mdp.pivot(index="mdp_id", columns="method")
summary = mdp.groupby("method")[["ibs", "ll"]].mean().sort_values("ibs")

comparators = [c for c in R.COMPARATORS if c not in pending]
names = [(b, met) for b in comparators for met in ("ibs", "ll")]
D = np.column_stack([wide[(met, b)] - wide[(met, "oeh_final")] for b, met in names])
n = len(D)
mu, se = D.mean(0), D.std(0, ddof=1) / np.sqrt(n)
rng = np.random.default_rng(20261004)  # cohort-7 bootstrap seed
T = []
for _ in range(10_000):
    Db = D[rng.integers(0, n, n)]
    T.append(((Db.mean(0) - mu) / (Db.std(0, ddof=1) / np.sqrt(n))).max())
crit = float(np.quantile(T, 0.95))
rows = []
for i, (b, met) in enumerate(names):
    base = wide[(met, b)].mean()
    rows.append({"tier": 1 if b in R.TIER1 else 2, "baseline": b,
                 "metric": "integrated_brier" if met == "ibs" else "seven_class_log_loss", "mdps": n,
                 "baseline_mean": base, "oeh_mean": wide[(met, "oeh_final")].mean(), "relative_gain": mu[i] / base,
                 "simultaneous_one_sided_lower_95": mu[i] - crit * se[i],
                 "mdps_improved": int((D[:, i] > 0).sum()), "mdps_worse": int((D[:, i] < 0).sum())})
table = pd.DataFrame(rows)
go = lambda t: "GO" if (t.simultaneous_one_sided_lower_95 > 0).all() else "NO-GO"
verdict = {"tier1": go(table[table.tier == 1]),
           "tier2_available": go(table[table.tier == 2]) if (table.tier == 2).any() else None,
           "all_available": go(table)}
OUT_DIR = RES / "score_comparators"
OUT_DIR.mkdir(exist_ok=True)
summary.to_csv(OUT_DIR / "summary.csv")
table.to_csv(OUT_DIR / "contrasts.csv", index=False)
mdp.to_csv(OUT_DIR / "mdp_scores.csv", index=False)
write_json(OUT_DIR / "verdict.json", {
    "amendment_sha256": sha256(RES / "protocol_colosseum_amendment_1_comparators.json"),
    "pending_comparators": pending, "joint_contrasts": len(names), "critical": crit, "mdps": n,
    "verdict": verdict, "reproduces_sealed_max_abs_diff": check,
    "predictions_sha256": sha256(RES / "comparator_predictions.csv")})
pd.set_option("display.width", 220)
print(summary.round(4).to_string(), "\n")
print(table.round(4).to_string(index=False), "\n")
print("verdict:", verdict, "| joint contrasts", len(names), "| critical", round(crit, 3))

"""Score the sealed Colosseum forecasts against the learner labels.

Primary (same rule as the paper): integrated Brier over six thresholds and
seven-class log loss, averaged within MDP then across MDPs; paired whole-MDP
bootstrap (10,000) with simultaneous max-t one-sided 95% lower bounds.
Registered contrasts: OEH vs B2 matched, OEH vs B2 + Colosseum hardness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RES as OUT, load_json, sha256, write_json

seal = load_json(OUT / "colosseum_preoutcome_seal.json")
if seal["predictions_sha256"] != sha256(OUT / "colosseum_predictions_preoutcome.csv"):
    raise SystemExit("Prediction file changed after the seal")
pred = pd.read_csv(OUT / "colosseum_predictions_preoutcome.csv")
lab = pd.read_csv(OUT / "colosseum_labels.csv")
KEYS = ["mdp_id", "policy_family", "algorithm"]
m = pred.merge(lab[KEYS + ["event"]], on=KEYS, validate="many_to_one")
P = m[[f"p_{i}" for i in range(7)]].to_numpy(float)
e = m.event.to_numpy(int)
m["ibs"] = ((np.cumsum(P, 1)[:, :6] - (e[:, None] <= np.arange(6)).astype(float)) ** 2).mean(1)
m["ll"] = -np.log(np.clip(P[np.arange(len(e)), e], 1e-15, 1))
m["expected_event"] = P @ np.arange(7)
m["family"] = m.mdp_id.str.extract(r"colosseum_(.+)_\d+$")[0]

mdp = m.groupby(["method", "mdp_id"])[["ibs", "ll"]].mean().reset_index()
summary = mdp.groupby("method")[["ibs", "ll"]].mean().rename(columns={"ibs": "integrated_brier",
                                                                      "ll": "seven_class_log_loss"})
wide = mdp.pivot(index="mdp_id", columns="method")
contrasts = [("b2_matched", "oeh_final"), ("b2_plus_colosseum", "oeh_final")]
names = [(b, c, met) for b, c in contrasts for met in ("ibs", "ll")]
D = np.column_stack([wide[(met, b)] - wide[(met, c)] for b, c, met in names])
n = len(D)
mu, se = D.mean(0), D.std(0, ddof=1) / np.sqrt(n)
rng = np.random.default_rng(20261014)
T = []
for _ in range(10_000):
    Db = D[rng.integers(0, n, n)]
    T.append(((Db.mean(0) - mu) / (Db.std(0, ddof=1) / np.sqrt(n))).max())
crit = float(np.quantile(T, 0.95))
rows = []
for i, (b, c, met) in enumerate(names):
    base = wide[(met, b)].mean()
    rows.append({"baseline": b, "candidate": c, "metric": "integrated_brier" if met == "ibs" else "seven_class_log_loss",
                 "mdps": n, "baseline_mean": base, "candidate_mean": wide[(met, c)].mean(),
                 "relative_gain": mu[i] / base, "mean_absolute_improvement": mu[i],
                 "simultaneous_one_sided_lower_95": mu[i] - crit * se[i],
                 "mdps_improved": int((D[:, i] > 0).sum()), "mdps_worse": int((D[:, i] < 0).sum())})
table = pd.DataFrame(rows)
verdict = {k: ("GO" if (table[table.baseline == k].simultaneous_one_sided_lower_95 > 0).all() else "NO-GO")
           for k in ("b2_matched", "b2_plus_colosseum")}

by_family = m.groupby(["family", "method", "mdp_id"])[["ibs", "ll"]].mean().groupby(["family", "method"]).mean()
by_learner = m.groupby(["algorithm", "method", "mdp_id"])[["ibs", "ll"]].mean().groupby(["algorithm", "method"]).mean()
censor = lab.groupby("algorithm").event.apply(lambda s: float((s == 6).mean()))
dist = lab.event.value_counts().sort_index()

# Can each forecast rank the three learners within the same MDP-behavior pair?
within = []
for method, g in m.groupby("method"):
    rho = [spearmanr(x.expected_event, x.event).statistic for _, x in g.groupby(["mdp_id", "policy_family"])
           if x.event.nunique() > 1]
    within.append({"method": method, "groups_with_learner_differences": len(rho),
                   "mean_within_group_spearman": float(np.nanmean(rho)) if rho else None})
# Rank association of the hardness measures (MDP-only) vs learner requirement.
hard = pd.read_csv(OUT / "hardness_colosseum.csv")
ev = lab.groupby(["mdp_id", "algorithm"]).event.mean().unstack()
assoc = []
for learner in ev.columns:
    joined = hard.set_index("mdp_id").join(ev[learner], how="inner")
    for measure in ("diameter", "value_norm", "suboptimal_gaps"):
        assoc.append({"learner": learner, "measure": measure,
                      "spearman": float(spearmanr(joined[measure], joined[learner]).statistic)})
oeh_mean = m[m.method == "oeh_final"].groupby(["mdp_id", "algorithm"]).expected_event.mean().unstack()
for learner in ev.columns:
    assoc.append({"learner": learner, "measure": "oeh_final_expected_event",
                  "spearman": float(spearmanr(oeh_mean[learner], ev.loc[oeh_mean.index, learner]).statistic)})

OUT.joinpath("score").mkdir(exist_ok=True)
summary.to_csv(OUT / "score/summary.csv")
table.to_csv(OUT / "score/contrasts.csv", index=False)
mdp.to_csv(OUT / "score/mdp_scores.csv", index=False)
by_family.to_csv(OUT / "score/by_family.csv")
by_learner.to_csv(OUT / "score/by_learner.csv")
pd.DataFrame(within).to_csv(OUT / "score/within_group_learner_ranking.csv", index=False)
pd.DataFrame(assoc).to_csv(OUT / "score/hardness_association.csv", index=False)
write_json(OUT / "score/verdict.json", {"mdps": n, "critical": crit, "verdict": verdict,
                                         "censored_fraction": censor.to_dict(),
                                         "event_distribution": {int(k): int(v) for k, v in dist.items()},
                                         "predictions_sha256": seal["predictions_sha256"]})
pd.set_option("display.width", 200)
print(summary.round(6), "\n")
print(table.round(6).to_string(index=False), "\n")
print("verdict:", verdict, "critical", round(crit, 3))
print("censored >x50:", censor.round(3).to_dict(), "| event distribution:", dist.to_dict(), "\n")
print(by_family.round(4), "\n")
print(by_learner.round(4), "\n")
print(pd.DataFrame(within).round(3).to_string(index=False), "\n")
print(pd.DataFrame(assoc).pivot(index="measure", columns="learner", values="spearman").round(3))

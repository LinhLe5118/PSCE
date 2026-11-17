"""Single registered analysis of the fourth 120-MDP test.

Runs only after every core and external learner job is verified against the
sealed plans. Produces the difficulty gate with the team's frozen analysis,
the seven-learner selection study and the oracle offline-horizon comparison.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.analysis import run_oeh_final_120 as R
from scripts.analysis.run_unseen_validation import archive_imports, load_json, write_json

base = R.base
OUTPUT = R.OUTPUT
base.FORECAST_METHODS = (*R.COMPARATORS, R.CANDIDATE, *R.REPORTED)


def simultaneous_lower(effect: np.ndarray, draws: int, seed: int) -> tuple[np.ndarray, float]:
    """One-sided simultaneous 95% lower bounds by the max-t MDP-cluster bootstrap."""
    n = effect.shape[0]
    mean = effect.mean(0)
    se = effect.std(0, ddof=1) / math.sqrt(n)
    index = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    critical = float(np.quantile(np.max((effect[index].mean(1) - mean) / se, axis=1), 0.95))
    return mean - critical * se, critical


def external_outcomes() -> pd.DataFrame:
    plan_path = OUTPUT / "external_execution_plan.json"
    plan = load_json(plan_path)
    rows = []
    for job in plan["jobs"]:
        path = OUTPUT / "external_execution/jobs" / base.job_path(job).name
        record = load_json(path)
        outcome = record["outcome"]
        value = float(outcome["normalized_return"])
        if (record["job"] != job or outcome["status"] != "observed"
                or bool(outcome["success"]) != (value >= 0.95) or not math.isfinite(value)):
            raise RuntimeError(f"Invalid external job {path}")
        rows.append({"mdp_id": job["instance_id"], "policy_family": job["policy_family"],
                     "algorithm": job["method"], "training_seed": job["training_seed"],
                     "normalized_return": value, "success": int(value >= 0.95)})
    return pd.DataFrame(rows)


def selection_study(doc: dict) -> dict:
    core = pd.read_csv(OUTPUT / "all_core_budget_seed_outcomes.csv")
    core = core[core.budget == 10][["mdp_id", "policy_family", "algorithm", "training_seed",
                                    "normalized_return", "success"]]
    outcomes = pd.concat([core, external_outcomes()], ignore_index=True)
    family = {item["instance_id"]: item["family"] for item in base.dataset()["instances"]}
    outcomes = outcomes[outcomes.mdp_id.map(family) != "lock_structured"]
    group = outcomes.groupby(["mdp_id", "policy_family", "algorithm"], as_index=False).agg(
        ret=("normalized_return", "mean"), suc=("success", "mean"), seeds=("success", "size"))
    if len(group) != 420 * len(R.LIBRARY) or not (group.seeds == 10).all():
        raise RuntimeError("Incomplete seven-learner x10 panel")
    table = group.pivot_table(index=["mdp_id", "policy_family"], columns="algorithm", values=["ret", "suc"])
    oeh = load_json(OUTPUT / "oeh_choices_preoutcome.json")
    v5 = load_json(OUTPUT / "v5_choices_preoutcome.json")
    rules = {f"fixed:{name}": {key: name for key in oeh} for name in R.LIBRARY}
    rules["oeh_final_selection"] = oeh
    rules["oeh_v5_router"] = {key: value for key, value in v5.items() if key in oeh}
    per_group = []
    for (mdp, behavior), row in table.iterrows():
        key = f"{mdp}|{behavior}"
        record = {"mdp_id": mdp, "policy_family": behavior}
        for rule, choice in rules.items():
            record[f"{rule}|ret"] = row[("ret", choice[key])]
            record[f"{rule}|suc"] = row[("suc", choice[key])]
        record["random|ret"] = float(np.mean([row[("ret", m)] for m in R.LIBRARY]))
        record["random|suc"] = float(np.mean([row[("suc", m)] for m in R.LIBRARY]))
        best = max(R.LIBRARY, key=lambda m: (row[("ret", m)], row[("suc", m)]))
        record["oracle_group|ret"], record["oracle_group|suc"] = row[("ret", best)], row[("suc", best)]
        per_group.append(record)
    groups = pd.DataFrame(per_group)
    groups.to_csv(OUTPUT / "selection_group_outcomes.csv", index=False)
    mdp = groups.drop(columns="policy_family").groupby("mdp_id").mean()
    if len(mdp) != 84:
        raise RuntimeError("Expected 84 stochastic MDPs")
    names = sorted({column.split("|")[0] for column in mdp.columns})
    summary = pd.DataFrame([{"rule": name, "mean_normalized_return": mdp[f"{name}|ret"].mean(),
                             "seed_success_rate": mdp[f"{name}|suc"].mean()} for name in names])
    summary.to_csv(OUTPUT / "selection_summary.csv", index=False)
    fixed = [f"fixed:{name}" for name in R.LIBRARY]
    best_fixed = max(fixed, key=lambda r: (mdp[f"{r}|ret"].mean(), r == "fixed:cfqi_tabular"))
    contrasts, effects = [], []
    for baseline in (best_fixed, "oeh_v5_router"):
        for endpoint in ("ret", "suc"):
            effect = (mdp[f"oeh_final_selection|{endpoint}"] - mdp[f"{baseline}|{endpoint}"]).to_numpy()
            effects.append(effect)
            contrasts.append({"baseline": baseline, "endpoint": endpoint, "delta": float(effect.mean()),
                              "mdps": 84, "mdps_improved": int((effect > 0).sum()),
                              "mdps_worse": int((effect < 0).sum())})
    lower, critical = simultaneous_lower(np.column_stack(effects), 10000, 20261005)
    for row, bound in zip(contrasts, lower):
        row["simultaneous_one_sided_lower_95"] = float(bound)
    pd.DataFrame(contrasts).to_csv(OUTPUT / "selection_contrasts.csv", index=False)
    descriptive = [{"baseline": rule, "endpoint": endpoint,
                    "delta": float((mdp[f"oeh_final_selection|{endpoint}"] - mdp[f"{rule}|{endpoint}"]).mean())}
                   for rule in (*fixed, "random") for endpoint in ("ret", "suc")]
    pd.DataFrame(descriptive).to_csv(OUTPUT / "selection_vs_every_learner.csv", index=False)
    counts = pd.Series(list(oeh.values())).value_counts().to_dict()
    return {"best_fixed": best_fixed, "critical": critical,
            "gate": "GO" if all(row["simultaneous_one_sided_lower_95"] > 0 for row in contrasts) else "NO-GO",
            "choice_counts": {k: int(v) for k, v in counts.items()}}


def _rank_target(events: pd.Series) -> np.ndarray:
    return events.to_numpy(dtype=float)  # censored events (6) already rank last


def horizon_study() -> dict:
    labels = pd.read_csv(OUTPUT / "labels.csv")
    horizon = pd.read_csv(OUTPUT / "horizon_preoutcome.csv")
    oracle = pd.read_csv(OUTPUT / "oracle_descriptors_preoutcome.csv")
    manifest = base.dataset()
    size = {item["instance_id"]: math.log(item["num_states"] * item["num_actions"]) for item in manifest["instances"]}
    cfqi = labels[labels.algorithm == "cfqi_tabular"][["mdp_id", "policy_family", "event", "panel"]]
    best = labels.groupby(["mdp_id", "policy_family"], as_index=False).event.min().rename(columns={"event": "min_event"})
    frame = cfqi.merge(best, on=["mdp_id", "policy_family"]).merge(horizon, on=["mdp_id", "policy_family"])
    frame = frame.merge(oracle, on="mdp_id", how="left")
    frame["size_log_sa"] = frame.mdp_id.map(size)
    big = 1e6  # an infinite horizon ranks after every finite one
    frame["oracle_oeh_h_rank"] = frame.oracle_oeh_h.fillna(big)
    frame["oracle_stationary_eh_rank"] = frame.oracle_stationary_eh.fillna(big)
    frame["data_oeh_h_rank"] = frame.data_oeh_h.fillna(big)
    for column in ("oracle_log_concentrability", "oracle_log_clipped_concentrability", "oracle_log_intrinsic_bound"):
        frame[f"{column}_rank"] = frame[column].fillna(big)
    frame.to_csv(OUTPUT / "horizon_panel.csv", index=False)
    metrics = {"oracle OEH-H": "oracle_oeh_h_rank", "data-only OEH-H": "data_oeh_h_rank",
               "stationary EH (gap bound)": "oracle_stationary_eh_rank",
               "published stochastic EH": "stochastic_eh_hbar_h100",
               "EH minimum QVI depth": "stochastic_eh_min_qvi_k_h100",
               "single-policy concentrability C*": "oracle_log_concentrability_rank",
               "clipped concentrability": "oracle_log_clipped_concentrability_rank",
               "Yin-Wang intrinsic bound": "oracle_log_intrinsic_bound_rank",
               "size log(SA)": "size_log_sa"}
    rows = []
    primary = frame[frame.panel == "primary"].reset_index(drop=True)
    mdps = sorted(primary.mdp_id.unique())
    position = {m: i for i, m in enumerate(mdps)}
    members = [np.flatnonzero(primary.mdp_id.map(position).to_numpy() == i) for i in range(len(mdps))]
    rng = np.random.default_rng(20261006)
    draws = rng.integers(0, len(mdps), size=(10000, len(mdps)))
    for target in ("event", "min_event"):
        y = primary[target].to_numpy(dtype=float)
        stats = {}
        for label, column in metrics.items():
            x = primary[column].to_numpy(dtype=float)
            stats[label] = float(spearmanr(x, y, nan_policy="omit").statistic)
        rows += [{"target": target, "metric": k, "spearman": v} for k, v in stats.items()]
        # Paired MDP-cluster bootstrap of rho(oracle OEH-H) - rho(EH comparators).
        for comparator in ("stationary EH (gap bound)", "published stochastic EH", "single-policy concentrability C*",
                           "clipped concentrability", "Yin-Wang intrinsic bound", "size log(SA)"):
            diffs = []
            for draw in draws[:2000]:
                index = np.concatenate([members[i] for i in draw])
                a = spearmanr(primary["oracle_oeh_h_rank"].to_numpy()[index], y[index]).statistic
                b = spearmanr(primary[metrics[comparator]].to_numpy()[index], y[index], nan_policy="omit").statistic
                diffs.append(a - b)
            rows.append({"target": target, "metric": f"oracle OEH-H minus {comparator}",
                         "spearman": stats["oracle OEH-H"] - stats[comparator],
                         "bootstrap_95_low": float(np.quantile(diffs, 0.025)),
                         "bootstrap_95_high": float(np.quantile(diffs, 0.975))})
    pd.DataFrame(rows).to_csv(OUTPUT / "horizon_comparison.csv", index=False)
    return {"rows": rows}


def analyze() -> None:
    doc, manifest = base.verified_preoutcome()
    base.analyze()
    contrasts = pd.read_csv(OUTPUT / "contrasts.csv")
    def tier_verdict(names):
        rows = contrasts[contrasts.baseline.isin(names)]
        return "GO" if bool(((rows.relative_gain >= 0.05) & (rows.simultaneous_one_sided_lower_95 > 0)).all()) else "NO-GO"
    tiers = {"tier1": tier_verdict(R.TIER1), "tier2": tier_verdict(R.TIER2),
             "all_comparators": tier_verdict((*R.TIER1, *R.TIER2))}
    selection = selection_study(doc)
    horizon = horizon_study()
    manifest_path = OUTPUT / "result_manifest.json"
    result = load_json(manifest_path)
    result.update({"tier_verdicts": tiers, "selection_gate": selection["gate"], "selection": selection,
                   "horizon_study_rows": len(horizon["rows"])})
    write_json(manifest_path, result)
    print(json.dumps({"difficulty_gate_all": result["difficulty_gate"], "tiers": tiers,
                      "selection_gate": selection["gate"]}, indent=2))

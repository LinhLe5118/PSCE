"""Refit the frozen v5 router on E1 plus the opened fourth cohort, grouped CV."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from scripts.analysis.run_unseen_validation import _load_e1_training, archive_imports
from scripts.analysis.oeh_final_analysis import simultaneous_lower
from scripts.analysis import run_oeh_final_120 as R

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/oeh_final_unseen_120_2026-10-02"


def fourth_panel(e1: dict) -> tuple[dict, dict]:
    from evaluation import e1_pipeline
    from oeh.reference import FEATURES

    names = e1_pipeline._feature_names()
    pilot = pd.read_csv(OUT / "pilot_features.csv")
    features = {(row.mdp_id, row.policy_family, row.algorithm):
                {block: tuple(float(getattr(row, field)) for field in names[block]) for block in FEATURES}
                for row in pilot.itertuples(index=False)}
    core = pd.read_csv(OUT / "all_core_budget_seed_outcomes.csv")
    core = core[core.budget == 10]
    all_rows = [core]
    plan = json.loads((OUT / "external_execution_plan.json").read_text())
    external = []
    for job in plan["jobs"]:
        record = json.loads((OUT / "external_execution/jobs" / R.base.job_path(job).name).read_text())
        external.append({"mdp_id": job["instance_id"], "policy_family": job["policy_family"],
                         "algorithm": job["method"], "training_seed": job["training_seed"],
                         "normalized_return": record["outcome"]["normalized_return"],
                         "success": record["outcome"]["success"]})
    all_rows.append(pd.DataFrame(external))
    rows = pd.concat(all_rows, ignore_index=True)
    rows = rows[rows.mdp_id.str.contains("lock_structured") == False]
    runs = {}
    for (mdp, behavior, method), part in rows.groupby(["mdp_id", "policy_family", "algorithm"]):
        runs.setdefault((mdp, behavior), {})[method] = {
            int(row.training_seed): (float(row.normalized_return), int(row.success))
            for row in part.itertuples(index=False)}
    assert len(runs) == 420
    assert all(set(item) == set(e1["candidate_methods"]) and all(len(v) == 10 for v in item.values()) for item in runs.values())
    return runs, features


def main() -> None:
    archive_imports()
    from oeh.unseen import fit_unseen_routes

    e1 = _load_e1_training()
    runs, features = fourth_panel(e1)
    primary = sorted({mdp for mdp, _ in runs})
    base = pd.read_csv(OUT / "selection_group_outcomes.csv")
    v5 = json.loads((OUT / "v5_choices_preoutcome.json").read_text())
    results = {}
    target = ROOT / "artifacts/oeh_router_v6_development_2026-10-04"
    target.mkdir(exist_ok=True)
    for fold, (fit, val) in enumerate(GroupKFold(5).split(primary, groups=primary)):
        checkpoint = target / f"choices_fold_{fold}.json"
        if checkpoint.exists():
            results.update({tuple(k.split("|")): v for k, v in json.loads(checkpoint.read_text()).items()})
            continue
        train_ids, val_ids = {primary[i] for i in fit}, {primary[i] for i in val}
        p = copy.deepcopy(e1)
        p["runs"].update({key: value for key, value in runs.items() if key[0] in train_ids})
        p["features"].update({key: value for key, value in features.items() if key[0] in train_ids})
        p["outer"].update({key: 0 for key in train_ids})
        p["panels"]["stochastic_primary"] = tuple(sorted(set(p["panels"]["stochastic_primary"]) | train_ids))
        query = {key: value for key, value in features.items() if key[0] in val_ids}
        choices, _, _ = fit_unseen_routes(p, query, {key: "stochastic_primary" for key in val_ids})
        results.update(choices)
        checkpoint.write_text(json.dumps({"|".join(k): v for k, v in choices.items()}))
        print("fold", fold, "choices", len(choices), flush=True)
    assert len(results) == 420
    frame = base.set_index(["mdp_id", "policy_family"])
    effects = []
    for key, method in results.items():
        row = frame.loc[key]
        old = v5[f"{key[0]}|{key[1]}"]
        effects.append({"mdp_id": key[0], "success": row[f"fixed:{method}|suc"] - row[f"fixed:{old}|suc"],
                        "return": row[f"fixed:{method}|ret"] - row[f"fixed:{old}|ret"],
                        "old": old, "new": method})
    effect = pd.DataFrame(effects)
    grouped = effect.groupby("mdp_id")[["success", "return"]].mean()
    lower, critical = simultaneous_lower(grouped.to_numpy(), 10000, 20261004)
    print("switches", (effect.old != effect.new).sum(), "point", grouped.mean().to_dict(),
          "simultaneous_lower", lower.tolist(), "critical", critical, flush=True)
    effect.to_csv(target / "oof_effects.csv", index=False)
    (target / "summary.json").write_text(json.dumps({"switches": int((effect.old != effect.new).sum()),
                                                     "point": grouped.mean().to_dict(),
                                                     "simultaneous_lower": lower.tolist(),
                                                     "critical": critical}, indent=2))


if __name__ == "__main__":
    main()

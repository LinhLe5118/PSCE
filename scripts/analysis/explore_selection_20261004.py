"""Exploratory grouped-CV learner-selection study on the opened fourth cohort."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/oeh_final_unseen_120_2026-10-02"


def main() -> None:
    y = pd.read_csv(OUT / "selection_group_outcomes.csv")
    y["key"] = y.mdp_id + "|" + y.policy_family
    v5 = json.loads((OUT / "v5_choices_preoutcome.json").read_text())
    algorithms = [c[6:-4] for c in y.columns if c.startswith("fixed:") and c.endswith("|suc")]
    features = pd.read_csv(OUT / "pilot_features.csv")
    features = features[features.algorithm == "cfqi_tabular"].drop(columns="algorithm")
    family = y.mdp_id.str.extract(r"oehfinal120_(.+)_\d{3}$")[0]
    features["family"] = features.mdp_id.str.extract(r"oehfinal120_(.+)_\d{3}$")[0]
    pred = pd.read_csv(OUT / "predictions_preoutcome.csv")
    pred = pred[(pred.method == "oeh_final") & (pred.algorithm == "cfqi_tabular")]
    pred = pred[["mdp_id", "policy_family", *[f"p_{i}" for i in range(7)]]]
    features = features.merge(pred, on=["mdp_id", "policy_family"])
    jsonrows = []
    for file in (OUT / "selection_groups").glob("*.json"):
        row = json.loads(file.read_text())
        item = {"mdp_id": row["mdp_id"], "policy_family": row["policy_family"]}
        if row["normalized_return"]:
            values = np.array(row["normalized_return"])
            for i, learner in enumerate(row["learners"]):
                item[f"post_suc_{learner}"] = (values[:, i] >= .95).mean()
                item[f"post_ret_{learner}"] = values[:, i].mean()
        jsonrows.append(item)
    features = features.merge(pd.DataFrame(jsonrows), on=["mdp_id", "policy_family"])
    frame = y.merge(features, on=["mdp_id", "policy_family"], validate="one_to_one")
    x = frame.drop(columns=[c for c in frame if c.startswith("fixed:") or c.endswith("|suc") or c.endswith("|ret") or c in ("mdp_id", "policy_family", "key", "family")]).select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)
    x = pd.concat([x, pd.get_dummies(frame.family, prefix="family")], axis=1)
    suc = frame[[f"fixed:{a}|suc" for a in algorithms]].to_numpy()
    ret = frame[[f"fixed:{a}|ret" for a in algorithms]].to_numpy()
    vi = np.array([algorithms.index(v5[k]) for k in frame.key])
    base_suc = suc[np.arange(len(frame)), vi]
    base_ret = ret[np.arange(len(frame)), vi]
    print("v5", base_suc.mean(), base_ret.mean(), "oracle", suc.max(1).mean())
    for model_name in ("extra", "hist", "catboost"):
        oof_suc = np.zeros_like(suc)
        oof_ret = np.zeros_like(ret)
        for fit, val in GroupKFold(5).split(x, groups=frame.mdp_id):
            if model_name == "catboost":
                for outcome, table in ((suc, oof_suc), (ret, oof_ret)):
                    model = CatBoostRegressor(iterations=450, depth=4, learning_rate=.04, loss_function="MultiRMSE", thread_count=8, verbose=False, random_seed=42)
                    model.fit(x.iloc[fit], outcome[fit])
                    table[val] = model.predict(x.iloc[val])
                continue
            for j in range(len(algorithms)):
                for outcome, table in ((suc, oof_suc), (ret, oof_ret)):
                    if model_name == "extra":
                        model = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=7, max_features=.8, n_jobs=4, random_state=42)
                    else:
                        model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=1, random_state=42)
                    model.fit(x.iloc[fit], outcome[fit, j])
                    table[val, j] = model.predict(x.iloc[val])
        for floor in (0, .02, .05, .1):
            score = oof_suc + .15 * oof_ret
            choices = score.argmax(1)
            advantage = score[np.arange(len(frame)), choices] - score[np.arange(len(frame)), vi]
            choice = np.where(advantage > floor, choices, vi)
            print(model_name, floor, "override", (choice != vi).sum(), "success", suc[np.arange(len(frame)), choice].mean(), "ret", ret[np.arange(len(frame)), choice].mean())
            if model_name in ("extra", "catboost") and floor == .1:
                from scripts.analysis.oeh_final_analysis import simultaneous_lower
                effects = pd.DataFrame({"mdp_id": frame.mdp_id,
                                        "v5_suc": suc[np.arange(len(frame)), choice] - base_suc,
                                        "v5_ret": ret[np.arange(len(frame)), choice] - base_ret,
                                        "fixed_suc": suc[np.arange(len(frame)), choice] - suc[:, algorithms.index("cfqi_tabular")],
                                        "fixed_ret": ret[np.arange(len(frame)), choice] - ret[:, algorithms.index("cfqi_tabular")]})
                array = effects.groupby("mdp_id").mean().to_numpy()
                bound, critical = simultaneous_lower(array, 10000, 20261004)
                print("extra MDP effects", array.mean(0), "simultaneous lower", bound, "critical", critical)


if __name__ == "__main__":
    main()

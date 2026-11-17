"""Grouped development study of an E1 plus opened-fourth learner selector."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from scripts.analysis.run_unseen_validation import _load_e1_training, archive_imports

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/oeh_final_unseen_120_2026-10-02"


def main() -> None:
    archive_imports()
    from evaluation import e1_pipeline
    from oeh.reference import METHODS as ANCHORS

    panel = _load_e1_training()
    algorithms = list(panel["candidate_methods"])
    fields = e1_pipeline._feature_names()["B2_OEH3_LCOEH4"]
    pilot = pd.read_csv(OUT / "pilot_features.csv")
    lookup = {(r.mdp_id, r.policy_family, r.algorithm): r for r in pilot.itertuples(index=False)}
    fourth = pd.read_csv(OUT / "selection_group_outcomes.csv")
    v5 = json.loads((OUT / "v5_choices_preoutcome.json").read_text())

    def vector(values: list[np.ndarray], mdp: str) -> list[float]:
        match = re.match(r"^(?:evaluation|oehfinal120)_(.+)_\d{3}$", mdp)
        if match is None:
            raise ValueError(mdp)
        family = match.group(1)
        names = ("bottleneck", "forward_layered", "uniform_garnet", "lock_structured")
        return np.concatenate([*values, [float(family == x) for x in names]]).tolist()

    x_e1, y_e1_s, y_e1_r = [], [], []
    for (mdp, behavior), group in sorted(panel["runs"].items()):
        if mdp not in panel["panels"]["stochastic_primary"]:
            continue
        x_e1.append(vector([np.asarray(panel["features"][(mdp, behavior, a)]["B2_OEH3_LCOEH4"]) for a in ANCHORS], mdp))
        y_e1_r.append([np.mean([v[0] for v in group[a].values()]) for a in algorithms])
        y_e1_s.append([np.mean([v[1] for v in group[a].values()]) for a in algorithms])
    x_4 = []
    for r in fourth.itertuples(index=False):
        x_4.append(vector([np.asarray([getattr(lookup[(r.mdp_id, r.policy_family, a)], f) for f in fields], dtype=float) for a in ANCHORS], r.mdp_id))
    x_e1 = np.nan_to_num(np.array(x_e1), posinf=1e6, neginf=-1e6)
    x_4 = np.nan_to_num(np.array(x_4), posinf=1e6, neginf=-1e6)
    y_e1_r, y_e1_s = np.array(y_e1_r), np.array(y_e1_s)
    y_4_r = fourth[[f"fixed:{a}|ret" for a in algorithms]].to_numpy()
    y_4_s = fourth[[f"fixed:{a}|suc" for a in algorithms]].to_numpy()
    vi = np.array([algorithms.index(v5[f"{m}|{b}"]) for m, b in zip(fourth.mdp_id, fourth.policy_family)])
    print("sizes", x_e1.shape, x_4.shape, "v5", y_4_s[np.arange(len(vi)), vi].mean(), y_4_r[np.arange(len(vi)), vi].mean(), flush=True)
    for use_fourth in (False, True):
        pred_s = np.zeros_like(y_4_s)
        pred_r = np.zeros_like(y_4_r)
        for fold, (fit, val) in enumerate(GroupKFold(5).split(x_4, groups=fourth.mdp_id)):
            xtr = np.concatenate([x_e1, x_4[fit]]) if use_fourth else x_e1
            for outcome, target, result in ((y_e1_s, y_4_s, pred_s), (y_e1_r, y_4_r, pred_r)):
                ytr = np.concatenate([outcome, target[fit]]) if use_fourth else outcome
                model = ExtraTreesRegressor(n_estimators=250, min_samples_leaf=10, max_features=.8, n_jobs=4, random_state=42)
                model.fit(xtr, ytr)
                result[val] = model.predict(x_4[val])
            print("fold", use_fourth, fold, flush=True)
        for weight in (0, .05, .15, .3):
            scores = pred_s + weight * pred_r
            for floor in (0, .02, .05, .1):
                idx = scores.argmax(1)
                idx = np.where(scores[np.arange(len(vi)), idx] - scores[np.arange(len(vi)), vi] > floor, idx, vi)
                print(use_fourth, weight, floor, "switches", int((idx != vi).sum()), "success", round(y_4_s[np.arange(len(vi)), idx].mean(), 5), "return", round(y_4_r[np.arange(len(vi)), idx].mean(), 5), flush=True)


if __name__ == "__main__":
    main()

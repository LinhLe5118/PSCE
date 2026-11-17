"""Post-outcome diagnostic of the frozen third candidate on the second suite.

This is exploratory: the second suite's labels are now open and these
predictions were not sealed before they were observed. The third fresh suite
remains the only prospective test of this candidate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import EH, FIELDS, KEYS, P, ROOT, sha256
from scripts.analysis.develop_cross_learner_context import CONTEXT, with_context
from scripts.analysis.develop_eh_oracle_three_way import aft_probabilities, numeric
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.develop_oeh_difficulty import fit_thresholds, score_probabilities
from scripts.analysis.develop_tree_capacity import multiclass
from scripts.analysis.run_oeh_120_cross_capacity import (
    OUTPUT as THIRD, SECOND_OUTPUT as SECOND, training,
)


OUTPUT = ROOT / "artifacts/oeh_cross_capacity_second_opened_posthoc_2026-10-01"
METHODS_TO_SCORE = ("b2_three_way", "b2_plus_eh_oracle_three_way", "oeh_cross_capacity")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Posthoc diagnostic already exists")
    protocol_path = THIRD / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["candidate"] != "oeh_cross_capacity":
        raise RuntimeError("Unexpected frozen third candidate")
    train = training()
    query = pd.read_csv(SECOND / "pilot_features.csv")
    query = query.merge(
        pd.read_csv(SECOND / "oracle_descriptors_preoutcome.csv")[["mdp_id", *EH]],
        on="mdp_id", validate="many_to_one",
    )
    query = with_context(query)
    labels = pd.read_csv(SECOND / "labels.csv")
    query = query.merge(labels[KEYS + ["panel", "event"]], on=KEYS, validate="one_to_one")
    if (len(query) != 1800 or query.mdp_id.nunique() != 120
            or set(train.mdp_id) & set(query.mdp_id)):
        raise RuntimeError("Opened holdout or train/query separation invalid")
    fields = {
        "b2_three_way": FEATURES_B2,
        "b2_plus_eh_oracle_three_way": FIELDS,
        "oeh_cross_capacity": (*OEH_FEATURES, *CONTEXT),
    }
    scored = []
    for panel in ("primary", "control"):
        for learner in METHODS:
            source = train[(train.panel == panel) & (train.algorithm == learner)]
            target = query[(query.panel == panel) & (query.algorithm == learner)]
            if len(source) != (1260 if panel == "primary" else 540) or len(target) != (420 if panel == "primary" else 180):
                raise RuntimeError("Wrong source or target stratum")
            y = source.event.to_numpy(dtype=int)
            for method in METHODS_TO_SCORE:
                names = fields[method]
                x = numeric(source) if method == "b2_plus_eh_oracle_three_way" else np.nan_to_num(
                    source[list(names)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
                q = numeric(target) if method == "b2_plus_eh_oracle_three_way" else np.nan_to_num(
                    target[list(names)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
                components = np.asarray((aft_probabilities(x, y, q),
                                         fit_thresholds(x, y, q),
                                         multiclass(x, y, q, protocol["multiclass_capacity"])))
                probabilities = np.einsum("i,ijk->jk", protocol["weights"][method], components)
                brier, logloss = score_probabilities(probabilities, target.event.to_numpy(dtype=int))
                frame = target[KEYS + ["panel", "event"]].copy()
                frame["method"] = method
                frame["integrated_brier"] = brier
                frame["seven_class_log_loss"] = logloss
                scored.append(frame)
            print(f"Posthoc opened-suite diagnostic: {panel}/{learner}", flush=True)
    scores = pd.concat(scored, ignore_index=True)
    mdp = scores.groupby(["mdp_id", "panel", "method"], as_index=False)[
        ["integrated_brier", "seven_class_log_loss"]].mean()
    summary = mdp.groupby(["panel", "method"], as_index=False)[
        ["integrated_brier", "seven_class_log_loss"]].mean()
    primary = mdp[mdp.panel == "primary"].set_index(["mdp_id", "method"])
    ids = sorted(mdp.loc[mdp.panel == "primary", "mdp_id"].unique())
    if len(ids) != 84:
        raise RuntimeError("Wrong stochastic holdout MDP count")
    effects, contrasts = [], []
    for baseline in METHODS_TO_SCORE[:2]:
        for metric in ("integrated_brier", "seven_class_log_loss"):
            before = np.asarray([primary.loc[(name, baseline), metric] for name in ids])
            after = np.asarray([primary.loc[(name, "oeh_cross_capacity"), metric] for name in ids])
            effects.append(before - after)
            contrasts.append({"baseline": baseline, "metric": metric,
                              "relative_gain": float(1 - after.mean() / before.mean()),
                              "mean_absolute_improvement": float((before - after).mean())})
    effect = np.column_stack(effects)
    mean = effect.mean(axis=0)
    se = effect.std(axis=0, ddof=1) / np.sqrt(len(ids))
    draws = np.random.default_rng(20261004).integers(0, len(ids), size=(10000, len(ids)))
    critical = float(np.quantile(np.max((effect[draws].mean(axis=1) - mean) / se, axis=1), .95))
    for index, row in enumerate(contrasts):
        row["simultaneous_one_sided_lower_95"] = float(mean[index] - critical * se[index])
    OUTPUT.mkdir(parents=True)
    scores.to_csv(OUTPUT / "scored.csv", index=False)
    mdp.to_csv(OUTPUT / "mdp_scores.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUTPUT / "contrasts.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "posthoc_opened_holdout_not_independent",
        "note": "Candidate fixed before second outcomes; forecasts computed after outcome opening, so this is development evidence only.",
        "third_protocol_sha256": sha256(protocol_path),
        "second_result_manifest_sha256": sha256(SECOND / "result_manifest.json"),
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {name: sha256(OUTPUT / name) for name in
                            ("scored.csv", "mdp_scores.csv", "summary.csv", "contrasts.csv")},
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

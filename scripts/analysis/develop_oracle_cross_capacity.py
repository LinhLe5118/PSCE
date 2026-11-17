"""Matched full-MDP oracle-EH transfer control for the depth-3 OEH candidate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.compare_difficulty import METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import E1_FEATURES, E1_LABELS, E1_ORACLE, EH, KEYS, ROOT, sha256
from scripts.analysis.develop_eh_oracle_three_way import aft_probabilities, numeric
from scripts.analysis.develop_joint_learner_multiclass import load_pair
from scripts.analysis.develop_oeh_difficulty import fit_thresholds, score_probabilities
from scripts.analysis.develop_tree_capacity import VARIANTS, multiclass
from scripts.analysis.run_oeh_120_successor import DEV, FIRST_OUTPUT


OUTPUT = ROOT / "artifacts/oeh_oracle_cross_capacity_development_2026-10-01"


def joined(labels: Path, features: Path, oracle: Path) -> pd.DataFrame:
    frame = load_pair(labels, features)
    frame = frame.merge(pd.read_csv(oracle)[["mdp_id", *EH]], on="mdp_id",
                        validate="many_to_one")
    if frame[KEYS].duplicated().any():
        raise RuntimeError("Duplicate oracle task")
    return frame


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Oracle capacity artifact exists")
    e1 = joined(E1_LABELS, E1_FEATURES, E1_ORACLE)
    opened = {
        "dev_first": joined(*DEV[0]),
        "dev_second": joined(*DEV[1]),
        "failed_120": joined(FIRST_OUTPUT / "labels.csv", FIRST_OUTPUT / "pilot_features.csv",
                             FIRST_OUTPUT / "oracle_descriptors_preoutcome.csv"),
    }
    weights = json.loads((FIRST_OUTPUT / "protocol.json").read_text(encoding="utf-8"))["weights"]["b2_plus_eh_oracle_three_way"]
    rows = []
    for query_name, query in opened.items():
        train = pd.concat((e1, *(frame for name, frame in opened.items()
                                 if name != query_name)), ignore_index=True)
        if set(train.mdp_id) & set(query.mdp_id):
            raise RuntimeError("Oracle cross-cohort leak")
        for learner in METHODS:
            source = train[(train.panel == "primary") & (train.algorithm == learner)]
            target = query[(query.panel == "primary") & (query.algorithm == learner)]
            x, q = numeric(source), numeric(target)
            y = source.event.to_numpy(dtype=int)
            parts = np.asarray((aft_probabilities(x, y, q), fit_thresholds(x, y, q),
                                multiclass(x, y, q, VARIANTS["depth3_200"])))
            p = np.einsum("i,ijk->jk", weights, parts)
            brier, logloss = score_probabilities(p, target.event.to_numpy(dtype=int))
            result = target[[*KEYS, "event"]].copy()
            result["cohort"] = query_name
            result["method"] = "eh_oracle_depth3_200"
            result["integrated_brier"] = brier
            result["seven_class_log_loss"] = logloss
            rows.append(result)
        print(f"Oracle EH capacity: {query_name}", flush=True)
    scored = pd.concat(rows, ignore_index=True)
    summary = scored.groupby("cohort", as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("integrated_brier", "mean"),
        seven_class_log_loss=("seven_class_log_loss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "exploratory_three_opened_cohorts_not_independent",
        "oracle_access": "true full-MDP descriptors, same three-part architecture and meta-training",
        "multiclass_capacity": VARIANTS["depth3_200"], "weights": weights,
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                         for path in (E1_LABELS, E1_FEATURES, E1_ORACLE,
                                      *(p for cohort in DEV for p in cohort),
                                      FIRST_OUTPUT / "labels.csv",
                                      FIRST_OUTPUT / "pilot_features.csv",
                                      FIRST_OUTPUT / "oracle_descriptors_preoutcome.csv")},
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {name: sha256(OUTPUT / name)
                            for name in ("scored.csv", "summary.csv")},
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

"""Matched oracle-EH expanded meta-training on opened development cohorts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.compare_difficulty import METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import (
    DEV_ORACLE, E1_ORACLE, EH, FIELDS, KEYS, P, ROOT, sha256,
)
from scripts.analysis.develop_eh_oracle_three_way import aft_probabilities, numeric
from scripts.analysis.develop_multiclass_oeh import fit_multiclass
from scripts.analysis.develop_oeh_difficulty import fit_thresholds, score_probabilities
from scripts.analysis.develop_oeh_pooled_meta_training import (
    E1_FEATURES, E1_LABELS, FIRST, SECOND,
)


SECOND_ORACLE = ROOT / "artifacts/oeh_b2_eh_oracle_second_development_2026-10-01/oracle_descriptors.csv"
WEIGHTS = ROOT / "artifacts/oeh_eh_oracle_three_way_development_2026-10-01/manifest.json"
OUTPUT = ROOT / "artifacts/oeh_eh_oracle_pooled_meta_development_2026-10-01"


def joined(labels: Path, features: Path, oracle: Path) -> pd.DataFrame:
    frame = pd.read_csv(labels).merge(pd.read_csv(features), on=KEYS,
                                      validate="one_to_one")
    frame = frame.merge(pd.read_csv(oracle)[["mdp_id", *EH]], on="mdp_id",
                        validate="many_to_one")
    return frame


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    e1 = joined(E1_LABELS, E1_FEATURES, E1_ORACLE)
    first = joined(*FIRST, DEV_ORACLE)
    second = joined(*SECOND, SECOND_ORACLE)
    if (len(e1) != 1800 or len(first) != 900 or len(second) != 900
            or set(e1.mdp_id) & set(first.mdp_id)
            or set(e1.mdp_id) & set(second.mdp_id)
            or set(first.mdp_id) & set(second.mdp_id)):
        raise RuntimeError("Oracle EH cohorts incomplete or overlapping")
    weight = np.asarray(json.loads(WEIGHTS.read_text(encoding="utf-8"))[
        "e1_oof_weights_aft_direct_multi"], dtype=float)
    rows = []
    for cohort, addition, target in (("second_open", first, second),
                                     ("first_open", second, first)):
        source = pd.concat((e1, addition), ignore_index=True)
        for panel in ("primary", "control"):
            for learner in METHODS:
                tr = source[(source.panel == panel) & (source.algorithm == learner)]
                qu = target[(target.panel == panel) & (target.algorithm == learner)]
                if len(tr) != (630 if panel == "primary" else 270) or len(qu) != (210 if panel == "primary" else 90):
                    raise RuntimeError("Wrong EH oracle stratum")
                if set(tr.mdp_id) & set(qu.mdp_id):
                    raise RuntimeError("Training/query MDP overlap")
                x, q = numeric(tr), numeric(qu)
                y = tr.event.to_numpy(dtype=int)
                parts = np.asarray((aft_probabilities(x, y, q),
                                    fit_thresholds(x, y, q), fit_multiclass(x, y, q)))
                p = np.einsum("i,ijk->jk", weight, parts)
                brier, logloss = score_probabilities(p, qu.event.to_numpy(dtype=int))
                scored = qu[KEYS + ["panel", "event"]].copy()
                scored["cohort"] = cohort
                scored["method"] = "b2_plus_eh_oracle_pooled_meta"
                scored["brier"] = brier
                scored["logloss"] = logloss
                for i, col in enumerate(P):
                    scored[col] = p[:, i]
                rows.append(scored)
            print(f"Expanded oracle EH: {cohort}/{panel}", flush=True)
    scored = pd.concat(rows, ignore_index=True)
    if len(scored) != 1800 or scored.duplicated(KEYS).any():
        raise RuntimeError("Incomplete expanded EH oracle forecasts")
    summary = scored.groupby(["cohort", "panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored_predictions.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "exploratory_two_opened_cohorts_cross_fit_not_independent_confirmation",
        "scope": "Same expanded meta-training and three-way architecture; EH sees true MDP",
        "weights_aft_direct_multi": weight.tolist(),
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in (E1_LABELS, E1_FEATURES, E1_ORACLE,
                                       *FIRST, DEV_ORACLE, *SECOND, SECOND_ORACLE, WEIGHTS)},
        "script_sha256": sha256(Path(__file__)),
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

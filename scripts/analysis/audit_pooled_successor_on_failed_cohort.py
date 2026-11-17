"""Post-hoc diagnostic of the already-frozen pooled successor on the failed cohort.

This deliberately opens the completed first 120-MDP outcomes. It must never be
presented as a second independent confirmation or used to modify the sealed
successor protocol.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import EH, FIELDS, KEYS, ROOT, sha256
from scripts.analysis.develop_eh_oracle_three_way import aft_probabilities, numeric
from scripts.analysis.develop_multiclass_oeh import fit_multiclass
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.develop_oeh_difficulty import fit_thresholds, score_probabilities
from scripts.analysis.run_oeh_120_successor import FIRST_OUTPUT, OUTPUT as SUCCESSOR_OUTPUT, training


OUTPUT = ROOT / "artifacts/oeh_pooled_failed_cohort_posthoc_2026-10-01"
METHODS_AND_FIELDS = (
    ("b2_pooled", FEATURES_B2, "b2_three_way"),
    ("oeh_pooled", OEH_FEATURES, "oeh_three_way"),
    ("b2_eh_pooled", FIELDS, "b2_plus_eh_oracle_three_way"),
)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Post-hoc diagnostic already exists")
    manifest = json.loads((FIRST_OUTPUT / "result_manifest.json").read_text(encoding="utf-8"))
    if manifest["difficulty_gate"] != "NO-GO":
        raise RuntimeError("First independent test has not failed")
    protocol = json.loads((SUCCESSOR_OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    frozen_sha = sha256(SUCCESSOR_OUTPUT / "protocol.json")
    if frozen_sha != (SUCCESSOR_OUTPUT / "protocol.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("Successor protocol seal mismatch")
    source = training()
    query = pd.read_csv(FIRST_OUTPUT / "pilot_features.csv")
    query = query.merge(pd.read_csv(FIRST_OUTPUT / "oracle_descriptors_preoutcome.csv")[["mdp_id", *EH]],
                        on="mdp_id", validate="many_to_one")
    query = query.merge(pd.read_csv(FIRST_OUTPUT / "labels.csv")[[*KEYS, "panel", "event"]],
                        on=KEYS, validate="one_to_one")
    if len(query) != 1800 or set(query.mdp_id) & set(source.mdp_id):
        raise RuntimeError("Diagnostic query overlaps expanded training")
    rows = []
    for panel in ("primary", "control"):
        for learner in METHODS:
            train = source[(source.panel == panel) & (source.algorithm == learner)]
            target = query[(query.panel == panel) & (query.algorithm == learner)]
            if len(train) != (840 if panel == "primary" else 360) or len(target) != (420 if panel == "primary" else 180):
                raise RuntimeError("Incorrect post-hoc stratum")
            y = train.event.to_numpy(dtype=int)
            for name, fields, weight_name in METHODS_AND_FIELDS:
                x = numeric(train) if name == "b2_eh_pooled" else np.nan_to_num(
                    train[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
                q = numeric(target) if name == "b2_eh_pooled" else np.nan_to_num(
                    target[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
                parts = np.asarray((aft_probabilities(x, y, q),
                                    fit_thresholds(x, y, q), fit_multiclass(x, y, q)))
                p = np.einsum("i,ijk->jk", protocol["weights"][weight_name], parts)
                brier, logloss = score_probabilities(p, target.event.to_numpy(dtype=int))
                frame = target[[*KEYS, "panel", "event"]].copy()
                frame["method"] = name
                frame["integrated_brier"] = brier
                frame["seven_class_log_loss"] = logloss
                rows.append(frame)
            print(f"Post-hoc pooled diagnostic: {panel}/{learner}", flush=True)
    scored = pd.concat(rows, ignore_index=True)
    summary = scored.groupby(["panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("integrated_brier", "mean"),
        seven_class_log_loss=("seven_class_log_loss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "posthoc_opened_failed_cohort_not_independent_confirmation",
        "successor_protocol_sha256_before_this_diagnostic": frozen_sha,
        "first_result_manifest_sha256": sha256(FIRST_OUTPUT / "result_manifest.json"),
        "artifact_sha256": {name: sha256(OUTPUT / name) for name in ("scored.csv", "summary.csv")},
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

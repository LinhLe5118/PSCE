"""Blinded, late-added architecture-matched B2 forecasts on fresh MDPs.

This comparator was specified after learner execution started, but before
the analyst opened/scored any fresh outcomes. It does not amend the original
confirmation protocol or its registered gate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS, sha256
from scripts.analysis.confirm_oeh_difficulty import OUTPUT as CONFIRMATION, verified_preoutcome
from scripts.analysis.develop_oeh_blend import KEYS, PROBABILITY_COLUMNS
from scripts.analysis.develop_oeh_difficulty import ROOT, E1, OLD, fit_thresholds

ARCHITECTURE = ROOT / "artifacts/oeh_difficulty_architecture_control_2026-10-01"
OUTPUT = ROOT / "artifacts/oeh_difficulty_matched_b2_confirmation_2026-10-01"


def main():
    import numpy as np
    import pandas as pd

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite forecasts: {OUTPUT}")
    protocol, manifest = verified_preoutcome()
    development = json.loads((ARCHITECTURE / "manifest.json").read_text(encoding="utf-8"))
    if (development["status"] != "development_only_open_60_mdp_not_independent_confirmation"
            or sha256(ARCHITECTURE / "predictions.csv") != development["artifact_sha256"]["predictions.csv"]):
        raise RuntimeError("Architecture control artifact changed")
    weight = float(development["weight_on_direct"])
    if not 0 <= weight <= 1:
        raise RuntimeError("Invalid E1-OOF blend weight")
    train = pd.read_csv(E1 / "labels.csv").merge(
        pd.read_csv(OLD / "features.csv"), on=["mdp_id", "policy_family", "algorithm"],
        validate="one_to_one")
    pilot = pd.read_csv(CONFIRMATION / "pilot_features.csv")
    aft = pd.read_csv(CONFIRMATION / "predictions_preoutcome.csv")
    aft = aft[aft.method == "b2_generic"]
    aft = aft[KEYS + PROBABILITY_COLUMNS]
    query = pilot.merge(aft[KEYS], on=["mdp_id", "policy_family", "algorithm"],
                        validate="one_to_one")
    train = train.sort_values(KEYS).reset_index(drop=True)
    query = query.sort_values(KEYS).reset_index(drop=True)
    aft = query[KEYS].merge(aft, on=KEYS, validate="one_to_one")
    if (len(train) != 1800 or len(query) != 900 or len(aft) != 900
            or not aft[KEYS].equals(query[KEYS])
            or set(train.mdp_id) & set(query.mdp_id)):
        raise RuntimeError("Incomplete or overlapping training/query rows")
    direct = np.full((len(query), 7), np.nan)
    for panel in ("primary", "control"):
        for learner in METHODS:
            tr = train[(train.panel == panel) & (train.algorithm == learner)]
            positions = np.flatnonzero((query.panel == panel) & (query.algorithm == learner))
            if len(tr) != (420 if panel == "primary" else 180) or len(positions) != (210 if panel == "primary" else 90):
                raise RuntimeError("Panel/learner stratum mismatch")
            direct[positions] = fit_thresholds(tr[list(FEATURES_B2)].to_numpy(),
                                               tr.event.to_numpy(),
                                               query.iloc[positions][list(FEATURES_B2)].to_numpy())
    if not np.isfinite(direct).all():
        raise RuntimeError("Incomplete direct B2 forecasts")
    blended = weight * direct + (1. - weight) * aft[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if (not np.isfinite(blended).all() or (blended < 0).any()
            or not np.allclose(blended.sum(axis=1), 1.)):
        raise RuntimeError("Invalid B2 mixture probabilities")
    rows = query[KEYS].copy()
    rows["method"] = "b2_matched_blend"
    for index, name in enumerate(PROBABILITY_COLUMNS):
        rows[name] = blended[:, index]
    OUTPUT.mkdir(parents=True)
    rows.to_csv(OUTPUT / "predictions_blinded.csv", index=False)
    source_files = (E1 / "labels.csv", OLD / "features.csv",
                    CONFIRMATION / "pilot_features.csv",
                    CONFIRMATION / "predictions_preoutcome.csv",
                    ARCHITECTURE / "manifest.json")
    seal = {
        "status": "blinded_additional_comparator_frozen_before_analyst_opened_fresh_outcomes",
        "timing": "Added after learner execution started; not in original registered gate",
        "registered_gate_unchanged": True,
        "new_60_mdp_outcome_files_read_by_this_script": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmation_protocol_sha256": sha256(CONFIRMATION / "protocol.json"),
        "confirmation_preoutcome_seal_sha256": sha256(CONFIRMATION / "preoutcome_seal.json"),
        "architecture_development_manifest_sha256": sha256(ARCHITECTURE / "manifest.json"),
        "weight_on_direct": weight,
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in source_files},
        "script_sha256": sha256(Path(__file__)),
        "predictions_sha256": sha256(OUTPUT / "predictions_blinded.csv"),
        "rows": len(rows),
    }
    (OUTPUT / "seal.json").write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
    print(f"Sealed {len(rows)} late-added, blinded B2 forecasts: {seal['predictions_sha256']}")


if __name__ == "__main__":
    main()

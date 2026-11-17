"""Seal a prespecified ablation without direct OEH-B on BRIDGE, before labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import run_oeh_confirmatory_20261004 as C
from scripts.analysis import run_oeh_final_120 as R
from scripts.analysis import oeh_final_forecast as F
from scripts.analysis.develop_cross_learner_context import with_context
from scripts.analysis.run_week8_matched_ablation import arms

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
FEATURES = ROOT / "artifacts/week8/bridge_pilot_features_preoutcome_2026-10-07"
HORIZON = ROOT / "artifacts/week8/bridge_horizon_preoutcome_2026-10-07"
OUTPUT = ROOT / "artifacts/week8/bridge_reduced_forecast_preoutcome_2026-10-07"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def horizon_rows(path: Path) -> pd.DataFrame:
    h = pd.read_csv(path)
    if len(h) != 70 or h[["mdp_id", "policy_family"]].duplicated().any():
        raise RuntimeError("Incomplete horizon table")
    return pd.DataFrame({
        "mdp_id": h.mdp_id, "policy_family": h.policy_family,
        "oehh_value": h.data_oeh_h.fillna(0.0),
        "oehh_identified": (h.data_oeh_h_status == "identified").astype(float),
        "oehh_trap": (h.data_oeh_h_status == "coverage_trap").astype(float),
        "oehh_statistical": h.data_oeh_h_statistical.fillna(0.0),
        "oehh_coverage": h.data_oeh_h_coverage.fillna(0.0),
        "oehh_depth_fraction": h.data_oeh_h_depth.fillna(101.0) / 100.0,
        "oehh_log_concentrability": h.data_log_concentrability.fillna(np.inf),
        "oehh_log_clipped_concentrability": h.data_log_clipped_concentrability.fillna(np.inf),
        "oehh_log_intrinsic_bound": h.data_log_intrinsic_bound.fillna(np.inf),
        "oehh_behavior_ceiling": h.data_behavior_ceiling.fillna(0.0),
        "oehh_support_ceiling": h.data_support_ceiling.fillna(0.0),
        **{f"oehh_{kind}_x{b}": h.get(f"data_{kind}_x{b}", pd.Series(np.nan, index=h.index)).fillna(1.0)
           for kind in ("risk", "miss") for b in F.BUDGETS},
        "oehh_risk_slack": h.get("data_risk_slack", pd.Series(np.nan, index=h.index)).fillna(-1.0),
    })


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    fm = json.loads((FEATURES / "manifest.json").read_text(encoding="utf-8"))
    hm = json.loads((HORIZON / "manifest.json").read_text(encoding="utf-8"))
    if (fm["status"] != "SEALED_30_COLUMN_PILOT_FEATURES_NO_TEST_LABELS" or
            hm["status"] != "SEALED_HORIZON_PREOUTCOME_NO_TEST_LABELS" or
            fm["test_labels_read"] or hm["test_labels_read"] or
            sha256(FEATURES / "features.csv") != fm["features_sha256"] or
            sha256(HORIZON / "horizon_preoutcome.csv") != hm["horizon_csv_sha256"]):
        raise RuntimeError("Preoutcome input seal failed")
    query = pd.read_csv(FEATURES / "features.csv")
    query = query.merge(horizon_rows(HORIZON / "horizon_preoutcome.csv"),
                        on=["mdp_id", "policy_family"], validate="many_to_one")
    query = with_context(query)
    query["panel"] = "primary"
    query = query.sort_values(KEYS).reset_index(drop=True)
    fields = arms()["oeh_without_direct_oehb"]
    if (len(query) != 210 or query.mdp_id.nunique() != 14 or
            query[KEYS].duplicated().any() or set(fields) - set(query.columns)):
        raise RuntimeError("Incomplete reduced OEH query")
    C.configure()
    historical = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    train = R.meta_training(R.OPENED, historical["oeh_final"]["structure_tag"])
    if len(train) != 10800 or train.mdp_id.nunique() != 720:
        raise RuntimeError("Historical meta-training panel changed")
    probability, tuning = F.matched_forecast(train, query, fields)
    if (probability.shape != (210, 7) or not np.isfinite(probability).all() or
            not np.allclose(probability.sum(1), 1)):
        raise RuntimeError("Invalid seven-class reduced OEH law")
    OUTPUT.mkdir(parents=True)
    predictions = query[KEYS].copy()
    predictions["method"] = "oeh_without_direct_oehb"
    predictions[P] = probability
    predictions.to_csv(OUTPUT / "predictions_before_label_read.csv", index=False)
    (OUTPUT / "tuning.json").write_text(json.dumps(tuning, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "SEALED_REDUCED_OEH_ABLATION_NO_TEST_LABELS",
        "role": "supplementary resource-contingency ablation, not full OEH 108",
        "test_labels_read": False, "task_count": 14, "rows": len(predictions),
        "feature_columns": list(fields), "meta_training_mdps": 720,
        "source_sha256": {"pilot_features.csv": sha256(FEATURES / "features.csv"),
                          "horizon.csv": sha256(HORIZON / "horizon_preoutcome.csv"),
                          "historical_protocol.json": sha256(SOURCE / "protocol.json"),
                          "matched_forecast.py": sha256(Path(F.__file__)),
                          "ablation_runner.py": sha256(Path(arms.__code__.co_filename)),
                          "script": sha256(Path(__file__))},
        "output_sha256": {"predictions": sha256(OUTPUT / "predictions_before_label_read.csv"),
                          "tuning": sha256(OUTPUT / "tuning.json")},
    }, indent=2) + "\n", encoding="utf-8")
    print("Sealed 210 BRIDGE reduced-OEH ablation forecasts before test labels", flush=True)


if __name__ == "__main__":
    main()

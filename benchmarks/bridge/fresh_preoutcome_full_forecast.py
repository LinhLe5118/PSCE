"""Seal the final 108-feature OEH forecasts before fresh BRIDGE labels."""
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
from .fresh_adapter import PANEL

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
FEATURES = ROOT / "artifacts/week8/bridge_fresh_pilot_features_preoutcome_2026-10-08"
HORIZON = ROOT / "artifacts/week8/bridge_fresh_horizon_preoutcome_2026-10-08"
OEHB = ROOT / "artifacts/week8/bridge_fresh_full_oehb_preoutcome_2026-10-08"
OUTPUT = ROOT / "artifacts/week8/bridge_fresh_full_forecast_preoutcome_2026-10-08"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(directory: Path, status: str, csv_file: str | None = None) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != status or manifest["test_labels_read"]:
        raise RuntimeError(f"Bad preoutcome manifest: {directory}")
    if csv_file is not None:
        field = "features_sha256" if csv_file == "features.csv" else "horizon_csv_sha256"
        if sha256(directory / csv_file) != manifest[field]:
            raise RuntimeError(f"Preoutcome CSV hash changed: {directory / csv_file}")
    return manifest


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protocol_file = PANEL / "protocol.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    fm = checked(FEATURES, "SEALED_FRESH_30_COLUMN_PILOT_FEATURES_NO_TEST_LABELS", "features.csv")
    hm = checked(HORIZON, "SEALED_FRESH_HORIZON_PREOUTCOME_NO_TEST_LABELS", "horizon_preoutcome.csv")
    bm = checked(OEHB, "SEALED_FRESH_FULL_OEHB_PREOUTCOME_NO_TEST_LABELS")
    if any(m["groups"] != 25 or m["protocol_sha256"] != sha256(protocol_file) for m in (fm, hm, bm)):
        raise RuntimeError("Incomplete fresh pilot groups")
    files = list((OEHB / "groups").glob("*.json"))
    if len(files) != 25 or set(bm["group_sha256"]) != {file.name for file in files}:
        raise RuntimeError("OEH-B group set changed")
    for file in files:
        if (sha256(file) != bm["group_sha256"][file.name] or
                json.loads(file.read_text(encoding="utf-8"))["status"] != "FRESH_FULL_OEHB_PREOUTCOME_ONLY"):
            raise RuntimeError(f"Bad full OEH-B checkpoint: {file.name}")
    horizon = pd.read_csv(HORIZON / "horizon_preoutcome.csv")
    # The historical feature is depth/horizon. The old assembler uses /100
    # because its original benchmark always had H=100. Give it the equivalent
    # numerator for this public task-specific H, preserving the feature's unit.
    depth = horizon["data_oeh_h_depth"].astype(float)
    horizon["data_oeh_h_depth"] = depth * 100.0 / horizon["public_horizon"].astype(float)
    OUTPUT.mkdir(parents=True)
    model_horizon = OUTPUT / "horizon_model_input_before_labels.csv"
    horizon.to_csv(model_horizon, index=False)
    query = pd.read_csv(FEATURES / "features.csv")
    own = F.oeh_feature_rows(OEHB / "groups", model_horizon)
    query = query.merge(own, on=KEYS, validate="one_to_one")
    query = with_context(query)
    query["panel"] = "primary"
    query = query.sort_values(KEYS).reset_index(drop=True)
    fields = tuple(R.CANDIDATE_VARIANTS["plus_prior_profile"])
    if (len(fields) != 108 or len(query) != 75 or query.mdp_id.nunique() != 5 or
            query[KEYS].duplicated().any() or set(fields) - set(query.columns)):
        raise RuntimeError("Incomplete 108-feature fresh OEH query")
    C.configure()
    historical = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    train = R.meta_training(R.OPENED, historical["oeh_final"]["structure_tag"])
    if len(train) != 10800 or train.mdp_id.nunique() != 720:
        raise RuntimeError("Historical meta-training panel changed")
    probability, tuning = F.matched_forecast(train, query, fields)
    if (probability.shape != (75, 7) or not np.isfinite(probability).all() or
            not np.allclose(probability.sum(1), 1)):
        raise RuntimeError("Invalid seven-class fresh full OEH law")
    predictions = query[KEYS].copy()
    predictions["method"] = "oeh_full_108"
    predictions[P] = probability
    predictions.to_csv(OUTPUT / "predictions_before_label_read.csv", index=False)
    (OUTPUT / "tuning.json").write_text(json.dumps(tuning, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "SEALED_FRESH_FULL_108_FORECAST_NO_TEST_LABELS",
        "test_labels_read": False, "task_count": 5, "rows": 75,
        "feature_columns": list(fields), "meta_training_mdps": 720,
        "protocol_sha256": sha256(protocol_file),
        "horizon_depth_mapping": "public depth / public task horizon; H=100 training equivalent",
        "source_sha256": {"pilot_features": sha256(FEATURES / "features.csv"),
                          "raw_horizon": sha256(HORIZON / "horizon_preoutcome.csv"),
                          "model_horizon": sha256(model_horizon),
                          "oehb_manifest": sha256(OEHB / "manifest.json"),
                          "historical_protocol": sha256(SOURCE / "protocol.json"),
                          "matched_forecast": sha256(Path(F.__file__)),
                          "script": sha256(Path(__file__))},
        "output_sha256": {"predictions": sha256(OUTPUT / "predictions_before_label_read.csv"),
                          "tuning": sha256(OUTPUT / "tuning.json")}},
        indent=2) + "\n", encoding="utf-8")
    print("Sealed 75 fresh full OEH forecasts before any learner labels", flush=True)


if __name__ == "__main__":
    main()

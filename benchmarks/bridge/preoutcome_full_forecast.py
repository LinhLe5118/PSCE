"""Seal full 108-column OEH BRIDGE forecasts before reading test labels."""
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

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
FEATURES = ROOT / "artifacts/week8/bridge_pilot_features_preoutcome_2026-10-07"
HORIZON = ROOT / "artifacts/week8/bridge_horizon_preoutcome_2026-10-07"
OEHB = ROOT / "artifacts/week8/bridge_oehb_preoutcome_2026-10-07"
OUTPUT = ROOT / "artifacts/week8/bridge_full_forecast_preoutcome_2026-10-07"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_manifest(directory: Path, status: str, csv: str | None = None) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != status or manifest["test_labels_read"]:
        raise RuntimeError(f"Invalid preoutcome manifest: {directory}")
    if csv is not None:
        key = "features_sha256" if csv == "features.csv" else "horizon_csv_sha256"
        if sha256(directory / csv) != manifest[key]:
            raise RuntimeError(f"Preoutcome CSV hash changed: {csv}")
    return manifest


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    fm = checked_manifest(FEATURES, "SEALED_30_COLUMN_PILOT_FEATURES_NO_TEST_LABELS", "features.csv")
    hm = checked_manifest(HORIZON, "SEALED_HORIZON_PREOUTCOME_NO_TEST_LABELS", "horizon_preoutcome.csv")
    bm = checked_manifest(OEHB, "SEALED_OEHB_PREOUTCOME_NO_TEST_LABELS")
    if fm["groups"] != 70 or hm["groups"] != 70 or bm["groups"] != 70:
        raise RuntimeError("Incomplete BRIDGE pilot groups")
    files = list((OEHB / "groups").glob("*.json"))
    if len(files) != 70 or set(bm["group_sha256"]) != {path.name for path in files}:
        raise RuntimeError("OEH-B group set changed")
    for path in files:
        if sha256(path) != bm["group_sha256"][path.name]:
            raise RuntimeError(f"OEH-B group hash changed: {path.name}")
        if json.loads(path.read_text(encoding="utf-8"))["status"] != "OEHB_PREOUTCOME_ONLY":
            raise RuntimeError(f"OEH-B group failed: {path.name}")
    query = pd.read_csv(FEATURES / "features.csv")
    own = F.oeh_feature_rows(OEHB / "groups", HORIZON / "horizon_preoutcome.csv")
    query = query.merge(own, on=KEYS, validate="one_to_one")
    query = with_context(query)
    query["panel"] = "primary"
    query = query.sort_values(KEYS).reset_index(drop=True)
    fields = tuple(R.CANDIDATE_VARIANTS["plus_prior_profile"])
    if (len(fields) != 108 or len(query) != 210 or query.mdp_id.nunique() != 14 or
            query[KEYS].duplicated().any() or set(fields) - set(query.columns)):
        raise RuntimeError("Incomplete full OEH 108-column query")
    C.configure()
    historical = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    train = R.meta_training(R.OPENED, historical["oeh_final"]["structure_tag"])
    if len(train) != 10800 or train.mdp_id.nunique() != 720:
        raise RuntimeError("Historical meta-training panel changed")
    probability, tuning = F.matched_forecast(train, query, fields)
    if (probability.shape != (210, 7) or not np.isfinite(probability).all() or
            not np.allclose(probability.sum(1), 1)):
        raise RuntimeError("Invalid seven-class full OEH law")
    OUTPUT.mkdir(parents=True)
    predictions = query[KEYS].copy()
    predictions["method"] = "oeh_full_108"
    predictions[P] = probability
    predictions.to_csv(OUTPUT / "predictions_before_label_read.csv", index=False)
    (OUTPUT / "tuning.json").write_text(json.dumps(tuning, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "SEALED_FULL_108_FORECAST_NO_TEST_LABELS",
        "test_labels_read": False, "task_count": 14, "rows": len(predictions),
        "feature_columns": list(fields), "meta_training_mdps": 720,
        "source_sha256": {"pilot_features.csv": sha256(FEATURES / "features.csv"),
                          "horizon.csv": sha256(HORIZON / "horizon_preoutcome.csv"),
                          "oehb_manifest.json": sha256(OEHB / "manifest.json"),
                          "historical_protocol.json": sha256(SOURCE / "protocol.json"),
                          "matched_forecast.py": sha256(Path(F.__file__)),
                          "script": sha256(Path(__file__))},
        "output_sha256": {"predictions": sha256(OUTPUT / "predictions_before_label_read.csv"),
                          "tuning": sha256(OUTPUT / "tuning.json")},
    }, indent=2) + "\n", encoding="utf-8")
    print("Sealed 210 BRIDGE full OEH forecasts before test labels", flush=True)


if __name__ == "__main__":
    main()

"""Seal B2 matched forecasts for BRIDGE before any candidate test label exists."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import run_oeh_confirmatory_20261004 as C
from scripts.analysis import run_oeh_final_120 as R
from scripts.analysis import oeh_final_forecast as F
from scripts.analysis.compare_difficulty import FEATURES_B2

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
FEATURES = ROOT / "artifacts/week8/bridge_pilot_features_preoutcome_2026-10-07"
OUTPUT = ROOT / "artifacts/week8/bridge_b2_forecast_preoutcome_2026-10-07"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads((FEATURES / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["status"] != "SEALED_30_COLUMN_PILOT_FEATURES_NO_TEST_LABELS" or
            manifest["test_labels_read"] or
            sha256(FEATURES / "features.csv") != manifest["features_sha256"]):
        raise RuntimeError("BRIDGE pilot feature seal failed")
    query = pd.read_csv(FEATURES / "features.csv")
    if (len(query) != 210 or query.mdp_id.nunique() != 14 or
            query[KEYS].duplicated().any() or
            set(FEATURES_B2) - set(query.columns)):
        raise RuntimeError("Incomplete B2 query")
    query = query.sort_values(KEYS).reset_index(drop=True)
    query["panel"] = "primary"
    C.configure()
    historical = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    train = R.meta_training(R.OPENED, historical["oeh_final"]["structure_tag"])
    if len(train) != 10800 or train.mdp_id.nunique() != 720:
        raise RuntimeError("Historical meta-training panel changed")
    probability, tuning = F.matched_forecast(train, query, FEATURES_B2)
    if (probability.shape != (210, 7) or not np.isfinite(probability).all() or
            not np.allclose(probability.sum(1), 1)):
        raise RuntimeError("Invalid seven-class B2 law")
    OUTPUT.mkdir(parents=True)
    predictions = query[KEYS].copy()
    predictions["method"] = "b2_matched"
    predictions[P] = probability
    predictions.to_csv(OUTPUT / "predictions_before_label_read.csv", index=False)
    (OUTPUT / "tuning.json").write_text(json.dumps(tuning, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "SEALED_B2_FORECAST_NO_TEST_LABELS",
        "test_labels_read": False,
        "task_count": 14, "rows": len(predictions), "meta_training_mdps": 720,
        "source_sha256": {"pilot_features.csv": sha256(FEATURES / "features.csv"),
                          "pilot_feature_manifest.json": sha256(FEATURES / "manifest.json"),
                          "historical_protocol.json": sha256(SOURCE / "protocol.json"),
                          "matched_forecast.py": sha256(Path(F.__file__)),
                          "script": sha256(Path(__file__))},
        "output_sha256": {"predictions": sha256(OUTPUT / "predictions_before_label_read.csv"),
                          "tuning": sha256(OUTPUT / "tuning.json")},
    }, indent=2) + "\n", encoding="utf-8")
    print("Sealed 210 BRIDGE B2 forecasts before test labels", flush=True)


if __name__ == "__main__":
    main()

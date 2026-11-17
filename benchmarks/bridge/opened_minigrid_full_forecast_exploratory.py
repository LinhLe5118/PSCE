"""Fit the unchanged 108-feature OEH head on opened MiniGrid pilot features.

The 14 test tasks already have known learner labels. This extension is explicitly
exploratory even though this script does not read those labels while forecasting.
"""

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
OEHB = ROOT / "artifacts/week8/bridge_opened_minigrid_full_oehb_exploratory_2026-10-09"
OUTPUT = ROOT / "artifacts/week8/bridge_opened_minigrid_full_forecast_exploratory_2026-10-09"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    fm = json.loads((FEATURES / "manifest.json").read_text(encoding="utf-8"))
    hm = json.loads((HORIZON / "manifest.json").read_text(encoding="utf-8"))
    bm = json.loads((OEHB / "manifest.json").read_text(encoding="utf-8"))
    if (fm["groups"] != 70 or hm["groups"] != 70 or bm["groups"] != 70 or
            bm["status"] != "EXPLORATORY_FULL_OEHB_COMPLETE_AFTER_LABEL_OPEN"):
        raise RuntimeError("Incomplete opened MiniGrid feature grid")
    if sha256(FEATURES / "features.csv") != fm["features_sha256"]:
        raise RuntimeError("Pilot feature hash changed")
    if sha256(HORIZON / "horizon_preoutcome.csv") != hm["horizon_csv_sha256"]:
        raise RuntimeError("Horizon descriptor hash changed")
    groups = sorted((OEHB / "groups").glob("*.json"))
    if len(groups) != 70 or {file.name for file in groups} != set(bm["group_sha256"]):
        raise RuntimeError("OEH-B checkpoint grid changed")
    for file in groups:
        record = json.loads(file.read_text(encoding="utf-8"))
        if (sha256(file) != bm["group_sha256"][file.name] or
                record["status"] != "EXPLORATORY_FULL_OEHB_AFTER_LABEL_OPEN"):
            raise RuntimeError(f"Bad OEH-B checkpoint: {file.name}")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    query = pd.read_csv(FEATURES / "features.csv")
    own = F.oeh_feature_rows(OEHB / "groups", HORIZON / "horizon_preoutcome.csv")
    query = query.merge(own, on=KEYS, validate="one_to_one")
    query = with_context(query)
    query["panel"] = "primary"
    query = query.sort_values(KEYS).reset_index(drop=True)
    fields = tuple(R.CANDIDATE_VARIANTS["plus_prior_profile"])
    if (len(fields) != 108 or len(query) != 210 or query.mdp_id.nunique() != 14 or
            query[KEYS].duplicated().any() or set(fields) - set(query.columns)):
        raise RuntimeError("Incomplete full 108-feature query")
    C.configure()
    historical = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    train = R.meta_training(R.OPENED, historical["oeh_final"]["structure_tag"])
    if len(train) != 10800 or train.mdp_id.nunique() != 720:
        raise RuntimeError("Historical meta-training set changed")
    probability, tuning = F.matched_forecast(train, query, fields)
    if (probability.shape != (210, 7) or not np.isfinite(probability).all() or
            not np.allclose(probability.sum(1), 1)):
        raise RuntimeError("Invalid seven-class probability grid")
    OUTPUT.mkdir(parents=True)
    predictions = query[KEYS].copy()
    predictions["method"] = "oeh_full_108_post_outcome_exploratory"
    predictions[P] = probability
    file = OUTPUT / "predictions_exploratory.csv"
    predictions.to_csv(file, index=False)
    tuning_file = OUTPUT / "tuning.json"
    tuning_file.write_text(json.dumps(tuning, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "EXPLORATORY_FULL_108_FORECAST_AFTER_LABEL_OPEN",
        "labels_already_opened_at_run": True,
        "labels_read_by_this_forecast_script": False,
        "tasks": 14, "rows": 210, "features": 108,
        "source_sha256": {
            "pilot_features": sha256(FEATURES / "features.csv"),
            "horizon": sha256(HORIZON / "horizon_preoutcome.csv"),
            "oehb_manifest": sha256(OEHB / "manifest.json"),
            "historical_protocol": sha256(SOURCE / "protocol.json"),
            "matched_forecast": sha256(Path(F.__file__)),
            "script": sha256(Path(__file__)),
        },
        "output_sha256": {"predictions": sha256(file), "tuning": sha256(tuning_file)},
    }, indent=2) + "\n", encoding="utf-8")
    print("Wrote 210 exploratory full OEH forecasts on 14 opened MiniGrid tasks")


if __name__ == "__main__":
    main()

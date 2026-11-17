"""Continue frozen third-cohort forecasting from verified pilot group outputs."""
from __future__ import annotations

import pandas as pd

from scripts.analysis import run_oeh_120_unseen as base
from scripts.analysis.develop_b2_eh_oracle_aft import KEYS
from scripts.analysis.run_oeh_120_cross_capacity import DATASET, OUTPUT, prepare
from scripts.analysis.run_unseen_validation import archive_imports, load_json


base.OUTPUT = OUTPUT
base.DATASET = DATASET


def reused_pilot_features(manifest: dict) -> pd.DataFrame:
    base.protocol()
    base.dataset()
    path = OUTPUT / "pilot_features.csv"
    seal = load_json(OUTPUT / "pilot_group_shards/merge_seal.json")
    if (seal["groups"] != 600 or seal["rows"] != 1800
            or seal["protocol_sha256"] != base.sha256(OUTPUT / "protocol.json")
            or seal["dataset_manifest_sha256"] != base.sha256(DATASET / "manifest.json")
            or seal["pilot_features_sha256"] != base.sha256(path)
            or seal["learner_outcomes_observed"] != 0):
        raise RuntimeError("Preoutcome pilot merge seal mismatch")
    frame = pd.read_csv(path)
    if len(frame) != 1800 or frame[KEYS].duplicated().any():
        raise RuntimeError("Reused pilot feature grid incomplete")
    return frame


def main() -> None:
    archive_imports()
    base.pilot_features = reused_pilot_features
    prepare()


if __name__ == "__main__":
    main()

"""Resume final pre-outcome forecast sealing from verified completed pilot features."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.analysis.compare_difficulty import FEATURES_B2
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis import run_oeh_120_unseen as run
from scripts.analysis.run_unseen_validation import archive_imports


def main() -> None:
    output = run.OUTPUT / "pilot_feature_reuse_preoutcome.json"
    if ((run.OUTPUT / "preoutcome_seal.json").exists()
            or (run.OUTPUT / "execution_plan.json").exists()):
        raise RuntimeError("Resume must precede learner outcomes and forecast seal")
    run.protocol()
    manifest = run.dataset()
    path = run.OUTPUT / "pilot_features.csv"
    frame = pd.read_csv(path)
    expected_mdps = {row["instance_id"] for row in manifest["instances"]}
    if (len(frame) != 1800 or set(frame.mdp_id) != expected_mdps
            or frame[run.KEYS].duplicated().any()
            or not (frame.groupby("mdp_id").size() == 15).all()
            or not np.isfinite(frame[list(dict.fromkeys((*FEATURES_B2, *OEH_FEATURES)))].to_numpy(dtype=float)).all()):
        raise RuntimeError("Completed pilot features are incomplete or invalid")
    record = {
        "status": "preoutcome_resume_completed_pilot_features",
        "reason": "Previous preparation completed all 600 pilot groups before the resolved-kernel fingerprint check stopped it",
        "rows": len(frame), "mdps": frame.mdp_id.nunique(),
        "pilot_features_sha256": run.sha256(path),
        "protocol_sha256": run.sha256(run.OUTPUT / "protocol.json"),
        "dataset_manifest_sha256": run.sha256(run.DATASET / "manifest.json"),
        "learner_outcomes_observed": 0,
    }
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != record:
            raise RuntimeError("Pilot-feature resume record changed")
    else:
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    archive_imports()
    run.pilot_features = lambda _: frame
    run.prepare()


if __name__ == "__main__":
    main()

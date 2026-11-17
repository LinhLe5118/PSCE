"""Exploratory temperature/censor-tilt grid on the opened failed cohort."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.develop_b2_eh_oracle_aft import KEYS, P, ROOT, sha256
from scripts.analysis.develop_oeh_difficulty import score_probabilities
from scripts.analysis.run_oeh_120_unseen import OUTPUT as FIRST


OUTPUT = ROOT / "artifacts/oeh_calibration_opened_posthoc_2026-10-01"
METHODS = ("b2_three_way", "b2_plus_eh_oracle_three_way", "oeh_three_way")
TEMPERATURES = (.8, .9, 1., 1.1, 1.2)
TILTS = (-.2, -.1, 0., .1, .2)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Calibration diagnostic already exists")
    manifest = json.loads((FIRST / "result_manifest.json").read_text(encoding="utf-8"))
    if manifest["difficulty_gate"] != "NO-GO":
        raise RuntimeError("No completed failed cohort")
    labels = pd.read_csv(FIRST / "labels.csv")
    frame = pd.read_csv(FIRST / "predictions_preoutcome.csv")
    frame = frame.merge(labels[[*KEYS, "event"]], on=KEYS, validate="many_to_one")
    frame = frame[(frame.panel == "primary") & frame.method.isin(METHODS)]
    rows = []
    for method in METHODS:
        data = frame[frame.method == method]
        p = data[P].to_numpy(dtype=float)
        y = data.event.to_numpy(dtype=int)
        for temperature in TEMPERATURES:
            for tilt in TILTS:
                adjusted = np.maximum(p, 1e-12) ** (1. / temperature)
                adjusted[:, -1] *= np.exp(tilt)
                adjusted /= adjusted.sum(axis=1, keepdims=True)
                brier, logloss = score_probabilities(adjusted, y)
                rows.append({"method": method, "temperature": temperature,
                             "censor_log_tilt": tilt,
                             "integrated_brier": float(brier.mean()),
                             "seven_class_log_loss": float(logloss.mean())})
    scores = pd.DataFrame(rows)
    best = scores.sort_values("seven_class_log_loss").groupby("method").head(3)
    OUTPUT.mkdir(parents=True)
    scores.to_csv(OUTPUT / "scores.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "posthoc_tuned_on_failed_120_not_confirmation",
        "first_result_manifest_sha256": sha256(FIRST / "result_manifest.json"),
        "first_predictions_sha256": sha256(FIRST / "predictions_preoutcome.csv"),
        "script_sha256": sha256(Path(__file__)),
        "scores_sha256": sha256(OUTPUT / "scores.csv"),
    }, indent=2) + "\n", encoding="utf-8")
    print(best.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

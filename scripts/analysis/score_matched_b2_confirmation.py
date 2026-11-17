"""Score the late-added blinded B2 architecture control after confirmation."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.compare_difficulty import sha256
from scripts.analysis.confirm_oeh_difficulty import (
    OUTPUT as CONFIRMATION, simultaneous_lower,
)
from scripts.analysis.develop_oeh_difficulty import ROOT, score_probabilities
from scripts.analysis.develop_oeh_blend import KEYS, PROBABILITY_COLUMNS

OUTPUT = ROOT / "artifacts/oeh_difficulty_matched_b2_confirmation_2026-10-01"


def main():
    import numpy as np
    import pandas as pd

    seal = json.loads((OUTPUT / "seal.json").read_text(encoding="utf-8"))
    result_path = CONFIRMATION / "result_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result["status"] != "complete" or result["jobs"] != 54000
            or seal["confirmation_protocol_sha256"] != result["protocol_sha256"]
            or seal["confirmation_preoutcome_seal_sha256"] != result["preoutcome_seal_sha256"]
            or sha256(OUTPUT / "predictions_blinded.csv") != seal["predictions_sha256"]):
        raise RuntimeError("Blinded forecast or confirmation seal mismatch")
    for name, digest in result["artifact_sha256"].items():
        if sha256(CONFIRMATION / name) != digest:
            raise RuntimeError(f"Confirmation output changed: {name}")
    for relative, digest in seal["source_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Blinded control input changed: {relative}")
    outputs = (OUTPUT / "scored_predictions.csv", OUTPUT / "mdp_scores.csv",
               OUTPUT / "summary.json", OUTPUT / "score_manifest.json")
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite matched-control scores")
    forecasts = pd.read_csv(OUTPUT / "predictions_blinded.csv")
    labels = pd.read_csv(CONFIRMATION / "labels.csv")
    if (len(forecasts) != 900 or len(labels) != 900
            or forecasts.duplicated(KEYS).any() or labels.duplicated(KEYS).any()):
        raise RuntimeError("Incomplete/duplicate forecast or label panel")
    aligned = forecasts.merge(labels[KEYS + ["event"]], on=KEYS,
                              validate="one_to_one", indicator=True)
    if len(aligned) != 900 or not aligned._merge.eq("both").all():
        raise RuntimeError("Forecast/label identity mismatch")
    aligned = aligned.drop(columns=["_merge"])
    brier, logloss = score_probabilities(
        aligned[PROBABILITY_COLUMNS].to_numpy(dtype=float),
        aligned.event.to_numpy(dtype=int))
    scored = aligned[KEYS + ["event", "method"]].copy()
    scored["integrated_brier"] = brier
    scored["seven_class_log_loss"] = logloss
    scored.to_csv(outputs[0], index=False)
    mdp_scores = scored.groupby(["mdp_id", "panel"], as_index=False).agg(
        tasks=("event", "size"), integrated_brier=("integrated_brier", "mean"),
        seven_class_log_loss=("seven_class_log_loss", "mean"))
    if len(mdp_scores) != 60 or not mdp_scores.tasks.eq(15).all():
        raise RuntimeError("Incomplete per-MDP task scores")
    mdp_scores.to_csv(outputs[1], index=False)
    original = pd.read_csv(CONFIRMATION / "mdp_scores.csv")
    original = original[original.method == "oeh_blend"]
    paired = mdp_scores.merge(original, on=["mdp_id", "panel"],
                              suffixes=("_b2", "_oeh"), validate="one_to_one")
    if len(paired) != 60:
        raise RuntimeError("Incomplete MDP pairing")
    summary = {"status": "late_added_blinded_control_not_original_registered_gate",
               "confirmation_result_sha256": sha256(result_path),
               "forecast_seal_sha256": sha256(OUTPUT / "seal.json"),
               "panels": {}}
    for panel, count in (("primary", 42), ("control", 18)):
        part = paired[paired.panel == panel].sort_values("mdp_id")
        if len(part) != count:
            raise RuntimeError(f"Wrong {panel} MDP count")
        metrics = {}
        differences = []
        for metric in ("integrated_brier", "seven_class_log_loss"):
            b = part[f"{metric}_b2"].to_numpy(dtype=float)
            o = part[f"{metric}_oeh"].to_numpy(dtype=float)
            differences.append(b - o)
            metrics[metric] = {
                "b2_matched": float(b.mean()), "oeh_blend": float(o.mean()),
                "relative_oeh_gain": float((b.mean() - o.mean()) / b.mean()),
                "mdps_favoring_oeh": int((b > o).sum()),
            }
        _, lower, critical = simultaneous_lower(np.array(differences).T)
        for metric, bound in zip(metrics, lower):
            metrics[metric]["exploratory_simultaneous_one_sided_lower_95"] = float(bound)
        summary["panels"][panel] = {
            "mdps": count, "exploratory_critical_value": critical,
            "metrics": metrics,
        }
    outputs[2].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "complete_exploratory_late_added_comparator",
        "comparison_scope": "Same architecture B2, not an original published estimator",
        "result_sha256": sha256(result_path),
        "seal_sha256": sha256(OUTPUT / "seal.json"),
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {path.name: sha256(path) for path in outputs[:3]},
    }
    outputs[3].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

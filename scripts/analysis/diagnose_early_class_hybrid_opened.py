"""Posthoc pilot-only B2/OEH probability hybrid on two opened 120-MDP suites."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.develop_b2_eh_oracle_aft import KEYS, P, ROOT, sha256
from scripts.analysis.develop_oeh_difficulty import score_probabilities
from scripts.analysis.run_oeh_120_successor import FIRST_OUTPUT, OUTPUT as SECOND


OUTPUT = ROOT / "artifacts/oeh_early_hybrid_opened_posthoc_2026-10-01"
COHORTS = {"first": FIRST_OUTPUT, "second": SECOND}
ALPHAS = (0., .25, .5, .75, 1., 1.25)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Early-hybrid diagnostic exists")
    all_rows = []
    source_hashes = {}
    for cohort, folder in COHORTS.items():
        predictions_path = folder / "predictions_preoutcome.csv"
        labels_path = folder / "labels.csv"
        source_hashes[cohort] = {"predictions": sha256(predictions_path),
                                 "labels": sha256(labels_path)}
        predictions = pd.read_csv(predictions_path)
        labels = pd.read_csv(labels_path)[KEYS + ["panel", "event"]]
        selected = {name: predictions[predictions.method == name].sort_values(KEYS).reset_index(drop=True)
                    for name in ("b2_three_way", "b2_plus_eh_oracle_three_way", "oeh_three_way")}
        keys = selected["oeh_three_way"][KEYS]
        if any(not keys.equals(frame[KEYS]) for frame in selected.values()):
            raise RuntimeError("Baseline and OEH prediction keys differ")
        truth = keys.merge(labels, on=KEYS, validate="one_to_one")
        if len(truth) != 1800:
            raise RuntimeError("Incomplete opened cohort")
        oeh = selected["oeh_three_way"][P].to_numpy(dtype=float)
        b2 = selected["b2_three_way"][P].to_numpy(dtype=float)
        eh = selected["b2_plus_eh_oracle_three_way"][P].to_numpy(dtype=float)
        candidates = {"b2": b2, "eh_oracle": eh, "oeh": oeh}
        for alpha in ALPHAS[1:]:
            hybrid = oeh.copy()
            hybrid[:, :2] *= np.power(np.clip(b2[:, :2], 1e-12, 1.) /
                                           np.clip(oeh[:, :2], 1e-12, 1.), alpha)
            hybrid /= hybrid.sum(axis=1, keepdims=True)
            candidates[f"early_b2_alpha_{alpha:g}"] = hybrid
        for method, p in candidates.items():
            brier, logloss = score_probabilities(p, truth.event.to_numpy(dtype=int))
            frame = truth.copy()
            frame["cohort"] = cohort
            frame["method"] = method
            frame["integrated_brier"] = brier
            frame["seven_class_log_loss"] = logloss
            all_rows.append(frame)
    scores = pd.concat(all_rows, ignore_index=True)
    summary = scores.groupby(["cohort", "panel", "method"], as_index=False)[
        ["integrated_brier", "seven_class_log_loss"]].mean()
    OUTPUT.mkdir(parents=True)
    scores.to_csv(OUTPUT / "scored.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "posthoc_opened_suites_not_confirmation",
        "rule": "Use B2/OEH probability ratio to modify classes x1 and x2, then renormalize",
        "alphas": ALPHAS,
        "source_sha256": source_hashes,
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {name: sha256(OUTPUT / name)
                            for name in ("scored.csv", "summary.csv")},
    }, indent=2) + "\n", encoding="utf-8")
    print(summary[summary.panel == "primary"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

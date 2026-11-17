"""Development-only matched-architecture B2 control for the OEH blend.

Both models use the same six-threshold direct fitter, grouped E1 folds, AFT
anchor, and globally fitted OOF mixture. Only the pilot feature set differs.
The already-open 60-MDP cohort cannot establish independent superiority.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS, sha256
from scripts.analysis.develop_oeh_blend import (
    KEYS, PROBABILITY_COLUMNS, fit_blend_weight,
)
from scripts.analysis.develop_oeh_difficulty import (
    ROOT, E1, OPEN_LABELS, _load, fit_thresholds, score_probabilities,
)

OUTPUT = ROOT / "artifacts/oeh_difficulty_architecture_control_2026-10-01"
DIRECT = ROOT / "artifacts/oeh_difficulty_development_2026-10-01"
E1_PREDICTIONS = E1 / "predictions.csv"
FRESH_PREDICTIONS = OPEN_LABELS.parent / "predictions_preoutcome.csv"


def selected(path, method):
    import pandas as pd

    data = pd.read_csv(path)
    data = data[data.method == method][KEYS + PROBABILITY_COLUMNS]
    if data.duplicated(KEYS).any():
        raise RuntimeError(f"Duplicate forecasts for {method}")
    return data


def main():
    import numpy as np
    import pandas as pd

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {OUTPUT}")
    source, query, paths = _load()
    source = source.sort_values(KEYS).reset_index(drop=True)
    query = query.sort_values(KEYS).reset_index(drop=True)
    original_manifest = json.loads((E1 / "manifest.json").read_text(encoding="utf-8"))
    fresh_seal = json.loads((FRESH_PREDICTIONS.parent / "preoutcome_seal.json").read_text(encoding="utf-8"))
    direct_manifest = json.loads((DIRECT / "manifest.json").read_text(encoding="utf-8"))
    if (sha256(E1_PREDICTIONS) != original_manifest["output_hashes"]["predictions.csv"]
            or sha256(FRESH_PREDICTIONS) != fresh_seal["artifact_sha256"]["predictions_preoutcome.csv"]
            or sha256(DIRECT / "predictions.csv") != direct_manifest["artifacts"]["predictions.csv"]):
        raise RuntimeError("A sealed or development forecast changed")
    anchors = (
        selected(E1_PREDICTIONS, "xgboost_aft_b2"),
        selected(FRESH_PREDICTIONS, "b2_generic"),
        selected(DIRECT / "predictions.csv", "b2_direct"),
    )
    aligned = []
    for anchor in anchors:
        merged = source[KEYS].merge(anchor, on=KEYS, validate="one_to_one") if len(anchor) == len(source) else query[KEYS].merge(anchor, on=KEYS, validate="one_to_one")
        expected = source if len(anchor) == len(source) else query
        if len(merged) != len(expected) or not merged[KEYS].equals(expected[KEYS]):
            raise RuntimeError("Forecast identity or order changed")
        aligned.append(merged[PROBABILITY_COLUMNS].to_numpy(dtype=float))
    aft_oof, aft_query, direct_query = aligned
    direct_oof = np.full((len(source), 7), np.nan)
    for panel in ("primary", "control"):
        for learner in METHODS:
            train = np.flatnonzero((source.panel == panel) & (source.algorithm == learner))
            for fold in range(5):
                fit = train[source.iloc[train].fold.to_numpy() != fold]
                val = train[source.iloc[train].fold.to_numpy() == fold]
                if (not len(fit) or not len(val)
                        or set(source.iloc[fit].mdp_id) & set(source.iloc[val].mdp_id)):
                    raise RuntimeError("Invalid grouped fold")
                direct_oof[val] = fit_thresholds(
                    source.iloc[fit][list(FEATURES_B2)].to_numpy(),
                    source.iloc[fit].event.to_numpy(),
                    source.iloc[val][list(FEATURES_B2)].to_numpy())
            print(f"Matched B2 OOF: {panel}/{learner}", flush=True)
    if not np.isfinite(direct_oof).all():
        raise RuntimeError("Incomplete direct B2 predictions")
    weight, oof_loss = fit_blend_weight(direct_oof, aft_oof,
                                        source.event.to_numpy(dtype=int))
    forecasts = weight * direct_query + (1. - weight) * aft_query
    brier, logloss = score_probabilities(forecasts, query.event.to_numpy(dtype=int))
    rows = query[KEYS + ["event"]].copy()
    rows["method"] = "b2_matched_blend"
    rows["brier"] = brier
    rows["logloss"] = logloss
    for index, name in enumerate(PROBABILITY_COLUMNS):
        rows[name] = forecasts[:, index]
    scores = rows.groupby("panel", as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    rows.to_csv(OUTPUT / "predictions.csv", index=False)
    scores.to_csv(OUTPUT / "scores.csv", index=False)
    manifest = {
        "status": "development_only_open_60_mdp_not_independent_confirmation",
        "purpose": "architecture-matched B2 ablation; no OEH mechanism features",
        "promoted": False,
        "weight_on_direct": weight,
        "e1_oof_logloss": oof_loss,
        "input_sha256": {**{name: sha256(path) for name, path in paths.items()},
                         "e1_aft_predictions": sha256(E1_PREDICTIONS),
                         "fresh_aft_predictions": sha256(FRESH_PREDICTIONS),
                         "fresh_direct_predictions": sha256(DIRECT / "predictions.csv")},
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {name: sha256(OUTPUT / name)
                            for name in ("predictions.csv", "scores.csv")},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    print(f"OOF weight on direct B2: {weight:.8f}", flush=True)
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

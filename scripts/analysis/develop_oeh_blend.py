"""Development-only probability calibration for direct OEH difficulty forecasts.

The sole mixture weight minimizes grouped out-of-fold log loss on historical E1
meta-training rows. The already-open 60 MDPs are inspected only afterward;
they cannot be claimed as confirmation for this revised candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analysis.compare_difficulty import sha256, METHODS
from scripts.analysis.develop_oeh_difficulty import (
    ROOT, E1, OLD, NEW, OPEN_LABELS, OEH_MECHANISM,
    _load, fit_thresholds, score_probabilities,
)

FEATURES = tuple(dict.fromkeys((
    "log_pilot_transitions", "log_state_action_count",
    "observed_sa_fraction", "observed_state_fraction",
    "normalized_visitation_entropy", "log1p_min_positive_sa_count",
    "trajectory_return_mean", "trajectory_return_std", "trajectory_return_iqr",
    "terminal_trajectory_fraction", "weighted_transition_entropy",
    "observed_cell_transition_entropy", "low_count_cell_fraction",
    "empirical_stochastic_cell_fraction", "low_coverage_entropy",
    "high_coverage_entropy", "bootstrap_return_mean_std",
    "bootstrap_rare_exposure_mean_std", *OEH_MECHANISM,
)))
OLD_E1_PREDICTIONS = E1 / "predictions.csv"
OLD_NEW_PREDICTIONS = OPEN_LABELS.parent / "predictions_preoutcome.csv"
OUTPUT = ROOT / "artifacts/oeh_difficulty_blend_development_2026-10-01"
KEYS = ["mdp_id", "policy_family", "algorithm", "panel"]
PROBABILITY_COLUMNS = [f"p_{index}" for index in range(7)]


def fit_blend_weight(direct, anchor, event):
    """Global log-score-optimal weight from out-of-fold training predictions."""
    import numpy as np
    from scipy.optimize import minimize_scalar

    a = np.asarray(direct, dtype=float)
    b = np.asarray(anchor, dtype=float)
    y = np.asarray(event, dtype=int)
    if (a.shape != b.shape or a.shape != (len(y), 7)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.allclose(a.sum(axis=1), 1.) or not np.allclose(b.sum(axis=1), 1.)
            or (a < 0).any() or (b < 0).any()):
        raise ValueError("Invalid out-of-fold probability arrays")
    chosen = np.arange(len(y))

    def loss(weight):
        likelihood = weight * a[chosen, y] + (1. - weight) * b[chosen, y]
        return float(-np.log(np.maximum(likelihood, 1e-12)).mean())

    optimum = minimize_scalar(loss, bounds=(0., 1.), method="bounded",
                              options={"xatol": 1e-8})
    candidates = [(0., loss(0.)), (1., loss(1.)),
                  (float(optimum.x), float(optimum.fun))]
    return min(candidates, key=lambda item: item[1])


def _anchor(path, method):
    import pandas as pd

    frame = pd.read_csv(path)
    frame = frame[frame.method == method][KEYS + PROBABILITY_COLUMNS]
    if frame.duplicated(KEYS).any():
        raise RuntimeError("Duplicate anchor forecast identity")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd

    source, query, paths = _load()
    original_manifest = json.loads((E1 / "manifest.json").read_text(encoding="utf-8"))
    open_seal = json.loads((OPEN_LABELS.parent / "preoutcome_seal.json").read_text(encoding="utf-8"))
    if (sha256(OLD_E1_PREDICTIONS) != original_manifest["output_hashes"]["predictions.csv"]
            or sha256(OLD_NEW_PREDICTIONS) != open_seal["artifact_sha256"]["predictions_preoutcome.csv"]):
        raise RuntimeError("Anchor predictions differ from preexisting seals")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite development artifact: {args.output}")
    source = source.sort_values(KEYS).reset_index(drop=True)
    query = query.sort_values(KEYS).reset_index(drop=True)
    old_source = _anchor(OLD_E1_PREDICTIONS, "xgboost_aft_b2_oeh")
    old_query = _anchor(OLD_NEW_PREDICTIONS, "b2_plus_oeh")
    source_join = source[KEYS].merge(old_source, on=KEYS, validate="one_to_one")
    query_join = query[KEYS].merge(old_query, on=KEYS, validate="one_to_one")
    if (len(source_join) != len(source) or len(query_join) != len(query)
            or not source_join[KEYS].equals(source[KEYS])
            or not query_join[KEYS].equals(query[KEYS])):
        raise RuntimeError("Anchor forecast identity/order mismatch")
    direct_oof = np.full((len(source), 7), np.nan)
    direct_query = np.full((len(query), 7), np.nan)
    for panel in ("primary", "control"):
        for algorithm in METHODS:
            tr = np.flatnonzero((source.panel == panel) & (source.algorithm == algorithm))
            qu = np.flatnonzero((query.panel == panel) & (query.algorithm == algorithm))
            for fold in range(5):
                fit = tr[source.iloc[tr].fold.to_numpy() != fold]
                val = tr[source.iloc[tr].fold.to_numpy() == fold]
                if not len(fit) or not len(val) or set(source.iloc[fit].mdp_id) & set(source.iloc[val].mdp_id):
                    raise RuntimeError("Invalid MDP-grouped calibration split")
                direct_oof[val] = fit_thresholds(source.iloc[fit][list(FEATURES)].to_numpy(),
                                                 source.iloc[fit].event.to_numpy(),
                                                 source.iloc[val][list(FEATURES)].to_numpy())
            direct_query[qu] = fit_thresholds(source.iloc[tr][list(FEATURES)].to_numpy(),
                                              source.iloc[tr].event.to_numpy(),
                                              query.iloc[qu][list(FEATURES)].to_numpy())
            print(f"Calibrated {panel}/{algorithm}", flush=True)
    if not np.isfinite(direct_oof).all() or not np.isfinite(direct_query).all():
        raise RuntimeError("Incomplete direct forecast grid")
    anchor_oof = source_join[PROBABILITY_COLUMNS].to_numpy()
    anchor_query = query_join[PROBABILITY_COLUMNS].to_numpy()
    weight, oof_logloss = fit_blend_weight(direct_oof, anchor_oof,
                                          source.event.to_numpy())
    forecasts = {
        "oeh_direct": direct_query,
        "oeh_blend": weight * direct_query + (1. - weight) * anchor_query,
        "oeh_aft_anchor": anchor_query,
    }
    rows = []
    for method, probs in forecasts.items():
        brier, logloss = score_probabilities(probs, query.event.to_numpy())
        for index, item in query.iterrows():
            rows.append({**{key: item[key] for key in KEYS}, "event": int(item.event),
                         "method": method, "brier": float(brier[index]),
                         "logloss": float(logloss[index]),
                         **{name: float(probs[index, category])
                            for category, name in enumerate(PROBABILITY_COLUMNS)}})
    detail = pd.DataFrame(rows)
    scores = detail.groupby(["panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    args.output.mkdir(parents=True)
    detail.to_csv(args.output / "predictions.csv", index=False)
    scores.to_csv(args.output / "scores.csv", index=False)
    manifest = {"status": "development_only_open_60_mdp_not_independent_confirmation",
                "scientific_revision": "direct_oeh_and_aft_out_of_fold_logloss_blend_v1",
                "promoted": False,
                "input_sha256": {**{name: sha256(path) for name, path in paths.items()},
                                 "e1_aft_oof": sha256(OLD_E1_PREDICTIONS),
                                 "fresh_aft_preoutcome": sha256(OLD_NEW_PREDICTIONS)},
                "script_sha256": sha256(Path(__file__)), "features": FEATURES,
                "weight_on_direct": weight, "e1_oof_logloss": oof_logloss,
                "weight_fit": "single global log-score optimum over all grouped E1 out-of-fold predictions",
                "artifacts": {name: sha256(args.output / name) for name in ("predictions.csv", "scores.csv")}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"OOF weight on direct: {weight:.6f}", flush=True)
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

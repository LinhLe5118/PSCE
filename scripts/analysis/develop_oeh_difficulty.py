"""Exploratory OEH difficulty revision on the already-open 60-MDP development set.

This script cannot certify a new method. It fits only on historical E1 labels,
scores the previously inspected 60 MDPs, and writes an explicit development
artifact. A fresh, pre-outcome-sealed test is required before an E1 comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analysis.compare_difficulty import FEATURES_B2, FEATURES_OEH, GRID, METHODS, sha256


ROOT = Path(__file__).resolve().parents[2]
E1 = ROOT / "artifacts/difficulty_e1_posthoc_2026-09-30"
OLD = ROOT / "artifacts/week6/oehv4e_confirmatory_v1_1/pilot_features"
NEW = ROOT / "artifacts/independent_validation_2026-09-30"
OPEN_LABELS = ROOT / "artifacts/independent_difficulty_2026-09-30/labels.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/oeh_difficulty_development_2026-10-01"

# The nearly constant LCOEH requirement/censor indicators approximate an
# unsupported planning guarantee, not the learner-specific 8/10 endpoint.
# Scale is kept so the standalone OEH score has budget context.
OEH_MECHANISM = (
    "log_pilot_transitions", "log_state_action_count",
    "oeh3_log1p_requirement", "oeh3_selected_qvi_fraction",
    "oeh3_risk_at_pilot", "oeh3_policy_instability",
    "lcoeh4_risk_at_pilot", "lcoeh4_contraction_rate",
    "lcoeh4_pessimistic_return", "lcoeh4_return_se",
    "lcoeh4_validation_support_mass", "lcoeh4_trajectory_independence_fraction",
)
FEATURE_SETS = {
    "b2_direct": FEATURES_B2,
    "oeh_direct": OEH_MECHANISM,
    "b2_oeh_direct": tuple(dict.fromkeys((*FEATURES_B2, *OEH_MECHANISM))),
}
PARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "tree_method": "hist", "learning_rate": .05, "max_depth": 2,
    "min_child_weight": 5, "lambda": 1, "seed": 20261001,
    "nthread": 1, "verbosity": 0,
}
ROUNDS = 200


def probabilities_from_cdf(raw):
    """Project six marginal event probabilities to a valid event distribution."""
    import numpy as np

    values = np.asarray(raw, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(GRID) or not np.isfinite(values).all():
        raise ValueError("Expected finite six-threshold CDF")
    # Cumulative maximum is a parameter-free order projection. It preserves
    # any already-valid CDF and never uses target labels.
    cdf = np.maximum.accumulate(np.clip(values, 0., 1.), axis=1)
    return np.column_stack((cdf[:, 0], np.diff(cdf, axis=1), 1. - cdf[:, -1]))


def score_probabilities(probabilities, event):
    import numpy as np

    probs = np.asarray(probabilities, dtype=float)
    actual = np.asarray(event, dtype=int)
    if (probs.ndim != 2 or probs.shape != (len(actual), len(GRID) + 1)
            or not np.isfinite(probs).all() or (probs < -1e-12).any()
            or not np.allclose(probs.sum(axis=1), 1.)
            or (actual < 0).any() or (actual > len(GRID)).any()):
        raise ValueError("Invalid event probabilities or labels")
    truth = actual[:, None] <= np.arange(len(GRID))[None, :]
    brier = np.mean((np.cumsum(probs, axis=1)[:, :len(GRID)] - truth) ** 2, axis=1)
    logloss = -np.log(np.maximum(probs[np.arange(len(actual)), actual], 1e-12))
    return brier, logloss


def fit_thresholds(x_train, event, x_query):
    """Six direct, equally specified binary risk models for the six budgets."""
    import numpy as np
    import xgboost as xgb

    train = xgb.DMatrix(np.asarray(x_train, dtype=float))
    query = xgb.DMatrix(np.asarray(x_query, dtype=float))
    y = np.asarray(event, dtype=int)
    cdf = np.empty((len(x_query), len(GRID)), dtype=float)
    for budget_index in range(len(GRID)):
        label = (y <= budget_index).astype(float)
        if np.unique(label).size == 1:
            cdf[:, budget_index] = label[0]
        else:
            train.set_label(label)
            model = xgb.train(PARAMS, train, num_boost_round=ROUNDS)
            cdf[:, budget_index] = model.predict(query)
    return probabilities_from_cdf(cdf)


def _load():
    import numpy as np
    import pandas as pd

    original_manifest = json.loads((E1 / "manifest.json").read_text(encoding="utf-8"))
    original_seal = json.loads((OLD / "feature_seal.json").read_text(encoding="utf-8"))
    open_manifest = json.loads((OPEN_LABELS.parent / "result_manifest.json").read_text(encoding="utf-8"))
    paths = {
        "e1_labels": E1 / "labels.csv",
        "e1_features": OLD / "features.csv",
        "development_labels": OPEN_LABELS,
        "development_features": NEW / "pilot_features.csv",
    }
    if (sha256(paths["e1_labels"]) != original_manifest["output_hashes"]["labels.csv"]
            or sha256(paths["e1_features"]) != original_seal["features_sha256"]
            or sha256(paths["development_labels"]) != open_manifest["artifact_sha256"]["labels.csv"]):
        raise RuntimeError("A sealed input changed")
    keys = ["mdp_id", "policy_family", "algorithm"]
    source = pd.read_csv(paths["e1_labels"]).merge(pd.read_csv(paths["e1_features"]), on=keys, validate="one_to_one")
    query = pd.read_csv(paths["development_labels"]).merge(pd.read_csv(paths["development_features"]), on=keys, validate="one_to_one")
    if (len(source) != 1800 or len(query) != 900 or source.mdp_id.nunique() != 120
            or query.mdp_id.nunique() != 60 or set(source.mdp_id) & set(query.mdp_id)
            or set(source.algorithm) != set(METHODS) or set(query.algorithm) != set(METHODS)
            or set(source.event) - set(range(7)) or set(query.event) - set(range(7))
            or not np.isfinite(source[list(FEATURES_OEH)].to_numpy(dtype=float)).all()
            or not np.isfinite(query[list(FEATURES_OEH)].to_numpy(dtype=float)).all()):
        raise RuntimeError("Incomplete or invalid development data")
    return source, query, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd

    source, query, paths = _load()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite development artifact: {args.output}")
    rows = []
    for panel in ("primary", "control"):
        for algorithm in METHODS:
            train = source[(source.panel == panel) & (source.algorithm == algorithm)]
            test = query[(query.panel == panel) & (query.algorithm == algorithm)]
            if (len(train) != (420 if panel == "primary" else 180)
                    or len(test) != (210 if panel == "primary" else 90)):
                raise RuntimeError("Unexpected MDP/behavior/learner stratum")
            for method, fields in FEATURE_SETS.items():
                probs = fit_thresholds(train[list(fields)].to_numpy(), train.event.to_numpy(),
                                       test[list(fields)].to_numpy())
                brier, logloss = score_probabilities(probs, test.event.to_numpy())
                for index, (_, item) in enumerate(test.iterrows()):
                    rows.append({"mdp_id": item.mdp_id, "policy_family": item.policy_family,
                                 "algorithm": algorithm, "panel": panel, "event": int(item.event),
                                 "method": method, "brier": float(brier[index]),
                                 "logloss": float(logloss[index]),
                                 **{f"p_{category}": float(probs[index, category]) for category in range(7)}})
            print(f"Development: {panel}/{algorithm} complete", flush=True)
    detail = pd.DataFrame(rows)
    scores = detail.groupby(["panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    old = pd.read_csv(OPEN_LABELS.parent / "mdp_scores.csv")
    baseline = old[old.method.isin(("b2_generic", "b2_plus_oeh"))]
    for panel in ("primary", "control"):
        for method in ("b2_generic", "b2_plus_oeh"):
            part = baseline[(baseline.panel == panel) & (baseline.method == method)]
            scores.loc[len(scores)] = [panel, method, part.mdp_id.nunique(),
                                       part.integrated_brier.mean(), part.seven_class_log_loss.mean()]
    args.output.mkdir(parents=True)
    detail.to_csv(args.output / "predictions.csv", index=False)
    scores.to_csv(args.output / "scores.csv", index=False)
    manifest = {"status": "development_only_open_60_mdp_not_independent_confirmation",
                "scientific_revision": "direct_six_threshold_brier_model_v1",
                "candidate_is_not_frozen_or_promoted": True,
                "input_sha256": {name: sha256(path) for name, path in paths.items()},
                "script_sha256": sha256(Path(__file__)), "features": FEATURE_SETS,
                "fit": {"params": PARAMS, "rounds": ROUNDS, "panel_and_learner_fits": True,
                        "cdf_projection": "rowwise cumulative maximum"},
                "artifacts": {name: sha256(args.output / name) for name in ("predictions.csv", "scores.csv")}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

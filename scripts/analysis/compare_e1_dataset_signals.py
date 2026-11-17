"""Post-hoc E1 comparison of data-only quality and coverage signals.

Uses the sealed pilot feature table and the existing E1 censored labels.  The
features are *adaptations*, not published TQ/SACo: E1 lacks their matched
reference datasets.  E1 was reused during OEH development, so these are
descriptive out-of-fold scores, not independent confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analysis.compare_difficulty import FEATURES_B2, FEATURES_OEH, GRID, aft_interval, sha256


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/week6/oehv4e_confirmatory_v1_1"
PREVIOUS = ROOT / "artifacts/difficulty_e1_posthoc_2026-09-30"
OUTPUT = ROOT / "artifacts/e1_dataset_signals_posthoc_2026-09-30"
SIGNALS = {
    "coverage_adaptation": ("log_state_action_count", "log_pilot_transitions", "observed_sa_fraction", "observed_state_fraction"),
    "quality_adaptation": ("trajectory_return_mean", "trajectory_return_std", "trajectory_return_iqr", "terminal_trajectory_fraction"),
    "coverage_quality_adaptation": ("log_state_action_count", "log_pilot_transitions", "observed_sa_fraction", "observed_state_fraction", "trajectory_return_mean", "trajectory_return_std", "trajectory_return_iqr", "terminal_trajectory_fraction"),
    "b2_generic": FEATURES_B2,
    "b2_plus_oeh": FEATURES_OEH,
}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    import numpy as np
    import pandas as pd
    from scipy.special import ndtr
    import xgboost as xgb

    labels_path = PREVIOUS / "labels.csv"
    previous_manifest = json.loads((PREVIOUS / "manifest.json").read_text(encoding="utf-8"))
    if sha256(labels_path) != previous_manifest["output_hashes"]["labels.csv"]:
        raise RuntimeError("E1 labels changed since the prior audit")
    features_path = SOURCE / "pilot_features/features.csv"
    seal = json.loads((SOURCE / "pilot_features/feature_seal.json").read_text(encoding="utf-8"))
    if sha256(features_path) != seal["features_sha256"]:
        raise RuntimeError("E1 pilot features changed since input seal")

    keys = ["mdp_id", "policy_family", "algorithm"]
    labels = pd.read_csv(labels_path)
    features = pd.read_csv(features_path)
    frame = labels.merge(features, on=keys, validate="one_to_one", how="inner")
    if len(frame) != 1800 or frame.mdp_id.nunique() != 120 or frame.duplicated(keys).any():
        raise RuntimeError("Expected 1800 distinct E1 task/behavior/learner labels on 120 MDPs")
    if set(frame.event) - set(range(7)) or set(frame.fold) - set(range(5)):
        raise RuntimeError("Invalid frozen event/fold labels")
    if not np.isfinite(frame[list(FEATURES_OEH)].to_numpy(dtype=float)).all():
        raise RuntimeError("Nonfinite pilot features")
    params = {
        "objective": "survival:aft", "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.2,
        "tree_method": "hist", "learning_rate": .05, "max_depth": 2,
        "min_child_weight": 5, "lambda": 1, "seed": 20260930,
        "nthread": 1, "verbosity": 0,
    }
    y = frame.event.to_numpy(dtype=int)
    predictions = {name: np.full((len(frame), 7), np.nan) for name in SIGNALS}
    for panel in ("primary", "control"):
        for algorithm in sorted(frame.algorithm.unique()):
            group = (frame.panel == panel) & (frame.algorithm == algorithm)
            for fold in range(5):
                train = np.flatnonzero(group & (frame.fold != fold))
                test = np.flatnonzero(group & (frame.fold == fold))
                if not len(train) or not len(test) or set(frame.iloc[train].mdp_id) & set(frame.iloc[test].mdp_id):
                    raise RuntimeError("MDP-grouped split failed")
                lower, upper = np.asarray([aft_interval(int(v)) for v in y[train]], dtype=float).T
                for name, columns in SIGNALS.items():
                    dtrain = xgb.DMatrix(frame.iloc[train][list(columns)].to_numpy(dtype=float))
                    dtrain.set_float_info("label_lower_bound", lower)
                    dtrain.set_float_info("label_upper_bound", upper)
                    model = xgb.train(params, dtrain, num_boost_round=200)
                    median = model.predict(xgb.DMatrix(frame.iloc[test][list(columns)].to_numpy(dtype=float)))
                    if not np.isfinite(median).all() or (median <= 0).any():
                        raise RuntimeError("Invalid AFT prediction")
                    cdf = ndtr((np.log(np.asarray(GRID))[None, :] - np.log(median)[:, None]) / 1.2)
                    cats = np.maximum(np.column_stack((cdf[:, 0], np.diff(cdf, axis=1), 1 - cdf[:, -1])), 0)
                    predictions[name][test] = cats / cats.sum(axis=1, keepdims=True)

    detail = []
    summary = []
    for panel in ("primary", "control"):
        indices = np.flatnonzero(frame.panel == panel)
        for name, prob_all in predictions.items():
            probs = prob_all[indices]
            if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1):
                raise RuntimeError("Incomplete probability predictions")
            events = y[indices]
            targets = (events[:, None] <= np.arange(6)[None, :]).astype(float)
            brier = ((probs.cumsum(axis=1)[:, :6] - targets) ** 2).mean(axis=1)
            logloss = -np.log(np.maximum(probs[np.arange(len(events)), events], 1e-12))
            summary.append({"panel": panel, "method": name, "n": len(indices),
                            "censored_fraction": float(np.mean(events == 6)),
                            "integrated_brier": float(np.mean(brier)),
                            "event_category_accuracy": float(np.mean(probs.argmax(axis=1) == events)),
                            "seven_class_log_loss": float(np.mean(logloss))})
            for i, index in enumerate(indices):
                row = frame.iloc[index]
                detail.append({"mdp_id": row.mdp_id, "policy_family": row.policy_family,
                               "algorithm": row.algorithm, "panel": panel, "fold": int(row.fold),
                               "event": int(events[i]), "method": name, "brier": float(brier[i]),
                               "log_loss": float(logloss[i]),
                               **{f"p_{category}": float(probs[i, category]) for category in range(7)}})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    detail_frame = pd.DataFrame(detail)
    earlier_predictions = PREVIOUS / "predictions.csv"
    if sha256(earlier_predictions) != previous_manifest["output_hashes"]["predictions.csv"]:
        raise RuntimeError("Earlier E1 prediction audit changed")
    earlier = pd.read_csv(earlier_predictions)
    earlier = earlier[earlier.method.isin(("xgboost_aft_b2", "xgboost_aft_b2_oeh"))].copy()
    earlier.method = earlier.method.map({"xgboost_aft_b2": "b2_generic", "xgboost_aft_b2_oeh": "b2_plus_oeh"})
    joined = earlier.merge(detail_frame, on=keys + ["method"], validate="one_to_one", suffixes=("_earlier", "_current"))
    if len(joined) != 3600 or any(
        not np.allclose(joined[f"p_{category}_earlier"].to_numpy(), joined[f"p_{category}_current"].to_numpy(), rtol=0, atol=1e-15)
        for category in range(7)
    ):
        raise RuntimeError("Shared B2 predictions do not reproduce the earlier E1 audit to 1e-15")
    pd.DataFrame(summary).to_csv(OUTPUT / "comparison.csv", index=False)
    detail_frame.to_csv(OUTPUT / "predictions.csv", index=False)
    rng = np.random.default_rng(20260930)
    paired_rows = []
    for panel in ("primary", "control"):
        by_mdp = detail_frame[detail_frame.panel == panel].pivot_table(
            index="mdp_id", columns="method", values="brier", aggfunc="mean"
        )
        for baseline in ("coverage_adaptation", "quality_adaptation", "coverage_quality_adaptation", "b2_generic"):
            difference = (by_mdp[baseline] - by_mdp["b2_plus_oeh"]).to_numpy()
            samples = rng.integers(0, len(difference), size=(2000, len(difference)))
            bootstrap = difference[samples].mean(axis=1)
            paired_rows.append({"panel": panel, "contrast": f"{baseline}_minus_b2_plus_oeh",
                                "mdps": len(difference), "mean_brier_improvement": float(difference.mean()),
                                "conditional_mdp_bootstrap_95_low": float(np.quantile(bootstrap, .025)),
                                "conditional_mdp_bootstrap_95_high": float(np.quantile(bootstrap, .975))})
    pd.DataFrame(paired_rows).to_csv(OUTPUT / "paired_bootstrap.csv", index=False)
    manifest = {
        "status": "posthoc_original_E1_adaptively_reused_not_confirmation",
        "mdps": 120, "labels": 1800, "pilot": "x10 E1 pilot diagnostics",
        "target": "first sustained >=8/10 successes at normalized return >=0.95 across x1,x2,x5,x10,x20,x50; >x50 right censored",
        "panels": {"primary_stochastic": 84, "control": 36},
        "information_access": "data-only pilot features; no oracle transition kernels",
        "warning": "Coverage/quality adaptations are not exact TQ/SACo: matched reference datasets are unavailable.",
        "source": "https://proceedings.mlr.press/v199/schweighofer22a.html",
        "input_hashes": {str(path.relative_to(ROOT)).replace('\\', '/'): sha256(path) for path in (labels_path, features_path, earlier_predictions)},
        "shared_prediction_verification": "all 3600 B2/B2+OEH probability vectors match the earlier E1 audit within 1e-15 before CSV serialization",
        "folds": "same MDP-grouped SHA256 modulo 5 folds as prior E1 difficulty audit; fitted within each panel and learner",
        "features": {name: list(columns) for name, columns in SIGNALS.items()},
        "xgboost_version": xgb.__version__, "xgboost_params": params,
        "rounds": 200, "script_sha256": sha256(Path(__file__)),
        "uncertainty": "2000 whole-MDP resamples of fixed out-of-fold predictions; conditional descriptive intervals",
        "output_hashes": {name: sha256(OUTPUT / name) for name in ("comparison.csv", "predictions.csv", "paired_bootstrap.csv")},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()

"""Post-hoc, censored E1 dataset-requirement comparison.

This is a research reproduction script, separate from the immutable E1 runner.
It never trains policies or reads true MDP dynamics. Run with
``python -m scripts.analysis.compare_difficulty`` from the repository root.
Requires pandas, scipy, and xgboost in addition to the project dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


GRID = (1, 2, 5, 10, 20, 50)
METHODS = ("behavior_cloning_tabular", "cfqi_tabular", "iql_tabular")
FEATURES_B2 = (
    "log_pilot_transitions", "log_state_action_count", "observed_sa_fraction",
    "observed_state_fraction", "normalized_visitation_entropy",
    "log1p_min_positive_sa_count", "trajectory_return_mean",
    "trajectory_return_std", "trajectory_return_iqr", "terminal_trajectory_fraction",
    "weighted_transition_entropy", "observed_cell_transition_entropy",
    "low_count_cell_fraction", "empirical_stochastic_cell_fraction",
    "low_coverage_entropy", "high_coverage_entropy",
    "bootstrap_return_mean_std", "bootstrap_rare_exposure_mean_std",
)
FEATURES_OEH = FEATURES_B2 + (
    "oeh3_log1p_requirement", "oeh3_selected_qvi_fraction",
    "oeh3_risk_at_pilot", "oeh3_policy_instability",
    "lcoeh4_log1p_requirement", "lcoeh4_requirement_censored",
    "lcoeh4_risk_at_pilot", "lcoeh4_contraction_rate",
    "lcoeh4_pessimistic_return", "lcoeh4_return_se",
    "lcoeh4_validation_support_mass", "lcoeh4_trajectory_independence_fraction",
)


def suffix_event(counts: tuple[int, ...], minimum: int = 8) -> int:
    """Return first sustained-success grid index, or len(counts) if censored."""
    if len(counts) != len(GRID) or any(value < 0 or value > 10 for value in counts):
        raise ValueError("Expected six success counts between zero and ten")
    for index in range(len(counts)):
        if min(counts[index:]) >= minimum:
            return index
    return len(counts)


def aft_interval(event: int) -> tuple[float, float]:
    """Map a discrete first-passing budget to a latent-time interval."""
    if not 0 <= event <= len(GRID):
        raise ValueError("Invalid event category")
    if event == len(GRID):
        return float(GRID[-1]), float("inf")
    return (0.0 if event == 0 else float(GRID[event - 1])), float(GRID[event])


def outer_fold(mdp_id: str) -> int:
    return int(hashlib.sha256(mdp_id.encode("utf-8")).hexdigest(), 16) % 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/difficulty_e1_posthoc_2026-09-30"),
        help="New output directory; sealed source artifacts are read only.",
    )
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    from scipy.special import ndtr
    import xgboost as xgb

    root = Path(__file__).resolve().parents[2]
    source = root / "artifacts/week6/oehv4e_confirmatory_v1_1"
    results_path = source / "core_run_results.csv"
    features_path = source / "pilot_features/features.csv"
    expected_feature_hash = json.loads(
        (source / "pilot_features/feature_seal.json").read_text(encoding="utf-8")
    )["features_sha256"]
    execution_hashes = json.loads(
        (source / "e1_execution_summary.json").read_text(encoding="utf-8")
    )["artifact_sha256"]
    expected_results_hash = json.loads(
        (source / "s1_corrected/s1_verdict.json").read_text(encoding="utf-8")
    )["input_sha256"]["stochastic_primary_core_run_results.csv"]
    if sha256(features_path) != expected_feature_hash:
        raise RuntimeError("Pilot feature input differs from its E1 seal")
    if sha256(results_path) != execution_hashes["core_run_results.csv"]:
        raise RuntimeError("Complete core outcomes differ from the original E1 execution summary")
    if sha256(source / "stochastic_primary_core_run_results.csv") != expected_results_hash:
        raise RuntimeError("Primary core outcomes differ from the corrected S1 seal")
    if sha256(source / "deterministic_control_core_run_results.csv") != execution_hashes["deterministic_control_core_run_results.csv"]:
        raise RuntimeError("Control core outcomes differ from the original E1 execution summary")

    results = pd.read_csv(results_path)
    features = pd.read_csv(features_path)
    keys = ["mdp_id", "policy_family", "algorithm"]
    if len(results) != 108000 or len(features) != 1800:
        raise RuntimeError("Incomplete original E1 core result or pilot feature table")
    if set(results["algorithm"]) != set(METHODS) or set(results["run_status"]) != {"observed"}:
        raise RuntimeError("Incomplete core learner outcomes")
    if not np.array_equal(
        results["success"].to_numpy(dtype=bool),
        results["normalized_return"].to_numpy(dtype=float) >= .95,
    ):
        raise RuntimeError("Success labels disagree with the frozen threshold")
    if results.duplicated(keys + ["training_size_multiplier", "training_seed"]).any():
        raise RuntimeError("Duplicate core seed identity")

    counts = results.groupby(keys + ["training_size_multiplier"])["success"].agg(["sum", "size"])
    if (counts["size"] != 10).any() or len(counts) != 1800 * len(GRID):
        raise RuntimeError("Every group/learner/budget must have ten seeds")
    wide = counts["sum"].unstack("training_size_multiplier").reindex(columns=GRID)
    if wide.isna().any().any():
        raise RuntimeError("Missing training budget")
    frame = features.set_index(keys).loc[wide.index].reset_index()
    frame["event"] = [suffix_event(tuple(int(x) for x in row)) for row in wide.to_numpy()]
    frame["panel"] = np.where(frame["mdp_id"].str.contains("lock_structured"), "control", "primary")
    frame["fold"] = frame["mdp_id"].map(outer_fold)
    if frame[frame.panel == "primary"].mdp_id.nunique() != 84 or frame[frame.panel == "control"].mdp_id.nunique() != 36:
        raise RuntimeError("E1 panel membership mismatch")
    if frame[keys].duplicated().any() or not np.isfinite(frame[list(FEATURES_OEH)].to_numpy(dtype=float)).all():
        raise RuntimeError("Invalid pilot feature rows")

    # Labels are categorical: six observed first-passing budgets plus >x50.
    # This gives a proper, censor-preserving six-threshold Brier and 7-class log loss.
    y = frame["event"].to_numpy(dtype=int)
    prediction = {name: np.empty((len(frame), 7), dtype=float) for name in
                  ("train_frequency", "xgboost_aft_b2", "xgboost_aft_b2_oeh")}
    xgb_params = {
        "objective": "survival:aft", "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.2,
        "tree_method": "hist", "learning_rate": .05, "max_depth": 2,
        "min_child_weight": 5, "lambda": 1, "seed": 20260930, "nthread": 1,
        "verbosity": 0,
    }
    for panel in ("primary", "control"):
        for algorithm in METHODS:
            mask = (frame.panel == panel) & (frame.algorithm == algorithm)
            for fold in range(5):
                train = np.flatnonzero(mask & (frame.fold != fold))
                test = np.flatnonzero(mask & (frame.fold == fold))
                if not len(train) or not len(test):
                    raise RuntimeError("Empty grouped outer fold")
                prediction["train_frequency"][test] = np.bincount(y[train], minlength=7) / len(train)
                lower, upper = np.asarray([aft_interval(int(v)) for v in y[train]], dtype=float).T
                for name, columns in (
                    ("xgboost_aft_b2", FEATURES_B2),
                    ("xgboost_aft_b2_oeh", FEATURES_OEH),
                ):
                    dtrain = xgb.DMatrix(frame.iloc[train][list(columns)].to_numpy(dtype=float))
                    dtrain.set_float_info("label_lower_bound", lower)
                    dtrain.set_float_info("label_upper_bound", upper)
                    model = xgb.train(xgb_params, dtrain, num_boost_round=200)
                    median = model.predict(xgb.DMatrix(frame.iloc[test][list(columns)].to_numpy(dtype=float)))
                    if not np.isfinite(median).all() or (median <= 0).any():
                        raise RuntimeError("Invalid XGBoost-AFT prediction")
                    cdf = ndtr((np.log(np.asarray(GRID))[None, :] - np.log(median)[:, None]) / 1.2)
                    cats = np.column_stack((cdf[:, 0], np.diff(cdf, axis=1), 1 - cdf[:, -1]))
                    prediction[name][test] = np.maximum(cats, 0) / np.maximum(cats.sum(axis=1, keepdims=True), 1e-15)

    # The two raw OEH proxies are point predictions, so only deterministic
    # threshold Brier/accuracy are reported; no invented likelihood calibration.
    state_action = np.exp(frame["log_state_action_count"].to_numpy(dtype=float))
    thresholds = np.log1p(state_action[:, None] * np.asarray(GRID)[None, :])
    for column, censored_column, name in (
        ("oeh3_log1p_requirement", None, "oeh3_raw_proxy"),
        ("lcoeh4_log1p_requirement", "lcoeh4_requirement_censored", "lcoeh4_raw_proxy"),
    ):
        predicted_pass = frame[column].to_numpy(dtype=float)[:, None] <= thresholds
        proxy_event = np.where(predicted_pass.any(axis=1), predicted_pass.argmax(axis=1), 6)
        if censored_column:
            proxy_event = np.where(frame[censored_column].to_numpy(dtype=float) >= .5, 6, proxy_event)
        prediction[name] = np.eye(7)[proxy_event]

    rows = []
    prediction_rows = []
    for panel in ("primary", "control"):
        panel_idx = np.flatnonzero(frame.panel == panel)
        for name, probs in prediction.items():
            p = probs[panel_idx]
            actual = y[panel_idx]
            cdf = p.cumsum(axis=1)[:, :6]
            targets = (actual[:, None] <= np.arange(6)[None, :]).astype(float)
            row = {
                "panel": panel, "method": name, "n": len(panel_idx),
                "censored_fraction": float(np.mean(actual == 6)),
                "integrated_brier": float(np.mean((cdf - targets) ** 2)),
                "event_category_accuracy": float(np.mean(np.argmax(p, axis=1) == actual)),
            }
            if "raw_proxy" not in name:
                row["seven_class_log_loss"] = float(-np.mean(np.log(np.maximum(p[np.arange(len(actual)), actual], 1e-12))))
            rows.append(row)
            instance_brier = np.mean((cdf - targets) ** 2, axis=1)
            instance_log_loss = -np.log(np.maximum(p[np.arange(len(actual)), actual], 1e-12))
            for local_index, global_index in enumerate(panel_idx):
                prediction_rows.append({
                    **{key: frame.iloc[global_index][key] for key in keys},
                    "panel": panel, "fold": int(frame.iloc[global_index]["fold"]),
                    "event": int(actual[local_index]), "method": name,
                    "brier": float(instance_brier[local_index]),
                    "log_loss": float(instance_log_loss[local_index]) if "raw_proxy" not in name else float("nan"),
                    "correct_category": int(np.argmax(p[local_index]) == actual[local_index]),
                    **{f"p_{category}": float(p[local_index, category]) for category in range(7)},
                })

    # Resample independent MDP units, conditional on the fixed out-of-fold
    # predictions. This is descriptive uncertainty; it does not refit trees.
    detail = pd.DataFrame(prediction_rows)
    rng = np.random.default_rng(20260930)
    bootstrap_rows = []
    for panel in ("primary", "control"):
        panel_detail = detail[detail.panel == panel]
        mdp_means = panel_detail.pivot_table(
            index="mdp_id", columns="method", values="brier", aggfunc="mean"
        )
        paired = (mdp_means["xgboost_aft_b2"] - mdp_means["xgboost_aft_b2_oeh"]).to_numpy()
        samples = rng.integers(0, len(paired), size=(2000, len(paired)))
        distribution = paired[samples].mean(axis=1)
        bootstrap_rows.append({
            "panel": panel, "contrast": "xgboost_aft_b2_minus_xgboost_aft_b2_oeh",
            "mean_brier_improvement": float(paired.mean()),
            "conditional_mdp_bootstrap_95_low": float(np.quantile(distribution, .025)),
            "conditional_mdp_bootstrap_95_high": float(np.quantile(distribution, .975)),
            "mdps": len(paired), "draws": 2000,
        })

    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    frame[keys + ["panel", "fold", "event"]].to_csv(output / "labels.csv", index=False)
    detail.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "paired_bootstrap.csv", index=False)
    manifest = {
        "status": "posthoc_original_E1_adaptively_reused_not_confirmation",
        "source_hashes": {path.relative_to(root).as_posix(): sha256(path) for path in (results_path, features_path)},
        "script_sha256": sha256(Path(__file__)),
        "grid": GRID, "threshold": .95, "minimum_successes": 8,
        "candidate_learners": METHODS, "xgboost_version": xgb.__version__,
        "xgboost_params": xgb_params, "num_boost_round": 200,
        "outer_folds": "registered_sha256_mdp_mod_5",
        "panels_fitted_separately": True,
        "feature_sets": {"B2": FEATURES_B2, "B2_OEH": FEATURES_OEH},
        "model_status": "XGBoost-AFT is an adapted censored tabular comparator; not an independently proven SOTA for this task",
        "input_scope": "x10 pilot diagnostics; x1 through x50 core outcomes define labels",
        "uncertainty": "2000 whole-MDP bootstrap resamples of fixed out-of-fold predictions; conditional descriptive intervals, no model refit",
        "output_hashes": {name: sha256(output / name) for name in
                          ("comparison.csv", "labels.csv", "predictions.csv", "paired_bootstrap.csv", "protocol.json")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print("Wrote difficulty comparison artifacts")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Exploratory nested-CV screen of principled LC-OEH-v4 feature blocks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from evaluation.aggregation.folds import assign_folds  # noqa: E402
from evaluation.aggregation.correlation import holm_adjust  # noqa: E402
from evaluation.io import (read_csv_rows, reserve_output_dir, sha256_file,
                           write_csv_rows, write_json)  # noqa: E402
from lcoeh_v4 import LC_OEH_V4_FEATURES  # noqa: E402
from return_selection import ReturnInstance, nested_return_oof  # noqa: E402
from scripts.analysis.run_calibrated_comparison import (  # noqa: E402
    cluster_bootstrap_difference, per_instance_brier,
    per_instance_log_likelihood,
)
from stochastic_evaluation import SurvivalInstance, nested_grouped_oof  # noqa: E402
from stochastic_features import (B0_NAMES, B1_EXTRA_NAMES, B2_EXTRA_NAMES,
                                 OEH_V3_EXTRA_NAMES)  # noqa: E402


GRID = (1, 2, 5, 10, 20, 50)
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
# Return regression has 22--26 standardized columns and about 240 group rows
# per learner in each outer training split.  The inherited grid stopped at 1
# and every initial inner fit selected that boundary.  Extending the same
# decimal grid to 10 and 100 lets CV represent material shrinkage relative to
# X'X eigenvalues of order n; it does not nominate a preferred value.
RETURN_RIDGES = RIDGES + (10.0, 100.0)
ALGORITHMS = ("cfqi_tabular", "iql_tabular", "behavior_cloning_tabular")
B2 = B0_NAMES + B1_EXTRA_NAMES + B2_EXTRA_NAMES
RATE = ("lcoeh4_log1p_requirement", "lcoeh4_requirement_censored",
        "lcoeh4_risk_at_pilot", "lcoeh4_contraction_rate",
        "lcoeh4_trajectory_independence_fraction")
PMS = ("lcoeh4_pessimistic_return", "lcoeh4_return_se",
       "lcoeh4_validation_support_mass",
       "lcoeh4_trajectory_independence_fraction")
CORE = ("lcoeh4_risk_at_pilot", "lcoeh4_contraction_rate",
        "lcoeh4_pessimistic_return", "lcoeh4_validation_support_mass",
        "lcoeh4_trajectory_independence_fraction")
FEATURES = {
    "B2": B2,
    "B2_OEH3": B2 + OEH_V3_EXTRA_NAMES,
    "B2_LC_RATE": B2 + RATE,
    "B2_LC_PMS": B2 + PMS,
    "B2_LC_CORE": B2 + CORE,
    "B2_LCOEH4": B2 + LC_OEH_V4_FEATURES,
    # Hybrid blocks test the scientific hypothesis that generic EH risk and
    # learner-conditional validation uncertainty are complementary.
    "B2_OEH3_LC_PMS": B2 + OEH_V3_EXTRA_NAMES + PMS,
    "B2_OEH3_LC_CORE": B2 + OEH_V3_EXTRA_NAMES + CORE,
    "B2_OEH3_LCOEH4": B2 + OEH_V3_EXTRA_NAMES + LC_OEH_V4_FEATURES,
}
BOOTSTRAP_DRAWS = 10_000


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(
        f"20260920|{label}".encode()).digest()[:4], "little")


def _outcomes(local_dir: Path):
    rows = (read_csv_rows(local_dir / "rl_run_results.csv")
            + read_csv_rows(local_dir / "bc_sanity_results.csv"))
    grouped = defaultdict(list)
    for row in rows:
        if int(float(row["training_size_multiplier"])) == 10:
            grouped[(row["mdp_id"], row["policy_family"], row["algorithm"])].append(
                float(row["normalized_return"]))
    result = {key: float(np.mean(values)) for key, values in grouped.items()}
    if any(key[:2] + (algorithm,) not in result
           for key in {item[:2] for item in result} for algorithm in ALGORITHMS):
        raise ValueError("incomplete budget-10 outcome panel")
    return result


def _pooled_predictive(predictions):
    rows = []
    for item in predictions:
        hazard = np.asarray(item["hazards"], dtype=float)
        rows.append({
            **{name: item[name] for name in
               ("mdp_id", "policy_family", "algorithm", "feature_set")},
            "log_likelihood": per_instance_log_likelihood(
                hazard, int(item["grid_index"]), bool(item["observed"])),
            "integrated_brier": per_instance_brier(
                hazard, int(item["grid_index"]), bool(item["observed"])),
        })
    lookup = {(r["mdp_id"], r["policy_family"], r["algorithm"], r["feature_set"]): r
              for r in rows}
    contrasts = []
    for feature_set in FEATURES:
        if feature_set == "B2":
            continue
        newer = [r for r in rows if r["feature_set"] == feature_set]
        older = [lookup[(r["mdp_id"], r["policy_family"], r["algorithm"], "B2")]
                 for r in newer]
        for metric in ("log_likelihood", "integrated_brier"):
            difference, low, high, p_value = cluster_bootstrap_difference(
                [r["mdp_id"] for r in newer], [r[metric] for r in newer],
                [r[metric] for r in older], num_resamples=BOOTSTRAP_DRAWS,
                confidence=.95, seed=_seed(f"predictive|{feature_set}|{metric}"))
            contrasts.append({
                "feature_set": feature_set, "contrast": f"{feature_set}-B2",
                "metric": metric, "difference": difference,
                "ci_low": low, "ci_high": high, "p_value": p_value,
                "direction_better": (difference > 0 if metric == "log_likelihood"
                                     else difference < 0),
            })
    for metric in ("log_likelihood", "integrated_brier"):
        indices = [index for index, row in enumerate(contrasts)
                   if row["metric"] == metric]
        adjusted, significant = holm_adjust(
            [contrasts[index]["p_value"] for index in indices])
        for index, p_holm, passes in zip(indices, adjusted, significant):
            contrasts[index]["p_holm"] = p_holm
            contrasts[index]["significant_after_holm"] = passes
    return rows, contrasts


def _selector_rows(predictions, outcomes, lc_features, lc_point_scores):
    outer = assign_folds({key[0] for key in outcomes}, num_folds=5)
    prediction_map = {
        (r["mdp_id"], r["policy_family"], r["algorithm"], r["feature_set"]):
        float(r["predicted_return"]) for r in predictions
    }
    groups = sorted({key[:2] for key in outcomes})
    rows = []
    methods = list(FEATURES) + ["DIRECT_LC_POINT", "DIRECT_LC_PESSIMISTIC"]
    for method in methods:
        for mdp_id, family in groups:
            fold = outer[mdp_id]
            train_groups = [group for group in groups if outer[group[0]] != fold]
            fixed = max(ALGORITHMS, key=lambda algorithm: np.mean([
                outcomes[group + (algorithm,)] for group in train_groups]))
            if method == "DIRECT_LC_POINT":
                selected = max(ALGORITHMS, key=lambda algorithm:
                    lc_point_scores[(mdp_id, family, algorithm)])
            elif method == "DIRECT_LC_PESSIMISTIC":
                selected = max(ALGORITHMS, key=lambda algorithm:
                    float(lc_features[(mdp_id, family, algorithm)][
                        "lcoeh4_pessimistic_return"]))
            else:
                selected = max(ALGORITHMS, key=lambda algorithm:
                    prediction_map[(mdp_id, family, algorithm, method)])
            oracle = max(ALGORITHMS, key=lambda algorithm:
                         outcomes[(mdp_id, family, algorithm)])
            rows.append({
                "method": method, "mdp_id": mdp_id, "policy_family": family,
                "outer_fold": fold, "selected_algorithm": selected,
                "fixed_algorithm": fixed, "oracle_algorithm": oracle,
                "selected_return": outcomes[(mdp_id, family, selected)],
                "fixed_return": outcomes[(mdp_id, family, fixed)],
                "oracle_return": outcomes[(mdp_id, family, oracle)],
            })
    summaries = []
    for method in methods:
        panel = [row for row in rows if row["method"] == method]
        difference, low, high, p_value = cluster_bootstrap_difference(
            [r["mdp_id"] for r in panel], [r["selected_return"] for r in panel],
            [r["fixed_return"] for r in panel], num_resamples=BOOTSTRAP_DRAWS,
            confidence=.95, seed=_seed(f"selector|{method}"))
        by_mdp = defaultdict(list)
        for row in panel:
            by_mdp[row["mdp_id"]].append(
                float(row["selected_return"]) - float(row["fixed_return"]))
        cluster_effects = np.asarray([np.mean(value) for value in by_mdp.values()])
        effect = float(np.mean(cluster_effects))
        sd = float(np.std(cluster_effects, ddof=1))
        # One-sided alpha=.05, 80% power. Constants are Normal quantiles,
        # fixed before the fresh cohort and not selected from candidate gates.
        required_clusters = (math.ceil(((1.6448536269514722 + .8416212335729143)
                                        * sd / effect) ** 2)
                             if effect > 0 else None)
        summaries.append({
            "method": method, "mean_selected_return": float(np.mean(
                [r["selected_return"] for r in panel])),
            "mean_fixed_return": float(np.mean([r["fixed_return"] for r in panel])),
            "mean_oracle_return": float(np.mean([r["oracle_return"] for r in panel])),
            "difference_vs_fixed": difference, "ci_low": low,
            "ci_high": high, "p_value": p_value,
            "cluster_effect_sd": sd,
            "fresh_clusters_for_80pct_power_one_sided": (
                "" if required_clusters is None else required_clusters),
        })
    adjusted, significant = holm_adjust(
        [row["p_value"] for row in summaries])
    for row, p_holm, passes in zip(summaries, adjusted, significant):
        row["p_holm"] = p_holm
        row["significant_after_holm"] = passes
    return rows, summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-b3-dir", type=Path, required=True)
    parser.add_argument("--lcoeh-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    reserve_output_dir(args.output_dir)
    base_rows = read_csv_rows(args.b0_b3_dir / "features.csv")
    labels = read_csv_rows(args.b0_b3_dir / "labels.csv")
    lc_rows = read_csv_rows(args.lcoeh_dir / "features.csv")
    lc_curve_rows = read_csv_rows(args.lcoeh_dir / "curves.csv")
    base = {(r["mdp_id"], r["policy_family"]): r for r in base_rows}
    lc = {(r["mdp_id"], r["policy_family"], r["algorithm"]): r for r in lc_rows}
    lc_point = {
        (r["mdp_id"], r["policy_family"], r["algorithm"]):
        float(r["crossfit_return_mean"])
        for r in lc_curve_rows if int(float(r["multiplier"])) == 10
    }
    outcomes = _outcomes(args.local_dir)
    expected = {(r["mdp_id"], r["policy_family"], r["algorithm"]) for r in labels}
    if set(lc) != expected or set(lc_point) != expected or set(outcomes) != expected:
        raise ValueError("LC features, outcomes, and labels are not the same panel")

    survival, returns = [], []
    for label in labels:
        key = (label["mdp_id"], label["policy_family"], label["algorithm"])
        values = {**base[key[:2]], **lc[key]}
        nested = {name: tuple(float(values[field]) for field in fields)
                  for name, fields in FEATURES.items()}
        survival.append(SurvivalInstance(
            mdp_id=key[0], policy_family=key[1], algorithm=key[2],
            grid_index=int(label["grid_index"]),
            observed=str(label["observed"]).lower() == "true", features=nested))
        returns.append(ReturnInstance(
            mdp_id=key[0], policy_family=key[1], algorithm=key[2],
            normalized_return=outcomes[key], features=nested))

    survival_predictions = nested_grouped_oof(
        survival, feature_sets=tuple(FEATURES), grid=GRID, ridge_grid=RIDGES)
    score_rows, predictive = _pooled_predictive(survival_predictions)
    return_predictions = nested_return_oof(
        returns, feature_sets=tuple(FEATURES), ridge_grid=RETURN_RIDGES)
    selection, selection_summary = _selector_rows(
        return_predictions, outcomes, lc, lc_point)

    write_csv_rows(args.output_dir / "predictive_scores.csv", score_rows,
                   list(score_rows[0]))
    write_csv_rows(args.output_dir / "predictive_contrasts.csv", predictive,
                   list(predictive[0]))
    write_csv_rows(args.output_dir / "return_oof_predictions.csv", return_predictions,
                   list(return_predictions[0]))
    write_csv_rows(args.output_dir / "selection_rows.csv", selection,
                   list(selection[0]))
    write_csv_rows(args.output_dir / "selection_summary.csv", selection_summary,
                   list(selection_summary[0]))
    report = {
        "status": "complete_exploratory_screen",
        "development_only": True,
        "consumed_v3_cohort": True,
        "confirmatory_claim_allowed": False,
        "new_development_dataset_generated": False,
        "sealed_evaluation_accessed": False,
        "research_objective_complete": False,
        "exploratory_best_predictive_block": "B2_OEH3_LC_PMS",
        "gates": {
            "mechanical_complete": True,
            "predictive_screen_after_holm": True,
            "positive_selector_screen_after_holm": False,
            "independent_confirmation": False,
            "sota_claim_supported": False,
        },
        "candidate_feature_sets": {key: list(value) for key, value in FEATURES.items()},
        "survival_ridge_grid": list(RIDGES),
        "return_ridge_grid": list(RETURN_RIDGES),
        "return_grid_extension_reason": (
            "All initial return fits selected the inherited upper boundary 1; "
            "the decimal grid was extended to 100 because standardized X'X "
            "eigenvalues scale with the roughly 240 training rows per learner."),
        "outer_folds": 5, "inner_folds": 4,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "inputs_sha256": {
            "base_features": sha256_file(args.b0_b3_dir / "features.csv"),
            "labels": sha256_file(args.b0_b3_dir / "labels.csv"),
            "lcoeh_features": sha256_file(args.lcoeh_dir / "features.csv"),
            "lcoeh_curves": sha256_file(args.lcoeh_dir / "curves.csv"),
            "rl_runs": sha256_file(args.local_dir / "rl_run_results.csv"),
            "bc_runs": sha256_file(args.local_dir / "bc_sanity_results.csv"),
        },
        "source_sha256": {
            name: sha256_file(ROOT / name) for name in (
                "code/lcoeh_v4.py", "code/return_selection.py",
                "code/stochastic_evaluation.py",
                "scripts/analysis/screen_lcoeh_v4.py",
            )
        },
        "predictive_contrasts": predictive,
        "selection_summary": selection_summary,
    }
    write_json(args.output_dir / "report.json", report)
    lines = ["# LC-OEH-v4 exploratory candidate screen", "",
             "This adaptive screen uses the already-consumed OEH-v3 development "
             "cohort. It is not confirmatory, and it did not generate another "
             "development dataset.", "",
             "## Learner selection at budget 10", "",
             "| Method | Mean return | Delta vs fixed | 95% CI | Holm p | Fresh MDPs (80% power) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in selection_summary:
        lines.append(f"| {row['method']} | {row['mean_selected_return']:.4f} | "
                     f"{row['difference_vs_fixed']:+.4f} | "
                     f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
                     f"{row['p_holm']:.4f} | "
                     f"{row['fresh_clusters_for_80pct_power_one_sided']} |")
    lines += ["", "No hyperparameter was chosen on an outer test fold. The return "
              "ridge grid extends the registered decimal grid to 10 and 100 because "
              "the initial inner fits all selected its former upper boundary 1. "
              "The hybrid blocks were an adaptive final screen on the same consumed "
              "cohort, so their corrected results remain exploratory.", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

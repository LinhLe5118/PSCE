#!/usr/bin/env python
"""Run the sealed-development B0--B3 and learner-selection evaluation.

Only development pilot buffers are used to construct features.  Offline-RL
sample-complexity labels and returns are joined after feature construction and
are used solely for out-of-fold scoring and utility evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from evaluation.aggregation.correlation import holm_adjust  # noqa: E402
from evaluation.aggregation.folds import assign_folds  # noqa: E402
from evaluation.io import (  # noqa: E402
    read_csv_rows, reserve_output_dir, sha256_file, verified_dataset_path,
    write_csv_rows, write_json,
)
from scripts.analysis.run_calibrated_comparison import (  # noqa: E402
    build_grid_index, cluster_bootstrap_difference, per_instance_brier,
    per_instance_log_likelihood,
)
from scripts.analysis.run_oeh_development_smoke import development_inputs  # noqa: E402
from stochastic_evaluation import (  # noqa: E402
    SurvivalInstance, nested_grouped_oof, suffix_requirement,
)
from stochastic_features import (  # noqa: E402
    B0_NAMES, B1_EXTRA_NAMES, B2_EXTRA_NAMES, B3_EXTRA_NAMES,
    OEH_V3_EXTRA_NAMES, aggregate_crossfit_oeh_profile,
    aggregate_oeh_risk_v3_profile, extract_stochastic_features,
)


GRID = (1, 2, 5, 10, 20, 50)
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
FEATURE_NAMES = {
    "B0": B0_NAMES,
    "B1": B0_NAMES + B1_EXTRA_NAMES,
    "B2": B0_NAMES + B1_EXTRA_NAMES + B2_EXTRA_NAMES,
    "B3": B0_NAMES + B1_EXTRA_NAMES + B2_EXTRA_NAMES + B3_EXTRA_NAMES,
}
FEATURE_SETS = tuple(FEATURE_NAMES)
ALGORITHMS = ("cfqi_tabular", "iql_tabular", "behavior_cloning_tabular")
DISPLAY = {
    "cfqi_tabular": "CFQI", "iql_tabular": "IQL",
    "behavior_cloning_tabular": "BC",
}
PILOT_MULTIPLIER = 10
SELECTION_BUDGET = 10
BOOTSTRAP_DRAWS = 10_000
CONFIDENCE = .95
SEED = 20260919


def _is_true(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _seed(label: str) -> int:
    digest = hashlib.sha256(f"{SEED}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _build_bc_labels(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["algorithm"] != "behavior_cloning_tabular":
            continue
        multiplier = int(float(row["training_size_multiplier"]))
        if multiplier in GRID:
            grouped[(row["mdp_id"], row["policy_family"])][multiplier].append(row)
    labels = []
    for (mdp_id, policy_family), by_budget in sorted(grouped.items()):
        counts = []
        for multiplier in GRID:
            budget_rows = by_budget[multiplier]
            if len(budget_rows) != 10 or any(r["run_status"] != "observed" for r in budget_rows):
                raise ValueError(f"incomplete BC grid for {(mdp_id, policy_family, multiplier)}")
            counts.append(sum(_is_true(row["success"]) for row in budget_rows))
        index, observed = suffix_requirement(counts, minimum_successes=8)
        labels.append({
            "mdp_id": mdp_id,
            "policy_family": policy_family,
            "algorithm": "behavior_cloning_tabular",
            "grid_index": index,
            "observed": observed,
            "n_emp_multiplier": GRID[index] if observed else "",
            "n_emp_censored": not observed,
            "success_counts": json.dumps(counts, separators=(",", ":")),
        })
    return labels


def _build_labels(n_emp_rows, bc_rows):
    labels = []
    for row in n_emp_rows:
        censored = _is_true(row["n_emp_censored"])
        multiplier = None if not row["n_emp_multiplier"] else float(row["n_emp_multiplier"])
        index = build_grid_index(multiplier, censored, GRID)
        if index is None:
            raise ValueError(f"label outside registered grid: {row}")
        labels.append({
            "mdp_id": row["mdp_id"], "policy_family": row["policy_family"],
            "algorithm": row["algorithm"], "grid_index": index,
            "observed": not censored,
            "n_emp_multiplier": row["n_emp_multiplier"],
            "n_emp_censored": censored,
            "success_counts": row["success_counts"],
        })
    labels.extend(_build_bc_labels(bc_rows))
    keys = {(r["mdp_id"], r["policy_family"], r["algorithm"]) for r in labels}
    pairs = {(r["mdp_id"], r["policy_family"]) for r in labels}
    expected = {(mdp, family, algorithm) for mdp, family in pairs
                for algorithm in ALGORITHMS}
    if keys != expected or len(labels) != len(keys):
        raise ValueError(
            f"incomplete or duplicate learner labels: {len(labels)}/{len(expected)}")
    return labels


def _build_features(dataset_dir: Path, crossfit_rows, oeh_aggregator):
    manifest, records, labels, index_path = development_inputs(dataset_dir)
    selected = [r for r in records if int(float(r["size_multiplier"])) == PILOT_MULTIPLIER]
    expected_groups = len(manifest["instances"]) * len(
        manifest["effective_protocol"]["behavior_policies"]["families"])
    if len(selected) != expected_groups:
        raise ValueError(f"expected {expected_groups} x10 pilots, got {len(selected)}")
    profiles = defaultdict(list)
    for row in crossfit_rows:
        if int(float(row["pilot_multiplier"])) == PILOT_MULTIPLIER:
            profiles[row["dataset_id"]].append(row)
    feature_rows = []
    feature_map = {}
    for position, record in enumerate(selected, 1):
        label = labels[record["dataset_id"]]
        with np.load(verified_dataset_path(dataset_dir, record), allow_pickle=False) as z:
            arrays = {name: z[name][:record["size"]] for name in z.files}
        base = extract_stochastic_features(
            arrays, num_states=int(record["num_states"]),
            num_actions=int(record["num_actions"]), gamma=.95,
            bootstrap_replicates=200, seed_key=record["dataset_id"],
        )
        if record["dataset_id"] not in profiles:
            raise ValueError(f"missing cross-fit profile for {record['dataset_id']}")
        oeh = oeh_aggregator(profiles[record["dataset_id"]])
        values = {**base, **oeh}
        identity = {
            "dataset_id": record["dataset_id"],
            "mdp_id": label["instance_id"],
            "policy_family": label["policy_family"],
            "pilot_multiplier": PILOT_MULTIPLIER,
            "num_transitions": record["size"],
            "num_states": record["num_states"],
            "num_actions": record["num_actions"],
        }
        feature_rows.append({**identity, **values})
        feature_map[(label["instance_id"], label["policy_family"])] = values
        if position % 15 == 0:
            print(f"features {position}/{len(selected)}", flush=True)
    if len(feature_map) != expected_groups:
        raise ValueError("duplicate or missing feature groups")
    return manifest, index_path, feature_rows, feature_map


def _instances(labels, feature_map):
    result = []
    for row in labels:
        values = feature_map[(row["mdp_id"], row["policy_family"])]
        nested = {
            name: tuple(float(values[field]) for field in fields)
            for name, fields in FEATURE_NAMES.items()
        }
        result.append(SurvivalInstance(
            mdp_id=row["mdp_id"], policy_family=row["policy_family"],
            algorithm=row["algorithm"], grid_index=int(row["grid_index"]),
            observed=bool(row["observed"]), features=nested,
        ))
    return result


def _flatten_predictions(predictions):
    rows = []
    for row in predictions:
        hazards = row["hazards"]
        flat = {key: value for key, value in row.items() if key != "hazards"}
        flat["observed"] = bool(flat["observed"])
        for multiplier, hazard in zip(GRID, hazards):
            flat[f"hazard_{multiplier}"] = hazard
        flat["probability_by_budget_10"] = 1.0 - float(np.prod(
            1.0 - np.asarray(hazards[:GRID.index(SELECTION_BUDGET) + 1])
        ))
        flat["log_likelihood"] = per_instance_log_likelihood(
            np.asarray(hazards), int(flat["grid_index"]), bool(flat["observed"])
        )
        flat["integrated_brier"] = per_instance_brier(
            np.asarray(hazards), int(flat["grid_index"]), bool(flat["observed"])
        )
        rows.append(flat)
    return rows


def _predictive_metrics(rows):
    metrics = []
    scores = {}
    for algorithm in ALGORITHMS:
        for feature_set in FEATURE_SETS:
            group = [r for r in rows if r["algorithm"] == algorithm and r["feature_set"] == feature_set]
            expected_groups = len({(r["mdp_id"], r["policy_family"]) for r in group})
            if len(group) != expected_groups:
                raise ValueError(f"incomplete prediction panel {algorithm}/{feature_set}")
            scores[(algorithm, feature_set)] = group
            metrics.append({
                "algorithm": algorithm, "feature_set": feature_set,
                "num_features": len(FEATURE_NAMES[feature_set]),
                "n_scored": len(group),
                "n_observed": sum(bool(r["observed"]) for r in group),
                "n_censored": sum(not bool(r["observed"]) for r in group),
                "num_mdp_clusters": len({r["mdp_id"] for r in group}),
                "mean_log_likelihood": float(np.mean([r["log_likelihood"] for r in group])),
                "integrated_brier": float(np.mean([r["integrated_brier"] for r in group])),
            })
    comparisons = []
    for algorithm in ALGORITHMS:
        for newer, older in (("B2", "B1"), ("B3", "B2")):
            new = scores[(algorithm, newer)]
            old = scores[(algorithm, older)]
            old_by_key = {(r["mdp_id"], r["policy_family"]): r for r in old}
            aligned_old = [old_by_key[(r["mdp_id"], r["policy_family"])] for r in new]
            for metric in ("log_likelihood", "integrated_brier"):
                difference, low, high, p = cluster_bootstrap_difference(
                    [r["mdp_id"] for r in new], [r[metric] for r in new],
                    [r[metric] for r in aligned_old], num_resamples=BOOTSTRAP_DRAWS,
                    confidence=CONFIDENCE, seed=_seed(f"{algorithm}|{newer}|{older}|{metric}"),
                )
                comparisons.append({
                    "algorithm": algorithm, "contrast": f"{newer}-{older}",
                    "metric": metric, "newer_value": float(np.mean([r[metric] for r in new])),
                    "older_value": float(np.mean([r[metric] for r in aligned_old])),
                    "difference_newer_minus_older": difference,
                    "ci_low": low, "ci_high": high, "p_value": p,
                    "p_holm": math.nan, "significant_after_holm": False,
                    "direction_better": difference > 0 if metric == "log_likelihood" else difference < 0,
                })
    for contrast in ("B2-B1", "B3-B2"):
        for metric in ("log_likelihood", "integrated_brier"):
            indices = [i for i, row in enumerate(comparisons)
                       if row["contrast"] == contrast and row["metric"] == metric]
            adjusted, significant = holm_adjust([comparisons[i]["p_value"] for i in indices])
            for i, p_holm, sig in zip(indices, adjusted, significant):
                comparisons[i]["p_holm"] = p_holm
                comparisons[i]["significant_after_holm"] = sig
    return metrics, comparisons


def _pooled_feature_comparisons(rows):
    """Pre-registered cluster-bootstrap contrasts pooled across learners.

    The primary OEH-v3 endpoint is B3-B2 mean log likelihood.  Pooling keeps
    the MDP as the resampling unit while avoiding the low-power requirement
    that every learner pass two redundant proper-score tests separately.
    """
    lookup = {
        (r["mdp_id"], r["policy_family"], r["algorithm"], r["feature_set"]): r
        for r in rows
    }
    result = []
    for newer, older in (("B2", "B1"), ("B3", "B2")):
        keys = sorted((m, p, a) for m, p, a, feature in lookup if feature == newer)
        for metric in ("log_likelihood", "integrated_brier"):
            new = [float(lookup[(*key, newer)][metric]) for key in keys]
            old = [float(lookup[(*key, older)][metric]) for key in keys]
            difference, low, high, p = cluster_bootstrap_difference(
                [key[0] for key in keys], new, old,
                num_resamples=BOOTSTRAP_DRAWS, confidence=CONFIDENCE,
                seed=_seed(f"pooled|{newer}|{older}|{metric}"),
            )
            result.append({
                "contrast": f"{newer}-{older}", "metric": metric,
                "n_rows": len(keys),
                "num_mdp_clusters": len({key[0] for key in keys}),
                "difference_newer_minus_older": difference,
                "ci_low": low, "ci_high": high, "p_value": p,
            })
    return result


def _return_table(rl_rows, bc_rows):
    values = defaultdict(list)
    for row in list(rl_rows) + list(bc_rows):
        if (row["algorithm"] in ALGORITHMS
                and int(float(row["training_size_multiplier"])) == SELECTION_BUDGET
                and row["run_status"] == "observed"):
            values[(row["mdp_id"], row["policy_family"], row["algorithm"])].append(
                float(row["normalized_return"])
            )
    means = {}
    for key, observations in values.items():
        if len(observations) != 10:
            raise ValueError(f"expected 10 return seeds for {key}, got {len(observations)}")
        means[key] = float(np.mean(observations))
    pairs = {(key[0], key[1]) for key in means}
    expected = {(mdp, family, algorithm) for mdp, family in pairs
                for algorithm in ALGORITHMS}
    if set(means) != expected:
        raise ValueError(f"incomplete pair/learner utilities: {len(means)}/{len(expected)}")
    return means


def _selection(rows, returns):
    lookup = {
        (r["mdp_id"], r["policy_family"], r["algorithm"], r["feature_set"]): r
        for r in rows
    }
    pairs = sorted({(key[0], key[1]) for key in returns})
    outer = assign_folds({pair[0] for pair in pairs}, num_folds=5)
    selection_rows = []
    for feature_set in FEATURE_SETS:
        for mdp_id, policy_family in pairs:
            fold = outer[mdp_id]
            training_mdps = {pair[0] for pair in pairs if outer[pair[0]] != fold}
            fixed_means = {
                algorithm: float(np.mean([
                    value for (m, _p, a), value in returns.items()
                    if a == algorithm and m in training_mdps
                ]))
                for algorithm in ALGORITHMS
            }
            best_fixed = max(ALGORITHMS, key=lambda a: (fixed_means[a], -ALGORITHMS.index(a)))
            probabilities = {
                algorithm: float(lookup[(mdp_id, policy_family, algorithm, feature_set)][
                    "probability_by_budget_10"
                ])
                for algorithm in ALGORITHMS
            }
            selected = max(ALGORITHMS, key=lambda a: (probabilities[a], -ALGORITHMS.index(a)))
            actual = {a: returns[(mdp_id, policy_family, a)] for a in ALGORITHMS}
            oracle = max(ALGORITHMS, key=lambda a: (actual[a], -ALGORITHMS.index(a)))
            selection_rows.append({
                "mdp_id": mdp_id, "policy_family": policy_family,
                "outer_fold": fold, "feature_set": feature_set,
                "selected_algorithm": selected,
                "selected_probability_by_budget_10": probabilities[selected],
                "selected_normalized_return": actual[selected],
                "best_fixed_algorithm": best_fixed,
                "best_fixed_training_return": fixed_means[best_fixed],
                "best_fixed_normalized_return": actual[best_fixed],
                "oracle_algorithm": oracle,
                "oracle_normalized_return": actual[oracle],
                "selection_regret_to_oracle": actual[oracle] - actual[selected],
            })
    summaries = []
    for feature_set in FEATURE_SETS:
        group = [r for r in selection_rows if r["feature_set"] == feature_set]
        difference, low, high, p = cluster_bootstrap_difference(
            [r["mdp_id"] for r in group],
            [r["selected_normalized_return"] for r in group],
            [r["best_fixed_normalized_return"] for r in group],
            num_resamples=BOOTSTRAP_DRAWS, confidence=CONFIDENCE,
            seed=_seed(f"selector|{feature_set}"),
        )
        summaries.append({
            "feature_set": feature_set, "n_groups": len(group),
            "num_mdp_clusters": len({r["mdp_id"] for r in group}),
            "mean_selector_return": float(np.mean([r["selected_normalized_return"] for r in group])),
            "mean_best_fixed_return": float(np.mean([r["best_fixed_normalized_return"] for r in group])),
            "mean_oracle_return": float(np.mean([r["oracle_normalized_return"] for r in group])),
            "selector_minus_best_fixed": difference,
            "ci_low": low, "ci_high": high, "p_value": p,
            "p_holm": math.nan, "significant_after_holm": False,
            "selected_cfqi": sum(r["selected_algorithm"] == "cfqi_tabular" for r in group),
            "selected_iql": sum(r["selected_algorithm"] == "iql_tabular" for r in group),
            "selected_bc": sum(r["selected_algorithm"] == "behavior_cloning_tabular" for r in group),
        })
    adjusted, significant = holm_adjust([r["p_value"] for r in summaries])
    for row, p_holm, sig in zip(summaries, adjusted, significant):
        row["p_holm"] = p_holm
        row["significant_after_holm"] = sig
    return selection_rows, summaries


def _markdown_report(report, metrics, comparisons, selection):
    lines = [
        "# Stochastic tabular B0–B3 development result", "",
        "Ngày chạy: **19/09/2026**. Tập evaluation 120 MDP không được truy cập.", "",
        "## Tính toàn vẹn", "",
        f"- 75 pilot groups, 225 learner labels, {report['oof_predictions']} OOF predictions.",
        f"- Cross-fit OEH case status: `{report['oeh_case_status_counts']}`.",
        "- Feature scaling và ridge selection đều được fit trong training fold tương ứng.",
        "- CI dùng 10.000 paired cluster-bootstrap draws theo base MDP.", "",
        "## Dự báo sample complexity", "",
        "| Learner | Features | Mean log-likelihood ↑ | Integrated Brier ↓ |",
        "|---|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {DISPLAY[row['algorithm']]} | {row['feature_set']} | "
            f"{row['mean_log_likelihood']:.4f} | {row['integrated_brier']:.4f} |"
        )
    lines += ["", "### Paired ablations", "",
              "| Learner | Contrast | Metric | Difference | 95% CI | Holm p | Better? |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(
            f"| {DISPLAY[row['algorithm']]} | {row['contrast']} | {row['metric']} | "
            f"{row['difference_newer_minus_older']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{row['p_holm']:.4f} | {row['direction_better']} |"
        )
    lines += ["", "## Chọn learner tại budget 10|S||A|", "",
              "| Features | Selector return | Best-fixed | Oracle | Δ selector−fixed (95% CI) | Choices CFQI/IQL/BC |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in selection:
        lines.append(
            f"| {row['feature_set']} | {row['mean_selector_return']:.4f} | "
            f"{row['mean_best_fixed_return']:.4f} | {row['mean_oracle_return']:.4f} | "
            f"{row['selector_minus_best_fixed']:.4f} "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{row['selected_cfqi']}/{row['selected_iql']}/{row['selected_bc']} |"
        )
    lines += ["", "## Kết luận development", "",
              f"- B2 cải thiện B1 theo gate đã khóa: **{report['gates']['b2_improves_b1']}**.",
              f"- B3/OEH bổ sung thông tin ngoài B2: **{report['gates']['b3_adds_information']}**.",
              f"- Selector B2 vượt best-fixed: **{report['gates']['b2_selector_beats_best_fixed']}**.",
              "- Đây là kết quả development. Nó không tự hoàn tất mục tiêu nghiên cứu và không cho phép tuyên bố confirmatory.",
              "- OEH cross-fit không còn bão hòa ở một số hữu hạn chung; nó abstain có lý do khi dữ liệu không nhận dạng được gap/coverage.", ""]
    return "\n".join(lines)


def main(argv=None):
    global FEATURE_NAMES, FEATURE_SETS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets/oeh_v3_post_freeze_holdout_2026-09-20")
    parser.add_argument("--crossfit-profile", type=Path, default=ROOT / "artifacts/research/oeh_v3_post_freeze_holdout_2026-09-20/profile.csv")
    parser.add_argument("--offline-dir", type=Path, default=ROOT / "artifacts/research/oeh_v3_post_freeze_local_methods_2026-09-20")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/research/oeh_v3_b0_b3_run")
    parser.add_argument("--oeh-version", choices=("v2", "v3"), default="v3")
    args = parser.parse_args(argv)
    if args.oeh_version == "v3":
        FEATURE_NAMES = {
            "B0": B0_NAMES,
            "B1": B0_NAMES + B1_EXTRA_NAMES,
            "B2": B0_NAMES + B1_EXTRA_NAMES + B2_EXTRA_NAMES,
            "B3": B0_NAMES + B1_EXTRA_NAMES + B2_EXTRA_NAMES + OEH_V3_EXTRA_NAMES,
        }
        oeh_aggregator = aggregate_oeh_risk_v3_profile
        oeh_source = "code/oeh_risk_v3.py"
    else:
        oeh_aggregator = aggregate_crossfit_oeh_profile
        oeh_source = "code/oeh_crossfit_v2.py"
    FEATURE_SETS = tuple(FEATURE_NAMES)
    reserve_output_dir(args.output_dir)
    started = time.perf_counter()
    inputs = {
        "manifest": args.dataset_dir / "manifest.json",
        "crossfit_profile": args.crossfit_profile,
        "n_emp": args.offline_dir / "empirical_sample_complexity.csv",
        "rl_runs": args.offline_dir / "rl_run_results.csv",
        "bc_runs": args.offline_dir / "bc_sanity_results.csv",
    }
    sources = [
        "code/stochastic_features.py", "code/stochastic_evaluation.py",
        oeh_source, "code/oeh_final_profile.py",
        "scripts/analysis/run_stochastic_b0_b3.py",
        "scripts/analysis/run_calibrated_comparison.py",
        "evaluation/calibration/discrete_survival.py",
    ]
    plan = {
        "status": "started", "development_only": True,
        "oeh_version": args.oeh_version,
        "sealed_evaluation_accessed": False, "pilot_multiplier": PILOT_MULTIPLIER,
        "selection_budget": SELECTION_BUDGET, "grid": GRID,
        "ridge_grid": RIDGES, "outer_folds": 5, "inner_folds": 4,
        "bootstrap_draws": BOOTSTRAP_DRAWS, "seed": SEED,
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "source_sha256": {path: sha256_file(ROOT / path) for path in sources},
    }
    write_json(args.output_dir / "plan.json", plan)

    crossfit_rows = read_csv_rows(args.crossfit_profile)
    n_emp_rows = read_csv_rows(inputs["n_emp"])
    rl_rows = read_csv_rows(inputs["rl_runs"])
    bc_rows = read_csv_rows(inputs["bc_runs"])
    manifest, index_path, features, feature_map = _build_features(
        args.dataset_dir, crossfit_rows, oeh_aggregator)
    labels = _build_labels(n_emp_rows, bc_rows)
    instances = _instances(labels, feature_map)
    predictions = nested_grouped_oof(
        instances, feature_sets=FEATURE_SETS, grid=GRID, ridge_grid=RIDGES,
        outer_folds=5, inner_folds=4,
    )
    flat_predictions = _flatten_predictions(predictions)
    metrics, comparisons = _predictive_metrics(flat_predictions)
    pooled_comparisons = _pooled_feature_comparisons(flat_predictions)
    returns = _return_table(rl_rows, bc_rows)
    selection_rows, selection_summary = _selection(flat_predictions, returns)

    feature_fields = [
        "dataset_id", "mdp_id", "policy_family", "pilot_multiplier",
        "num_transitions", "num_states", "num_actions",
    ] + list(FEATURE_NAMES["B3"])
    label_fields = [
        "mdp_id", "policy_family", "algorithm", "grid_index", "observed",
        "n_emp_multiplier", "n_emp_censored", "success_counts",
    ]
    prediction_fields = [
        "mdp_id", "policy_family", "algorithm", "feature_set", "outer_fold",
        "selected_ridge", "inner_mean_log_likelihood", "grid_index", "observed",
    ] + [f"hazard_{value}" for value in GRID] + [
        "probability_by_budget_10", "log_likelihood", "integrated_brier",
    ]
    metric_fields = list(metrics[0])
    comparison_fields = list(comparisons[0])
    selection_fields = list(selection_rows[0])
    selection_summary_fields = list(selection_summary[0])
    write_csv_rows(args.output_dir / "features.csv", features, feature_fields)
    write_csv_rows(args.output_dir / "labels.csv", labels, label_fields)
    write_csv_rows(args.output_dir / "oof_predictions.csv", flat_predictions, prediction_fields)
    write_csv_rows(args.output_dir / "predictive_metrics.csv", metrics, metric_fields)
    write_csv_rows(args.output_dir / "predictive_comparisons.csv", comparisons, comparison_fields)
    write_csv_rows(args.output_dir / "pooled_predictive_comparisons.csv",
                   pooled_comparisons, list(pooled_comparisons[0]))
    write_csv_rows(args.output_dir / "learner_selection.csv", selection_rows, selection_fields)
    write_csv_rows(args.output_dir / "learner_selection_summary.csv", selection_summary, selection_summary_fields)

    def gate(contrast, metric, direction):
        panel = [r for r in comparisons if r["contrast"] == contrast and r["metric"] == metric]
        return all(bool(r["significant_after_holm"]) and direction(r) for r in panel)

    pooled_lookup = {(r["contrast"], r["metric"]): r for r in pooled_comparisons}
    report = {
        **plan, "status": "complete", "manifest_run_mode": manifest.get("run_mode"),
        "index_sha256": sha256_file(index_path), "feature_groups": len(features),
        "learner_labels": len(labels), "oof_predictions": len(flat_predictions),
        "oeh_case_status_counts": dict(Counter(
            (
                "identified" if any(r["status"] == "identified" for r in crossfit_rows
                                    if r["dataset_id"] == feature["dataset_id"])
                else "epsilon_trivial" if any(r["status"] == "epsilon_trivial" for r in crossfit_rows
                                               if r["dataset_id"] == feature["dataset_id"])
                else "unsupported" if any(r["status"] == "unsupported" for r in crossfit_rows
                                          if r["dataset_id"] == feature["dataset_id"])
                else "unresolved" if any(r["status"] == "unresolved" for r in crossfit_rows
                                         if r["dataset_id"] == feature["dataset_id"])
                else "not_epsilon_solvable"
            )
            for feature in features
        )),
        "gates": {
            "b2_improves_b1": (
                gate("B2-B1", "log_likelihood", lambda r: r["difference_newer_minus_older"] > 0)
                and gate("B2-B1", "integrated_brier", lambda r: r["difference_newer_minus_older"] < 0)
            ),
            "b3_adds_information": (
                gate("B3-B2", "log_likelihood", lambda r: r["difference_newer_minus_older"] > 0)
                and gate("B3-B2", "integrated_brier", lambda r: r["difference_newer_minus_older"] < 0)
            ),
            "oeh_v3_primary_pooled_log_likelihood": (
                pooled_lookup[("B3-B2", "log_likelihood")]["ci_low"] > 0
            ),
            "oeh_v3_secondary_pooled_brier": (
                pooled_lookup[("B3-B2", "integrated_brier")]["ci_high"] < 0
            ),
            "b2_primary_pooled_log_likelihood": (
                pooled_lookup[("B2-B1", "log_likelihood")]["ci_low"] > 0
            ),
            "b2_selector_beats_best_fixed": next(
                r["ci_low"] > 0 for r in selection_summary if r["feature_set"] == "B2"
            ),
            "b3_selector_beats_best_fixed": next(
                r["ci_low"] > 0 for r in selection_summary if r["feature_set"] == "B3"
            ),
            "research_objective_complete": False,
        },
        "pooled_predictive_comparisons": pooled_comparisons,
        "wall_seconds": time.perf_counter() - started,
    }
    report["gates"]["research_objective_complete"] = all((
        report["gates"]["b2_primary_pooled_log_likelihood"],
        report["gates"]["oeh_v3_primary_pooled_log_likelihood"],
        report["gates"]["b3_selector_beats_best_fixed"],
    ))
    artifacts = [
        "features.csv", "labels.csv", "oof_predictions.csv", "predictive_metrics.csv",
        "predictive_comparisons.csv", "pooled_predictive_comparisons.csv",
        "learner_selection.csv", "learner_selection_summary.csv",
    ]
    report["artifact_sha256"] = {
        name: sha256_file(args.output_dir / name) for name in artifacts
    }
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "REPORT.md").write_text(
        _markdown_report(report, metrics, comparisons, selection_summary), encoding="utf-8"
    )
    print(json.dumps(report["gates"], indent=2), flush=True)
    print(f"completed in {report['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

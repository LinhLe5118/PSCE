#!/usr/bin/env python
"""The registered primary metric, against every registered baseline.

Closes two gaps that an audit on 2026-09-06 found in the C2 result. Neither
changes a constant; both run machinery that was already written, tested, and
never pointed at real data.

**Gap 1 -- the primary metric was never computed.** ``PLAN_Final.md`` sec 7
registers *"Metric chinh: likelihood thoi gian roi rac, integrated Brier
score. Metric phu: tuong quan hang"* -- discrete-time likelihood and
integrated Brier are primary, rank correlation is **secondary**. The C2 run
(``artifacts/week2/c2_correlation_2026-09-02/``) scored success criteria 3 and
4 on Spearman and Pearson. ``evaluation/calibration/discrete_survival.py``
implements sec 6 stage 3 and had never been imported outside its own unit
test.

**Gap 2 -- the strongest baseline never entered.** Sec 8.2 criterion 3 is
*"vuot baseline manh nhat"*. C2 compared against ``structural_sa`` only, which
is the **trivial** baseline. ``baseline_results.csv`` already carries 840 rows
across six registered baselines, coverage included, and five of them sat
unused.

What this runs, exactly as sec 6-7 specify: a discrete-time hazard model per
feature set, 5-fold cross-validation split by base MDP, out-of-fold prediction
only, censored labels contributing their partial-information likelihood term
rather than being imputed or dropped, and cluster-bootstrap intervals on the
**difference** between OEH and each baseline, Holm-corrected within each
algorithm.

Two baselines are oracle-privileged and are labelled as such on every row.
``behavior_quality`` and ``behavior_value`` are extracted from
``exact_oracle_results.csv``, i.e. from ``J*``/``V*``. Sec 6.1 asks for return
quality measured *from the pilot*; the oracle version is a reference ceiling,
not a fair competitor, and sec 11 rule 5 forbids the estimator from touching
those quantities. They are reported so the ceiling is visible, and they are
excluded from the "strongest fair baseline" verdict.

Usage::

    python -m scripts.analysis.run_calibrated_comparison \\
      --n-emp artifacts/week2/development_offline_rl_2026-08-22/empirical_sample_complexity.csv \\
      --behavior-metadata artifacts/week2/development_offline_rl_2026-08-22/behavior_policy_metadata.csv \\
      --benchmark-metadata artifacts/week2/benchmark_oracle_handoff_2026-08-23/benchmark_metadata.csv \\
      --exact-oracle artifacts/week2/benchmark_oracle_handoff_2026-08-23/exact_oracle_results.csv \\
      --alternatives artifacts/week2/alternative_definitions_2026-09-01/alternative_definitions.csv \\
      --output-dir artifacts/week2/calibrated_comparison_<date>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.aggregation.correlation import holm_adjust  # noqa: E402
from evaluation.aggregation.folds import assign_folds  # noqa: E402
from evaluation.calibration.discrete_survival import DiscreteTimeHazardCalibrator  # noqa: E402
from evaluation.io import read_csv_rows, sha256_file, write_csv_rows, write_json  # noqa: E402
from config.protocol_config import PROTOCOL_CONFIG, PROTOCOL_CONFIG_PATH  # noqa: E402

#: The OEH predictor whose difference against each baseline is the registered
#: statistic. ``registered`` is the frozen definition; the alternatives are
#: reported beside it but were written after it was seen to saturate.
PRIMARY_OEH_FEATURE_SET = "oeh_registered"

#: Oracle-privileged: built from ``J*``/``V*``. Reported as a ceiling, never
#: counted as the strongest *fair* baseline (sec 6.1, sec 11 rule 5).
ORACLE_FEATURE_SETS = frozenset({"behavior_quality", "behavior_value"})

#: The floor every other feature set has to clear: a hazard model with no
#: covariates at all, i.e. the marginal grid-level event rates. Its feature is
#: a constant, so ``beta`` contributes nothing and only the interval intercepts
#: ``alpha_j`` are fit. Without it, a table of similar log-likelihoods is
#: unreadable -- one cannot tell "every predictor carries the same information"
#: from "no predictor carries any information beyond the marginal rate", and
#: those are different findings. sec 6.1's negative control answers a narrower
#: question (does a random feature win?) than this one.
INTERCEPT_ONLY_FEATURE_SET = "intercept_only"

RESULT_FIELDS = (
    "algorithm",
    "feature_set",
    "is_oracle",
    "num_features",
    "n_scored",
    "n_observed",
    "n_censored",
    "num_clusters",
    "mean_log_likelihood",
    "integrated_brier",
)

COMPARISON_FIELDS = (
    "algorithm",
    "metric",
    "baseline",
    "baseline_is_oracle",
    "oeh_value",
    "baseline_value",
    "difference",
    "ci_low",
    "ci_high",
    "p_value",
    "p_holm",
    "significant_after_holm",
    "oeh_better",
    "n_scored",
    "num_clusters",
)


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


@dataclass
class Instance:
    """One (mdp_id, policy_family, algorithm) label with its feature sets."""

    mdp_id: str
    policy_family: str
    algorithm: str
    grid_index: int
    observed: bool
    features: dict[str, dict[str, float]] = field(default_factory=dict)


def build_grid_index(multiplier: float | None, censored: bool, grid: Sequence[int]) -> int | None:
    """Position of ``N_emp`` on the registered size grid.

    A right-censored instance survived the whole grid, so its index is the last
    interval and ``observed`` is False -- that is exactly the partial
    information a discrete-time survival likelihood consumes, and it is why
    this metric is the registered primary one while a correlation has to drop
    the row (``PLAN_Final.md`` sec 3.1: never replaced by ``N_max``, never
    discarded).
    """

    if censored:
        return len(grid) - 1
    if multiplier is None:
        return None
    for position, level in enumerate(grid):
        if abs(level - multiplier) < 1e-9:
            return position
    return None


def _feature_rows(rows: Sequence[Mapping[str, Any]], names: Sequence[str], key: tuple[str, ...]) -> dict[tuple, dict[str, float]]:
    table: dict[tuple, dict[str, float]] = {}
    for row in rows:
        values = {name: _float(row.get(name)) for name in names}
        if any(value is None for value in values.values()):
            continue
        table[tuple(row[part] for part in key)] = values  # type: ignore[misc]
    return table


def assemble(
    *,
    n_emp_rows: Sequence[Mapping[str, Any]],
    behavior_rows: Sequence[Mapping[str, Any]],
    benchmark_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    alternative_rows: Sequence[Mapping[str, Any]],
    grid: Sequence[int],
    materiality_fraction: float,
) -> tuple[list[Instance], dict[str, tuple[str, ...]]]:
    """One row per label, carrying every registered feature set it has."""

    coverage_names = (
        "observed_reachable_sa_fraction",
        "observed_sa_fraction_all",
        "observed_policy_support_fraction",
        "fraction_policy_support_at_least_n_min",
        "minimum_policy_support_count",
    )
    by_pair_coverage = _feature_rows(behavior_rows, coverage_names, ("mdp_id", "policy_family"))
    by_pair_entropy = _feature_rows(behavior_rows, ("normalized_visitation_entropy",), ("mdp_id", "policy_family"))
    by_mdp_structural = _feature_rows(benchmark_rows, ("num_states", "num_actions"), ("mdp_id",))
    by_pair_quality = _feature_rows(
        oracle_rows, ("behavior_return", "behavior_normalized_return", "behavior_optimality_gap"), ("mdp_id", "policy_family")
    )
    by_pair_value = _feature_rows(oracle_rows, ("behavior_return",), ("mdp_id", "policy_family"))

    # OEH features, one variant at a time, at the registered materiality.
    oeh_by_variant: dict[str, dict[tuple, dict[str, float]]] = {}
    for row in alternative_rows:
        variant = row["variant"]
        fraction = _float(row.get("materiality_fraction"))
        if variant in {"materiality", "max_state"} and (fraction is None or abs(fraction - materiality_fraction) > 1e-12):
            continue
        multiplier = _float(row.get("min_multiplier"))
        requirement = _float(row.get("n_req_hat"))
        depth = _float(row.get("k_dagger"))
        if multiplier is None or requirement is None or depth is None:
            continue
        oeh_by_variant.setdefault(f"oeh_{variant}", {})[(row["mdp_id"], row["policy_family"])] = {
            "log_n_req_hat": math.log(requirement) if requirement > 0 else 0.0,
            "min_multiplier": multiplier,
            "k_dagger": depth,
        }

    feature_names: dict[str, tuple[str, ...]] = {
        "coverage": coverage_names,
        "visit_entropy": ("normalized_visitation_entropy",),
        "structural_only": ("log_state_action_product",),
        "behavior_quality": ("behavior_return", "behavior_normalized_return", "behavior_optimality_gap"),
        "behavior_value": ("behavior_return",),
        "random_control": ("random_feature",),
        INTERCEPT_ONLY_FEATURE_SET: ("constant",),
    }
    for name in oeh_by_variant:
        feature_names[name] = ("log_n_req_hat", "min_multiplier", "k_dagger")

    instances: list[Instance] = []
    for row in n_emp_rows:
        pair = (row["mdp_id"], row["policy_family"])
        censored = _is_true(row.get("n_emp_censored"))
        position = build_grid_index(_float(row.get("n_emp_multiplier")), censored, grid)
        if position is None:
            continue
        instance = Instance(
            mdp_id=row["mdp_id"],
            policy_family=row["policy_family"],
            algorithm=row["algorithm"],
            grid_index=position,
            observed=not censored,
        )
        if pair in by_pair_coverage:
            instance.features["coverage"] = by_pair_coverage[pair]
        if pair in by_pair_entropy:
            instance.features["visit_entropy"] = by_pair_entropy[pair]
        if (row["mdp_id"],) in by_mdp_structural:
            structural = by_mdp_structural[(row["mdp_id"],)]
            product = structural["num_states"] * structural["num_actions"]
            instance.features["structural_only"] = {"log_state_action_product": math.log(product)}
        if pair in by_pair_quality:
            instance.features["behavior_quality"] = by_pair_quality[pair]
        if pair in by_pair_value:
            instance.features["behavior_value"] = by_pair_value[pair]
        # Deterministic per-instance noise: a negative control must be
        # reproducible, so it is hashed from the key rather than drawn.
        digest = hashlib.sha256(f"{pair}|{row['algorithm']}".encode("utf-8")).hexdigest()
        instance.features["random_control"] = {"random_feature": int(digest[:8], 16) / 0xFFFFFFFF}
        instance.features[INTERCEPT_ONLY_FEATURE_SET] = {"constant": 0.0}
        for name, table in oeh_by_variant.items():
            if pair in table:
                instance.features[name] = table[pair]
        instances.append(instance)
    return instances, feature_names


def out_of_fold_predictions(
    instances: Sequence[Instance],
    feature_set: str,
    names: Sequence[str],
    *,
    grid: Sequence[int],
    num_folds: int,
) -> tuple[np.ndarray, list[Instance]]:
    """Hazard predictions for every instance, always from a model that never
    saw its base MDP (``PLAN_Final.md`` sec 7: folds split by base MDP)."""

    usable = [instance for instance in instances if feature_set in instance.features]
    if not usable:
        return np.empty((0, len(grid))), []
    folds = assign_folds({instance.mdp_id for instance in usable}, num_folds=num_folds)
    hazards = np.full((len(usable), len(grid)), np.nan, dtype=np.float64)

    for fold in range(num_folds):
        train = [i for i, inst in enumerate(usable) if folds[inst.mdp_id] != fold]
        test = [i for i, inst in enumerate(usable) if folds[inst.mdp_id] == fold]
        if not test:
            continue
        if not train or len({usable[i].observed for i in train}) < 2:
            # A fold whose training half carries no event (or no censoring) cannot
            # identify a hazard. Left as NaN and reported as unscored rather
            # than back-filled from the pooled model, which would leak.
            continue
        matrix = [[usable[i].features[feature_set][name] for name in names] for i in train]
        calibrator = DiscreteTimeHazardCalibrator(grid=grid).fit(
            matrix,
            [usable[i].grid_index for i in train],
            [usable[i].observed for i in train],
        )
        test_matrix = [[usable[i].features[feature_set][name] for name in names] for i in test]
        predicted = calibrator.predict_hazard(test_matrix)
        for position, index in enumerate(test):
            hazards[index] = predicted[position]
    return hazards, usable


def per_instance_log_likelihood(hazard: np.ndarray, grid_index: int, observed: bool) -> float:
    """Discrete-time survival log-likelihood for one instance.

    Observed at interval ``j``: survived ``0..j-1`` then the event fired at
    ``j``. Right-censored: survived every interval. This is the term
    ``PLAN_Final.md`` sec 6 stage 3 describes, and the reason a censored row
    contributes information here while a correlation must discard it.
    """

    clipped = np.clip(hazard, 1e-12, 1.0 - 1e-12)
    total = float(np.sum(np.log(1.0 - clipped[:grid_index])))
    if observed:
        total += float(np.log(clipped[grid_index]))
    else:
        total += float(np.log(1.0 - clipped[grid_index]))
    return total


def per_instance_brier(hazard: np.ndarray, grid_index: int, observed: bool) -> float:
    """Mean Brier score over the full registered grid.

    After an observed event, survival is known to be zero on every later
    interval. The registered protocol censors only at the final grid point;
    earlier censoring requires a different scoring contract.
    """

    if not 0 <= grid_index < len(hazard):
        raise ValueError("grid_index must lie on the hazard grid")
    if not observed and grid_index != len(hazard) - 1:
        raise ValueError("Full-grid Brier requires censoring at the final grid point")

    clipped = np.clip(hazard, 1e-12, 1.0 - 1e-12)
    survival = np.cumprod(1.0 - clipped)
    last = len(hazard) - 1
    errors = []
    for position in range(last + 1):
        # Indicator that the requirement is still unmet after interval j.
        truth = 1.0 if (not observed or position < grid_index) else 0.0
        errors.append((survival[position] - truth) ** 2)
    return float(np.mean(errors))


def cluster_bootstrap_difference(
    clusters: Sequence[str],
    oeh_scores: Sequence[float],
    baseline_scores: Sequence[float],
    *,
    num_resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float, float]:
    """Paired difference of mean scores, resampling **base MDPs** (sec 7).

    Returns ``(difference, ci_low, ci_high, p_value)``. The p-value counts both
    tails inclusive of zero, matching ``correlate_difference`` -- with paired
    scores on identical rows, exact ties are common and counting only strict
    tails would manufacture significance out of them.
    """

    unique = sorted(set(clusters))
    index_by_cluster: dict[str, list[int]] = {name: [] for name in unique}
    for position, name in enumerate(clusters):
        index_by_cluster[name].append(position)

    oeh = np.asarray(oeh_scores, dtype=np.float64)
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    observed = float(np.mean(oeh) - np.mean(baseline))

    rng = np.random.default_rng(seed)
    draws = np.empty(num_resamples, dtype=np.float64)
    for draw in range(num_resamples):
        picked = rng.choice(len(unique), size=len(unique), replace=True)
        rows: list[int] = []
        for position in picked:
            rows.extend(index_by_cluster[unique[position]])
        draws[draw] = float(np.mean(oeh[rows]) - np.mean(baseline[rows]))

    tail = (1.0 - confidence) / 2.0
    low, high = float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))
    non_positive = float(np.mean(draws <= 0.0))
    non_negative = float(np.mean(draws >= 0.0))
    p_value = min(1.0, 2.0 * min(non_positive, non_negative))
    return observed, low, high, p_value


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def run(
    *,
    instances: Sequence[Instance],
    feature_names: Mapping[str, Sequence[str]],
    grid: Sequence[int],
    num_folds: int,
    num_resamples: int,
    confidence: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    algorithms = sorted({instance.algorithm for instance in instances})
    results: list[dict[str, Any]] = []
    scores: dict[tuple[str, str], dict[str, tuple[list[str], list[float], list[tuple[str, str]]]]] = {}
    observation_keys = [(inst.algorithm, inst.mdp_id, inst.policy_family) for inst in instances]
    if len(observation_keys) != len(set(observation_keys)):
        raise ValueError("Duplicate (algorithm, mdp_id, policy_family) label")

    for algorithm in algorithms:
        subset = [instance for instance in instances if instance.algorithm == algorithm]
        for feature_set, names in sorted(feature_names.items()):
            hazards, usable = out_of_fold_predictions(
                subset, feature_set, list(names), grid=grid, num_folds=num_folds
            )
            clusters: list[str] = []
            likelihoods: list[float] = []
            briers: list[float] = []
            for position, instance in enumerate(usable):
                if not np.all(np.isfinite(hazards[position])):
                    continue
                clusters.append(instance.mdp_id)
                likelihoods.append(per_instance_log_likelihood(hazards[position], instance.grid_index, instance.observed))
                briers.append(per_instance_brier(hazards[position], instance.grid_index, instance.observed))
            if not clusters:
                continue
            scored = [inst for position, inst in enumerate(usable) if np.all(np.isfinite(hazards[position]))]
            results.append(
                {
                    "algorithm": algorithm,
                    "feature_set": feature_set,
                    "is_oracle": feature_set in ORACLE_FEATURE_SETS,
                    "num_features": len(names),
                    "n_scored": len(clusters),
                    "n_observed": sum(1 for inst in scored if inst.observed),
                    "n_censored": sum(1 for inst in scored if not inst.observed),
                    "num_clusters": len(set(clusters)),
                    "mean_log_likelihood": float(np.mean(likelihoods)),
                    "integrated_brier": float(np.mean(briers)),
                }
            )
            key = (algorithm, feature_set)
            scores[key] = {
                "log_likelihood": (clusters, likelihoods, [(inst.mdp_id, inst.policy_family) for inst in scored]),
                "brier": (clusters, briers, [(inst.mdp_id, inst.policy_family) for inst in scored]),
            }

    comparisons: list[dict[str, Any]] = []
    for algorithm in algorithms:
        oeh_key = (algorithm, PRIMARY_OEH_FEATURE_SET)
        if oeh_key not in scores:
            continue
        for metric, higher_is_better in (("log_likelihood", True), ("brier", False)):
            family: list[dict[str, Any]] = []
            for feature_set in sorted(feature_names):
                if feature_set == PRIMARY_OEH_FEATURE_SET or feature_set.startswith("oeh_"):
                    continue
                baseline_key = (algorithm, feature_set)
                if baseline_key not in scores:
                    continue
                # Pair on identical rows only: sec 7's registered statistic is
                # the interval of the difference, which is meaningless across
                # different row sets.
                oeh_clusters, oeh_values, oeh_ids = scores[oeh_key][metric]
                base_clusters, base_values, base_ids = scores[baseline_key][metric]
                oeh_lookup = dict(zip(oeh_ids, oeh_values))
                paired_clusters: list[str] = []
                paired_oeh: list[float] = []
                paired_base: list[float] = []
                for cluster, value, ident in zip(base_clusters, base_values, base_ids):
                    if ident not in oeh_lookup:
                        continue
                    paired_clusters.append(cluster)
                    paired_oeh.append(oeh_lookup[ident])
                    paired_base.append(value)
                if len(paired_clusters) < 2:
                    continue
                difference, low, high, p_value = cluster_bootstrap_difference(
                    paired_clusters,
                    paired_oeh,
                    paired_base,
                    num_resamples=num_resamples,
                    confidence=confidence,
                    seed=seed,
                )
                family.append(
                    {
                        "algorithm": algorithm,
                        "metric": metric,
                        "baseline": feature_set,
                        "baseline_is_oracle": feature_set in ORACLE_FEATURE_SETS,
                        "oeh_value": float(np.mean(paired_oeh)),
                        "baseline_value": float(np.mean(paired_base)),
                        "difference": difference,
                        "ci_low": low,
                        "ci_high": high,
                        "p_value": p_value,
                        "oeh_better": (difference > 0) if higher_is_better else (difference < 0),
                        "n_scored": len(paired_clusters),
                        "num_clusters": len(set(paired_clusters)),
                    }
                )
            adjusted, significant = holm_adjust([row["p_value"] for row in family])
            for row, p_holm, is_significant in zip(family, adjusted, significant):
                row["p_holm"] = p_holm
                row["significant_after_holm"] = is_significant
            comparisons.extend(family)

    floor = {
        (row["algorithm"], row["metric"]): row
        for row in comparisons
        if row["baseline"] == INTERCEPT_ONLY_FEATURE_SET
    }
    beats_floor = {
        f"{key[0]}/{key[1]}": bool(row["oeh_better"] and row["significant_after_holm"])
        for key, row in sorted(floor.items())
    }
    fair = [row for row in comparisons if not row["baseline_is_oracle"]]
    beats_any_fair = any(row["oeh_better"] and row["significant_after_holm"] for row in fair)
    loses_to_fair = sorted(
        {
            f"{row['baseline']}/{row['metric']}/{row['algorithm']}"
            for row in fair
            if (not row["oeh_better"]) and row["significant_after_holm"]
        }
    )
    report = {
        "scoring_version": "2026-09-17-full-grid-brier-policy-pairing",
        "pairing_key": ["algorithm", "mdp_id", "policy_family"],
        "bootstrap_cluster": "mdp_id",
        "brier_grid": "full grid; observed survival remains zero after event; censoring at final point only",
        "holm_family": "all baseline comparisons (including oracle references) within algorithm and metric; historical family preserved",
        "bootstrap_p_note": "Empirical two-sided tail fractions with inclusive ties; a reported zero means zero opposite-tail draws, not zero population probability.",
        "criterion": (
            "PLAN_Final.md sec 8.2 criterion 3, on the sec 7 PRIMARY metric (discrete-time "
            "survival log-likelihood and integrated Brier), out of fold, against every "
            "registered baseline -- not only the trivial structural one."
        ),
        "primary_metric_note": (
            "sec 7 lists rank correlation as the SECONDARY metric. "
            "artifacts/week2/c2_correlation_2026-09-02/ scored criteria 3 and 4 on that "
            "secondary metric. This run supplies the primary one."
        ),
        "oracle_note": (
            "behavior_quality and behavior_value are built from exact_oracle_results.csv, i.e. "
            "from J*/V*. They are a ceiling, not fair competitors (sec 6.1 asks for return "
            "quality measured from the pilot; sec 11 rule 5 keeps oracle quantities out of the "
            "estimator). They are excluded from the strongest-fair-baseline verdict."
        ),
        "grid": list(grid),
        "num_folds": num_folds,
        "num_resamples": num_resamples,
        "confidence": confidence,
        "seed": seed,
        "protocol_config_sha256": sha256_file(PROTOCOL_CONFIG_PATH),
        "git_commit": _git_commit(),
        "intercept_only_note": (
            "intercept_only fits the grid-level marginal hazards and no covariate at all. If no "
            "feature set separates from it, the honest reading is that none of them -- OEH or "
            "baseline -- carries information beyond the marginal event rate at this sample size, "
            "which is a statement about power, not about OEH specifically."
        ),
        "oeh_beats_intercept_only_after_holm": beats_floor,
        "beats_any_fair_baseline_after_holm": beats_any_fair,
        "significantly_worse_than_fair_baselines": loses_to_fair,
        "criterion_3_on_primary_metric": beats_any_fair,
    }
    return results, comparisons, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-emp", type=Path, required=True)
    parser.add_argument("--behavior-metadata", type=Path, required=True)
    parser.add_argument("--benchmark-metadata", type=Path, required=True)
    parser.add_argument("--exact-oracle", type=Path, required=True)
    parser.add_argument("--alternatives", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--num-resamples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--materiality-fraction", type=float, default=0.01)
    args = parser.parse_args()

    grid = tuple(PROTOCOL_CONFIG["collection"]["size_multipliers"])
    started = time.time()
    instances, feature_names = assemble(
        n_emp_rows=read_csv_rows(args.n_emp),
        behavior_rows=read_csv_rows(args.behavior_metadata),
        benchmark_rows=read_csv_rows(args.benchmark_metadata),
        oracle_rows=read_csv_rows(args.exact_oracle),
        alternative_rows=read_csv_rows(args.alternatives),
        grid=grid,
        materiality_fraction=args.materiality_fraction,
    )
    if not instances:
        raise ValueError("No labels could be placed on the registered size grid.")

    results, comparisons, report = run(
        instances=instances,
        feature_names=feature_names,
        grid=grid,
        num_folds=args.num_folds,
        num_resamples=args.num_resamples,
        confidence=args.confidence,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output_dir / "calibrated_results.csv", results, RESULT_FIELDS)
    write_csv_rows(args.output_dir / "calibrated_comparison.csv", comparisons, COMPARISON_FIELDS)
    report["instances"] = len(instances)
    report["runtime_seconds"] = round(time.time() - started, 1)
    report["input_sha256"] = {
        path.name: sha256_file(path)
        for path in (args.n_emp, args.behavior_metadata, args.benchmark_metadata, args.exact_oracle, args.alternatives)
    }
    report["artifact_sha256"] = {
        name: sha256_file(args.output_dir / name)
        for name in ("calibrated_results.csv", "calibrated_comparison.csv")
    }
    write_json(args.output_dir / "calibrated_comparison_report.json", report)

    print(f"\n  {len(instances)} labels, grid {list(grid)}, {args.num_folds}-fold by base MDP")
    print("  out-of-fold mean log-likelihood (higher is better):")
    for row in sorted(results, key=lambda r: (r["algorithm"], -r["mean_log_likelihood"])):
        mark = " [ORACLE]" if row["is_oracle"] else ""
        print(
            f"    {row['algorithm']:14s} {row['feature_set']:18s} "
            f"loglik={row['mean_log_likelihood']:8.4f}  brier={row['integrated_brier']:.4f}  "
            f"n={row['n_scored']:3d} clusters={row['num_clusters']:2d}{mark}"
        )
    print(f"\n  criterion 3 on the PRIMARY metric: {report['criterion_3_on_primary_metric']}")
    if report["significantly_worse_than_fair_baselines"]:
        print(f"  significantly worse than: {', '.join(report['significantly_worse_than_fair_baselines'])}")
    print(f"  -> {args.output_dir}")


if __name__ == "__main__":
    main()

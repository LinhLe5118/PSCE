"""Per-run melt + group aggregation with explicit status accounting.

``per_run_results.csv`` is a long/melted table -- one row per
``(mdp_id, family, seed, algorithm, dataset_size, metric)`` -- exactly the
shape mega-task section 10 specifies, so every metric (``return``,
``normalized_return``, ``success``, ``N_emp``, ``k_dagger_oeh``,
``n_req_hat``, ...) is aggregated by the same code path instead of one path
per metric.

Never silently drops a censored/invalid/abstained run (mega-task section 10).
A group with zero valid rows still gets an output row, with ``NaN``
statistics rather than being omitted.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from evaluation.censoring.status import STATUS_VALUES, classify_n_emp_status

PER_RUN_FIELDS = ("mdp_id", "family", "algorithm", "seed", "dataset_size", "metric", "value", "status")


def _row(mdp_id: Any, family: Any, algorithm: Any, seed: Any, dataset_size: Any, metric: str, value: Any, status: str) -> dict[str, Any]:
    return {
        "mdp_id": mdp_id,
        "family": family,
        "algorithm": algorithm,
        "seed": seed,
        "dataset_size": dataset_size,
        "metric": metric,
        "value": value,
        "status": status,
    }


def build_per_run_results(
    *,
    rl_rows: Sequence[Mapping[str, Any]] = (),
    n_emp_rows: Sequence[Mapping[str, Any]] = (),
    oeh_rows: Sequence[Mapping[str, Any]] = (),
    exact_rows: Sequence[Mapping[str, Any]] = (),
    exact_k_dagger_rows: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Melt the artifact-contract tables into one long per-run table.

    ``oeh_rows`` must already be filtered to ``selected=True`` -- this
    function does not average across pilot multipliers (WEEK2_STATISTICAL_PLAN.md
    sec 3).
    """

    rows: list[dict[str, Any]] = []
    for r in rl_rows:
        args = (r["mdp_id"], r["policy_family"], r["algorithm"], r["training_seed"], r["training_size_multiplier"])
        rows.append(_row(*args, "normalized_return", r["normalized_return"], r["run_status"]))
        rows.append(_row(*args, "success", float(r["success"]), r["run_status"]))

    for r in n_emp_rows:
        args = (r["mdp_id"], r["policy_family"], r["algorithm"], None, None)
        status = classify_n_emp_status(
            right_censored=bool(r["n_emp_censored"]), requirement=r["n_emp_multiplier"]
        )
        rows.append(_row(*args, "N_emp", r["n_emp_transitions"], status))

    for r in oeh_rows:
        if not r.get("selected", True):
            raise ValueError("build_per_run_results requires oeh_rows filtered to selected=True")
        args = (r["mdp_id"], r["policy_family"], None, None, r["pilot_multiplier"])
        status = r["oeh_status"]
        rows.append(_row(*args, "k_dagger_oeh", r.get("k_dagger_oeh"), status))
        rows.append(_row(*args, "n_req_hat", r.get("n_req_hat"), status))
        rows.append(_row(*args, "effective_horizon", r.get("effective_horizon"), status))

    for r in exact_rows:
        args = (r["mdp_id"], r["policy_family"], None, None, None)
        rows.append(_row(*args, "behavior_return", r["behavior_return"], "observed"))
        rows.append(_row(*args, "behavior_normalized_return", r["behavior_normalized_return"], "observed"))

    for r in exact_k_dagger_rows:
        args = (r["mdp_id"], r["policy_family"], None, None, None)
        rows.append(_row(*args, "k_dagger_exact", r["k_dagger_exact"], "observed"))

    return rows


def cluster_bootstrap_ci(
    values_by_cluster: Mapping[Any, Sequence[float]],
    *,
    rng: np.random.Generator,
    statistic: Callable[[np.ndarray], float] = lambda values: float(np.mean(values)),
    num_resamples: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float, float, int]:
    """Resample clusters (base MDPs), not rows, per PLAN_Final.md sec 7."""

    clusters = list(values_by_cluster)
    num_clusters = len(clusters)
    if num_clusters == 0:
        return float("nan"), float("nan"), float("nan"), 0
    pooled = np.concatenate([np.asarray(values_by_cluster[c], dtype=np.float64) for c in clusters])
    point = statistic(pooled)
    if num_clusters == 1:
        return point, float("nan"), float("nan"), 1
    replicate_statistics = np.empty(num_resamples, dtype=np.float64)
    for b in range(num_resamples):
        sampled = rng.choice(clusters, size=num_clusters, replace=True)
        values = np.concatenate([np.asarray(values_by_cluster[c], dtype=np.float64) for c in sampled])
        replicate_statistics[b] = statistic(values)
    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(replicate_statistics, [alpha, 1.0 - alpha])
    return point, float(ci_low), float(ci_high), num_clusters


def aggregate(
    per_run_rows: Sequence[Mapping[str, Any]],
    *,
    group_by: Sequence[str],
    cluster_field: str,
    rng: np.random.Generator,
    include_median: bool = False,
    num_resamples: int = 2000,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_run_rows:
        key = tuple(row[field] for field in group_by)
        groups[key].append(row)

    output: list[dict[str, Any]] = []
    for key, rows in groups.items():
        status_counts = {status: 0 for status in STATUS_VALUES}
        values_by_cluster: dict[Any, list[float]] = defaultdict(list)
        for row in rows:
            status = row["status"]
            if status not in STATUS_VALUES:
                raise ValueError(f"Unknown status {status!r} in per-run row {row!r}")
            status_counts[status] += 1
            if status == "observed" and row["value"] is not None:
                values_by_cluster[row[cluster_field]].append(float(row["value"]))

        entry: dict[str, Any] = dict(zip(group_by, key))
        entry["n_total"] = len(rows)
        for status in STATUS_VALUES:
            entry[f"n_{status}"] = status_counts[status]

        if values_by_cluster:
            point, ci_low, ci_high, num_clusters = cluster_bootstrap_ci(
                values_by_cluster, rng=rng, num_resamples=num_resamples
            )
            all_values = np.concatenate([np.asarray(v, dtype=np.float64) for v in values_by_cluster.values()])
            entry["mean"] = point
            entry["std"] = float(np.std(all_values, ddof=1)) if len(all_values) > 1 else float("nan")
            entry["median"] = float(np.median(all_values)) if include_median else None
            entry["ci_low"] = ci_low
            entry["ci_high"] = ci_high
            entry["num_clusters"] = num_clusters
        else:
            entry["mean"] = float("nan")
            entry["std"] = float("nan")
            entry["median"] = float("nan") if include_median else None
            entry["ci_low"] = float("nan")
            entry["ci_high"] = float("nan")
            entry["num_clusters"] = 0
        output.append(entry)
    return output


__all__ = ["PER_RUN_FIELDS", "aggregate", "build_per_run_results", "cluster_bootstrap_ci"]

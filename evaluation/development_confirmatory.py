"""Development-only adapters used by the Week 6 confirmatory preflight.

This module deliberately keeps the development probes separate from the
sealed confirmatory command.  The train/score adapter calls the registered
tabular learners and exact policy scorer on a development buffer.  The
bootstrap adapter consumes the tracked development feature/outcome panel and
refits the two frozen return models inside every paired cluster draw.

Neither adapter can read an evaluation record: the training adapter rejects
non-development manifests and the refit panel verifies its own development
reports before it is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from evaluation.aggregation.folds import assign_folds
from evaluation.io import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL_FILES = {
    "base_features": "artifacts/research/oeh_v3_post_freeze_b0_b3_2026-09-20/features.csv",
    "learner_features": "artifacts/research/lcoeh_v4_exploratory_2026-09-20/features.csv",
    "returns": "artifacts/research/lcoeh_v4_exploratory_final_2026-09-20/return_oof_predictions.csv",
    "base_report": "artifacts/research/oeh_v3_post_freeze_b0_b3_2026-09-20/report.json",
    "learner_report": "artifacts/research/lcoeh_v4_exploratory_2026-09-20/report.json",
    "return_report": "artifacts/research/lcoeh_v4_exploratory_final_2026-09-20/report.json",
}

METHOD_TO_ALGORITHM = {
    "cfqi_tabular": "cfqi_tabular",
    "iql_tabular": "iql_tabular",
    "behavior_cloning_tabular": "behavior_cloning_tabular",
}
FEATURE_SETS = ("B2_OEH3", "B2_OEH3_LCOEH4")
ALGORITHMS = tuple(METHOD_TO_ALGORITHM.values())
RETURN_RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class DevelopmentPanelRow:
    mdp_id: str
    policy_family: str
    algorithm: str
    normalized_return: float
    features: Mapping[str, tuple[float, ...]]


@dataclass(frozen=True)
class DevelopmentRefitPanel:
    rows: tuple[DevelopmentPanelRow, ...]
    cluster_strata: Mapping[str, str]
    observed: Mapping[str, float]
    fit_and_score: Callable[[Mapping[str, int], int], Mapping[str, float]]
    source_hashes: Mapping[str, str]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_vector(row: Mapping[str, str], fields: Sequence[str], *, label: str) -> tuple[float, ...]:
    values = tuple(float(row[field]) for field in fields)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite or empty feature vector for {label}")
    return values


def _panel_paths(root: Path, panel_dir: Path | None) -> dict[str, Path]:
    base = (panel_dir or root).resolve()
    if panel_dir is None:
        return {key: (root / value).resolve() for key, value in DEFAULT_PANEL_FILES.items()}
    return {
        "base_features": base / "oeh_v3_post_freeze_b0_b3_2026-09-20" / "features.csv",
        "learner_features": base / "lcoeh_v4_exploratory_2026-09-20" / "features.csv",
        "returns": base / "lcoeh_v4_exploratory_final_2026-09-20" / "return_oof_predictions.csv",
        "base_report": base / "oeh_v3_post_freeze_b0_b3_2026-09-20" / "report.json",
        "learner_report": base / "lcoeh_v4_exploratory_2026-09-20" / "report.json",
        "return_report": base / "lcoeh_v4_exploratory_final_2026-09-20" / "report.json",
    }


def _assert_development_report(path: Path) -> None:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("sealed_evaluation_accessed") is True:
        raise ValueError(f"Refit panel report claims sealed evaluation access: {path}")
    exploratory = "exploratory" in str(payload.get("method", "")).lower()
    if (
        payload.get("development_only") is not True
        and payload.get("confirmatory_claim_allowed") is not False
        and not exploratory
    ):
        raise ValueError(f"Refit panel is not marked development-only/exploratory: {path}")


def load_development_refit_panel(
    *, root: Path = ROOT, panel_dir: Path | None = None, max_clusters: int | None = 15
) -> DevelopmentRefitPanel:
    """Load the tracked development panel and return a real refit callback.

    The default 15-cluster slice mirrors the registered development cohort and
    keeps the 2,000-draw capacity run a dry-run rather than a full analysis.
    Passing ``None`` uses every tracked development cluster.
    """

    paths = _panel_paths(root.resolve(), panel_dir)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing development refit input: {path}")
    for key in ("base_report", "learner_report", "return_report"):
        _assert_development_report(paths[key])

    base_rows = _read_csv(paths["base_features"])
    learner_rows = _read_csv(paths["learner_features"])
    return_rows = _read_csv(paths["returns"])
    if not base_rows or not learner_rows or not return_rows:
        raise ValueError("Development refit panel inputs must be non-empty")

    identity = {"dataset_id", "mdp_id", "policy_family", "pilot_multiplier",
                "num_transitions", "num_states", "num_actions"}
    base_fields = tuple(field for field in base_rows[0] if field not in identity)
    learner_fields = tuple(field for field in learner_rows[0] if field.startswith("lcoeh4_"))
    if not base_fields or not learner_fields:
        raise ValueError("Development panel is missing B2/OEH3 or LC-OEH4 features")

    base_by_group: dict[tuple[str, str], dict[str, str]] = {}
    for row in base_rows:
        key = (str(row["mdp_id"]), str(row["policy_family"]))
        if key in base_by_group:
            raise ValueError(f"Duplicate base feature group: {key}")
        base_by_group[key] = row
    all_mdp_ids = sorted({key[0] for key in base_by_group})
    if max_clusters is not None:
        if max_clusters < 5:
            raise ValueError("max_clusters must be at least five")
        selected_mdp_ids = set(all_mdp_ids[:max_clusters])
    else:
        selected_mdp_ids = set(all_mdp_ids)

    learner_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in learner_rows:
        key = (str(row["mdp_id"]), str(row["policy_family"]), str(row["algorithm"]))
        if key in learner_by_key:
            raise ValueError(f"Duplicate learner feature row: {key}")
        learner_by_key[key] = row

    outcomes: dict[tuple[str, str, str], float] = {}
    for row in return_rows:
        if str(row.get("feature_set", "")) != "B2":
            continue
        key = (str(row["mdp_id"]), str(row["policy_family"]), str(row["algorithm"]))
        value = float(row["normalized_return"])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite development return for {key}")
        previous = outcomes.get(key)
        if previous is not None and not math.isclose(previous, value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Inconsistent development return for {key}")
        outcomes[key] = value

    rows: list[DevelopmentPanelRow] = []
    for key in sorted(learner_by_key):
        mdp_id, policy_family, algorithm = key
        if mdp_id not in selected_mdp_ids:
            continue
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unregistered algorithm in development panel: {algorithm}")
        if (mdp_id, policy_family) not in base_by_group or key not in outcomes:
            raise ValueError(f"Incomplete development refit panel at {key}")
        base = base_by_group[(mdp_id, policy_family)]
        learner = learner_by_key[key]
        rows.append(
            DevelopmentPanelRow(
                mdp_id=mdp_id,
                policy_family=policy_family,
                algorithm=algorithm,
                normalized_return=outcomes[key],
                features={
                    "B2_OEH3": _finite_vector(base, base_fields, label=f"{key}/B2_OEH3"),
                    "B2_OEH3_LCOEH4": _finite_vector(
                        {**base, **learner},
                        base_fields + learner_fields,
                        label=f"{key}/B2_OEH3_LCOEH4",
                    ),
                },
            )
        )
    if not rows:
        raise ValueError("No complete development refit rows")

    mdp_ids = sorted({row.mdp_id for row in rows})
    if len(mdp_ids) < 5:
        raise ValueError("Development refit panel needs at least five base MDPs")
    groups = {(row.mdp_id, row.policy_family) for row in rows}
    expected = {(mdp, family, algorithm) for mdp, family in groups for algorithm in ALGORITHMS}
    actual = {(row.mdp_id, row.policy_family, row.algorithm) for row in rows}
    if actual != expected:
        raise ValueError("Development refit panel does not contain a complete learner panel")

    cluster_strata = {
        mdp_id: str(fold)
        for mdp_id, fold in assign_folds(mdp_ids, num_folds=5).items()
    }
    row_by_key = {
        (row.mdp_id, row.policy_family, row.algorithm): row for row in rows
    }
    group_keys = sorted(groups)
    outer = {mdp_id: int(fold) for mdp_id, fold in assign_folds(mdp_ids, num_folds=5).items()}
    for fold in range(5):
        if not any(value == fold for value in outer.values()):
            raise ValueError(f"Development panel has no outer-fold {fold}")

    def fit_and_score(multiplicities: Mapping[str, int], _draw_index: int) -> Mapping[str, float]:
        weights = {str(key): int(value) for key, value in multiplicities.items()}
        if any(value < 0 for value in weights.values()):
            raise ValueError("Bootstrap multiplicities cannot be negative")
        if sum(weights.values()) <= 0:
            raise ValueError("Bootstrap sample is empty")
        return _refit_selector_endpoints(rows, group_keys, outer, weights)

    observed = fit_and_score({mdp_id: 1 for mdp_id in mdp_ids}, -1)
    source_hashes: dict[str, str] = {}
    root_resolved = root.resolve()
    for key, path in paths.items():
        if not (key.endswith("features") or key in {"returns", "base_report", "learner_report", "return_report"}):
            continue
        try:
            label = str(path.relative_to(root_resolved)).replace("\\", "/")
        except ValueError:
            label = f"external_panel:{key}"
        source_hashes[label] = sha256_file(path)
    return DevelopmentRefitPanel(
        rows=tuple(rows),
        cluster_strata=cluster_strata,
        observed=observed,
        fit_and_score=fit_and_score,
        source_hashes=source_hashes,
    )


def _weighted_design(
    rows: Sequence[DevelopmentPanelRow],
    weights: Mapping[str, int],
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    usable = [row for row in rows if int(weights.get(row.mdp_id, 0)) > 0]
    if not usable:
        raise ValueError("Weighted ridge received no positive-weight rows")
    matrix = np.asarray([row.features[feature_set] for row in usable], dtype=float)
    target = np.asarray([row.normalized_return for row in usable], dtype=float)
    sample_weight = np.asarray([float(weights[row.mdp_id]) for row in usable], dtype=float)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise ValueError("Non-finite weighted ridge input")
    total = float(sample_weight.sum())
    mean = np.sum(matrix * sample_weight[:, None], axis=0) / total
    scale = np.sqrt(np.sum(np.square(matrix - mean) * sample_weight[:, None], axis=0) / total)
    scale[scale <= 1e-12] = 1.0
    standardized = (matrix - mean) / scale
    design = np.column_stack((np.ones(len(usable)), standardized))
    return design, target, sample_weight, np.column_stack((mean, scale))


def _fit_weighted_ridge(
    rows: Sequence[DevelopmentPanelRow],
    weights: Mapping[str, int],
    feature_set: str,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design, target, sample_weight, standardizer = _weighted_design(rows, weights, feature_set)
    penalty = np.diag([0.0] + [float(ridge)] * (design.shape[1] - 1))
    weighted_design = design * sample_weight[:, None]
    coefficients = np.linalg.solve(
        design.T @ weighted_design + penalty,
        design.T @ (sample_weight * target),
    )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Weighted ridge produced non-finite coefficients")
    return coefficients, standardizer[:, 0], standardizer[:, 1]


def _predict_weighted_ridge(
    model: tuple[np.ndarray, np.ndarray, np.ndarray],
    rows: Sequence[DevelopmentPanelRow],
    feature_set: str,
) -> np.ndarray:
    coefficients, mean, scale = model
    matrix = np.asarray([row.features[feature_set] for row in rows], dtype=float)
    design = np.column_stack((np.ones(len(rows)), (matrix - mean) / scale))
    result = design @ coefficients
    if not np.all(np.isfinite(result)):
        raise ValueError("Weighted ridge prediction was non-finite")
    return result


def _select_ridge(
    rows: Sequence[DevelopmentPanelRow],
    weights: Mapping[str, int],
    train_mdp_ids: Sequence[str],
    feature_set: str,
    *,
    folds: int = 4,
) -> float:
    positive = sorted({mdp_id for mdp_id in train_mdp_ids if int(weights.get(mdp_id, 0)) > 0})
    if len(positive) < folds:
        raise ValueError(f"Need {folds} positive-weight inner groups, found {len(positive)}")
    inner = assign_folds(positive, num_folds=folds)
    scores: dict[float, float] = {}
    train_rows = [
        row
        for row in rows
        if row.mdp_id in set(train_mdp_ids) and int(weights.get(row.mdp_id, 0)) > 0
    ]
    for ridge in RETURN_RIDGE_GRID:
        squared: list[float] = []
        total_weight = 0.0
        for fold in range(folds):
            inner_train = [row for row in train_rows if inner[row.mdp_id] != fold]
            inner_test = [row for row in train_rows if inner[row.mdp_id] == fold and int(weights.get(row.mdp_id, 0)) > 0]
            if not inner_test:
                continue
            model = _fit_weighted_ridge(inner_train, weights, feature_set, ridge)
            predicted = _predict_weighted_ridge(model, inner_test, feature_set)
            observed = np.asarray([row.normalized_return for row in inner_test], dtype=float)
            row_weights = np.asarray([float(weights[row.mdp_id]) for row in inner_test], dtype=float)
            squared.extend((row_weights * np.square(predicted - observed)).tolist())
            total_weight += float(row_weights.sum())
        if total_weight <= 0.0 or not squared:
            scores[ridge] = float("inf")
        else:
            scores[ridge] = float(sum(squared) / total_weight)
    selected = min(scores, key=lambda value: (scores[value], -value))
    if not math.isfinite(scores[selected]):
        raise ValueError("All inner ridge candidates failed")
    return float(selected)


def _batch_weighted_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    ridges: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit all registered ridge candidates from one weighted design matrix."""

    total = float(sample_weight.sum())
    if total <= 0.0:
        raise ValueError("Weighted ridge received no positive mass")
    mean = np.sum(matrix * sample_weight[:, None], axis=0) / total
    scale = np.sqrt(
        np.sum(np.square(matrix - mean) * sample_weight[:, None], axis=0) / total
    )
    scale[scale <= 1e-12] = 1.0
    standardized = (matrix - mean) / scale
    design = np.column_stack((np.ones(len(matrix)), standardized))
    gram = design.T @ (design * sample_weight[:, None])
    rhs = design.T @ (sample_weight * target)
    penalties = np.zeros((len(ridges), design.shape[1], design.shape[1]), dtype=float)
    for index, ridge in enumerate(ridges):
        penalties[index, 1:, 1:] = np.eye(design.shape[1] - 1) * float(ridge)
    coefficients = np.linalg.solve(
        gram[None, :, :] + penalties,
        np.broadcast_to(rhs, (len(ridges), len(rhs)))[:, :, None],
    )[:, :, 0]
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Batched weighted ridge produced non-finite coefficients")
    return coefficients, mean, scale


def _fast_model_for_panel(
    panel: Sequence[DevelopmentPanelRow],
    *,
    feature_set: str,
    outer: Mapping[str, int],
    outer_fold: int,
    weights: Mapping[str, int],
) -> tuple[dict[tuple[str, str], float], float]:
    """Tune and fit one learner/feature model using vectorized ridge solves."""

    train_indices = [
        index
        for index, row in enumerate(panel)
        if outer[row.mdp_id] != outer_fold and int(weights.get(row.mdp_id, 0)) > 0
    ]
    test_indices = [
        index
        for index, row in enumerate(panel)
        if outer[row.mdp_id] == outer_fold and int(weights.get(row.mdp_id, 0)) > 0
    ]
    if not train_indices or not test_indices:
        raise ValueError("Outer fold has no positive-weight train/test rows")
    train_mdp_ids = sorted({panel[index].mdp_id for index in train_indices})
    if len(train_mdp_ids) < 4:
        raise ValueError("Need four positive-weight outer-training MDPs")
    inner = assign_folds(train_mdp_ids, num_folds=4)
    matrix = np.asarray([row.features[feature_set] for row in panel], dtype=float)
    target = np.asarray([row.normalized_return for row in panel], dtype=float)
    weights_array = np.asarray(
        [float(weights.get(row.mdp_id, 0)) for row in panel], dtype=float
    )
    scores = np.zeros(len(RETURN_RIDGE_GRID), dtype=float)
    score_mass = np.zeros(len(RETURN_RIDGE_GRID), dtype=float)
    for inner_fold in range(4):
        inner_train = [
            index for index in train_indices if inner[panel[index].mdp_id] != inner_fold
        ]
        inner_test = [
            index for index in train_indices if inner[panel[index].mdp_id] == inner_fold
        ]
        if not inner_test:
            continue
        coefficients, mean, scale = _batch_weighted_ridge(
            matrix[inner_train],
            target[inner_train],
            weights_array[inner_train],
            RETURN_RIDGE_GRID,
        )
        test_design = np.column_stack(
            (np.ones(len(inner_test)), (matrix[inner_test] - mean) / scale)
        )
        predicted = test_design @ coefficients.T
        row_weights = weights_array[inner_test]
        errors = np.square(predicted - target[inner_test, None])
        scores += np.sum(row_weights[:, None] * errors, axis=0)
        score_mass += float(row_weights.sum())
    valid = score_mass > 0.0
    if not np.any(valid):
        raise ValueError("All inner ridge candidates failed")
    scores[valid] /= score_mass[valid]
    scores[~valid] = np.inf
    ridge_index = min(
        range(len(RETURN_RIDGE_GRID)),
        key=lambda index: (scores[index], -float(RETURN_RIDGE_GRID[index])),
    )
    selected = float(RETURN_RIDGE_GRID[ridge_index])
    coefficients, mean, scale = _batch_weighted_ridge(
        matrix[train_indices],
        target[train_indices],
        weights_array[train_indices],
        (selected,),
    )
    test_design = np.column_stack(
        (np.ones(len(test_indices)), (matrix[test_indices] - mean) / scale)
    )
    predictions = test_design @ coefficients[0]
    by_group = {
        (panel[index].mdp_id, panel[index].policy_family): float(value)
        for index, value in zip(test_indices, predictions)
    }
    return by_group, selected


def _refit_selector_endpoints(
    rows: Sequence[DevelopmentPanelRow],
    group_keys: Sequence[tuple[str, str]],
    outer: Mapping[str, int],
    weights: Mapping[str, int],
) -> dict[str, float]:
    """Refit both frozen return components and score paired endpoints."""

    by_algorithm = {
        algorithm: [row for row in rows if row.algorithm == algorithm]
        for algorithm in ALGORITHMS
    }
    row_by_key = {
        (row.mdp_id, row.policy_family, row.algorithm): row for row in rows
    }
    selected_values: list[tuple[str, float, float]] = []
    for fold in range(5):
        test_groups = [
            group
            for group in group_keys
            if outer[group[0]] == fold and int(weights.get(group[0], 0)) > 0
        ]
        if not test_groups:
            continue
        predictions: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
        for algorithm in ALGORITHMS:
            for feature_set in FEATURE_SETS:
                prediction, _ = _fast_model_for_panel(
                    by_algorithm[algorithm],
                    feature_set=feature_set,
                    outer=outer,
                    outer_fold=fold,
                    weights=weights,
                )
                predictions[(algorithm, feature_set)] = prediction

        fixed_scores = {algorithm: 0.0 for algorithm in ALGORITHMS}
        fixed_mass = {algorithm: 0.0 for algorithm in ALGORITHMS}
        for row in rows:
            if outer[row.mdp_id] == fold:
                continue
            weight = float(weights.get(row.mdp_id, 0))
            fixed_scores[row.algorithm] += weight * row.normalized_return
            fixed_mass[row.algorithm] += weight
        if any(mass <= 0.0 for mass in fixed_mass.values()):
            raise ValueError("Outer training sample is empty for a learner")
        fixed_algorithm = max(
            ALGORITHMS,
            key=lambda algorithm: (
                fixed_scores[algorithm] / fixed_mass[algorithm],
                -ALGORITHMS.index(algorithm),
            ),
        )
        for mdp_id, policy_family in test_groups:
            predicted_by_algorithm = {
                algorithm: float(
                    np.mean(
                        [
                            predictions[(algorithm, feature_set)][(mdp_id, policy_family)]
                            for feature_set in FEATURE_SETS
                        ]
                    )
                )
                for algorithm in ALGORITHMS
            }
            selected_algorithm = max(
                ALGORITHMS,
                key=lambda algorithm: (
                    predicted_by_algorithm[algorithm],
                    -ALGORITHMS.index(algorithm),
                ),
            )
            selected_values.append(
                (
                    mdp_id,
                    row_by_key[(mdp_id, policy_family, selected_algorithm)].normalized_return,
                    row_by_key[(mdp_id, policy_family, fixed_algorithm)].normalized_return,
                )
            )

    if not selected_values:
        raise ValueError("No positive-weight outer test groups")
    by_mdp: dict[str, list[tuple[float, float]]] = {}
    for mdp_id, selected, fixed in selected_values:
        by_mdp.setdefault(mdp_id, []).append((float(selected), float(fixed)))
    weighted_selected = 0.0
    weighted_fixed = 0.0
    denominator = 0.0
    for mdp_id, values in by_mdp.items():
        weight = float(weights.get(mdp_id, 0))
        if weight <= 0.0:
            continue
        weighted_selected += weight * float(np.mean([value[0] for value in values]))
        weighted_fixed += weight * float(np.mean([value[1] for value in values]))
        denominator += weight
    if denominator <= 0.0:
        raise ValueError("No positive-weight MDP clusters after outer refit")
    return {
        "v4e": weighted_selected / denominator,
        "fold_best_fixed": weighted_fixed / denominator,
    }


def _legacy_refit_selector_endpoints(
    rows: Sequence[DevelopmentPanelRow],
    group_keys: Sequence[tuple[str, str]],
    outer: Mapping[str, int],
    weights: Mapping[str, int],
) -> dict[str, float]:
    selected_values: list[tuple[str, float, float]] = []
    for fold in range(5):
        test_groups = [
            group for group in group_keys
            if outer[group[0]] == fold and int(weights.get(group[0], 0)) > 0
        ]
        train_mdp_ids = [mdp_id for mdp_id, value in outer.items() if value != fold]
        if not test_groups:
            continue
        models: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for algorithm in ALGORITHMS:
            train_rows = [
                row for row in rows
                if row.algorithm == algorithm and row.mdp_id in set(train_mdp_ids)
            ]
            for feature_set in FEATURE_SETS:
                ridge = _select_ridge(
                    train_rows, weights, train_mdp_ids, feature_set, folds=4
                )
                models[(algorithm, feature_set)] = _fit_weighted_ridge(
                    train_rows, weights, feature_set, ridge
                )

        fixed_scores = {
            algorithm: 0.0 for algorithm in ALGORITHMS
        }
        fixed_denominator = 0.0
        for row in rows:
            if row.mdp_id in set(train_mdp_ids):
                weight = float(weights.get(row.mdp_id, 0))
                fixed_scores[row.algorithm] += weight * row.normalized_return
                fixed_denominator += weight
        if fixed_denominator <= 0.0:
            raise ValueError("Outer training sample is empty")
        fixed_algorithm = max(
            ALGORITHMS,
            key=lambda algorithm: (fixed_scores[algorithm] / fixed_denominator, -ALGORITHMS.index(algorithm)),
        )
        for mdp_id, policy_family in test_groups:
            group_rows = {
                algorithm: next(
                    row for row in rows
                    if row.mdp_id == mdp_id and row.policy_family == policy_family and row.algorithm == algorithm
                )
                for algorithm in ALGORITHMS
            }
            predicted = {}
            for algorithm, row in group_rows.items():
                component_scores = [
                    float(_predict_weighted_ridge(models[(algorithm, feature_set)], [row], feature_set)[0])
                    for feature_set in FEATURE_SETS
                ]
                predicted[algorithm] = float(np.mean(component_scores))
            selected_algorithm = max(
                ALGORITHMS,
                key=lambda algorithm: (predicted[algorithm], -ALGORITHMS.index(algorithm)),
            )
            selected_values.append(
                (
                    mdp_id,
                    group_rows[selected_algorithm].normalized_return,
                    group_rows[fixed_algorithm].normalized_return,
                )
            )

    if not selected_values:
        raise ValueError("No positive-weight outer test groups")
    by_mdp: dict[str, list[tuple[float, float]]] = {}
    for mdp_id, selected, fixed in selected_values:
        by_mdp.setdefault(mdp_id, []).append((float(selected), float(fixed)))
    weighted_selected = 0.0
    weighted_fixed = 0.0
    denominator = 0.0
    for mdp_id, values in by_mdp.items():
        weight = float(weights.get(mdp_id, 0))
        if weight <= 0.0:
            continue
        weighted_selected += weight * float(np.mean([value[0] for value in values]))
        weighted_fixed += weight * float(np.mean([value[1] for value in values]))
        denominator += weight
    if denominator <= 0.0:
        raise ValueError("No positive-weight MDP clusters after outer refit")
    return {
        "v4e": weighted_selected / denominator,
        "fold_best_fixed": weighted_fixed / denominator,
    }


def build_development_train_score_callback(
    dataset_dir: Path,
    records: Sequence[Mapping[str, Any]],
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Build the real CFQI/IQL/BC train-and-score callback for development."""

    import json
    import sys

    if str(ROOT / "code") not in sys.path:
        sys.path.insert(0, str(ROOT / "code"))
    from config.protocol_config import PROTOCOL_CONFIG
    from environments.benchmark_environments import make_benchmark_mdp
    from scripts.training.run_empirical_requirement import (
        _load_run,
        _score_policy,
        _training_seed_index,
    )
    from empirical_requirement import (
        behavior_cloning_tabular,
        cfqi_tabular,
        iql_tabular,
    )

    manifest_path = dataset_dir.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_mode") != "review":
        raise ValueError("Real confirmatory callback requires a review development bundle")
    instances = {str(item["instance_id"]): item for item in manifest.get("instances", [])}
    if any(item.get("benchmark_split") != "development" for item in records):
        raise ValueError("Real confirmatory callback refuses non-development records")
    indexed: dict[tuple[str, str, int, int], Mapping[str, Any]] = {}
    for record in records:
        if record.get("stream") != "training":
            continue
        key = (
            str(record["instance_id"]),
            str(record["policy_family"]),
            int(record["size_multiplier"]),
            int(_training_seed_index(dict(record))),
        )
        if key in indexed:
            raise ValueError(f"Duplicate development training record: {key}")
        indexed[key] = record
    mdps: dict[str, Any] = {}
    transitions_cache: dict[str, dict[str, np.ndarray]] = {}
    learning = PROTOCOL_CONFIG["learning"]

    def runner(job: Mapping[str, Any]) -> Mapping[str, Any]:
        method = str(job["method"])
        if method not in METHOD_TO_ALGORITHM:
            raise ValueError(f"Unsupported confirmatory method: {method}")
        key = (
            str(job["instance_id"]),
            str(job["policy_family"]),
            int(job["size_multiplier"]),
            int(job["training_seed"]),
        )
        record = indexed.get(key)
        if record is None:
            raise KeyError(f"Missing development training record for {key}")
        if str(record["sha256"]).lower() != str(job["buffer_sha256"]).lower():
            raise ValueError(f"Development buffer hash mismatch for {key}")
        instance = instances[key[0]]
        if key[0] not in mdps:
            mdps[key[0]] = make_benchmark_mdp(type("Spec", (), instance)())
        mdp = mdps[key[0]]
        transitions = _load_run(dataset_dir.resolve(), dict(record), transitions_cache)
        if method == "cfqi_tabular":
            policy = cfqi_tabular(
                transitions,
                num_states=mdp.num_states,
                num_actions=mdp.num_actions,
                gamma=mdp.gamma,
                iterations=int(learning["iterations"]),
                tolerance=float(learning["tolerance"]),
            )
        elif method == "iql_tabular":
            policy = iql_tabular(
                transitions,
                num_states=mdp.num_states,
                num_actions=mdp.num_actions,
                gamma=mdp.gamma,
                expectile=float(learning["iql_expectile"]),
                inverse_temperature=float(learning["iql_inverse_temperature"]),
                iterations=int(learning["iterations"]),
                tolerance=float(learning["tolerance"]),
            )
        else:
            policy = behavior_cloning_tabular(
                transitions,
                num_states=mdp.num_states,
                num_actions=mdp.num_actions,
            )
        normalized_return, run_status = _score_policy(
            mdp,
            np.asarray(policy, dtype=float),
            random_return=float(instance["random_return"]),
            dynamic_range=float(instance["dynamic_range"]),
        )
        if run_status != "observed":
            raise ValueError(f"Development train/score returned invalid policy for {key}")
        return {
            "status": "observed",
            "run_status": run_status,
            "method": method,
            "normalized_return": float(normalized_return),
            "success": bool(normalized_return >= 0.95),
            "buffer_sha256": str(job["buffer_sha256"]),
            "evaluation_opened": False,
            "callback": "registered_development_train_score_v1",
        }

    return runner


__all__ = [
    "ALGORITHMS",
    "DEFAULT_PANEL_FILES",
    "DevelopmentPanelRow",
    "DevelopmentRefitPanel",
    "FEATURE_SETS",
    "build_development_train_score_callback",
    "load_development_refit_panel",
]

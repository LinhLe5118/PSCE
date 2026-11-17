"""Artifact schema validation against ``docs/week2/WEEK2_ARTIFACT_CONTRACT.md``.

Fails loudly. Never fills a missing column with a default, never coerces a bad
dtype silently, never drops a duplicate row, never converts an invalid status
into a nearby valid one (mega-task section 17).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evaluation.censoring.status import STATUS_VALUES
from evaluation.io import read_csv_rows

Row = Mapping[str, Any]
RowRule = Callable[[dict[str, Any]], str | None]
TableRule = Callable[[list[dict[str, Any]]], str | None]


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str  # "str" | "int" | "float" | "bool" | "json"
    required: bool = True
    enum: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ArtifactSchema:
    name: str
    key_columns: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    row_rules: tuple[RowRule, ...] = ()
    table_rules: tuple[TableRule, ...] = ()


def _coerce(raw: str, column: ColumnSpec, *, context: str) -> Any:
    value = raw.strip()
    if value == "":
        if column.required:
            raise SchemaError(f"{context}: required column '{column.name}' is empty")
        return None
    if column.dtype == "str":
        result: Any = value
    elif column.dtype == "int":
        try:
            result = int(value)
        except ValueError as error:
            raise SchemaError(
                f"{context}: column '{column.name}' expected int, got {raw!r}"
            ) from error
    elif column.dtype == "float":
        try:
            result = float(value)
        except ValueError as error:
            raise SchemaError(
                f"{context}: column '{column.name}' expected float, got {raw!r}"
            ) from error
    elif column.dtype == "bool":
        lowered = value.lower()
        if lowered in {"true", "1"}:
            result = True
        elif lowered in {"false", "0"}:
            result = False
        else:
            raise SchemaError(
                f"{context}: column '{column.name}' expected bool, got {raw!r}"
            )
    elif column.dtype == "json":
        try:
            result = json.loads(value)
        except json.JSONDecodeError as error:
            raise SchemaError(
                f"{context}: column '{column.name}' is not valid JSON: {raw!r}"
            ) from error
    else:
        raise AssertionError(f"Unhandled dtype {column.dtype!r}")
    if column.enum is not None and result not in column.enum:
        raise SchemaError(
            f"{context}: column '{column.name}' has value {result!r}, "
            f"expected one of {column.enum}"
        )
    return result


def _status_column(name: str, *, required: bool = True) -> ColumnSpec:
    return ColumnSpec(name=name, dtype="str", required=required, enum=STATUS_VALUES)


def _empirical_sample_complexity_rules() -> tuple[RowRule, ...]:
    def censoring_consistency(row: dict[str, Any]) -> str | None:
        censored = row["n_emp_censored"]
        multiplier = row["n_emp_multiplier"]
        transitions = row["n_emp_transitions"]
        if censored and (multiplier is not None or transitions is not None):
            return "n_emp_censored=True but n_emp_multiplier/n_emp_transitions are not empty"
        if not censored and (multiplier is None or transitions is None):
            return "n_emp_censored=False but n_emp_multiplier/n_emp_transitions are empty"
        return None

    return (censoring_consistency,)


def _oeh_results_selected_rule(rows: list[dict[str, Any]]) -> str | None:
    counts: dict[tuple[Any, ...], int] = {}
    for row in rows:
        key = (row["mdp_id"], row["policy_family"])
        counts[key] = counts.get(key, 0) + (1 if row["selected"] else 0)
    bad = {key: count for key, count in counts.items() if count != 1}
    if bad:
        return f"expected exactly one selected=True row per (mdp_id, policy_family), found {bad}"
    return None


def _oeh_results_rules() -> tuple[RowRule, ...]:
    def status_consistency(row: dict[str, Any]) -> str | None:
        status = row["oeh_status"]
        if status == "abstained":
            if not row["abstain_reason"]:
                return "oeh_status=abstained requires a non-empty abstain_reason"
            if row["k_dagger_oeh"] is not None or row["effective_horizon"] is not None or row["n_req_hat"] is not None:
                return "oeh_status=abstained must leave k_dagger_oeh/effective_horizon/n_req_hat empty"
        elif status == "censored":
            if row["gap_occupancy_at_k_dagger"] is None:
                return "oeh_status=censored requires gap_occupancy_at_k_dagger"
            if row["k_dagger_oeh"] is None:
                return "oeh_status=censored must still carry k_dagger_oeh"
        elif status == "observed":
            if row["k_dagger_oeh"] is None or row["effective_horizon"] is None or row["n_req_hat"] is None:
                return "oeh_status=observed requires k_dagger_oeh/effective_horizon/n_req_hat"
        return None

    return (status_consistency,)


ARTIFACT_SCHEMAS: dict[str, ArtifactSchema] = {
    "benchmark_metadata": ArtifactSchema(
        name="benchmark_metadata",
        key_columns=("mdp_id",),
        columns=(
            ColumnSpec("mdp_id", "str"),
            ColumnSpec(
                "family",
                "str",
                enum=("uniform_garnet", "forward_layered", "lock_structured", "bottleneck",
                      "combination_lock", "grid_maze", "riverswim"),
            ),
            ColumnSpec("benchmark_split", "str", enum=("evaluation", "development", "ood")),
            ColumnSpec("num_states", "int"),
            ColumnSpec("num_actions", "int"),
            ColumnSpec("gamma", "float"),
            ColumnSpec("branching_factor", "float"),
            ColumnSpec("mixing_time_proxy", "float", required=False),
            ColumnSpec("dynamic_range", "float"),
            ColumnSpec("excluded_dynamic_range", "bool"),
            ColumnSpec("git_commit", "str"),
            ColumnSpec("config_hash", "str"),
        ),
    ),
    "exact_oracle_results": ArtifactSchema(
        name="exact_oracle_results",
        key_columns=("mdp_id", "policy_family"),
        columns=(
            ColumnSpec("mdp_id", "str"),
            ColumnSpec("policy_family", "str"),
            ColumnSpec("behavior_return", "float"),
            ColumnSpec("behavior_normalized_return", "float"),
            ColumnSpec("behavior_optimality_gap", "float"),
            ColumnSpec("optimal_return", "float"),
            ColumnSpec("random_return", "float"),
            ColumnSpec("uses_q_star", "bool"),
        ),
    ),
    "rl_run_results": ArtifactSchema(
        name="rl_run_results",
        key_columns=("mdp_id", "policy_family", "algorithm", "training_size_multiplier", "training_seed"),
        columns=(
            ColumnSpec("mdp_id", "str"),
            ColumnSpec("policy_family", "str"),
            ColumnSpec("algorithm", "str", enum=("cfqi_tabular", "iql_tabular")),
            ColumnSpec("training_size_multiplier", "int"),
            ColumnSpec("training_seed", "int"),
            ColumnSpec("normalized_return", "float"),
            ColumnSpec("success", "bool"),
            ColumnSpec("run_status", "str", enum=("observed", "invalid")),
        ),
    ),
    "behavior_policy_metadata": ArtifactSchema(
        name="behavior_policy_metadata",
        key_columns=("mdp_id", "policy_family"),
        columns=(
            ColumnSpec("mdp_id", "str"),
            ColumnSpec("policy_family", "str"),
            ColumnSpec("policy_parameters", "json"),
            ColumnSpec("observed_reachable_sa_fraction", "float"),
            ColumnSpec("observed_sa_fraction_all", "float"),
            ColumnSpec("observed_policy_support_fraction", "float"),
            ColumnSpec("fraction_policy_support_at_least_n_min", "float"),
            ColumnSpec("minimum_policy_support_count", "int"),
            ColumnSpec("normalized_visitation_entropy", "float"),
            ColumnSpec("num_transitions", "int"),
            ColumnSpec("num_trajectories_started", "int"),
        ),
    ),
    "empirical_sample_complexity": ArtifactSchema(
        name="empirical_sample_complexity",
        key_columns=("mdp_id", "policy_family", "algorithm"),
        columns=(
            ColumnSpec("mdp_id", "str"),
            ColumnSpec("policy_family", "str"),
            ColumnSpec("algorithm", "str", enum=("cfqi_tabular", "iql_tabular")),
            ColumnSpec("n_emp_multiplier", "int", required=False),
            ColumnSpec("n_emp_transitions", "int", required=False),
            ColumnSpec("n_emp_censored", "bool"),
            ColumnSpec("success_counts", "json"),
            ColumnSpec("success_rates", "json"),
        ),
        row_rules=_empirical_sample_complexity_rules(),
    ),
    "oeh_results": ArtifactSchema(
        name="oeh_results",
        key_columns=("mdp_id", "policy_family", "pilot_multiplier"),
        columns=(
            ColumnSpec("mdp_id", "str"),
            ColumnSpec("policy_family", "str"),
            ColumnSpec("pilot_multiplier", "int"),
            ColumnSpec("selected", "bool"),
            _status_column("oeh_status"),
            ColumnSpec("abstain_reason", "str", required=False),
            ColumnSpec("k_dagger_oeh", "int", required=False),
            ColumnSpec("effective_horizon", "float", required=False),
            ColumnSpec("n_req_hat", "float", required=False),
            ColumnSpec("valid_bootstrap_fraction", "float", required=False),
            ColumnSpec("gap_occupancy_at_k_dagger", "float", required=False),
        ),
        row_rules=_oeh_results_rules(),
        table_rules=(_oeh_results_selected_rule,),
    ),
    "exact_k_curve": ArtifactSchema(
        name="exact_k_curve",
        key_columns=("mdp_id", "policy_family", "depth"),
        columns=(
            ColumnSpec("mdp_id", "str"),
            ColumnSpec("policy_family", "str"),
            ColumnSpec("depth", "int"),
            ColumnSpec("delta_hat", "float", required=False),
            ColumnSpec("sigma_hat", "float", required=False),
            ColumnSpec("multiplier", "float"),
            ColumnSpec("k_dagger_exact", "int"),
            ColumnSpec("valid_sample_fraction", "float"),
        ),
    ),
    "abstention_results": ArtifactSchema(
        name="abstention_results",
        key_columns=("mdp_id", "policy_family", "pilot_multiplier"),
        columns=(
            ColumnSpec("mdp_id", "str"),
            ColumnSpec("policy_family", "str"),
            ColumnSpec("pilot_multiplier", "int"),
            ColumnSpec("oeh_status", "str", enum=("abstained", "censored")),
            ColumnSpec("reason", "str"),
        ),
    ),
}


def validate_rows(rows: Sequence[Row], schema: ArtifactSchema, *, context: str) -> list[dict[str, Any]]:
    coerced_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        row_context = f"{context}[row {index}]"
        missing = [column.name for column in schema.columns if column.name not in row]
        if missing:
            raise SchemaError(f"{row_context}: missing required column(s) {missing}")
        coerced: dict[str, Any] = {}
        for column in schema.columns:
            coerced[column.name] = _coerce(str(row[column.name]), column, context=row_context)
        key = tuple(coerced[name] for name in schema.key_columns)
        if key in seen_keys:
            raise SchemaError(f"{row_context}: duplicate key {schema.key_columns}={key}")
        seen_keys.add(key)
        for rule in schema.row_rules:
            problem = rule(coerced)
            if problem is not None:
                raise SchemaError(f"{row_context}: {problem}")
        coerced_rows.append(coerced)
    for rule in schema.table_rules:
        problem = rule(coerced_rows)
        if problem is not None:
            raise SchemaError(f"{context}: {problem}")
    return coerced_rows


def validate_artifact(path: Path, contract_name: str) -> list[dict[str, Any]]:
    if contract_name not in ARTIFACT_SCHEMAS:
        raise SchemaError(f"No registered schema named {contract_name!r}")
    schema = ARTIFACT_SCHEMAS[contract_name]
    rows = read_csv_rows(path)
    return validate_rows(rows, schema, context=str(path))


__all__ = ["ARTIFACT_SCHEMAS", "ArtifactSchema", "ColumnSpec", "SchemaError", "validate_artifact", "validate_rows"]

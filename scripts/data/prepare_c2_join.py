"""Prepare the Phase-B C2 handoff table without changing the raw RL labels.

The alternative-definition runner produces one requirement estimate per
``(mdp_id, policy_family, variant)``.  Empirical sample complexity is measured
per algorithm as well, so this module performs a validated many-to-one join
and repeats each data-only requirement for CFQI and IQL.

Right-censored ``N_emp`` observations remain blank.  They are retained in the
handoff table but are never converted to ``N_max`` or marked eligible for the
point-correlation analysis.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluation.io import read_csv_rows, sha256_file, write_csv_rows, write_json


REQUIRED_VARIANTS: tuple[str, ...] = (
    "registered",
    "materiality",
    "max_state",
    "probability",
)
MATERIALITY_VARIANTS = frozenset({"materiality", "max_state"})
N_REQ_STATUSES = frozenset({"finite", "unbounded", "abstained", "invalid"})

OUTPUT_FIELDS: tuple[str, ...] = (
    "mdp_id",
    "policy_family",
    "algorithm",
    "variant",
    "materiality_fraction",
    "oeh_status",
    "n_req_status",
    "min_multiplier",
    "n_req_hat",
    "n_emp_multiplier",
    "n_emp_transitions",
    "n_emp_censored",
    "correlation_eligible",
    "num_states",
    "num_actions",
    "structural_baseline_sa",
    "success_counts",
    "success_rates",
)


class C2JoinError(ValueError):
    """An input violates the pre-specified C2 handoff contract."""


def _require_columns(
    rows: Sequence[Mapping[str, Any]], required: Iterable[str], *, source: str
) -> None:
    if not rows:
        raise C2JoinError(f"{source}: no rows")
    missing = sorted(set(required) - set(rows[0]))
    if missing:
        raise C2JoinError(f"{source}: missing columns {missing}")


def _parse_bool(value: Any, *, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise C2JoinError(f"{label}: expected a boolean, got {value!r}")


def _parse_positive_int(value: Any, *, label: str) -> int:
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise C2JoinError(f"{label}: expected a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise C2JoinError(f"{label}: expected a positive integer, got {value!r}")
    return parsed


def _parse_requirement(value: Any, *, label: str) -> float:
    try:
        parsed = float(str(value).strip())
    except ValueError as exc:
        raise C2JoinError(f"{label}: expected a positive number or inf, got {value!r}") from exc
    if math.isnan(parsed) or parsed <= 0.0:
        raise C2JoinError(f"{label}: expected a positive number or inf, got {value!r}")
    return parsed


def _unique_by(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    source: str,
) -> dict[tuple[str, ...], Mapping[str, Any]]:
    indexed: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(str(row[field]).strip() for field in fields)
        if any(not item for item in key):
            raise C2JoinError(f"{source}: empty join key in {dict(row)!r}")
        if key in indexed:
            raise C2JoinError(f"{source}: duplicate key {key}")
        indexed[key] = row
    return indexed


def validate_n_emp_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_rows: int | None = 150
) -> list[dict[str, Any]]:
    """Validate labels while preserving right-censored blank values."""

    required = (
        "mdp_id",
        "policy_family",
        "algorithm",
        "n_emp_multiplier",
        "n_emp_transitions",
        "n_emp_censored",
    )
    _require_columns(rows, required, source="N_emp")
    if expected_rows is not None and len(rows) != expected_rows:
        raise C2JoinError(f"N_emp: expected {expected_rows} rows, got {len(rows)}")
    _unique_by(rows, ("mdp_id", "policy_family", "algorithm"), source="N_emp")

    validated: list[dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        key = (row["mdp_id"], row["policy_family"], row["algorithm"])
        censored = _parse_bool(row["n_emp_censored"], label=f"N_emp {key}")
        multiplier = str(row["n_emp_multiplier"]).strip()
        transitions = str(row["n_emp_transitions"]).strip()
        if censored and (multiplier or transitions):
            raise C2JoinError(
                f"N_emp {key}: censored rows must keep multiplier/transitions blank"
            )
        if not censored and (not multiplier or not transitions):
            raise C2JoinError(
                f"N_emp {key}: uncensored rows require multiplier and transitions"
            )
        if not censored:
            _parse_positive_int(multiplier, label=f"N_emp multiplier {key}")
            _parse_positive_int(transitions, label=f"N_emp transitions {key}")
        entry["n_emp_censored"] = censored
        validated.append(entry)
    return validated


def _selected_alternative_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    materiality_fraction: float,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    required = (
        "mdp_id",
        "policy_family",
        "variant",
        "materiality_fraction",
        "oeh_status",
        "min_multiplier",
        "n_req_hat",
    )
    _require_columns(rows, required, source="alternative definitions")

    selected: list[dict[str, Any]] = []
    variants_present = {str(row["variant"]).strip() for row in rows}
    missing_variants = sorted(set(REQUIRED_VARIANTS) - variants_present)
    if missing_variants:
        raise C2JoinError(f"alternative definitions: missing variants {missing_variants}")

    # The current runner emits only the first configured fraction when an OEH
    # pilot abstains, because no materiality-dependent curve exists to rescore.
    # Prefer an exact fixed-fraction row whenever one exists; otherwise retain
    # one blank abstention/invalid row and normalize its label to the fixed C2
    # fraction. This preserves failed pilots without pretending they were
    # numerically scored at another threshold.
    exact_materiality_keys: set[tuple[str, str, str]] = set()
    abstention_fallback: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for raw in rows:
        variant = str(raw.get("variant", "")).strip()
        if variant not in MATERIALITY_VARIANTS:
            continue
        key = (
            str(raw.get("mdp_id", "")).strip(),
            str(raw.get("policy_family", "")).strip(),
            variant,
        )
        try:
            fraction = float(str(raw.get("materiality_fraction", "")).strip())
        except ValueError:
            continue
        if math.isclose(fraction, materiality_fraction, rel_tol=0.0, abs_tol=1e-12):
            exact_materiality_keys.add(key)
        elif str(raw.get("oeh_status", "")).strip().lower() in {"abstained", "invalid"}:
            abstention_fallback.setdefault(key, raw)

    for raw in rows:
        variant = str(raw["variant"]).strip()
        if variant not in REQUIRED_VARIANTS:
            continue
        fraction_raw = str(raw["materiality_fraction"]).strip()
        if variant in MATERIALITY_VARIANTS:
            try:
                fraction = float(fraction_raw)
            except ValueError as exc:
                raise C2JoinError(
                    f"alternative definitions: invalid materiality fraction {fraction_raw!r}"
                ) from exc
            if not math.isclose(
                fraction, materiality_fraction, rel_tol=0.0, abs_tol=1e-12
            ):
                key = (
                    str(raw["mdp_id"]).strip(),
                    str(raw["policy_family"]).strip(),
                    variant,
                )
                if key in exact_materiality_keys or abstention_fallback.get(key) is not raw:
                    continue

        entry = dict(raw)
        oeh_status = str(raw["oeh_status"]).strip().lower()
        n_req_raw = str(raw["n_req_hat"]).strip()
        multiplier_raw = str(raw["min_multiplier"]).strip()
        explicit_status = str(raw.get("n_req_status", "")).strip().lower()

        if oeh_status in {"abstained", "invalid"}:
            if n_req_raw or multiplier_raw:
                raise C2JoinError(
                    f"{raw['mdp_id'], raw['policy_family'], variant}: "
                    f"{oeh_status} rows must leave requirement fields blank"
                )
            inferred_status = oeh_status
        else:
            if not n_req_raw or not multiplier_raw:
                raise C2JoinError(
                    f"{raw['mdp_id'], raw['policy_family'], variant}: "
                    "non-abstained row requires min_multiplier and n_req_hat"
                )
            n_req = _parse_requirement(n_req_raw, label="n_req_hat")
            multiplier = _parse_requirement(multiplier_raw, label="min_multiplier")
            if math.isinf(n_req) != math.isinf(multiplier):
                raise C2JoinError(
                    f"{raw['mdp_id'], raw['policy_family'], variant}: "
                    "n_req_hat and min_multiplier must both be finite or both unbounded"
                )
            inferred_status = "unbounded" if math.isinf(n_req) else "finite"

        if explicit_status:
            if explicit_status not in N_REQ_STATUSES:
                raise C2JoinError(f"unknown n_req_status {explicit_status!r}")
            if explicit_status != inferred_status:
                raise C2JoinError(
                    f"{raw['mdp_id'], raw['policy_family'], variant}: "
                    f"n_req_status={explicit_status!r} conflicts with {inferred_status!r}"
                )
        entry["n_req_status"] = inferred_status
        entry["materiality_fraction"] = (
            materiality_fraction if variant in MATERIALITY_VARIANTS else ""
        )
        selected.append(entry)

    indexed = _unique_by(
        selected,
        ("mdp_id", "policy_family", "variant"),
        source="selected alternative definitions",
    )
    return {(key[0], key[1], key[2]): dict(row) for key, row in indexed.items()}


def _metadata_by_mdp(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    _require_columns(
        rows, ("mdp_id", "num_states", "num_actions"), source="benchmark metadata"
    )
    indexed = _unique_by(rows, ("mdp_id",), source="benchmark metadata")
    result: dict[str, dict[str, Any]] = {}
    for (mdp_id,), row in indexed.items():
        states = _parse_positive_int(row["num_states"], label=f"{mdp_id} num_states")
        actions = _parse_positive_int(row["num_actions"], label=f"{mdp_id} num_actions")
        result[mdp_id] = {
            "num_states": states,
            "num_actions": actions,
            "structural_baseline_sa": states * actions,
        }
    return result


def build_c2_join(
    n_emp_rows: Sequence[Mapping[str, Any]],
    alternative_rows: Sequence[Mapping[str, Any]],
    metadata_rows: Sequence[Mapping[str, Any]],
    *,
    materiality_fraction: float = 0.01,
    expected_n_emp_rows: int | None = 150,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the validated long-form C2 handoff and its audit counts."""

    if not 0.0 <= materiality_fraction < 1.0:
        raise C2JoinError("materiality_fraction must lie in [0, 1)")
    n_emp = validate_n_emp_rows(n_emp_rows, expected_rows=expected_n_emp_rows)
    alternatives = _selected_alternative_rows(
        alternative_rows, materiality_fraction=materiality_fraction
    )
    metadata = _metadata_by_mdp(metadata_rows)

    expected_pairs = {
        (str(row["mdp_id"]).strip(), str(row["policy_family"]).strip())
        for row in n_emp
    }
    alternative_pairs = {(key[0], key[1]) for key in alternatives}
    if alternative_pairs != expected_pairs:
        missing = sorted(expected_pairs - alternative_pairs)
        extra = sorted(alternative_pairs - expected_pairs)
        raise C2JoinError(
            f"alternative pair coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    for pair in sorted(expected_pairs):
        missing = [variant for variant in REQUIRED_VARIANTS if (*pair, variant) not in alternatives]
        if missing:
            raise C2JoinError(f"alternative definitions {pair}: missing variants {missing}")
        if pair[0] not in metadata:
            raise C2JoinError(f"benchmark metadata: missing mdp_id {pair[0]!r}")

    joined: list[dict[str, Any]] = []
    for empirical in sorted(
        n_emp,
        key=lambda row: (row["mdp_id"], row["policy_family"], row["algorithm"]),
    ):
        mdp_id = str(empirical["mdp_id"]).strip()
        policy = str(empirical["policy_family"]).strip()
        algorithm = str(empirical["algorithm"]).strip()
        for variant in REQUIRED_VARIANTS:
            alternative = alternatives[(mdp_id, policy, variant)]
            n_req_status = alternative["n_req_status"]
            eligible = not empirical["n_emp_censored"] and n_req_status == "finite"
            joined.append(
                {
                    "mdp_id": mdp_id,
                    "policy_family": policy,
                    "algorithm": algorithm,
                    "variant": variant,
                    "materiality_fraction": alternative["materiality_fraction"],
                    "oeh_status": alternative["oeh_status"],
                    "n_req_status": n_req_status,
                    "min_multiplier": alternative["min_multiplier"],
                    "n_req_hat": alternative["n_req_hat"],
                    "n_emp_multiplier": empirical["n_emp_multiplier"],
                    "n_emp_transitions": empirical["n_emp_transitions"],
                    "n_emp_censored": empirical["n_emp_censored"],
                    "correlation_eligible": eligible,
                    **metadata[mdp_id],
                    "success_counts": empirical.get("success_counts", ""),
                    "success_rates": empirical.get("success_rates", ""),
                }
            )

    expected_joined = len(n_emp) * len(REQUIRED_VARIANTS)
    if len(joined) != expected_joined:
        raise AssertionError(f"expected {expected_joined} joined rows, got {len(joined)}")

    emp_by_algorithm: dict[str, Counter[str]] = defaultdict(Counter)
    for row in n_emp:
        status = "censored" if row["n_emp_censored"] else "observed"
        emp_by_algorithm[str(row["algorithm"])][status] += 1
    requirement_counts: dict[str, Counter[str]] = defaultdict(Counter)
    eligible_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in joined:
        requirement_counts[str(row["variant"])][str(row["n_req_status"])] += 1
        if row["correlation_eligible"]:
            eligible_counts[str(row["variant"])][str(row["algorithm"])] += 1

    report = {
        "contract": {
            "censored_n_emp_substitution": "forbidden",
            "point_correlations": "finite N_emp and finite n_req_hat only",
            "bootstrap_cluster": "mdp_id",
            "materiality_fraction": materiality_fraction,
        },
        "n_emp": {
            "rows": len(n_emp),
            "unique_mdp_policy_pairs": len(expected_pairs),
            "by_algorithm": {
                algorithm: dict(sorted(counts.items()))
                for algorithm, counts in sorted(emp_by_algorithm.items())
            },
        },
        "joined_rows": len(joined),
        "requirement_status_by_variant": {
            variant: dict(sorted(requirement_counts[variant].items()))
            for variant in REQUIRED_VARIANTS
        },
        "correlation_eligible_by_variant_algorithm": {
            variant: dict(sorted(eligible_counts[variant].items()))
            for variant in REQUIRED_VARIANTS
        },
    }
    return joined, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the validated Phase-B C2 join.")
    parser.add_argument("--n-emp", type=Path, required=True)
    parser.add_argument("--alternatives", type=Path, required=True)
    parser.add_argument("--benchmark-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--materiality-fraction", type=float, default=0.01)
    parser.add_argument("--expected-n-emp-rows", type=int, default=150)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty; refusing to overwrite it")
    inputs = {
        "n_emp": args.n_emp,
        "alternatives": args.alternatives,
        "benchmark_metadata": args.benchmark_metadata,
    }
    joined, report = build_c2_join(
        read_csv_rows(args.n_emp),
        read_csv_rows(args.alternatives),
        read_csv_rows(args.benchmark_metadata),
        materiality_fraction=args.materiality_fraction,
        expected_n_emp_rows=args.expected_n_emp_rows,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "c2_joined_requirements.csv"
    write_csv_rows(output_csv, joined, OUTPUT_FIELDS)
    report["provenance"] = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sorted(inputs.items())
    }
    report["artifact_sha256"] = {output_csv.name: sha256_file(output_csv)}
    write_json(args.output_dir / "c2_join_report.json", report)


if __name__ == "__main__":
    main()

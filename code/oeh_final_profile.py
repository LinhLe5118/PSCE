"""System contract for the final OEH profile.

The information-theoretic target is defined in
``docs/research/OEH_FINAL_FORMULATION_2026-09-19.md``.  This adapter does not
pretend that the current empirical-Bernstein plug-in is the exact KL optimizer.
It gives the existing development estimator the final output shape and keeps the
data-budget selection separate from the EH-reference scalar.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def _finite_nonnegative(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0.0 else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "True", "TRUE"):
        return True
    if value in (0, "0", "false", "False", "FALSE", "", None):
        return False
    raise ValueError(f"Invalid Boolean value: {value!r}")


def build_final_proxy_profile(
        records: Iterable[Mapping[str, Any]], *,
        compute_budget: int | None = None,
        requirement_key: str = "eb_sample_requirement",
        evidence_key: str = "eb_evidence") -> dict[str, Any]:
    """Convert legacy EB depth records to the final OEH profile contract.

    ``compute_budget`` is measured in the original-paper QVI index: ``Q^1`` is
    the behavior-policy leaf followed by supported Bellman backups.  Missing or
    unresolved requirements are represented by ``None`` plus a right-censoring
    flag, avoiding non-standard JSON infinities.
    """
    source = [dict(item) for item in records]
    if not source:
        raise ValueError("At least one depth record is required")
    qvi_indices = [int(item["qvi_index"]) for item in source]
    if len(set(qvi_indices)) != len(qvi_indices) or min(qvi_indices) < 1:
        raise ValueError("QVI indices must be unique positive integers")
    maximum = max(qvi_indices)
    budget = maximum if compute_budget is None else int(compute_budget)
    if budget < 1:
        raise ValueError("compute_budget must be positive")

    profile = []
    explicit_statuses = {
        "identified", "epsilon_trivial", "unresolved", "unsupported",
        "not_epsilon_solvable"
    }
    for item in sorted(source, key=lambda row: int(row["qvi_index"])):
        qvi_index = int(item["qvi_index"])
        admissible = _as_bool(item.get("admissible"))
        identified = _as_bool(item.get("identified"))
        requirement = _finite_nonnegative(item.get(requirement_key))
        evidence = _finite_nonnegative(item.get(evidence_key))
        explicit_status = item.get("crossfit_status")
        if explicit_status is not None and explicit_status not in explicit_statuses:
            raise ValueError(f"Invalid explicit profile status: {explicit_status!r}")
        if explicit_status is not None:
            status = str(explicit_status)
        elif not admissible:
            status = "not_epsilon_solvable"
        elif not identified or requirement is None:
            status = "unresolved"
        else:
            status = "identified"
        resolved = status in {"identified", "epsilon_trivial"}
        usable = resolved and qvi_index <= budget
        profile.append({
            "depth": int(item.get("depth", qvi_index-1)),
            "qvi_index": qvi_index,
            "status": status,
            "status_basis": ("trajectory_crossfit_eb_oeh_development_proxy"
                             if explicit_status is not None else
                             "eb_oeh_plugin_development_proxy"),
            "within_compute_budget": qvi_index <= budget,
            "requirement_censored_right": not resolved,
            "n_hat_eb": requirement if resolved else None,
            "log_a_n_hat_eb": evidence if resolved else None,
            "eh_reference": (qvi_index+evidence
                             if resolved and evidence is not None
                             else None),
            "eligible_for_selection": usable,
        })

    eligible = [item for item in profile if item["eligible_for_selection"]]
    data_choice = min(
        eligible, key=lambda item: (item["n_hat_eb"], item["qvi_index"]),
        default=None)
    eh_choice = min(
        eligible, key=lambda item: (item["eh_reference"], item["qvi_index"]),
        default=None)

    def selection(item: Mapping[str, Any] | None, objective: str):
        if item is None:
            return {
                "status": "abstained", "objective": objective,
                "qvi_index": None, "depth": None, "n_hat_eb": None,
            }
        return {
            "status": "selected", "objective": objective,
            "qvi_index": item["qvi_index"], "depth": item["depth"],
            "n_hat_eb": item["n_hat_eb"],
        }

    return {
        "formula_version": "oeh_identification_profile_v1",
        "estimator": "eb_oeh_plugin_development_proxy",
        "exact_kl_target_implemented": False,
        "cross_fitted": False,
        "compute_budget_qvi": budget,
        "profile": profile,
        "primary_data_selection": selection(data_choice, "min_n_hat_eb"),
        "eh_reference_selection": selection(eh_choice, "min_k_plus_log_a_n_hat_eb"),
    }


__all__ = ["build_final_proxy_profile"]

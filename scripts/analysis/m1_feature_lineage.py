"""M1 feature lineage of the sealed 108-column ``plus_prior_profile`` candidate.

Every column of the seventh-cohort protocol (``oeh_final.fields``) is assigned
to exactly one information block, its producer and sealed source artifact, the
upstream inputs it depends on, the compute component that pays for it and its
learner scope. The ablation arms are defined from these blocks, never from
outcome results, and keep the sealed column order.

Nothing in this module reads learner outcomes. Lineage is a description of the
existing pipeline; it changes no algorithm, protocol or sealed artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS
from scripts.analysis import oeh_final_forecast as F

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
PROTOCOL = COHORT / "protocol.json"
BUDGETS = F.BUDGETS

# Sealed pre-outcome producers. Paths are relative to the cohort directory.
PREOUTCOME_SOURCES = ("pilot_features.csv", "oeh_b_groups/*.json", "horizon_preoutcome.csv")
# Cohort-7 files that hold learner outcomes or anything scored against them.
OUTCOME_FILES = ("labels.csv", "all_core_budget_seed_outcomes.csv", "scored_predictions.csv",
                 "mdp_scores.csv", "summary.csv", "contrasts.csv", "confirmatory_verdict.json",
                 "result_manifest.json", "selection_group_outcomes.csv", "selection_summary.csv",
                 "selection_contrasts.csv", "selection_vs_every_learner.csv",
                 "horizon_comparison.csv", "horizon_panel.csv", "execution_complete.json")
OUTCOME_DIRS = ("execution", "external_execution")
# Columns that carry outcomes or true-MDP (privileged) information.
LABEL_COLUMNS = ("event", "panel", "cohort", *(f"successes_x{b}" for b in BUDGETS))
PRIVILEGED_PREFIXES = ("oracle_", "eh_", "tq_", "saco_", "policy_", "true_")

COMPONENTS = {
    "b2_statistics": {
        "producer": "code/stochastic_features.py::extract_stochastic_features",
        "inputs": ["pilot_x10", "public_S_A", "gamma_0.95"],
        "note": "200 trajectory-bootstrap replicates; one call per (MDP, behavior)",
    },
    "oeh3_risk": {
        "producer": "code/oeh_risk_v3.py::estimate_oeh_risk_v3 + "
                    "code/stochastic_features.py::aggregate_oeh_risk_v3_profile",
        "inputs": ["pilot_x10", "public_S_A", "public_rho", "gamma_0.95"],
        "note": "k_max=100; learner-agnostic; one call per (MDP, behavior)",
    },
    "lcoeh4_crossfit": {
        "producer": "code/lcoeh_v4.py::estimate_lcoeh_v4",
        "inputs": ["pilot_prefix_x2", "pilot_prefix_x5", "pilot_x10", "public_S_A", "public_rho",
                   "gamma_0.95", "learner_code_BC_CFQI_IQL"],
        "note": "trains each core learner on 4/5 of the pilot trajectories at three nested prefixes; "
                "one call returns all three learners",
    },
    "posterior_fit": {
        "producer": "code/oeh/posterior.py::pilot_statistics + fit_posterior",
        "inputs": ["pilot_x10", "public_S_A", "public_rho", "gamma_0.95", "structure_A2"],
        "note": "empirical-Bayes Dirichlet concentrations; also a prerequisite of OEH-H",
    },
    "oehb_simulation": {
        "producer": "code/oeh/predictive.py::posterior_outcomes + requirement_forecast "
                    "(scripts/analysis/run_oeh_final_120.py::oeh_b_group)",
        "inputs": ["posterior_fit", "learner_code_BC_CFQI_IQL", "64_posterior_draws",
                   "6_budgets_x_10_seeds"],
        "note": "replays the three core learners on simulated nested streams; one call per (MDP, behavior)",
    },
    "oehh_horizon": {
        "producer": "code/oeh/horizon.py::offline_effective_horizon(form=risk) + offline_theory_measures + "
                    "asymptotic_ceilings + risk_profile on "
                    "scripts/analysis/develop_oeh_horizon.py::posterior_mean_model "
                    "(scripts/analysis/run_oeh_final_120.py::horizon_group)",
        "inputs": ["posterior_fit", "pilot_x10_observed_mask", "public_rho", "gamma_0.95", "tau_0.95",
                   "horizon_100"],
        "note": "posterior-mean model only, no learner replay; one call per (MDP, behavior)",
    },
}
# Compute prerequisites: a component also needs every component listed here.
REQUIRES = {"oehb_simulation": ("posterior_fit",), "oehh_horizon": ("posterior_fit",)}

BLOCKS = {
    "G1": "B2 generic pilot statistics",
    "G2": "OEH-B summaries of the row's learner",
    "G3": "OEH-B posterior concentrations",
    "G4": "OEH-H descriptors and ceilings",
    "G5": "OEH-H risk and missing-mass profile",
    "G6": "OEH-B CDF context of all core learners",
    "G7": "Legacy OEH3/LCOEH4 diagnostics of the row",
    "G8": "Legacy LCOEH4 context of all core learners",
}
# Arms keep every block except the dropped ones. "Direct OEH-B" is the name
# required by the Week 8 plan: OEH-H still uses the posterior mean.
ARMS = {
    "b2_matched": ("G2", "G3", "G4", "G5", "G6", "G7", "G8"),
    "oeh_full_108": (),
    "oeh_without_direct_oehb": ("G2", "G3", "G6"),
    "oeh_without_oehh_risk": ("G4", "G5"),
    "oeh_without_legacy_diagnostics": ("G7", "G8"),
}

B2_MEANING = {
    "log_pilot_transitions": "log pilot size N0",
    "log_state_action_count": "log |S||A| (public metadata)",
    "observed_sa_fraction": "fraction of state-action pairs seen in the pilot",
    "observed_state_fraction": "fraction of states seen in the pilot",
    "normalized_visitation_entropy": "entropy of state-action visitation / log |S||A|",
    "log1p_min_positive_sa_count": "log1p of the smallest positive state-action count",
    "trajectory_return_mean": "mean discounted pilot trajectory return",
    "trajectory_return_std": "std of discounted pilot trajectory return",
    "trajectory_return_iqr": "IQR of discounted pilot trajectory return",
    "terminal_trajectory_fraction": "fraction of pilot trajectories reaching a terminal",
    "weighted_transition_entropy": "count-weighted normalized next-state entropy",
    "observed_cell_transition_entropy": "mean normalized next-state entropy over seen cells",
    "low_count_cell_fraction": "fraction of cells with 1-2 visits",
    "empirical_stochastic_cell_fraction": "fraction of cells (>=2 visits) with several successors",
    "low_coverage_entropy": "mean next-state entropy of cells with <3 visits",
    "high_coverage_entropy": "mean next-state entropy of cells with >=3 visits",
    "bootstrap_return_mean_std": "trajectory-bootstrap std of mean return",
    "bootstrap_rare_exposure_mean_std": "trajectory-bootstrap std of rare-transition exposure",
}
OEHH_SOURCE = {
    "oehh_value": ("data_oeh_h", "nan->0"),
    "oehh_identified": ("data_oeh_h_status", "status=='identified'"),
    "oehh_trap": ("data_oeh_h_status", "status=='coverage_trap'"),
    "oehh_statistical": ("data_oeh_h_statistical", "nan->0"),
    "oehh_coverage": ("data_oeh_h_coverage", "nan->0"),
    "oehh_depth_fraction": ("data_oeh_h_depth", "nan->101, then /100"),
    "oehh_log_concentrability": ("data_log_concentrability", "nan->inf->1e6 in matrix"),
    "oehh_log_clipped_concentrability": ("data_log_clipped_concentrability", "nan->inf->1e6 in matrix"),
    "oehh_log_intrinsic_bound": ("data_log_intrinsic_bound", "nan->inf->1e6 in matrix"),
    "oehh_behavior_ceiling": ("data_behavior_ceiling", "nan->0"),
    "oehh_support_ceiling": ("data_support_ceiling", "nan->0"),
    **{f"oehh_{kind}_x{b}": (f"data_{kind}_x{b}", "nan->1") for kind in ("risk", "miss") for b in BUDGETS},
    "oehh_risk_slack": ("data_risk_slack", "nan->-1"),
}


def sealed_fields() -> tuple[str, ...]:
    """The 108 sealed candidate columns, verified against the runner definition."""
    from scripts.analysis import run_oeh_final_120 as R

    doc = json.loads(PROTOCOL.read_text(encoding="utf-8"))["oeh_final"]
    fields = tuple(doc["fields"])
    if doc["candidate_variant"] != "plus_prior_profile" or doc["candidate_law"]:
        raise RuntimeError("Sealed candidate is not plus_prior_profile without the law component")
    if fields != tuple(R.CANDIDATE_VARIANTS["plus_prior_profile"]):
        raise RuntimeError("Sealed fields differ from run_oeh_final_120.CANDIDATE_VARIANTS")
    if len(fields) != 108 or len(set(fields)) != 108:
        raise RuntimeError("Sealed candidate must contain 108 distinct columns")
    return fields


def block_of(column: str) -> str:
    if column in FEATURES_B2:
        return "G1"
    if column in F.OWN:
        return "G2"
    if column in ("oehb_log_transition_concentration", "oehb_log_behavior_concentration"):
        return "G3"
    if column in F.PROFILE:
        return "G5"
    if column.startswith("oehh_"):
        return "G4"
    if column in F.CONTEXT:
        return "G6"
    if column.startswith(("oeh3_", "lcoeh4_")):
        return "G7"
    if column.startswith("other_") and "_lcoeh4_" in column:
        return "G8"
    raise KeyError(f"Column without lineage: {column}")


def _context_learner(column: str) -> str:
    for method in METHODS:
        if column.startswith(f"other_{method}_"):
            return method
    raise KeyError(column)


def _row(position: int, column: str) -> dict:
    block = block_of(column)
    row = {"position": position, "column": column, "block": block, "block_label": BLOCKS[block],
           "outcome_access": False, "privileged_access": False}
    if block == "G1":
        row.update(meaning=B2_MEANING[column], source_artifact="pilot_features.csv", source_field=column,
                   component="b2_statistics", learner_scope="shared", uses_posterior=False,
                   uses_learner_replay=False, encoding="finite by construction")
    elif block == "G2":
        kind, index = column.split("_")[1], column.rsplit("_", 1)[1]
        meaning = {"cdf": f"posterior-predictive P(E <= class {index}) after KT smoothing",
                   "success": f"mean simulated seed success (NR>=0.95) at budget {index}",
                   "return": f"mean simulated normalized return at budget {index}"}[kind]
        field = {"cdf": "learners[l].probabilities (cumulative)", "success": "learners[l].success_rate",
                 "return": "learners[l].mean_normalized_return"}[kind]
        row.update(meaning=meaning, source_artifact="oeh_b_groups/*.json", source_field=field,
                   component="oehb_simulation", learner_scope="own", uses_posterior=True,
                   uses_learner_replay=True, encoding="missing success/return lists -> 0")
    elif block == "G3":
        kind = column.split("_")[2]
        row.update(meaning=f"log empirical-Bayes {kind} Dirichlet concentration",
                   source_artifact="oeh_b_groups/*.json", source_field=f"posterior.{kind}_concentration",
                   component="posterior_fit", learner_scope="shared", uses_posterior=True,
                   uses_learner_replay=False, encoding="log; nonfinite or zero -> 0")
    elif block in ("G4", "G5"):
        source, encoding = OEHH_SOURCE[column]
        meaning = ("modeled noise regret at budget" if column.startswith("oehh_risk_x") else
                   "modeled exponential missing mass at budget" if column.startswith("oehh_miss_x") else
                   "NR slack used by the risk inversion" if column == "oehh_risk_slack" else
                   "OEH-H descriptor or ceiling on the posterior-mean model")
        row.update(meaning=meaning, source_artifact="horizon_preoutcome.csv", source_field=source,
                   component="oehh_horizon", learner_scope="shared", uses_posterior=True,
                   uses_learner_replay=False, encoding=encoding)
    elif block == "G6":
        learner = _context_learner(column)
        row.update(meaning=f"OEH-B CDF of {learner} copied to every learner row of the group",
                   source_artifact="oeh_b_groups/*.json", source_field=f"learners[{learner}].probabilities",
                   component="oehb_simulation", learner_scope="cross_learner", uses_posterior=True,
                   uses_learner_replay=True, encoding="as G2")
    elif block == "G7":
        oeh3 = column.startswith("oeh3_")
        row.update(meaning=("OEH3 return-weighted identification risk summary" if oeh3 else
                            "LCOEH4 cross-fitted learner diagnostic of the row's learner"),
                   source_artifact="pilot_features.csv", source_field=column,
                   component="oeh3_risk" if oeh3 else "lcoeh4_crossfit",
                   learner_scope="shared" if oeh3 else "own", uses_posterior=False,
                   uses_learner_replay=not oeh3, encoding="finite by construction")
    else:
        learner = _context_learner(column)
        row.update(meaning=f"LCOEH4 diagnostic of {learner} copied to every learner row of the group",
                   source_artifact="pilot_features.csv",
                   source_field=column.replace(f"other_{learner}_", "") + f" (algorithm={learner})",
                   component="lcoeh4_crossfit", learner_scope="cross_learner", uses_posterior=False,
                   uses_learner_replay=True, encoding="finite by construction")
    component = row["component"]
    row["depends_on"] = ";".join(COMPONENTS[component]["inputs"])
    row["producer"] = COMPONENTS[component]["producer"]
    return row


def lineage() -> list[dict]:
    rows = [_row(i, column) for i, column in enumerate(sealed_fields())]
    check_lineage(rows)
    return rows


def check_lineage(rows: list[dict]) -> None:
    columns = [row["column"] for row in rows]
    counts = {block: sum(row["block"] == block for row in rows) for block in BLOCKS}
    expected = {"G1": 18, "G2": 18, "G3": 2, "G4": 11, "G5": 13, "G6": 18, "G7": 10, "G8": 18}
    if counts != expected or len(columns) != 108 or len(set(columns)) != 108:
        raise RuntimeError(f"Block partition changed: {counts}")
    for row in rows:
        if row["outcome_access"] or row["privileged_access"]:
            raise RuntimeError(f"Column with outcome or privileged access: {row['column']}")
        if row["source_artifact"] not in PREOUTCOME_SOURCES:
            raise RuntimeError(f"Column not produced by a sealed pre-outcome artifact: {row['column']}")
        if row["column"] in LABEL_COLUMNS or row["column"].startswith(PRIVILEGED_PREFIXES):
            raise RuntimeError(f"Outcome-like column name: {row['column']}")


def arms(rows: list[dict] | None = None) -> dict[str, tuple[str, ...]]:
    """Column tuples of every arm, in sealed order."""
    rows = rows if rows is not None else lineage()
    result = {name: tuple(row["column"] for row in rows if row["block"] not in dropped)
              for name, dropped in ARMS.items()}
    if result["b2_matched"] != tuple(FEATURES_B2) or len(result["oeh_full_108"]) != 108:
        raise RuntimeError("Arm definitions no longer match B2 and the sealed candidate")
    return result


def components_of(columns) -> tuple[str, ...]:
    """Compute components (with prerequisites) needed to produce ``columns``."""
    by_column = {row["column"]: row["component"] for row in lineage()}
    needed = {by_column[column] for column in columns}
    for name in list(needed):
        needed.update(REQUIRES.get(name, ()))
    return tuple(name for name in COMPONENTS if name in needed)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

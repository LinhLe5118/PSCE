#!/usr/bin/env python
"""C2 correlation runner -- ``n_req_hat`` vs ``N_emp`` against the ``S*A`` baseline.

Owner: Linh (Workstream 4 -- Statistical Evaluation & Research Protocol).

Two modes, and the difference between them is the point of the file:

``--synthetic``
    Runs the whole table on generated data with a known planted structure. No
    real artifact is read, no registry gate applies, and the output directory
    is labelled as a dry run. This is the deliverable for days 1-3: the frame
    exists and is exercised before any real coefficient is visible, so day 4 is
    a data swap and not a design decision (``PHASE_B_PLAN.md`` week 1).

real inputs
    Refuses to run until the ``n_emp_censored`` handling rule is recorded in
    ``artifacts/week2/ARM_REGISTRY.md``. That is ``EXPERIMENT_RULE.md`` rule 2
    -- the rule is fixed *before* a coefficient is seen -- enforced in code
    rather than trusted to discipline, because the person running this script
    is also the person who would be tempted to adjust the rule after seeing the
    number. The registry entry is Workstream 2's (Nguyen Quoc Viet) to write.

Usage::

    python -m scripts.analysis.run_c2_correlation --synthetic --output-dir artifacts/week2/c2_correlation_dryrun_<date>

    python -m scripts.analysis.run_c2_correlation \\
      --alternative-definitions artifacts/week2/alternative_definitions_<date>/alternative_definitions.csv \\
      --empirical-sample-complexity artifacts/week2/development_offline_rl_<date>/empirical_sample_complexity.csv \\
      --benchmark-metadata artifacts/week2/benchmark_oracle_handoff_<date>/benchmark_metadata.csv \\
      --censoring-rule complete_case \\
      --output-dir artifacts/week2/c2_correlation_<date>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from evaluation.aggregation.c2_analysis import (  # noqa: E402
    BASELINE_COMPARISON_FIELDS,
    CENSORING_RULES,
    DEFAULT_CENSORING_RULE,
    FAMILY_ROBUSTNESS_FIELDS,
    OUTPUT_FIELDS,
    JoinedPair,
    beats_structural_baseline,
    multiplicity_report,
    run_baseline_comparison,
    run_c2_analysis,
    run_family_robustness,
    success_criteria_3_and_4,
)
from evaluation.io import read_csv_rows, sha256_file, write_csv_rows, write_json  # noqa: E402
from config.protocol_config import PROTOCOL_CONFIG, PROTOCOL_CONFIG_PATH  # noqa: E402

#: Declared before the first run, per ``EXPERIMENT_RULE.md`` rule 6. Changing
#: it after a real run has been seen is a new registry entry, not an edit.
ANALYSIS_SEED = 20260901
ARM_REGISTRY_PATH = ROOT / "artifacts" / "week2" / "ARM_REGISTRY.md"
#: The machine-readable marker in ARM_REGISTRY.md, e.g.
#: ``censoring_rule: complete_case``.
_REGISTRY_RULE_PATTERN = re.compile(r"`?censoring_rule`?\s*:\s*`?([a-z_]+)`?", re.IGNORECASE)

#: Registry entries are written for people, in Vietnamese, and their names are
#: not this module's vocabulary. "Quy tac C2-2" (registered 2026-08-31 by
#: Workstream 2) clause 2 reads: *Pearson/Spearman chi tinh tren cac dong dong
#: thoi co N_emp huu han va n_req_hat huu han* -- which is exactly
#: ``complete_case``. The binding is spelled out here so the translation from
#: the human entry to the code path is auditable in one place rather than
#: assumed at the call site. A heading alone never unblocks a run: the
#: machine-readable ``censoring_rule:`` line must also be present and agree, so
#: what unblocks the gate is something a person signed, not something a regex
#: inferred.
REGISTERED_RULE_ALIASES: dict[str, str] = {"c2-2": "complete_case"}
_REGISTRY_HEADING_PATTERN = re.compile(r"Quy\s+t\w*c\s+(C2-\d+)", re.IGNORECASE)


def registry_rule_headings(registry_path: Path = ARM_REGISTRY_PATH) -> list[str]:
    """Named censoring-rule entries present in the registry, lowercased."""

    if not registry_path.exists():
        return []
    text = registry_path.read_text(encoding="utf-8")
    # Later entries cite the rule by name too, so the same heading appears more
    # than once; deduplicate in first-seen order rather than reporting one rule
    # four times in a mismatch message.
    found = [match.group(1).lower() for match in _REGISTRY_HEADING_PATTERN.finditer(text)]
    return list(dict.fromkeys(found))

VERDICT_FIELDS: tuple[str, ...] = (
    "variant",
    "materiality_fraction",
    "analysis_role",
    "scale",
    "method",
    "algorithm_scope",
    "coefficient",
    "ci_low",
    "ci_high",
    "n_pairs",
    "num_clusters",
    "ci_excludes_zero",
    "undefined_reason",
)


# --------------------------------------------------------------------------
# Registry gate (EXPERIMENT_RULE.md rule 2)
# --------------------------------------------------------------------------


def registered_censoring_rule(registry_path: Path = ARM_REGISTRY_PATH) -> str | None:
    """The rule recorded in ``ARM_REGISTRY.md``, or ``None`` if none is."""

    if not registry_path.exists():
        return None
    for match in _REGISTRY_RULE_PATTERN.finditer(registry_path.read_text(encoding="utf-8")):
        candidate = match.group(1).lower()
        if candidate in CENSORING_RULES:
            return candidate
    return None


def assert_rule_registered(requested: str, *, registry_path: Path = ARM_REGISTRY_PATH) -> str:
    recorded = registered_censoring_rule(registry_path)
    if recorded is None:
        raise SystemExit(
            "Refusing to correlate real data: no n_emp_censored handling rule is recorded in\n"
            f"  {registry_path}\n"
            "EXPERIMENT_RULE.md rule 2 requires the rule to be fixed before any correlation\n"
            "coefficient is looked at. Workstream 2 (Nguyen Quoc Viet) owns this entry. Add a\n"
            "line of the form\n\n"
            "    censoring_rule: complete_case\n\n"
            "with the reasoning, then rerun. Admissible values:\n  - "
            + "\n  - ".join(sorted(CENSORING_RULES))
            + "\n\nTo exercise the analysis without real data, use --synthetic."
        )
    if recorded != requested:
        raise SystemExit(
            f"--censoring-rule {requested!r} does not match the rule recorded in "
            f"{registry_path}: {recorded!r}.\nThe registry is the pre-registration; the flag is "
            "not allowed to override it (EXPERIMENT_RULE.md rule 2)."
        )
    headings = registry_rule_headings(registry_path)
    bound = {REGISTERED_RULE_ALIASES[name] for name in headings if name in REGISTERED_RULE_ALIASES}
    if bound and recorded not in bound:
        raise SystemExit(
            f"The registry's named rule entries {sorted(headings)} bind to {sorted(bound)}, but "
            f"the machine-readable line says {recorded!r}. Those must agree before a real run "
            "(EXPERIMENT_RULE.md rule 2); reconcile the registry rather than picking one here."
        )
    return recorded


# --------------------------------------------------------------------------
# Real-input join
# --------------------------------------------------------------------------


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _bool(value: str | None) -> bool:
    if value is None:
        raise ValueError("missing boolean field")
    text = value.strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unparseable boolean {value!r}")


def join_pairs(
    definition_rows: list[dict[str, Any]],
    n_emp_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
) -> tuple[list[JoinedPair], dict[str, Any]]:
    """Join on ``(mdp_id, policy_family)`` x ``algorithm``.

    One OEH row fans out to one row per algorithm, exactly as
    ``alternative_definitions_2026-09-01/README.md`` describes the key. Any
    definition row with no matching label row, or any label row with no
    matching definition row, is counted and reported -- never dropped
    silently, because a shrinking join is the most plausible way for this
    analysis to end up quietly reporting a different sample than it claims.
    """

    metadata_by_mdp = {row["mdp_id"]: row for row in metadata_rows}
    labels: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in n_emp_rows:
        labels.setdefault((row["mdp_id"], row["policy_family"]), []).append(row)

    pairs: list[JoinedPair] = []
    unmatched_definitions = 0
    missing_metadata: set[str] = set()
    matched_label_keys: set[tuple[str, str, str]] = set()

    for row in definition_rows:
        key = (row["mdp_id"], row["policy_family"])
        metadata = metadata_by_mdp.get(row["mdp_id"])
        if metadata is None:
            missing_metadata.add(row["mdp_id"])
            continue
        label_rows = labels.get(key)
        if not label_rows:
            unmatched_definitions += 1
            continue
        for label in label_rows:
            matched_label_keys.add((*key, label["algorithm"]))
            pairs.append(
                JoinedPair(
                    mdp_id=row["mdp_id"],
                    policy_family=row["policy_family"],
                    algorithm=label["algorithm"],
                    variant=row["variant"],
                    materiality_fraction=float(row["materiality_fraction"]),
                    n_req_hat=_optional_float(row.get("n_req_hat")),
                    oeh_status=row["oeh_status"],
                    n_emp_transitions=_optional_float(label.get("n_emp_transitions")),
                    n_emp_censored=_bool(label.get("n_emp_censored")),
                    num_states=int(metadata["num_states"]),
                    num_actions=int(metadata["num_actions"]),
                )
            )

    all_label_keys = {(row["mdp_id"], row["policy_family"], row["algorithm"]) for row in n_emp_rows}
    join_report = {
        "definition_rows": len(definition_rows),
        "label_rows": len(n_emp_rows),
        "joined_pairs": len(pairs),
        "definition_rows_without_label": unmatched_definitions,
        "label_rows_without_definition": len(all_label_keys - matched_label_keys),
        "mdp_ids_missing_metadata": sorted(missing_metadata),
    }
    return pairs, join_report


#: Columns ``scripts/data/prepare_c2_join.py`` writes (``c2_join_2026-09-02/README.md``).
JOINED_TABLE_FIELDS: tuple[str, ...] = (
    "mdp_id",
    "policy_family",
    "algorithm",
    "variant",
    "materiality_fraction",
    "oeh_status",
    "n_req_hat",
    "n_emp_transitions",
    "n_emp_censored",
    "correlation_eligible",
    "num_states",
    "num_actions",
)


def read_joined_table(rows: list[dict[str, Any]]) -> tuple[list[JoinedPair], dict[str, Any]]:
    """Read the registered C2 handoff (``c2_joined_requirements.csv``).

    This is the preferred real-mode input. ``ARM_REGISTRY.md`` rule C2-2 names
    ``scripts/data/prepare_c2_join.py`` as *the* pipeline implementing the rule, so redoing
    the join here would be a second implementation of a registered procedure
    rather than a use of it.

    What this function does add is a **cross-check**: the handoff carries its
    own ``correlation_eligible`` flag, computed independently by Workstream 2,
    and eligibility is re-derived here from the raw fields. A row where the two
    disagree is a hard failure. Two independent implementations of one
    registered rule agreeing is worth something; one implementation agreeing
    with itself is not.
    """

    missing_columns = [name for name in JOINED_TABLE_FIELDS if rows and name not in rows[0]]
    if missing_columns:
        raise SystemExit(
            f"The joined table is missing required columns {missing_columns}. Expected the "
            "schema scripts/data/prepare_c2_join.py writes (c2_joined_requirements.csv)."
        )

    pairs: list[JoinedPair] = []
    eligible_flagged = 0
    disagreements: list[str] = []
    for row in rows:
        n_req = _optional_float(row["n_req_hat"])
        n_emp = _optional_float(row["n_emp_transitions"])
        censored = _bool(row["n_emp_censored"])
        pairs.append(
            JoinedPair(
                mdp_id=row["mdp_id"],
                policy_family=row["policy_family"],
                algorithm=row["algorithm"],
                variant=row["variant"],
                materiality_fraction=_optional_float(row["materiality_fraction"]),
                n_req_hat=n_req,
                oeh_status=row["oeh_status"],
                n_emp_transitions=n_emp,
                n_emp_censored=censored,
                num_states=int(row["num_states"]),
                num_actions=int(row["num_actions"]),
            )
        )

        flagged = _bool(row["correlation_eligible"])
        derived = n_req is not None and np.isfinite(n_req) and not censored and n_emp is not None
        eligible_flagged += int(flagged)
        if flagged != derived:
            disagreements.append(
                f"{row['mdp_id']}/{row['policy_family']}/{row['algorithm']}/{row['variant']}: "
                f"handoff says correlation_eligible={flagged}, raw fields imply {derived}"
            )

    if disagreements:
        raise SystemExit(
            "The handoff's correlation_eligible flag disagrees with its own raw fields on "
            f"{len(disagreements)} row(s). Rule C2-2 clause 2 defines eligibility, and the two "
            "implementations must agree before any coefficient is computed.\n  "
            + "\n  ".join(disagreements[:5])
        )

    report = {
        "source": "scripts.data.prepare_c2_join.py handoff (ARM_REGISTRY.md rule C2-2)",
        "joined_rows": len(pairs),
        "correlation_eligible_rows": eligible_flagged,
        "eligibility_cross_check": "independently re-derived; agrees on every row",
        "label_rows": len({(p.mdp_id, p.policy_family, p.algorithm) for p in pairs}),
        "variant_keys": sorted({f"{p.variant}@{p.materiality_fraction}" for p in pairs}),
    }
    return pairs, report


# --------------------------------------------------------------------------
# Synthetic mode
# --------------------------------------------------------------------------

SYNTHETIC_VARIANTS: dict[str, str] = {
    "synthetic_signal": (
        "planted: the pilot-dependent multiplier tracks N_emp, so the per_sa cell must come out "
        "strongly positive. If it does not, the analysis is broken, not the data."
    ),
    "synthetic_structural_only": (
        "planted: n_req_hat = N_0 with a multiplier drawn independently of N_emp -- the sec 10b "
        "degeneracy. Raw scale inherits S*A and looks predictive; per_sa must collapse to noise."
    ),
    "synthetic_constant_multiplier": (
        "planted: min_k m_k == 1 and one fixed N_0 multiplier, so the per_sa predictor is a "
        "constant. The cell must report undefined with a reason, never 0.0."
    ),
}


def synthetic_pairs(
    *,
    num_mdps: int = 15,
    families: tuple[str, ...] = (
        "q_star_softmax",
        "perturbed_reward",
        "structured_random",
        "two_policy_mixture",
        "shifted_dynamics",
    ),
    algorithms: tuple[str, str] = ("cfqi_tabular", "iql_tabular"),
    censored_label_fraction: float = 0.2,
    infinite_n_req_fraction: float = 0.15,
    seed: int = ANALYSIS_SEED,
) -> list[JoinedPair]:
    """A 150-row-per-variant fixture shaped like the real join.

    Sizes, family names, algorithm names, censoring proportions and the
    ``{2,5,10}`` pilot-budget grid all mirror the development split, so a bug
    that only appears at the real table's shape has somewhere to appear here.
    Values are invented; nothing here is a result.
    """

    rng = np.random.default_rng(seed)
    sizes = [(int(s), int(a)) for s, a in zip(rng.integers(8, 40, num_mdps), rng.integers(2, 5, num_mdps))]
    pilot_grid = np.array(PROTOCOL_CONFIG["collection"]["pilot_size_multipliers"], dtype=float)

    pairs: list[JoinedPair] = []
    for index, (num_states, num_actions) in enumerate(sizes):
        mdp_id = f"synthetic_mdp_{index:03d}"
        sa = num_states * num_actions
        for family in families:
            # The label: a per-SA difficulty that varies by MDP and family.
            difficulty = float(np.exp(rng.normal(0.8, 0.6)))
            censored = bool(rng.random() < censored_label_fraction)
            n_emp = None if censored else float(min(50.0, difficulty) * sa)
            n0_multiplier = float(rng.choice(pilot_grid))
            for algorithm in algorithms:
                for variant in SYNTHETIC_VARIANTS:
                    if variant == "synthetic_signal":
                        multiplier = difficulty * float(np.exp(rng.normal(0.0, 0.25)))
                    elif variant == "synthetic_structural_only":
                        multiplier = 1.0
                    else:
                        multiplier = 1.0
                    budget = (2.0 if variant == "synthetic_constant_multiplier" else n0_multiplier) * sa
                    n_req = budget * multiplier
                    if rng.random() < infinite_n_req_fraction and variant != "synthetic_constant_multiplier":
                        n_req = float("inf")
                    pairs.append(
                        JoinedPair(
                            mdp_id=mdp_id,
                            policy_family=family,
                            algorithm=algorithm,
                            variant=variant,
                            materiality_fraction=0.01,
                            n_req_hat=n_req,
                            oeh_status="observed",
                            n_emp_transitions=n_emp,
                            n_emp_censored=censored,
                            num_states=num_states,
                            num_actions=num_actions,
                        )
                    )
    return pairs


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover - environment
        return "unknown"


def _provenance(inputs: dict[str, Path], *, censoring_rule: str, synthetic: bool) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "protocol_config_sha256": sha256_file(PROTOCOL_CONFIG_PATH),
        "analysis_seed": ANALYSIS_SEED,
        "censoring_rule": censoring_rule,
        "censoring_rule_description": CENSORING_RULES[censoring_rule],
        "synthetic": synthetic,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true", help="Run on generated data; no registry gate.")
    parser.add_argument(
        "--joined-table",
        type=Path,
        help="c2_joined_requirements.csv from scripts/data/prepare_c2_join.py (the registered C2-2 handoff).",
    )
    parser.add_argument("--alternative-definitions", type=Path)
    parser.add_argument("--empirical-sample-complexity", type=Path)
    parser.add_argument("--benchmark-metadata", type=Path)
    parser.add_argument("--censoring-rule", choices=sorted(CENSORING_RULES), default=None)
    parser.add_argument("--num-resamples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    args = parser.parse_args(argv)

    inputs: dict[str, Path] = {}
    join_report: dict[str, Any]

    if args.synthetic:
        censoring_rule = args.censoring_rule or DEFAULT_CENSORING_RULE
        pairs = synthetic_pairs(seed=args.seed)
        join_report = {"synthetic": True, "joined_pairs": len(pairs), "variants": list(SYNTHETIC_VARIANTS)}
    elif args.joined_table is not None:
        if args.censoring_rule is None:
            parser.error("real mode needs --censoring-rule; it must match ARM_REGISTRY.md")
        censoring_rule = assert_rule_registered(args.censoring_rule)
        inputs = {"joined_table": args.joined_table}
        pairs, join_report = read_joined_table(read_csv_rows(args.joined_table))
        if not pairs:
            raise SystemExit("The joined table is empty.")
    else:
        required = {
            "alternative_definitions": args.alternative_definitions,
            "empirical_sample_complexity": args.empirical_sample_complexity,
            "benchmark_metadata": args.benchmark_metadata,
        }
        missing = [name for name, path in required.items() if path is None]
        if missing:
            parser.error(
                "real mode needs --joined-table, or all of --"
                + "/--".join(name.replace("_", "-") for name in missing)
            )
        if args.censoring_rule is None:
            parser.error("real mode needs --censoring-rule; it must match ARM_REGISTRY.md")
        censoring_rule = assert_rule_registered(args.censoring_rule)
        inputs = {name: path for name, path in required.items() if path is not None}
        pairs, join_report = join_pairs(
            read_csv_rows(inputs["alternative_definitions"]),
            read_csv_rows(inputs["empirical_sample_complexity"]),
            read_csv_rows(inputs["benchmark_metadata"]),
        )
        if not pairs:
            raise SystemExit("The join produced no rows. Check the key columns before rerunning.")

    rng = np.random.default_rng(args.seed)
    rows = run_c2_analysis(
        pairs,
        censoring_rule=censoring_rule,
        rng=rng,
        censored_n_emp_lower_bound_multiplier=float(max(PROTOCOL_CONFIG["collection"]["size_multipliers"])),
        num_resamples=args.num_resamples,
        confidence=args.confidence,
    )
    verdicts = beats_structural_baseline(rows)
    head_to_head = run_baseline_comparison(
        pairs,
        censoring_rule=censoring_rule,
        rng=np.random.default_rng(args.seed),
        censored_n_emp_lower_bound_multiplier=float(max(PROTOCOL_CONFIG["collection"]["size_multipliers"])),
        num_resamples=args.num_resamples,
        confidence=args.confidence,
    )

    robustness = run_family_robustness(
        pairs,
        censoring_rule=censoring_rule,
        rng=np.random.default_rng(args.seed),
        censored_n_emp_lower_bound_multiplier=float(max(PROTOCOL_CONFIG["collection"]["size_multipliers"])),
        num_resamples=args.num_resamples,
        confidence=args.confidence,
    )
    criteria = success_criteria_3_and_4(robustness)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "c2_correlations.csv", rows, OUTPUT_FIELDS)
    write_csv_rows(output_dir / "c2_verdicts.csv", verdicts, VERDICT_FIELDS)
    write_csv_rows(
        output_dir / "c2_baseline_comparison.csv", head_to_head, BASELINE_COMPARISON_FIELDS
    )
    write_csv_rows(
        output_dir / "c2_family_robustness.csv", robustness, FAMILY_ROBUSTNESS_FIELDS
    )
    write_json(
        output_dir / "c2_decision_inputs.json",
        {
            "scope": "development-only; C2 head-to-head against the structural-only baseline",
            "synthetic": args.synthetic,
            "warning": (
                "SYNTHETIC -- generated data, not a decision input. Do not feed to the gate."
                if args.synthetic
                else None
            ),
            "generated": datetime.now(timezone.utc).date().isoformat(),
            "censoring_rule": censoring_rule,
            "analysis_seed": args.seed,
            "criteria": criteria,
            "note": (
                "These two fields were null in artifacts/week2/decision_gate_input.json "
                "(2026-08-24) because no baseline comparison had been run on real data. They "
                "are now measured. A False here is a measured failure, not an absent input."
            ),
        },
    )
    write_json(
        output_dir / "c2_correlation_report.json",
        {
            "provenance": _provenance(inputs, censoring_rule=censoring_rule, synthetic=args.synthetic),
            "join": join_report,
            "num_cells": len(rows),
            "bootstrap": {"num_resamples": args.num_resamples, "confidence": args.confidence},
            "multiplicity": multiplicity_report(rows),
            "success_criteria": criteria,
            "synthetic_variant_expectations": SYNTHETIC_VARIANTS if args.synthetic else None,
        },
    )

    label = "SYNTHETIC DRY RUN -- not a result" if args.synthetic else "real inputs"
    print(f"[{label}] {len(rows)} cells over {len(pairs)} joined pairs -> {output_dir}")
    for verdict in verdicts:
        print(
            f"  {verdict['variant']:<32} per_sa spearman = {verdict['coefficient']:+.3f} "
            f"[{verdict['ci_low']:+.3f}, {verdict['ci_high']:+.3f}] "
            f"n={verdict['n_pairs']} clusters={verdict['num_clusters']}"
            + (f"  ({verdict['undefined_reason']})" if verdict["undefined_reason"] else "")
        )
    per_sa = multiplicity_report(rows)["by_scale"]["per_sa"]
    print(
        f"  -- per_sa: {per_sa['cells_whose_ci_excludes_zero']} of "
        f"{per_sa['cells_with_a_defined_coefficient']} defined cells exclude zero "
        f"(~{per_sa['expected_by_chance_at_95_percent']} expected by chance, uncorrected)"
    )
    print(
        f"  -- criterion 3 (beats baseline, Holm): {criteria['beats_strongest_baseline_ci_excludes_zero']}"
        f"  |  criterion 4 (non-Q* families): {criteria['holds_on_non_q_star_families']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

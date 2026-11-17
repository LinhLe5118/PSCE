#!/usr/bin/env python
"""Can a difficulty statistic work on this dataset? Decide before paying for it.

This is the practitioner-facing entry point of the thesis: the piece that makes
the pipeline difficulty-**aware** rather than merely difficulty-estimating. Given
a pilot buffer, it reports whether a resolution-failure statistic of the judged
admissibility class can be non-degenerate on that data at all, why not when it
cannot, and what would have to change.

The theory it executes is ``docs/DERIVATIONS.md``:

* **Theorem 1** bounds the registered loss by ``0.169971 * sigma_bar``, so
  escaping saturation *requires* ``sigma_bar >= epsilon / 0.169971 = 0.2942``.
* **Corollary 2.2** extends that to the whole degree-one class.
* **Section 4.3** gives a stronger, instance-specific statement: when the exact
  ceiling ``max_k (1/2) sum_s w_k Delta_k`` falls below ``epsilon``, no amount of
  dispersion suffices. That is a proof, not a bound, and it outranks the
  ``sigma_bar`` condition in the verdict.

Both sides of every one of those inequalities are computable from the pilot
alone -- no oracle, no training run, no label.

Usage::

    python -m scripts.analysis.run_feasibility_check --buffer-dir handover/pilots --output-dir artifacts/feasibility_<date>
    python -m scripts.analysis.run_feasibility_check --reference          # built-in demo instance, no files needed

Buffers are ``.npz`` in the ``transitions_from_arrays`` schema, exactly the
handover format ``scripts/validation/run_positive_control.py`` documents: the eight transition
fields plus ``num_states``, ``num_actions``, ``initial_state_distribution``,
either in the same file or in a JSON sidecar of the same stem.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

import numpy as np  # noqa: E402

from adaptive_pilot import transitions_from_arrays  # noqa: E402
from feasibility import (  # noqa: E402
    REPORT_FIELDS,
    FeasibilityReport,
    assess,
    assess_from_record,
)
from oeh_estimator import OEHConfig, estimate_oeh  # noqa: E402
from positive_control import (  # noqa: E402
    assert_registered,
    load_candidate,
    reference_control_buffer,
)

from evaluation.io import read_csv_rows, sha256_file, write_csv_rows, write_json  # noqa: E402

DEFAULT_SEED = 20260902


def check_buffer(
    arrays: Any,
    initial_state_distribution: Any,
    *,
    num_states: int,
    num_actions: int,
    label: str,
    seed: int,
    definition: str = "registered",
    config: OEHConfig | None = None,
) -> FeasibilityReport:
    """Run the registered estimator on one buffer and judge feasibility.

    ``assert_registered`` is applied for the same reason the positive control
    applies it: a feasibility verdict computed under a tweaked ``epsilon`` or
    ``n_min`` describes the tweak, not the dataset. ``epsilon`` matters most --
    ``m_k = min{mu : L_k(mu) <= epsilon}``, so lowering it manufactures
    feasibility.
    """

    cfg = assert_registered(config or OEHConfig.load())
    result = estimate_oeh(
        transitions_from_arrays(arrays),
        np.asarray(initial_state_distribution, dtype=np.float64),
        num_states=int(num_states),
        num_actions=int(num_actions),
        config=cfg,
        rng=np.random.default_rng(seed),
    )
    return assess(result, epsilon=cfg.epsilon, definition=definition, label=label)


def from_audit(
    audit_path: Path, definitions_path: Path | None, *, definition: str
) -> list[FeasibilityReport]:
    """Judge every pilot in a committed audit artifact, running nothing.

    DEMO_PLAN.md sec 3 forbids training on stage, and re-estimating 825 buffers
    is hours of compute for numbers already on disk. ``sigma_bar`` is joined in
    from the alternative-definitions artifact because the audit file does not
    carry it; without that join the Theorem 1 tier is simply unavailable and the
    verdict falls back to the ceiling, which is stated rather than silently
    downgraded.
    """

    sigma_by_key: dict[tuple[str, str], float] = {}
    if definitions_path is not None:
        for row in read_csv_rows(definitions_path):
            if row.get("variant") != "registered":
                continue
            value = row.get("sigma_weighted_mean", "")
            if str(value).strip():
                sigma_by_key[(row["mdp_id"], row["policy_family"])] = float(value)

    reports: list[FeasibilityReport] = []
    for row in read_csv_rows(audit_path):
        key = (row["mdp_id"], row["policy_family"])
        reports.append(
            assess_from_record(
                row, definition=definition, sigma_bar_value=sigma_by_key.get(key)
            )
        )
    return reports


def _reference(seed: int, definition: str) -> FeasibilityReport:
    """The built-in instance, so the tool demonstrates a ``feasible`` verdict.

    A checker that only ever says "no" is indistinguishable from a broken
    checker, which is the same argument the positive control rests on.
    """

    arrays, rho0, num_states, num_actions = reference_control_buffer()
    return check_buffer(
        arrays,
        rho0,
        num_states=num_states,
        num_actions=num_actions,
        label="reference_instance",
        seed=seed,
        definition=definition,
    )


def _print(report: FeasibilityReport) -> None:
    mark = {
        "feasible": "FEASIBLE",
        "underpowered": "UNDERPOWERED",
        "blocked_by_theorem_1": "BLOCKED",
        "impossible_at_any_dispersion": "IMPOSSIBLE",
        "no_curve": "NO CURVE",
    }[report.verdict]
    print(f"\n  {report.label}  [{mark}]  definition={report.definition}")
    if report.sigma_bar is not None:
        print(
            f"    sigma_bar        {report.sigma_bar:.5g}"
            f"   (need {report.required_sigma_bar:.5g}"
            f", have {report.sigma_headroom:.3g}x)"
        )
    if report.exact_ceiling is not None:
        print(
            f"    exact ceiling    {report.exact_ceiling:.5g}"
            f"   (need {report.epsilon:g}"
            f", have {report.ceiling_headroom:.3g}x)"
        )
    if report.dispersion_multiplier_needed is not None:
        print(f"    sigma inflation  {report.dispersion_multiplier_needed:.4g}x would suffice")
    elif report.verdict not in {"feasible", "no_curve"}:
        print("    sigma inflation  no finite factor suffices")
    print(f"    mechanism        {report.mechanism}")
    print(f"    {report.explanation}")
    if report.remedy:
        print(f"    -> {report.remedy}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--buffer-dir", type=Path, help="Directory of .npz pilot buffers.")
    parser.add_argument(
        "--audit",
        type=Path,
        help="audit_loss_ceiling.csv from a committed saturation audit; judges every pilot in it "
        "without re-running the estimator.",
    )
    parser.add_argument(
        "--definitions",
        type=Path,
        help="alternative_definitions.csv, joined for sigma_weighted_mean when using --audit.",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Also judge the built-in instance, which is known to be feasible.",
    )
    parser.add_argument("--definition", default="registered", help="Which statistic to judge.")
    parser.add_argument("--output-dir", type=Path, help="Write CSV + JSON here.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    if args.buffer_dir is None and args.audit is None and not args.reference:
        parser.error(
            "give --audit (a committed artifact), --buffer-dir (.npz pilots), "
            "or --reference (the built-in instance)"
        )

    reports: list[FeasibilityReport] = []
    inputs: dict[str, str] = {}

    if args.reference:
        reports.append(_reference(args.seed, args.definition))

    if args.audit is not None:
        inputs[args.audit.name] = sha256_file(args.audit)
        if args.definitions is not None:
            inputs[args.definitions.name] = sha256_file(args.definitions)
        reports.extend(from_audit(args.audit, args.definitions, definition=args.definition))

    if args.buffer_dir is not None:
        paths = sorted(args.buffer_dir.glob("*.npz"))
        if not paths:
            raise SystemExit(f"{args.buffer_dir} contains no .npz buffers.")
        for path in paths:
            candidate = load_candidate(path)
            inputs[path.name] = sha256_file(path)
            reports.append(
                check_buffer(
                    candidate["arrays"],
                    candidate["initial_state_distribution"],
                    num_states=candidate["num_states"],
                    num_actions=candidate["num_actions"],
                    label=path.stem,
                    seed=args.seed,
                    definition=args.definition,
                )
            )

    if len(reports) <= 8:
        for report in reports:
            _print(report)
    else:
        by_mechanism: dict[str, int] = {}
        for report in reports:
            by_mechanism[report.mechanism] = by_mechanism.get(report.mechanism, 0) + 1
        print(f"\n  {len(reports)} pilots judged. Blocking mechanism:")
        for mechanism, n in sorted(by_mechanism.items(), key=lambda kv: -kv[1]):
            print(f"    {mechanism:<24}{n:>4}")

    counts: dict[str, int] = {}
    for report in reports:
        counts[report.verdict] = counts.get(report.verdict, 0) + 1
    print("\n  " + ", ".join(f"{verdict}: {n}" for verdict, n in sorted(counts.items())))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rows = [report.as_row() for report in reports]
        write_csv_rows(args.output_dir / "feasibility.csv", rows, REPORT_FIELDS)
        write_json(
            args.output_dir / "feasibility_report.json",
            {
                "generated": datetime.now(timezone.utc).isoformat(),
                "seed": args.seed,
                "definition": args.definition,
                "criterion": {
                    "theorem_1": "L_k(1) <= 0.169971 * sigma_bar, so escape needs "
                    "sigma_bar >= epsilon / 0.169971",
                    "exact_ceiling": "max_k (1/2) sum_s w_k(s) Delta_k(s) < epsilon proves "
                    "impossibility at any dispersion (DERIVATIONS.md sec 4.3)",
                },
                "verdict_counts": counts,
                "inputs_sha256": inputs,
                "caveat": "Feasible is not informative. Whether an admissible statistic predicts "
                "the empirical requirement better than |S||A| is a separate, measured question; "
                "on this benchmark it did not (artifacts/week2/c2_correlation_2026-09-02/).",
            },
        )
        print(f"\n  -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

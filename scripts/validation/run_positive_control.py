"""Verify positive-control candidates with the registered OEH estimator.

Takes the buffers Workstream 1 hands over and answers one question per
candidate: does the **unmodified** registered estimator return ``m > 1`` at an
interior depth?  Until at least one does, every negative result in the Week 2
record is ambiguous between "the metric does not work" and "our code does not
work", and that ambiguity is the first question a committee asks.

A failure is reported with the condition that failed and the distance to
Theorem 1's necessary bound ``sigma_bar >= epsilon / 0.16997 = 0.2942``, so the
instance builder gets a direction rather than a verdict.

The built-in reference control (``--include-reference``, on by default) is run
alongside the handover.  It is not a benchmark instance -- it exists to show
the harness can return a pass at all, so a run where every candidate fails is
still informative.

Usage::

    python -m scripts.validation.run_positive_control \\
        --candidate-dir handover/positive_control \\
        --output-dir artifacts/week2/positive_control_<date>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from evaluation.io import sha256_file, write_csv_rows, write_json  # noqa: E402
from oeh_estimator import OEHConfig  # noqa: E402
from positive_control import (  # noqa: E402
    REFERENCE_CONTROL,
    VERDICT_FIELDS,
    PositiveControlVerdict,
    load_candidate,
    reference_control_buffer,
    required_sigma_bar,
    verdict_row,
    verify,
)


def _reference(seed: int) -> PositiveControlVerdict:
    arrays, initial, states, actions = reference_control_buffer()
    return verify(
        arrays,
        initial,
        num_states=states,
        num_actions=actions,
        label="reference_control",
        seed=seed,
    )


def run(
    candidate_dir: Path | None,
    *,
    seed: int,
    include_reference: bool,
) -> tuple[list[PositiveControlVerdict], dict[str, Any]]:
    config = OEHConfig.load()
    started = time.time()
    verdicts: list[PositiveControlVerdict] = []
    if include_reference:
        verdicts.append(_reference(int(REFERENCE_CONTROL["estimator_seed"])))

    candidates = (
        sorted(candidate_dir.glob("*.npz")) if candidate_dir is not None else []
    )
    if candidate_dir is not None and not candidates:
        raise SystemExit(f"{candidate_dir} contains no .npz candidate buffers.")
    for path in candidates:
        candidate = load_candidate(path)
        verdicts.append(
            verify(
                candidate["arrays"],
                candidate["initial_state_distribution"],
                num_states=candidate["num_states"],
                num_actions=candidate["num_actions"],
                label=path.stem,
                seed=seed,
            )
        )

    handover = [item for item in verdicts if item.label != "reference_control"]
    report = {
        "status": "positive control for the registered OEH estimator",
        "estimator": "registered, unmodified (protocol_config.json)",
        "epsilon": config.epsilon,
        "k_max": config.k_max,
        "required_sigma_bar": required_sigma_bar(config.epsilon),
        "required_sigma_bar_note": (
            "Theorem 1: L_k(1) <= 0.16997 * sum_s w_k(s) sigma_k(s), so an "
            "instance below this cannot reach epsilon at any depth whatever "
            "its action gaps are."
        ),
        "criterion": "max_k m_k > 1 AND 0 < k_dagger < K_max AND max_k L_k(1) >= epsilon",
        "seed": seed,
        "candidates": len(handover),
        "candidates_passed": sum(1 for item in handover if item.passed),
        "reference_included": include_reference,
        "reference_passed": next(
            (item.passed for item in verdicts if item.label == "reference_control"),
            None,
        ),
        "runtime_seconds": round(time.time() - started, 1),
        "verdicts": [verdict_row(item) for item in verdicts],
    }
    return verdicts, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        help="Directory of handover .npz buffers; omit to run the reference only.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip the built-in reference control.",
    )
    args = parser.parse_args()

    if args.output_dir is not None and args.output_dir.exists() and any(
        args.output_dir.iterdir()
    ):
        raise SystemExit(
            f"{args.output_dir} is not empty; refusing to overwrite an artifact directory."
        )

    verdicts, report = run(
        args.candidate_dir,
        seed=args.seed,
        include_reference=not args.no_reference,
    )

    for verdict in verdicts:
        mark = "PASS" if verdict.passed else "fail"
        headroom = (
            f"{verdict.sigma_headroom:.2f}x"
            if verdict.sigma_headroom is not None
            else "n/a"
        )
        print(
            f"[{mark}] {verdict.label}: status={verdict.oeh_status} "
            f"m={verdict.max_multiplier} k†={verdict.k_dagger} "
            f"sigma_bar={verdict.sigma_bar} ({headroom} of required)"
        )
        if not verdict.passed:
            print(f"       {verdict.diagnosis}")

    if args.output_dir is not None:
        rows = [verdict_row(item) for item in verdicts]
        path = args.output_dir / "positive_control.csv"
        write_csv_rows(path, rows, VERDICT_FIELDS)
        report["artifact_sha256"] = {"positive_control.csv": sha256_file(path)}
        write_json(args.output_dir / "positive_control_report.json", report)
        print(f"\nwrote {path.parent}")


if __name__ == "__main__":
    main()

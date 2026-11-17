"""Run the development-only Week 1 validation over an existing dataset bundle.

This script intentionally reads the estimator-facing buffers before it reads
oracle audit metadata.  It never materializes an evaluation instance and
records abstention/censoring rather than converting either state into a value.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from adaptive_pilot import select_effective_pilot_with_oeh  # noqa: E402
from environments.behavior_policies import behavior_policy_suite  # noqa: E402
from exact_evidence_curve import exact_evidence_curve  # noqa: E402
from oeh_estimator import OEHConfig  # noqa: E402
from environments.benchmark_environments import make_benchmark_mdp  # noqa: E402
from config.protocol_config import PROTOCOL_CONFIG  # noqa: E402


def _support_mask(result: Any) -> np.ndarray:
    mask = np.zeros((result.num_states, result.num_actions), dtype=np.bool_)
    for state, actions in enumerate(result.supported_actions):
        mask[state, list(actions)] = True
    return mask


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def run(dataset_dir: Path, *, oracle_seed: int) -> dict[str, Any]:
    """Validate all pilot streams in a development-only generated bundle."""

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["run_mode"] != "review":
        raise ValueError("Week 1 validation requires an explicitly development-only review run.")
    records = [
        item
        for item in manifest["datasets"]
        if item["stream"] == "pilot" and item["budget_role"] == "adaptive_pilot_candidate"
        and item["size_multiplier"] == 10
    ]
    if len(records) != 75:
        raise ValueError(f"Expected 75 development policy pilot records, found {len(records)}.")
    if any(item["benchmark_split"] != "development" for item in records):
        raise ValueError("Validation bundle must not include evaluation records.")

    outcomes: list[dict[str, Any]] = []
    per_family: dict[str, Counter[str]] = defaultdict(Counter)
    coverage: list[float] = []
    policy_quality: dict[str, list[float]] = defaultdict(list)
    bootstrap_validity: list[float] = []
    selected_depth_error: list[float] = []
    oracle_defined = 0
    oracle_undefined: Counter[str] = Counter()

    for record in records:
        with np.load(dataset_dir / record["file"], allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        selection = select_effective_pilot_with_oeh(
            arrays,
            np.asarray(record["initial_state_distribution"], dtype=np.float64),
            num_states=int(record["num_states"]),
            num_actions=int(record["num_actions"]),
            bootstrap_seed=20260817,
        )
        result = selection.result
        state = "abstain" if result.abstained else "censor" if result.censored_right else "valid"
        per_family[record["policy_family"]][state] += 1
        coverage.append(float(record["coverage"]["observed_reachable_sa_fraction"]))
        policy_quality[record["policy_family"]].append(float(record["policy_normalized_return"]))
        if result.valid_bootstrap_fraction is not None:
            bootstrap_validity.append(float(result.valid_bootstrap_fraction))

        outcome: dict[str, Any] = {
            "instance_id": record["instance_id"],
            "family": record["policy_family"],
            "selection_status": selection.status,
            "outcome": state,
            "k_dagger": result.k_dagger,
            "valid_bootstrap_fraction": result.valid_bootstrap_fraction,
            "reason": result.reason,
            "coverage": record["coverage"],
            "policy_normalized_return": record["policy_normalized_return"],
        }
        # The oracle is invoked only after the data-only selection is complete,
        # and only for a selected, supported estimator result.
        if state == "valid":
            try:
                spec = next(item for item in manifest["instances"] if item["instance_id"] == record["instance_id"])
                mdp = make_benchmark_mdp(type("Spec", (), spec)())
                policy = next(
                    item
                    for item in behavior_policy_suite(
                        mdp,
                        seed=int(spec["seed"]) + 3_000_000,
                        variant_index=int(spec["ordinal"]),
                    )
                    if item.family == record["policy_family"]
                )
                curve = exact_evidence_curve(
                    mdp,
                    policy.probabilities,
                    action_mask=_support_mask(result),
                    config=OEHConfig.load(),
                    rng=np.random.default_rng([oracle_seed, int(record["mdp_seed"])]),
                )
                outcome["oracle_k_dagger"] = curve.k_dagger
                outcome["oracle_valid_sample_fraction"] = curve.valid_sample_fraction
                selected_depth_error.append(abs(int(result.k_dagger) - curve.k_dagger))
                oracle_defined += 1
            except (RuntimeError, ValueError) as error:
                oracle_undefined[type(error).__name__ + ": " + str(error)] += 1
        outcomes.append(outcome)

    result = {
        "scope": "development-only: 15 instances x 5 behavior-policy families",
        "protocol_status_at_run": manifest["protocol_status"],
        "protocol_schema_version": PROTOCOL_CONFIG["schema_version"],
        "manifest_sha256": __import__("hashlib").sha256((dataset_dir / "manifest.json").read_bytes()).hexdigest(),
        "outcome_counts": dict(Counter(item["outcome"] for item in outcomes)),
        "selection_status_counts": dict(Counter(item["selection_status"] for item in outcomes)),
        "per_family_outcomes": {family: dict(counts) for family, counts in per_family.items()},
        "coverage_observed_reachable_sa_fraction": _summary(coverage),
        "policy_normalized_return": {family: _summary(values) for family, values in policy_quality.items()},
        "bootstrap_valid_fraction": _summary(bootstrap_validity),
        "oracle_comparison": {
            "defined": oracle_defined,
            "undefined": dict(oracle_undefined),
            "absolute_k_dagger_error": _summary(selected_depth_error),
        },
        "outcomes": outcomes,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle-seed", type=int, default=20260817)
    args = parser.parse_args()
    summary = run(args.dataset_dir, oracle_seed=args.oracle_seed)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "outcomes"}, indent=2))


if __name__ == "__main__":
    main()

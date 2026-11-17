"""Reproduce the Week-8 data/label/hash audit from committed compact artifacts.

This intentionally does not claim to audit omitted raw buffers.  Missing files are
reported explicitly so that a public-clone audit cannot be confused with a full
end-to-end regeneration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "oeh_confirmatory_unseen_120_2026-10-04"
BUDGETS = (1, 2, 5, 10, 20, 50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def check_hashes() -> dict:
    protocol = load_json(ARTIFACT / "protocol.json")
    preoutcome = load_json(ARTIFACT / "preoutcome_seal.json")
    result = load_json(ARTIFACT / "result_manifest.json")
    expected_protocol = (ARTIFACT / "protocol.sha256").read_text(encoding="ascii").strip()

    entries: list[tuple[str, Path, str]] = [
        ("protocol.json", ARTIFACT / "protocol.json", expected_protocol)
    ]
    entries.extend(
        (name, ARTIFACT / name, expected)
        for name, expected in preoutcome["artifact_sha256"].items()
    )
    entries.extend(
        (name, ARTIFACT / name, expected)
        for name, expected in result["artifact_sha256"].items()
    )
    entries.extend(
        (name, ROOT / Path(name), expected)
        for name, expected in protocol["source_sha256"].items()
    )

    matched, mismatched, missing = [], [], []
    for name, path, expected in entries:
        if not path.is_file():
            missing.append(name)
        elif sha256(path) == expected:
            matched.append(name)
        else:
            mismatched.append(name)

    dataset_seal = load_json(ARTIFACT / "dataset_seal.json")
    omitted_dataset_files = [
        name
        for name in dataset_seal["sha256"]
        if not (ROOT / Path(name)).is_file()
    ]
    return {
        "checked_entries": len(entries),
        "matched": len(matched),
        "mismatched": mismatched,
        "missing_versioned_inputs": missing,
        "omitted_dataset_seal_targets": omitted_dataset_files,
    }


def expected_event(successes: dict[int, int]) -> int:
    for index, budget in enumerate(BUDGETS):
        if all(successes[later] >= 8 for later in BUDGETS[index:]):
            return index
    return 6


def check_labels() -> dict:
    outcomes = load_csv(ARTIFACT / "all_core_budget_seed_outcomes.csv")
    labels = load_csv(ARTIFACT / "labels.csv")
    jobs = {
        (
            row["mdp_id"],
            row["policy_family"],
            row["algorithm"],
            int(row["budget"]),
            int(row["training_seed"]),
        )
        for row in outcomes
    }
    threshold_mismatches = sum(
        int(row["success"]) != (float(row["normalized_return"]) >= 0.95)
        for row in outcomes
    )

    grouped: dict[tuple[str, str, str], Counter[int]] = defaultdict(Counter)
    for row in outcomes:
        key = (row["mdp_id"], row["policy_family"], row["algorithm"])
        grouped[key][int(row["budget"])] += int(row["success"])

    count_mismatches = 0
    event_mismatches = 0
    event_counts: Counter[int] = Counter()
    panel_counts: dict[str, Counter[str]] = defaultdict(Counter)
    learner_primary: dict[str, Counter[str]] = defaultdict(Counter)
    behavior_primary: dict[str, Counter[str]] = defaultdict(Counter)
    for row in labels:
        key = (row["mdp_id"], row["policy_family"], row["algorithm"])
        counts = grouped[key]
        count_mismatches += sum(
            int(row[f"successes_x{budget}"]) != counts[budget]
            for budget in BUDGETS
        )
        expected = expected_event(counts)
        event = int(row["event"])
        event_mismatches += event != expected
        event_counts[event] += 1
        panel_counts[row["panel"]]["total"] += 1
        panel_counts[row["panel"]]["censored"] += event == 6
        if row["panel"] == "primary":
            learner_primary[row["algorithm"]]["total"] += 1
            learner_primary[row["algorithm"]]["censored"] += event == 6
            behavior_primary[row["policy_family"]]["total"] += 1
            behavior_primary[row["policy_family"]]["censored"] += event == 6

    kernel_audit = load_json(ARTIFACT / "actual_kernel_audit_preoutcome.json")
    fingerprints = list(kernel_audit["actual_kernel_fingerprints"].values())
    return {
        "outcome_rows": len(outcomes),
        "unique_jobs": len(jobs),
        "duplicate_jobs": len(outcomes) - len(jobs),
        "threshold_mismatches": threshold_mismatches,
        "label_rows": len(labels),
        "success_count_mismatches": count_mismatches,
        "event_mismatches": event_mismatches,
        "event_counts": dict(sorted(event_counts.items())),
        "panel_censoring": panel_counts,
        "primary_learner_censoring": learner_primary,
        "primary_behavior_censoring": behavior_primary,
        "kernel_fingerprints": len(fingerprints),
        "unique_kernel_fingerprints": len(set(fingerprints)),
        "audit_says_all_new": kernel_audit["all_120_unique_and_new"],
        "resolved_seed_changes": kernel_audit["changed_requested_seed_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-available",
        action="store_true",
        help="fail on a mismatch in available compact artifacts (omitted inputs remain reported)",
    )
    args = parser.parse_args()
    report = {"hash_audit": check_hashes(), "label_audit": check_labels()}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict_available:
        hashes = report["hash_audit"]
        labels = report["label_audit"]
        failed = bool(hashes["mismatched"]) or any(
            labels[name]
            for name in (
                "duplicate_jobs",
                "threshold_mismatches",
                "success_count_mismatches",
                "event_mismatches",
            )
        )
        failed |= labels["kernel_fingerprints"] != labels["unique_kernel_fingerprints"]
        raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

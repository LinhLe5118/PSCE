"""Audit the completed OEH difficulty confirmation against the later 5% rule.

This does not turn tabular adaptations into original published methods, and
never authorizes an E1 run. It records numerical and literature-scope gates
separately after the pre-outcome-sealed analysis is complete.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.analysis.compare_difficulty import sha256
from scripts.analysis.confirm_oeh_difficulty import OUTPUT, BASELINES

METRICS = ("integrated_brier", "seven_class_log_loss")
MIN_GAIN = 0.05


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    result_path = OUTPUT / "result_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result.get("status") != "complete" or result.get("jobs") != 54000
            or result.get("mdps") != 60):
        raise RuntimeError("Confirmation result is incomplete")
    for name, digest in result["artifact_sha256"].items():
        if sha256(OUTPUT / name) != digest:
            raise RuntimeError(f"Confirmation artifact changed: {name}")
    amendment_path = OUTPUT / "acceptance_amendment_2026-10-01.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if (amendment["original_protocol_sha256"] != result["protocol_sha256"]
            or amendment["minimum_relative_improvement"] != MIN_GAIN
            or set(amendment["scores"]) != set(METRICS)):
        raise RuntimeError("Acceptance amendment differs from result")
    summary = rows(OUTPUT / "summary.csv")
    primary = {row["method"]: row for row in summary if row["panel"] == "primary"}
    if set(primary) != {*BASELINES, "b2_plus_oeh", "oeh_blend"}:
        raise RuntimeError("Primary method set is incomplete")
    contrasts = rows(OUTPUT / "contrasts.csv")
    expected = {(metric, baseline) for metric in METRICS for baseline in BASELINES}
    found = {(row["metric"], row["baseline"]) for row in contrasts}
    if found != expected or len(contrasts) != len(expected):
        raise RuntimeError("Incomplete or duplicate registered contrasts")
    metrics = {}
    for metric in METRICS:
        baseline = min(BASELINES, key=lambda name: float(primary[name][metric]))
        reference = float(primary[baseline][metric])
        candidate = float(primary["oeh_blend"][metric])
        if reference <= 0:
            raise RuntimeError("Relative gain requires positive baseline score")
        gains = [row for row in contrasts if row["metric"] == metric]
        minimum_lower = min(float(row["simultaneous_one_sided_lower_95"])
                            for row in gains)
        metrics[metric] = {
            "strongest_registered_comparator": baseline,
            "comparator_score": reference,
            "oeh_score": candidate,
            "relative_gain": (reference - candidate) / reference,
            "minimum_simultaneous_lower_bound_across_five_comparators": minimum_lower,
            "point_gain_at_least_5_percent": (reference - candidate) / reference >= MIN_GAIN,
            "all_five_simultaneous_lower_bounds_positive": minimum_lower > 0,
        }
    numerical_go = (result["difficulty_gate"] == "GO"
                    and all(item["point_gain_at_least_5_percent"]
                            and item["all_five_simultaneous_lower_bounds_positive"]
                            for item in metrics.values()))
    output = {
        "schema": "oeh-difficulty-later-5pct-audit-v1",
        "confirmation_result_sha256": sha256(result_path),
        "acceptance_amendment_sha256": sha256(amendment_path),
        "confirmation_original_gate": result["difficulty_gate"],
        "numerical_5pct_gate": "GO" if numerical_go else "NO-GO",
        "metrics": metrics,
        "published_original_method_gate": "NOT_ESTABLISHED",
        "reason": "No original published method audited so far has the same pilot-only inputs and learner-specific seven-category endpoint; tabular adaptations are not originals.",
        "overall_user_criterion": "NOT_ESTABLISHED",
        "e1_comparison_authorized_by_this_audit": False,
    }
    path = OUTPUT / "margin_audit.json"
    if path.exists():
        raise FileExistsError(f"Refusing to replace prior audit: {path}")
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

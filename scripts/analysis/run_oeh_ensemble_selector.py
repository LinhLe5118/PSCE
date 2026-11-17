#!/usr/bin/env python
"""Evaluate one fixed equal-weight OEH selector ensemble on development OOF.

The two components are the generic OEH-v3 return model and its full
learner-conditional LC-OEH augmentation.  Equal weighting is fixed by
symmetry; no weight, margin, or feature subset is selected from outcomes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.aggregation.correlation import holm_adjust  # noqa: E402
from evaluation.io import (read_csv_rows, reserve_output_dir, sha256_file,
                           write_csv_rows, write_json)  # noqa: E402
from scripts.analysis.run_calibrated_comparison import (  # noqa: E402
    cluster_bootstrap_difference,
)
from scripts.analysis.screen_lcoeh_v4 import ALGORITHMS  # noqa: E402


COMPONENTS = ("B2_OEH3", "B2_OEH3_LCOEH4")
METHOD = "EQUAL_ENSEMBLE_B2_OEH3_FULL_LCOEH4"
BOOTSTRAP_DRAWS = 10_000


def _seed() -> int:
    return int.from_bytes(hashlib.sha256(
        b"20260920|equal-oeh-ensemble").digest()[:4], "little")


def _holm_family(previous, current):
    rows = []
    for raw in previous:
        row = dict(raw)
        for name in ("mean_selected_return", "mean_fixed_return",
                     "mean_oracle_return", "difference_vs_fixed",
                     "ci_low", "ci_high", "p_value"):
            row[name] = float(row[name])
        rows.append(row)
    rows.append(current)
    adjusted, significant = holm_adjust([row["p_value"] for row in rows])
    for row, p_holm, passes in zip(rows, adjusted, significant):
        row["p_holm_combined_family"] = p_holm
        row["positive_gate"] = bool(
            row["difference_vs_fixed"] > 0 and row["ci_low"] > 0
            and passes)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-screen-dir", type=Path, required=True)
    parser.add_argument("--ensemble-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    reserve_output_dir(args.output_dir)

    predictions_path = args.ensemble_input_dir / "return_oof_predictions.csv"
    selection_path = args.ensemble_input_dir / "selection_rows.csv"
    predictions = read_csv_rows(predictions_path)
    prediction = {
        (row["mdp_id"], row["policy_family"], row["algorithm"],
         row["feature_set"]): row for row in predictions
        if row["feature_set"] in COMPONENTS
    }
    base_rows = [row for row in read_csv_rows(selection_path)
                 if row["method"] == "B2_OEH3"]
    if len(base_rows) != 300:
        raise ValueError("expected the complete 300-group OEH development panel")

    rows = []
    for base in base_rows:
        group = (base["mdp_id"], base["policy_family"])
        scores = {}
        outcomes = {}
        for algorithm in ALGORITHMS:
            components = [prediction[group + (algorithm, name)]
                          for name in COMPONENTS]
            scores[algorithm] = float(np.mean([
                float(row["predicted_return"]) for row in components]))
            observed = {float(row["normalized_return"]) for row in components}
            if len(observed) != 1:
                raise ValueError("ensemble components disagree on outcome")
            outcomes[algorithm] = observed.pop()
        selected = max(ALGORITHMS,
                       key=lambda name: (scores[name], -ALGORITHMS.index(name)))
        fixed = base["fixed_algorithm"]
        oracle = max(ALGORITHMS,
                     key=lambda name: (outcomes[name], -ALGORITHMS.index(name)))
        rows.append({
            "method": METHOD, "mdp_id": group[0], "policy_family": group[1],
            "outer_fold": base["outer_fold"],
            "selected_algorithm": selected, "fixed_algorithm": fixed,
            "oracle_algorithm": oracle,
            "selected_return": outcomes[selected],
            "fixed_return": outcomes[fixed], "oracle_return": outcomes[oracle],
        })

    difference, low, high, p_value = cluster_bootstrap_difference(
        [row["mdp_id"] for row in rows],
        [row["selected_return"] for row in rows],
        [row["fixed_return"] for row in rows],
        num_resamples=BOOTSTRAP_DRAWS, confidence=.95, seed=_seed())
    current = {
        "method": METHOD,
        "mean_selected_return": float(np.mean(
            [row["selected_return"] for row in rows])),
        "mean_fixed_return": float(np.mean(
            [row["fixed_return"] for row in rows])),
        "mean_oracle_return": float(np.mean(
            [row["oracle_return"] for row in rows])),
        "difference_vs_fixed": difference, "ci_low": low,
        "ci_high": high, "p_value": p_value,
        "switches": sum(row["selected_algorithm"] != row["fixed_algorithm"]
                        for row in rows),
        "oracle_matches": sum(row["selected_algorithm"] == row["oracle_algorithm"]
                              for row in rows),
    }
    previous = read_csv_rows(
        args.previous_screen_dir / "combined_selector_family.csv")
    combined = _holm_family(previous, current)
    result = combined[-1]

    write_csv_rows(args.output_dir / "selection_rows.csv", rows, list(rows[0]))
    write_csv_rows(args.output_dir / "selection_summary.csv", [result],
                   list(result))
    write_csv_rows(args.output_dir / "combined_selector_family.csv", combined,
                   list(result))
    report = {
        "status": "complete_final_exploratory_selector_screen",
        "development_only": True, "consumed_v3_cohort": True,
        "confirmatory_claim_allowed": False,
        "new_development_dataset_generated": False,
        "sealed_evaluation_accessed": False,
        "method": METHOD, "components": list(COMPONENTS),
        "weights": [.5, .5], "weight_selection": "fixed_by_symmetry",
        "multiplicity_family_size": len(combined),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "gates": {
            "positive_interval": bool(result["ci_low"] > 0),
            "positive_after_combined_holm": result["positive_gate"],
            "independent_confirmation": False,
            "sota_claim_supported": False,
        },
        "inputs_sha256": {
            "return_predictions": sha256_file(predictions_path),
            "selection_rows": sha256_file(selection_path),
            "previous_family": sha256_file(
                args.previous_screen_dir / "combined_selector_family.csv"),
        },
        "source_sha256": sha256_file(
            ROOT / "scripts/analysis/run_oeh_ensemble_selector.py"),
        "selection_summary": result,
    }
    write_json(args.output_dir / "report.json", report)
    text = f"""# Fixed equal-weight OEH selector ensemble

This is the final exploratory selector screen on the consumed development
cohort. No new development data were generated and sealed evaluation was not
accessed.

| Return | Delta vs fixed | 95% CI | Raw p | Combined Holm p | Switches | Gate |
|---:|---:|---:|---:|---:|---:|---|
| {result['mean_selected_return']:.4f} | {result['difference_vs_fixed']:+.4f} | [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] | {result['p_value']:.4f} | {result['p_holm_combined_family']:.4f} | {result['switches']} | {result['positive_gate']} |

The family contains all {len(combined)} selectors examined in the LC-OEH,
paired-advantage, and ensemble screens. Equal component weights were not fit to
the return labels. This result remains adaptive development evidence.
"""
    (args.output_dir / "REPORT.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

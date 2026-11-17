#!/usr/bin/env python
"""Exploratory paired-advantage learner selector for the consumed OEH cohort.

The decision is whether an alternative learner improves return over the best
fixed learner at budget 10.  The model therefore predicts that paired return
difference directly instead of estimating three noisy absolute returns.  All
reference selection, ridge tuning, scaling, and conservative residual scores
are learned inside the relevant outer-training split.
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
sys.path.insert(0, str(ROOT / "code"))

from evaluation.aggregation.correlation import holm_adjust  # noqa: E402
from evaluation.io import (read_csv_rows, reserve_output_dir, sha256_file,
                           write_csv_rows, write_json)  # noqa: E402
from return_selection import (AdvantageInstance,
                              nested_advantage_oof)  # noqa: E402
from scripts.analysis.run_calibrated_comparison import (  # noqa: E402
    cluster_bootstrap_difference,
)
from scripts.analysis.screen_lcoeh_v4 import (  # noqa: E402
    ALGORITHMS, B2, OEH_V3_EXTRA_NAMES, PMS, RETURN_RIDGES, _outcomes,
)


FEATURE_SET = "PAIR_OEH3_LC_PMS_DIFF"
FEATURE_NAMES = B2 + OEH_V3_EXTRA_NAMES + tuple(
    f"delta_{name}" for name in PMS)
BOOTSTRAP_DRAWS = 10_000


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(
        f"20260920|paired-advantage|{label}".encode()).digest()[:4], "little")


def _instances(base, lc, outcomes):
    rows = []
    groups = sorted({key[:2] for key in outcomes})
    for mdp_id, family in groups:
        shared_row = base[(mdp_id, family)]
        shared = tuple(float(shared_row[name])
                       for name in B2 + OEH_V3_EXTRA_NAMES)
        for reference in ALGORITHMS:
            for alternative in ALGORITHMS:
                if alternative == reference:
                    continue
                ref_lc = lc[(mdp_id, family, reference)]
                alt_lc = lc[(mdp_id, family, alternative)]
                differences = tuple(float(alt_lc[name]) - float(ref_lc[name])
                                    for name in PMS)
                rows.append(AdvantageInstance(
                    mdp_id=mdp_id, policy_family=family,
                    reference_algorithm=reference,
                    alternative_algorithm=alternative,
                    advantage=(outcomes[(mdp_id, family, alternative)]
                               - outcomes[(mdp_id, family, reference)]),
                    features={FEATURE_SET: shared + differences},
                ))
    return rows


def _selection_rows(predictions, outcomes):
    by_group = defaultdict(list)
    for row in predictions:
        by_group[(row["mdp_id"], row["policy_family"])].append(row)
    rows = []
    for mode, score_name in (("POINT", "predicted_advantage"),
                             ("INNER_TUNED", "predicted_advantage"),
                             ("CONSERVATIVE", "conservative_advantage")):
        method = f"{FEATURE_SET}_{mode}"
        for group in sorted(by_group):
            panel = by_group[group]
            references = {row["reference_algorithm"] for row in panel}
            if len(references) != 1:
                raise ValueError("outer fold produced inconsistent references")
            reference = next(iter(references))
            best = max(panel, key=lambda row: (float(row[score_name]),
                                               row["alternative_algorithm"]))
            threshold = (float(best["decision_threshold"])
                         if mode == "INNER_TUNED" else 0.0)
            selected = (best["alternative_algorithm"]
                        if float(best[score_name]) > threshold else reference)
            oracle = max(ALGORITHMS,
                         key=lambda algorithm: outcomes[group + (algorithm,)])
            rows.append({
                "method": method,
                "mdp_id": group[0], "policy_family": group[1],
                "outer_fold": best["outer_fold"],
                "reference_algorithm": reference,
                "selected_algorithm": selected,
                "oracle_algorithm": oracle,
                "winning_predicted_advantage": best[score_name],
                "decision_threshold": threshold,
                "selected_return": outcomes[group + (selected,)],
                "fixed_return": outcomes[group + (reference,)],
                "oracle_return": outcomes[group + (oracle,)],
            })
    return rows


def _summaries(rows):
    output = []
    for method in sorted({row["method"] for row in rows}):
        panel = [row for row in rows if row["method"] == method]
        difference, low, high, p_value = cluster_bootstrap_difference(
            [row["mdp_id"] for row in panel],
            [row["selected_return"] for row in panel],
            [row["fixed_return"] for row in panel],
            num_resamples=BOOTSTRAP_DRAWS, confidence=.95,
            seed=_seed(method))
        output.append({
            "method": method,
            "mean_selected_return": float(np.mean(
                [row["selected_return"] for row in panel])),
            "mean_fixed_return": float(np.mean(
                [row["fixed_return"] for row in panel])),
            "mean_oracle_return": float(np.mean(
                [row["oracle_return"] for row in panel])),
            "difference_vs_fixed": difference,
            "ci_low": low, "ci_high": high, "p_value": p_value,
            "switches": sum(row["selected_algorithm"]
                            != row["reference_algorithm"] for row in panel),
            "oracle_matches": sum(row["selected_algorithm"]
                                  == row["oracle_algorithm"] for row in panel),
        })
    return output


def _combined_holm(previous, current):
    combined = []
    for source, rows in (("previous_screen", previous),
                         ("paired_screen", current)):
        for raw in rows:
            row = dict(raw)
            row["screen_source"] = source
            for name in ("mean_selected_return", "mean_fixed_return",
                         "mean_oracle_return", "difference_vs_fixed",
                         "ci_low", "ci_high", "p_value"):
                row[name] = float(row[name])
            combined.append(row)
    adjusted, significant = holm_adjust([row["p_value"] for row in combined])
    for row, p_holm, passes in zip(combined, adjusted, significant):
        row["p_holm_combined_family"] = p_holm
        row["positive_gate"] = bool(
            row["difference_vs_fixed"] > 0 and row["ci_low"] > 0
            and passes)
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-b3-dir", type=Path, required=True)
    parser.add_argument("--lcoeh-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--previous-screen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    reserve_output_dir(args.output_dir)

    base_rows = read_csv_rows(args.b0_b3_dir / "features.csv")
    lc_rows = read_csv_rows(args.lcoeh_dir / "features.csv")
    base = {(row["mdp_id"], row["policy_family"]): row for row in base_rows}
    lc = {(row["mdp_id"], row["policy_family"], row["algorithm"]): row
          for row in lc_rows}
    outcomes = _outcomes(args.local_dir)
    groups = {key[:2] for key in outcomes}
    if set(base) != groups or set(lc) != set(outcomes):
        raise ValueError("base, LC, and outcome panels are not aligned")

    instances = _instances(base, lc, outcomes)
    predictions = nested_advantage_oof(
        instances, feature_sets=(FEATURE_SET,), ridge_grid=RETURN_RIDGES,
        algorithms=ALGORITHMS, lower_quantile=.05)
    selection = _selection_rows(predictions, outcomes)
    summaries = _summaries(selection)
    previous = read_csv_rows(args.previous_screen_dir / "selection_summary.csv")
    combined = _combined_holm(previous, summaries)
    current_combined = [row for row in combined
                        if row["screen_source"] == "paired_screen"]

    write_csv_rows(args.output_dir / "advantage_oof_predictions.csv",
                   predictions, list(predictions[0]))
    write_csv_rows(args.output_dir / "selection_rows.csv",
                   selection, list(selection[0]))
    write_csv_rows(args.output_dir / "selection_summary.csv",
                   current_combined, list(current_combined[0]))
    write_csv_rows(args.output_dir / "combined_selector_family.csv",
                   combined, list(combined[0]))

    any_gate = any(row["positive_gate"] for row in current_combined)
    report = {
        "status": "complete_exploratory_paired_advantage_screen",
        "development_only": True,
        "consumed_v3_cohort": True,
        "confirmatory_claim_allowed": False,
        "new_development_dataset_generated": False,
        "sealed_evaluation_accessed": False,
        "feature_set": FEATURE_SET,
        "feature_names": list(FEATURE_NAMES),
        "decision_estimand": "alternative return minus best-fixed return at budget 10",
        "reference_rule": "best fixed mean return in each outer-training split",
        "switch_rule": "strictly positive outer-OOF predicted paired advantage",
        "conservative_rule": (
            "point prediction plus the training-MDP 5th percentile of the "
            "minimum inner-OOF residual across behavior policies"),
        "conservative_is_formal_conformal_bound": False,
        "ridge_grid": list(RETURN_RIDGES),
        "outer_folds": 5, "inner_folds": 4,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "multiplicity_family_size": len(combined),
        "gates": {
            "paired_selector_positive_after_combined_holm": any_gate,
            "independent_confirmation": False,
            "sota_claim_supported": False,
        },
        "inputs_sha256": {
            "base_features": sha256_file(args.b0_b3_dir / "features.csv"),
            "lcoeh_features": sha256_file(args.lcoeh_dir / "features.csv"),
            "rl_runs": sha256_file(args.local_dir / "rl_run_results.csv"),
            "bc_runs": sha256_file(args.local_dir / "bc_sanity_results.csv"),
            "previous_selector_summary": sha256_file(
                args.previous_screen_dir / "selection_summary.csv"),
        },
        "source_sha256": {
            name: sha256_file(ROOT / name) for name in (
                "code/return_selection.py",
                "scripts/analysis/run_paired_advantage_selector.py",
            )
        },
        "selection_summary": current_combined,
    }
    write_json(args.output_dir / "report.json", report)
    lines = [
        "# Paired-advantage OEH learner selector", "",
        "This is an exploratory screen on the already-consumed OEH-v3 cohort. "
        "No new development data were generated and the sealed evaluation set "
        "was not accessed.", "",
        "| Method | Return | Delta vs fixed | 95% CI | Raw p | Combined Holm p | Switches | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in current_combined:
        lines.append(
            f"| {row['method']} | {row['mean_selected_return']:.4f} | "
            f"{row['difference_vs_fixed']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{row['p_value']:.4f} | {row['p_holm_combined_family']:.4f} | "
            f"{row['switches']} | {row['positive_gate']} |")
    lines += ["", f"The Holm family contains all {len(combined)} learner "
              "selectors from the earlier LC-OEH screen and this paired screen. "
              "The conservative score is a training-only robustness rule, not "
              "a claimed finite-sample conformal interval.", ""]
    (args.output_dir / "REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export the sealed 120-MDP offline-difficulty comparison for the paper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.analysis.run_oeh_120_unseen import COMPARATORS, CANDIDATE, OUTPUT, ROOT, sha256


TARGET = ROOT / "tests/fixtures/paper_history/unseen_difficulty_table.tex"
NAMES = {
    "train_frequency": "Training frequency",
    "coverage_adaptation": "Coverage adaptation",
    "quality_adaptation": "Quality adaptation",
    "coverage_quality_adaptation": "Coverage + quality",
    "b2_generic": "Generic $B2$ AFT",
    "b2_three_way": "$B2$ three-part",
    "b2_plus_eh_oracle_three_way": "$B2$ + oracle EH",
    CANDIDATE: "\\OEH{} three-part",
}


def render() -> str:
    manifest = json.loads((OUTPUT / "result_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["mdps"] != 120 or manifest["stochastic_primary_mdps"] != 84:
        raise RuntimeError("Final evaluation is incomplete")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256(OUTPUT / name) != digest:
            raise RuntimeError(f"Final result hash mismatch: {name}")
    summary = pd.read_csv(OUTPUT / "summary.csv")
    primary = summary[summary.panel == "primary"].set_index("method")
    if set(primary.index) != {*COMPARATORS, CANDIDATE} or not (primary.mdps == 84).all():
        raise RuntimeError("Final primary comparator panel is incomplete")
    contrasts = pd.read_csv(OUTPUT / "contrasts.csv")
    if (len(contrasts) != 2 * len(COMPARATORS)
            or set(contrasts.baseline) != set(COMPARATORS)
            or set(contrasts.metric) != {"integrated_brier", "seven_class_log_loss"}):
        raise RuntimeError("Final joint contrasts are incomplete")
    gate = all((contrasts.relative_gain >= .05) & (contrasts.simultaneous_one_sided_lower_95 > 0))
    if manifest["difficulty_gate"] != ("GO" if gate else "NO-GO"):
        raise RuntimeError("Stored verdict differs from registered gate")
    lines = [
        "% Generated from the sealed 120-MDP evaluation; do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{New 84-stochastic-MDP offline difficulty evaluation. All three-part rows use matched architectures; oracle EH has true-MDP input. Lower is better.}",
        "\\label{tab:unseen-difficulty}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Forecast & Brier & Log loss \\\\",
        "\\midrule",
    ]
    for method in (*COMPARATORS, CANDIDATE):
        row = primary.loc[method]
        before = "\\textbf{" if method == CANDIDATE else ""
        after = "}" if method == CANDIDATE else ""
        lines.append(f"{NAMES[method]} & {before}{row.integrated_brier:.4f}{after} & "
                     f"{before}{row.seven_class_log_loss:.4f}{after} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = render()
    if args.check:
        if not TARGET.exists() or TARGET.read_text(encoding="utf-8") != output:
            raise RuntimeError("Paper table differs from sealed evaluation")
        print("Final offline-difficulty paper table verified")
    else:
        TARGET.write_text(output, encoding="utf-8")
        print("Wrote tests/fixtures/paper_history/unseen_difficulty_table.tex")


if __name__ == "__main__":
    main()

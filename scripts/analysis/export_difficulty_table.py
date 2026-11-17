"""Generate/check the paper difficulty table against its post-hoc artifact."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/difficulty_e1_posthoc_2026-09-30/comparison.csv"
DESTINATION = ROOT / "tests/fixtures/paper_history/difficulty_table.tex"
METHODS = (
    ("train_frequency", "Fold-local event frequency"),
    ("xgboost_aft_b2", "XGBoost-AFT, generic $B2$"),
    ("xgboost_aft_b2_oeh", r"XGBoost-AFT, $B2+$\OEH{}"),
    ("oeh3_raw_proxy", "OEH3 raw proxy"),
    ("lcoeh4_raw_proxy", "LCOEH4 raw proxy"),
)


def render(rows: list[dict[str, str]]) -> str:
    by_key = {(row["panel"], row["method"]): row for row in rows}
    expected = {(panel, method) for panel in ("primary", "control") for method, _ in METHODS}
    if len(rows) != len(expected) or set(by_key) != expected:
        raise ValueError("Incomplete or duplicated difficulty comparison")
    lines = [
        "% Generated from artifacts/difficulty_e1_posthoc_2026-09-30/comparison.csv.",
        r"\begin{table}[t]",
        r"\caption{Post-hoc E1 empirical-requirement prediction. Lower six-threshold Brier is better; raw proxies are uncalibrated point predictions. All cases, including $>x50$, are scored.}",
        r"\label{tab:difficulty}",
        r"\centering\footnotesize",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Predictor & Stoch. (84 MDPs) & Control (36) \\",
        r"\midrule",
    ]
    for method, name in METHODS:
        primary = float(by_key["primary", method]["integrated_brier"])
        control = float(by_key["control", method]["integrated_brier"])
        if not (0 <= primary <= 1 and 0 <= control <= 1):
            raise ValueError("Invalid Brier score")
        values = (f"{primary:.4f}", f"{control:.4f}")
        if method == "xgboost_aft_b2_oeh":
            values = tuple(r"\textbf{" + number + "}" for number in values)
        lines.append(name + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        rendered = render(list(csv.DictReader(handle)))
    if args.check:
        if not DESTINATION.is_file() or DESTINATION.read_text(encoding="utf-8") != rendered:
            raise SystemExit("Difficulty paper table differs from comparison artifact")
        print("Difficulty paper table matches comparison artifact")
    else:
        DESTINATION.write_text(rendered, encoding="utf-8")
        print("Wrote tests/fixtures/paper_history/difficulty_table.tex")


if __name__ == "__main__":
    main()

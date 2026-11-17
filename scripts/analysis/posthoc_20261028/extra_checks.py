"""Extra read-only checks for the post hoc run of 28/10 (added 29/10; changes none of the numbers of the four scripts).

1. The refitted "without planning risk" law equals the sealed oeh_without_horizon forecasts on all four panels of
   Tests 2 and 3 (score_gated_ablation.py checks only gambler and mdpp).
2. The log-loss floor: score_gated_ablation.py floors at 1e-15, the registered Eq. (5.1) at 1e-12. Counts the rows
   whose probability of the observed class is below 1e-12; if there are none, both floors give the same scores.
3. SHA-256 of every output table and, if the folder out/ of the original machine is given, every numeric cell that
   differs from it by more than 0.0005.

    python scripts/analysis/posthoc_20261028/extra_checks.py [path/to/original/out]
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "out"
P = [f"p_{i}" for i in range(7)]
KEYS = ["mdp_id", "policy_family", "algorithm"]


def main() -> None:
    pred = pd.read_csv(OUT / "gated_ablation" / "predictions.csv.gz")
    for test, folder, panels in (("Test 2", "oeh_fifth_panel_2026-10-15", ("frozenlake", "gambler")),
                                 ("Test 3", "oeh_panel6_2026-10-15", ("mdpp", "labmaze"))):
        for panel in panels:
            sealed = pd.read_csv(ROOT / "artifacts" / folder / panel / "predictions_preoutcome.csv")
            sealed = sealed[sealed.method == "oeh_without_horizon"]
            mine = pred[(pred.method == "no_H") & (pred.test == test) & (pred.source_panel == panel)]
            both = mine.merge(sealed[KEYS + P], on=KEYS, suffixes=("", "_s"), validate="one_to_one")
            diff = np.abs(both[P].to_numpy() - both[[p + "_s" for p in P]].to_numpy()).max()
            print(f"no_H vs sealed oeh_without_horizon, {test}/{panel}: {len(both)} rows, max |diff| = {diff:.2e}")

    scored = pd.read_csv(OUT / "gated_ablation" / "scored.csv.gz")
    p_obs = scored[P].to_numpy()[np.arange(len(scored)), scored.event.to_numpy()]
    small = scored[p_obs < 1e-12]
    print(f"rows with p(observed class) < 1e-12: {len(small)}"
          + ("" if small.empty else "\n" + small.groupby(["test", "method"]).size().to_string()))

    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.suffix in (".csv", ".gz")
                   and "work" not in p.relative_to(OUT).parts)
    print("SHA-256 of the outputs:")
    for path in files:
        print(f"  {hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(OUT).as_posix()}")

    if len(sys.argv) > 1:
        ref, worst = Path(sys.argv[1]), 0.0
        for path in files:
            other = ref / path.relative_to(OUT)
            if not other.exists():
                print(f"  {path.relative_to(OUT).as_posix()}: missing on the original machine")
                continue
            a, b = pd.read_csv(path), pd.read_csv(other)
            if a.shape != b.shape or list(a.columns) != list(b.columns):
                print(f"  {path.relative_to(OUT).as_posix()}: shape or columns differ, {a.shape} vs {b.shape}")
                continue
            num = list(a.select_dtypes("number").columns)
            diff = (a[num] - b[num]).abs()
            if num:
                worst = max(worst, float(np.nan_to_num(diff.to_numpy(), nan=0.0).max()))
            for col in num:
                for i in np.flatnonzero(diff[col].to_numpy() > 5e-4):
                    print(f"  {path.relative_to(OUT).as_posix()} row {i} {col}: {a[col].iat[i]} vs {b[col].iat[i]}")
            text_a, text_b = a.drop(columns=num).astype(str), b.drop(columns=num).astype(str)
            if (text_a != text_b).any().any():
                print(f"  {path.relative_to(OUT).as_posix()}: text columns differ")
        print(f"largest numeric difference to the original outputs: {worst:.2e}")


if __name__ == "__main__":
    main()

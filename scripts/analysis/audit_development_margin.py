"""Exploratory MDP-paired 5% margin audit for saved development forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
KEYS = ["mdp_id", "policy_family", "algorithm", "panel", "event"]


def audit(path: Path, baseline: str, candidate: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    raw = pd.read_csv(path)
    required = set(KEYS + ["method", "brier", "logloss"])
    if not required.issubset(raw):
        raise ValueError("Scored forecast columns missing")
    b2 = raw[(raw.method == baseline) & (raw.panel == "primary")]
    oeh = raw[(raw.method == candidate) & (raw.panel == "primary")]
    paired = b2[KEYS + ["brier", "logloss"]].merge(
        oeh[KEYS + ["brier", "logloss"]], on=KEYS,
        suffixes=("_baseline", "_candidate"), validate="one_to_one")
    if len(paired) != 630 or paired.mdp_id.nunique() != 42:
        raise ValueError("Expected 42 complete stochastic MDPs and 630 pairs")
    oracle_path = path.parent / "oracle_descriptors.csv"
    if oracle_path.exists():
        oracle = pd.read_csv(oracle_path)
        if "kernel_fingerprint" in oracle:
            paired = paired.merge(oracle[["mdp_id", "kernel_fingerprint"]],
                                  on="mdp_id", validate="many_to_one")
            cluster_key = "kernel_fingerprint"
        else:
            cluster_key = "mdp_id"
    else:
        cluster_key = "mdp_id"
    mdp = paired.groupby(cluster_key, as_index=False).agg(
        **{f"{metric}_{method}": (f"{metric}_{method}", "mean")
           for metric in ("brier", "logloss") for method in ("baseline", "candidate")})
    effects = np.column_stack([
        .95 * mdp[f"{metric}_baseline"].to_numpy()
        - mdp[f"{metric}_candidate"].to_numpy()
        for metric in ("brier", "logloss")
    ])
    center = effects.mean(axis=0)
    se = effects.std(axis=0, ddof=1) / np.sqrt(len(mdp))
    if (se <= 0).any():
        raise ValueError("Degenerate MDP effect")
    rng = np.random.default_rng(20261001)
    draws = rng.integers(0, len(mdp), size=(2000, len(mdp)))
    critical = float(np.quantile(np.max((effects[draws].mean(axis=1) - center) / se,
                                        axis=1), .95))
    lower = center - critical * se
    result = {
        "status": "exploratory_opened_cohort_not_confirmation",
        "baseline": baseline, "candidate": candidate,
        "mdps": 42, "independent_kernel_clusters": len(mdp),
        "draws": 2000, "margin": .05,
        "simultaneous_critical_value": critical,
        "metrics": {
            metric: {
                "relative_gain": float(1 - mdp[f"{metric}_candidate"].mean() /
                                       mdp[f"{metric}_baseline"].mean()),
                "mean_excess_over_5pct": float(center[i]),
                "simultaneous_one_sided_lower_95": float(lower[i]),
            } for i, metric in enumerate(("brier", "logloss"))
        },
        "development_margin_gate": "GO" if (lower > 0).all() else "NO-GO",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="Artifact directory relative to repo root")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    directory = (ROOT / args.artifact).resolve()
    if not directory.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Artifact must be under repository artifacts/")
    path = directory / "scored_predictions.csv"
    output = directory / "paired_margin_audit.json"
    print(json.dumps(audit(path, args.baseline, args.candidate, output), indent=2))


if __name__ == "__main__":
    main()

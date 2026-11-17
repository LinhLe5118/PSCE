"""Read-only audit of the published 155-row sticky BRIDGE summary.

This is an author-metadata reanalysis of online sample complexity, not an OEH
offline benchmark and not a rerun of PPO/DQN/SQIRL/GORP.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "metadata/summary_sticky.csv"
OUTPUT = HERE / "expanded_panel_2026-10-08/published_155_online_metadata_audit.json"
PAIRS = (("sqirl", "ppo"), ("sqirl", "dqn"), ("ppo", "dqn"))
PAPER_TABLE_2 = {"ppo": 98, "dqn": 78, "sqirl": 77, "gorp": 29}
PAPER_TABLE_3 = {"sqirl/ppo": {"rho": 0.80, "median_ratio": 1.19},
                 "sqirl/dqn": {"rho": 0.56, "median_ratio": 0.85},
                 "ppo/dqn": {"rho": 0.55, "median_ratio": 0.56}}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = pd.read_csv(SOURCE)
    if len(source) != 155 or source.mdp_name.duplicated().any():
        raise ValueError("Expected 155 distinct published sticky MDPs")
    finite = {}
    pairs = {}
    for name in PAPER_TABLE_2:
        sample = pd.to_numeric(source[f"sample_complexity_{name}"], errors="coerce").to_numpy(float)
        finite[name] = int(np.isfinite(sample).sum())
    for first, second in PAIRS:
        a = pd.to_numeric(source[f"sample_complexity_{first}"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(source[f"sample_complexity_{second}"], errors="coerce").to_numpy(float)
        keep = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        rho = float(spearmanr(a[keep], b[keep]).statistic) if keep.sum() >= 3 else None
        pairs[f"{first}/{second}"] = {"both_finite": int(keep.sum()),
                                      "rho": rho,
                                      "median_ratio": float(np.median(a[keep] / b[keep])) if keep.any() else None}
    result = {"status": "PUBLISHED_ONLINE_METADATA_REANALYSIS_NOT_OFFLINE_OEH_RESULT",
              "source_url": "https://zenodo.org/records/12556298",
              "paper_url": "https://aima.cs.berkeley.edu/~russell/papers/iclr24-horizon.pdf",
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "mdps": len(source),
              "origin_counts": source.origin.value_counts().to_dict(),
              "finite_sample_complexity_counts_from_v1_1_1_csv": finite,
              "both_finite_pairwise_from_v1_1_1_csv": pairs,
              "paper_reported_table_2": PAPER_TABLE_2,
              "paper_reported_table_3": PAPER_TABLE_3,
              "interpretation": ("CSV finite-counts and pairwise statistics are a version/field audit only. "
                                 "Any mismatch with the paper must be investigated before claiming an exact reproduction. "
                                 "The paper evaluates online learner sample complexity, whereas OEH evaluates offline "
                                 "seven-class learner requirement on team-generated logs.")}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("mdps", "origin_counts",
                                       "finite_sample_complexity_counts_from_v1_1_1_csv",
                                       "both_finite_pairwise_from_v1_1_1_csv")}, indent=2))


if __name__ == "__main__":
    main()

"""Descriptive MDP-cluster uncertainty for BRIDGE original scalar scores.

Runs only after the full TQ/SACo reference grid is complete. It does not
select or tune a comparator and does not create a new confirmatory test set.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[2]
ERI = ROOT / "artifacts/week8/bridge_original_eri_eh_exploratory_2026-10-08"
TQSACO = ROOT / "artifacts/week8/bridge_original_tqsaco_exploratory_2026-10-08"
LABELS = ROOT / "artifacts/week8/bridge_reduced_test_labels_2026-10-07"
OUT = ROOT / "artifacts/week8/bridge_original_scores_uncertainty_2026-10-08"
METHODS = {
    "ERI original formula": "eri_original",
    "TQ original formula / author DQN reference": "tq_original",
    "SACo original formula / author replay reference": "saco_original",
    "BRIDGE author EPW": "eh_author_epw",
    "BRIDGE author k_min": "eh_author_k_min",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def correlations(score: np.ndarray, event: np.ndarray) -> tuple[float, float]:
    if len(score) < 3 or np.unique(score).size < 2 or np.unique(event).size < 2:
        return np.nan, np.nan
    return float(spearmanr(score, event).statistic), float(kendalltau(score, event, variant="b").statistic)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    for directory in (ERI, TQSACO):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for filename, expected in manifest["output_sha256"].items():
            if sha256(directory / filename) != expected:
                raise RuntimeError(f"Score artifact changed: {directory / filename}")
    labels_manifest = json.loads((LABELS / "manifest.json").read_text(encoding="utf-8"))
    if sha256(LABELS / "labels.csv") != labels_manifest["labels_sha256"]:
        raise RuntimeError("Labels changed")
    left = pd.read_csv(ERI / "raw_scores.csv")
    right = pd.read_csv(TQSACO / "raw_scores.csv")
    if len(left) != 70 or len(right) != 70:
        raise RuntimeError("Incomplete 14-task / 5-behavior score grid")
    raw = left.merge(right, on=["mdp_id", "policy_family"], validate="one_to_one")
    labels = pd.read_csv(LABELS / "labels.csv")
    labels["mdp_id"] = labels.mdp_id.str.removesuffix("_sticky25")
    joined = labels.merge(raw, on=["mdp_id", "policy_family"], validate="many_to_one")
    if len(joined) != 210:
        raise RuntimeError("Score/label grid mismatch")
    tasks = np.asarray(sorted(joined.mdp_id.unique()))
    if len(tasks) != 14:
        raise RuntimeError("Unexpected number of task clusters")
    rng = np.random.default_rng(20261008)
    draws = rng.integers(0, len(tasks), size=(1000, len(tasks)))
    results = []
    for learner, group in joined.groupby("algorithm"):
        for method, field in METHODS.items():
            subset = group[np.isfinite(group[field])].copy()
            score = subset[field].to_numpy(float)
            event = subset.event.to_numpy(int)
            rho, tau = correlations(score, event)
            parts = [(sub[field].to_numpy(float), sub.event.to_numpy(int))
                     for task in tasks for sub in (subset[subset.mdp_id == task],)]
            boot = []
            for draw in draws:
                s = np.concatenate([parts[index][0] for index in draw])
                e = np.concatenate([parts[index][1] for index in draw])
                value, _ = correlations(s, e)
                if np.isfinite(value):
                    boot.append(value)
            interval = np.percentile(boot, [2.5, 97.5]) if boot else [np.nan, np.nan]
            results.append({"algorithm": learner, "method": method, "rows": len(subset),
                            "mdps_with_defined_score": subset.mdp_id.nunique(),
                            "defined_fraction": len(subset) / len(group),
                            "censored_fraction_on_defined_rows": float(subset.censored.mean()) if len(subset) else np.nan,
                            "spearman_score_vs_harder_event": rho,
                            "kendall_tau_b_score_vs_harder_event": tau,
                            "spearman_mdp_bootstrap_95_low": interval[0],
                            "spearman_mdp_bootstrap_95_high": interval[1],
                            "valid_bootstrap": len(boot), "requested_bootstrap": len(draws),
                            "note": "exploratory; original scalar score, not seven-class forecast"})
    OUT.mkdir(parents=True)
    frame = pd.DataFrame(results)
    frame.to_csv(OUT / "rank_uncertainty.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "status": "COMPLETE_DESCRIPTIVE_OPENED_BRIDGE_PANEL",
        "task_clusters": 14, "bootstrap_draws": 1000, "bootstrap_seed": 20261008,
        "source_sha256": {"eri_raw": sha256(ERI / "raw_scores.csv"),
                          "tqsaco_raw": sha256(TQSACO / "raw_scores.csv"),
                          "labels": sha256(LABELS / "labels.csv"),
                          "script": sha256(Path(__file__))},
        "output_sha256": sha256(OUT / "rank_uncertainty.csv"),
        "warning": "Already opened BRIDGE outcomes; CI is descriptive and does not prove method superiority."
    }, indent=2) + "\n", encoding="utf-8")
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

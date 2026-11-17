"""Exploratory rank audit of published dataset scores versus OEH difficulty.

ERI is evaluated only where the historical pilot mean return was positive.
TQ/SACo require references generated from the true MDP and are privileged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
KEYS = ["mdp_id", "policy_family", "algorithm", "panel"]
P = [f"p_{i}" for i in range(7)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rho(easy_score: np.ndarray, event: np.ndarray) -> float:
    if len(easy_score) < 3 or np.unique(easy_score).size < 2 or np.unique(event).size < 2:
        return float("nan")
    return float(spearmanr(easy_score, -event).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/week8/original_score_rank_exploratory_2026-10-07")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    seal = json.loads((SOURCE / "preoutcome_seal.json").read_text(encoding="utf-8"))
    result = json.loads((SOURCE / "result_manifest.json").read_text(encoding="utf-8"))
    paths = ("predictions_preoutcome.csv", "published_preoutcome.csv", "tqsaco_original_preoutcome.csv",
             "oracle_descriptors_preoutcome.csv")
    for name in paths:
        if sha256(SOURCE / name) != seal["artifact_sha256"][name]:
            raise RuntimeError(f"Preoutcome file changed: {name}")
    if sha256(SOURCE / "labels.csv") != result["artifact_sha256"]["labels.csv"]:
        raise RuntimeError("Sealed labels changed")
    labels = pd.read_csv(SOURCE / "labels.csv")
    pred = pd.read_csv(SOURCE / "predictions_preoutcome.csv")
    pred = pred[pred.method == "oeh_final"][[*KEYS, *P]]
    if len(pred) != 1800 or pred[KEYS].duplicated().any():
        raise RuntimeError("Incomplete OEH forecast")
    pred["oeh_easy_score"] = -(pred[P].to_numpy(float) @ np.arange(7))
    base = labels[[*KEYS, "event"]].merge(pred[[*KEYS, "oeh_easy_score"]], on=KEYS, validate="one_to_one")
    published = pd.read_csv(SOURCE / "published_preoutcome.csv")
    original = pd.read_csv(SOURCE / "tqsaco_original_preoutcome.csv")
    base = base.merge(published[["mdp_id", "policy_family", "eri", "eri_eligible"]],
                      on=["mdp_id", "policy_family"], validate="many_to_one")
    base = base.merge(original[["mdp_id", "policy_family", "tq_original", "saco_original"]],
                      on=["mdp_id", "policy_family"], validate="many_to_one")
    out.mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(20261007)
    for (panel, learner), group in base.groupby(["panel", "algorithm"], observed=True):
        for name, column, access in (("eri_original_formula", "eri", "pilot_only"),
                                     ("tq_original_reference", "tq_original", "privileged_reference"),
                                     ("saco_original_reference", "saco_original", "privileged_reference")):
            eligible = group[(group.eri_eligible > 0) if name.startswith("eri") else np.ones(len(group), bool)]
            eligible = eligible[np.isfinite(eligible[column]) & np.isfinite(eligible.oeh_easy_score)]
            observed = rho(eligible[column].to_numpy(), eligible.event.to_numpy())
            oeh = rho(eligible.oeh_easy_score.to_numpy(), eligible.event.to_numpy())
            clusters = eligible.mdp_id.unique()
            samples = []
            for _ in range(1000):
                draw = rng.choice(clusters, size=len(clusters), replace=True)
                positions = np.concatenate([np.flatnonzero(eligible.mdp_id.to_numpy() == mdp) for mdp in draw])
                target = eligible.event.to_numpy()[positions]
                a = rho(eligible.oeh_easy_score.to_numpy()[positions], target)
                b = rho(eligible[column].to_numpy()[positions], target)
                if np.isfinite(a) and np.isfinite(b):
                    samples.append(a - b)
            low, high = (np.quantile(samples, [.025, .975]) if samples else [np.nan, np.nan])
            rows.append(dict(panel=panel, algorithm=learner, method=name, information=access,
                             rows=len(eligible), mdps=len(clusters), eligible_fraction=len(eligible) / len(group),
                             score_spearman_easy=observed, oeh_spearman_easy_on_same_rows=oeh,
                             oeh_minus_score=oeh - observed, paired_cluster_bootstrap_low=low,
                             paired_cluster_bootstrap_high=high, valid_bootstrap=len(samples),
                             censored_fraction=float((eligible.event == 6).mean())))
    table = pd.DataFrame(rows)
    table.to_csv(out / "rank_audit.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "status": "complete_exploratory_opened_cohort", "source_sha256": {n: sha256(SOURCE / n) for n in (*paths, "labels.csv")},
        "script_sha256": sha256(Path(__file__)), "bootstrap_mdps": 1000,
        "warning": "Score directions and endpoints differ; TQ/SACo use privileged true-MDP references. These are rank diagnostics, not matched seven-class forecasters or claims of superiority over original systems.",
        "output_sha256": sha256(out / "rank_audit.csv"),
    }, indent=2) + "\n", encoding="utf-8")
    print(table[table.panel == "primary"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

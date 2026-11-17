"""Exploratory difficulty diagnostics from an already sealed OEH cohort.

This script reads the frozen predictions and labels, never refits or changes
them. Decision and subgroup analyses are post-hoc on this opened cohort; they
must be frozen on development before use on any independent benchmark.
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
BUDGETS = (1, 2, 5, 10, 20, 50)
PROBS = [f"p_{i}" for i in range(7)]
METHODS = ("oeh_final", "b2_matched", "b2_plus_eri", "b2_plus_bwd")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source() -> dict:
    seal = json.loads((SOURCE / "preoutcome_seal.json").read_text(encoding="utf-8"))
    result = json.loads((SOURCE / "result_manifest.json").read_text(encoding="utf-8"))
    if result["status"] != "complete" or result["preoutcome_seal_sha256"] != sha256(SOURCE / "preoutcome_seal.json"):
        raise RuntimeError("The sealed cohort is incomplete or its preoutcome seal changed")
    expected = {"predictions_preoutcome.csv": seal["artifact_sha256"]["predictions_preoutcome.csv"],
                "labels.csv": result["artifact_sha256"]["labels.csv"]}
    for name, digest in expected.items():
        if sha256(SOURCE / name) != digest:
            raise RuntimeError(f"Source artifact changed: {name}")
    return expected


def choose_budget(probabilities: np.ndarray, q: float = .8) -> np.ndarray:
    """Smallest operational budget with P(E <= budget) >= q; -1 abstains."""
    if probabilities.ndim != 2 or probabilities.shape[1] != 7 or not (0 < q <= 1):
        raise ValueError("Expected seven-category probabilities and 0 < q <= 1")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1):
        raise ValueError("Invalid probability rows")
    cumulative = np.cumsum(probabilities, axis=1)
    candidates = np.array([3, 4, 5])  # x10, x20, x50; pilot already costs x10
    reached = cumulative[:, candidates] >= q
    return np.where(reached.any(1), candidates[reached.argmax(1)], -1)


def score_rows(frame: pd.DataFrame) -> pd.DataFrame:
    p = frame[PROBS].to_numpy(float)
    y = frame.event.to_numpy(int)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise RuntimeError("Malformed sealed probability distribution")
    cdf = np.cumsum(p, axis=1)[:, :6]
    target = y[:, None] <= np.arange(6)
    scored = frame[KEYS + ["method", "event"]].copy()
    scored["brier"] = np.mean((cdf - target) ** 2, axis=1)
    scored["log_loss"] = -np.log(np.maximum(p[np.arange(len(p)), y], 1e-12))
    scored["expected_class"] = p @ np.arange(7)
    selected = choose_budget(p)
    scored["chosen_event_index"] = selected
    scored["chosen_budget"] = np.where(selected < 0, np.nan, np.array([1, 2, 5, 10, 20, 50, 0])[np.maximum(selected, 0)])
    scored["sufficient_when_chosen"] = np.where(selected < 0, np.nan, (y <= selected).astype(float))
    for k, budget in enumerate(BUDGETS):
        scored[f"cdf_x{budget}"] = cdf[:, k]
        scored[f"observed_by_x{budget}"] = target[:, k].astype(int)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/week8/difficulty_diagnostics_exploratory_2026-10-07")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    verified = verify_source()
    labels = pd.read_csv(SOURCE / "labels.csv")
    predictions = pd.read_csv(SOURCE / "predictions_preoutcome.csv")
    if labels[KEYS[:3]].duplicated().any() or len(labels) != 1800 or set(labels.event) - set(range(7)):
        raise RuntimeError("Unexpected label panel")
    if predictions[KEYS + ["method"]].duplicated().any():
        raise RuntimeError("Duplicate forecast identity")
    predictions = predictions[predictions.method.isin(METHODS)].copy()
    frame = predictions.merge(labels[KEYS + ["event"]], on=KEYS, validate="many_to_one")
    if len(frame) != 1800 * len(METHODS):
        raise RuntimeError("Incomplete matched forecast panel")
    scored = score_rows(frame)
    output.mkdir(parents=True)
    scored.to_csv(output / "row_scores.csv", index=False)

    distribution = (labels.groupby(["panel", "algorithm", "policy_family", "event"], observed=True)
                    .size().rename("rows").reset_index())
    distribution.to_csv(output / "difficulty_distribution.csv", index=False)
    metrics = (scored.groupby(["panel", "method", "algorithm"], observed=True)
               .agg(rows=("event", "size"), mdps=("mdp_id", "nunique"), brier=("brier", "mean"),
                    log_loss=("log_loss", "mean"), censored_fraction=("event", lambda x: float((x == 6).mean())),
                    abstention_fraction=("chosen_event_index", lambda x: float((x < 0).mean())),
                    success_when_selected=("sufficient_when_chosen", "mean"),
                    mean_chosen_budget=("chosen_budget", "mean"))
               .reset_index())
    metrics.to_csv(output / "metrics_by_learner.csv", index=False)

    calibration = []
    for (panel, method, learner), group in scored.groupby(["panel", "method", "algorithm"], observed=True):
        for budget in BUDGETS:
            prob = group[f"cdf_x{budget}"].to_numpy(float)
            truth = group[f"observed_by_x{budget}"].to_numpy(float)
            for bin_index in range(5):
                left, right = bin_index / 5, (bin_index + 1) / 5
                mask = (prob >= left) & ((prob < right) if bin_index < 4 else (prob <= right))
                if mask.any():
                    calibration.append(dict(panel=panel, method=method, algorithm=learner,
                                            budget=budget, bin=bin_index, rows=int(mask.sum()),
                                            predicted=float(prob[mask].mean()), observed=float(truth[mask].mean())))
    pd.DataFrame(calibration).to_csv(output / "calibration_by_budget.csv", index=False)

    ranks = []
    for (panel, method, learner), group in scored.groupby(["panel", "method", "algorithm"], observed=True):
        relation = spearmanr(-group.expected_class, -group.event)
        ranks.append(dict(panel=panel, method=method, algorithm=learner,
                          rows=len(group), mdps=group.mdp_id.nunique(),
                          spearman_easy_score=float(relation.statistic),
                          censored_fraction=float((group.event == 6).mean())))
    pd.DataFrame(ranks).to_csv(output / "difficulty_rank.csv", index=False)

    fixed = []
    for (panel, learner), group in labels.groupby(["panel", "algorithm"], observed=True):
        for index in (3, 4, 5):
            fixed.append(dict(panel=panel, algorithm=learner, budget=BUDGETS[index],
                              rows=len(group), success_fraction=float((group.event <= index).mean())))
    pd.DataFrame(fixed).to_csv(output / "fixed_budget_reference.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "status": "complete_exploratory_opened_cohort", "source": str(SOURCE),
        "source_sha256": verified, "script_sha256": sha256(Path(__file__)),
        "decision_q": .8, "operational_budgets": [10, 20, 50],
        "warning": "Decision q and subgroup analyses were chosen after this cohort opened; descriptive only. Freeze independently before BRIDGE outcomes.",
        "rows": len(labels), "mdps": labels.mdp_id.nunique(), "methods": list(METHODS),
    }, indent=2), encoding="utf-8")
    print(f"Exploratory difficulty diagnostics: {output.relative_to(ROOT)}")
    print(metrics[metrics.panel == "primary"].to_string(index=False))


if __name__ == "__main__":
    main()

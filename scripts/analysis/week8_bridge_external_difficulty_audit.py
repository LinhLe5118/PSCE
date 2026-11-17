"""Exploratory BRIDGE difficulty ranking: reduced OEH versus published scores.

This cannot evaluate full OEH or establish superiority at the authors' native
endpoints. All BRIDGE requirement labels were opened before this analysis.
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
WEEK8 = ROOT / "artifacts/week8"
SCORED = WEEK8 / "bridge_reduced_test_analysis_2026-10-07"
ERI_EH = WEEK8 / "bridge_original_eri_eh_exploratory_2026-10-08"
TQ_SACO = WEEK8 / "bridge_original_tqsaco_exploratory_2026-10-08"
KEYS = ["mdp_id", "policy_family"]
PROB = [f"p_{i}" for i in range(7)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path: Path, manifest: Path, key: str) -> pd.DataFrame:
    expected = json.loads(manifest.read_text(encoding="utf-8"))["output_sha256"][key]
    if sha256(path) != expected:
        raise RuntimeError(f"Sealed source hash mismatch: {path}")
    return pd.read_csv(path)


def rank_rho(score: np.ndarray, event: np.ndarray) -> float:
    if len(score) < 3 or np.unique(score).size < 2 or np.unique(event).size < 2:
        return float("nan")
    return float(spearmanr(score, -event).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=WEEK8 / "bridge_reduced_vs_original_difficulty_exploratory_2026-10-08",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    scored_file = SCORED / "scored.csv"
    eri_file = ERI_EH / "raw_scores.csv"
    tq_file = TQ_SACO / "raw_scores.csv"
    scored = verified(scored_file, SCORED / "manifest.json", "scored.csv")
    eri = verified(eri_file, ERI_EH / "manifest.json", "raw_scores.csv")
    tq = verified(tq_file, TQ_SACO / "manifest.json", "raw_scores.csv")
    forecast = scored[scored.method == "oeh_without_direct_oehb"].copy()
    if len(forecast) != 210 or forecast[[*KEYS, "algorithm"]].duplicated().any():
        raise RuntimeError("Expected exactly 210 unique reduced-OEH rows")
    forecast["oeh_easy"] = -(forecast[PROB].to_numpy(float) @ np.arange(7))
    base = forecast[[*KEYS, "algorithm", "event", "oeh_easy"]]
    base = base.merge(
        eri[[*KEYS, "eri_original", "eri_defined", "eh_author_epw", "eh_author_k_min"]],
        on=KEYS, validate="many_to_one",
    ).merge(
        tq[[*KEYS, "tq_original", "tq_defined", "saco_original"]],
        on=KEYS, validate="many_to_one",
    )
    if len(base) != 210 or base[[*KEYS, "algorithm"]].duplicated().any():
        raise RuntimeError("Score merge dropped or duplicated rows")

    candidates = (
        ("ERI", "eri_original", "pilot_only", 1, "eri_defined"),
        ("TQ", "tq_original", "true_mdp_reference", 1, "tq_defined"),
        ("SACo", "saco_original", "true_mdp_reference", 1, None),
        ("EH_EPWs", "eh_author_epw", "author_true_mdp_metadata", -1, None),
        ("EH_k_min", "eh_author_k_min", "author_true_mdp_metadata", -1, None),
    )
    rng = np.random.default_rng(20261008)
    rows = []
    for learner, group in base.groupby("algorithm", sort=True):
        for method, column, access, direction, defined_column in candidates:
            eligible = group.copy()
            if defined_column:
                eligible = eligible[eligible[defined_column].astype(bool)]
            eligible = eligible[np.isfinite(eligible[column]) & np.isfinite(eligible.oeh_easy)]
            score = eligible[column].to_numpy(float) * direction
            oeh = eligible.oeh_easy.to_numpy(float)
            event = eligible.event.to_numpy(int)
            rho_original = rank_rho(score, event)
            rho_oeh = rank_rho(oeh, event)
            clusters = eligible.mdp_id.unique()
            offsets = [np.flatnonzero(eligible.mdp_id.to_numpy() == mdp) for mdp in clusters]
            draws = []
            for _ in range(1000):
                positions = np.concatenate([offsets[i] for i in rng.integers(len(offsets), size=len(offsets))])
                a, b = rank_rho(oeh[positions], event[positions]), rank_rho(score[positions], event[positions])
                if np.isfinite(a) and np.isfinite(b):
                    draws.append(a - b)
            low, high = np.quantile(draws, [.025, .975]) if draws else (np.nan, np.nan)
            rows.append({
                "learner": learner, "published_score": method, "access": access,
                "rows": len(eligible), "mdps": len(clusters),
                "score_spearman_easy": rho_original,
                "reduced_oeh_spearman_easy_same_rows": rho_oeh,
                "reduced_oeh_minus_score": rho_oeh - rho_original,
                "paired_cluster_bootstrap_low": low,
                "paired_cluster_bootstrap_high": high,
                "valid_bootstrap": len(draws),
                "censored_fraction": float((event == 6).mean()) if len(event) else np.nan,
            })
    result = pd.DataFrame(rows)
    output.mkdir(parents=True)
    result.to_csv(output / "rank_audit.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "status": "complete_exploratory_opened_bridge_labels",
        "method": "oeh_without_direct_oehb (reduced ablation, not full OEH)",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (scored_file, eri_file, tq_file)},
        "script_sha256": sha256(Path(__file__)),
        "output_sha256": sha256(output / "rank_audit.csv"),
        "bootstrap_mdps": 1000,
        "warning": "Different native endpoints/access; scores rank E_l only as a transfer diagnostic. CIs descriptive, labels already opened. EH signs oriented high=easier by negating its published difficulty descriptor.",
    }, indent=2) + "\n", encoding="utf-8")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

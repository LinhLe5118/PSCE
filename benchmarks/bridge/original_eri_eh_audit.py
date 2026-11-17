"""Exploratory BRIDGE audit of original ERI and author-published EH metadata.

ERI is computed from each pilot alone. The EH fields are read from the BRIDGE
authors' pinned sticky-action summary; they have privileged MDP access and a
different horizon interpretation from our finite-horizon offline label.
Neither quantity is a seven-class requirement forecast.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.analysis.oeh_final_published import episode_returns

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "datasets/bridge_v1_1_1/offline_pilot_preoutcome_v1"
LABELS = ROOT / "artifacts/week8/bridge_reduced_test_labels_2026-10-07"
AUTHOR = ROOT / "benchmarks/bridge/metadata/summary_sticky.csv"
SOURCE = ROOT / "benchmarks/bridge/source_manifest.json"
OUT = ROOT / "artifacts/week8/bridge_original_eri_eh_exploratory_2026-10-08"
NAMES = {"BC": "behavior_cloning_tabular", "CFQI": "cfqi_tabular", "IQL": "iql_tabular"}


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rho(score: np.ndarray, event: np.ndarray) -> float:
    if len(score) < 3 or np.unique(score).size < 2 or np.unique(event).size < 2:
        return float("nan")
    return float(spearmanr(score, event).statistic)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if hash_file(AUTHOR, "md5") != source["source_csv_md5"]["summary_sticky.csv"]:
        raise RuntimeError("Published BRIDGE sticky summary hash differs")
    pilot_manifest = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    if pilot_manifest["status"] != "SEALED_PILOT_ONLY_NO_TEST_LABELS" or len(pilot_manifest["pilot_groups"]) != 70:
        raise RuntimeError("Pilot manifest is incomplete")
    author = pd.read_csv(AUTHOR)
    author = author[author.mdp_name.isin(pilot_manifest["tasks"])].copy()
    if len(author) != 14 or author.mdp_name.duplicated().any():
        raise RuntimeError("Author metadata does not contain exactly 14 selected tasks")
    author = author.set_index("mdp_name")
    records = []
    for row in pilot_manifest["pilot_groups"]:
        path = PILOT / row["file"]
        if hash_file(path) != row["sha256"]:
            raise RuntimeError(f"Pilot hash mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        returns = episode_returns(arrays)
        mean = float(returns.mean())
        valid = bool(np.isfinite(mean) and mean > 0)
        published = author.loc[row["mdp_id"]]
        records.append({"mdp_id": row["mdp_id"], "policy_family": row["policy_family"],
                        "eri_original": float((returns.max() - mean) / mean) if valid else np.nan,
                        "eri_defined": valid, "pilot_episodes": len(returns),
                        "pilot_mean_return": mean,
                        "eh_author_epw": float(published.epw),
                        "eh_author_k_min": float(published.k_min),
                        "eh_author_horizon": int(published.horizon),
                        "eri_access": "pilot_only",
                        "eh_access": "privileged_author_published_MDP_analysis"})
    raw = pd.DataFrame(records).sort_values(["mdp_id", "policy_family"])
    if len(raw) != 70 or raw[["mdp_id", "policy_family"]].duplicated().any():
        raise RuntimeError("Original score table is incomplete")

    label_manifest = json.loads((LABELS / "manifest.json").read_text(encoding="utf-8"))
    label_path = LABELS / "labels.csv"
    if (label_manifest["status"] != "BRIDGE_REDUCED_TEST_LABELS_COMPLETE" or
            hash_file(label_path) != label_manifest["labels_sha256"]):
        raise RuntimeError("BRIDGE labels are not sealed")
    labels = pd.read_csv(label_path)
    if not labels.mdp_id.str.endswith("_sticky25").all():
        raise RuntimeError("Unexpected label task identity")
    labels["mdp_id"] = labels.mdp_id.str.removesuffix("_sticky25")
    joined = labels.merge(raw, on=["mdp_id", "policy_family"], validate="many_to_one")
    if len(joined) != 210 or joined[["mdp_id", "policy_family", "algorithm"]].duplicated().any():
        raise RuntimeError("Original score/label join mismatch")

    summary = []
    for algorithm, group in joined.groupby("algorithm"):
        for method, field, defined, access in (
            ("ERI original formula", "eri_original", group.eri_defined, "pilot_only"),
            ("BRIDGE author EPW", "eh_author_epw", np.isfinite(group.eh_author_epw),
             "privileged_published_MDP"),
            ("BRIDGE author k_min", "eh_author_k_min", np.isfinite(group.eh_author_k_min),
             "privileged_published_MDP"),
        ):
            subset = group[defined & np.isfinite(group[field])]
            value = rho(subset[field].to_numpy(float), subset.event.to_numpy(int))
            summary.append({"algorithm": algorithm, "method": method, "access": access,
                            "rows": len(subset), "mdps": subset.mdp_id.nunique(),
                            "defined_fraction": len(subset) / len(group),
                            "spearman_score_vs_harder_event": value,
                            "note": "exploratory; score endpoint differs from seven-class E_l"})
    OUT.mkdir(parents=True)
    raw.to_csv(OUT / "raw_scores.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "rank_summary.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "status": "COMPLETE_EXPLORATORY_ORIGINAL_ERI_AND_AUTHOR_EH_METADATA",
        "comparison_scope": "original scalar scores versus offline E_l rank; no Brier/log-loss comparison",
        "test_outcomes_already_open": True,
        "source_sha256": {"pilot_manifest": hash_file(PILOT / "manifest.json"),
                          "labels_manifest": hash_file(LABELS / "manifest.json"),
                          "author_sticky_summary": hash_file(AUTHOR),
                          "script": hash_file(Path(__file__))},
        "rows": 70, "learner_rows": 210,
        "author_horizon_warning": "Published EPW/k_min are MDP-based and not matched to our finite-horizon offline learner target.",
        "output_sha256": {name: hash_file(OUT / name) for name in ("raw_scores.csv", "rank_summary.csv")},
    }, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

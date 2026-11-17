"""Score sealed fresh BRIDGE laws and separate original-method rank signals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .fresh_adapter import PANEL

ROOT = Path(__file__).resolve().parents[2]
B2 = ROOT / "artifacts/week8/bridge_fresh_b2_forecast_preoutcome_2026-10-08"
PUBLISHED_CF = ROOT / "artifacts/week8/bridge_fresh_published_xgboost_cf_preoutcome_2026-10-08"
PUBLISHED_CF_FULL = ROOT / "artifacts/week8/bridge_fresh_published_xgboost_cf_full108_preoutcome_2026-10-08"
FULL = ROOT / "artifacts/week8/bridge_fresh_full_forecast_preoutcome_2026-10-08"
ORIGINAL = ROOT / "artifacts/week8/bridge_fresh_original_scores_preoutcome_2026-10-08"
LABELS = ROOT / "artifacts/week8/bridge_fresh_test_labels_2026-10-08"
OUT = ROOT / "artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_prediction(directory: Path, status: str, protocol_hash: str) -> tuple[pd.DataFrame, str]:
    manifest_file = directory / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    file = directory / "predictions_before_label_read.csv"
    if (manifest["status"] != status or manifest["test_labels_read"] or
            manifest["protocol_sha256"] != protocol_hash or
            manifest["output_sha256"]["predictions"] != sha256(file)):
        raise RuntimeError(f"Bad prelabel predictions: {directory}")
    return pd.read_csv(file), sha256(manifest_file)


def rank(score: np.ndarray, event: np.ndarray) -> float:
    good = np.isfinite(score) & np.isfinite(event)
    if good.sum() < 3 or np.unique(score[good]).size < 2 or np.unique(event[good]).size < 2:
        return float("nan")
    return float(spearmanr(score[good], event[good]).statistic)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    protocol_file, seal_file = PANEL / "protocol.json", PANEL / "prelabel_seal.json"
    protocol_hash = sha256(protocol_file)
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    seal = json.loads(seal_file.read_text(encoding="utf-8"))
    labels_manifest_file, labels_file = LABELS / "manifest.json", LABELS / "labels.csv"
    lm = json.loads(labels_manifest_file.read_text(encoding="utf-8"))
    if (protocol["status"] != "FRESH_BRIDGE_PROTOCOL_FROZEN_BEFORE_LABELS" or
            seal["status"] != "FRESH_PRELABEL_SEAL_READY" or
            seal["sha256"]["protocol"] != protocol_hash or
            lm["status"] != "FRESH_BRIDGE_TEST_LABELS_COMPLETE" or
            lm["prelabel_seal_sha256"] != sha256(seal_file) or
            lm["labels_sha256"] != sha256(labels_file)):
        raise RuntimeError("Fresh scoring seal failed")
    full, full_manifest_hash = read_prediction(FULL, "SEALED_FRESH_FULL_108_FORECAST_NO_TEST_LABELS", protocol_hash)
    b2, b2_manifest_hash = read_prediction(B2, "SEALED_FRESH_B2_FORECAST_NO_TEST_LABELS", protocol_hash)
    published_cf, published_cf_manifest_hash = read_prediction(
        PUBLISHED_CF, "SEALED_FRESH_PUBLISHED_XGBOOST_CF_ADAPTATION_NO_TEST_LABELS", protocol_hash)
    published_cf_full, published_cf_full_manifest_hash = read_prediction(
        PUBLISHED_CF_FULL, "SEALED_FRESH_XGBOOST_CF_FULL108_ARCHITECTURE_CONTROL_NO_TEST_LABELS", protocol_hash)
    if (seal["sha256"].get("published_cf_manifest") != sha256(PUBLISHED_CF / "manifest.json") or
            seal["sha256"].get("published_cf_predictions") != sha256(PUBLISHED_CF / "predictions_before_label_read.csv")):
        raise RuntimeError("Published CF was not frozen with the prelabel seal")
    if (seal["sha256"].get("published_cf_full108_manifest") != sha256(PUBLISHED_CF_FULL / "manifest.json") or
            seal["sha256"].get("published_cf_full108_predictions") != sha256(PUBLISHED_CF_FULL / "predictions_before_label_read.csv")):
        raise RuntimeError("Full-input CF was not frozen with the prelabel seal")
    labels = pd.read_csv(labels_file)
    if (len(labels) != 75 or labels[KEYS].duplicated().any() or
            set(labels.mdp_id) != set(protocol["evaluation_tasks"])):
        raise RuntimeError("Fresh label grid mismatch")
    predictions = pd.concat([full, b2, published_cf, published_cf_full], ignore_index=True)
    scored = predictions.merge(labels[KEYS + ["event", "label_environment_id"]],
                               on=KEYS, validate="many_to_one")
    if len(scored) != 300 or scored[KEYS + ["method"]].duplicated().any():
        raise RuntimeError("Fresh forecast/label join mismatch")
    p = scored[P].to_numpy(float)
    y = scored.event.to_numpy(int)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1) or np.any((y < 0) | (y > 6)):
        raise RuntimeError("Invalid law/event")
    cdf = np.cumsum(p, axis=1)[:, :6]
    scored["brier"] = np.mean((cdf - (y[:, None] <= np.arange(6))) ** 2, axis=1)
    scored["log_loss"] = -np.log(np.maximum(p[np.arange(len(p)), y], 1e-12))
    scored["expected_event"] = p @ np.arange(7)
    summary = (scored.groupby(["algorithm", "method"], observed=True)
               .agg(rows=("event", "size"), tasks=("mdp_id", "nunique"),
                    brier=("brier", "mean"), log_loss=("log_loss", "mean"),
                    censored_fraction=("event", lambda v: float((v == 6).mean())))
               .reset_index())
    task_ids = sorted(labels.mdp_id.unique())
    rng = np.random.default_rng(int(protocol["bootstrap"]["seed"]))
    paired = []
    for learner in sorted(labels.algorithm.unique()):
        part = scored[scored.algorithm == learner]
        for metric in ("brier", "log_loss"):
            pivot = part.pivot(index=["mdp_id", "policy_family"], columns="method", values=metric)
            for comparator in ("b2_matched_internal_diagnostic",
                               "published_xgboost_cf_2026_adapted_b2",
                               "xgboost_cf_2026_full108_architecture_control"):
                delta = pivot["oeh_full_108"] - pivot[comparator]
                by_task = delta.groupby(level="mdp_id").mean().reindex(task_ids)
                if by_task.isna().any() or len(delta) != 25:
                    raise RuntimeError(f"Missing paired task rows: {comparator}")
                samples = rng.choice(by_task.to_numpy(), size=(10000, 5), replace=True).mean(axis=1)
                paired.append({"algorithm": learner, "metric": metric,
                               "comparison": f"oeh_full_108 minus {comparator}",
                               "delta": float(by_task.mean()),
                               "ci95_low_descriptive": float(np.quantile(samples, .025)),
                               "ci95_high_descriptive": float(np.quantile(samples, .975)),
                               "bootstrap_unit": "MDP task, keeping its five behaviors together"})
    original_manifest_file = ORIGINAL / "manifest.json"
    original_manifest = json.loads(original_manifest_file.read_text(encoding="utf-8"))
    original_file = ORIGINAL / "scores_before_label_read.csv"
    if (original_manifest["status"] != "SEALED_FRESH_ORIGINAL_SCALAR_SCORES_NO_TEST_LABELS" or
            original_manifest["scores_sha256"] != sha256(original_file) or
            original_manifest["protocol_sha256"] != protocol_hash):
        raise RuntimeError("Original scalar scores changed")
    original = pd.read_csv(original_file)
    joined = labels.merge(original, on=["mdp_id", "policy_family"], validate="many_to_one")
    joined = joined.merge(full[KEYS + P], on=KEYS, validate="one_to_one")
    if len(joined) != 75:
        raise RuntimeError("Original rank join mismatch")
    joined["oeh_expected_event"] = joined[P].to_numpy(float) @ np.arange(7)
    rank_rows = []
    methods = (("OEH expected event", "oeh_expected_event", "pilot_only"),
               ("ERI original formula", "eri_original", "pilot_only"),
               ("TQ original formula / author DQN reference", "tq_original", "privileged_true_MDP_online_reference"),
               ("SACo original formula / author DQN reference", "saco_original", "privileged_true_MDP_online_reference"),
               ("EH author EPW", "eh_author_epw", "privileged_author_MDP_summary"),
               ("EH author k_min", "eh_author_k_min", "privileged_author_MDP_summary"))
    for learner, group in joined.groupby("algorithm"):
        for method, field, access in methods:
            values = pd.to_numeric(group[field], errors="coerce").to_numpy(float)
            event = group.event.to_numpy(float)
            rank_rows.append({"algorithm": learner, "method": method,
                              "access": access, "defined_rows": int(np.isfinite(values).sum()),
                              "tasks": int(group.mdp_id.nunique()),
                              "spearman_raw_score_vs_harder_event": rank(values, event),
                              "scope": "Scalar rank association only; original methods do not forecast seven E_l classes"})
    # Secondary transfer to a common ranking target, using the original
    # high-is-better direction of TQ/SACo. This is not a seven-class comparison.
    rank_pairs = []
    rank_rng = np.random.default_rng(int(protocol["bootstrap"]["seed"]) + 1)
    for learner, group in joined.groupby("algorithm"):
        for method, field in (("TQ original formula", "tq_original"),
                              ("SACo original formula", "saco_original")):
            sub = group[np.isfinite(pd.to_numeric(group[field], errors="coerce"))]
            task_names = sorted(sub.mdp_id.unique())
            pieces = []
            for name in task_names:
                part = sub[sub.mdp_id == name]
                pieces.append((part[field].to_numpy(float),
                               -part.oeh_expected_event.to_numpy(float),
                               -part.event.to_numpy(float)))
            values = sub[field].to_numpy(float)
            easier = -sub.event.to_numpy(float)
            own = -sub.oeh_expected_event.to_numpy(float)
            method_rho, own_rho = rank(values, easier), rank(own, easier)
            replicates = []
            for _ in range(int(protocol["bootstrap"]["replicates"])):
                selected = rank_rng.integers(0, len(pieces), size=len(pieces))
                sample = tuple(np.concatenate([pieces[int(i)][j] for i in selected])
                               for j in range(3))
                rho_method, rho_own = rank(sample[0], sample[2]), rank(sample[1], sample[2])
                if np.isfinite(rho_method) and np.isfinite(rho_own):
                    replicates.append(rho_own - rho_method)
            rank_pairs.append({"algorithm": learner, "method": method,
                               "rows": len(sub), "tasks": len(task_names),
                               "oeh_rho_same_rows": own_rho,
                               "original_rho": method_rho,
                               "delta_oeh_minus_original": own_rho - method_rho,
                               "ci95_low_descriptive": float(np.quantile(replicates, .025)) if replicates else np.nan,
                               "ci95_high_descriptive": float(np.quantile(replicates, .975)) if replicates else np.nan,
                               "defined_bootstrap_draws": len(replicates),
                               "bootstrap_draws": int(protocol["bootstrap"]["replicates"]),
                               "scope": "Exploratory rank transfer, not same-output forecaster or SOTA claim"})
    OUT.mkdir(parents=True)
    scored.to_csv(OUT / "scored.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "paired_task_bootstrap.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(OUT / "original_rank_comparison.csv", index=False)
    pd.DataFrame(rank_pairs).to_csv(OUT / "original_rank_paired_transfer.csv", index=False)
    lines = ["# Fresh BRIDGE Atari: full OEH transfer\n\n",
             "Five published BRIDGE Atari MDPs; offline pilot and E_l labels generated by this project under the frozen prelabel protocol. All methods and failures are retained.\n\n",
             "`brier` is the mean squared error of the six cumulative event probabilities (ordinal Brier/RPS). Lower Brier and log loss are better.\n\n",
             "## Seven-class forecasts\n\n",
             summary.to_markdown(index=False, floatfmt=".6f"), "\n\n",
             "## Paired difference by MDP (negative favors OEH)\n\n",
             pd.DataFrame(paired).to_markdown(index=False, floatfmt=".6f"), "\n\n",
             "XGBoost-CF is a published general survival formulation adapted to the seven budget classes with pilot-only B2 inputs; it is not a published offline-RL requirement forecaster or a field SOTA result. The full-108-input CF row uses OEH descriptors and is an architecture control, not an independent-method comparator. With only five MDP clusters, bootstrap intervals are descriptive and may be unstable.\n\n",
             "## Published-method scalar scores\n\n",
             pd.DataFrame(rank_rows).to_markdown(index=False, floatfmt=".6f"), "\n\n",
             "## Exploratory paired rank transfer to easier learner requirement\n\n",
             pd.DataFrame(rank_pairs).to_markdown(index=False, floatfmt=".6f"), "\n\n",
             "TQ/SACo use the authors' formulas and DQN reference code retrained on our sticky, reward-scaled adapter; they are not the authors' published task scores. TQ/SACo and EH use privileged MDP information; their scalar scores cannot be assigned OEH's Brier/log loss or called a direct seven-class SOTA comparator. Author EH values are metadata for the published sticky MDP; the reachable-state count of our augmented adapter differs for some tasks, so the EH rows are contextual diagnostics rather than an exact same-representation comparison. A positive/negative raw rank correlation has method-specific meaning.\n"]
    (OUT / "RESULTS.md").write_text("".join(lines), encoding="utf-8")
    source = {"protocol": protocol_hash, "prelabel_seal": sha256(seal_file),
              "full_forecast_manifest": full_manifest_hash,
              "b2_internal_manifest": b2_manifest_hash,
              "published_cf_manifest": published_cf_manifest_hash,
              "published_cf_full108_manifest": published_cf_full_manifest_hash,
              "original_scores_manifest": sha256(original_manifest_file),
              "labels_manifest": sha256(labels_manifest_file)}
    (OUT / "manifest.json").write_text(json.dumps({
        "status": "COMPLETE_FRESH_BRIDGE_ATARI_FULL_OEH_TRANSFER",
        "tasks": 5, "behaviors_per_task": 5, "learners": 3,
        "rows_per_forecaster": 75,
        "role": "third-party published MDPs; team-generated offline pilots and labels",
        "source_sha256": source, "source_script_sha256": sha256(Path(__file__)),
        "output_sha256": {name: sha256(OUT / name) for name in
                          ("scored.csv", "summary.csv", "paired_task_bootstrap.csv",
                           "original_rank_comparison.csv", "original_rank_paired_transfer.csv",
                           "RESULTS.md")}},
        indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(paired).to_string(index=False), flush=True)
    print("Completed fresh BRIDGE full OEH analysis; original scores kept in rank-only table", flush=True)


if __name__ == "__main__":
    main()

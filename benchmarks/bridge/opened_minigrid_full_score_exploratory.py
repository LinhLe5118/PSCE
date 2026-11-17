"""Score full OEH and controls on the same 14 opened MiniGrid task rows."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "artifacts/week8"
FULL = W / "bridge_opened_minigrid_full_forecast_exploratory_2026-10-09"
B2 = W / "bridge_b2_forecast_preoutcome_2026-10-07"
REDUCED = W / "bridge_reduced_forecast_preoutcome_2026-10-07"
LABELS = W / "bridge_reduced_test_labels_2026-10-07"
TQ = W / "bridge_original_tqsaco_exploratory_2026-10-08/raw_scores.csv"
ERI_EH = W / "bridge_original_eri_eh_exploratory_2026-10-08/raw_scores.csv"
OUT = W / "bridge_opened_minigrid_full_analysis_exploratory_2026-10-09"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_prediction(directory: Path, filename: str, expected_status: str) -> pd.DataFrame:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != expected_status:
        raise RuntimeError(f"Unexpected forecast status: {directory}")
    file = directory / filename
    if manifest["output_sha256"]["predictions"] != sha256(file):
        raise RuntimeError(f"Forecast hash mismatch: {file}")
    frame = pd.read_csv(file)
    if len(frame) != 210 or frame[KEYS].duplicated().any():
        raise RuntimeError(f"Incomplete 210 forecast rows: {directory}")
    return frame


def correlation(left, right):
    good = np.isfinite(left) & np.isfinite(right)
    if good.sum() < 3 or len(np.unique(left[good])) < 2 or len(np.unique(right[good])) < 2:
        return float("nan")
    return float(spearmanr(left[good], right[good]).statistic)


def verified_scores(path: Path, columns: list[str]) -> pd.DataFrame:
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if (not manifest["status"].startswith("COMPLETE_EXPLORATORY_ORIGINAL_") or
            manifest["output_sha256"][path.name] != sha256(path)):
        raise RuntimeError(f"Original score source changed: {path}")
    frame = pd.read_csv(path)
    keys = ["mdp_id", "policy_family"]
    if len(frame) != 70 or frame[keys].duplicated().any():
        raise RuntimeError(f"Incomplete original score rows: {path}")
    return frame[keys + columns]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    predictions = pd.concat([
        read_prediction(FULL, "predictions_exploratory.csv",
                        "EXPLORATORY_FULL_108_FORECAST_AFTER_LABEL_OPEN"),
        read_prediction(B2, "predictions_before_label_read.csv",
                        "SEALED_B2_FORECAST_NO_TEST_LABELS"),
        read_prediction(REDUCED, "predictions_before_label_read.csv",
                        "SEALED_REDUCED_OEH_ABLATION_NO_TEST_LABELS"),
    ], ignore_index=True)
    labels_file = LABELS / "labels.csv"
    lm = json.loads((LABELS / "manifest.json").read_text(encoding="utf-8"))
    if lm["labels_sha256"] != sha256(labels_file):
        raise RuntimeError("Opened labels changed")
    labels = pd.read_csv(labels_file)
    labels["mdp_id"] = labels.mdp_id.str.removesuffix("_sticky25")
    if len(labels) != 210 or labels[KEYS].duplicated().any():
        raise RuntimeError("Label task mapping is not one-to-one")
    scored = predictions.merge(labels[KEYS + ["event"]], on=KEYS, validate="many_to_one")
    if (len(scored) != 630 or scored[KEYS + ["method"]].duplicated().any() or
            scored.mdp_id.nunique() != 14):
        raise RuntimeError("Incomplete paired prediction/label comparison")
    p = scored[P].to_numpy(float)
    y = scored.event.to_numpy(int)
    if (not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-7) or
            np.any((y < 0) | (y > 6))):
        raise RuntimeError("Invalid probabilities or event class")
    cdf = np.cumsum(p, axis=1)[:, :6]
    scored["integrated_brier"] = np.mean((cdf - (y[:, None] <= np.arange(6))) ** 2, axis=1)
    scored["seven_class_log_loss"] = -np.log(np.maximum(p[np.arange(len(p)), y], 1e-12))
    scored["expected_event"] = p @ np.arange(7)
    summary = (scored.groupby(["algorithm", "method"], observed=True)
               .agg(rows=("event", "size"), tasks=("mdp_id", "nunique"),
                    brier=("integrated_brier", "mean"),
                    log_loss=("seven_class_log_loss", "mean"),
                    censored_fraction=("event", lambda x: float((x == 6).mean())))
               .reset_index())
    full_name = "oeh_full_108_post_outcome_exploratory"
    rng = np.random.default_rng(20261009)
    pairs = []
    for learner, frame in scored.groupby("algorithm"):
        for metric in ("integrated_brier", "seven_class_log_loss"):
            pivot = frame.pivot(index=["mdp_id", "policy_family"], columns="method", values=metric)
            for comparator in ("b2_matched", "oeh_without_direct_oehb"):
                by_task = (pivot[full_name] - pivot[comparator]).groupby(level="mdp_id").mean()
                if len(by_task) != 14 or by_task.isna().any():
                    raise RuntimeError("Missing paired task")
                samples = rng.choice(by_task.to_numpy(), size=(10000, 14), replace=True).mean(axis=1)
                pairs.append({"algorithm": learner, "metric": metric,
                              "comparison": f"full OEH minus {comparator}",
                              "delta": float(by_task.mean()),
                              "ci95_low_descriptive": float(np.quantile(samples, .025)),
                              "ci95_high_descriptive": float(np.quantile(samples, .975)),
                              "bootstrap_unit": "published MDP; five behaviors stay together",
                              "inference_status": "post_outcome_exploratory"})
    tq = verified_scores(TQ, ["tq_original", "saco_original"])
    eri_eh = verified_scores(ERI_EH, ["eri_original", "eri_defined",
                                      "eh_author_epw", "eh_author_k_min"])
    full = scored[scored.method == full_name].merge(
        tq, on=["mdp_id", "policy_family"], validate="many_to_one")
    full = full.merge(eri_eh, on=["mdp_id", "policy_family"], validate="many_to_one")
    if len(full) != 210 or full[KEYS].duplicated().any():
        raise RuntimeError("Original score join did not preserve 210 common rows")
    rank_rows = []
    rank_rng = np.random.default_rng(20261010)
    original_scores = (
        ("TQ", "tq_original", 1, "privileged_true_MDP_online_reference"),
        ("SACo", "saco_original", 1, "privileged_true_MDP_online_reference"),
        ("ERI", "eri_original", 1, "pilot_only_conditional_positive_mean_return"),
        ("EH author EPW", "eh_author_epw", -1, "privileged_author_true_MDP_summary"),
        ("EH author k_min", "eh_author_k_min", -1, "privileged_author_true_MDP_summary"),
    )
    for learner, frame in full.groupby("algorithm"):
        for name, column, sign, access in original_scores:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            eligible = np.isfinite(values)
            if name == "ERI":
                eligible &= frame.eri_defined.to_numpy(bool)
            valid = frame.loc[eligible]
            if not len(valid):
                raise RuntimeError(f"No defined rows for {name}: {learner}")
            original_easy = sign * values[eligible]
            oeh_easy = -valid.expected_event.to_numpy(float)
            observed_easy = -valid.event.to_numpy(float)
            original_rho = correlation(original_easy, observed_easy)
            oeh_rho = correlation(oeh_easy, observed_easy)
            tasks = sorted(valid.mdp_id.unique())
            pieces = [np.flatnonzero(valid.mdp_id.to_numpy() == task) for task in tasks]
            replicates = []
            for _ in range(2000):
                draw = rank_rng.integers(len(pieces), size=len(pieces))
                idx = np.concatenate([pieces[int(i)] for i in draw])
                a = correlation(oeh_easy[idx], observed_easy[idx])
                b = correlation(original_easy[idx], observed_easy[idx])
                if np.isfinite(a) and np.isfinite(b):
                    replicates.append(a - b)
            rank_rows.append({
                "algorithm": learner, "original_score": name,
                "rows": len(valid), "tasks": len(tasks),
                "eligible_fraction": len(valid) / len(frame),
                "original_spearman_easier": original_rho,
                "full_oeh_spearman_easier_same_rows": oeh_rho,
                "delta_oeh_minus_original": oeh_rho - original_rho,
                "ci95_low_descriptive": float(np.quantile(replicates, .025)) if replicates else np.nan,
                "ci95_high_descriptive": float(np.quantile(replicates, .975)) if replicates else np.nan,
                "valid_bootstrap_draws": len(replicates),
                "bootstrap_unit": "published MDP; five behaviors stay together",
                "score_direction": "higher is easier" if sign == 1 else "lower is easier",
                "access": access,
                "inference_status": "post_outcome_exploratory_different_original_endpoint",
            })
    OUT.mkdir(parents=True)
    scored.to_csv(OUT / "scored.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUT / "paired_task_bootstrap.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(OUT / "original_score_rank.csv", index=False)
    manifest = {"status": "COMPLETE_POST_OUTCOME_EXPLORATORY_14_MINIGRID",
                "tasks": 14, "rows_per_method": 210,
                "labels_already_opened_before_full_oeh_run": True,
                "metrics": ["integrated_brier", "seven_class_log_loss", "same_row_original_scalar_rank"],
                "rank_warning": "Original scalar scores have different endpoints and information access; ERI direction is descriptive, not a prediction that larger ERI implies less offline data. No scalar score has seven-class Brier/log loss.",
                "source_sha256": {
                    "full": sha256(FULL / "manifest.json"),
                    "b2": sha256(B2 / "manifest.json"),
                    "reduced": sha256(REDUCED / "manifest.json"),
                    "labels": sha256(LABELS / "manifest.json"),
                    "tq_saco": sha256(TQ), "eri_eh": sha256(ERI_EH),
                    "script": sha256(Path(__file__)),
                },
                "output_sha256": {file: sha256(OUT / file) for file in
                                  ("scored.csv", "summary.csv", "paired_task_bootstrap.csv",
                                   "original_score_rank.csv")}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

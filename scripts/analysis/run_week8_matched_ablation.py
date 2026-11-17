"""Matched feature ablation on an opened OEH cohort (exploratory only).

Predictions are written before this process reads query labels. All arms use
the historical matched_forecast implementation and identical meta-training
rows, folds, learners, tuning rule and seven-class target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import run_oeh_confirmatory_20261004 as C
from scripts.analysis import oeh_final_forecast as F
from scripts.analysis import run_oeh_final_120 as R
from scripts.analysis.compare_difficulty import FEATURES_B2
from scripts.analysis.develop_cross_learner_context import with_context

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arms() -> dict[str, tuple[str, ...]]:
    full = tuple(R.CANDIDATE_VARIANTS["plus_prior_profile"])
    direct = set(F.OWN) | set(F.CONTEXT) | set(F.HORIZON[-2:])
    horizon = set(F.HORIZON[:11]) | set(F.PROFILE)
    legacy = (set(R.OEH_FEATURES) - set(FEATURES_B2)) | set(R.LC_CONTEXT)
    result = {
        "b2_matched": tuple(FEATURES_B2),
        "oeh_full_108": full,
        "oeh_without_direct_oehb": tuple(x for x in full if x not in direct),
        "oeh_without_oehh_risk": tuple(x for x in full if x not in horizon),
        "oeh_without_legacy_diagnostics": tuple(x for x in full if x not in legacy),
        "b2_plus_eri_adaptation": (*FEATURES_B2, "eri", "eri_eligible"),
        "b2_plus_bwd_adaptation": (*FEATURES_B2, "bwd"),
    }
    if len(full) != 108 or len(set(full)) != 108 or len(direct | horizon | legacy | set(FEATURES_B2)) != 108:
        raise RuntimeError("Feature lineage changed; review all ablation definitions")
    for name, fields in result.items():
        if len(fields) != len(set(fields)) or not set(FEATURES_B2).issubset(fields):
            raise RuntimeError(f"Invalid field set: {name}")
    return result


def query_features() -> pd.DataFrame:
    pilot = pd.read_csv(SOURCE / "pilot_features.csv")
    oeh = F.oeh_feature_rows(SOURCE / "oeh_b_groups", SOURCE / "horizon_preoutcome.csv")
    published = pd.read_csv(SOURCE / "published_preoutcome.csv")
    panel = pd.read_csv(SOURCE / "predictions_preoutcome.csv", usecols=[*KEYS, "panel"])
    panel = panel.drop_duplicates(KEYS)
    query = pilot.merge(oeh, on=KEYS, validate="one_to_one")
    query = query.merge(published, on=KEYS[:2], validate="many_to_one")
    query = query.merge(panel, on=KEYS, validate="one_to_one")
    query = with_context(query)
    if len(query) != 1800 or query[KEYS].duplicated().any() or query.mdp_id.nunique() != 120:
        raise RuntimeError("Incomplete query feature set")
    return query.sort_values(KEYS).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/week8/matched_ablation_exploratory_2026-10-07")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    seal = json.loads((SOURCE / "preoutcome_seal.json").read_text(encoding="utf-8"))
    for name in ("pilot_features.csv", "horizon_preoutcome.csv", "published_preoutcome.csv", "predictions_preoutcome.csv"):
        if sha256(SOURCE / name) != seal["artifact_sha256"][name]:
            raise RuntimeError(f"Preoutcome file changed: {name}")
    C.configure()
    doc = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    tag = doc["oeh_final"]["structure_tag"]
    start = time.perf_counter()
    train = R.meta_training(R.OPENED, tag)
    query = query_features()
    variants = arms()
    if len(train) != 10800 or train.mdp_id.nunique() != 720 or set(train.mdp_id) & set(query.mdp_id):
        raise RuntimeError("Meta-training split mismatch")
    required = set().union(*(set(fields) for fields in variants.values()))
    for label, frame in (("train", train), ("query", query)):
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"Missing {label} columns: {sorted(missing)}")
    output.mkdir(parents=True)
    lineage = {name: {"columns": list(fields), "column_count": len(fields)} for name, fields in variants.items()}
    (output / "feature_lineage.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    rows, run_times, weights = [], [], {}
    for name, fields in variants.items():
        before = time.perf_counter()
        probability, tuning = F.matched_forecast(train, query, fields)
        elapsed = time.perf_counter() - before
        prediction = query[[*KEYS, "panel"]].copy()
        prediction["method"] = name
        prediction[P] = probability
        rows.append(prediction)
        run_times.append({"method": name, "seconds": elapsed, "columns": len(fields)})
        weights[name] = tuning
        print(f"{name}: {elapsed:.1f}s, {len(fields)} columns", flush=True)
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(output / "predictions_before_label_read.csv", index=False)
    (output / "prediction_seal.json").write_text(json.dumps({
        "prediction_sha256": sha256(output / "predictions_before_label_read.csv"),
        "labels_read": False, "query_source_sha256": {name: sha256(SOURCE / name) for name in
                                                   ("pilot_features.csv", "horizon_preoutcome.csv", "published_preoutcome.csv")},
        "total_prelabel_seconds": time.perf_counter() - start,
    }, indent=2) + "\n", encoding="utf-8")

    # Only the exploratory scoring phase below reads query labels.
    result = json.loads((SOURCE / "result_manifest.json").read_text(encoding="utf-8"))
    if sha256(SOURCE / "labels.csv") != result["artifact_sha256"]["labels.csv"]:
        raise RuntimeError("Query labels changed")
    labels = pd.read_csv(SOURCE / "labels.csv")[[*KEYS, "panel", "event"]]
    scored = predictions.merge(labels, on=[*KEYS, "panel"], validate="many_to_one")
    p = scored[P].to_numpy(float)
    y = scored.event.to_numpy(int)
    if len(scored) != 1800 * len(variants) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
        raise RuntimeError("Invalid seven-class panel")
    cdf = np.cumsum(p, axis=1)[:, :6]
    scored["brier"] = ((cdf - (y[:, None] <= np.arange(6))) ** 2).mean(1)
    scored["log_loss"] = -np.log(np.maximum(p[np.arange(len(p)), y], 1e-12))
    scored.to_csv(output / "scored.csv", index=False)
    summary = (scored.groupby(["panel", "algorithm", "method"], observed=True)
               .agg(rows=("event", "size"), mdps=("mdp_id", "nunique"), brier=("brier", "mean"),
                    log_loss=("log_loss", "mean"), censored_fraction=("event", lambda z: (z == 6).mean()))
               .reset_index())
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(run_times).to_csv(output / "runtime.csv", index=False)
    (output / "tuning.json").write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "status": "complete_exploratory_opened_cohort", "source_sha256": {
            "protocol.json": sha256(SOURCE / "protocol.json"),
            "preoutcome_seal.json": sha256(SOURCE / "preoutcome_seal.json"),
            "result_manifest.json": sha256(SOURCE / "result_manifest.json")},
        "script_sha256": sha256(Path(__file__)), "matched_forecast_sha256": sha256(Path(F.__file__)),
        "meta_training_rows": len(train), "meta_training_mdps": train.mdp_id.nunique(),
        "query_rows": len(query), "query_mdps": query.mdp_id.nunique(),
        "label": "exploratory: seventh cohort already opened; no new confirmatory claim",
        "output_sha256": {name: sha256(output / name) for name in
                          ("feature_lineage.json", "predictions_before_label_read.csv", "prediction_seal.json",
                           "scored.csv", "summary.csv", "runtime.csv", "tuning.json")},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Exploratory ablation complete: {output.relative_to(ROOT)}", flush=True)
    print(summary[summary.panel == "primary"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

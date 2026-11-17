"""Opened second-cohort B2+oracle-EH AFT transfer diagnostic.

Reconstructs oracle features from the original pinned generator. The cohort
has already been opened for development; no independent claim is made.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
import xgboost as xgb

from scripts.analysis.compare_difficulty import GRID, METHODS, aft_interval
from scripts.analysis.develop_b2_eh_oracle_aft import (
    E1_FEATURES, E1_LABELS, E1_ORACLE, EH, FIELDS, KEYS, P, ROOT, sha256,
)
from scripts.analysis.develop_oeh_difficulty import score_probabilities
from scripts.analysis.e1_oracle_qvi_diagnostic import finite_horizon_online_bounds
from scripts.analysis.independent_difficulty import XGB_PARAMS
from scripts.analysis.run_unseen_validation import archive_imports


DATASET = ROOT / "datasets/oeh_difficulty_confirmation_2026-10-01/manifest.json"
FEATURES = ROOT / "artifacts/oeh_difficulty_confirmation_2026-10-01/pilot_features.csv"
LABELS = ROOT / "artifacts/oeh_difficulty_confirmation_2026-10-01/labels.csv"
PREDICTIONS = ROOT / "artifacts/oeh_difficulty_confirmation_2026-10-01/predictions_preoutcome.csv"
OUTPUT = ROOT / "artifacts/oeh_b2_eh_oracle_second_development_2026-10-01"


def reconstruct_oracle() -> pd.DataFrame:
    _, _, generator, _ = archive_imports()
    from environments.benchmark_environments import make_benchmark_mdp

    manifest = json.loads(DATASET.read_text(encoding="utf-8"))
    e1 = json.loads((ROOT / "datasets/week6_e1_oehv4e_confirmatory_v1_1/manifest.json").read_text(encoding="utf-8"))
    if len(manifest["instances"]) != 60 or manifest["generator_provenance"]["source_sha256"] != e1["generator_provenance"]["source_sha256"]:
        raise RuntimeError("New cohort or source provenance mismatch")
    e1_oracle = pd.read_csv(E1_ORACLE)
    prior_oracle = pd.read_csv(ROOT / "artifacts/independent_difficulty_2026-09-30/oracle_preoutcome.csv")
    fingerprints = dict(zip(e1_oracle.kernel_fingerprint_reconstructed, e1_oracle.mdp_id))
    fingerprints.update(zip(prior_oracle.kernel_fingerprint, prior_oracle.mdp_id))
    prior_fingerprints = set(fingerprints)
    duplicate_kernels = []
    rows = []
    for instance in sorted(manifest["instances"], key=lambda item: item["instance_id"]):
        mdp = make_benchmark_mdp(type("Spec", (), instance)())
        dynamic, optimum, random = generator.dynamic_range(mdp)
        for key, value in (("dynamic_range", dynamic), ("optimal_return", optimum),
                           ("random_return", random)):
            if not np.isclose(value, float(instance[key]), rtol=0, atol=1e-10):
                raise RuntimeError(f"Oracle audit mismatch: {instance['instance_id']}/{key}")
        fingerprint = hashlib.sha256()
        for array in (mdp.P, mdp.R, mdp.initial_state_distribution):
            fingerprint.update(array.tobytes())
        fingerprint.update(json.dumps(list(mdp.terminal_states)).encode("ascii"))
        digest = fingerprint.hexdigest()
        if digest in prior_fingerprints:
            raise RuntimeError(f"Duplicate MDP kernel across cohorts: {instance['instance_id']} ({instance['family']}) matches {fingerprints[digest]}")
        if digest in fingerprints:
            duplicate_kernels.append((instance["instance_id"], fingerprints[digest]))
        else:
            fingerprints[digest] = instance["instance_id"]
        measured = finite_horizon_online_bounds(mdp)
        rows.append({"mdp_id": instance["instance_id"], "panel":
                     "control" if instance["family"] == "lock_structured" else "primary",
                     "kernel_fingerprint": digest, **{field: measured[field] for field in EH}})
        if len(rows) % 20 == 0:
            print(f"Reconstructed oracle EH descriptors: {len(rows)}/60", flush=True)
    if len(rows) != 60:
        raise RuntimeError("Oracle reconstruction incomplete")
    if duplicate_kernels:
        print(f"Within-cohort duplicate kernels: {duplicate_kernels}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {OUTPUT}")
    oracle = reconstruct_oracle()
    source = pd.read_csv(E1_LABELS).merge(pd.read_csv(E1_FEATURES), on=KEYS,
                                           validate="one_to_one")
    source = source.merge(pd.read_csv(E1_ORACLE)[["mdp_id", *EH]], on="mdp_id",
                          validate="many_to_one")
    query = pd.read_csv(LABELS).merge(pd.read_csv(FEATURES), on=KEYS,
                                       validate="one_to_one")
    if query[["mdp_id", "panel"]].drop_duplicates().merge(
            oracle[["mdp_id", "panel"]], on=["mdp_id", "panel"],
            validate="one_to_one").shape[0] != 60:
        raise RuntimeError("Oracle panel identities do not match labels")
    query = query.merge(oracle[["mdp_id", *EH]], on="mdp_id", validate="many_to_one")
    if len(source) != 1800 or len(query) != 900:
        raise RuntimeError("E1 or second-cohort labels incomplete")
    rows = []
    for panel in ("primary", "control"):
        for learner in METHODS:
            tr = source[(source.panel == panel) & (source.algorithm == learner)]
            qu = query[(query.panel == panel) & (query.algorithm == learner)]
            if len(tr) != (420 if panel == "primary" else 180) or len(qu) != (210 if panel == "primary" else 90):
                raise RuntimeError("Wrong panel/learner size")
            x = tr[list(FIELDS)].to_numpy(dtype=float)
            q = qu[list(FIELDS)].to_numpy(dtype=float)
            x[~np.isfinite(x)] = np.nan
            q[~np.isfinite(q)] = np.nan
            lower, upper = np.asarray([aft_interval(int(value)) for value in tr.event], dtype=float).T
            matrix = xgb.DMatrix(x)
            matrix.set_float_info("label_lower_bound", lower)
            matrix.set_float_info("label_upper_bound", upper)
            model = xgb.train(XGB_PARAMS, matrix, num_boost_round=200)
            median = model.predict(xgb.DMatrix(q))
            if not np.isfinite(median).all() or (median <= 0).any():
                raise RuntimeError("Invalid EH AFT median")
            cdf = ndtr((np.log(np.asarray(GRID))[None, :] - np.log(median)[:, None]) / 1.2)
            p = np.maximum(np.column_stack((cdf[:, 0], np.diff(cdf, axis=1), 1 - cdf[:, -1])), 0)
            p /= p.sum(axis=1, keepdims=True)
            brier, logloss = score_probabilities(p, qu.event.to_numpy(dtype=int))
            frame = qu[KEYS + ["panel", "event"]].copy()
            frame["method"] = "b2_plus_eh_oracle_aft"
            frame["brier"] = brier
            frame["logloss"] = logloss
            for index, name in enumerate(P):
                frame[name] = p[:, index]
            rows.append(frame)
            print(f"Fitted oracle EH AFT: {panel}/{learner}", flush=True)
    oracle_forecast = pd.concat(rows, ignore_index=True)
    saved = pd.read_csv(PREDICTIONS)
    saved = saved[saved.method == "b2_plus_oeh"]
    joined = query[KEYS + ["panel", "event"]].merge(saved[KEYS + P], on=KEYS,
                                                       validate="one_to_one")
    if len(joined) != 900:
        raise RuntimeError("Frozen B2+OEH prediction count mismatch")
    brier, logloss = score_probabilities(joined[P].to_numpy(dtype=float),
                                         joined.event.to_numpy(dtype=int))
    joined["method"] = "b2_plus_oeh_aft"
    joined["brier"] = brier
    joined["logloss"] = logloss
    scored = pd.concat([oracle_forecast, joined], ignore_index=True)
    summary = scored.groupby(["panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    oracle.to_csv(OUTPUT / "oracle_descriptors.csv", index=False)
    scored.to_csv(OUTPUT / "scored_predictions.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    manifest = {
        "status": "development_only_second_open_cohort_not_confirmation",
        "scope": "Same AFT, B2+full-MDP oracle EH versus B2+pilot-only OEH; offline target transfer only",
        "eh_features": EH,
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in (DATASET, E1_LABELS, E1_FEATURES, E1_ORACLE,
                                       FEATURES, LABELS, PREDICTIONS)},
        "script_sha256": sha256(Path(__file__)),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

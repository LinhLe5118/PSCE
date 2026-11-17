"""Conditional fresh 120-MDP test of cross-learner depth-3 OEH.

This runner may be frozen only if the second new 120-MDP test is complete and
NO-GO. That second cohort is never used for meta-training. E1, both opened
60-MDP development cohorts, and the failed first 120-MDP cohort provide labels.
The third 120-MDP cohort is entirely new and remains unseen until its forecasts
have been sealed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import run_oeh_120_unseen as base
from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS
from scripts.analysis.compare_e1_dataset_signals import SIGNALS
from scripts.analysis.develop_b2_eh_oracle_aft import E1_FEATURES, E1_LABELS, E1_ORACLE, EH, FIELDS, KEYS, P, ROOT, sha256
from scripts.analysis.develop_cross_learner_context import CONTEXT, with_context
from scripts.analysis.develop_eh_oracle_three_way import aft_probabilities, numeric
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.develop_oeh_difficulty import fit_thresholds
from scripts.analysis.develop_tree_capacity import VARIANTS, multiclass
from scripts.analysis.run_unseen_validation import ARCHIVE, archive_imports, load_json, write_json


OUTPUT = ROOT / "artifacts/oeh_cross_capacity_unseen_120_2026-10-01"
DATASET = ROOT / "datasets/oeh_cross_capacity_unseen_120_2026-10-01"
FIRST_OUTPUT = ROOT / "artifacts/oeh_difficulty_unseen_120_2026-10-01"
FIRST_DATASET = ROOT / "datasets/oeh_difficulty_unseen_120_2026-10-01"
SECOND_OUTPUT = ROOT / "artifacts/oeh_pooled_successor_unseen_120_2026-10-01"
SECOND_DATASET = ROOT / "datasets/oeh_pooled_successor_unseen_120_2026-10-01"
CROSS_GATE = ROOT / "artifacts/oeh_cross_capacity_gate_development_2026-10-01/manifest.json"
DEV = (
    (ROOT / "artifacts/independent_difficulty_2026-09-30/labels.csv",
     ROOT / "artifacts/independent_validation_2026-09-30/pilot_features.csv",
     ROOT / "artifacts/independent_difficulty_2026-09-30/oracle_preoutcome.csv"),
    (ROOT / "artifacts/oeh_difficulty_confirmation_2026-10-01/labels.csv",
     ROOT / "artifacts/oeh_difficulty_confirmation_2026-10-01/pilot_features.csv",
     ROOT / "artifacts/oeh_b2_eh_oracle_second_development_2026-10-01/oracle_descriptors.csv"),
)
NAMESPACE = "oeh-cross-capacity-third-new-120-v1-20261001"
CANDIDATE = "oeh_cross_capacity"

# The frozen first-run module implements dataset checks, the matched learner
# execution plan, label construction, scores, and joint 14-contrast analysis.
# This conditional variant changes its output namespace and preoutcome training.
base.OUTPUT = OUTPUT
base.DATASET = DATASET
base.CANDIDATE = CANDIDATE
base.FORECAST_METHODS = (*base.COMPARATORS, CANDIDATE)


def training() -> pd.DataFrame:
    e1 = pd.read_csv(E1_LABELS).merge(pd.read_csv(E1_FEATURES), on=KEYS,
                                      validate="one_to_one")
    e1 = e1.merge(pd.read_csv(E1_ORACLE)[["mdp_id", *EH]], on="mdp_id",
                  validate="many_to_one")
    parts = [e1]
    for labels, features, oracle in DEV:
        frame = pd.read_csv(labels).merge(pd.read_csv(features), on=KEYS,
                                          validate="one_to_one")
        frame = frame.merge(pd.read_csv(oracle)[["mdp_id", *EH]], on="mdp_id",
                            validate="many_to_one")
        if len(frame) != 900 or frame.mdp_id.nunique() != 60:
            raise RuntimeError(f"Incomplete opened development cohort: {labels}")
        parts.append(frame)
    failed = pd.read_csv(FIRST_OUTPUT / "labels.csv").merge(
        pd.read_csv(FIRST_OUTPUT / "pilot_features.csv"), on=KEYS,
        validate="one_to_one")
    failed = failed.merge(pd.read_csv(FIRST_OUTPUT / "oracle_descriptors_preoutcome.csv")[["mdp_id", *EH]],
                          on="mdp_id", validate="many_to_one")
    if len(failed) != 1800 or failed.mdp_id.nunique() != 120:
        raise RuntimeError("First opened failed cohort incomplete")
    parts.append(failed)
    if len(e1) != 1800 or e1.mdp_id.nunique() != 120:
        raise RuntimeError("Historical E1 training cohort incomplete")
    result = pd.concat(parts, ignore_index=True)
    if len(result) != 5400 or result.mdp_id.nunique() != 360 or result[KEYS].duplicated().any():
        raise RuntimeError("Expanded meta-training cohort overlap or missing rows")
    return with_context(result)


def freeze() -> None:
    if OUTPUT.exists() or DATASET.exists():
        raise FileExistsError("Third-test namespace already exists")
    second = load_json(SECOND_OUTPUT / "result_manifest.json")
    if second["status"] != "complete" or second["difficulty_gate"] != "NO-GO":
        raise RuntimeError("Third test is conditional on completed second NO-GO")
    if sha256(SECOND_OUTPUT / "protocol.json") != second["protocol_sha256"]:
        raise RuntimeError("Second test protocol seal changed")
    gate = load_json(CROSS_GATE)
    if gate["development_gate"] != "GO" or gate["mdps"] != 168:
        raise RuntimeError("Cross-capacity development gate did not pass")
    training()
    archive_imports()
    from environments.benchmark_environments import benchmark_specs, make_benchmark_mdp

    e1_manifest = load_json(base.E1_MANIFEST)
    e1_seal = load_json(ROOT / "artifacts/week6/E1_INPUT_SEAL_2026-09-26.json")
    if sha256(base.E1_MANIFEST) != e1_seal["manifest_sha256"]:
        raise RuntimeError("E1 input seal mismatch")
    seen = (base.prior_kernels()
            | set(load_json(FIRST_OUTPUT / "protocol.json")["kernel_fingerprints"].values())
            | set(load_json(SECOND_OUTPUT / "protocol.json")["kernel_fingerprints"].values()))
    used_seeds = set()
    for path in (base.E1_MANIFEST,
                 ROOT / "datasets/independent_validation_2026-09-30/manifest.json",
                 ROOT / "datasets/oeh_difficulty_confirmation_2026-10-01/manifest.json",
                 FIRST_DATASET / "manifest.json",
                 SECOND_DATASET / "manifest.json"):
        used_seeds.update(int(row["seed"]) for row in load_json(path)["instances"])
    specs, kernels = [], {}
    templates = [row for row in benchmark_specs() if row.split == "evaluation"]
    for family, count in base.FAMILY_COUNTS.items():
        selected = sorted((row for row in templates if row.family == family),
                          key=lambda row: row.ordinal)
        if len(selected) != count:
            raise RuntimeError("Historical recipe counts changed")
        for ordinal, template in enumerate(selected):
            instance_id = f"oehcross120_{family}_{ordinal:03d}"
            for retry in range(100):
                digest = hashlib.sha256(f"{NAMESPACE}|{template.instance_id}|{retry}".encode()).digest()
                seed = 100_000_000 + int.from_bytes(digest[:8], "little") % 1_000_000_000
                if seed in used_seeds:
                    continue
                spec = replace(template, instance_id=instance_id, seed=seed)
                kernel = base.fingerprint(make_benchmark_mdp(spec))
                if kernel not in seen:
                    break
            else:
                raise RuntimeError(f"No fresh seed for {instance_id}")
            used_seeds.add(seed)
            seen.add(kernel)
            specs.append(spec)
            kernels[instance_id] = kernel
    if len(specs) != 120 or len(kernels) != 120:
        raise RuntimeError("Incomplete third cohort")
    old_weights = load_json(FIRST_OUTPUT / "protocol.json")["weights"]
    weights = {**old_weights, CANDIDATE: old_weights["oeh_three_way"]}
    sources = (Path(__file__), Path(base.__file__), E1_LABELS, E1_FEATURES, E1_ORACLE,
               base.E1_MANIFEST, CROSS_GATE, FIRST_OUTPUT / "result_manifest.json",
               FIRST_OUTPUT / "protocol.json", FIRST_DATASET / "manifest.json",
               FIRST_OUTPUT / "labels.csv", FIRST_OUTPUT / "pilot_features.csv",
               FIRST_OUTPUT / "oracle_descriptors_preoutcome.csv",
               SECOND_OUTPUT / "result_manifest.json",
               SECOND_OUTPUT / "protocol.json", SECOND_DATASET / "manifest.json",
               *[path for cohort in DEV for path in cohort],
               ROOT / "scripts/analysis/develop_oeh_difficulty.py",
               ROOT / "scripts/analysis/develop_cross_learner_context.py",
               ROOT / "scripts/analysis/develop_tree_capacity.py",
               ROOT / "scripts/analysis/develop_eh_oracle_three_way.py",
               ARCHIVE / "evaluation/e1_pipeline.py",
               ARCHIVE / "scripts/data/generate_offline_datasets.py",
               ARCHIVE / "environments/benchmark_environments.py")
    doc = {
        "schema": "oeh-unseen-120-difficulty-v1",
        "status": "frozen_before_dataset_buffers_or_learner_outcomes",
        "namespace": NAMESPACE,
        "requested_specs": [spec.to_dict() for spec in specs],
        "kernel_fingerprints": kernels,
        "family_counts": base.FAMILY_COUNTS,
        "stochastic_primary_mdps": 84, "control_mdps": 36,
        "pilot_multipliers": [2, 5, 10], "budgets": base.GRID,
        "training_seed_indices": list(range(10)), "behavior_families": 5,
        "core_learners": METHODS, "expected_jobs": base.EXPECTED_JOBS,
        "target": "first complete suffix with >=8/10 seeds normalized return >=0.95; else >x50",
        "candidate": base.CANDIDATE, "comparators": base.COMPARATORS,
        "forecast_architecture": "matched AFT/direct-threshold/depth-3 multiclass; E1, two 60-MDP development cohorts, and failed first 120-MDP cohort for meta-training; OEH adds other learners' pilot context",
        "weights": weights,
        "multiclass_capacity": VARIANTS["depth3_200"],
        "decision_rule": "on 84 stochastic MDPs: OEH point relative gains >=5% in integrated Brier and seven-class log loss versus every comparator, and all simultaneous one-sided 95% MDP-cluster absolute-improvement lower bounds >0",
        "bootstrap": {"draws": 10000, "seed": 20261003,
                      "joint_contrasts": 2 * len(base.COMPARATORS)},
        "control_rule": "report 36 control MDPs separately; never select candidate on them",
        "scope": "new untouched kernels; expanded training excludes second 120-MDP test; EH is an oracle descriptor transfer, not the published online theorem",
        "original_generator_sha256": e1_manifest["generator_provenance"]["source_sha256"],
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in sources},
    }
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "protocol.json", doc)
    (OUTPUT / "protocol.sha256").write_text(sha256(OUTPUT / "protocol.json") + "\n", encoding="ascii")
    print("Frozen conditional cross-capacity OEH on 120 new requested kernels", flush=True)


def prepare() -> None:
    if (OUTPUT / "preoutcome_seal.json").exists():
        raise FileExistsError("Successor forecasts already sealed")
    doc = base.protocol()
    if DATASET.exists():
        manifest = base.dataset()
    else:
        _, _, generator, _ = archive_imports()
        from environments.benchmark_environments import BenchmarkSpec, make_benchmark_mdp
        specs = [BenchmarkSpec(**item) for item in doc["requested_specs"]]
        manifest = generator.generate_benchmark_bundle(
            DATASET, specs=specs, pilot_seeds=(0,), training_seeds=tuple(range(10)),
            max_size=None, horizon=100, run_mode="test",
            deviation_reason="Conditional cross-capacity OEH third fresh 120-MDP test",
            pilot_size_multipliers=(2, 5, 10))
        if manifest["instance_count"] != 120 or manifest["physical_buffer_count"] != 6600:
            raise RuntimeError("Incomplete third dataset")
        old = (base.prior_kernels()
               | set(load_json(FIRST_OUTPUT / "protocol.json")["kernel_fingerprints"].values())
               | set(load_json(SECOND_OUTPUT / "protocol.json")["kernel_fingerprints"].values()))
        actual = {}
        for item in manifest["instances"]:
            mdp = make_benchmark_mdp(type("Spec", (), item)())
            actual[item["instance_id"]] = base.fingerprint(mdp)
        if len(actual) != 120 or len(set(actual.values())) != 120 or set(actual.values()) & old:
            raise RuntimeError("Resolved third kernels are not all new")
        changed = {name: [doc["kernel_fingerprints"][name], digest]
                   for name, digest in actual.items()
                   if doc["kernel_fingerprints"][name] != digest}
        if changed:
            (OUTPUT / "protocol_before_resolved_kernel_amendment.json").write_bytes(
                (OUTPUT / "protocol.json").read_bytes())
            previous = sha256(OUTPUT / "protocol.json")
            doc["kernel_fingerprints"] = actual
            doc["preoutcome_resolved_kernel_amendment"] = {
                "changed": changed, "old_protocol_sha256": previous,
                "learner_outcomes_observed": 0,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json(OUTPUT / "protocol.json", doc)
            (OUTPUT / "protocol.sha256").write_text(sha256(OUTPUT / "protocol.json") + "\n", encoding="ascii")
        write_json(OUTPUT / "actual_kernel_audit_preoutcome.json", {
            "all_120_unique_and_new": True, "changed_requested_seed_count": len(changed),
            "actual_kernel_fingerprints": actual,
            "learner_outcomes_observed": 0})
        paths = (DATASET / "manifest.json", DATASET / "data_only_index.json")
        write_json(OUTPUT / "dataset_seal.json", {
            "protocol_sha256": sha256(OUTPUT / "protocol.json"),
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                       for path in paths},
        })
        doc = base.protocol()
    query = base.pilot_features(manifest)
    oracle = base.oracle_descriptors(manifest, doc)
    train = training()
    query = query.merge(oracle[["mdp_id", *EH]], on="mdp_id", validate="many_to_one")
    family = {item["instance_id"]: item["family"] for item in manifest["instances"]}
    query["panel"] = query.mdp_id.map(
        lambda item: "control" if family[item] == "lock_structured" else "primary")
    query = with_context(query)
    if (len(query) != 1800 or query.mdp_id.nunique() != 120
            or set(train.mdp_id) & set(query.mdp_id)):
        raise RuntimeError("Third training/query overlap or incomplete pilot")
    field_sets = {"b2_three_way": FEATURES_B2,
                  CANDIDATE: (*OEH_FEATURES, *CONTEXT),
                  "b2_plus_eh_oracle_three_way": FIELDS}
    forecasts = []
    for panel in ("primary", "control"):
        for learner in METHODS:
            source = train[(train.panel == panel) & (train.algorithm == learner)]
            selected = query[(query.panel == panel) & (query.algorithm == learner)]
            if len(source) != (1260 if panel == "primary" else 540) or len(selected) != (420 if panel == "primary" else 180):
                raise RuntimeError("Wrong expanded training/query stratum")
            y = source.event.to_numpy(dtype=int)
            probability = {"train_frequency": np.repeat(
                (np.bincount(y, minlength=7) / len(y))[None, :], len(selected), axis=0)}
            for name, fields in field_sets.items():
                x = numeric(source) if name == "b2_plus_eh_oracle_three_way" else np.nan_to_num(
                    source[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
                q = numeric(selected) if name == "b2_plus_eh_oracle_three_way" else np.nan_to_num(
                    selected[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
                parts = np.asarray((aft_probabilities(x, y, q),
                                    fit_thresholds(x, y, q),
                                    multiclass(x, y, q, doc["multiclass_capacity"])))
                probability[name] = np.einsum("i,ijk->jk", doc["weights"][name], parts)
                if name == "b2_three_way":
                    probability["b2_generic"] = parts[0]
            for name in ("coverage_adaptation", "quality_adaptation", "coverage_quality_adaptation"):
                fields = SIGNALS[name]
                x = np.nan_to_num(source[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
                q = np.nan_to_num(selected[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
                probability[name] = aft_probabilities(x, y, q)
            for name in base.FORECAST_METHODS:
                p = probability[name]
                if (p.shape != (len(selected), 7) or not np.isfinite(p).all()
                        or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.)):
                    raise RuntimeError(f"Invalid third forecast: {name}")
                frame = selected[KEYS + ["panel"]].copy()
                frame["method"] = name
                for i, col in enumerate(P):
                    frame[col] = p[:, i]
                forecasts.append(frame)
            print(f"Cross-capacity forecasts: {panel}/{learner}", flush=True)
    predictions = pd.concat(forecasts, ignore_index=True)
    if len(predictions) != 1800 * len(base.FORECAST_METHODS) or predictions.duplicated(KEYS + ["method"]).any():
        raise RuntimeError("Incomplete third forecasts")
    predictions.to_csv(OUTPUT / "predictions_preoutcome.csv", index=False)
    write_json(OUTPUT / "preoutcome_seal.json", {
        "protocol_sha256": sha256(OUTPUT / "protocol.json"),
        "dataset_seal_sha256": sha256(OUTPUT / "dataset_seal.json"),
        "learner_outcomes_observed": 0,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {name: sha256(OUTPUT / name) for name in
                            ("pilot_features.csv", "oracle_descriptors_preoutcome.csv",
                             "predictions_preoutcome.csv")},
    })
    print("Sealed 14,400 cross-capacity forecasts before any new learner outcome", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "evaluate", "analyze"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    {"freeze": freeze, "prepare": prepare,
     "evaluate": lambda: base.evaluate(args.resume),
     "analyze": base.analyze}[args.phase]()


if __name__ == "__main__":
    main()

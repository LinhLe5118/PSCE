"""Freeze an untouched 60-MDP validation cohort before any new outcome is read."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from scripts.analysis.compare_difficulty import sha256
from scripts.analysis.e1_oracle_qvi_diagnostic import archived_source_hash


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "tmp/oeh_development_2026-09-30"
OUTPUT = ROOT / "artifacts/independent_validation_2026-09-30"
FAMILY_COUNTS = {"uniform_garnet": 15, "forward_layered": 18,
                 "bottleneck": 9, "lock_structured": 18}
NAMESPACE = "oeh-unseen-validation-v1-20260930"


def select_specs(specs):
    """Select half of each E1 recipe family without looking at outcomes."""
    from dataclasses import replace

    selected = []
    for family, count in FAMILY_COUNTS.items():
        available = [spec for spec in specs if spec.split == "evaluation" and spec.family == family]
        ordered = sorted(available, key=lambda spec: hashlib.sha256(
            f"{NAMESPACE}|select|{spec.instance_id}".encode("utf-8")
        ).hexdigest())
        if len(available) != 2 * count:
            raise RuntimeError(f"Unexpected E1 recipe count: {family}")
        for index, template in enumerate(sorted(ordered[:count], key=lambda item: item.ordinal)):
            digest = hashlib.sha256(f"{NAMESPACE}|seed|{template.instance_id}".encode("utf-8")).digest()
            seed = 100_000_000 + int.from_bytes(digest[:8], "little") % 1_000_000_000
            selected.append(replace(template, instance_id=f"validation_{family}_{index:03d}", seed=seed))
    if len(selected) != 60 or len({spec.seed for spec in selected}) != 60:
        raise RuntimeError("Duplicate or incomplete validation cohort")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Validation protocol already exists: {output}")

    e1_manifest_path = ROOT / "datasets/week6_e1_oehv4e_confirmatory_v1_1/manifest.json"
    e1_manifest = json.loads(e1_manifest_path.read_text(encoding="utf-8"))
    seal = json.loads((ROOT / "artifacts/week6/E1_INPUT_SEAL_2026-09-26.json").read_text(encoding="utf-8"))
    if sha256(e1_manifest_path) != seal["manifest_sha256"]:
        raise RuntimeError("Historical E1 input seal mismatch")
    if archived_source_hash(ARCHIVE, e1_manifest["generator_provenance"]["source_files"]) != e1_manifest["generator_provenance"]["source_sha256"]:
        raise RuntimeError("Historical generator source mismatch")
    sys.path.insert(0, str(ARCHIVE))
    from environments.benchmark_environments import benchmark_specs

    specs = select_specs(benchmark_specs())
    e1_seeds = {record["seed"] for record in e1_manifest["instances"]}
    if e1_seeds & {spec.seed for spec in specs}:
        raise RuntimeError("New seed aliases E1")
    files = [
        ROOT / "code/oeh/unseen.py", ROOT / "code/oeh/core.py", ROOT / "code/oeh/router.py",
        ROOT / "artifacts/week6/oehv4e_confirmatory_v1_1/core_run_results.csv",
        ROOT / "artifacts/week6/oehv4e_confirmatory_v1_1/external_run_results.csv",
        ROOT / "artifacts/week6/oehv4e_confirmatory_v1_1/pilot_features/features.csv",
        ARCHIVE / "evaluation/e1_pipeline.py", ARCHIVE / "evaluation/e1_external.py",
        ARCHIVE / "scripts/data/generate_week6_e1_bundle.py",
        ROOT / "artifacts/provenance/sources/evaluation/e1_reuse.py",
        ROOT / "scripts/analysis/run_unseen_validation.py",
        Path(__file__),
    ]
    document = {
        "schema": "oeh-independent-validation-v1",
        "status": "protocol_frozen_before_new_mdp_materialization_or_outcomes",
        "cohort": "60 new MDPs under original benchmark recipe distributions; not a new 120-MDP E1",
        "recipe_namespace": NAMESPACE,
        "requested_specs": [spec.to_dict() for spec in specs],
        "family_counts": FAMILY_COUNTS,
        "primary_families": ["uniform_garnet", "forward_layered", "bottleneck"],
        "control_family": "lock_structured",
        "mdp_count": 60, "primary_count": 42, "control_count": 18,
        "pilot_size_multipliers": [2, 5, 10], "training_size_multiplier": 10,
        "pilot_seed_indices": [0], "training_seed_indices": list(range(10)),
        "behavior_families": 5, "horizon": 100, "gamma": .95,
        "candidate_methods": ["cfqi_tabular", "iql_tabular", "behavior_cloning_tabular",
                              "pib_spibb_n10_author", "vi_lcb_cb005_author_empirical",
                              "discrete_bcq_author", "discrete_cql_d3rlpy"],
        "meta_training": "all sealed E1 x10 labels/pilot features; stochastic and control models fit separately",
        "query_information": "pilot-only features for three core anchors plus public predeclared benchmark family stratum; no new training/evaluation outcomes or oracle dynamics",
        "routing": "frozen quadratic paired forecasts, four grouped inner folds, original ridge grid, equal two-block ensemble and joint batch frontier; original fold-0 inner-CV salt fixed before validation",
        "primary_endpoints": ["mean_normalized_return", "seed_success_at_normalized_return_ge_0.95"],
        "primary_comparison": "OEH minus each of seven fixed learners on identical MDP-behavior-training seeds; all outcomes retained",
        "success_rule": "both endpoints must have positive simultaneous one-sided 95% whole-MDP bootstrap lower bounds against all seven fixed learners",
        "uncertainty": "2000 whole-MDP bootstrap draws, seed 20260921, one-sided max-error simultaneous lower bounds across 14 contrasts; no model refits on validation outcomes",
        "analysis_unit": "MDP; average 10 seeds within behavior, 5 behaviors within MDP, then average MDPs equally",
        "additional_targets_descriptive": ["OEH mean normalized return >0.8", "OEH success rate >0.5"],
        "no_retuning_or_interim_endpoint_access": True,
        "original_e1_manifest_sha256": seal["manifest_sha256"],
        "original_generator_source_sha256": e1_manifest["generator_provenance"]["source_sha256"],
        "runner_sha256": sha256(ROOT / "scripts/analysis/run_unseen_validation.py"),
        "input_source_sha256": {str(path.relative_to(ROOT)).replace('\\', '/'): sha256(path) for path in files},
    }
    output.mkdir(parents=True)
    path = output / "protocol.json"
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "protocol.sha256").write_text(sha256(path) + "\n", encoding="ascii")
    print(f"Frozen {len(specs)} fresh MDP recipes; protocol SHA-256 {sha256(path)}")


if __name__ == "__main__":
    main()

"""OEH v2 / M3: an extra meta-training cohort with three new logging-behavior families.

Development data (labels are used to fit and select OEH v2); it is never a test
panel. 168 fresh stochastic kernels (two per primary template of the registered
recipe, distinct from every earlier cohort including the eighth), each logged by
three behaviors that the five registered families do not cover:

* ``ratio_baseline``  softmax on Q* sharpened until J >= rand + (rho+1)/2 * range,
  then random states lose argmax mass until J <= rand + rho * range
  (the procedure of Laroche et al. 2019, implemented here, not their code);
* ``epsilon_star``    epsilon-greedy around pi*;
* ``random_mixture``  mixture of uniform and one random deterministic policy.

Parameters rotate deterministically over the MDP ordinal. Pilot x10, ten seeds,
BC/CFQI/IQL, the same label rule. Phases: ``freeze``, ``prepare``, ``evaluate``.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import run_oeh_eighth_cohort_20261014 as E

ROOT = E.ROOT
OUTPUT = ROOT / "artifacts/oeh_v2_behavior_cohort_2026-10-14"
DATASET = ROOT / "datasets/oeh_v2_behavior_cohort_2026-10-14"
NAMESPACE = "oeh-v2-behavior-meta-cohort-v1-20261014"
FAMILIES = ("ratio_baseline", "epsilon_star", "random_mixture")
RHO = (0.1, 0.3, 0.5, 0.7, 0.9)
EPSILON = (0.1, 0.3, 0.5, 0.7, 0.9)
WEIGHT = (0.2, 0.4, 0.6, 0.8, 1.0)
REPLICAS = 2


def _fresh_specs():
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    from environments.benchmark_environments import benchmark_specs, make_benchmark_mdp
    from scripts.analysis import run_oeh_confirmatory_20261004 as C
    from scripts.analysis import run_oeh_final_120 as R
    from scripts.analysis import run_oeh_120_unseen as base

    C.configure()
    seen = set(R.seen_kernels())
    seen |= set(E.load_json(E.COHORT7 / "actual_kernel_audit_preoutcome.json")["actual_kernel_fingerprints"].values())
    eighth = E.load_json(E.OUTPUT / "internal/dataset_seal.json")
    seen |= set(eighth["actual_kernel_fingerprints"].values())
    used = {int(item["seed"]) for item in E.load_json(E.COHORT7 / "protocol.json")["requested_specs"]}
    used |= {int(item["seed"]) for item in E.load_json(E.OUTPUT / "protocol.json")["panels"]["internal"]["requested_specs"]}
    for spec in R.OPENED.values():
        used |= {int(row["seed"]) for row in E.load_json(spec["dataset"] / "manifest.json")["instances"]}
    templates = [row for row in benchmark_specs() if row.split == "evaluation"]
    specs, kernels = [], {}
    for family, count in E.PRIMARY_COUNTS.items():
        selected = sorted((row for row in templates if row.family == family), key=lambda row: row.ordinal)
        for replica in range(REPLICAS):
            for ordinal, template in enumerate(selected):
                instance_id = f"v2b_{family}_{replica * count + ordinal:03d}"
                for retry in range(5000):
                    digest = hashlib.sha256(f"{NAMESPACE}|{template.instance_id}|{replica}|{retry}".encode()).digest()
                    seed = 100_000_000 + int.from_bytes(digest[:8], "little") % 1_000_000_000
                    if seed in used:
                        continue
                    spec = replace(template, instance_id=instance_id, seed=seed)
                    kernel = base.fingerprint(make_benchmark_mdp(spec))
                    if kernel not in seen:
                        break
                else:
                    raise RuntimeError(f"No fresh kernel for {instance_id}")
                used.add(seed)
                seen.add(kernel)
                specs.append(spec)
                kernels[instance_id] = kernel
    return specs, kernels


def freeze() -> None:
    if OUTPUT.exists() or DATASET.exists():
        raise FileExistsError("namespace exists")
    specs, kernels = _fresh_specs()
    doc = {"schema": "oeh-v2-behavior-meta-cohort-v1", "namespace": NAMESPACE, "frozen_at_utc": E.now(),
           "role": "development meta-training cohort for OEH v2 (M3); labels are fitted and used for selection",
           "mdps": len(specs), "behaviors": list(FAMILIES),
           "parameters": {"ratio_baseline_rho": RHO, "epsilon_star_epsilon": EPSILON, "random_mixture_weight": WEIGHT,
                          "rotation": "index = MDP ordinal mod 5"},
           "requested_specs": [s.to_dict() for s in specs], "kernel_fingerprints": kernels,
           "source_sha256": {E.rel(Path(__file__)): E.sha256(Path(__file__))}}
    OUTPUT.mkdir(parents=True)
    E.write_json(OUTPUT / "protocol.json", doc)
    (OUTPUT / "protocol.sha256").write_text(E.sha256(OUTPUT / "protocol.json") + "\n", encoding="ascii")
    print(f"frozen {len(specs)} MDPs {E.sha256(OUTPUT / 'protocol.json')}", flush=True)


def new_behaviors(mdp, *, seed: int, variant_index: int):
    import environments.behavior_policies as B
    from exact_solver import dynamic_range, evaluate_policy, value_iteration

    rng = np.random.default_rng(seed)
    k = variant_index % 5
    S, A = mdp.num_states, mdp.num_actions
    q = value_iteration(mdp).q_values
    star = q.argmax(1)
    _, j_star, j_rand = dynamic_range(mdp)
    span = j_star - j_rand

    def value(pi):
        return evaluate_policy(mdp, pi).expected_return

    rho = RHO[k]
    beta = 1.0
    pi = B._stable_softmax((1 - mdp.gamma) * q, 1.0 / beta)
    for _ in range(200):  # sharpen until the softmax reaches the intermediate target
        if value(pi) >= j_rand + (rho + 1) / 2 * span:
            break
        beta *= 1.5
        pi = B._stable_softmax((1 - mdp.gamma) * q, 1.0 / beta)
    for _ in range(20_000):  # perturb random states until the baseline target
        if value(pi) <= j_rand + rho * span:
            break
        s = rng.integers(S)
        pi[s, star[s]] *= 0.9
        pi[s] /= pi[s].sum()
    out = [B.BehaviorPolicy(family="ratio_baseline", probabilities=pi, parameters={"rho": rho, "beta": beta},
                            seed=seed + 11, uses_q_star=True)]
    eps = EPSILON[k]
    out.append(B.BehaviorPolicy(family="epsilon_star", probabilities=B._epsilon_soft(star, A, eps),
                                parameters={"epsilon": eps}, seed=seed + 12, uses_q_star=True))
    w = WEIGHT[k]
    det = np.eye(A)[rng.integers(0, A, size=S)]
    out.append(B.BehaviorPolicy(family="random_mixture", probabilities=w * np.full((S, A), 1.0 / A) + (1 - w) * det,
                                parameters={"weight_uniform": w}, seed=seed + 13, uses_q_star=False))
    return out


def prepare(workers: int) -> None:
    from scripts.analysis.run_unseen_validation import archive_imports
    from scripts.analysis import run_oeh_final_120 as R
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis.develop_cross_learner_context import with_context
    from scripts.analysis.develop_cohort8_candidates_20261014 import add_fields

    doc = E.load_json(OUTPUT / "protocol.json")
    if not (OUTPUT / "dataset_seal.json").exists():
        _, _, generator, _ = archive_imports()
        import environments.behavior_policies as B
        from environments.benchmark_environments import BenchmarkSpec
        B.POLICY_FAMILIES = (*B.POLICY_FAMILIES, *FAMILIES)
        generator.behavior_policy_suite = new_behaviors
        specs = [BenchmarkSpec(**item) for item in doc["requested_specs"]]
        manifest = generator.generate_benchmark_bundle(
            DATASET, specs=specs, pilot_seeds=(0,), training_seeds=tuple(range(10)), max_size=None, horizon=100,
            run_mode="test", deviation_reason="OEH v2 M3: extra behavior families", pilot_size_multipliers=(2, 5, 10))
        E.write_json(OUTPUT / "dataset_seal.json", {"generated_at_utc": E.now(), "instances": manifest["instance_count"],
                                                    "buffers": manifest["physical_buffer_count"],
                                                    "manifest_sha256": E.sha256(DATASET / "manifest.json")})
    R._repo_first()
    seven = E.load_json(E.COHORT7 / "protocol.json")["oeh_final"]
    target = OUTPUT / "inputs"
    target.mkdir(exist_ok=True)
    features = E.pilot_features(DATASET, target / "pilot_features.csv", workers)
    pub = E.published(DATASET, target / "published.csv", workers, None)
    R.run_oeh_b(DATASET, target / "oeh_b_groups", seven["structure"], seven["requirement_draws"], f"{NAMESPACE}", workers, "cpu")
    R.run_horizon(DATASET, target / "horizon.csv", seven["structure"], workers, oracle=False)
    curve = R.run_curve(DATASET, target / "curve.csv", seven["structure"], seven["requirement_draws"], workers)
    query = features.merge(F.oeh_feature_rows(target / "oeh_b_groups", target / "horizon.csv"), on=E.KEYS, validate="one_to_one")
    query = query.merge(R.curve_features(curve), on=E.KEYS, validate="one_to_one")
    query = query.merge(pub, on=["mdp_id", "policy_family"], validate="many_to_one")
    query = add_fields(with_context(query))
    query["panel"], query["cohort"] = "primary", "v2_behavior"
    query.to_csv(target / "query.csv.gz", index=False)
    print(f"query {len(query)} rows", flush=True)


def evaluate(workers: int) -> None:
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor
    from scripts.analysis.run_unseen_validation import archive_imports
    from scripts.analysis.compare_difficulty import suffix_event
    from scripts.analysis import run_oeh_final_120 as R

    _, _, _, seed_index = archive_imports()
    R._repo_first()
    manifest = E.load_json(DATASET / "manifest.json")
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    by_mdp = defaultdict(list)
    for row in manifest["datasets"]:
        if row.get("stream") == "training":
            by_mdp[row["instance_id"]].append({**row, "seed_index": seed_index(row)})
    tasks = [{"instance": instances[m], "dataset": str(DATASET), "colosseum": None,
              "records": sorted(r, key=lambda x: (x["policy_family"], x["size_multiplier"], x["seed_index"]))}
             for m, r in sorted(by_mdp.items())]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outcomes = pd.DataFrame([row for result in pool.map(E._mdp_jobs, tasks) for row in result])
    counts = outcomes.groupby(["mdp_id", "policy_family", "algorithm", "budget"]).success.agg(["sum", "size"])
    if (counts["size"] != 10).any():
        raise RuntimeError("ten seeds required")
    wide = counts["sum"].unstack("budget").reindex(columns=list(E.BUDGETS))
    labels = wide.reset_index()
    labels["event"] = [suffix_event(tuple(int(x) for x in r)) for r in wide.to_numpy()]
    labels = labels.rename(columns={b: f"successes_x{b}" for b in E.BUDGETS})
    outcomes.to_csv(OUTPUT / "outcomes.csv.gz", index=False)
    labels.to_csv(OUTPUT / "labels.csv", index=False)
    E.write_json(OUTPUT / "outcome_run.json", {"jobs": len(outcomes), "completed_at_utc": E.now(),
                                               "labels_sha256": E.sha256(OUTPUT / "labels.csv")})
    print(f"{len(outcomes)} jobs, {len(labels)} labels", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "evaluate"))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    {"freeze": freeze, "prepare": lambda: prepare(args.workers), "evaluate": lambda: evaluate(args.workers)}[args.phase]()


if __name__ == "__main__":
    main()

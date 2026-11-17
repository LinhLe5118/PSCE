"""SPIBB benchmark panel: MDPs **and** logging behavior from another group's procedure.

Laroche, Trichelair & Tachet des Combes (ICML 2019) random-MDP benchmark,
author code pinned in ``tmp/external_spibb`` (commit 833f32b), used unchanged:
``garnets.Garnets(50, 4, 4, self_transitions=0)``, goal = the state the authors
pick as "farther" (reward 1 on reaching it, then terminal), gamma 0.95, and the
authors' baseline ``generate_baseline_policy(gamma, softmax_target_perf_ratio=
(rho + 1) / 2, baseline_target_perf_ratio=rho)`` for rho in {0.1, ..., 0.9}.
Each MDP has exactly this one logging behavior. The project's generator turns
it into an x10 pilot and the x1..x50 training streams; labels, forecaster and
gate are those of the eighth cohort (sealed 720-MDP fit, same comparators,
same fixed-sequence steps). Meta-training has no SPIBB-style behavior: this
is the distribution-shift test of the handover plan (C1).

Phases: ``freeze``, ``prepare``, ``evaluate``, ``analyze``.
"""
from __future__ import annotations

import argparse
import contextlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from scripts.analysis import run_oeh_eighth_cohort_20261014 as E

ROOT = E.ROOT
OUTPUT = ROOT / "artifacts/oeh_spibb_panel_2026-10-14"
DATASET = ROOT / "datasets/oeh_spibb_panel_2026-10-14"
KERNELS = OUTPUT / "kernels"
SPIBB = ROOT / "tmp/external_spibb"
NAMESPACE = "oeh-spibb-panel-v1-20261014"
RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PER_RATIO = 12
FAMILY = "spibb_baseline"
KEYS, P = E.KEYS, E.P


def spibb_mdp(source: str, mdp_id: str, TabularMDP):
    with np.load(Path(source) / f"{mdp_id}.npz") as z:
        mdp = TabularMDP(environment_id=mdp_id, P=z["P"].astype(np.float64), R=z["R"].astype(np.float64),
                         initial_state_distribution=z["rho"].astype(np.float64),
                         terminal_states=tuple(int(s) for s in z["terminal"]), gamma=0.95, metadata={})
    mdp.validate()
    return mdp


def _garnet(task: dict) -> dict:
    sys.path.insert(0, str(SPIBB))
    import garnets
    np.random.seed(task["seed"])
    with contextlib.redirect_stdout(io.StringIO()):
        garnet = garnets.Garnets(50, 4, 4, self_transitions=0)
        pi_b, _, pi_star_perf, pi_b_perf, pi_rand_perf = garnet.generate_baseline_policy(
            0.95, softmax_target_perf_ratio=(task["ratio"] + 1) / 2, baseline_target_perf_ratio=task["ratio"])
    final = int(garnet.final_state)
    S, A = 50, 4
    P = np.array(garnet.transition_function, dtype=float)
    P[final] = 0.0
    P[final, :, final] = 1.0
    R = np.zeros((S, A, S))
    R[:, :, final] = 1.0
    R[final] = 0.0
    rho = np.zeros(S)
    rho[0] = 1.0
    np.savez(Path(task["out"]) / f"{task['mdp_id']}.npz", P=P, R=R, rho=rho, terminal=np.array([final]), pi_b=pi_b)
    return {"mdp_id": task["mdp_id"], "seed": task["seed"], "ratio": task["ratio"], "final_state": final,
            "author_pi_star_perf": float(pi_star_perf), "author_pi_b_perf": float(pi_b_perf),
            "author_pi_rand_perf": float(pi_rand_perf)}


def freeze() -> None:
    if OUTPUT.exists() or DATASET.exists():
        raise FileExistsError("SPIBB namespace exists")
    eighth = E.protocol()
    doc = {
        "schema": "oeh-spibb-panel-v1", "namespace": NAMESPACE, "frozen_at_utc": E.now(),
        "status": "frozen_before_mdp_generation_or_learner_outcomes",
        "panels": {"spibb": {"mdps": len(RATIOS) * PER_RATIO, "behaviors": 1,
                             "source": "Laroche et al. 2019 random-MDP benchmark, author code tmp/external_spibb @ 833f32b",
                             "garnet": {"states": 50, "actions": 4, "next_states": 4, "self_transitions": 0},
                             "baseline_ratios": list(RATIOS), "per_ratio": PER_RATIO,
                             "seeds": "numpy global seed = 20261014000 + index (authors' code uses the global RNG)",
                             "selection": "every generated MDP that passes the project's dynamic-range filter"}},
        "data": eighth["data"], "forecaster": eighth["forecaster"], "candidates": eighth["candidates"],
        "comparators": eighth["comparators"], "gate": {**eighth["gate"],
                                                       "reading": "step 1 is the SPIBB transfer test (handover C1)"},
        "descriptive": eighth["descriptive"], "eighth_protocol_sha256": E.sha256(E.OUTPUT / "protocol.json"),
        "source_sha256": {E.rel(p): E.sha256(p) for p in (Path(__file__), SPIBB / "garnets.py", SPIBB / "spibb.py",
                                                          SPIBB / "spibb_utils.py")},
    }
    OUTPUT.mkdir(parents=True)
    E.write_json(OUTPUT / "protocol.json", doc)
    (OUTPUT / "protocol.sha256").write_text(E.sha256(OUTPUT / "protocol.json") + "\n", encoding="ascii")
    print(f"Frozen SPIBB protocol {E.sha256(OUTPUT / 'protocol.json')}", flush=True)


def _use_namespace() -> None:
    E.OUTPUT = OUTPUT
    E.DATASET = DATASET


def generate(workers: int) -> dict:
    if (OUTPUT / "dataset_seal.json").exists():
        return E.load_json(DATASET / "manifest.json")
    KERNELS.mkdir(parents=True, exist_ok=True)
    tasks = [{"mdp_id": f"spibb_r{int(ratio * 10)}_{k:03d}", "ratio": ratio, "seed": 2026101400 + i * PER_RATIO + k,
              "out": str(KERNELS)} for i, ratio in enumerate(RATIOS) for k in range(PER_RATIO)]
    with ProcessPoolExecutor(max_workers=workers) as pool:  # before archive_imports: workers import this module
        records = list(pool.map(_garnet, tasks))
    from scripts.analysis.run_unseen_validation import archive_imports
    _, _, generator, _ = archive_imports()
    import environments.behavior_policies as behavior
    from environments.benchmark_environments import BenchmarkSpec, DYNAMIC_RANGE_THRESHOLD
    from environments.tabular_environments import TabularMDP
    from exact_solver import passes_dynamic_range_filter
    registry, specs, policies, selection = {}, [], {}, []
    for record in records:
        mdp = spibb_mdp(str(KERNELS), record["mdp_id"], TabularMDP)
        keep = bool(passes_dynamic_range_filter(mdp, threshold=DYNAMIC_RANGE_THRESHOLD))
        selection.append({**record, "passes_filter": keep})
        if not keep:
            continue
        with np.load(KERNELS / f"{record['mdp_id']}.npz") as z:
            policies[record["mdp_id"]] = z["pi_b"]
        spec = BenchmarkSpec(instance_id=record["mdp_id"], split="evaluation", family="spibb_garnet",
                             seed=9_900_000 + 1_000 * len(specs), ordinal=len(specs), family_ordinal=len(specs))
        registry[spec.instance_id] = mdp
        specs.append(spec)
    E.write_json(OUTPUT / "spibb_selection.json", {"records": selection, "selected": len(specs)})
    behavior.POLICY_FAMILIES = (*behavior.POLICY_FAMILIES, FAMILY)
    generator.resolve_dynamic_range = lambda spec: (spec, registry[spec.instance_id], 0)
    generator.behavior_policy_suite = lambda mdp, seed, variant_index: [behavior.BehaviorPolicy(
        family=FAMILY, probabilities=np.asarray(policies[mdp.environment_id], float), parameters={"source": "spibb author"},
        seed=seed, uses_q_star=True)]
    manifest = generator.generate_benchmark_bundle(
        DATASET, specs=specs, pilot_seeds=(0,), training_seeds=tuple(range(10)), max_size=None, horizon=100,
        run_mode="test", deviation_reason="SPIBB panel: author MDPs and baseline policies", pilot_size_multipliers=(2, 5, 10))
    E.write_json(OUTPUT / "dataset_seal.json", {
        "protocol_sha256": E.sha256(OUTPUT / "protocol.json"), "sealed_at_utc": E.now(), "learner_outcomes_observed": 0,
        "selected_mdps": len(specs), "kernels_sha256": {p.stem: E.sha256(p) for p in sorted(KERNELS.glob("*.npz"))},
        "sha256": {E.rel(p): E.sha256(p) for p in (DATASET / "manifest.json", DATASET / "data_only_index.json")}})
    return manifest


def _published_task(task):
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    import environments.benchmark_environments as B
    from environments.tabular_environments import TabularMDP
    B.make_benchmark_mdp = lambda spec: spibb_mdp(task["kernels"], spec.instance_id, TabularMDP)
    from scripts.analysis import oeh_final_published as Pub
    return Pub._task(task)


def prepare(workers: int) -> None:
    from scripts.analysis import run_oeh_final_120 as R
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis.develop_cross_learner_context import with_context
    from scripts.analysis.develop_cohort8_candidates_20261014 import add_fields

    _use_namespace()
    target = E.panel_dir("spibb")
    if (target / "preoutcome_seal.json").exists():
        raise FileExistsError("SPIBB panel already sealed")
    doc = E.protocol()
    manifest = generate(workers)
    R._repo_first()  # spawned workers must import this runner, not the archived package
    target.mkdir(parents=True, exist_ok=True)
    seven = E.load_json(E.COHORT7 / "protocol.json")["oeh_final"]
    features = E.pilot_features(DATASET, target / "pilot_features.csv", workers)
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    rows = sorted((r for r in manifest["datasets"] if r.get("stream") == "pilot" and int(r.get("size_multiplier", -1)) == 10),
                  key=lambda r: r["instance_id"])
    out = target / "published_preoutcome.csv"
    if not out.exists():
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pd.DataFrame(list(pool.map(_published_task, [{"instance": instances[r["instance_id"]], "row": r,
                                                           "dataset": str(DATASET), "kernels": str(KERNELS)} for r in rows],
                                       chunksize=4))).to_csv(out, index=False)
    pub = pd.read_csv(out)
    R._repo_first()
    R.run_oeh_b(DATASET, target / "oeh_b_groups", seven["structure"], seven["requirement_draws"], f"{NAMESPACE}|spibb",
                workers, "cpu")
    R.run_horizon(DATASET, target / "horizon_preoutcome.csv", seven["structure"], workers, oracle=False)
    curve = R.run_curve(DATASET, target / "curve_preoutcome.csv", seven["structure"], seven["requirement_draws"], workers)
    query = features.merge(F.oeh_feature_rows(target / "oeh_b_groups", target / "horizon_preoutcome.csv"), on=KEYS,
                           validate="one_to_one")
    query = query.merge(R.curve_features(curve), on=KEYS, validate="one_to_one")
    query = query.merge(pub, on=["mdp_id", "policy_family"], validate="many_to_one")
    query = add_fields(with_context(query))
    query["panel"] = "primary"
    selected = E.load_json(OUTPUT / "spibb_selection.json")["selected"]
    if len(query) != selected * 3 or query[KEYS].duplicated().any():
        raise RuntimeError(f"Incomplete SPIBB query: {len(query)}")
    query = query.sort_values(KEYS).reset_index(drop=True)
    E.seal("spibb", query, E.forecast("spibb", query, workers))


def _mdp_jobs(task):
    """Same learners, evaluation and success rule as run_oeh_eighth_cohort._mdp_jobs."""
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    sys.path.insert(0, str(ROOT / "benchmarks/colosseum/driver"))
    import expectile_fix
    expectile_fix.install()
    expectile_fix.CALLS["fallback"] = 0
    from config.protocol_config import PROTOCOL_CONFIG
    from scripts.training.run_empirical_requirement import _load_run, _score_policy
    from empirical_requirement import behavior_cloning_tabular, cfqi_tabular, iql_tabular
    from environments.tabular_environments import TabularMDP

    learning = PROTOCOL_CONFIG["learning"]
    instance = task["instance"]
    mdp = spibb_mdp(task["kernels"], instance["instance_id"], TabularMDP)
    cache, rows = {}, []
    for record in task["records"]:
        transitions = _load_run(Path(task["dataset"]), record, cache)
        for method in ("cfqi_tabular", "iql_tabular", "behavior_cloning_tabular"):
            if method == "cfqi_tabular":
                policy = cfqi_tabular(transitions, num_states=mdp.num_states, num_actions=mdp.num_actions, gamma=mdp.gamma,
                                      iterations=int(learning["iterations"]), tolerance=float(learning["tolerance"]))
            elif method == "iql_tabular":
                policy = iql_tabular(transitions, num_states=mdp.num_states, num_actions=mdp.num_actions, gamma=mdp.gamma,
                                     expectile=float(learning["iql_expectile"]),
                                     inverse_temperature=float(learning["iql_inverse_temperature"]),
                                     iterations=int(learning["iterations"]), tolerance=float(learning["tolerance"]))
            else:
                policy = behavior_cloning_tabular(transitions, num_states=mdp.num_states, num_actions=mdp.num_actions)
            value, status = _score_policy(mdp, np.asarray(policy, dtype=float), random_return=float(instance["random_return"]),
                                          dynamic_range=float(instance["dynamic_range"]))
            if status != "observed":
                raise RuntimeError(f"invalid score {instance['instance_id']}/{method}")
            rows.append({"mdp_id": instance["instance_id"], "policy_family": record["policy_family"], "algorithm": method,
                         "budget": int(record["size_multiplier"]), "training_seed": int(record["seed_index"]),
                         "buffer_sha256": record["sha256"], "normalized_return": value, "success": bool(value >= 0.95),
                         "expectile_fallbacks_so_far": expectile_fix.CALLS["fallback"]})
    return rows


def evaluate(workers: int) -> None:
    from scripts.analysis.run_unseen_validation import archive_imports
    from scripts.analysis.compare_difficulty import suffix_event

    _use_namespace()
    target = E.panel_dir("spibb")
    seal = E.load_json(target / "preoutcome_seal.json")
    for name, digest in seal["artifact_sha256"].items():
        if E.sha256(target / name) != digest:
            raise RuntimeError(f"Sealed file changed: {name}")
    if (target / "outcomes.csv.gz").exists():
        raise FileExistsError("Outcomes already computed")
    _, _, _, seed_index = archive_imports()
    from scripts.analysis import run_oeh_final_120 as R
    R._repo_first()
    manifest = E.load_json(DATASET / "manifest.json")
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    by_mdp = defaultdict(list)
    for row in manifest["datasets"]:
        if row.get("stream") == "training":
            by_mdp[row["instance_id"]].append({**row, "seed_index": seed_index(row)})
    if sum(map(len, by_mdp.values())) != len(instances) * 6 * 10:
        raise RuntimeError("Incomplete SPIBB training grid")
    tasks = [{"instance": instances[m], "dataset": str(DATASET), "kernels": str(KERNELS),
              "records": sorted(r, key=lambda x: (x["size_multiplier"], x["seed_index"]))} for m, r in sorted(by_mdp.items())]
    tick = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outcomes = pd.DataFrame([row for result in pool.map(_mdp_jobs, tasks) for row in result])
    counts = outcomes.groupby(["mdp_id", "policy_family", "algorithm", "budget"]).success.agg(["sum", "size"])
    if (counts["size"] != 10).any():
        raise RuntimeError("every group/learner/budget needs ten seeds")
    wide = counts["sum"].unstack("budget").reindex(columns=list(E.BUDGETS))
    labels = wide.reset_index()
    labels["event"] = [suffix_event(tuple(int(x) for x in r)) for r in wide.to_numpy()]
    labels = labels.rename(columns={b: f"successes_x{b}" for b in E.BUDGETS})
    outcomes.to_csv(target / "outcomes.csv.gz", index=False)
    labels.to_csv(target / "labels.csv", index=False)
    E.write_json(target / "outcome_run.json", {
        "jobs": len(outcomes), "seconds": time.perf_counter() - tick, "completed_at_utc": E.now(),
        "preoutcome_seal_sha256": E.sha256(target / "preoutcome_seal.json"),
        "mdps_with_expectile_fallback": int((outcomes.groupby("mdp_id").expectile_fallbacks_so_far.max() > 0).sum()),
        "outcomes_sha256": E.sha256(target / "outcomes.csv.gz"), "labels_sha256": E.sha256(target / "labels.csv")})
    print(f"spibb: {len(outcomes)} jobs, {len(labels)} labels", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "evaluate", "analyze"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "prepare":
        prepare(args.workers)
    elif args.phase == "evaluate":
        evaluate(args.workers)
    else:
        _use_namespace()
        result = E.analyze("spibb")
        for name, step in result["steps"].items():
            print(name, step["verdict"], round(step["critical"], 3))
            for r in step["contrasts"]:
                print(f"  {r['baseline']:>20s} -> {r['candidate']:<10s} {r['metric']:<8s} gain {r['relative_gain']:+.3f} "
                      f"LB {r['simultaneous_lower_95']:+.4f}")


if __name__ == "__main__":
    main()

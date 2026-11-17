"""One-shot fourth fresh 120-MDP test of the final OEH candidate.

Phases, in order: ``freeze`` (new kernels and the complete protocol), ``prepare``
(datasets, pilot-only features, OEH-B/OEH-H, every forecast and every learner
choice, then a pre-outcome seal), ``evaluate-core`` / ``evaluate-external``
(sharded, resumable learner jobs verified against the sealed plan) and
``analyze`` (labels, the registered difficulty gate, the learner-selection
study and the oracle horizon comparison). Six earlier cohorts are opened and
serve only as meta-training data. No outcome of this cohort exists before the
seal, and nothing is re-selected after it.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

from scripts.analysis import run_oeh_120_cross_capacity as cc  # noqa: F401  (frozen comparator module)
from scripts.analysis import run_oeh_120_unseen as base
from scripts.analysis.compare_difficulty import FEATURES_B2, GRID, METHODS
from scripts.analysis.compare_e1_dataset_signals import SIGNALS
from scripts.analysis.develop_b2_eh_oracle_aft import E1_FEATURES, E1_LABELS, E1_ORACLE, EH, KEYS, P, ROOT, sha256
from scripts.analysis.develop_cross_learner_context import CONTEXT as LC_CONTEXT, with_context
from scripts.analysis.develop_eh_oracle_three_way import aft_probabilities
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.run_unseen_validation import archive_imports, load_json, write_json
from scripts.analysis import oeh_final_forecast as F

OUTPUT = ROOT / "artifacts/oeh_final_unseen_120_2026-10-02"
DATASET = ROOT / "datasets/oeh_final_unseen_120_2026-10-02"
DEVELOPMENT = ROOT / "artifacts/oeh_final_development_2026-10-01"
NAMESPACE = "oeh-final-fourth-new-120-v1-20261002"
CANDIDATE = "oeh_final"
# Tier 1: the same pilot-only information as OEH. Tier 2: privileged information
# from the true MDP (oracle EH) or from reference datasets generated in it (TQ/SACo).
TIER1 = ("b2_matched", "b2_plus_eri", "b2_plus_bwd", "train_frequency",
         "coverage_adaptation", "quality_adaptation", "coverage_quality_adaptation", "b2_generic")
TIER2 = ("b2_plus_eh_oracle_matched", "b2_plus_tq_saco_original", "published_metrics_only")
COMPARATORS = (*TIER1, *TIER2)
PUBLISHED = ("tq_original", "saco_original", "eri", "eri_eligible", "bwd")
# Reported without a gate: the unfitted posterior law, and the previous OEH kept as
# evidence of the saturation of its formula.
REPORTED = ("oeh_b_raw", "prior_oeh_matched")
EXTERNAL = ("pib_spibb_n10_author", "vi_lcb_cb005_author_empirical",
            "discrete_bcq_author", "discrete_cql_d3rlpy")
LIBRARY = (*METHODS, *EXTERNAL)

base.OUTPUT = OUTPUT
base.DATASET = DATASET
base.CANDIDATE = CANDIDATE
base.COMPARATORS = COMPARATORS
base.FORECAST_METHODS = (*COMPARATORS, CANDIDATE, *REPORTED)

OPENED = {
    "e1": dict(labels=E1_LABELS, features=E1_FEATURES, oracle=E1_ORACLE,
               dataset=ROOT / "datasets/week6_e1_oehv4e_confirmatory_v1_1", mdps=120),
    "first60": dict(labels=ROOT / "artifacts/independent_difficulty_2026-09-30/labels.csv",
                    features=ROOT / "artifacts/independent_validation_2026-09-30/pilot_features.csv",
                    oracle=ROOT / "artifacts/independent_difficulty_2026-09-30/oracle_preoutcome.csv",
                    dataset=ROOT / "datasets/independent_validation_2026-09-30", mdps=60),
    "second60": dict(labels=ROOT / "artifacts/oeh_difficulty_confirmation_2026-10-01/labels.csv",
                     features=ROOT / "artifacts/oeh_difficulty_confirmation_2026-10-01/pilot_features.csv",
                     oracle=ROOT / "artifacts/oeh_b2_eh_oracle_second_development_2026-10-01/oracle_descriptors.csv",
                     dataset=ROOT / "datasets/oeh_difficulty_confirmation_2026-10-01", mdps=60),
    "first120": dict(labels=ROOT / "artifacts/oeh_difficulty_unseen_120_2026-10-01/labels.csv",
                     features=ROOT / "artifacts/oeh_difficulty_unseen_120_2026-10-01/pilot_features.csv",
                     oracle=ROOT / "artifacts/oeh_difficulty_unseen_120_2026-10-01/oracle_descriptors_preoutcome.csv",
                     dataset=ROOT / "datasets/oeh_difficulty_unseen_120_2026-10-01", mdps=120),
    "second120": dict(labels=ROOT / "artifacts/oeh_pooled_successor_unseen_120_2026-10-01/labels.csv",
                      features=ROOT / "artifacts/oeh_pooled_successor_unseen_120_2026-10-01/pilot_features.csv",
                      oracle=ROOT / "artifacts/oeh_pooled_successor_unseen_120_2026-10-01/oracle_descriptors_preoutcome.csv",
                      dataset=ROOT / "datasets/oeh_pooled_successor_unseen_120_2026-10-01", mdps=120),
    "third120": dict(labels=ROOT / "artifacts/oeh_cross_capacity_unseen_120_2026-10-01/labels.csv",
                     features=ROOT / "artifacts/oeh_cross_capacity_unseen_120_2026-10-01/pilot_features.csv",
                     oracle=ROOT / "artifacts/oeh_cross_capacity_unseen_120_2026-10-01/oracle_descriptors_preoutcome.csv",
                     dataset=ROOT / "datasets/oeh_cross_capacity_unseen_120_2026-10-01", mdps=120),
}


def _repo_first() -> None:
    """Keep this repository's ``scripts`` package ahead of the archived checkout.

    ``archive_imports`` prepends the archived generator; spawned worker
    processes inherit ``sys.path`` and must import this runner by name.
    """
    sys.path[:] = [str(ROOT)] + [item for item in sys.path if item != str(ROOT)]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Shared per-group computations (also used for meta-training cohorts)
# --------------------------------------------------------------------------- #

def pilot_rows(dataset: Path) -> list[dict]:
    manifest = load_json(dataset / "manifest.json")
    rows = sorted((row for row in manifest["datasets"]
                   if row.get("stream") == "pilot" and int(row.get("size_multiplier", -1)) == 10
                   and row.get("benchmark_split", "evaluation") == "evaluation"),
                  key=lambda row: (row["instance_id"], row["policy_family"]))
    return rows


def _group_name(row: dict) -> str:
    return hashlib.sha256(f"{row['instance_id']}|{row['policy_family']}".encode()).hexdigest()[:24]


def _claim(path: Path) -> bool:
    try:
        os.close(os.open(str(path.with_suffix(".claim")), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def oeh_b_group(task: dict) -> str:
    """Posterior-predictive requirement outcomes for one pilot (core learners)."""
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from oeh import predictive

    row, path = task["row"], Path(task["path"])
    if path.exists() or not _claim(path):
        return str(path)
    batched = None
    if task["backend"] == "gpu":
        from scripts.analysis import oeh_final_gpu
        batched = oeh_final_gpu.learners()
    tick = time.perf_counter()
    with np.load(Path(task["dataset"]) / row["file"], allow_pickle=False) as loaded:
        arrays = {name: loaded[name][:int(row["size"])] for name in loaded.files}
    seed = list(task["seed"])
    outcomes = predictive.posterior_outcomes(
        arrays, row["initial_state_distribution"], num_states=int(row["num_states"]),
        num_actions=int(row["num_actions"]), gamma=0.95, draws=task["draws"], seed=seed,
        batched=batched, **task["structure"])
    forecast = predictive.requirement_forecast(outcomes)
    nr = outcomes.normalized_return
    document = {
        "mdp_id": row["instance_id"], "policy_family": row["policy_family"],
        "pilot_sha256": row["sha256"], "structure": task["structure"], "draws": task["draws"],
        "seed": seed, "posterior": outcomes.posterior, "sampling": outcomes.sampling,
        "learners": {name: {
            "probabilities": value["probabilities"].tolist(),
            "events": value["events"].tolist(), "median_event": value["median_event"],
            "censored_fraction": value["censored_fraction"],
            "mean_normalized_return": nr[:, :, :, i].mean((0, 1)).tolist() if len(nr) else None,
            "success_rate": (nr[:, :, :, i] >= 0.95).mean((0, 1)).tolist() if len(nr) else None,
        } for i, (name, value) in enumerate(forecast.items())},
        "runtime_seconds": time.perf_counter() - tick, "backend": task["backend"],
    }
    temporary = path.with_suffix(".writing")
    temporary.write_text(json.dumps(document), encoding="utf-8")
    temporary.replace(path)
    path.with_suffix(".claim").unlink(missing_ok=True)
    return str(path)


def run_oeh_b(dataset: Path, target: Path, structure: dict, draws: int, namespace: str,
              workers: int, backend: str) -> None:
    target.mkdir(parents=True, exist_ok=True)
    tasks = []
    for row in pilot_rows(dataset):
        path = target / f"{_group_name(row)}.json"
        if path.exists():
            if load_json(path)["pilot_sha256"] != row["sha256"]:
                raise RuntimeError(f"Stale OEH-B result {path}")
            continue
        if path.with_suffix(".claim").exists():
            continue
        digest = hashlib.sha256(f"{namespace}|{row['instance_id']}|{row['policy_family']}".encode()).digest()
        tasks.append({"row": row, "path": str(path), "dataset": str(dataset), "structure": structure,
                      "draws": draws, "seed": [int.from_bytes(digest[:8], "little")],
                      "backend": backend})
    print(f"OEH-B {target.name}: {len(tasks)} pending ({backend})", flush=True)
    _repo_first()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, _ in enumerate(pool.map(oeh_b_group, tasks), 1):
            if done % 50 == 0 or done == len(tasks):
                print(f"OEH-B {target.name}: {done}/{len(tasks)}", flush=True)


def horizon_group(task: dict) -> dict:
    """Data-only OEH-H on the posterior-mean model, plus the oracle panel if asked."""
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from oeh import horizon as H
    from scripts.analysis.develop_oeh_horizon import posterior_mean_model

    row = task["row"]
    with np.load(Path(task["dataset"]) / row["file"], allow_pickle=False) as loaded:
        arrays = {name: loaded[name][:int(row["size"])] for name in loaded.files}
    S, A = int(row["num_states"]), int(row["num_actions"])
    rho = np.asarray(row["initial_state_distribution"], dtype=float)
    Pm, Rm, bm, term = posterior_mean_model(arrays, rho, S, A, 0.95, **task["structure"])
    data = H.offline_effective_horizon(Pm, Rm, bm, rho, 0.95, terminal=term, tau=0.95, horizon=100, form="risk")
    theory = H.offline_theory_measures(Pm, Rm, bm, rho, 0.95, terminal=term, horizon=100)
    observed = np.zeros((S, A), dtype=bool)
    observed[np.asarray(arrays["state"], dtype=int), np.asarray(arrays["action"], dtype=int)] = True
    theory.update(H.asymptotic_ceilings(Pm, Rm, bm, rho, 0.95, supported=observed, terminal=term))
    profile = H.risk_profile(Pm, Rm, bm, rho, 0.95, terminal=term, tau=0.95, horizon=100,
                             budgets=np.array(F.BUDGETS) * S * A)
    theory.update({f"risk_x{b}": float(v) for b, v in zip(F.BUDGETS, profile["noise_regret"])})
    theory.update({f"miss_x{b}": float(v) for b, v in zip(F.BUDGETS, profile["missing_mass"])})
    theory["risk_slack"] = profile["slack"]
    clean = lambda x: None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x  # noqa: E731
    result = {"mdp_id": row["instance_id"], "policy_family": row["policy_family"],
              **{f"data_{k}": clean(v) for k, v in data.as_dict().items()},
              **{f"data_{k}": clean(v) for k, v in theory.items()}}
    if task.get("oracle"):
        archive_imports()
        from environments.behavior_policies import behavior_policy_suite
        from environments.benchmark_environments import make_benchmark_mdp

        instance = task["instance"]
        mdp = make_benchmark_mdp(type("Spec", (), instance)())
        terminal = np.zeros(mdp.num_states, dtype=bool)
        terminal[list(mdp.terminal_states)] = True
        policies = behavior_policy_suite(mdp, seed=int(instance["seed"]) + 3_000_000,
                                         variant_index=int(instance["ordinal"]))
        beta = next(item.probabilities for item in policies if item.family == row["policy_family"])
        oracle = H.offline_effective_horizon(mdp.P, mdp.R, beta, mdp.initial_state_distribution,
                                             mdp.gamma, terminal=terminal, tau=0.95, horizon=100, form="risk")
        eh, eh_k = H.stationary_eh(mdp.P, mdp.R, mdp.gamma, max_depth=100, terminal=terminal)
        theory = H.offline_theory_measures(mdp.P, mdp.R, beta, mdp.initial_state_distribution, mdp.gamma,
                                           terminal=terminal, horizon=100)
        result.update({f"oracle_{k}": clean(v) for k, v in theory.items()})
        result.update({f"oracle_{k}": clean(v) for k, v in oracle.as_dict().items()})
        result.update({"oracle_stationary_eh": clean(eh), "oracle_stationary_eh_k": eh_k})
    return result


REQUIRED_HORIZON = ("data_oeh_h", "data_log_clipped_concentrability", "data_behavior_ceiling", "data_miss_x50")


def run_horizon(dataset: Path, out: Path, structure: dict, workers: int, oracle: bool) -> pd.DataFrame:
    if out.exists():
        frame = pd.read_csv(out)
        if all(column in frame.columns for column in REQUIRED_HORIZON):
            return frame
    manifest = load_json(dataset / "manifest.json")
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    tasks = [{"row": row, "dataset": str(dataset), "structure": structure, "oracle": oracle,
              "instance": instances[row["instance_id"]]} for row in pilot_rows(dataset)]
    _repo_first()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frame = pd.DataFrame(list(pool.map(horizon_group, tasks, chunksize=4)))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


CURVE_PREFIXES = (1, 2, 5, 10)


def curve_group(task: dict) -> dict:
    """Learners trained on pilot prefixes, evaluated in posterior MDP draws."""
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from oeh import posterior as post
    from oeh.tabular import CORE_LEARNERS, count_model, normalized_returns

    row = task["row"]
    with np.load(Path(task["dataset"]) / row["file"], allow_pickle=False) as loaded:
        arrays = {name: loaded[name][:int(row["size"])] for name in loaded.files}
    S, A = int(row["num_states"]), int(row["num_actions"])
    stats = post.pilot_statistics(arrays, row["initial_state_distribution"], num_states=S, num_actions=A, gamma=0.95)
    model = post.fit_posterior(stats, **task["structure"])
    digest = hashlib.sha256(f"oeh-final-curve|{row['instance_id']}|{row['policy_family']}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    sample, _ = post.sample_mdps(model, rng, task["draws"], minimum_dynamic_range=0.01 / (1 - 0.95))
    result = {"mdp_id": row["instance_id"], "policy_family": row["policy_family"]}
    names = list(CORE_LEARNERS)
    for j in CURVE_PREFIXES:
        size = j * S * A
        counts = count_model(arrays["state"][:size], arrays["action"][:size], arrays["reward"][:size],
                             arrays["next_state"][:size], arrays["terminal"][:size],
                             num_states=S, num_actions=A)
        policies = np.stack([CORE_LEARNERS[name](counts, 0.95)[0] for name in names])
        if len(sample.P):
            nr = normalized_returns(sample.P, sample.expected_rewards,
                                    np.broadcast_to(sample.rho, (len(sample.P), S)),
                                    np.broadcast_to(policies, (len(sample.P),) + policies.shape), 0.95)
            means, success = nr.mean(0), (nr >= 0.95).mean(0)
        else:
            means, success = np.zeros(len(names)), np.zeros(len(names))
        for i, name in enumerate(names):
            result[f"{name}|curve_mean_x{j}"] = float(means[i])
            result[f"{name}|curve_success_x{j}"] = float(success[i])
    return result


def run_curve(dataset: Path, out: Path, structure: dict, draws: int, workers: int) -> pd.DataFrame:
    if out.exists():
        return pd.read_csv(out)
    tasks = [{"row": row, "dataset": str(dataset), "structure": structure, "draws": draws}
             for row in pilot_rows(dataset)]
    _repo_first()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frame = pd.DataFrame(list(pool.map(curve_group, tasks, chunksize=4)))
    frame.to_csv(out, index=False)
    return frame


def curve_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per learner with its own curve and others' x10 success."""
    rows = []
    for record in frame.to_dict("records"):
        for learner in METHODS:
            item = {"mdp_id": record["mdp_id"], "policy_family": record["policy_family"], "algorithm": learner}
            for j in CURVE_PREFIXES:
                item[f"oehc_mean_x{j}"] = record[f"{learner}|curve_mean_x{j}"]
                item[f"oehc_success_x{j}"] = record[f"{learner}|curve_success_x{j}"]
            for other in METHODS:
                item[f"other_{other}_oehc_success_x10"] = record[f"{other}|curve_success_x10"]
            rows.append(item)
    return pd.DataFrame(rows)


CURVE = tuple([f"oehc_mean_x{j}" for j in CURVE_PREFIXES] + [f"oehc_success_x{j}" for j in CURVE_PREFIXES]
              + [f"other_{m}_oehc_success_x10" for m in METHODS])
CEILING_STRUCTURE = {"reward_prior": "exchangeable", "transition_base": "uniform", "behavior_prior": "mixture"}
CEILINGS = ("bayes_behavior_ceiling_mean", "bayes_behavior_ceiling_sd",
            "bayes_support_ceiling_mean", "bayes_support_ceiling_sd")


def ceiling_group(task: dict) -> dict:
    """Posterior distribution of behavior and support-optimal normalized returns."""
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from oeh import posterior as post

    row = task["row"]
    with np.load(Path(task["dataset"]) / row["file"], allow_pickle=False) as loaded:
        arrays = {name: loaded[name][:int(row["size"])] for name in loaded.files}
    S, A = int(row["num_states"]), int(row["num_actions"])
    stats = post.pilot_statistics(arrays, row["initial_state_distribution"], num_states=S, num_actions=A, gamma=0.95)
    model = post.fit_posterior(stats, **CEILING_STRUCTURE)
    digest = hashlib.sha256(f"oeh-final-ceiling|{row['instance_id']}|{row['policy_family']}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    sample, _ = post.sample_mdps(model, rng, task["draws"], minimum_dynamic_range=0.01 / (1 - 0.95))
    result = {"mdp_id": row["instance_id"], "policy_family": row["policy_family"]}
    if not len(sample.P):
        return {**result, **{name: 0.0 for name in CEILINGS}}
    from oeh.horizon import asymptotic_ceilings

    observed = stats.action_counts > 0
    values = np.array([[asymptotic_ceilings(sample.P[b], sample.R[b], sample.beta[b], sample.rho, 0.95,
                                            supported=observed, terminal=sample.terminal)[key]
                        for key in ("behavior_ceiling", "support_ceiling")] for b in range(len(sample.P))])
    return {**result, "bayes_behavior_ceiling_mean": float(values[:, 0].mean()),
            "bayes_behavior_ceiling_sd": float(values[:, 0].std()),
            "bayes_support_ceiling_mean": float(values[:, 1].mean()),
            "bayes_support_ceiling_sd": float(values[:, 1].std())}


def run_ceiling(dataset: Path, out: Path, draws: int, workers: int) -> pd.DataFrame:
    if out.exists():
        return pd.read_csv(out)
    tasks = [{"row": row, "dataset": str(dataset), "draws": draws} for row in pilot_rows(dataset)]
    _repo_first()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frame = pd.DataFrame(list(pool.map(ceiling_group, tasks, chunksize=4)))
    frame.to_csv(out, index=False)
    return frame


# --------------------------------------------------------------------------- #
# Meta-training panel
# --------------------------------------------------------------------------- #

def opened_frame(name: str, structure_tag: str) -> pd.DataFrame:
    spec = OPENED[name]
    frame = pd.read_csv(spec["labels"]).merge(pd.read_csv(spec["features"]), on=KEYS, validate="one_to_one")
    oracle = pd.read_csv(spec["oracle"])
    frame = frame.merge(oracle[["mdp_id", *EH]], on="mdp_id", validate="many_to_one")
    oeh = F.oeh_feature_rows(DEVELOPMENT / "groups" / name / structure_tag,
                             DEVELOPMENT / f"horizon_data_{name}_{structure_tag}.csv")
    frame = frame.merge(oeh, on=KEYS, validate="one_to_one")
    published = pd.read_csv(DEVELOPMENT / f"published_{name}.csv")
    frame = frame.merge(published, on=["mdp_id", "policy_family"], validate="many_to_one")
    original = DEVELOPMENT / f"tqsaco_original_{name}.csv"
    if original.exists():
        frame = frame.merge(pd.read_csv(original)[["mdp_id", "policy_family", "tq_original", "saco_original"]],
                            on=["mdp_id", "policy_family"], validate="many_to_one")
    curve_path = DEVELOPMENT / f"curve_{name}_{structure_tag}.csv"
    if curve_path.exists():
        frame = frame.merge(curve_features(pd.read_csv(curve_path)), on=KEYS, validate="one_to_one")
    ceiling_path = DEVELOPMENT / f"ceiling_{name}.csv"
    if ceiling_path.exists():
        frame = frame.merge(pd.read_csv(ceiling_path), on=["mdp_id", "policy_family"], validate="many_to_one")
    if len(frame) != spec["mdps"] * 15 or frame.mdp_id.nunique() != spec["mdps"]:
        raise RuntimeError(f"Incomplete opened cohort {name}")
    frame["cohort"] = name
    return with_context(frame)


def meta_training(cohorts, structure_tag: str) -> pd.DataFrame:
    frame = pd.concat([opened_frame(name, structure_tag) for name in cohorts], ignore_index=True)
    if frame[KEYS].duplicated().any():
        raise RuntimeError("Overlapping meta-training rows")
    return frame


FIELD_SETS = {
    "b2_matched": FEATURES_B2,
    "b2_plus_eh_oracle_matched": (*FEATURES_B2, *EH),
    "prior_oeh_matched": (*OEH_FEATURES, *LC_CONTEXT),
    "b2_plus_tq_saco_original": (*FEATURES_B2, "tq_original", "saco_original"),
    "b2_plus_eri": (*FEATURES_B2, "eri", "eri_eligible"),
    "b2_plus_bwd": (*FEATURES_B2, "bwd"),
    "published_metrics_only": PUBLISHED,
    CANDIDATE: F.FINAL_FIELDS,
}
CANDIDATE_VARIANTS = {
    "base": F.FINAL_FIELDS,
    # Union with the previous OEH diagnostics (OEH3, LCOEH4, cross-learner context).
    "plus_prior": tuple(dict.fromkeys((*F.FINAL_FIELDS, *OEH_FEATURES, *LC_CONTEXT))),
    # A4: plus the OEH-H risk profile at the protocol budgets.
    "plus_prior_profile": tuple(dict.fromkeys((*F.FINAL_FIELDS, *OEH_FEATURES, *LC_CONTEXT, *F.PROFILE))),
}
# A5: plus the real-data learning curve from pilot prefixes.
CANDIDATE_VARIANTS["plus_prior_profile_curve"] = tuple(dict.fromkeys((*CANDIDATE_VARIANTS["plus_prior_profile"], *CURVE)))
# A8: plus reference-free Bayesian ceilings (exchangeable-reward posterior).
CANDIDATE_VARIANTS["plus_prior_profile_curve_ceiling"] = tuple(dict.fromkeys(
    (*CANDIDATE_VARIANTS["plus_prior_profile_curve"], *CEILINGS)))


CANDIDATE_LAW = False  # A7: add the recalibrated OEH-B law as an ensemble component


def use_candidate(variant: str, law: bool = False) -> None:
    global CANDIDATE_LAW
    FIELD_SETS[CANDIDATE] = CANDIDATE_VARIANTS[variant]
    CANDIDATE_LAW = law


def all_forecasts(train: pd.DataFrame, query: pd.DataFrame, raw: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Every registered forecast for ``query``; weights are learned on ``train``."""
    tables, tuning = [], {}
    for name, fields in FIELD_SETS.items():
        probability, weights = F.matched_forecast(train, query, fields, law=(name == CANDIDATE and CANDIDATE_LAW))
        tables.append((name, probability))
        tuning[name] = weights
    generic = np.full((len(query), 7), np.nan)
    adaptations = {name: np.full((len(query), 7), np.nan) for name in
                   ("coverage_adaptation", "quality_adaptation", "coverage_quality_adaptation")}
    frequency = np.full((len(query), 7), np.nan)
    for panel in ("primary", "control"):
        for learner in METHODS:
            source = train[(train.panel == panel) & (train.algorithm == learner)]
            qi = np.flatnonzero(((query.panel == panel) & (query.algorithm == learner)).to_numpy())
            if not len(qi):
                continue
            y = source.event.to_numpy(dtype=int)
            frequency[qi] = np.bincount(y, minlength=7) / len(y)
            selected = query.iloc[qi]
            generic[qi] = aft_probabilities(F._matrix(source, FEATURES_B2), y, F._matrix(selected, FEATURES_B2))
            for name in adaptations:
                adaptations[name][qi] = aft_probabilities(F._matrix(source, SIGNALS[name]), y,
                                                          F._matrix(selected, SIGNALS[name]))
    tables += [("b2_generic", generic), ("train_frequency", frequency), *adaptations.items()]
    frames = []
    for name, probability in tables:
        if not np.isfinite(probability).all() or not np.allclose(probability.sum(1), 1.0):
            raise RuntimeError(f"Invalid forecast {name}")
        frame = query[KEYS + ["panel"]].copy()
        frame["method"] = name
        frame[P] = probability
        frames.append(frame)
    if raw is not None:
        frames.append(raw.merge(query[KEYS + ["panel"]], on=KEYS, validate="one_to_one")[KEYS + ["panel", "method", *P]])
    return pd.concat(frames, ignore_index=True), tuning


def raw_oeh_b(group_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(group_dir.glob("*.json")):
        document = load_json(path)
        for learner, value in document["learners"].items():
            rows.append({"mdp_id": document["mdp_id"], "policy_family": document["policy_family"],
                         "algorithm": learner, "method": "oeh_b_raw",
                         **{f"p_{i}": p for i, p in enumerate(value["probabilities"])}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #

def seen_kernels() -> set[str]:
    seen = set(base.prior_kernels())
    for name in ("oeh_difficulty_unseen_120_2026-10-01", "oeh_pooled_successor_unseen_120_2026-10-01",
                 "oeh_cross_capacity_unseen_120_2026-10-01"):
        seen |= set(load_json(ROOT / "artifacts" / name / "protocol.json")["kernel_fingerprints"].values())
        audit = ROOT / "artifacts" / name / "actual_kernel_audit_preoutcome.json"
        if audit.exists():
            seen |= set(load_json(audit).get("actual_kernel_fingerprints", {}).values())
    return seen


def freeze(args) -> None:
    if OUTPUT.exists() or DATASET.exists():
        raise FileExistsError("Fourth-test namespace already exists")
    selection = load_json(DEVELOPMENT / "selection.json")
    if selection.get("status") != "frozen":
        raise RuntimeError("Development selection is not frozen")
    third = load_json(ROOT / "artifacts/oeh_cross_capacity_unseen_120_2026-10-01/result_manifest.json")
    if third["status"] != "complete":
        raise RuntimeError("Third test must be complete before freezing the fourth")
    archive_imports()
    from environments.benchmark_environments import benchmark_specs, make_benchmark_mdp

    seen = seen_kernels()
    used = set()
    for spec in OPENED.values():
        used |= {int(row["seed"]) for row in load_json(spec["dataset"] / "manifest.json")["instances"]}
    specs, kernels, substitutions = [], {}, {}
    templates = [row for row in benchmark_specs() if row.split == "evaluation"]

    def fresh(template, instance_id):
        for retry in range(5000):  # exhaustive enough for the smallest lock space (2^5)
            digest = hashlib.sha256(f"{NAMESPACE}|{template.instance_id}|{retry}".encode()).digest()
            seed = 100_000_000 + int.from_bytes(digest[:8], "little") % 1_000_000_000
            if seed in used:
                continue
            spec = replace(template, instance_id=instance_id, seed=seed)
            kernel = base.fingerprint(make_benchmark_mdp(spec))
            if kernel not in seen:
                return spec, kernel
        return None, None

    for family, count in base.FAMILY_COUNTS.items():
        selected = sorted((row for row in templates if row.family == family), key=lambda row: row.ordinal)
        if len(selected) != count:
            raise RuntimeError("Historical recipe counts changed")
        for ordinal, template in enumerate(selected):
            instance_id = f"oehfinal120_{family}_{ordinal:03d}"
            spec, kernel = fresh(template, instance_id)
            if spec is None and family == "lock_structured":
                # The deterministic lock space of this length is exhausted by earlier
                # cohorts; use the next longer registered lock length instead.
                size = make_benchmark_mdp(template).num_states
                longer = sorted({make_benchmark_mdp(row).num_states for row in selected} - {size})
                for target in (value for value in longer if value > size):
                    for candidate in (row for row in selected if make_benchmark_mdp(row).num_states == target):
                        spec, kernel = fresh(candidate, instance_id)
                        if spec is not None:
                            substitutions[instance_id] = {"template": template.instance_id, "states": size,
                                                          "substitute_template": candidate.instance_id,
                                                          "substitute_states": target}
                            break
                    if spec is not None:
                        break
            if spec is None:
                raise RuntimeError(f"No fresh seed for {instance_id}")
            used.add(spec.seed)
            seen.add(kernel)
            specs.append(spec)
            kernels[instance_id] = kernel
    if len(specs) != 120:
        raise RuntimeError("Incomplete fourth cohort")
    e1_manifest = load_json(base.E1_MANIFEST)
    sources = [Path(__file__), Path(base.__file__), Path(cc.__file__), Path(F.__file__),
               ROOT / "scripts/analysis/oeh_final_gpu.py", ROOT / "scripts/analysis/oeh_final_learners.py",
               ROOT / "scripts/analysis/develop_oeh_horizon.py",
               *[ROOT / "code/oeh" / name for name in ("tabular.py", "posterior.py", "predictive.py",
                                                       "horizon.py", "selection.py", "unseen.py")],
               DEVELOPMENT / "registry.jsonl", DEVELOPMENT / "selection.json",
               base.ARCHIVE / "evaluation/e1_pipeline.py", base.ARCHIVE / "evaluation/e1_external.py",
               base.ARCHIVE / "scripts/data/generate_offline_datasets.py",
               base.ARCHIVE / "environments/benchmark_environments.py"]
    for spec in OPENED.values():
        sources += [spec["labels"], spec["features"], spec["oracle"]]
    doc = {
        "schema": "oeh-unseen-120-difficulty-v1",
        "status": "frozen_before_dataset_buffers_or_learner_outcomes",
        "namespace": NAMESPACE, "requested_specs": [spec.to_dict() for spec in specs],
        "kernel_fingerprints": kernels, "family_counts": base.FAMILY_COUNTS,
        "control_recipe_substitutions": substitutions,
        "stochastic_primary_mdps": 84, "control_mdps": 36,
        "pilot_multipliers": [2, 5, 10], "budgets": GRID,
        "training_seed_indices": list(range(10)), "behavior_families": 5,
        "core_learners": METHODS, "expected_jobs": base.EXPECTED_JOBS,
        "target": "first complete suffix with >=8/10 seeds normalized return >=0.95; else >x50",
        "candidate": CANDIDATE, "comparators": COMPARATORS, "reported_unfitted": REPORTED,
        "oeh_final": {
            "structure": selection["structure"], "structure_tag": selection["arm"],
            "candidate_variant": selection["candidate_variant"],
            "candidate_law": selection.get("candidate_law", False),
            "select_capacity": selection.get("select_capacity", False),
            "requirement_draws": selection["draws"], "selection_draws": selection["selection_draws"],
            "fields": list(CANDIDATE_VARIANTS[selection["candidate_variant"]]), "meta_training_cohorts": list(OPENED),
            "forecaster": "matched AFT/direct/depth3_200 multiclass with OOF-log-loss convex weights, identical for every fitted comparator",
        },
        "decision_rule": "on 84 stochastic MDPs: OEH point relative gains >=5% in integrated Brier and seven-class log loss versus every comparator, and all simultaneous one-sided 95% MDP-cluster absolute-improvement lower bounds >0 (simultaneous over every tier-1 and tier-2 contrast)",
        "tiers": {"tier1_primary_claim": list(TIER1), "tier2_privileged_information": list(TIER2),
                  "verdicts": ["tier1 (primary claim of superiority over the published state of the art with the same information)",
                               "tier2 (privileged-information comparators, same rule)",
                               "all comparators (tier1 and tier2, strict)"],
                  "reported_without_gate": list(REPORTED),
                  "reason": "team audit PUBLISHED_DIFFICULTY_METHOD_AUDIT_2026-10-01 requires reference-dependent and oracle scores in separate tables; tiering was decided after the development results and is recorded as such"},
        "comparator_sources": {
            "tq_saco_original": "authors' code ml-jku/OfflineRL commit 8cc8eec (online DQN replay reference, exact unique-pair count, max online return); evaluation reproduced bit-for-bit",
            "bwd": "no public code; reproduced from the paper's equations with exact tabular SARSA fixed point and exact (epsilon -> 0) transport; one-hot actions, uniform reference",
            "eri": "exact formula of Swazinna et al. on eligible datasets",
            "eh": "port of the published definitions; the authors' Julia code accepts only deterministic transition tables",
            "learners": "author SPIBB, VI-LCB port, author discrete BCQ, pinned d3rlpy CQL; registered tabular CFQI/IQL/BC"},
        "bootstrap": {"draws": 10000, "seed": 20261004, "joint_contrasts": 2 * len(COMPARATORS)},
        "control_rule": "report 36 control MDPs separately; never select candidate on them",
        "selection_study": {
            "panel": "84 stochastic primary MDPs (420 groups), budget x10, ten training seeds",
            "library": list(LIBRARY),
            "rule": "argmax posterior-predictive P(normalized return >= 0.95), ties by expected return then registry order; identical for all learners",
            "comparators": ["every fixed learner", "frozen E1-trained OEH-v5 router", "uniform random learner (expectation)", "per-group oracle (bound only)"],
            "endpoints": ["mean normalized return", "seed success rate"],
            "decision_rule": "simultaneous one-sided 95% MDP-cluster lower bounds >0 for OEH selection minus the best fixed learner (CFQI if tied) and minus OEH-v5 on both endpoints",
        },
        "horizon_study": {
            "metric": "oracle OEH-H on the true MDP and behavior (DP form)",
            "comparators": ["stationary gap-bound EH", "published stochastic EH descriptors", "size reference log(S*A)"],
            "target": "CFQI requirement event; also the minimum event over core learners",
            "statistic": "Spearman rank correlation with censored events ranked last; 10000 MDP-cluster bootstrap of the paired difference",
        },
        "scope": "new untouched kernels; six opened cohorts are meta-training only; EH is an oracle descriptor, not the published online theorem",
        "original_generator_sha256": e1_manifest["generator_provenance"]["source_sha256"],
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in sources},
    }
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "protocol.json", doc)
    (OUTPUT / "protocol.sha256").write_text(sha256(OUTPUT / "protocol.json") + "\n", encoding="ascii")
    print(f"Frozen fourth 120-MDP protocol {sha256(OUTPUT / 'protocol.json')}", flush=True)


def generate() -> dict:
    doc = base.protocol()
    if DATASET.exists():
        return base.dataset()
    _, _, generator, _ = archive_imports()
    from environments.benchmark_environments import BenchmarkSpec, make_benchmark_mdp

    specs = [BenchmarkSpec(**item) for item in doc["requested_specs"]]
    manifest = generator.generate_benchmark_bundle(
        DATASET, specs=specs, pilot_seeds=(0,), training_seeds=tuple(range(10)),
        max_size=None, horizon=100, run_mode="test",
        deviation_reason="Fourth fresh 120-MDP test of the final OEH candidate",
        pilot_size_multipliers=(2, 5, 10))
    if manifest["instance_count"] != 120 or manifest["physical_buffer_count"] != 6600:
        raise RuntimeError("Incomplete fourth dataset")
    actual = {item["instance_id"]: base.fingerprint(make_benchmark_mdp(type("Spec", (), item)()))
              for item in manifest["instances"]}
    if len(set(actual.values())) != 120 or set(actual.values()) & seen_kernels():
        raise RuntimeError("Resolved fourth kernels are not all distinct and new")
    changed = {name: [doc["kernel_fingerprints"][name], digest] for name, digest in actual.items()
               if doc["kernel_fingerprints"][name] != digest}
    if changed:
        # Dynamic-range resolution re-seeded some requested recipes. Record the
        # resolved kernels before any learner outcome, as the third test did.
        (OUTPUT / "protocol_before_resolved_kernel_amendment.json").write_bytes(
            (OUTPUT / "protocol.json").read_bytes())
        previous = sha256(OUTPUT / "protocol.json")
        doc["kernel_fingerprints"] = actual
        doc["preoutcome_resolved_kernel_amendment"] = {
            "changed": changed, "old_protocol_sha256": previous,
            "learner_outcomes_observed": 0, "recorded_at_utc": now()}
        write_json(OUTPUT / "protocol.json", doc)
        (OUTPUT / "protocol.sha256").write_text(sha256(OUTPUT / "protocol.json") + "\n", encoding="ascii")
    write_json(OUTPUT / "actual_kernel_audit_preoutcome.json", {
        "all_120_unique_and_new": True, "changed_requested_seed_count": len(changed),
        "actual_kernel_fingerprints": actual, "learner_outcomes_observed": 0})
    paths = (DATASET / "manifest.json", DATASET / "data_only_index.json")
    write_json(OUTPUT / "dataset_seal.json", {
        "protocol_sha256": sha256(OUTPUT / "protocol.json"), "sealed_at_utc": now(),
        "sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths}})
    return base.dataset()


def _features_task(task):
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    archive_imports()
    from evaluation import e1_pipeline
    return e1_pipeline._group_features(Path(task["dataset"]), tuple(task["group"]), task["prefixes"])


def pilot_features(manifest: dict, workers: int) -> pd.DataFrame:
    path = OUTPUT / "pilot_features.csv"
    if path.exists():
        return pd.read_csv(path)
    index = load_json(DATASET / "data_only_index.json")["datasets"]
    records = {row["data_only_dataset_id"]: row for row in manifest["datasets"]}
    grouped = defaultdict(dict)
    for item in index:
        if item.get("stream") != "pilot" or int(item.get("size_multiplier", -1)) not in (2, 5, 10):
            continue
        row = records[item["dataset_id"]]
        if row["sha256"] != item["sha256"] or row["file"] != item["file"]:
            raise RuntimeError("Pilot index differs from manifest")
        grouped[(row["instance_id"], row["policy_family"])][int(row["size_multiplier"])] = row
    if len(grouped) != 600 or any(set(v) != {2, 5, 10} for v in grouped.values()):
        raise RuntimeError("Incomplete pilot grid")
    tasks = [{"dataset": str(DATASET), "group": list(key), "prefixes": value}
             for key, value in sorted(grouped.items())]
    _repo_first()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for result in pool.map(_features_task, tasks, chunksize=2) for row in result]
    frame = pd.DataFrame(rows)
    if len(frame) != 1800 or frame[KEYS].duplicated().any():
        raise RuntimeError("Incomplete pilot features")
    frame.to_csv(path, index=False)
    return frame


def selection_group(task: dict) -> dict:
    """Posterior-predictive x10 outcomes of all seven learners on shared datasets."""
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from oeh import predictive
    from oeh.selection import select_learner
    from scripts.analysis import oeh_final_gpu
    from scripts.analysis.oeh_final_learners import external_learners

    path = Path(task["path"])
    if path.exists() or not _claim(path):
        return {}
    row = task["row"]
    with np.load(Path(task["dataset"]) / row["file"], allow_pickle=False) as loaded:
        arrays = {name: loaded[name][:int(row["size"])] for name in loaded.files}
    tick = time.perf_counter()
    outcomes = predictive.posterior_outcomes(
        arrays, row["initial_state_distribution"], num_states=int(row["num_states"]),
        num_actions=int(row["num_actions"]), gamma=0.95, draws=task["draws"], seed=task["seed"],
        budgets=(10,), seeds=1, batched={name: oeh_final_gpu.learners()[name] for name in METHODS},
        per_dataset=external_learners(), **task["structure"])
    values = outcomes.normalized_return[:, 0, 0, :]
    if list(outcomes.learners) != list(LIBRARY):
        raise RuntimeError("Library order changed")
    choice = select_learner(values, outcomes.learners) if len(values) else None
    document = {"mdp_id": row["instance_id"], "policy_family": row["policy_family"],
                "pilot_sha256": row["sha256"], "learners": list(outcomes.learners),
                "normalized_return": values.tolist(), "sampling": outcomes.sampling,
                "selected": choice.learner if choice else METHODS[1],
                "fallback": choice is None,
                "success_probability": choice.success_probability if choice else None,
                "expected_return": choice.expected_return if choice else None,
                "runtime_seconds": time.perf_counter() - tick}
    temporary = path.with_suffix(".writing")
    temporary.write_text(json.dumps(document), encoding="utf-8")
    temporary.replace(path)
    path.with_suffix(".claim").unlink(missing_ok=True)
    return document


def run_selection(doc: dict, manifest: dict, workers: int) -> None:
    family = {item["instance_id"]: item["family"] for item in manifest["instances"]}
    target = OUTPUT / "selection_groups"
    target.mkdir(parents=True, exist_ok=True)
    tasks = []
    for row in pilot_rows(DATASET):
        if family[row["instance_id"]] == "lock_structured":
            continue
        path = target / f"{_group_name(row)}.json"
        if path.exists() or path.with_suffix(".claim").exists():
            continue
        digest = hashlib.sha256(f"{NAMESPACE}|selection|{row['instance_id']}|{row['policy_family']}".encode()).digest()
        tasks.append({"row": row, "path": str(path), "dataset": str(DATASET),
                      "structure": doc["oeh_final"]["structure"], "draws": doc["oeh_final"]["selection_draws"],
                      "seed": [int.from_bytes(digest[:8], "little")]})
    print(f"Selection: {len(tasks)} pending", flush=True)
    _repo_first()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, _ in enumerate(pool.map(selection_group, tasks), 1):
            if done % 20 == 0 or done == len(tasks):
                print(f"Selection: {done}/{len(tasks)}", flush=True)


def v5_choices(query_features: pd.DataFrame, manifest: dict) -> dict:
    """Frozen E1-trained OEH-v5 router for the unlabeled fourth cohort."""
    from scripts.analysis.run_unseen_validation import _load_e1_training
    from oeh.unseen import fit_unseen_routes
    from oeh.reference import FEATURES as BLOCKS

    archive_imports()
    from evaluation import e1_pipeline
    names = e1_pipeline._feature_names()
    features = {(row.mdp_id, row.policy_family, row.algorithm):
                {block: tuple(float(getattr(row, field)) for field in names[block]) for block in BLOCKS}
                for row in query_features.itertuples(index=False)}
    panels = {item["instance_id"]: ("deterministic_control" if item["family"] == "lock_structured"
                                    else "stochastic_primary") for item in manifest["instances"]}
    choices, _, _ = fit_unseen_routes(_load_e1_training(), features, panels)
    return {f"{mdp}|{behavior}": method for (mdp, behavior), method in choices.items()}


def prepare(args) -> None:
    if (OUTPUT / "preoutcome_seal.json").exists():
        raise FileExistsError("Fourth-test forecasts already sealed")
    doc = base.protocol()
    manifest = generate()
    archive_imports()   # generator modules for oracle descriptors and reference data
    _repo_first()
    structure, tag = doc["oeh_final"]["structure"], doc["oeh_final"]["structure_tag"]
    use_candidate(doc["oeh_final"]["candidate_variant"], doc["oeh_final"]["candidate_law"])
    F.SELECT_CAPACITY = doc["oeh_final"]["select_capacity"]
    if list(FIELD_SETS[CANDIDATE]) != doc["oeh_final"]["fields"]:
        raise RuntimeError("Candidate fields differ from the frozen protocol")
    query = pilot_features(manifest, args.workers)
    oracle = base.oracle_descriptors(manifest, doc)
    from scripts.analysis.oeh_final_published import compute as published_scores
    from scripts.analysis.oeh_final_tqsaco_original import compute as tqsaco_original
    published = published_scores(DATASET, OUTPUT / "published_preoutcome.csv", args.workers)
    original = tqsaco_original(DATASET, OUTPUT / "tqsaco_original_preoutcome.csv", args.workers)
    published = published.merge(original[["mdp_id", "policy_family", "tq_original", "saco_original"]],
                                on=["mdp_id", "policy_family"], validate="one_to_one")
    run_oeh_b(DATASET, OUTPUT / "oeh_b_groups", structure, doc["oeh_final"]["requirement_draws"],
              NAMESPACE, args.workers, args.backend)
    horizon = run_horizon(DATASET, OUTPUT / "horizon_preoutcome.csv", structure, args.workers, oracle=True)
    if len(horizon) != 600:
        raise RuntimeError("Incomplete OEH-H panel")
    family = {item["instance_id"]: item["family"] for item in manifest["instances"]}
    oeh = F.oeh_feature_rows(OUTPUT / "oeh_b_groups", OUTPUT / "horizon_preoutcome.csv")
    curve = run_curve(DATASET, OUTPUT / "curve_preoutcome.csv", structure,
                      doc["oeh_final"]["requirement_draws"], args.workers)
    ceiling = run_ceiling(DATASET, OUTPUT / "ceiling_preoutcome.csv",
                          doc["oeh_final"]["requirement_draws"], args.workers)
    query = query.merge(oracle[["mdp_id", *EH]], on="mdp_id", validate="many_to_one")
    query = query.merge(oeh, on=KEYS, validate="one_to_one")
    query = query.merge(curve_features(curve), on=KEYS, validate="one_to_one")
    query = query.merge(ceiling, on=["mdp_id", "policy_family"], validate="many_to_one")
    query = query.merge(published, on=["mdp_id", "policy_family"], validate="many_to_one")
    query["panel"] = query.mdp_id.map(lambda m: "control" if family[m] == "lock_structured" else "primary")
    query = with_context(query)
    train = meta_training(OPENED, tag)
    if len(query) != 1800 or set(train.mdp_id) & set(query.mdp_id):
        raise RuntimeError("Fourth training/query overlap or incomplete")
    predictions, tuning = all_forecasts(train, query, raw_oeh_b(OUTPUT / "oeh_b_groups"))
    predictions.to_csv(OUTPUT / "predictions_preoutcome.csv", index=False)
    write_json(OUTPUT / "forecast_tuning_preoutcome.json", tuning)
    write_json(OUTPUT / "v5_choices_preoutcome.json", v5_choices(pd.read_csv(OUTPUT / "pilot_features.csv"), manifest))
    run_selection(doc, manifest, args.selection_workers)
    rows = [load_json(path) for path in sorted((OUTPUT / "selection_groups").glob("*.json"))]
    if len(rows) != 420:
        raise RuntimeError("Incomplete selection panel")
    write_json(OUTPUT / "oeh_choices_preoutcome.json",
               {f"{row['mdp_id']}|{row['policy_family']}": row["selected"] for row in rows})
    sealed = ("pilot_features.csv", "oracle_descriptors_preoutcome.csv", "horizon_preoutcome.csv",
              "published_preoutcome.csv", "tqsaco_original_preoutcome.csv",
              "curve_preoutcome.csv", "ceiling_preoutcome.csv",
              "predictions_preoutcome.csv", "forecast_tuning_preoutcome.json",
              "v5_choices_preoutcome.json", "oeh_choices_preoutcome.json")
    write_json(OUTPUT / "preoutcome_seal.json", {
        "protocol_sha256": sha256(OUTPUT / "protocol.json"),
        "dataset_seal_sha256": sha256(OUTPUT / "dataset_seal.json"),
        "learner_outcomes_observed": 0, "sealed_at_utc": now(),
        "artifact_sha256": {name: sha256(OUTPUT / name) for name in sealed}})
    print("Sealed all fourth-test forecasts and learner choices before any outcome", flush=True)


def evaluate_core_shard(args) -> None:
    doc, manifest = base.verified_preoutcome()
    pipeline, _, _, seed_index = archive_imports()
    _repo_first()
    planned = base.jobs(manifest, seed_index)
    plan_path = OUTPUT / "execution_plan.json"
    if not plan_path.exists():
        write_json(plan_path, {"protocol_sha256": sha256(OUTPUT / "protocol.json"),
                               "preoutcome_seal_sha256": sha256(OUTPUT / "preoutcome_seal.json"),
                               "dataset_manifest_sha256": sha256(DATASET / "manifest.json"),
                               "jobs": planned, "job_count": len(planned)})
    plan = load_json(plan_path)
    if plan["jobs"] != planned or plan["preoutcome_seal_sha256"] != sha256(OUTPUT / "preoutcome_seal.json"):
        raise RuntimeError("Core plan differs from the sealed plan")
    plan_sha = sha256(plan_path)
    (OUTPUT / "execution/jobs").mkdir(parents=True, exist_ok=True)
    callback = pipeline.build_core_train_score_callback(DATASET, manifest)
    for index, job in enumerate(planned):
        if index % args.shards != args.shard:
            continue
        path = base.job_path(job)
        if path.exists():
            base.verify_job(job, path, plan_sha)
            continue
        tick = time.perf_counter()
        outcome = callback(job)
        write_json(path, {"job": job, "execution_plan_sha256": plan_sha, "outcome": dict(outcome),
                          "runtime_seconds": time.perf_counter() - tick, "completed_at_utc": now()})
        base.verify_job(job, path, plan_sha)
    print(f"Core shard {args.shard}/{args.shards} complete", flush=True)


def external_plan(manifest: dict, seed_index) -> list[dict]:
    family = {item["instance_id"]: item["family"] for item in manifest["instances"]}
    jobs = []
    for row in manifest["datasets"]:
        if (row.get("stream") != "training" or int(row["size_multiplier"]) != 10
                or family[row["instance_id"]] == "lock_structured"):
            continue
        for method in EXTERNAL:
            jobs.append({"instance_id": row["instance_id"], "policy_family": row["policy_family"],
                         "size_multiplier": 10, "training_seed": seed_index(row), "method": method,
                         "buffer_file": row["file"], "buffer_sha256": row["sha256"]})
    jobs.sort(key=lambda job: (job["instance_id"], job["policy_family"], job["training_seed"], job["method"]))
    if len(jobs) != 420 * 10 * 4:
        raise RuntimeError("Incomplete external x10 plan")
    return jobs


def evaluate_external_shard(args) -> None:
    doc, manifest = base.verified_preoutcome()
    _, external, _, seed_index = archive_imports()
    _repo_first()
    planned = external_plan(manifest, seed_index)
    plan_path = OUTPUT / "external_execution_plan.json"
    if not plan_path.exists():
        write_json(plan_path, {"preoutcome_seal_sha256": sha256(OUTPUT / "preoutcome_seal.json"),
                               "jobs": planned, "job_count": len(planned),
                               "components": external.verify_external_components(ROOT / "tmp")})
    plan = load_json(plan_path)
    if plan["jobs"] != planned:
        raise RuntimeError("External plan differs from the sealed plan")
    plan_sha = sha256(plan_path)
    external.verify_external_components(ROOT / "tmp")
    callback = external.build_external_train_score_callback(DATASET, manifest, ROOT / "tmp")
    target = OUTPUT / "external_execution/jobs"
    target.mkdir(parents=True, exist_ok=True)
    for index, job in enumerate(planned):
        if index % args.shards != args.shard:
            continue
        path = target / base.job_path(job).name
        if path.exists():
            continue
        tick = time.perf_counter()
        outcome = callback(job)
        write_json(path, {"job": job, "execution_plan_sha256": plan_sha, "outcome": dict(outcome),
                          "runtime_seconds": time.perf_counter() - tick, "completed_at_utc": now()})
    print(f"External shard {args.shard}/{args.shards} complete", flush=True)


def oeh_b_helper(args) -> None:
    """Extra OEH-B worker pool (e.g. GPU) sharing claims with ``prepare``."""
    doc = base.protocol()
    base.dataset()
    run_oeh_b(DATASET, OUTPUT / "oeh_b_groups", doc["oeh_final"]["structure"],
              doc["oeh_final"]["requirement_draws"], NAMESPACE, args.workers, args.backend)


def meta(args) -> None:
    """OEH-B and data-only OEH-H for one opened meta-training cohort and arm."""
    structure = json.loads(args.structure)
    spec = OPENED[args.cohort]
    target = DEVELOPMENT / "groups" / args.cohort / args.tag
    complete = target.exists() and len(list(target.glob("*.json"))) == spec["mdps"] * 5
    if not complete:
        run_oeh_b(spec["dataset"], target, structure, args.draws, f"oeh-final-meta|{args.cohort}",
                  args.workers, args.backend)
    if args.backend == "cpu" and not args.skip_horizon:
        run_horizon(spec["dataset"], DEVELOPMENT / f"horizon_data_{args.cohort}_{args.tag}.csv",
                    structure, args.workers, oracle=False)
        run_curve(spec["dataset"], DEVELOPMENT / f"curve_{args.cohort}_{args.tag}.csv",
                  structure, args.draws, args.workers)
        run_ceiling(spec["dataset"], DEVELOPMENT / f"ceiling_{args.cohort}.csv", args.draws, args.workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "oeh-b", "meta", "evaluate-core",
                                          "evaluate-external", "complete-core", "analyze"))
    parser.add_argument("--cohort", choices=sorted(OPENED))
    parser.add_argument("--tag")
    parser.add_argument("--structure", default="{}")
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--skip-horizon", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--selection-workers", type=int, default=4)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args)
    elif args.phase == "prepare":
        prepare(args)
    elif args.phase == "oeh-b":
        oeh_b_helper(args)
    elif args.phase == "meta":
        meta(args)
    elif args.phase == "evaluate-core":
        evaluate_core_shard(args)
    elif args.phase == "evaluate-external":
        evaluate_external_shard(args)
    elif args.phase == "complete-core":
        base.evaluate(resume=True)
    else:
        from scripts.analysis.oeh_final_analysis import analyze
        analyze()


if __name__ == "__main__":
    main()

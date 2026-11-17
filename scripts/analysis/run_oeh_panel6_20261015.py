"""Panel 6 for OEH (task cards T8/T9): two benchmark sources not used before, new questions only.

Families (approved at checkpoint H1 on 2026-10-15; draft ``artifacts/week8/panel6_protocol_draft_2026-10-15``):
* ``mdpp``: MDP Playground 1.0.0 discrete toy MDPs (Rajan et al., 2021);
* ``labmaze``: labmaze 1.0.6 random mazes with four-way slippery moves (dynamics set by the team).
Kernels come from the source libraries in an isolated venv (``scripts/panel6/export_kernels.py``); selection, fingerprints
and the continuing conversion are in ``scripts/analysis/panel6_families.py``. Behaviors, pilot, ten seeds,
BC/CFQI/IQL and budgets x1..x200 are those of panel 5.

Registered questions (fixed sequence; MDP = unit, both families pooled; one-sided max-t with the original SE):
* F1: OEH vs B2 + prefix/FQE on the extrapolation endpoint (``ibs_ge``, ``ll_ext``); GO if both lower bounds > 0;
* F2 (only after F1 GO): OEH vs B2 on the same endpoint;
* F3: decision regret (tau = 0.8, lambda = 100) vs both comparators, registered estimate without a GO/NO-GO verdict.
The panel-5 contrast (OEH vs B2 + diagnostics) is not re-tested. Everything else is descriptive.

Phases:
* ``dryrun``: development draws only (numpy seed 1, never the panel seeds); learners on 12 MDPs per family to check
  that labels are not degenerate (rule fixed before the run: stop if one class holds > 80% of a family's labels);
* ``freeze``: writes ``protocol.json`` and ``protocol.sha256`` before any panel data;
* ``prepare --panel P``: kernels, selection, data, pilot inputs and sealed forecasts (checkpoint H2);
* ``evaluate --panel P``: learners and labels, only after the human authorizes the label step (checkpoint H3);
* ``analyze``: the registered analysis.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts.analysis import panel6_families as F
from scripts.analysis import run_oeh_eighth_cohort_20261014 as E
from scripts.analysis import oeh_pipeline as O

ROOT = E.ROOT
OUTPUT = ROOT / "artifacts/oeh_panel6_2026-10-15"
DATASETS = ROOT / "datasets/oeh_panel6_2026-10-15"
DRYRUN = ROOT / "artifacts/week8/panel6_dryrun_2026-10-15"
DRAFT = ROOT / "artifacts/week8/panel6_protocol_draft_2026-10-15/protocol_draft.json"
NAMESPACE = "oeh-panel6-v1-20261015"
KEYS, P = O.KEYS, O.P
EXTENDED_BUDGETS = (1, 2, 5, 10, 20, 50, 100, 200)
PANELS = {"mdpp": {"mdps": 96, "candidates": 150}, "labmaze": {"mdps": 96, "candidates": 150}}
CANDIDATE_SEEDS = {"mdpp": 20301115, "labmaze": 20301116}
DATA_SEED_BASE = {"mdpp": 70_000_000, "labmaze": 72_000_000}  # draft had 30M/31M: inside earlier training-stream seeds
DRYRUN_SEED_BASE = {"mdpp": 76_000_000, "labmaze": 78_000_000}
SEED_REFERENCE = "datasets/oeh_fifth_panel_2026-10-15/gambler/manifest.json"
BOOTSTRAP = {"draws": 10_000, "seed": 20261216, "rule": "max-t with the original standard error (one-sided)"}
TAU, LAMBDA = 0.8, 100.0
EXPORT_VERIFY_STEPS = 100
DRYRUN_MDPS, DRYRUN_MAX_CLASS_SHARE = 12, 0.80
GATES = {
    "F1": [("b2_plus_prefix_fqe", "ibs_ge"), ("b2_plus_prefix_fqe", "ll_ext")],
    "F2": [("b2_matched", "ibs_ge"), ("b2_matched", "ll_ext")],
}
REGISTERED_ESTIMATE = {"F3": [("b2_plus_prefix_fqe", "regret"), ("b2_matched", "regret")]}
DESCRIPTIVE_METHODS = {
    "oeh": "OEH (final method, oeh_v2_spec.json)", "oeh_full_head": "OEH-F, full 108-field head (no shift gate)",
    "robust": "B2 + diagnostics (robust head)", "b2_matched": "B2", "xgboost_cf_18": "XGBoost-CF on B2",
    "xgboost_cf_108": "XGBoost-CF on the 108 fields", "prefix_fqe_raw": "prefix + FQE, zero-shot law (T2)",
    "b2_plus_prefix_fqe": "B2 + prefix/FQE (T2)", "lce_raw": "LCE", "b2_plus_lce": "B2 + LCE", "b2_plus_cb": "B2 + CB",
    "knn_analog": "k-NN analog on B2 (k = 25)", "logistic": "multinomial logistic regression on B2",
    "s4": "T4 candidate S4 (B2 + P + D, one head, gated); descriptive only", "oeh_without_horizon": "OEH without the OEH-H block",
}
S4_GATE = {"lambda_max": 1.0, "eps_max": 0.02, "d0": 3.0}


def panel_dir(panel: str) -> Path:
    return OUTPUT / panel


def protocol() -> dict:
    if E.sha256(OUTPUT / "protocol.json") != (OUTPUT / "protocol.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("protocol changed after freeze")
    return E.load_json(OUTPUT / "protocol.json")


# --------------------------------------------------------------------------- #
# Kernels and data
# --------------------------------------------------------------------------- #

def select(panel: str, items: list[dict], raw: Path, kernels: Path, mdps: int) -> tuple[list[str], list[dict]]:
    """First ``mdps`` candidates in list order passing the registered checks; continuing kernels go to ``kernels``."""
    from scripts.analysis.run_oeh_final_120 import seen_kernels

    seen, earlier = set(seen_kernels()), F.earlier_signatures()
    files = sum(map(len, F.earlier_kernel_files().values()))
    if files < 800:  # amendment A1: count kernel files (884), not distinct signatures (749; Colosseum 2029 = v2test)
        raise RuntimeError(f"only {files} earlier external kernel files found; set PANEL6_EARLIER_ROOT to the main checkout")
    kernels.mkdir(parents=True, exist_ok=True)
    chosen, log, signatures = [], [], set()
    for c in items:
        if len(chosen) == mdps:
            break
        T, R, rho = F.load(raw / f"{c['mdp_id']}.npz")
        S, A = R.shape
        sig, fp = F.signature(T, R, rho), F.fingerprint(T, R, rho)
        reason = ("size" if S * A > F.SA_CAP else "seen_exact" if fp in seen else "earlier_panel" if sig in earlier
                  else "duplicate" if sig in signatures else "")
        if not reason:
            np.savez(kernels / f"{c['mdp_id']}.npz", T=T, R=R, rho=rho)
            value, passes = F.dynamic_range(kernels, c["mdp_id"])
            if not passes:
                (kernels / f"{c['mdp_id']}.npz").unlink()
                reason = "dynamic_range"
        log.append({**{k: v for k, v in c.items() if k != "family"}, "states": S, "actions_count": A, "signature": sig,
                    "fingerprint": fp, "selected": not reason, "reason": reason})
        if not reason:
            chosen.append(c["mdp_id"])
            signatures.add(sig)
    if len(chosen) != mdps:
        raise RuntimeError(f"{panel}: only {len(chosen)} candidates pass")
    return chosen, log


def build_dataset(panel: str, ids: list[str], kernels: Path, dataset: Path, seed_base: int, reason: str) -> dict:
    from scripts.analysis.run_unseen_validation import archive_imports
    _, _, generator, _ = archive_imports()
    from environments.benchmark_environments import BenchmarkSpec
    from environments.tabular_environments import TabularMDP

    registry, specs = {}, []
    for i, mdp_id in enumerate(ids):
        registry[mdp_id] = E.colosseum_mdp(str(kernels), mdp_id, TabularMDP)
        specs.append(BenchmarkSpec(instance_id=mdp_id, split="evaluation", family=panel, seed=seed_base + 1_000 * i,
                                   ordinal=i, family_ordinal=i))
    generator.resolve_dynamic_range = lambda s: (s, registry[s.instance_id], 0)
    generator.SIZE_MULTIPLIERS = EXTENDED_BUDGETS
    return generator.generate_benchmark_bundle(
        dataset, specs=specs, pilot_seeds=(0,), training_seeds=tuple(range(10)), max_size=None, horizon=100,
        run_mode="test", deviation_reason=reason, pilot_size_multipliers=(2, 5, 10))


def earlier_seeds() -> set[int]:
    """Instance, policy and stream seeds of every earlier dataset manifest readable on this machine."""
    out = set()
    for path in (Path(os.environ.get("PANEL6_EARLIER_ROOT", ROOT)) / "datasets").glob("**/manifest.json"):
        if DATASETS in path.parents:
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue  # Git LFS pointers
        if not isinstance(doc, dict):
            continue
        for item in doc.get("instances", []):
            if isinstance(item, dict) and isinstance(item.get("seed"), int):
                out.add(item["seed"])
        for row in doc.get("datasets", []):
            if isinstance(row, dict):
                out.update(v for k in ("mdp_seed", "policy_seed", "stream_seed") if isinstance(v := row.get(k), int))
    return out


def seed_offsets() -> set[int]:
    """Offsets of every derived seed (policy, pilot, training streams) from the instance seed, read from panel 5's manifest
    (the generator's convention)."""
    doc = json.loads((Path(os.environ.get("PANEL6_EARLIER_ROOT", ROOT)) / SEED_REFERENCE).read_text(encoding="utf-8"))
    first = doc["instances"][0]
    rows = [r for r in doc["datasets"] if r["instance_id"] == first["instance_id"]]
    return {0} | {r[k] - first["seed"] for r in rows for k in ("mdp_seed", "policy_seed", "stream_seed") if isinstance(r.get(k), int)}


def planned_seeds(base: int, count: int) -> set[int]:
    offsets = seed_offsets()
    return {base + 1_000 * i + o for i in range(count) for o in offsets}


def panel_seed_plan() -> set[int]:
    return set().union(*(planned_seeds(DATA_SEED_BASE[f], PANELS[f]["candidates"]) for f in PANELS))


def generated_seeds(manifest: dict) -> set[int]:
    out = {i["seed"] for i in manifest["instances"]}
    for row in manifest["datasets"]:
        out.update(row[k] for k in ("mdp_seed", "policy_seed", "stream_seed") if isinstance(row.get(k), int))
    return out


def learner_labels(dataset: Path, kernels: Path, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scripts.analysis.run_unseen_validation import archive_imports
    from scripts.analysis.compare_difficulty import suffix_event
    from scripts.analysis import run_oeh_final_120 as R

    _, _, _, seed_index = archive_imports()
    R._repo_first()
    manifest = E.load_json(dataset / "manifest.json")
    instances = {i["instance_id"]: i for i in manifest["instances"]}
    by_mdp = defaultdict(list)
    for row in manifest["datasets"]:
        if row.get("stream") == "training":
            by_mdp[row["instance_id"]].append({**row, "seed_index": seed_index(row)})
    if sum(map(len, by_mdp.values())) != len(instances) * 5 * len(EXTENDED_BUDGETS) * 10:
        raise RuntimeError("incomplete training grid")
    source = str(kernels)
    tasks = [{"instance": instances[m], "dataset": str(dataset), "colosseum": source, "kernels": source,
              "records": sorted(r, key=lambda x: (x["policy_family"], x["size_multiplier"], x["seed_index"]))}
             for m, r in sorted(by_mdp.items())]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outcomes = pd.DataFrame([row for result in pool.map(E._mdp_jobs, tasks) for row in result])
    counts = outcomes.groupby(["mdp_id", "policy_family", "algorithm", "budget"]).success.agg(["sum", "size"])
    if (counts["size"] != 10).any():
        raise RuntimeError("ten seeds required")
    wide = counts["sum"].unstack("budget").reindex(columns=list(EXTENDED_BUDGETS))
    labels = wide.reset_index()
    labels["event"] = [suffix_event(tuple(int(x) for x in r[:6])) for r in wide.to_numpy()]
    labels["event_extended"] = [extended_event(tuple(int(x) for x in r)) for r in wide.to_numpy()]
    return outcomes, labels.rename(columns={b: f"successes_x{b}" for b in EXTENDED_BUDGETS})


def extended_event(counts) -> int:
    for index in range(len(counts)):
        if min(counts[index:]) >= 8:
            return index
    return len(counts)


# --------------------------------------------------------------------------- #
# Dry run on development draws (approved at H1)
# --------------------------------------------------------------------------- #

def dryrun(workers: int) -> dict:
    if (DRYRUN / "manifest.json").exists():
        raise FileExistsError("dry run already recorded")
    DRYRUN.mkdir(parents=True, exist_ok=True)
    frames, versions = [], {}
    for panel in PANELS:
        work = F.WORK / "dryrun" / panel
        items = F.candidates(panel, DRYRUN_MDPS, F.DEV_SEED, "dry_")
        meta = F.export(items, work / "episodic", 0)
        versions[panel] = {key: v for key, v in meta.items() if key != "kernels"}
        kernels = work / "kernels"
        kernels.mkdir(parents=True, exist_ok=True)
        for c in items:
            T, R, rho = F.load(work / "episodic" / f"{c['mdp_id']}.npz")
            np.savez(kernels / f"{c['mdp_id']}.npz", T=T, R=R, rho=rho)
        dataset = work / "dataset"
        if not (dataset / "manifest.json").exists():
            if planned_seeds(DRYRUN_SEED_BASE[panel], DRYRUN_MDPS) & (earlier_seeds() | panel_seed_plan()):
                raise RuntimeError("dry-run seeds overlap earlier or panel seeds")
            build_dataset(panel, [c["mdp_id"] for c in items], kernels, dataset, DRYRUN_SEED_BASE[panel],
                          f"OEH panel 6 dry run on development draws: {panel}")
        _, labels = learner_labels(dataset, kernels, workers)
        frames.append(labels.assign(family=panel))
    labels = pd.concat(frames, ignore_index=True)
    labels.to_csv(DRYRUN / "labels.csv", index=False)
    share = labels.groupby("family").event.value_counts(normalize=True).unstack(fill_value=0.0)
    share.columns = [f"class_{c}" for c in share.columns]
    share["max_share"] = share.max(1)
    share.reset_index().to_csv(DRYRUN / "class_shares.csv", index=False)
    by_learner = labels.groupby(["family", "algorithm"]).event.value_counts().unstack(fill_value=0)
    by_learner.reset_index().to_csv(DRYRUN / "class_counts_by_learner.csv", index=False)
    degenerate = sorted(share.index[share.max_share > DRYRUN_MAX_CLASS_SHARE])
    doc = {"status": "development_draws_only", "rule": f"stop before freeze if the most frequent class holds more than "
           f"{DRYRUN_MAX_CLASS_SHARE:.0%} of a family's labels (fixed before the run)",
           "mdps_per_family": DRYRUN_MDPS, "dev_seed": F.DEV_SEED, "degenerate_families": degenerate,
           "verdict": "STOP" if degenerate else "PROCEED", "versions": versions, "created_utc": E.now(),
           "forecasts_scored": 0,
           "outputs_sha256": {n: E.sha256(DRYRUN / n) for n in ("labels.csv", "class_shares.csv", "class_counts_by_learner.csv")},
           "source_sha256": {E.rel(Path(__file__)): E.sha256(Path(__file__))}}
    E.write_json(DRYRUN / "manifest.json", doc)
    return doc


# --------------------------------------------------------------------------- #
# Freeze
# --------------------------------------------------------------------------- #

def freeze() -> None:
    if OUTPUT.exists() or DATASETS.exists():
        raise FileExistsError("namespace exists")
    dry = E.load_json(DRYRUN / "manifest.json")
    if dry["verdict"] != "PROCEED":
        raise RuntimeError("the dry run asked to stop before freeze")
    draft = E.load_json(DRAFT)
    seeds = earlier_seeds()
    plan = {f: planned_seeds(DATA_SEED_BASE[f], PANELS[f]["mdps"]) for f in PANELS}
    if (plan["mdpp"] & plan["labmaze"]) or any(p & seeds for p in plan.values()):
        raise RuntimeError("planned seeds overlap each other or an earlier manifest")
    families = {f: {**F.FAMILIES[f], **PANELS[f], "candidate_seed": CANDIDATE_SEEDS[f], "data_seed_base": DATA_SEED_BASE[f],
                    "candidates_list": F.candidates(f, PANELS[f]["candidates"], CANDIDATE_SEEDS[f], "p6_"),
                    "library": draft["families"][f]["library"]} for f in PANELS}
    doc = {
        "schema": "oeh-panel6-v1", "namespace": NAMESPACE, "frozen_at_utc": E.now(),
        "status": "frozen_before_any_panel_data_or_learner_outcomes",
        "approval": "checkpoint H1 approved by Linh on 2026-10-15 with every recommendation of the T8 draft",
        "method": draft["method"], "oeh_spec": O.load_spec(), "oeh_spec_sha256": E.sha256(O.SPEC),
        "families": families, "kernel_source": draft["kernel_source"], "export_verify_steps": EXPORT_VERIFY_STEPS,
        "conversion": draft["conversion"], "selection": draft["selection"], "data": draft["data"],
        "questions": draft["questions"], "endpoints": draft["endpoints"],
        "gates": {"order": "fixed sequence F1 -> F2; F2 is tested only if F1 is GO",
                  "unit": draft["gates"]["unit"], "bootstrap": BOOTSTRAP,
                  **{g: {"contrasts": [[c, "oeh", m] for c, m in v], "go": "both simultaneous lower bounds > 0"} for g, v in GATES.items()}},
        "registered_estimates": {g: {"contrasts": [[c, "oeh", m] for c, m in v], "tau": TAU, "lambda": LAMBDA,
                                     "report": "mean effect and simultaneous one-sided 95% lower bounds; no GO/NO-GO verdict "
                                               "(H1: underpowered as a gate)"} for g, v in REGISTERED_ESTIMATE.items()},
        "not_tested": draft["not_tested"],
        "sample_size": draft["sample_size"],
        "descriptive": {"methods": DESCRIPTIVE_METHODS, "also": draft["descriptive"]["also"],
                        "conditional_methods": {"tabpfn": "TabPFN v2 on B2 (task card T3): predictions are added and sealed "
                                                          "separately before the label step (checkpoint H3) if the licence "
                                                          "token is available; otherwise reported as not run"},
                        "dropped": {"lc_pfn": "not used: needs Python < 3.12 and torch <= 1.11, and fits a curve to only "
                                              "four pilot prefixes (H1 decision)"}},
        "dryrun": {"path": E.rel(DRYRUN), "manifest_sha256": E.sha256(DRYRUN / "manifest.json"), "verdict": dry["verdict"],
                   "rule": dry["rule"]},
        "seed_check": {"earlier_seeds_read": len(seeds), "planned_seeds": {f: len(p) for f, p in plan.items()}, "overlap": 0,
                       "offsets_from": SEED_REFERENCE,
                       "change_from_draft": "data seed bases moved from 30M/31M to 70M/72M: 30M+ holds the training-stream "
                                            "seeds of earlier panels (instance seed + 20,000,000); the draft required this check"},
        "disclosures": draft["disclosures"] + [
            f"Dry run (approved at H1): learners were run on {DRYRUN_MDPS} development MDPs per family drawn with numpy seed 1 "
            "to check the label distribution; no forecast was scored on them (artifacts/week8/panel6_dryrun_2026-10-15)."],
        "draft_sha256": E.sha256(DRAFT),
        "source_sha256": {E.rel(p): E.sha256(p) for p in (Path(__file__), F.EXPORTER, ROOT / "scripts/analysis/panel6_families.py",
                                                         ROOT / "scripts/analysis/oeh_pipeline.py", ROOT / "scripts/analysis/oeh_v2_forecast.py",
                                                         ROOT / "scripts/analysis/baseline_prefix_fqe.py",
                                                         ROOT / "scripts/analysis/extrapolation_endpoint.py",
                                                         ROOT / "scripts/analysis/simplified_candidates_t4.py",
                                                         ROOT / "scripts/analysis/oeh_generic_baselines.py")},
    }
    OUTPUT.mkdir(parents=True)
    E.write_json(OUTPUT / "protocol.json", doc)
    (OUTPUT / "protocol.sha256").write_text(E.sha256(OUTPUT / "protocol.json") + "\n", encoding="ascii")
    print(f"frozen panel-6 protocol {E.sha256(OUTPUT / 'protocol.json')}", flush=True)


# --------------------------------------------------------------------------- #
# Forecasts
# --------------------------------------------------------------------------- #

def prefix_fqe(dataset: Path, query: pd.DataFrame, target: Path, workers: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Raw and calibrated prefix + FQE laws (task card T2, unchanged), with the meta-training features of T2."""
    from scripts.analysis import baseline_prefix_fqe as B
    from scripts.analysis import oeh_final_forecast as Fc
    from scripts.analysis import run_oeh_final_120 as R
    from scripts.analysis.compare_difficulty import FEATURES_B2
    from scripts.analysis.develop_cohort8_candidates_20261014 import meta_panel

    out = target / "prefix_fqe_features.csv.gz"
    if not out.exists():
        tasks = [{"row": row, "dataset": str(dataset)} for row in R.pilot_rows(dataset)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pd.DataFrame([r for part in pool.map(B._task, tasks, chunksize=4) for r in part]).to_csv(out, index=False)
    feats = pd.read_csv(out)
    meta = {p.name.removesuffix(".csv.gz"): pd.read_csv(p) for p in sorted((B.OUTPUT / "features").glob("meta_*.csv.gz"))}
    meta["meta_fourth120"]["mdp_id"] = "opened_fourth120/" + meta["meta_fourth120"]["mdp_id"].astype(str)
    t = meta_panel(sort=False).merge(pd.concat(meta.values(), ignore_index=True)[KEYS + list(B.FEATURES)], on=KEYS,
                                      how="left", validate="many_to_one")
    if t[B.FEATURES[0]].isna().any():
        raise RuntimeError("meta-training prefix/FQE features incomplete")
    qq = query[KEYS + list(FEATURES_B2)].merge(feats, on=KEYS, validate="one_to_one")
    if len(qq) != len(query):
        raise RuntimeError("prefix/FQE features missing for some rows")
    qq = query[KEYS].merge(qq, on=KEYS, validate="one_to_one")  # query row order
    qq["panel"] = "primary"
    calibrated, _ = Fc.matched_forecast(t, qq, tuple(FEATURES_B2) + B.FEATURES)
    return qq[[f"raw_p_{i}" for i in range(7)]].to_numpy(float), calibrated, feats


def s4_law(train: pd.DataFrame, query: pd.DataFrame, d: np.ndarray, workers: int, work: Path) -> np.ndarray:
    """T4 candidate S4 exactly as in ``simplified_candidates_t4`` (single multiclass heads, locked gate)."""
    from scripts.analysis import develop_oeh_v2_heads_20261014 as H
    from scripts.analysis import oeh_v2_forecast as V
    from scripts.analysis.simplified_candidates_t4 import field_sets

    sets = field_sets()
    work.mkdir(parents=True, exist_ok=True)
    tpath, qpath = work / "_train_s4.pkl", work / "_query_s4.pkl"
    train.to_pickle(tpath)
    query.to_pickle(qpath)
    specs = [("B2+P+D", False), ("B2+P+D", True), ("B2+D", False)]
    tasks = [{"name": f"{n}{'|fin' if fin else ''}", "fields": list(sets[n]), "kind": "multiclass", "finite_only": fin,
              "train": str(tpath), "query": str(qpath)} for n, fin in specs]
    try:
        with ProcessPoolExecutor(max_workers=min(workers, 3)) as pool:
            heads = {name: p for name, p, _ in pool.map(H._fit, tasks)}
    finally:
        tpath.unlink(missing_ok=True)
        qpath.unlink(missing_ok=True)
    pi = np.stack([V.class_prior(train)[a] for a in query.algorithm])
    return V.v2_law(heads["B2+P+D"], heads["B2+P+D|fin"], heads["B2+D"], d, pi, **S4_GATE)


def forecasts(dataset: Path, query: pd.DataFrame, target: Path, workers: int) -> pd.DataFrame:
    from scripts.analysis import oeh_generic_baselines as G
    from scripts.analysis.develop_cohort8_candidates_20261014 import lce_law

    spec = O.load_spec()
    train = O.training_panel(spec)
    sets = E.forecast_sets()
    extra = {name: (sets[name], False) for name in ("b2_matched", "b2_plus_eri", "xgboost_cf_18", "xgboost_cf_108", "b2_plus_lce",
                                                     "b2_plus_cb", "oeh_without_oehh_risk")}
    extra["oeh_without_oehh_risk_finite"] = (sets["oeh_without_oehh_risk"], True)
    tables = O.fit_heads(train, query, workers, target, extra)
    tuning = tables.pop("_tuning")
    law, d = O.final_law(tables, train, query, spec)
    no_h, _ = O.final_law(tables, train, query, spec, full="oeh_without_oehh_risk", full_finite="oeh_without_oehh_risk_finite")
    raw, calibrated, _ = prefix_fqe(dataset, query, target, workers)
    laws = {"oeh": law, "oeh_full_head": tables["full"], "robust": tables["robust"], "b2_matched": tables["b2_matched"],
            "xgboost_cf_18": tables["xgboost_cf_18"], "xgboost_cf_108": tables["xgboost_cf_108"], "prefix_fqe_raw": raw,
            "b2_plus_prefix_fqe": calibrated, "lce_raw": lce_law(query), "b2_plus_lce": tables["b2_plus_lce"],
            "b2_plus_cb": tables["b2_plus_cb"], "knn_analog": G.knn_analog(train, query), "logistic": G.logistic(train, query),
            "s4": s4_law(train, query, d, workers, target), "oeh_without_horizon": no_h}
    if set(laws) != set(DESCRIPTIVE_METHODS):
        raise RuntimeError("forecast set differs from the protocol")
    E.write_json(target / "forecast_tuning_preoutcome.json", tuning)
    E.check_sealed_tuning({"oeh_final": tuning["full"], "b2_matched": tuning["b2_matched"], "b2_plus_eri": tuning["b2_plus_eri"],
                           "prior_oeh_matched": tuning["robust"]}, target)
    frames = []
    for method, probability in laws.items():
        frame = query[KEYS].copy()
        frame["method"], frame["distance"] = method, d
        frame[P] = probability
        frames.append(frame)
    pred = pd.concat(frames, ignore_index=True)
    p = pred[P].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-9):
        raise RuntimeError("invalid seven-class forecast")
    return pred


def prepare(panel: str, workers: int) -> None:
    doc = protocol()
    target, dataset = panel_dir(panel), DATASETS / panel
    if (target / "preoutcome_seal.json").exists():
        raise FileExistsError(f"{panel} already sealed")
    target.mkdir(parents=True, exist_ok=True)
    fam = doc["families"][panel]
    if not (target / "dataset_seal.json").exists():
        raw = F.WORK / NAMESPACE / panel / "episodic"
        meta = F.export(fam["candidates_list"], raw, doc["export_verify_steps"])
        bad = [m for m, e in meta["kernels"].items() if "verify" in e and not (e["verify"]["support_ok"] and e["verify"]["bonferroni_p"] > 1e-3
                                                                               and e["verify"]["max_abs_reward_error"] == 0)]
        if bad:
            raise RuntimeError(f"exported kernels disagree with the source environment: {bad[:5]}")
        chosen, log = select(panel, fam["candidates_list"], raw, target / "kernels", fam["mdps"])
        manifest = build_dataset(panel, chosen, target / "kernels", dataset, fam["data_seed_base"], f"OEH panel 6: {panel}")
        overlap = generated_seeds(manifest) & earlier_seeds()
        if overlap:
            raise RuntimeError(f"generated seeds overlap earlier manifests: {sorted(overlap)[:5]}")
        pd.DataFrame(log).to_csv(target / "selection.csv", index=False)
        E.write_json(target / "export_meta.json", {k: v for k, v in meta.items() if k != "kernels"} |
                     {"kernels": {m: {k: v for k, v in e.items() if k != "source"} for m, e in meta["kernels"].items()}})
        E.write_json(target / "dataset_seal.json", {
            "protocol_sha256": E.sha256(OUTPUT / "protocol.json"), "sealed_at_utc": E.now(), "learner_outcomes_observed": 0,
            "instances": manifest["instance_count"], "buffers": manifest["physical_buffer_count"],
            "selection_sha256": E.sha256(target / "selection.csv"), "export_meta_sha256": E.sha256(target / "export_meta.json"),
            "kernels_sha256": {p.stem: E.sha256(p) for p in sorted((target / "kernels").glob("*.npz"))},
            "sha256": {E.rel(p): E.sha256(p) for p in (dataset / "manifest.json", dataset / "data_only_index.json")}})
    query = O.pilot_inputs(dataset, target, workers, target / "kernels", "colosseum", f"{NAMESPACE}|{panel}")
    seal = E.load_json(target / "dataset_seal.json")
    if len(query) != seal["instances"] * 5 * 3:
        raise RuntimeError(f"incomplete {panel} query: {len(query)}")
    query.to_csv(target / "query_preoutcome.csv.gz", index=False)
    pred = forecasts(dataset, query, target, workers)
    pred.to_csv(target / "predictions_preoutcome.csv", index=False)
    names = ["pilot_features.csv", "published.csv", "horizon.csv", "curve.csv", "query_preoutcome.csv.gz", "prefix_fqe_features.csv.gz",
             "predictions_preoutcome.csv", "forecast_tuning_preoutcome.json", "selection.csv", "export_meta.json"]
    groups = sorted((target / "oeh_b_groups").glob("*.json"))
    E.write_json(target / "preoutcome_seal.json", {
        "panel": panel, "protocol_sha256": E.sha256(OUTPUT / "protocol.json"), "learner_outcomes_observed": 0, "sealed_at_utc": E.now(),
        "artifact_sha256": {n: E.sha256(target / n) for n in names}, "oeh_b_groups": len(groups),
        "oeh_b_groups_sha256": hashlib.sha256("".join(E.sha256(p) for p in groups).encode()).hexdigest()})
    print(f"sealed {panel}: {len(pred)} predictions", flush=True)


# --------------------------------------------------------------------------- #
# Labels (after H3) and the registered analysis
# --------------------------------------------------------------------------- #

def evaluate(panel: str, workers: int) -> None:
    target, dataset = panel_dir(panel), DATASETS / panel
    seal = E.load_json(target / "preoutcome_seal.json")
    for name, digest in seal["artifact_sha256"].items():
        if E.sha256(target / name) != digest:
            raise RuntimeError(f"sealed file changed: {name}")
    if seal["protocol_sha256"] != E.sha256(OUTPUT / "protocol.json"):
        raise RuntimeError("protocol changed after the seal")
    if (target / "outcomes.csv.gz").exists():
        raise FileExistsError("outcomes exist")
    tick = time.perf_counter()
    outcomes, labels = learner_labels(dataset, target / "kernels", workers)
    outcomes.to_csv(target / "outcomes.csv.gz", index=False)
    labels.to_csv(target / "labels.csv", index=False)
    E.write_json(target / "outcome_run.json", {"jobs": len(outcomes), "seconds": time.perf_counter() - tick, "completed_at_utc": E.now(),
                                               "preoutcome_seal_sha256": E.sha256(target / "preoutcome_seal.json"),
                                               "labels_sha256": E.sha256(target / "labels.csv")})
    print(f"{panel}: {len(outcomes)} jobs, {len(labels)} labels", flush=True)


def scored() -> pd.DataFrame:
    from scripts.analysis import extrapolation_endpoint as X
    from scripts.analysis.colosseum_posthoc_20261014 import decision_costs

    frames = []
    for panel in PANELS:
        target = panel_dir(panel)
        run = E.load_json(target / "outcome_run.json")
        if run["preoutcome_seal_sha256"] != E.sha256(target / "preoutcome_seal.json"):
            raise RuntimeError("seal changed after outcomes")
        m = pd.read_csv(target / "predictions_preoutcome.csv").merge(
            pd.read_csv(target / "labels.csv")[KEYS + ["event", "event_extended"]], on=KEYS, validate="many_to_one")
        p, y = m[P].to_numpy(float), m.event.to_numpy(int)
        for k, v in X.row_scores(p, y).items():
            m[k] = v
        m["regret"] = decision_costs(np.cumsum(p, 1)[:, :6], y, TAU, LAMBDA)["regret"]
        m["source_panel"] = panel
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def strict_event(outcomes: pd.DataFrame, threshold: float, k: int = 8) -> pd.DataFrame:
    """Label variant (task card T6): success at NR >= ``threshold`` on the first six budgets, k of 10 seeds."""
    from scripts.analysis.threshold_sensitivity import counts, suffix_events

    c = counts(outcomes, threshold)
    out = c.reset_index()[KEYS]
    out["event"] = suffix_events(c.to_numpy(int), k)
    return out


def family_test(per: pd.DataFrame, contrasts) -> dict:
    from scripts.analysis.colosseum_posthoc_20261014 import max_t

    effect = np.column_stack([per[(m, c)] - per[(m, "oeh")] for c, m in contrasts])
    mean, lower, crit = max_t(effect, BOOTSTRAP["seed"], BOOTSTRAP["draws"])
    return {"contrasts": [{"comparator": c, "metric": m, "mean_improvement": float(mean[i]), "simultaneous_lower_95": float(lower[i])}
                          for i, (c, m) in enumerate(contrasts)], "critical": float(crit), "mdps": int(len(effect)),
            "all_lower_bounds_positive": bool(all(lower > 0))}


def analyze() -> dict:
    from scripts.analysis.cross_panel_reanalysis_20261014 import murphy

    doc = protocol()
    s = scored()
    s.to_csv(OUTPUT / "scored.csv.gz", index=False)
    metrics = ["ibs", "ll", "ibs_ge", "ibs_gt", "ll_ext", "regret"]
    per_mdp = s.groupby(["source_panel", "method", "mdp_id"])[metrics].mean().reset_index()
    wide = per_mdp.pivot_table(index=["source_panel", "mdp_id"], columns="method", values=metrics)
    gates, go = {}, True
    for name, contrasts in GATES.items():
        if not go:
            gates[name] = {"tested": False, "reason": "an earlier family in the fixed sequence was NO-GO"}
            continue
        result = family_test(wide, contrasts)
        result["verdict"] = "GO" if result["all_lower_bounds_positive"] else "NO-GO"
        gates[name] = {"tested": True, **result}
        go = result["verdict"] == "GO"
    estimates = {name: family_test(wide, contrasts) for name, contrasts in REGISTERED_ESTIMATE.items()}
    rows = []
    for scope, panels in [(p, (p,)) for p in PANELS] + [("pooled", tuple(PANELS))]:
        sub = per_mdp[per_mdp.source_panel.isin(panels)]
        w = sub.pivot_table(index=["source_panel", "mdp_id"], columns="method", values=metrics)
        for method in DESCRIPTIVE_METHODS:
            row = {"scope": scope, "method": method, **{m: float(w[(m, method)].mean()) for m in metrics}}
            for m in metrics:
                g = (w[(m, method)] - w[(m, "oeh")]).to_numpy()
                half = 1.96 * g.std(ddof=1) / np.sqrt(len(g))
                row[f"{m}_gain_of_oeh"], row[f"{m}_low95"], row[f"{m}_high95"] = g.mean(), g.mean() - half, g.mean() + half
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT / "summary.csv", index=False)
    s.groupby(["source_panel", "algorithm", "method", "mdp_id"])[metrics].mean().groupby(
        ["source_panel", "algorithm", "method"]).mean().reset_index().to_csv(OUTPUT / "by_learner.csv", index=False)
    pd.DataFrame([{"panel": panel, **r} for panel in PANELS
                  for r in murphy(s[(s.source_panel == panel) & s.method.isin(["oeh", "robust", "oeh_full_head", "b2_matched", "b2_plus_prefix_fqe"])])]
                 ).to_csv(OUTPUT / "murphy.csv", index=False)
    d = s[s.method == "oeh"].groupby(["source_panel", "mdp_id"]).distance.mean().reindex(wide.index)
    bins = pd.qcut(d, 4, labels=["q1", "q2", "q3", "q4"]).to_numpy()
    pd.DataFrame([{"distance_quartile": b, "method": c, "mdps": int((bins == b).sum()),
                   **{m: float((wide[(m, c)][bins == b] - wide[(m, "oeh")][bins == b]).mean()) for m in metrics}}
                  for b in ("q1", "q2", "q3", "q4") for c in ("b2_matched", "b2_plus_prefix_fqe", "robust")]
                 ).to_csv(OUTPUT / "gain_by_distance.csv", index=False)
    strict = []
    for panel in PANELS:
        outcomes = pd.read_csv(panel_dir(panel) / "outcomes.csv.gz")
        lab = strict_event(outcomes, 0.99)
        m = pd.read_csv(panel_dir(panel) / "predictions_preoutcome.csv").merge(lab, on=KEYS, validate="many_to_one")
        p, y = m[P].to_numpy(float), m.event.to_numpy(int)
        m["ibs"] = ((np.cumsum(p, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)
        m["ll"] = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
        strict.append(m.assign(source_panel=panel))
    st = pd.concat(strict).groupby(["method", "source_panel", "mdp_id"])[["ibs", "ll"]].mean().groupby("method").mean().reset_index()
    st.to_csv(OUTPUT / "label_variant_tau099.csv", index=False)
    result = {"completed_at_utc": E.now(), "protocol_sha256": E.sha256(OUTPUT / "protocol.json"), "gates": gates,
              "registered_estimates": estimates,
              "events": {f"{k[0]}|{k[1]}": v for k, v in s[s.method == "oeh"].groupby(["source_panel", "algorithm"]).event
                         .value_counts().unstack(fill_value=0).to_dict("index").items()}}
    E.write_json(OUTPUT / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("dryrun", "freeze", "prepare", "evaluate", "analyze"))
    parser.add_argument("--panel", choices=tuple(PANELS))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.phase == "dryrun":
        print(json.dumps({k: v for k, v in dryrun(args.workers).items() if k in ("verdict", "degenerate_families")}))
    elif args.phase == "freeze":
        freeze()
    elif args.phase == "prepare":
        prepare(args.panel, args.workers)
    elif args.phase == "evaluate":
        evaluate(args.panel, args.workers)
    else:
        result = analyze()
        for name, g in result["gates"].items():
            print(name, g.get("verdict", "not tested"))


if __name__ == "__main__":
    main()

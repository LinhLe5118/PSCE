"""Sealed test of OEH v2 (handover plan KE_HOACH_OEH_V2, Section 6).

Four new panels, one protocol frozen before any of their data exists:

* ``internal``       168 new stochastic MDPs of the registered recipe (five registered behaviors);
* ``colosseum2029``  72 new Colosseum kernels (authors' sampler, seed 2029, earlier kernels excluded);
* ``spibb2``         108 new SPIBB random MDPs with the authors' baseline behavior (author code);
* ``forest``         48 forest-management MDPs from MDPToolbox (Chades et al. 2014), five registered behaviors.

Training streams carry the extended grid x1..x200: the primary endpoint uses the
first six budgets (identical prefixes, verified), the extended endpoint all eight.
OEH v2 is frozen in ``oeh_v2_spec.json`` (parameters chosen on opened data).

Phases: ``freeze``; ``prepare --panel P`` (data, pilot-only inputs, forecasts, seal);
``evaluate --panel P`` (learners, only after that panel's seal); ``analyze``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts.analysis import run_oeh_eighth_cohort_20261014 as E
from scripts.analysis import run_spibb_panel_20261014 as S

ROOT = E.ROOT
OUTPUT = ROOT / "artifacts/oeh_v2_sealed_test_2026-10-14"
DATASETS = ROOT / "datasets/oeh_v2_sealed_test_2026-10-14"
NAMESPACE = "oeh-v2-sealed-test-v1-20261014"
SPEC = ROOT / "artifacts/week8/oeh_v2_development_2026-10-14/oeh_v2_spec.json"
KEYS, P = E.KEYS, E.P
PRIMARY_BUDGETS = (1, 2, 5, 10, 20, 50)
EXTENDED_BUDGETS = (1, 2, 5, 10, 20, 50, 100, 200)
EXTERNAL = ("colosseum2029", "spibb2", "forest")
PANELS = {"internal": {"mdps": 168, "behaviors": 5}, "colosseum2029": {"mdps": 72, "behaviors": 5},
          "spibb2": {"mdps": 108, "behaviors": 1}, "forest": {"mdps": 48, "behaviors": 5}}
BOOTSTRAP = {"draws": 10_000, "seed": 20261101, "rule": "max-t with the original standard error"}
STEPS = {
    "step1": {"what": "OEH v2 vs B2 + legacy diagnostics, pooled external MDPs", "pairs": [["legacy", "v2", "external"]]},
    "step2": {"what": "OEH v2 vs OEH current: internal non-inferiority (2% margin) and pooled external superiority",
              "pairs": [["oeh_final", "v2", "internal_noninferiority"], ["oeh_final", "v2", "external"]]},
    "step3": {"what": "OEH v2 vs B2, XGBoost-CF/18, LCE, CB on internal and pooled external",
              "pairs": [[c, "v2", g] for c in ("b2_matched", "xgboost_cf_18", "lce_raw", "b2_plus_lce", "cb_only", "b2_plus_cb")
                        for g in ("internal", "external")]},
}
MARGIN = 0.02
DECISION = {"tau": [0.5, 0.8, 0.9], "lambda": [50, 100], "headline": {"tau": 0.8, "lambda": 100}}
COLOSSEUM_SRC = Path("D:/Tools/colosseum_work/src")


def panel_dir(panel: str) -> Path:
    return OUTPUT / panel


def dataset_dir(panel: str) -> Path:
    return DATASETS / panel


def protocol() -> dict:
    path = OUTPUT / "protocol.json"
    if E.sha256(path) != (OUTPUT / "protocol.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("protocol changed after freeze")
    return E.load_json(path)


# --------------------------------------------------------------------------- #
# Freeze
# --------------------------------------------------------------------------- #

def _internal_specs() -> tuple[list, dict]:
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    from environments.benchmark_environments import benchmark_specs, make_benchmark_mdp
    from scripts.analysis import run_oeh_confirmatory_20261004 as C
    from scripts.analysis import run_oeh_final_120 as R
    from scripts.analysis import run_oeh_120_unseen as base

    C.configure()
    seen = set(R.seen_kernels())
    seen |= set(E.load_json(E.COHORT7 / "actual_kernel_audit_preoutcome.json")["actual_kernel_fingerprints"].values())
    seen |= set(E.load_json(E.OUTPUT / "internal/dataset_seal.json")["actual_kernel_fingerprints"].values())
    v2b = ROOT / "artifacts/oeh_v2_behavior_cohort_2026-10-14/protocol.json"
    used = {int(i["seed"]) for i in E.load_json(E.COHORT7 / "protocol.json")["requested_specs"]}
    used |= {int(i["seed"]) for i in E.load_json(E.OUTPUT / "protocol.json")["panels"]["internal"]["requested_specs"]}
    if v2b.exists():
        doc = E.load_json(v2b)
        seen |= set(doc["kernel_fingerprints"].values())
        used |= {int(i["seed"]) for i in doc["requested_specs"]}
    for spec in R.OPENED.values():
        used |= {int(row["seed"]) for row in E.load_json(spec["dataset"] / "manifest.json")["instances"]}
    templates = [row for row in benchmark_specs() if row.split == "evaluation"]
    specs, kernels = [], {}
    for family, count in E.PRIMARY_COUNTS.items():
        selected = sorted((row for row in templates if row.family == family), key=lambda row: row.ordinal)
        for replica in range(2):
            for ordinal, template in enumerate(selected):
                instance_id = f"v2t_{family}_{replica * count + ordinal:03d}"
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
                    raise RuntimeError(f"no fresh kernel for {instance_id}")
                used.add(seed)
                seen.add(kernel)
                specs.append(spec)
                kernels[instance_id] = kernel
    return specs, kernels


def _forest_candidates() -> list[dict]:
    rng = np.random.default_rng(20291014)
    seen, out = set(), []
    while len(out) < 80:
        item = {"S": int(rng.choice([6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64])), "p": round(float(rng.uniform(0.01, 0.3)), 3),
                "r1": float(rng.choice([1, 2, 4, 8, 16])), "r2": float(rng.choice([0.5, 1, 2, 4]))}
        key = tuple(item.values())
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def freeze() -> None:
    if OUTPUT.exists() or DATASETS.exists():
        raise FileExistsError("namespace exists")
    spec = E.load_json(SPEC)
    if spec.get("status") != "frozen" or spec.get("base_head") != "oeh_ensemble":
        raise RuntimeError("OEH v2 is not frozen on the OEH current head")
    specs, kernels = _internal_specs()
    doc = {
        "schema": "oeh-v2-sealed-test-v1", "namespace": NAMESPACE, "frozen_at_utc": E.now(),
        "status": "frozen_before_any_panel_data_or_learner_outcomes",
        "oeh_v2": spec, "oeh_v2_spec_sha256": E.sha256(SPEC),
        "panels": {
            "internal": {**PANELS["internal"], "recipe": "two fresh kernels per primary template; distinct from every earlier kernel",
                         "requested_specs": [s.to_dict() for s in specs], "kernel_fingerprints": kernels},
            "colosseum2029": {**PANELS["colosseum2029"], "sampler_seed": 2029, "draws_per_family": 600, "export_cap_per_family": 30,
                              "select_per_family": 18, "exclusion": "every kernel of the 2026, 2027 and 2028 runs",
                              "data_seed_base": 9_950_000, "source": "Colosseum @ 981b344, mean rewards, continuous"},
            "spibb2": {**PANELS["spibb2"], "source": "Laroche et al. 2019 author code @ 833f32b", "ratios": list(S.RATIOS),
                       "per_ratio": 12, "numpy_seed_base": 2226101400, "data_seed_base": 9_970_000},
            "forest": {**PANELS["forest"], "source": "pymdptoolbox mdptoolbox.example.forest (Chades et al. 2014; Possingham & Tuck)",
                       "start_state": 0, "gamma": 0.95, "candidates": _forest_candidates(),
                       "selection": "first 48 candidates passing the project's dynamic-range filter", "data_seed_base": 9_990_000},
        },
        "budgets": {"primary": list(PRIMARY_BUDGETS), "extended": list(EXTENDED_BUDGETS),
                    "extended_endpoint": "first budget of the eight with >=8/10 seeds at it and every larger budget; secondary"},
        "data": E.load_json(E.OUTPUT / "protocol.json")["data"],
        "methods": {"v2": "OEH v2 (oeh_v2_spec.json)", "oeh_final": "OEH current", "legacy": "B2 + legacy diagnostics",
                    "b2_matched": "B2", "xgboost_cf_18": "adapted XGBoost-CF on B2", "lce_raw": "LCE", "b2_plus_lce": "B2 + LCE",
                    "cb_only": "CB", "b2_plus_cb": "B2 + CB", "v2_no_direct_oehb": "OEH v2 built on the arm without direct OEH-B (descriptive)",
                 "b2_shift": "B2 with the v2 shrinkage eps(d) toward the learner prior (descriptive)",
                 "legacy_shift": "B2 + legacy with the v2 shrinkage eps(d) (descriptive)",
                    "factorial": "eight P/H/L block arms of OEH current (descriptive Shapley)"},
        "gate": {"unit": "MDP; IBS over the six primary thresholds and seven-class log loss (floor 1e-12)",
                 "pooled_external": "all MDPs of colosseum2029, spibb2 and forest, equal MDP weight",
                 "noninferiority": f"internal: lower bound of (1+{MARGIN}) * OEH current - v2 > 0 for IBS and log loss",
                 "bootstrap": BOOTSTRAP, "steps": STEPS,
                 "sequence": "step 2 only if step 1 is GO, step 3 only if step 2 is GO; closed steps are reported descriptively",
                 "success": "OEH v2 is adopted if steps 1 and 2 are GO"},
        "descriptive": {"per_panel_learner": True, "murphy": True, "decision_regret": DECISION, "extended_endpoint": True,
                        "cost": True, "shapley": True},
        "disclosures": [
            "Pipeline dry run before this freeze: 3 development forest MDPs from a different candidate draw "
            "(numpy seed 1, not the registered draw 20291014), D:/Tools/colosseum_work/forest_feas; its 45 labels were "
            "seen (BC and IQL mostly beyond x50, CFQI mostly at x1). A seed-1 draw of 80 candidates all passed the "
            "dynamic-range filter. No registered test MDP, kernel or buffer existed at the freeze.",
            "OEH v2 was frozen (oeh_v2_spec.json) before this protocol; M2 stopped at step 0 and M3 was not adopted by "
            "rules_m2_m3.json, so the training panel is the sealed 720-MDP meta panel."],
        "source_sha256": {E.rel(p): E.sha256(p) for p in (Path(__file__), ROOT / "scripts/analysis/oeh_v2_forecast.py",
                                                         ROOT / "scripts/analysis/run_oeh_eighth_cohort_20261014.py",
                                                         ROOT / "scripts/analysis/run_spibb_panel_20261014.py",
                                                         ROOT / "benchmarks/colosseum/driver/colosseum_export_fast.py")},
    }
    OUTPUT.mkdir(parents=True)
    E.write_json(OUTPUT / "protocol.json", doc)
    (OUTPUT / "protocol.sha256").write_text(E.sha256(OUTPUT / "protocol.json") + "\n", encoding="ascii")
    print(f"frozen OEH v2 test protocol {E.sha256(OUTPUT / 'protocol.json')}", flush=True)


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #

def _generate(panel: str, specs, registry=None, policies=None) -> dict:
    from scripts.analysis.run_unseen_validation import archive_imports
    _, _, generator, _ = archive_imports()
    import environments.behavior_policies as B
    if registry is not None:
        generator.resolve_dynamic_range = lambda spec: (spec, registry[spec.instance_id], 0)
    if policies is not None:
        B.POLICY_FAMILIES = (*B.POLICY_FAMILIES, S.FAMILY)
        generator.behavior_policy_suite = lambda mdp, seed, variant_index: [B.BehaviorPolicy(
            family=S.FAMILY, probabilities=np.asarray(policies[mdp.environment_id], float),
            parameters={"source": "spibb author"}, seed=seed, uses_q_star=True)]
    generator.SIZE_MULTIPLIERS = EXTENDED_BUDGETS
    return generator.generate_benchmark_bundle(
        dataset_dir(panel), specs=specs, pilot_seeds=(0,), training_seeds=tuple(range(10)), max_size=None, horizon=100,
        run_mode="test", deviation_reason=f"OEH v2 sealed test: {panel}", pilot_size_multipliers=(2, 5, 10))


def _registry_from_kernels(kernels: Path, ids, family: str, seed_base: int, loader):
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    from environments.benchmark_environments import BenchmarkSpec, DYNAMIC_RANGE_THRESHOLD
    from environments.tabular_environments import TabularMDP
    from exact_solver import passes_dynamic_range_filter

    registry, specs, selection = {}, [], []
    for mdp_id in ids:
        mdp = loader(str(kernels), mdp_id, TabularMDP)
        keep = bool(passes_dynamic_range_filter(mdp, threshold=DYNAMIC_RANGE_THRESHOLD))
        selection.append({"mdp_id": mdp_id, "passes_filter": keep})
        if keep:
            spec = BenchmarkSpec(instance_id=mdp_id, split="evaluation", family=family, seed=seed_base + 1_000 * len(specs),
                                 ordinal=len(specs), family_ordinal=len(specs))
            registry[mdp_id] = mdp
            specs.append(spec)
    return registry, specs, selection


def generate(panel: str, workers: int) -> None:
    doc = protocol()
    target = panel_dir(panel)
    target.mkdir(parents=True, exist_ok=True)
    if (target / "dataset_seal.json").exists():
        return
    spec_doc = doc["panels"][panel]
    kernels = target / "kernels"
    extra = {}
    if panel == "internal":
        from scripts.analysis.run_unseen_validation import archive_imports
        archive_imports()
        from environments.benchmark_environments import BenchmarkSpec
        manifest = _generate(panel, [BenchmarkSpec(**i) for i in spec_doc["requested_specs"]])
    elif panel == "colosseum2029":
        import subprocess
        trial = Path("D:/Tools/colosseum_work/trial")
        out = trial / "data" / "v2test"
        if not (out / "candidates.json").exists():
            subprocess.run([str(Path("D:/Tools/py39col/python.exe")), str(ROOT / "benchmarks/colosseum/driver/colosseum_export_fast.py"),
                            "--src", str(COLOSSEUM_SRC), "--out", str(out), "--prefix", "v2t_colosseum", "--seed", "2029",
                            "--draws", "600", "--cap", "30", "--exclude", str(trial / "data/colosseum"), str(trial / "data/confirm"),
                            str(trial / "data/c8tier")], check=True)
        cand = [r for r in json.loads((out / "candidates.json").read_text())["index"] if r["status"] == "candidate"]
        kernels.mkdir(exist_ok=True)
        for r in cand:
            (kernels / f"{r['mdp_id']}.npz").write_bytes((out / f"{r['mdp_id']}.npz").read_bytes())
        (target / "colosseum_candidates.json").write_bytes((out / "candidates.json").read_bytes())
        from scripts.analysis.run_unseen_validation import archive_imports
        archive_imports()
        from environments.benchmark_environments import BenchmarkSpec, DYNAMIC_RANGE_THRESHOLD
        from environments.tabular_environments import TabularMDP
        from exact_solver import passes_dynamic_range_filter
        registry, specs, selection = {}, [], []
        counts = defaultdict(int)
        for r in cand:  # build_colosseum.py convention: first 18 passing per family in sampler order
            mdp = E.colosseum_mdp(str(kernels), r["mdp_id"], TabularMDP)
            keep = bool(passes_dynamic_range_filter(mdp, threshold=DYNAMIC_RANGE_THRESHOLD))
            room = counts[r["family"]] < spec_doc["select_per_family"]
            selection.append({"mdp_id": r["mdp_id"], "family": r["family"], "passes_filter": keep, "kept": keep and room})
            if not (keep and room):
                continue
            spec = BenchmarkSpec(instance_id=r["mdp_id"], split="evaluation", family=f"colosseum_{r['family']}",
                                 seed=spec_doc["data_seed_base"] + 1_000 * len(specs), ordinal=len(specs),
                                 family_ordinal=counts[r["family"]])
            counts[r["family"]] += 1
            registry[spec.instance_id] = mdp
            specs.append(spec)
        if len(specs) != spec_doc["mdps"]:
            raise RuntimeError(f"colosseum2029: {len(specs)} kernels selected, {dict(counts)}")
        extra = {"selection": selection, "family_counts": dict(counts)}
        manifest = _generate(panel, specs, registry=registry)
    elif panel == "spibb2":
        kernels.mkdir(exist_ok=True)
        tasks = [{"mdp_id": f"spibb2_r{int(r * 10)}_{k:03d}", "ratio": r, "seed": spec_doc["numpy_seed_base"] + i * 12 + k,
                  "out": str(kernels)} for i, r in enumerate(S.RATIOS) for k in range(12)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(S._garnet, tasks))
        registry, specs, selection = _registry_from_kernels(kernels, [t["mdp_id"] for t in tasks], "spibb_garnet",
                                                            spec_doc["data_seed_base"], S.spibb_mdp)
        policies = {m: np.load(kernels / f"{m}.npz")["pi_b"] for m in registry}
        extra = {"selection": selection, "records": records}
        manifest = _generate(panel, specs, registry=registry, policies=policies)
    else:  # forest
        import mdptoolbox.example as ex
        kernels.mkdir(exist_ok=True)
        ids = []
        for i, c in enumerate(spec_doc["candidates"]):
            Pm, Rm = ex.forest(S=c["S"], r1=c["r1"], r2=c["r2"], p=c["p"])
            rho = np.zeros(c["S"])
            rho[0] = 1.0
            mdp_id = f"forest_{i:03d}"
            Rm = np.asarray(Rm, float)  # amendment 1: rewards scaled into the public bound [0, 1]
            np.savez(kernels / f"{mdp_id}.npz", T=np.transpose(np.asarray(Pm, float), (1, 0, 2)), R=Rm / Rm.max(), rho=rho)
            ids.append(mdp_id)
        reg_all, specs_all, selection = _registry_from_kernels(kernels, ids, "forest", spec_doc["data_seed_base"], E.colosseum_mdp)
        specs = specs_all[:spec_doc["mdps"]]
        if len(specs) != spec_doc["mdps"]:
            raise RuntimeError(f"forest: only {len(specs)} candidates pass the filter")
        registry = {s.instance_id: reg_all[s.instance_id] for s in specs}
        extra = {"selection": selection}
        manifest = _generate(panel, specs, registry=registry)
    E.write_json(target / "dataset_seal.json", {
        "protocol_sha256": E.sha256(OUTPUT / "protocol.json"), "sealed_at_utc": E.now(), "learner_outcomes_observed": 0,
        "instances": manifest["instance_count"], "buffers": manifest["physical_buffer_count"],
        "kernels_sha256": {p.stem: E.sha256(p) for p in sorted(kernels.glob("*.npz"))} if kernels.exists() else {},
        "sha256": {E.rel(p): E.sha256(p) for p in (dataset_dir(panel) / "manifest.json", dataset_dir(panel) / "data_only_index.json")},
        **extra})


# --------------------------------------------------------------------------- #
# Pilot-only inputs and forecasts
# --------------------------------------------------------------------------- #

def _loader_kind(panel: str) -> tuple[str | None, str]:
    kernels = str(panel_dir(panel) / "kernels")
    return {"internal": (None, "internal"), "colosseum2029": (kernels, "colosseum"), "forest": (kernels, "colosseum"),
            "spibb2": (kernels, "spibb")}[panel]


def build_query(panel: str, workers: int) -> pd.DataFrame:
    from scripts.analysis import run_oeh_final_120 as R
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis.develop_cross_learner_context import with_context
    from scripts.analysis.develop_cohort8_candidates_20261014 import add_fields

    target, dataset = panel_dir(panel), dataset_dir(panel)
    seven = E.load_json(E.COHORT7 / "protocol.json")["oeh_final"]
    R._repo_first()
    tick = {}
    t0 = time.perf_counter()
    features = E.pilot_features(dataset, target / "pilot_features.csv", workers)
    tick["pilot_features"] = time.perf_counter() - t0
    source, kind = _loader_kind(panel)
    out = target / "published_preoutcome.csv"
    t0 = time.perf_counter()
    if kind == "spibb":
        if not out.exists():
            manifest = E.load_json(dataset / "manifest.json")
            inst = {i["instance_id"]: i for i in manifest["instances"]}
            rows = sorted((r for r in manifest["datasets"] if r.get("stream") == "pilot" and int(r.get("size_multiplier", -1)) == 10),
                          key=lambda r: r["instance_id"])
            with ProcessPoolExecutor(max_workers=workers) as pool:
                pd.DataFrame(list(pool.map(S._published_task, [{"instance": inst[r["instance_id"]], "row": r, "dataset": str(dataset),
                                                                  "kernels": source} for r in rows], chunksize=4))).to_csv(out, index=False)
        pub = pd.read_csv(out)
    else:
        pub = E.published(dataset, out, workers, source)
    tick["published"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    R.run_oeh_b(dataset, target / "oeh_b_groups", seven["structure"], seven["requirement_draws"], f"{NAMESPACE}|{panel}", workers, "cpu")
    tick["oeh_b"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    R.run_horizon(dataset, target / "horizon_preoutcome.csv", seven["structure"], workers, oracle=False)
    tick["horizon"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    curve = R.run_curve(dataset, target / "curve_preoutcome.csv", seven["structure"], seven["requirement_draws"], workers)
    tick["curve"] = time.perf_counter() - t0
    query = features.merge(F.oeh_feature_rows(target / "oeh_b_groups", target / "horizon_preoutcome.csv"), on=KEYS, validate="one_to_one")
    query = query.merge(R.curve_features(curve), on=KEYS, validate="one_to_one")
    query = query.merge(pub, on=["mdp_id", "policy_family"], validate="many_to_one")
    query = add_fields(with_context(query))
    query["panel"] = "primary"
    seal = E.load_json(target / "dataset_seal.json")
    if len(query) != seal["instances"] * PANELS[panel]["behaviors"] * 3 or query[KEYS].duplicated().any():
        raise RuntimeError(f"incomplete {panel} query: {len(query)}")
    E.write_json(target / "input_runtime_seconds.json", tick)
    return query.sort_values(KEYS).reset_index(drop=True)


def training_panel(spec: dict) -> pd.DataFrame:
    from scripts.analysis.develop_cohort8_candidates_20261014 import meta_panel
    train = meta_panel(sort=False)
    for extra in spec.get("extra_training_cohorts", []):
        rows = pd.read_csv(ROOT / extra["query"]).merge(pd.read_csv(ROOT / extra["labels"])[KEYS + ["event"]], on=KEYS,
                                                        validate="one_to_one")
        rows["cohort"], rows["panel"] = extra["name"], "primary"
        cols = [c for c in train.columns if c in rows.columns]
        train = pd.concat([train, rows[cols]], ignore_index=True)
    return train


def forecast(panel: str, query: pd.DataFrame, workers: int) -> pd.DataFrame:
    from scripts.analysis.develop_cohort8_candidates_20261014 import lce_law
    from scripts.analysis import oeh_v2_forecast as V

    doc = protocol()
    spec = doc["oeh_v2"]
    target = panel_dir(panel)
    train = training_panel(spec)
    sets = E.forecast_sets()
    jobs = {name: (sets[name], False) for name in ("oeh_final", "b2_matched", "b2_plus_eri", "prior_oeh_matched", "b2_plus_lce",
                                                   "b2_plus_cb", "cb_only", "b2_plus_P", "b2_plus_H", "oeh_without_legacy_diagnostics",
                                                   "oeh_without_oehh_risk", "oeh_without_direct_oehb", "xgboost_cf_18")}
    jobs["cond_oeh"] = (sets["oeh_final"], True)
    jobs["cond_without_direct_oehb"] = (sets["oeh_without_direct_oehb"], True)
    jobs["cond_legacy"] = (sets["prior_oeh_matched"], True)
    tpath, qpath = target / "_train.pkl", target / "_query.pkl"
    train.to_pickle(tpath)
    query.to_pickle(qpath)
    tasks = [{"method": ("cond_full" if finite else name), "label": name, "fields": list(fields), "train": str(tpath), "query": str(qpath),
              "finite": finite} for name, (fields, finite) in sorted(jobs.items(), key=lambda kv: -len(kv[1][0]))]
    tables, tuning = {}, {}
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        for task, (method, probability, weights, seconds) in zip(tasks, pool.map(E._forecast_task, tasks)):
            tables[task["label"]], tuning[task["label"]] = probability, weights
            print(f"forecast {panel}/{task['label']}: {seconds:.0f}s", flush=True)
    tpath.unlink()
    qpath.unlink()
    d = V.knn_distance(train, query)
    prior = V.class_prior(train)
    pi = np.stack([prior[a] for a in query.algorithm])
    params = {k: spec["parameters"][k] for k in ("lambda_max", "eps_max", "d0")}
    own_finite = spec["parameters"]["q_oeh"] == "finite_only_head"  # else the base head's finite part
    robust = tables["cond_legacy"] if spec["parameters"]["q_robust"] == "legacy_finite_only_head" else tables["prior_oeh_matched"]
    tables["v2"] = V.v2_law(tables["oeh_final"], tables["cond_oeh" if own_finite else "oeh_final"], robust, d, pi, **params)
    tables["v2_no_direct_oehb"] = V.v2_law(tables["oeh_without_direct_oehb"],
                                           tables["cond_without_direct_oehb" if own_finite else "oeh_without_direct_oehb"],
                                           robust, d, pi, **params)
    eps = V.shift_weight(params["eps_max"], d, params["d0"])[:, None]
    for name, base in (("b2_shift", "b2_matched"), ("legacy_shift", "prior_oeh_matched")):  # shrinkage alone (descriptive)
        tables[name] = (1 - eps) * tables[base] + eps * pi
    tables["lce_raw"] = lce_law(query)
    tables["legacy"] = tables.pop("prior_oeh_matched")
    for name in ("cond_oeh", "cond_without_direct_oehb", "cond_legacy"):
        tables.pop(name)
    frames = []
    for method, probability in tables.items():
        frame = query[[*KEYS, "panel"]].copy()
        frame["method"], frame["distance"] = method, d
        frame[P] = probability
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    p = predictions[P].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-9):
        raise RuntimeError("invalid seven-class forecast")
    E.write_json(target / "forecast_tuning_preoutcome.json", tuning)
    if not spec.get("extra_training_cohorts"):
        E.check_sealed_tuning({"oeh_final": tuning["oeh_final"], "b2_matched": tuning["b2_matched"], "b2_plus_eri": tuning["b2_plus_eri"],
                               "prior_oeh_matched": tuning["legacy"] if "legacy" in tuning else tuning["prior_oeh_matched"]}, target)
    return predictions


def prepare(panel: str, workers: int) -> None:
    target = panel_dir(panel)
    if (target / "preoutcome_seal.json").exists():
        raise FileExistsError(f"{panel} already sealed")
    protocol()
    generate(panel, workers)
    from scripts.analysis import run_oeh_final_120 as R
    R._repo_first()
    query = build_query(panel, workers)
    predictions = forecast(panel, query, workers)
    predictions.to_csv(target / "predictions_preoutcome.csv", index=False)
    query.to_csv(target / "query_preoutcome.csv.gz", index=False)
    names = ["pilot_features.csv", "published_preoutcome.csv", "horizon_preoutcome.csv", "curve_preoutcome.csv",
             "query_preoutcome.csv.gz", "predictions_preoutcome.csv", "forecast_tuning_preoutcome.json"]
    groups = sorted((target / "oeh_b_groups").glob("*.json"))
    E.write_json(target / "preoutcome_seal.json", {
        "panel": panel, "protocol_sha256": E.sha256(OUTPUT / "protocol.json"), "learner_outcomes_observed": 0, "sealed_at_utc": E.now(),
        "artifact_sha256": {n: E.sha256(target / n) for n in names}, "oeh_b_groups": len(groups),
        "oeh_b_groups_sha256": hashlib.sha256("".join(E.sha256(p) for p in groups).encode()).hexdigest()})
    print(f"sealed {panel}: {len(predictions)} predictions", flush=True)


# --------------------------------------------------------------------------- #
# Learners and labels
# --------------------------------------------------------------------------- #

def extended_event(counts) -> int:
    for index in range(len(counts)):
        if min(counts[index:]) >= 8:
            return index
    return len(counts)


def evaluate(panel: str, workers: int) -> None:
    from scripts.analysis.run_unseen_validation import archive_imports
    from scripts.analysis.compare_difficulty import suffix_event
    from scripts.analysis import run_oeh_final_120 as R

    target = panel_dir(panel)
    seal = E.load_json(target / "preoutcome_seal.json")
    for name, digest in seal["artifact_sha256"].items():
        if E.sha256(target / name) != digest:
            raise RuntimeError(f"sealed file changed: {name}")
    if (target / "outcomes.csv.gz").exists():
        raise FileExistsError("outcomes exist")
    _, _, _, seed_index = archive_imports()
    R._repo_first()
    dataset = dataset_dir(panel)
    manifest = E.load_json(dataset / "manifest.json")
    instances = {i["instance_id"]: i for i in manifest["instances"]}
    by_mdp = defaultdict(list)
    for row in manifest["datasets"]:
        if row.get("stream") == "training":
            by_mdp[row["instance_id"]].append({**row, "seed_index": seed_index(row)})
    expected = len(instances) * PANELS[panel]["behaviors"] * len(EXTENDED_BUDGETS) * 10
    if sum(map(len, by_mdp.values())) != expected:
        raise RuntimeError("incomplete training grid")
    source, kind = _loader_kind(panel)
    tasks = [{"instance": instances[m], "dataset": str(dataset), "colosseum": source if kind == "colosseum" else None,
              "kernels": source, "records": sorted(r, key=lambda x: (x["policy_family"], x["size_multiplier"], x["seed_index"]))}
             for m, r in sorted(by_mdp.items())]
    worker = S._mdp_jobs if kind == "spibb" else E._mdp_jobs
    tick = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outcomes = pd.DataFrame([row for result in pool.map(worker, tasks) for row in result])
    counts = outcomes.groupby(["mdp_id", "policy_family", "algorithm", "budget"]).success.agg(["sum", "size"])
    if (counts["size"] != 10).any():
        raise RuntimeError("ten seeds required")
    wide = counts["sum"].unstack("budget").reindex(columns=list(EXTENDED_BUDGETS))
    labels = wide.reset_index()
    labels["event"] = [suffix_event(tuple(int(x) for x in r[:6])) for r in wide.to_numpy()]
    labels["event_extended"] = [extended_event(tuple(int(x) for x in r)) for r in wide.to_numpy()]
    labels = labels.rename(columns={b: f"successes_x{b}" for b in EXTENDED_BUDGETS})
    outcomes.to_csv(target / "outcomes.csv.gz", index=False)
    labels.to_csv(target / "labels.csv", index=False)
    E.write_json(target / "outcome_run.json", {"jobs": len(outcomes), "seconds": time.perf_counter() - tick, "completed_at_utc": E.now(),
                                               "preoutcome_seal_sha256": E.sha256(target / "preoutcome_seal.json"),
                                               "labels_sha256": E.sha256(target / "labels.csv")})
    print(f"{panel}: {len(outcomes)} jobs, {len(labels)} labels", flush=True)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def _scored(panel: str) -> pd.DataFrame:
    target = panel_dir(panel)
    run = E.load_json(target / "outcome_run.json")
    if run["preoutcome_seal_sha256"] != E.sha256(target / "preoutcome_seal.json"):
        raise RuntimeError("seal changed after outcomes")
    pred = pd.read_csv(target / "predictions_preoutcome.csv")
    lab = pd.read_csv(target / "labels.csv")
    m = pred.merge(lab[KEYS + ["event", "event_extended"]], on=KEYS, validate="many_to_one")
    p, y = m[P].to_numpy(float), m.event.to_numpy(int)
    m["brier"] = ((np.cumsum(p, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)
    m["log_loss"] = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    m["source_panel"] = panel
    return m


def analyze() -> dict:
    from scripts.analysis.colosseum_posthoc_20261014 import decision_costs, fixed_cdf, max_t

    protocol()
    scored = pd.concat([_scored(p) for p in PANELS], ignore_index=True)
    scored.to_csv(OUTPUT / "scored.csv.gz", index=False)
    per_mdp = scored.groupby(["source_panel", "method", "mdp_id"])[["brier", "log_loss"]].mean().reset_index()

    def effect(base: str, cand: str, group: str):
        rows = per_mdp[per_mdp.source_panel.isin(EXTERNAL if group in ("external",) else ("internal",))]
        wide = rows.pivot_table(index=["source_panel", "mdp_id"], columns="method", values=["brier", "log_loss"])
        cols = []
        for metric in ("brier", "log_loss"):
            b, c = wide[(metric, base)], wide[(metric, cand)]
            cols.append(((1 + MARGIN) * b - c) if group == "internal_noninferiority" else (b - c))
        return np.column_stack(cols), wide

    steps, open_gate = {}, True
    for name, step in STEPS.items():
        sets, labels_ = {}, []  # MDP set -> effect columns; contrasts on one set share their resampled MDPs
        for base, cand, group in step["pairs"]:
            eff, _ = effect(base, cand, group)
            key = "external" if group == "external" else "internal"
            sets.setdefault(key, []).append(eff)
            labels_.append((key, [(base, cand, group, "brier"), (base, cand, group, "log_loss")]))
        rows = []
        if len(sets) == 1:
            mean, lower, crit = max_t(np.column_stack(next(iter(sets.values()))), BOOTSTRAP["seed"], BOOTSTRAP["draws"])
            ordered = [lab for _, labs in labels_ for lab in labs]
        else:  # independent MDP sets: one index matrix per set, common max over every column
            rng = np.random.default_rng(BOOTSTRAP["seed"])
            means, ses, draws, ordered = [], [], [], []
            for key in ("internal", "external"):
                block = np.column_stack(sets[key])
                n = len(block)
                mu, se = block.mean(0), block.std(0, ddof=1) / np.sqrt(n)
                idx = rng.integers(0, n, size=(BOOTSTRAP["draws"], n))
                draws.append((block[idx].mean(axis=1) - mu) / se)
                means.append(mu)
                ses.append(se)
                ordered += [lab for k, labs in labels_ if k == key for lab in labs]
            crit = float(np.quantile(np.max(np.concatenate(draws, axis=1), axis=1), .95))
            mean = np.concatenate(means)
            lower = mean - crit * np.concatenate(ses)
        labels_ = ordered
        for i, (base, cand, group, metric) in enumerate(labels_):
            rows.append({"baseline": base, "candidate": cand, "group": group, "metric": metric, "mean_improvement": float(mean[i]),
                         "simultaneous_lower_95": float(lower[i])})
        passed = all(r["simultaneous_lower_95"] > 0 for r in rows)
        steps[name] = {"tested": open_gate, "critical": crit, "contrasts": rows,
                       "verdict": ("GO" if passed else "NO-GO") if open_gate else f"not tested (gate closed); descriptive {'GO' if passed else 'NO-GO'}"}
        open_gate = open_gate and passed
    summaries = []
    for scope, by in (("panel", ["source_panel"]), ("learner", ["source_panel", "algorithm"])):
        t = scored.groupby([*by, "method", "mdp_id"])[["brier", "log_loss"]].mean().groupby([*by, "method"]).mean().reset_index()
        t.insert(0, "scope", scope)
        summaries.append(t)
    ext = per_mdp[per_mdp.source_panel.isin(EXTERNAL)].groupby("method")[["brier", "log_loss"]].mean().reset_index()
    ext.insert(0, "scope", "pooled_external")
    summaries.append(ext)
    pd.concat(summaries).to_csv(OUTPUT / "summary.csv", index=False)
    regret = []
    for panel in PANELS:
        s = scored[scored.source_panel == panel]
        base = s[s.method == "v2"].sort_values(KEYS).reset_index(drop=True)
        for tau in DECISION["tau"]:
            for lam in DECISION["lambda"]:
                for method in ("v2", "oeh_final", "legacy", "b2_matched", "always_skip", "always_x50"):
                    frame, cdf = (base, fixed_cdf(method, len(base))) if method.startswith("always") else (
                        s[s.method == method].sort_values(KEYS).reset_index(drop=True), None)
                    if cdf is None:
                        cdf = np.cumsum(frame[P].to_numpy(float), 1)[:, :6]
                    parts = pd.DataFrame(decision_costs(cdf, frame.event.to_numpy(int), tau, lam))
                    parts["mdp_id"] = frame.mdp_id.to_numpy()
                    regret.append({"panel": panel, "tau": tau, "lambda": lam, "method": method,
                                   "mean_regret": float(parts.groupby("mdp_id").regret.mean().mean())})
    pd.DataFrame(regret).to_csv(OUTPUT / "decision_regret.csv", index=False)
    ext_rows = scored[(scored.method == "v2") & (scored.event == 6)]
    extended = ext_rows.groupby(["source_panel", "event_extended"]).size().unstack(fill_value=0)
    extended.to_csv(OUTPUT / "extended_endpoint.csv")
    from scripts.analysis.colosseum_posthoc_20261014 import shapley
    from scripts.analysis.cross_panel_reanalysis_20261014 import murphy
    arm_names = {"oeh_final": "oeh_full_108", "legacy": "b2_plus_L"}
    shap = []
    for scope, panels in [(p, (p,)) for p in PANELS] + [("pooled_external", EXTERNAL)]:
        rows = per_mdp[per_mdp.source_panel.isin(panels)].replace({"method": arm_names})
        rows = rows.assign(mdp_id=rows.source_panel + "/" + rows.mdp_id)
        for metric in ("brier", "log_loss"):
            values = shapley(rows, metric)
            shap.append({"scope": scope, "metric": metric, **values.mean().to_dict()})
    pd.DataFrame(shap).to_csv(OUTPUT / "shapley.csv", index=False)
    keep = ("v2", "oeh_final", "legacy", "b2_matched", "xgboost_cf_18")
    pd.DataFrame([{"panel": panel, **row} for panel in PANELS
                  for row in murphy(scored[(scored.source_panel == panel) & scored.method.isin(keep)])]).to_csv(
        OUTPUT / "murphy.csv", index=False)
    result = {"completed_at_utc": E.now(), "steps": steps, "protocol_sha256": E.sha256(OUTPUT / "protocol.json"),
              "adopt_v2": steps["step1"]["verdict"] == "GO" and steps["step2"]["verdict"] == "GO"}
    E.write_json(OUTPUT / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("freeze", "prepare", "evaluate", "analyze"))
    parser.add_argument("--panel", choices=tuple(PANELS))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "prepare":
        prepare(args.panel, args.workers)
    elif args.phase == "evaluate":
        evaluate(args.panel, args.workers)
    else:
        result = analyze()
        for name, step in result["steps"].items():
            print(name, step["verdict"], round(step["critical"], 3))
            for r in step["contrasts"]:
                print(f"  {r['baseline']:>14s} -> {r['candidate']} [{r['group']}] {r['metric']}: {r['mean_improvement']:+.4f} LB {r['simultaneous_lower_95']:+.4f}")


if __name__ == "__main__":
    main()

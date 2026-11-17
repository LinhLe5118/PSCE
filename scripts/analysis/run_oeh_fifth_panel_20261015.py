"""Fifth external panel for OEH (final method): two benchmark sources not used anywhere before.

* ``frozenlake``: Gymnasium 1.0.0 FrozenLake, maps from its own ``generate_random_map``,
  ``is_slippery=True``; 36 MDPs, the first passing the dynamic-range filter among 160
  seeded candidates (size 4-8, frozen probability 0.75-0.95);
* ``gambler``: the gambler's problem of Sutton & Barto (2018, Example 4.3), capital
  0..N, stake min(k, s, N - s), win probability p_h; 36 MDPs among 60 seeded
  candidates (N 8-20, p_h 0.25-0.55).

Both are episodic as published; they are converted to the continuing setting used for
every external panel (Colosseum's convention): a terminal state restarts from the initial
distribution, keeping the reward earned on entry. Rewards lie in [0, 1]. The five
registered behaviors, x10 pilot, ten seeds, BC/CFQI/IQL and budgets x1..x200 are unchanged.

Registered contrast family (one): OEH vs B2 + diagnostics on the 72 pooled MDPs,
integrated Brier and log loss, one-sided max-t with the original standard error.
Everything else is descriptive and listed in the protocol.

Phases: ``freeze``; ``prepare --panel P``; ``evaluate --panel P``; ``analyze``.
OEH forecasts are produced by the single entry point ``scripts.analysis.oeh_pipeline``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts.analysis import run_oeh_eighth_cohort_20261014 as E
from scripts.analysis import oeh_pipeline as O

ROOT = E.ROOT
OUTPUT = ROOT / "artifacts/oeh_fifth_panel_2026-10-15"
DATASETS = ROOT / "datasets/oeh_fifth_panel_2026-10-15"
NAMESPACE = "oeh-fifth-panel-v1-20261015"
KEYS, P = O.KEYS, O.P
EXTENDED_BUDGETS = (1, 2, 5, 10, 20, 50, 100, 200)
PANELS = {"frozenlake": {"mdps": 36, "candidates": 160}, "gambler": {"mdps": 36, "candidates": 60}}
BOOTSTRAP = {"draws": 10_000, "seed": 20261115, "rule": "max-t with the original standard error"}
DATA_SEED_BASE = {"frozenlake": 9_993_000, "gambler": 9_996_000}
DESCRIPTIVE_METHODS = {
    "oeh": "OEH (final method, oeh_v2_spec.json)", "oeh_full_head": "full 108-field head alone (no shift gate)",
    "robust": "B2 + diagnostics (registered comparator)", "b2_matched": "B2", "xgboost_cf_18": "adapted XGBoost-CF on B2",
    "b2_plus_lce": "B2 + LCE", "b2_plus_cb": "B2 + CB", "lce_raw": "LCE", "knn_analog": "k-NN analog on B2 (k = 25)",
    "logistic": "multinomial logistic regression on B2", "oeh_without_horizon": "OEH built without the OEH-H block (G4, G5)",
}


def panel_dir(panel: str) -> Path:
    return OUTPUT / panel


def protocol() -> dict:
    if E.sha256(OUTPUT / "protocol.json") != (OUTPUT / "protocol.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("protocol changed after freeze")
    return E.load_json(OUTPUT / "protocol.json")


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #

def continuing(T: np.ndarray, R: np.ndarray, rho: np.ndarray, terminal) -> tuple:
    """Episodic -> continuing: a terminal state restarts from the initial distribution."""
    for s in terminal:
        T[s] = rho[None, :]
        R[s] = 0.0
    return T, R, rho


def frozenlake_kernel(size: int, p: float, seed: int) -> tuple:
    from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map
    env = FrozenLakeEnv(desc=generate_random_map(size=size, p=p, seed=seed), is_slippery=True)
    S, A = env.observation_space.n, env.action_space.n
    T, R = np.zeros((S, A, S)), np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            for prob, nxt, reward, _ in env.unwrapped.P[s][a]:
                T[s, a, nxt] += prob
                R[s, a] += prob * reward
    rho = np.zeros(S)
    rho[0] = 1.0
    desc = env.unwrapped.desc.flatten()
    return continuing(T, R, rho, [s for s in range(S) if desc[s] in (b"H", b"G")])


def gambler_kernel(N: int, ph: float) -> tuple:
    K, S = N // 2, N + 1
    T, R = np.zeros((S, K, S)), np.zeros((S, K))
    for s in range(S):
        for k in range(K):
            if s in (0, N):
                T[s, k, s] = 1.0
                continue
            stake = min(k + 1, s, N - s)
            T[s, k, s + stake] += ph
            T[s, k, s - stake] += 1 - ph
            if s + stake == N:
                R[s, k] += ph
    rho = np.zeros(S)
    rho[1:N] = 1.0 / (N - 1)
    return continuing(T, R, rho, [0, N])


def candidates(panel: str) -> list[dict]:
    rng = np.random.default_rng({"frozenlake": 20291115, "gambler": 20291116}[panel])
    out, seen = [], set()
    while len(out) < PANELS[panel]["candidates"]:
        if panel == "frozenlake":
            item = {"size": int(rng.choice([4, 5, 6, 7, 8])), "p": round(float(rng.uniform(0.75, 0.95)), 3),
                    "map_seed": int(rng.integers(1, 2**31 - 1))}
        else:
            item = {"N": int(rng.integers(8, 21)), "p_h": round(float(rng.uniform(0.25, 0.55)), 3)}
        key = tuple(item.values())
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def kernel(panel: str, c: dict) -> tuple:
    return frozenlake_kernel(c["size"], c["p"], c["map_seed"]) if panel == "frozenlake" else gambler_kernel(c["N"], c["p_h"])


# --------------------------------------------------------------------------- #
# Freeze
# --------------------------------------------------------------------------- #

def freeze() -> None:
    import gymnasium
    if OUTPUT.exists() or DATASETS.exists():
        raise FileExistsError("namespace exists")
    spec = O.load_spec()
    doc = {
        "schema": "oeh-fifth-panel-v1", "namespace": NAMESPACE, "frozen_at_utc": E.now(),
        "status": "frozen_before_any_panel_data_or_learner_outcomes",
        "method": "OEH final method = oeh_v2_spec.json via scripts/analysis/oeh_pipeline.py (unchanged since the four-panel test)",
        "oeh_spec": spec, "oeh_spec_sha256": E.sha256(O.SPEC),
        "panels": {
            "frozenlake": {**PANELS["frozenlake"], "source": f"Gymnasium {gymnasium.__version__} FrozenLakeEnv + generate_random_map, is_slippery=True",
                           "candidates_list": candidates("frozenlake"), "data_seed_base": DATA_SEED_BASE["frozenlake"]},
            "gambler": {**PANELS["gambler"], "source": "Sutton & Barto (2018), Example 4.3 (gambler's problem); stake min(k, s, N - s), k = 1..N//2",
                        "candidates_list": candidates("gambler"), "data_seed_base": DATA_SEED_BASE["gambler"]},
        },
        "conversion": "episodic to continuing: terminal states restart from the initial distribution (FrozenLake: start cell; "
                      "gambler: uniform over capitals 1..N-1); reward kept on entry; rewards in [0, 1]",
        "selection": "first `mdps` candidates in list order passing the project's dynamic-range filter",
        "data": {"behaviors": 5, "pilot_multipliers": [2, 5, 10], "budgets": list(EXTENDED_BUDGETS), "training_seeds": 10,
                 "learners": ["behavior_cloning_tabular", "cfqi_tabular", "iql_tabular"],
                 "primary_label": "first six budgets (sustained 8/10 seeds at NR >= 0.95), seven classes"},
        "gate": {"family": [["robust", "oeh", "pooled", "brier"], ["robust", "oeh", "pooled", "log_loss"]],
                 "unit": "MDP (72 pooled, equal weight)", "bootstrap": BOOTSTRAP, "go": "both simultaneous lower bounds > 0"},
        "descriptive": {"methods": DESCRIPTIVE_METHODS, "per_family": True, "per_learner": True, "gain_by_distance": True,
                        "murphy": True, "decision_regret": {"tau": 0.8, "lambda": 100}},
        "disclosures": [
            "Feasibility only: dynamic-range pass rates were checked on development draws with numpy seed 1 (FrozenLake 12/40, "
            "gambler 40/40 after the continuing conversion; 0/40 for the episodic gambler, which motivated the conversion); "
            "no learner was run on any FrozenLake or gambler MDP before this freeze.",
            "A generation and pilot-input dry run on development draws (seed 1) checked the pipeline without learners.",
            "The OEH specification and the entry point were fixed before this protocol; the entry point reproduces the sealed "
            "four-panel forecasts exactly (artifacts/week8/oeh_pipeline_reproduction_2026-10-15)."],
        "source_sha256": {E.rel(p): E.sha256(p) for p in (Path(__file__), ROOT / "scripts/analysis/oeh_pipeline.py",
                                                         ROOT / "scripts/analysis/oeh_v2_forecast.py",
                                                         ROOT / "scripts/analysis/oeh_generic_baselines.py",
                                                         ROOT / "scripts/analysis/run_oeh_eighth_cohort_20261014.py")},
    }
    OUTPUT.mkdir(parents=True)
    E.write_json(OUTPUT / "protocol.json", doc)
    (OUTPUT / "protocol.sha256").write_text(E.sha256(OUTPUT / "protocol.json") + "\n", encoding="ascii")
    print(f"frozen fifth-panel protocol {E.sha256(OUTPUT / 'protocol.json')}", flush=True)


# --------------------------------------------------------------------------- #
# Data and forecasts
# --------------------------------------------------------------------------- #

def generate(panel: str, doc: dict, dataset: Path, target: Path) -> None:
    from scripts.analysis.run_unseen_validation import archive_imports
    _, _, generator, _ = archive_imports()
    from environments.benchmark_environments import BenchmarkSpec, DYNAMIC_RANGE_THRESHOLD
    from environments.tabular_environments import TabularMDP
    from exact_solver import passes_dynamic_range_filter

    spec_doc = doc["panels"][panel]
    kernels = target / "kernels"
    kernels.mkdir(parents=True, exist_ok=True)
    registry, specs, selection = {}, [], []
    for i, c in enumerate(spec_doc["candidates_list"]):
        if len(specs) == spec_doc["mdps"]:
            break
        mdp_id = f"{panel}_{i:03d}"
        T, R, rho = kernel(panel, c)
        np.savez(kernels / f"{mdp_id}.npz", T=T, R=R, rho=rho)
        mdp = E.colosseum_mdp(str(kernels), mdp_id, TabularMDP)
        keep = bool(passes_dynamic_range_filter(mdp, threshold=DYNAMIC_RANGE_THRESHOLD))
        selection.append({"mdp_id": mdp_id, **c, "passes_filter": keep})
        if not keep:
            (kernels / f"{mdp_id}.npz").unlink()
            continue
        spec = BenchmarkSpec(instance_id=mdp_id, split="evaluation", family=panel, seed=spec_doc["data_seed_base"] + 1_000 * len(specs),
                             ordinal=len(specs), family_ordinal=len(specs))
        registry[mdp_id] = mdp
        specs.append(spec)
    if len(specs) != spec_doc["mdps"]:
        raise RuntimeError(f"{panel}: only {len(specs)} candidates pass the filter")
    generator.resolve_dynamic_range = lambda s: (s, registry[s.instance_id], 0)
    generator.SIZE_MULTIPLIERS = EXTENDED_BUDGETS
    manifest = generator.generate_benchmark_bundle(
        dataset, specs=specs, pilot_seeds=(0,), training_seeds=tuple(range(10)), max_size=None, horizon=100,
        run_mode="test", deviation_reason=f"OEH fifth panel: {panel}", pilot_size_multipliers=(2, 5, 10))
    E.write_json(target / "dataset_seal.json", {
        "protocol_sha256": E.sha256(OUTPUT / "protocol.json"), "sealed_at_utc": E.now(), "learner_outcomes_observed": 0,
        "instances": manifest["instance_count"], "buffers": manifest["physical_buffer_count"], "selection": selection,
        "kernels_sha256": {p.stem: E.sha256(p) for p in sorted(kernels.glob("*.npz"))},
        "sha256": {E.rel(p): E.sha256(p) for p in (dataset / "manifest.json", dataset / "data_only_index.json")}})


def forecasts(query: pd.DataFrame, target: Path, workers: int) -> pd.DataFrame:
    from scripts.analysis import oeh_generic_baselines as G
    from scripts.analysis.develop_cohort8_candidates_20261014 import lce_law

    spec = O.load_spec()
    train = O.training_panel(spec)
    sets = E.forecast_sets()
    extra = {name: (sets[name], False) for name in ("b2_matched", "b2_plus_eri", "xgboost_cf_18", "b2_plus_lce", "b2_plus_cb",
                                                     "oeh_without_oehh_risk")}
    extra["oeh_without_oehh_risk_finite"] = (sets["oeh_without_oehh_risk"], True)
    tables = O.fit_heads(train, query, workers, target, extra)
    tuning = tables.pop("_tuning")
    law, d = O.final_law(tables, train, query, spec)
    no_h, _ = O.final_law(tables, train, query, spec, full="oeh_without_oehh_risk", full_finite="oeh_without_oehh_risk_finite")
    laws = {"oeh": law, "oeh_full_head": tables["full"], "robust": tables["robust"], "b2_matched": tables["b2_matched"],
            "xgboost_cf_18": tables["xgboost_cf_18"], "b2_plus_lce": tables["b2_plus_lce"], "b2_plus_cb": tables["b2_plus_cb"],
            "lce_raw": lce_law(query), "knn_analog": G.knn_analog(train, query), "logistic": G.logistic(train, query),
            "oeh_without_horizon": no_h}
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
    if not (target / "dataset_seal.json").exists():
        generate(panel, doc, dataset, target)
    query = O.pilot_inputs(dataset, target, workers, target / "kernels", "colosseum", f"{NAMESPACE}|{panel}")
    seal = E.load_json(target / "dataset_seal.json")
    if len(query) != seal["instances"] * 5 * 3:
        raise RuntimeError(f"incomplete {panel} query: {len(query)}")
    query.to_csv(target / "query_preoutcome.csv.gz", index=False)
    pred = forecasts(query, target, workers)
    pred.to_csv(target / "predictions_preoutcome.csv", index=False)
    names = ["pilot_features.csv", "published.csv", "horizon.csv", "curve.csv", "query_preoutcome.csv.gz",
             "predictions_preoutcome.csv", "forecast_tuning_preoutcome.json"]
    groups = sorted((target / "oeh_b_groups").glob("*.json"))
    E.write_json(target / "preoutcome_seal.json", {
        "panel": panel, "protocol_sha256": E.sha256(OUTPUT / "protocol.json"), "learner_outcomes_observed": 0, "sealed_at_utc": E.now(),
        "artifact_sha256": {n: E.sha256(target / n) for n in names}, "oeh_b_groups": len(groups),
        "oeh_b_groups_sha256": hashlib.sha256("".join(E.sha256(p) for p in groups).encode()).hexdigest()})
    print(f"sealed {panel}: {len(pred)} predictions", flush=True)


# --------------------------------------------------------------------------- #
# Learners and analysis
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

    target, dataset = panel_dir(panel), DATASETS / panel
    seal = E.load_json(target / "preoutcome_seal.json")
    for name, digest in seal["artifact_sha256"].items():
        if E.sha256(target / name) != digest:
            raise RuntimeError(f"sealed file changed: {name}")
    if (target / "outcomes.csv.gz").exists():
        raise FileExistsError("outcomes exist")
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
    source = str(target / "kernels")
    tasks = [{"instance": instances[m], "dataset": str(dataset), "colosseum": source, "kernels": source,
              "records": sorted(r, key=lambda x: (x["policy_family"], x["size_multiplier"], x["seed_index"]))}
             for m, r in sorted(by_mdp.items())]
    tick = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outcomes = pd.DataFrame([row for result in pool.map(E._mdp_jobs, tasks) for row in result])
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


def scored() -> pd.DataFrame:
    frames = []
    for panel in PANELS:
        target = panel_dir(panel)
        run = E.load_json(target / "outcome_run.json")
        if run["preoutcome_seal_sha256"] != E.sha256(target / "preoutcome_seal.json"):
            raise RuntimeError("seal changed after outcomes")
        m = pd.read_csv(target / "predictions_preoutcome.csv").merge(
            pd.read_csv(target / "labels.csv")[KEYS + ["event", "event_extended"]], on=KEYS, validate="many_to_one")
        p, y = m[P].to_numpy(float), m.event.to_numpy(int)
        m["brier"] = ((np.cumsum(p, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)
        m["log_loss"] = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
        m["source_panel"] = panel
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def analyze() -> dict:
    from scripts.analysis.colosseum_posthoc_20261014 import decision_costs, max_t
    from scripts.analysis.cross_panel_reanalysis_20261014 import murphy

    protocol()
    s = scored()
    s.to_csv(OUTPUT / "scored.csv.gz", index=False)
    per = s.groupby(["source_panel", "method", "mdp_id"])[["brier", "log_loss"]].mean().reset_index()
    wide = per.pivot_table(index=["source_panel", "mdp_id"], columns="method", values=["brier", "log_loss"])
    effect = np.column_stack([wide[(m, "robust")] - wide[(m, "oeh")] for m in ("brier", "log_loss")])
    mean, lower, crit = max_t(effect, BOOTSTRAP["seed"], BOOTSTRAP["draws"])
    gate = {"contrasts": [{"baseline": "robust", "candidate": "oeh", "metric": m, "mean_improvement": float(mean[i]),
                           "simultaneous_lower_95": float(lower[i])} for i, m in enumerate(("brier", "log_loss"))],
            "critical": float(crit), "mdps": int(len(effect))}
    gate["verdict"] = "GO" if all(c["simultaneous_lower_95"] > 0 for c in gate["contrasts"]) else "NO-GO"
    rows = []
    for scope, panels in [(p, (p,)) for p in PANELS] + [("pooled", tuple(PANELS))]:
        sub = per[per.source_panel.isin(panels)]
        w = sub.pivot_table(index=["source_panel", "mdp_id"], columns="method", values=["brier", "log_loss"])
        for method in DESCRIPTIVE_METHODS:
            row = {"scope": scope, "method": method, "brier": float(w[("brier", method)].mean()), "log_loss": float(w[("log_loss", method)].mean())}
            for metric in ("brier", "log_loss"):
                g = (w[(metric, method)] - w[(metric, "oeh")]).to_numpy()
                half = 1.96 * g.std(ddof=1) / np.sqrt(len(g))
                row[f"{metric}_gain_of_oeh"], row[f"{metric}_low95"], row[f"{metric}_high95"] = g.mean(), g.mean() - half, g.mean() + half
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT / "summary.csv", index=False)
    learner = s.groupby(["source_panel", "algorithm", "method", "mdp_id"])[["brier", "log_loss"]].mean().groupby(
        ["source_panel", "algorithm", "method"]).mean().reset_index()
    learner.to_csv(OUTPUT / "by_learner.csv", index=False)
    pd.DataFrame([{"panel": panel, **r} for panel in PANELS
                  for r in murphy(s[(s.source_panel == panel) & s.method.isin(["oeh", "robust", "oeh_full_head", "b2_matched"])])]).to_csv(
        OUTPUT / "murphy.csv", index=False)
    regret = []
    for panel in PANELS:
        sub = s[s.source_panel == panel]
        for method in ("oeh", "oeh_full_head", "robust", "b2_matched"):
            f = sub[sub.method == method].sort_values(KEYS)
            parts = pd.DataFrame(decision_costs(np.cumsum(f[P].to_numpy(float), 1)[:, :6], f.event.to_numpy(int), 0.8, 100))
            parts["mdp_id"] = f.mdp_id.to_numpy()
            regret.append({"panel": panel, "method": method, "mean_regret": float(parts.groupby("mdp_id").regret.mean().mean())})
    pd.DataFrame(regret).to_csv(OUTPUT / "decision_regret.csv", index=False)
    result = {"completed_at_utc": E.now(), "protocol_sha256": E.sha256(OUTPUT / "protocol.json"), "gate": gate,
              "events": s[s.method == "oeh"].groupby(["source_panel", "algorithm"]).event.value_counts().unstack(fill_value=0).to_dict("index")}
    E.write_json(OUTPUT / "result.json", {k: v for k, v in result.items() if k != "events"})
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
        g = result["gate"]
        print("gate", g["verdict"], round(g["critical"], 3), g["mdps"], "MDPs")
        for c in g["contrasts"]:
            print(f"  OEH vs B2 + diagnostics {c['metric']}: {c['mean_improvement']:+.4f} LB {c['simultaneous_lower_95']:+.4f}")


if __name__ == "__main__":
    main()

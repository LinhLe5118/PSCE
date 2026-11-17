"""M1: matched block ablation of the sealed 108-column OEH candidate (exploratory).

Phases, in order, each refusing to overwrite its own outputs:

``lineage``   write the per-column lineage and the five arm column lists;
``cost``      re-run every feature producer on a fixed sample of cohort-7 pilots,
              time it and check that it reproduces the sealed values;
``forecast``  fit every arm with ``oeh_final_forecast.matched_forecast`` on the
              same 720 opened meta-training MDPs, folds and tuning rule, then
              seal the seven-class predictions;
``score``     only now read the cohort-7 labels and score the sealed predictions.

The seventh cohort was opened on 2026-10-04, so every result is exploratory.
While ``lineage``, ``cost`` and ``forecast`` run, an audit hook refuses to open
any cohort-7 outcome file. Sealed artifacts are read, never written.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from scripts.analysis import m1_feature_lineage as L

ROOT = L.ROOT
COHORT = L.COHORT
DATASET = ROOT / "datasets/oeh_confirmatory_unseen_120_2026-10-04"
OUTPUT = ROOT / "artifacts/week8/m1_matched_ablation_2026-10-13"
PREVIOUS = ROOT / "artifacts/week8/matched_ablation_exploratory_2026-10-07"
NAMESPACE = "week8-m1-feature-lineage-matched-ablation-v1-20261013"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]
QUERY_FILES = ("pilot_features.csv", "horizon_preoutcome.csv", "predictions_preoutcome.csv")
# Sealed predictions the full and B2 arms must reproduce (same fit procedure).
REPRODUCE = {"oeh_full_108": "oeh_final", "b2_matched": "b2_matched"}
REPRODUCE_TOLERANCE = 1e-6
BOOTSTRAP = 2000
SEED = 20261013
PHASE_FILES = {
    "lineage": ("feature_lineage.csv", "feature_lineage.json"),
    "cost": ("cost_probe_groups.csv", "cost_summary.csv", "arm_feature_cost.csv", "cost_parity.json"),
    "forecast": ("predictions_before_label_read.csv", "tuning.json", "fit_runtime.csv",
                 "input_inventory.csv", "source_inventory.csv", "prediction_seal.json"),
    "score": ("scored.csv", "summary.csv", "paired_mdp_delta.csv", "calibration_by_budget.csv",
              "manifest.json"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Outcome guard and input recording
# --------------------------------------------------------------------------- #

def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(path))


FORBIDDEN_FILES = frozenset(_norm(COHORT / name) for name in L.OUTCOME_FILES)
FORBIDDEN_DIRS = tuple(_norm(COHORT / name) + os.sep for name in L.OUTCOME_DIRS)
_GUARD = {"active": False, "opened": set()}


class OutcomeAccessError(PermissionError):
    pass


def _hook(event, args):
    if event != "open" or not _GUARD["active"] or not args or isinstance(args[0], int) or args[0] is None:
        return
    try:
        path = _norm(os.fsdecode(args[0]))
    except (TypeError, ValueError):
        return
    if path in FORBIDDEN_FILES or path.startswith(FORBIDDEN_DIRS):
        raise OutcomeAccessError(f"Outcome file opened before the prediction seal: {path}")
    _GUARD["opened"].add(path)


_INSTALLED = False


@contextlib.contextmanager
def outcome_guard():
    """Refuse cohort-7 outcome files and record every other opened file."""
    global _INSTALLED
    if not _INSTALLED:
        sys.addaudithook(_hook)
        _INSTALLED = True
    _GUARD["active"], _GUARD["opened"] = True, set()
    try:
        yield _GUARD["opened"]
    finally:
        _GUARD["active"] = False


def input_inventory(opened) -> pd.DataFrame:
    root = _norm(ROOT) + os.sep
    data = (_norm(ROOT / "artifacts") + os.sep, _norm(ROOT / "datasets") + os.sep)
    rows = [{"path": rel(path), "bytes": os.path.getsize(path), "sha256": sha256(Path(path))}
            for path in sorted(opened) if path.startswith(root) and path.startswith(data) and os.path.isfile(path)]
    return pd.DataFrame(rows, columns=["path", "bytes", "sha256"])


def source_inventory() -> pd.DataFrame:
    root = _norm(ROOT) + os.sep
    files = {_norm(module.__file__) for module in list(sys.modules.values())
             if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")}
    rows = [{"path": rel(path), "sha256": sha256(Path(path))} for path in sorted(files) if path.startswith(root)]
    return pd.DataFrame(rows, columns=["path", "sha256"])


# --------------------------------------------------------------------------- #
# Output namespace
# --------------------------------------------------------------------------- #

def prepare_output(output: Path, phase: str) -> Path:
    output = output.resolve()
    for sealed in (COHORT, PREVIOUS):
        if output == sealed.resolve() or sealed.resolve() in output.parents:
            raise RuntimeError(f"Output must use a new namespace, not {rel(sealed)}")
    existing = [name for name in PHASE_FILES[phase] if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite {phase} outputs: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def verify_cohort_seal() -> dict:
    protocol = COHORT / "protocol.json"
    if sha256(protocol) != (COHORT / "protocol.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("Sealed protocol changed")
    seal = json.loads((COHORT / "preoutcome_seal.json").read_text(encoding="utf-8"))
    if seal["protocol_sha256"] != sha256(protocol) or seal["learner_outcomes_observed"] != 0:
        raise RuntimeError("Pre-outcome seal does not match the protocol")
    for name in QUERY_FILES:
        if sha256(COHORT / name) != seal["artifact_sha256"][name]:
            raise RuntimeError(f"Pre-outcome file changed: {name}")
    return {"protocol.json": sha256(protocol), "preoutcome_seal.json": sha256(COHORT / "preoutcome_seal.json"),
            **{name: seal["artifact_sha256"][name] for name in QUERY_FILES}}


def frozen_lineage(output: Path) -> tuple[list[dict], dict[str, tuple[str, ...]]]:
    """Lineage written by the ``lineage`` phase, verified against the code."""
    document = json.loads((output / "feature_lineage.json").read_text(encoding="utf-8"))
    rows = L.lineage()
    variants = L.arms(rows)
    if document["lineage_sha256"] != L.digest(rows) or document["arms_sha256"] != L.digest(variants):
        raise RuntimeError("Lineage changed after it was written; start a new namespace")
    if sha256(output / "feature_lineage.csv") != document["lineage_csv_sha256"]:
        raise RuntimeError("feature_lineage.csv changed")
    return rows, variants


# --------------------------------------------------------------------------- #
# Phase: lineage
# --------------------------------------------------------------------------- #

def run_lineage(output: Path) -> None:
    output = prepare_output(output, "lineage")
    rows = L.lineage()
    variants = L.arms(rows)
    table = pd.DataFrame(rows)
    for name, fields in variants.items():
        table[f"in_{name}"] = table.column.isin(fields)
    table.to_csv(output / "feature_lineage.csv", index=False)
    write_json(output / "feature_lineage.json", {
        "namespace": NAMESPACE, "written_at_utc": now(), "status": "lineage_frozen_before_forecast",
        "sealed_protocol": rel(L.PROTOCOL), "sealed_protocol_sha256": sha256(L.PROTOCOL),
        "blocks": L.BLOCKS, "components": L.COMPONENTS, "component_requires": L.REQUIRES,
        "arms": {name: {"dropped_blocks": list(L.ARMS[name]), "column_count": len(fields),
                        "columns": list(fields), "components": list(L.components_of(fields))}
                 for name, fields in variants.items()},
        "lineage_sha256": L.digest(rows), "arms_sha256": L.digest(variants),
        "lineage_csv_sha256": sha256(output / "feature_lineage.csv"),
        "source_sha256": {rel(path): sha256(path) for path in (Path(L.__file__), Path(__file__))},
        "note": "Arm membership follows information blocks fixed before any M1 outcome; "
                "'without direct OEH-B' keeps OEH-H, which still uses the posterior mean.",
    })
    print(table.groupby(["block", "component", "learner_scope"]).size().to_string(), flush=True)


# --------------------------------------------------------------------------- #
# Phase: cost (timing and parity on a fixed pilot sample)
# --------------------------------------------------------------------------- #

def pilot_groups(manifest: dict) -> dict[tuple[str, str], dict[int, dict]]:
    groups: dict[tuple[str, str], dict[int, dict]] = {}
    for row in manifest["datasets"]:
        size = int(row.get("size_multiplier", -1))
        if row.get("stream") == "pilot" and size in (2, 5, 10) and row.get("benchmark_split") == "evaluation":
            groups.setdefault((row["instance_id"], row["policy_family"]), {})[size] = row
    if len(groups) != 600 or any(set(value) != {2, 5, 10} for value in groups.values()):
        raise RuntimeError("Incomplete cohort-7 pilot grid")
    return groups


def cost_sample(manifest: dict, per_family: int) -> list[tuple[str, str]]:
    """``per_family`` instances per family (by hashed ID), all five behaviors."""
    families: dict[str, list[str]] = {}
    for item in manifest["instances"]:
        families.setdefault(item["family"], []).append(item["instance_id"])
    chosen = [mdp for family in sorted(families)
              for mdp in sorted(families[family], key=lambda m: hashlib.sha256(m.encode()).hexdigest())[:per_family]]
    groups = pilot_groups(manifest)
    return sorted(key for key in groups if key[0] in chosen)


def _load(row: dict) -> dict[str, np.ndarray]:
    path = DATASET / row["file"]
    if sha256(path) != row["sha256"]:
        raise RuntimeError(f"Pilot buffer changed: {row['file']}")
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name][:int(row["size"])] for name in loaded.files}


def _max_diff(expected, actual) -> float:
    a = np.asarray(expected, dtype=float)
    b = np.asarray(actual, dtype=float)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return float(np.max(np.where(same, 0.0, np.abs(a - b)), initial=0.0))


def cost_group(key: tuple[str, str], prefixes: dict[int, dict], structure: dict,
               sealed: dict) -> list[dict]:
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from stochastic_features import aggregate_oeh_risk_v3_profile, extract_stochastic_features
    from oeh_risk_v3 import estimate_oeh_risk_v3
    from lcoeh_v4 import LC_OEH_V4_FEATURES, estimate_lcoeh_v4
    from oeh import posterior as post, predictive
    from scripts.analysis import run_oeh_final_120 as R
    from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS

    record = prefixes[10]
    S, A = int(record["num_states"]), int(record["num_actions"])
    rho = record["initial_state_distribution"]
    arrays = {size: _load(row) for size, row in prefixes.items()}
    pilot = sealed["pilot"].loc[key]
    results = []

    def timed(component, call):
        tick = time.perf_counter()
        value = call()
        results.append({"mdp_id": key[0], "policy_family": key[1], "num_states": S, "num_actions": A,
                        "pilot_transitions": int(record["size"]), "component": component,
                        "seconds": time.perf_counter() - tick})
        return value

    b2 = timed("b2_statistics", lambda: extract_stochastic_features(
        arrays[10], num_states=S, num_actions=A, gamma=.95, bootstrap_replicates=200,
        seed_key=str(record["dataset_id"])))
    results[-1]["max_abs_diff_vs_sealed"] = max(
        _max_diff(pilot[name], b2[name]) for name in FEATURES_B2)

    def oeh3():
        risk = estimate_oeh_risk_v3(arrays[10], rho, num_states=S, num_actions=A, gamma=.95, k_max=100)
        depth = {int(item["depth"]): item for item in risk["records"]}
        return aggregate_oeh_risk_v3_profile([
            {**item, "risk_at_pilot": depth[int(item["depth"])]["risk_at_pilot"],
             "policy_instability": depth[int(item["depth"])]["policy_instability"]}
            for item in risk["profile"]])
    legacy = timed("oeh3_risk", oeh3)
    results[-1]["max_abs_diff_vs_sealed"] = max(
        _max_diff(pilot[name], legacy[name]) for name in legacy if name.startswith("oeh3_"))

    learner = timed("lcoeh4_crossfit", lambda: estimate_lcoeh_v4(
        arrays, rho, num_states=S, num_actions=A, gamma=.95, seed_key=f"{key[0]}|{key[1]}"))
    results[-1]["max_abs_diff_vs_sealed"] = max(
        _max_diff(sealed["pilot_by_learner"].loc[(*key, method)][name], learner[method][name])
        for method in METHODS for name in LC_OEH_V4_FEATURES if name in sealed["pilot_by_learner"].columns)

    document = sealed["oehb"][key]
    stats = timed("posterior_fit", lambda: post.fit_posterior(post.pilot_statistics(
        arrays[10], rho, num_states=S, num_actions=A, gamma=.95), **structure))
    results[-1]["max_abs_diff_vs_sealed"] = max(
        _max_diff(document["posterior"][f"{kind}_concentration"], getattr(stats, f"{kind}_concentration"))
        for kind in ("transition", "behavior"))

    def oehb():
        outcomes = predictive.posterior_outcomes(
            arrays[10], rho, num_states=S, num_actions=A, gamma=0.95, draws=document["draws"],
            seed=list(document["seed"]), batched=None, **structure)
        return predictive.requirement_forecast(outcomes)
    forecast = timed("oehb_simulation", oehb)
    results[-1]["max_abs_diff_vs_sealed"] = max(
        _max_diff(document["learners"][name]["probabilities"], value["probabilities"])
        for name, value in forecast.items())
    results[-1]["sealed_runtime_seconds"] = document["runtime_seconds"]

    horizon = timed("oehh_horizon", lambda: R.horizon_group(
        {"row": record, "dataset": str(DATASET), "structure": structure, "oracle": False}))
    expected = sealed["horizon"].loc[key]
    diffs, labels_match = [], True
    for name in expected.index:
        if not name.startswith("data_") or name not in horizon:
            continue
        actual = np.nan if horizon[name] is None else horizon[name]
        if isinstance(actual, str) or isinstance(expected[name], str):
            labels_match &= str(actual) == str(expected[name])
        else:
            diffs.append(_max_diff(expected[name], actual))
    results[-1]["max_abs_diff_vs_sealed"] = max(diffs) if labels_match and diffs else float("inf")
    return results


def run_cost(output: Path, per_family: int) -> None:
    output = prepare_output(output, "cost")
    rows, variants = frozen_lineage(output)
    verify_cohort_seal()
    structure = json.loads(L.PROTOCOL.read_text(encoding="utf-8"))["oeh_final"]["structure"]
    with outcome_guard():
        manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
        sample = cost_sample(manifest, per_family)
        groups = pilot_groups(manifest)
        pilot = pd.read_csv(COHORT / "pilot_features.csv")
        horizon = pd.read_csv(COHORT / "horizon_preoutcome.csv").set_index(["mdp_id", "policy_family"])
        oehb, logged = {}, []
        for path in sorted((COHORT / "oeh_b_groups").glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            oehb[(document["mdp_id"], document["policy_family"])] = document
            logged.append(document["runtime_seconds"])
        if len(oehb) != 600:
            raise RuntimeError("Incomplete sealed OEH-B groups")
        sealed = {"pilot": pilot.drop_duplicates(["mdp_id", "policy_family"]).set_index(["mdp_id", "policy_family"]),
                  "pilot_by_learner": pilot.set_index(KEYS), "horizon": horizon, "oehb": oehb}
        probe = []
        for index, key in enumerate(sample, 1):
            probe.extend(cost_group(key, groups[key], structure, sealed))
            print(f"cost {index}/{len(sample)} {key[0]}|{key[1]}", flush=True)
    table = pd.DataFrame(probe)
    table.to_csv(output / "cost_probe_groups.csv", index=False)
    summary = (table.groupby("component").seconds
               .agg(groups="size", mean="mean", median="median", p90=lambda s: s.quantile(.9), max="max")
               .reindex(list(L.COMPONENTS)).reset_index())
    summary["sealed_logged_groups"] = np.where(summary.component == "oehb_simulation", len(logged), 0)
    summary["sealed_logged_mean"] = np.where(summary.component == "oehb_simulation", np.mean(logged), np.nan)
    summary["sealed_logged_median"] = np.where(summary.component == "oehb_simulation", np.median(logged), np.nan)
    summary["sealed_logged_total"] = np.where(summary.component == "oehb_simulation", np.sum(logged), np.nan)
    summary.to_csv(output / "cost_summary.csv", index=False)
    per_group = table.pivot_table(index=["mdp_id", "policy_family"], columns="component", values="seconds")
    arm_rows = []
    for name, fields in variants.items():
        needed = list(L.components_of(fields))
        # Each sealed producer call (OEH-B, OEH-H) refits the posterior itself;
        # the standalone fit is counted only if no producer would pay for it.
        paid = [c for c in needed if c != "posterior_fit" or not {"oehb_simulation", "oehh_horizon"} & set(needed)]
        seconds = per_group[paid].sum(axis=1)
        arm_rows.append({"arm": name, "columns": len(fields), "components": ";".join(needed),
                         "sample_groups": len(seconds), "mean_seconds_per_group": seconds.mean(),
                         "median_seconds_per_group": seconds.median(),
                         "estimated_sequential_seconds_600_groups": seconds.mean() * 600})
    pd.DataFrame(arm_rows).to_csv(output / "arm_feature_cost.csv", index=False)
    parity = table.groupby("component").max_abs_diff_vs_sealed.max().to_dict()
    status = "PASS" if all(value <= 1e-8 for value in parity.values()) else "FAIL"
    write_json(output / "cost_parity.json", {
        "status": status, "tolerance": 1e-8, "max_abs_diff_vs_sealed": parity,
        "sample": [list(key) for key in sample], "per_family": per_family,
        "machine": {"cpu_count": os.cpu_count(), "platform": sys.platform, "python": sys.version.split()[0]},
        "timing": "single process, sequential, wall clock; OEH-B and OEH-H calls include their own posterior fit",
        "input_inventory": input_inventory(_GUARD["opened"]).to_dict("records"),
        "written_at_utc": now(),
    })
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(arm_rows).to_string(index=False), flush=True)
    if status != "PASS":
        raise RuntimeError(f"Recomputed features differ from sealed values: {parity}")


# --------------------------------------------------------------------------- #
# Phase: forecast
# --------------------------------------------------------------------------- #

def query_features() -> pd.DataFrame:
    """Cohort-7 query rows built only from sealed pre-outcome artifacts."""
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis.develop_cross_learner_context import with_context

    pilot = pd.read_csv(COHORT / "pilot_features.csv")
    oeh = F.oeh_feature_rows(COHORT / "oeh_b_groups", COHORT / "horizon_preoutcome.csv")
    panel = pd.read_csv(COHORT / "predictions_preoutcome.csv", usecols=[*KEYS, "panel"]).drop_duplicates(KEYS)
    query = pilot.merge(oeh, on=KEYS, validate="one_to_one").merge(panel, on=KEYS, validate="one_to_one")
    query = with_context(query)
    if len(query) != 1800 or query[KEYS].duplicated().any() or query.mdp_id.nunique() != 120:
        raise RuntimeError("Incomplete query feature set")
    return query.sort_values(KEYS).reset_index(drop=True)


def check_seven_class(frame: pd.DataFrame, rows: int) -> None:
    if not set(P) <= set(frame.columns):
        raise RuntimeError("Forecast lacks one of the seven class columns")
    p = frame[P].to_numpy(float)
    if (len(frame) != rows or frame[KEYS].duplicated().any() or p.shape[1] != 7
            or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1.0, atol=1e-9)):
        raise RuntimeError("Forecast is not a complete seven-class distribution per row")


def forecast_arms(train: pd.DataFrame, query: pd.DataFrame, variants: dict) -> tuple[pd.DataFrame, dict, list]:
    """Fit every arm by the matched procedure; ``query`` must carry no outcomes."""
    from scripts.analysis import oeh_final_forecast as F

    leaked = [c for c in query.columns if c in L.LABEL_COLUMNS and c != "panel" or c.startswith("successes_")]
    if leaked:
        raise RuntimeError(f"Query carries outcome columns: {leaked}")
    for name, fields in variants.items():
        missing = set(fields) - set(train.columns) | set(fields) - set(query.columns)
        if missing:
            raise RuntimeError(f"{name}: missing columns {sorted(missing)}")
        if set(fields) & set(L.LABEL_COLUMNS) or any(c.startswith(L.PRIVILEGED_PREFIXES) for c in fields):
            raise RuntimeError(f"{name}: outcome or privileged column in the feature set")
    tables, tuning, runtime = [], {}, []
    for name, fields in variants.items():
        tick = time.perf_counter()
        probability, weights = F.matched_forecast(train, query, fields)
        runtime.append({"arm": name, "columns": len(fields), "fit_seconds": time.perf_counter() - tick})
        frame = query[[*KEYS, "panel"]].copy()
        frame.insert(4, "arm", name)
        frame[P] = probability
        check_seven_class(frame, len(query))
        tables.append(frame)
        tuning[name] = weights
        print(f"{name}: {runtime[-1]['fit_seconds']:.1f}s, {len(fields)} columns", flush=True)
    return pd.concat(tables, ignore_index=True), tuning, runtime


def reproduction(predictions: pd.DataFrame) -> dict:
    sealed = pd.read_csv(COHORT / "predictions_preoutcome.csv")
    result = {}
    for arm, method in REPRODUCE.items():
        merged = predictions[predictions.arm == arm].merge(
            sealed[sealed.method == method], on=KEYS, suffixes=("", "_sealed"), validate="one_to_one")
        if len(merged) != 1800:
            raise RuntimeError(f"Cannot pair {arm} with sealed {method}")
        result[arm] = {"sealed_method": method, "max_abs_diff": float(np.abs(
            merged[P].to_numpy() - merged[[f"{c}_sealed" for c in P]].to_numpy()).max())}
    return result


def previous_run(predictions: pd.DataFrame) -> dict:
    path = PREVIOUS / "predictions_before_label_read.csv"
    if not path.exists():
        return {}
    old = pd.read_csv(path)
    result = {}
    for arm in predictions.arm.unique():
        merged = predictions[predictions.arm == arm].merge(old[old.method == arm], on=KEYS, suffixes=("", "_old"))
        if len(merged) == 1800:
            result[arm] = float(np.abs(merged[P].to_numpy() - merged[[f"{c}_old" for c in P]].to_numpy()).max())
    return result


def run_forecast(output: Path) -> None:
    output = prepare_output(output, "forecast")
    rows, variants = frozen_lineage(output)
    start = time.perf_counter()
    with outcome_guard() as opened:
        cohort_hashes = verify_cohort_seal()
        from scripts.analysis import run_oeh_confirmatory_20261004 as C
        from scripts.analysis import run_oeh_final_120 as R

        C.configure()
        tag = json.loads(L.PROTOCOL.read_text(encoding="utf-8"))["oeh_final"]["structure_tag"]
        train = R.meta_training(R.OPENED, tag)
        query = query_features()
        if len(train) != 10800 or train.mdp_id.nunique() != 720 or set(train.mdp_id) & set(query.mdp_id):
            raise RuntimeError("Meta-training split mismatch")
        predictions, tuning, runtime = forecast_arms(train, query, variants)
        reproduced = reproduction(predictions)
        inputs = input_inventory(opened)
    if any(item["max_abs_diff"] > REPRODUCE_TOLERANCE for item in reproduced.values()):
        write_json(output / "reproduction_failure.json", reproduced)
        raise RuntimeError(f"Arms do not reproduce the sealed fit: {reproduced}")
    predictions.to_csv(output / "predictions_before_label_read.csv", index=False)
    write_json(output / "tuning.json", tuning)
    pd.DataFrame(runtime).to_csv(output / "fit_runtime.csv", index=False)
    inputs.to_csv(output / "input_inventory.csv", index=False)
    sources = source_inventory()
    sources.to_csv(output / "source_inventory.csv", index=False)
    write_json(output / "prediction_seal.json", {
        "namespace": NAMESPACE, "sealed_at_utc": now(), "labels_read": False,
        "outcome_guard": {"forbidden_files": sorted(L.OUTCOME_FILES), "forbidden_dirs": list(L.OUTCOME_DIRS)},
        "prediction_sha256": sha256(output / "predictions_before_label_read.csv"),
        "tuning_sha256": sha256(output / "tuning.json"),
        "lineage_json_sha256": sha256(output / "feature_lineage.json"),
        "input_inventory_sha256": sha256(output / "input_inventory.csv"), "input_files": len(inputs),
        "source_inventory_sha256": sha256(output / "source_inventory.csv"), "source_files": len(sources),
        "cohort_sha256": cohort_hashes,
        "meta_training": {"rows": len(train), "mdps": int(train.mdp_id.nunique()),
                          "cohorts": sorted(train.cohort.unique().tolist())},
        "query": {"rows": len(query), "mdps": int(query.mdp_id.nunique()),
                  "panels": {k: int(v) for k, v in query.drop_duplicates("mdp_id").panel.value_counts().items()}},
        "procedure": "oeh_final_forecast.matched_forecast(law=False): AFT + direct thresholds + depth3_200 "
                     "multiclass, MDP-hash five folds, OOF log-loss convex weights, per panel and learner",
        "reproduces_sealed": reproduced, "tolerance": REPRODUCE_TOLERANCE,
        "previous_exploratory_max_abs_diff": previous_run(predictions),
        "total_prelabel_seconds": time.perf_counter() - start,
    })
    print(f"Sealed {len(predictions)} predictions in {rel(output)}", flush=True)


# --------------------------------------------------------------------------- #
# Phase: score (the only phase that reads cohort-7 labels)
# --------------------------------------------------------------------------- #

def score_rows(frame: pd.DataFrame) -> pd.DataFrame:
    p = frame[P].to_numpy(float)
    y = frame.event.to_numpy(int)
    cdf = np.cumsum(p, axis=1)[:, :6]
    frame = frame.copy()
    frame["brier"] = ((cdf - (y[:, None] <= np.arange(6))) ** 2).mean(1)
    frame["log_loss"] = -np.log(np.maximum(p[np.arange(len(p)), y], 1e-12))
    return frame


def paired(scored: pd.DataFrame, references=("oeh_full_108", "b2_matched")) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    pooled = scored.assign(algorithm="pooled")
    per_mdp = (pd.concat([scored, pooled]).groupby(["panel", "algorithm", "arm", "mdp_id"], observed=True)
               [["brier", "log_loss"]].mean().reset_index())
    rows = []
    for (panel, learner), group in per_mdp.groupby(["panel", "algorithm"], observed=True):
        for metric in ("brier", "log_loss"):
            pivot = group.pivot(index="mdp_id", columns="arm", values=metric)
            if pivot.isna().any().any():
                raise RuntimeError("Unpaired MDP rows")
            draws = rng.integers(0, len(pivot), size=(BOOTSTRAP, len(pivot)))
            for reference in references:
                for arm in pivot.columns.drop(reference):
                    delta = (pivot[arm] - pivot[reference]).to_numpy(float)
                    boot = delta[draws].mean(axis=1)
                    rows.append({"panel": panel, "algorithm": learner, "metric": metric,
                                 "comparison": f"{arm} minus {reference}", "mdps": len(delta),
                                 "mean_delta": delta.mean(), "relative_delta": delta.mean() / pivot[reference].mean(),
                                 "ci95_low": np.quantile(boot, .025), "ci95_high": np.quantile(boot, .975),
                                 "better_mdps": int((delta < 0).sum()), "worse_mdps": int((delta > 0).sum())})
    return pd.DataFrame(rows)


def run_score(output: Path) -> None:
    output = prepare_output(output, "score")
    seal = json.loads((output / "prediction_seal.json").read_text(encoding="utf-8"))
    for name, key in (("predictions_before_label_read.csv", "prediction_sha256"),
                      ("feature_lineage.json", "lineage_json_sha256")):
        if sha256(output / name) != seal[key]:
            raise RuntimeError(f"Sealed file changed: {name}")
    frozen_lineage(output)
    result = json.loads((COHORT / "result_manifest.json").read_text(encoding="utf-8"))
    if sha256(COHORT / "labels.csv") != result["artifact_sha256"]["labels.csv"]:
        raise RuntimeError("Cohort-7 labels changed")
    labels = pd.read_csv(COHORT / "labels.csv")[[*KEYS, "panel", "event"]]
    predictions = pd.read_csv(output / "predictions_before_label_read.csv")
    arms = list(dict.fromkeys(predictions.arm))
    scored = score_rows(predictions.merge(labels, on=[*KEYS, "panel"], validate="many_to_one"))
    if len(scored) != 1800 * len(arms) or scored.event.isna().any():
        raise RuntimeError("Labels do not cover every sealed prediction")
    scored.to_csv(output / "scored.csv", index=False)
    summary = (pd.concat([scored, scored.assign(algorithm="pooled")])
               .groupby(["panel", "algorithm", "arm"], observed=True)
               .agg(rows=("event", "size"), mdps=("mdp_id", "nunique"), brier=("brier", "mean"),
                    log_loss=("log_loss", "mean"), censored_fraction=("event", lambda z: (z == 6).mean()))
               .reset_index())
    summary["arm"] = pd.Categorical(summary.arm, arms, ordered=True)
    summary = summary.sort_values(["panel", "algorithm", "arm"])
    summary.to_csv(output / "summary.csv", index=False)
    paired(scored).to_csv(output / "paired_mdp_delta.csv", index=False)
    cdf = np.cumsum(scored[P].to_numpy(float), axis=1)[:, :6]
    calibration = pd.concat([
        scored[["panel", "algorithm", "arm"]].assign(
            budget=budget, forecast_cdf=cdf[:, j], observed_reach=(scored.event.to_numpy() <= j).astype(float))
        for j, budget in enumerate(L.BUDGETS)]).groupby(
        ["panel", "algorithm", "arm", "budget"], observed=True)[["forecast_cdf", "observed_reach"]].mean().reset_index()
    calibration.to_csv(output / "calibration_by_budget.csv", index=False)
    names = [name for phase in ("lineage", "cost", "forecast") for name in PHASE_FILES[phase]
             if (output / name).exists()]
    names += ["scored.csv", "summary.csv", "paired_mdp_delta.csv", "calibration_by_budget.csv"]
    write_json(output / "manifest.json", {
        "namespace": NAMESPACE, "status": "complete_exploratory_opened_cohort",
        "label": "exploratory: the seventh cohort was opened on 2026-10-04; no new confirmatory claim",
        "completed_at_utc": now(), "labels_sha256": sha256(COHORT / "labels.csv"),
        "result_manifest_sha256": sha256(COHORT / "result_manifest.json"),
        "bootstrap": f"{BOOTSTRAP} MDP-resampled means, percentile 95%, seed {SEED}; descriptive only",
        "delta_definition": "arm minus reference; negative is better for both losses",
        "metrics": "IBS over six CDF thresholds; seven-class log loss with 1e-12 floor",
        "score_source_sha256": sha256(Path(__file__)),
        "output_sha256": {name: sha256(output / name) for name in names},
    })
    view = summary[(summary.panel == "primary")][["algorithm", "arm", "brier", "log_loss", "censored_fraction"]]
    print(view.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=tuple(PHASE_FILES))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cost-per-family", type=int, default=2,
                        help="instances per MDP family in the cost probe (all five behaviors each)")
    args = parser.parse_args()
    if args.phase == "lineage":
        run_lineage(args.output)
    elif args.phase == "cost":
        run_cost(args.output, args.cost_per_family)
    elif args.phase == "forecast":
        run_forecast(args.output)
    else:
        run_score(args.output)


if __name__ == "__main__":
    main()

"""Post-hoc analyses of the opened 72-MDP Colosseum confirmation panel (14/10/2026).

The Colosseum learner outcomes were opened on 2026-10-14 (02:05 UTC), so every
analysis here is exploratory. Phases, each refusing to overwrite its outputs:

``freeze``    write the analysis plan (arms, comparators, decision grid, seeds,
              diagnostics) and hash its sealed inputs before anything is fitted;
``forecast``  fit the eight block-factorial arms and the two adapted XGBoost-CF
              arms on the sealed 720-MDP meta-training panel; the full and B2
              arms must reproduce the sealed Colosseum forecasts and the
              XGBoost-CF arms the archived cohort-7 forecasts; an audit hook
              refuses every Colosseum and cohort-7 outcome file in this phase;
``score``     read the labels and write scores, block Shapley values, max-t
              contrasts under the team's rule, decision regret, the posterior
              zero-draw diagnosis and the cost-by-size table.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
from itertools import combinations
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from scripts.analysis import m1_feature_lineage as L

ROOT = L.ROOT
CONFIRM = ROOT / "artifacts/week8/colosseum_confirm_2026-10-14"
EXPLORE = ROOT / "artifacts/week8/colosseum_exploratory_2026-10-13"
COHORT7 = L.COHORT
XGB18_C7 = ROOT / "artifacts/week8/published_xgboost_cf_120_exploratory_2026-10-08"
XGB108_C7 = ROOT / "artifacts/week8/published_xgboost_cf_120_full108_architecture_2026-10-08"
BRIDGE_OEHB = {"bridge_atari5": ROOT / "artifacts/week8/bridge_fresh_full_oehb_preoutcome_2026-10-08/groups",
               "bridge_expanded5": ROOT / "artifacts/week8/bridge_expanded_full_oehb_preoutcome_2026-10-08/groups"}
OUTPUT = ROOT / "artifacts/week8/colosseum_posthoc_2026-10-14"
NAMESPACE = "week8-colosseum-posthoc-v1-20261014"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]
BUDGETS = (1, 2, 5, 10, 20, 50)
REPRODUCE_TOLERANCE = 1e-6
BOOTSTRAP, SEED = 10_000, 20261014

# Block factorial on top of B2 (M1 lineage). P = posterior simulation (direct
# OEH-B), H = OEH-H horizon/risk, L = legacy OEH3/LCOEH4 and their context.
FACTORS = {"P": ("G2", "G3", "G6"), "H": ("G4", "G5"), "L": ("G7", "G8")}
M1_NAMES = {(): "b2_matched", ("P", "H", "L"): "oeh_full_108", ("H", "L"): "oeh_without_direct_oehb",
            ("P", "L"): "oeh_without_oehh_risk", ("P", "H"): "oeh_without_legacy_diagnostics"}
XGB18, XGB108 = "xgboost_cf_18", "xgboost_cf_108"
SEALED_METHOD = {"oeh_full_108": "oeh_final", "b2_matched": "b2_matched"}
# Decision analysis (KE_HOACH_THUC_NGHIEM A3), costs in multiples of |S||A|.
TAUS, LAMBDAS = (0.5, 0.8, 0.9), (50, 100)
DECISION_METHODS = ("oeh_final", "b2_matched", "b2_plus_eri", "prior_oeh_matched", XGB18, XGB108)
FIXED_RULES = ("always_x10", "always_x50", "always_skip")

OUTCOME_FILES = tuple(CONFIRM / name for name in ("colosseum_labels.csv", "colosseum_outcomes.csv",
                                                  "colosseum_outcome_run.json")) + tuple(
    EXPLORE / name for name in ("colosseum_labels.csv", "colosseum_outcomes.csv", "colosseum_outcome_run.json")) + tuple(
    COHORT7 / name for name in L.OUTCOME_FILES)
OUTCOME_DIRS = (CONFIRM / "score", CONFIRM / "score_comparators", EXPLORE / "score",
                *(COHORT7 / name for name in L.OUTCOME_DIRS))
PHASE_FILES = {
    "freeze": ("analysis_plan.json",),
    "forecast": ("predictions_before_label_read.csv", "tuning.json", "fit_runtime.csv", "reproduction.json",
                 "input_inventory.csv", "prediction_seal.json"),
    "score": ("scored.csv", "summary.csv", "maxt_contrasts.csv", "paired_mdp_delta.csv", "block_shapley.csv",
              "decision_regret.csv", "decision_contrasts.csv", "zero_draw_diagnosis.csv",
              "cost_by_size.csv", "manifest.json"),
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


def family(mdp_id: pd.Series) -> pd.Series:
    return mdp_id.astype(str).str.extract(r"colosseum_(.+)_\d+$")[0]


# --------------------------------------------------------------------------- #
# Outcome guard (same mechanism as the M1 runner)
# --------------------------------------------------------------------------- #

def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(path))


FORBIDDEN_FILES = frozenset(_norm(path) for path in OUTCOME_FILES)
FORBIDDEN_DIRS = tuple(_norm(path) + os.sep for path in OUTCOME_DIRS)
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
    global _INSTALLED
    if not _INSTALLED:
        sys.addaudithook(_hook)
        _INSTALLED = True
    _GUARD["active"], _GUARD["opened"] = True, set()
    try:
        yield _GUARD["opened"]
    finally:
        _GUARD["active"] = False


def prepare_output(output: Path, phase: str) -> Path:
    output = output.resolve()
    for sealed in (CONFIRM, EXPLORE, COHORT7):
        if output == sealed.resolve() or sealed.resolve() in output.parents:
            raise RuntimeError(f"Output must use a new namespace, not {rel(sealed)}")
    existing = [name for name in PHASE_FILES[phase] if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite {phase} outputs: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    return output


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #

def factorial_arms() -> dict[str, tuple[str, ...]]:
    """Columns of the 2^3 arms in sealed order; the five M1 arms keep their names."""
    rows = L.lineage()
    m1 = L.arms(rows)
    result = {}
    for size in range(4):
        for subset in combinations(FACTORS, size):
            blocks = {"G1", *(g for f in subset for g in FACTORS[f])}
            name = M1_NAMES.get(subset, "b2_plus_" + "".join(subset))
            result[name] = tuple(row["column"] for row in rows if row["block"] in blocks)
            if name in m1 and m1[name] != result[name]:
                raise RuntimeError(f"{name} differs from the M1 arm definition")
    return result


def subset_of(name: str) -> tuple[str, ...]:
    inverse = {v: k for k, v in M1_NAMES.items()}
    return inverse.get(name, tuple(name.removeprefix("b2_plus_")))


# --------------------------------------------------------------------------- #
# Phase: freeze
# --------------------------------------------------------------------------- #

SEALED_INPUTS = ("protocol_colosseum.json", "colosseum_preoutcome_seal.json", "colosseum_predictions_preoutcome.csv",
                 "colosseum_pilot_features.csv", "colosseum_horizon.csv", "comparator_predictions.csv",
                 "protocol_colosseum_amendment_1_comparators.json", "colosseum_labels.csv")


def run_freeze(output: Path) -> None:
    output = prepare_output(output, "freeze")
    seal = json.loads((CONFIRM / "colosseum_preoutcome_seal.json").read_text(encoding="utf-8"))
    if seal["predictions_sha256"] != sha256(CONFIRM / "colosseum_predictions_preoutcome.csv"):
        raise RuntimeError("Sealed Colosseum predictions changed")
    arms = factorial_arms()
    write_json(output / "analysis_plan.json", {
        "namespace": NAMESPACE, "frozen_at_utc": now(), "post_hoc": True,
        "status_at_freeze": ("Colosseum labels opened 2026-10-14T02:05Z and scored by the handover (primary GO/GO, "
                             "cohort-7 comparator set NO-GO tier 1). During code review, before this plan, the "
                             "registered contrasts were rescored with the team's max-t rule from the handover's "
                             "MDP scores: primary still GO (log-loss lower bound +0.0003); this is reported, not "
                             "chosen. The count of zero-draw OEH-B groups (42/360) was read from the pre-outcome "
                             "OEH-B files; no loss by draw status had been computed."),
        "analyses": {
            "A_rescore": "registered primary 4 contrasts and the 18 comparator contrasts under the team rule "
                         "(critical = 0.95 quantile of max_j (mean*_j - mean_j)/se_j with the original se; "
                         "seeds 20261014 and 20261004 as in the handover), next to the handover's per-draw rule",
            "B_xgboost_cf": "adapted XGBoost-CF/18 and /108 exactly as published_survival_120_* (per learner, "
                            "primary meta-training stratum, median imputation, 100 trees depth 3, seed 42); "
                            "main post-hoc family = OEH vs {B2 matched, XGBoost-CF/18} x {IBS, LL}, team max-t, "
                            f"{BOOTSTRAP} draws, seed {SEED}; XGBoost-CF/108 reported as architecture control",
            "C_block_factorial": {"factors": FACTORS, "arms": {k: len(v) for k, v in arms.items()},
                                  "m1_arms_defined_before_colosseum_labels": "2026-10-13T16:20Z",
                                  "new_arms_defined_now": [k for k in arms if k.startswith("b2_plus_")],
                                  "value": "v(S) = loss(B2) - loss(B2+S), per MDP, then Shapley over P/H/L; "
                                           "MDP bootstrap percentile 95%, descriptive; pooled, per learner, per family"},
            "D_decision": {"policy": "smallest budget x_k with forecast P(E<=k) >= tau, otherwise skip",
                           "cost": "collect: x_k (+ lambda if E > k); skip: lambda if E < 6 else 0; "
                                   "oracle: x_E if E < 6 else 0; regret = cost - oracle",
                           "tau": TAUS, "lambda": LAMBDAS, "lambda_miss": "equal to lambda",
                           "methods": DECISION_METHODS, "fixed_rules": FIXED_RULES,
                           "panels": ["colosseum_confirm_72", "cohort7_primary_84 (exploratory)"],
                           "aggregate": "mean over rows within MDP, then over MDPs; OEH minus method per MDP, "
                                        f"percentile 95% from {BOOTSTRAP} MDP draws, seed {SEED}"},
            "E_zero_draw": "OEH-B groups whose posterior sampler accepted no draw: rows, label mix, OEH/B2 scores "
                           "and probability of the true class, by family and draw status (descriptive)",
            "F_cost": "OEH-B runtime per MDP-behavior group against |S||A| for cohort 7, Colosseum (355/360 "
                      "groups ran with multi-threaded OpenBLAS) and the two full-OEH BRIDGE panels",
        },
        "not_run": ["learning-curve extrapolation baseline (needs Colosseum curve features)",
                    "leave-one-family-out, OEH-S, OEH-2T, new sealed cohort (need new protocol decisions)",
                    "learner choice by expected cost"],
        "sealed_inputs_sha256": {name: sha256(CONFIRM / name) for name in SEALED_INPUTS},
        "source_sha256": {rel(Path(__file__)): sha256(Path(__file__)), rel(Path(L.__file__)): sha256(Path(L.__file__))},
    })
    print(f"Plan frozen: {rel(output / 'analysis_plan.json')}", flush=True)


def check_plan(output: Path) -> dict:
    plan = json.loads((output / "analysis_plan.json").read_text(encoding="utf-8"))
    for name, digest in plan["sealed_inputs_sha256"].items():
        if sha256(CONFIRM / name) != digest:
            raise RuntimeError(f"Sealed input changed after the plan: {name}")
    return plan


# --------------------------------------------------------------------------- #
# Phase: forecast
# --------------------------------------------------------------------------- #

def colosseum_query() -> pd.DataFrame:
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis.develop_cross_learner_context import with_context

    pilot = pd.read_csv(CONFIRM / "colosseum_pilot_features.csv")
    oeh = F.oeh_feature_rows(CONFIRM / "oeh_b_groups", CONFIRM / "colosseum_horizon.csv")
    query = with_context(pilot.merge(oeh, on=KEYS, validate="one_to_one"))
    query["panel"] = "primary"
    if len(query) != 1080 or query.mdp_id.nunique() != 72 or query[KEYS].duplicated().any():
        raise RuntimeError("Incomplete Colosseum query")
    return query.sort_values(KEYS).reset_index(drop=True)


def cohort7_query() -> pd.DataFrame:
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis.develop_cross_learner_context import with_context

    pilot = pd.read_csv(COHORT7 / "pilot_features.csv")
    oeh = F.oeh_feature_rows(COHORT7 / "oeh_b_groups", COHORT7 / "horizon_preoutcome.csv")
    panel = pd.read_csv(COHORT7 / "predictions_preoutcome.csv", usecols=[*KEYS, "panel"]).drop_duplicates(KEYS)
    query = with_context(pilot.merge(oeh, on=KEYS, validate="one_to_one").merge(panel, on=KEYS, validate="one_to_one"))
    return query[query.panel == "primary"].sort_values(KEYS).reset_index(drop=True)


def xgboost_cf(train: pd.DataFrame, query: pd.DataFrame, fields) -> tuple[np.ndarray, dict]:
    """Adapted XGBoost-CF per (panel, learner), as in published_survival_120_*."""
    from sklearn.impute import SimpleImputer
    from xgboost import XGBClassifier
    from benchmarks.bridge.fresh_published_survival_forecast import FEATURES_B2, expanded_train, predict_law
    from scripts.analysis import oeh_final_forecast as F

    out, timing = np.full((len(query), 7), np.nan), {}
    for panel in sorted(query.panel.unique()):
        for learner in sorted(query.algorithm.unique()):
            tick = time.perf_counter()
            tr = train[(train.panel == panel) & (train.algorithm == learner)]
            qi = np.flatnonzero(((query.panel == panel) & (query.algorithm == learner)).to_numpy())
            if tr.mdp_id.nunique() < 10 or not len(qi):
                raise RuntimeError("Missing train/query stratum")
            if tuple(fields) == tuple(FEATURES_B2):  # published_survival_120_exploratory.fit_law
                imputer = SimpleImputer(strategy="median")
                x = imputer.fit_transform(tr[list(fields)].to_numpy(float))
                q = imputer.transform(query.iloc[qi][list(fields)].to_numpy(float))
            else:  # published_survival_120_full_architecture
                imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                x = imputer.fit_transform(F._matrix(tr, fields))
                q = imputer.transform(F._matrix(query.iloc[qi], fields))
            xt, yt = expanded_train(x, tr.event.to_numpy(int))
            model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42,
                                  eval_metric="logloss", n_jobs=1, tree_method="hist")
            model.fit(xt, yt)
            out[qi] = predict_law(model, q)
            timing[f"{panel}/{learner}"] = time.perf_counter() - tick
    if not np.isfinite(out).all():
        raise RuntimeError("Incomplete XGBoost-CF forecast")
    return out, timing


def _max_diff(new: pd.DataFrame, old: pd.DataFrame, rows: int) -> float:
    merged = new.merge(old, on=KEYS, suffixes=("", "_old"), validate="one_to_one")
    if len(merged) != rows:
        raise RuntimeError("Cannot pair reproduced and archived forecasts")
    return float(np.abs(merged[P].to_numpy() - merged[[f"{c}_old" for c in P]].to_numpy()).max())


def run_forecast(output: Path) -> None:
    output = prepare_output(output, "forecast")
    check_plan(output)
    from benchmarks.bridge.fresh_published_survival_forecast import FEATURES_B2
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis import run_oeh_confirmatory_20261004 as C
    from scripts.analysis import run_oeh_final_120 as R

    start = time.perf_counter()
    with outcome_guard() as opened:
        C.configure()
        tag = json.loads(L.PROTOCOL.read_text(encoding="utf-8"))["oeh_final"]["structure_tag"]
        train = R.meta_training(R.OPENED, tag)
        query = colosseum_query()
        if len(train) != 10800 or train.mdp_id.nunique() != 720 or set(train.mdp_id) & set(query.mdp_id):
            raise RuntimeError("Meta-training split mismatch")
        arms = factorial_arms()
        sets = {**arms, XGB18: tuple(FEATURES_B2), XGB108: arms["oeh_full_108"]}
        tables, tuning, runtime = [], {}, []
        for name, fields in sets.items():
            if set(fields) & set(L.LABEL_COLUMNS) or any(c.startswith(L.PRIVILEGED_PREFIXES) for c in fields):
                raise RuntimeError(f"{name}: outcome or privileged column")
            tick = time.perf_counter()
            if name.startswith("xgboost"):
                probability, weights = xgboost_cf(train, query, fields)
            else:
                probability, weights = F.matched_forecast(train, query, fields)
            runtime.append({"method": name, "columns": len(fields), "fit_seconds": time.perf_counter() - tick})
            frame = query[[*KEYS, "panel"]].copy()
            frame["method"] = name
            frame[P] = probability
            tables.append(frame)
            tuning[name] = weights
            print(f"{name}: {len(fields)} columns, {runtime[-1]['fit_seconds']:.1f}s", flush=True)
        predictions = pd.concat(tables, ignore_index=True)
        p = predictions[P].to_numpy(float)
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-9):
            raise RuntimeError("Forecast is not a seven-class distribution")
        sealed = pd.read_csv(CONFIRM / "colosseum_predictions_preoutcome.csv")
        reproduction = {arm: _max_diff(predictions[predictions.method == arm][KEYS + P],
                                       sealed[sealed.method == method][KEYS + P], 1080)
                        for arm, method in SEALED_METHOD.items()}
        # The XGBoost-CF code path must reproduce the archived cohort-7 forecasts.
        c7 = cohort7_query()
        for name, fields, folder, method in ((XGB18, FEATURES_B2, XGB18_C7, "published_xgboost_cf_2026_adapted_b2"),
                                             (XGB108, arms["oeh_full_108"], XGB108_C7,
                                              "xgboost_cf_2026_full108_architecture_control")):
            law, _ = xgboost_cf(train, c7, fields)
            new = c7[KEYS].copy()
            new[P] = law
            old = pd.read_csv(folder / "predictions.csv")
            old = old[old.panel == "primary"]
            if set(old.method) != {method}:
                raise RuntimeError(f"Unexpected archived method in {rel(folder)}")
            reproduction[f"{name}_cohort7"] = _max_diff(new, old[KEYS + P], 1260)
        inputs = sorted(opened)
    write_json(output / "reproduction.json", reproduction)
    if any(value > REPRODUCE_TOLERANCE for value in reproduction.values()):
        raise RuntimeError(f"Reproduction failed: {reproduction}")
    predictions.to_csv(output / "predictions_before_label_read.csv", index=False)
    write_json(output / "tuning.json", tuning)
    pd.DataFrame(runtime).to_csv(output / "fit_runtime.csv", index=False)
    root = _norm(ROOT) + os.sep
    pd.DataFrame([{"path": rel(path), "sha256": sha256(Path(path))} for path in inputs
                  if path.startswith(root) and os.path.isfile(path) and
                  path.startswith((_norm(ROOT / "artifacts") + os.sep, _norm(ROOT / "datasets") + os.sep))]
                 ).to_csv(output / "input_inventory.csv", index=False)
    import sklearn, xgboost
    write_json(output / "prediction_seal.json", {
        "namespace": NAMESPACE, "sealed_at_utc": now(), "labels_read": False,
        "prediction_sha256": sha256(output / "predictions_before_label_read.csv"),
        "plan_sha256": sha256(output / "analysis_plan.json"),
        "input_inventory_sha256": sha256(output / "input_inventory.csv"),
        "reproduction_max_abs_diff": reproduction, "tolerance": REPRODUCE_TOLERANCE,
        "meta_training": {"rows": len(train), "mdps": int(train.mdp_id.nunique())},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "total_prelabel_seconds": time.perf_counter() - start,
    })
    print(f"Sealed {len(predictions)} predictions; reproduction {reproduction}", flush=True)


# --------------------------------------------------------------------------- #
# Phase: score
# --------------------------------------------------------------------------- #

def score_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Integrated Brier over six thresholds; log loss with the handover's 1e-15 floor."""
    p, y = frame[P].to_numpy(float), frame.event.to_numpy(int)
    frame = frame.copy()
    frame["brier"] = ((np.cumsum(p, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)
    frame["log_loss"] = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))
    return frame


def max_t(effect: np.ndarray, seed: int, draws: int = BOOTSTRAP, studentize_each_draw: bool = False):
    """Simultaneous one-sided lower bounds; the default is the team rule (fixed se)."""
    n = len(effect)
    mean, se = effect.mean(0), effect.std(0, ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(seed)
    if studentize_each_draw:  # handover rule
        stats = []
        for _ in range(draws):
            boot = effect[rng.integers(0, n, n)]
            stats.append(((boot.mean(0) - mean) / (boot.std(0, ddof=1) / np.sqrt(n))).max())
        critical = float(np.quantile(stats, .95))
    else:  # run_oeh_120_unseen.analyze
        index = rng.integers(0, n, size=(draws, n))
        critical = float(np.quantile(np.max((effect[index].mean(axis=1) - mean) / se, axis=1), .95))
    return mean, mean - critical * se, critical


def contrast_family(mdp: pd.DataFrame, baselines, candidate: str, seed: int, family_name: str,
                    rules=("team",)) -> list[dict]:
    wide = mdp.pivot(index="mdp_id", columns="method")
    names = [(b, m) for b in baselines for m in ("brier", "log_loss")]
    effect = np.column_stack([wide[(m, b)] - wide[(m, candidate)] for b, m in names])
    rows = []
    for rule in rules:
        mean, lower, critical = max_t(effect, seed, studentize_each_draw=(rule == "handover"))
        for i, (b, m) in enumerate(names):
            rows.append({"family": family_name, "rule": rule, "baseline": b, "candidate": candidate, "metric": m,
                         "mdps": len(effect), "baseline_mean": wide[(m, b)].mean(),
                         "candidate_mean": wide[(m, candidate)].mean(),
                         "relative_gain": mean[i] / wide[(m, b)].mean(), "mean_improvement": mean[i],
                         "simultaneous_lower_95": lower[i], "critical": critical,
                         "mdps_better": int((effect[:, i] > 0).sum()), "mdps_worse": int((effect[:, i] < 0).sum())})
    return rows


def percentile_ci(values: np.ndarray, index: np.ndarray) -> tuple[float, float]:
    boot = values[index].mean(axis=1)
    return float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def shapley(per_mdp: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-MDP Shapley values of P/H/L for the loss improvement over B2."""
    arms = factorial_arms()
    wide = per_mdp.pivot(index="mdp_id", columns="method", values=metric)
    value = {subset_of(name): (wide["b2_matched"] - wide[name]).to_numpy(float) for name in arms}
    value = {tuple(sorted(k, key="PHL".index)): v for k, v in value.items()}
    n, players = 3, tuple(FACTORS)
    result = {}
    for player in players:
        others = [p for p in players if p != player]
        total = np.zeros(len(wide))
        for size in range(n):
            for subset in combinations(others, size):
                weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                with_player = tuple(sorted((*subset, player), key="PHL".index))
                total += weight * (value[with_player] - value[tuple(sorted(subset, key="PHL".index))])
        result[player] = total
    return pd.DataFrame(result, index=wide.index)


def decision_costs(cdf: np.ndarray, event: np.ndarray, tau: float, lam: float) -> dict[str, np.ndarray]:
    budgets = np.asarray(BUDGETS, float)
    reach = cdf >= tau
    collect = reach.any(1)
    k = np.where(collect, reach.argmax(1), -1)
    spent = np.where(collect, budgets[np.clip(k, 0, 5)], 0.0)
    failed = collect & (event > k)
    missed = ~collect & (event < 6)
    cost = spent + lam * (failed | missed)
    oracle = np.where(event < 6, budgets[np.clip(event, 0, 5)], 0.0)
    return {"regret": cost - oracle, "cost": cost, "spent": spent, "failed": failed.astype(float),
            "missed": missed.astype(float), "skipped": (~collect).astype(float)}


def fixed_cdf(rule: str, rows: int) -> np.ndarray:
    cdf = np.zeros((rows, 6))
    if rule == "always_x10":
        cdf[:, 3:] = 1.0
    elif rule == "always_x50":
        cdf[:, 5] = 1.0
    return cdf


def decision_panel(predictions: pd.DataFrame, labels: pd.DataFrame, panel: str) -> tuple[list, list]:
    rows, contrasts = [], []
    merged = predictions.merge(labels[KEYS + ["event"]], on=KEYS, validate="many_to_one")
    ids = sorted(merged.mdp_id.unique())
    index = np.random.default_rng(SEED).integers(0, len(ids), size=(BOOTSTRAP, len(ids)))
    base = merged[merged.method == "oeh_final"].sort_values(KEYS).reset_index(drop=True)
    for tau in TAUS:
        for lam in LAMBDAS:
            per_mdp = {}
            for name in (*DECISION_METHODS, *FIXED_RULES):
                if name in FIXED_RULES:
                    frame, cdf = base, fixed_cdf(name, len(base))
                else:
                    frame = merged[merged.method == name].sort_values(KEYS).reset_index(drop=True)
                    if len(frame) != len(base):
                        continue
                    cdf = np.cumsum(frame[P].to_numpy(float), 1)[:, :6]
                parts = pd.DataFrame(decision_costs(cdf, frame.event.to_numpy(int), tau, lam))
                parts["mdp_id"] = frame.mdp_id.to_numpy()
                by_mdp = parts.groupby("mdp_id").mean().reindex(ids)
                per_mdp[name] = by_mdp
                rows.append({"panel": panel, "tau": tau, "lambda": lam, "method": name, "mdps": len(ids),
                             **{f"mean_{c}": float(by_mdp[c].mean()) for c in parts.columns if c != "mdp_id"}})
            for name, frame in per_mdp.items():
                if name == "oeh_final":
                    continue
                delta = (per_mdp["oeh_final"].regret - frame.regret).to_numpy(float)
                low, high = percentile_ci(delta, index)
                contrasts.append({"panel": panel, "tau": tau, "lambda": lam, "comparator": name,
                                  "oeh_minus_comparator_regret": float(delta.mean()), "ci95_low": low,
                                  "ci95_high": high, "mdps_oeh_better": int((delta < 0).sum()),
                                  "mdps_oeh_worse": int((delta > 0).sum())})
    return rows, contrasts


def oehb_groups(folder: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(folder / "*.json"))):
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if "mdp_id" not in doc:
            continue
        rows.append({"mdp_id": doc["mdp_id"], "policy_family": doc["policy_family"],
                     "draws_returned": doc["sampling"].get("draws_returned"),
                     "runtime_seconds": doc.get("runtime_seconds"),
                     "states": doc.get("states"), "actions": doc.get("actions")})
    return pd.DataFrame(rows)


def cost_by_size() -> pd.DataFrame:
    sizes = {}
    pilot = pd.read_csv(CONFIRM / "colosseum_pilot_features.csv", usecols=["mdp_id", "log_state_action_count"])
    sizes.update(dict(zip(pilot.mdp_id, np.rint(np.exp(pilot.log_state_action_count)))))
    pilot = pd.read_csv(COHORT7 / "pilot_features.csv", usecols=["mdp_id", "log_state_action_count"])
    sizes.update(dict(zip(pilot.mdp_id, np.rint(np.exp(pilot.log_state_action_count)))))
    panels = {"cohort7": COHORT7 / "oeh_b_groups", "colosseum_confirm": CONFIRM / "oeh_b_groups", **BRIDGE_OEHB}
    frames = []
    for name, folder in panels.items():
        groups = oehb_groups(folder)
        if groups.empty:
            continue
        sa = groups.states * groups.actions if groups.states.notna().all() else groups.mdp_id.map(sizes)
        frames.append(groups.assign(panel=name, state_actions=sa.astype(float)))
    table = pd.concat(frames, ignore_index=True)
    edges = [0, 64, 128, 256, 1024, 4096, np.inf]
    table["size_bin"] = pd.cut(table.state_actions, edges, right=True).astype(str)
    return (table.groupby(["panel", "size_bin"], observed=True)
            .agg(groups=("runtime_seconds", "size"), min_sa=("state_actions", "min"), max_sa=("state_actions", "max"),
                 median_seconds=("runtime_seconds", "median"), p90_seconds=("runtime_seconds", lambda s: s.quantile(.9)),
                 total_hours=("runtime_seconds", lambda s: s.sum() / 3600),
                 zero_draw_groups=("draws_returned", lambda s: int((s == 0).sum())))
            .reset_index())


def run_score(output: Path) -> None:
    output = prepare_output(output, "score")
    check_plan(output)
    seal = json.loads((output / "prediction_seal.json").read_text(encoding="utf-8"))
    if sha256(output / "predictions_before_label_read.csv") != seal["prediction_sha256"]:
        raise RuntimeError("Post-hoc predictions changed after their seal")
    labels = pd.read_csv(CONFIRM / "colosseum_labels.csv")
    new = pd.read_csv(output / "predictions_before_label_read.csv")
    sealed = pd.read_csv(CONFIRM / "comparator_predictions.csv")
    comparators = sealed[~sealed.method.isin(["oeh_final", "b2_matched"])]
    predictions = pd.concat([new.replace({"method": {"oeh_full_108": "oeh_final"}}),
                             comparators[[*KEYS, "panel", "method", *P]]], ignore_index=True)
    scored = score_rows(predictions.merge(labels[KEYS + ["event"]], on=KEYS, validate="many_to_one"))
    scored["family"] = family(scored.mdp_id)
    scored.to_csv(output / "scored.csv", index=False)

    def per_mdp(frame, by=()):
        return frame.groupby([*by, "method", "mdp_id"])[["brier", "log_loss"]].mean().reset_index()

    summaries = []
    for scope, by in (("pooled", ()), ("learner", ("algorithm",)), ("family", ("family",))):
        table = per_mdp(scored, by).groupby([*by, "method"])[["brier", "log_loss"]].mean().reset_index()
        table.insert(0, "scope", scope)
        summaries.append(table)
    pd.concat(summaries, ignore_index=True).to_csv(output / "summary.csv", index=False)

    # A: registered contrasts under both rules; B: post-hoc main-table family.
    mdp = per_mdp(scored)
    tier = ["b2_matched", "b2_plus_eri", "b2_plus_bwd", "train_frequency", "coverage_adaptation", "quality_adaptation",
            "coverage_quality_adaptation", "b2_generic", "b2_plus_eh_oracle_matched"]
    primary = pd.read_csv(CONFIRM / "colosseum_predictions_preoutcome.csv")
    primary_mdp = per_mdp(score_rows(primary.merge(labels[KEYS + ["event"]], on=KEYS, validate="many_to_one")))
    contrasts = (contrast_family(primary_mdp, ["b2_matched", "b2_plus_colosseum"], "oeh_final", 20261014,
                                 "registered_primary_4", ("handover", "team"))
                 + contrast_family(mdp, tier, "oeh_final", 20261004, "amendment1_cohort7_set_18", ("handover", "team"))
                 + contrast_family(mdp, ["b2_matched", XGB18], "oeh_final", SEED, "posthoc_main_table_4")
                 + contrast_family(mdp, [XGB108], "oeh_final", SEED, "posthoc_architecture_control_2"))
    pd.DataFrame(contrasts).to_csv(output / "maxt_contrasts.csv", index=False)

    # C: paired deltas and block Shapley values.
    arms = list(factorial_arms())
    arms = ["oeh_final" if a == "oeh_full_108" else a for a in arms]
    deltas, shap_rows = [], []
    for scope, by in (("pooled", None), ("learner", "algorithm"), ("family", "family")):
        groups = [("all", scored)] if by is None else list(scored.groupby(by))
        for label, frame in groups:
            table = per_mdp(frame)
            ids = sorted(table.mdp_id.unique())
            index = np.random.default_rng(SEED).integers(0, len(ids), size=(BOOTSTRAP, len(ids)))
            for metric in ("brier", "log_loss"):
                wide = table.pivot(index="mdp_id", columns="method", values=metric).reindex(ids)
                for reference in ("oeh_final", "b2_matched"):
                    for other in [*arms, XGB18, XGB108, "b2_plus_eri", "prior_oeh_matched"]:
                        if other == reference or other not in wide:
                            continue
                        delta = (wide[other] - wide[reference]).to_numpy(float)
                        low, high = percentile_ci(delta, index)
                        deltas.append({"scope": scope, "group": label, "metric": metric,
                                       "comparison": f"{other} minus {reference}", "mdps": len(ids),
                                       "mean_delta": delta.mean(), "relative_delta": delta.mean() / wide[reference].mean(),
                                       "ci95_low": low, "ci95_high": high})
                shap = shapley(table.replace({"method": {"oeh_final": "oeh_full_108"}}), metric).reindex(ids)
                total = shap.sum(axis=1).to_numpy(float)
                for player in shap.columns:
                    values = shap[player].to_numpy(float)
                    low, high = percentile_ci(values, index)
                    shap_rows.append({"scope": scope, "group": label, "metric": metric, "block": player,
                                      "blocks": "+".join(FACTORS[player]), "mdps": len(ids),
                                      "shapley_mean": values.mean(), "ci95_low": low, "ci95_high": high,
                                      "full_minus_b2_gain": total.mean(),
                                      "share_of_gain": values.mean() / total.mean() if total.mean() else np.nan})
    pd.DataFrame(deltas).to_csv(output / "paired_mdp_delta.csv", index=False)
    pd.DataFrame(shap_rows).to_csv(output / "block_shapley.csv", index=False)

    # D: decision regret on Colosseum and (exploratory) cohort 7.
    rows, dcontrasts = decision_panel(predictions, labels, "colosseum_confirm_72")
    c7_labels = pd.read_csv(COHORT7 / "labels.csv")
    c7_labels = c7_labels[c7_labels.panel == "primary"]
    c7 = pd.read_csv(COHORT7 / "predictions_preoutcome.csv")
    c7 = c7[c7.panel == "primary"]
    for folder, name in ((XGB18_C7, XGB18), (XGB108_C7, XGB108)):
        extra = pd.read_csv(folder / "predictions.csv")
        c7 = pd.concat([c7, extra[extra.panel == "primary"].assign(method=name)], ignore_index=True)
    more, more_contrasts = decision_panel(c7[[*KEYS, "method", *P]], c7_labels, "cohort7_primary_84")
    pd.DataFrame(rows + more).to_csv(output / "decision_regret.csv", index=False)
    pd.DataFrame(dcontrasts + more_contrasts).to_csv(output / "decision_contrasts.csv", index=False)

    # E: posterior zero-draw diagnosis.
    draws = oehb_groups(CONFIRM / "oeh_b_groups")[["mdp_id", "policy_family", "draws_returned"]]
    diag = scored[scored.method.isin(["oeh_final", "b2_matched", "prior_oeh_matched"])].merge(
        draws, on=["mdp_id", "policy_family"], validate="many_to_one")
    diag["zero_draw"] = diag.draws_returned == 0
    diag["p_true"] = diag[P].to_numpy(float)[np.arange(len(diag)), diag.event.to_numpy(int)]
    diagnosis = (diag.groupby(["family", "zero_draw", "method"])
                 .agg(rows=("event", "size"), mdps=("mdp_id", "nunique"), censored_share=("event", lambda e: (e == 6).mean()),
                      first_budget_share=("event", lambda e: (e == 0).mean()), brier=("brier", "mean"),
                      log_loss=("log_loss", "mean"), median_p_true=("p_true", "median"),
                      p_true_below_002=("p_true", lambda p: (p < 0.02).mean()))
                 .reset_index())
    diagnosis.to_csv(output / "zero_draw_diagnosis.csv", index=False)

    # F: cost by size.
    cost_by_size().to_csv(output / "cost_by_size.csv", index=False)

    names = [n for phase in ("freeze", "forecast") for n in PHASE_FILES[phase]] + list(PHASE_FILES["score"][:-1])
    write_json(output / "manifest.json", {
        "namespace": NAMESPACE, "status": "complete_exploratory_opened_colosseum_panel",
        "label": "post hoc: Colosseum labels were opened before every analysis in this directory",
        "completed_at_utc": now(), "labels_sha256": sha256(CONFIRM / "colosseum_labels.csv"),
        "score_source_sha256": sha256(Path(__file__)),
        "output_sha256": {name: sha256(output / name) for name in names},
    })
    view = pd.DataFrame(contrasts)
    print(view[["family", "rule", "baseline", "metric", "relative_gain", "simultaneous_lower_95"]]
          .round(4).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=tuple(PHASE_FILES))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    {"freeze": run_freeze, "forecast": run_forecast, "score": run_score}[args.phase](args.output)


if __name__ == "__main__":
    main()

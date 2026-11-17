"""OEH v2 development, cheap arm: M6 (simpler head), M1 (two-stage head), M4 (shift-aware smoothing).

Fits on the sealed 720-MDP meta-training panel only. Five opened panels are used
for **selection only** (never fitted): cohort 7 and cohort 8 (inside the
generator) and Colosseum 2027, Colosseum 2028 and SPIBB (outside it). All rules
in ``RULES`` are written to ``rules.json`` before any fit. Phases:

``fit``     heads on the meta panel, predictions for the union of the five panels;
``select``  M6 decision, M1 x M4 grid, selection with the internal constraint,
            and leave-one-external-panel-out (LOPO) estimates of the selection;
``spec``    the frozen ``oeh_v2_spec.json``: M3 decided by ``rules_m2_m3.json``
            against the sealed OEH current, parameters from the adopted selection.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "artifacts"
OUTPUT = A / "week8/oeh_v2_development_2026-10-14/heads"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]
INTERNAL, EXTERNAL = ("cohort7", "cohort8"), ("colosseum2027", "colosseum2028", "spibb")
D0 = ("const", 1.5, 2.0, 3.0, 4.0)
RULES = {
    "training": "sealed 720-MDP meta panel (R.meta_training order); opened test panels are never fitted",
    "selection_panels": {"internal": list(INTERNAL), "external": list(EXTERNAL)},
    "metric": "MDP-averaged integrated Brier (6 thresholds) and seven-class log loss (floor 1e-12)",
    "relative": "candidate / OEH current - 1 per panel",
    "m6": {"heads": ["ensemble (current)", "multiclass depth3_200 only", "AFT only", "direct thresholds only",
                     "XGBoost-CF (discrete-time survival) on 108"],
           "rule": "adopt the simple head with the best external score if its internal relative IBS and LL are both <= +1% "
                   "and its external score <= +1%; otherwise keep the ensemble"},
    "m1": {"p6": "from the chosen base head", "q_oeh": ["base finite part renormalized", "finite-only head on the 108 fields"],
           "q_robust": ["B2+legacy finite part renormalized", "finite-only head on the B2+legacy fields"],
           "lambda": "lambda_max * min(1, d / d0), or lambda_max if d0 = const", "lambda_max": [0, 0.25, 0.5, 0.75, 1.0]},
    "m4": {"rule": "p'' = (1 - eps(d)) p' + eps(d) pi_learner, eps(d) = eps_max * min(1, d / d0)",
           "eps_max": [0, 0.01, 0.02, 0.05, 0.1], "pi_learner": "class distribution of the learner on meta primary rows"},
    "d": "mean distance to the 10 nearest meta-training primary rows in standardized B2 statistics, per MDP-behavior group",
    "d0": list(D0), "d0_shared": "one d0 for M1 and M4",
    "objective": "external score = mean over the three external panels of (rel IBS + rel LL) / 2, lower is better",
    "constraint": "internal (cohort 7 and 8 pooled over MDPs) rel IBS <= +0.02 and rel LL <= +0.02",
    "parsimony": "among candidates within 0.002 of the best external score, take the smallest (lambda_max, eps_max) and "
                 "prefer d0 = const, then larger d0",
    "lopo": "for each external panel: select on the other two external panels under the same constraint, report the held-out panel",
    "success_check": "point estimate vs B2+legacy on every external panel, both metrics (descriptive here; the gate is the final test)",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def union_query() -> pd.DataFrame:
    from scripts.analysis.colosseum_posthoc_20261014 import cohort7_query
    parts = {"cohort7": cohort7_query(),
             "cohort8": pd.read_csv(A / "oeh_eighth_cohort_2026-10-14/internal/query_preoutcome.csv.gz"),
             "colosseum2027": pd.read_csv(A / "week8/colosseum_posthoc_part2_2026-10-14/query.csv.gz"),
             "colosseum2028": pd.read_csv(A / "oeh_eighth_cohort_2026-10-14/colosseum/query_preoutcome.csv.gz"),
             "spibb": pd.read_csv(A / "oeh_spibb_panel_2026-10-14/spibb/query_preoutcome.csv.gz")}
    frames = []
    for name, q in parts.items():
        q = q.copy()
        q["source_panel"], q["panel"] = name, "primary"
        frames.append(q)
    cols = set.intersection(*(set(f.columns) for f in frames))
    query = pd.concat([f[sorted(cols)] for f in frames], ignore_index=True)
    if query[KEYS].duplicated().any():
        raise RuntimeError("panel keys collide")
    return query


def labels() -> pd.DataFrame:
    srcs = {"cohort7": A / "oeh_confirmatory_unseen_120_2026-10-04/labels.csv",
            "cohort8": A / "oeh_eighth_cohort_2026-10-14/internal/labels.csv",
            "colosseum2027": A / "week8/colosseum_confirm_2026-10-14/colosseum_labels.csv",
            "colosseum2028": A / "oeh_eighth_cohort_2026-10-14/colosseum/labels.csv",
            "spibb": A / "oeh_spibb_panel_2026-10-14/spibb/labels.csv"}
    frames = []
    for name, path in srcs.items():
        lab = pd.read_csv(path)
        if "panel" in lab:
            lab = lab[lab.panel == "primary"]
        frames.append(lab[KEYS + ["event"]].assign(source_panel=name))
    return pd.concat(frames, ignore_index=True)


def distances(train: pd.DataFrame, query: pd.DataFrame, k: int = 10) -> pd.Series:
    from scipy.spatial import cKDTree
    from scripts.analysis.compare_difficulty import FEATURES_B2

    cols = list(FEATURES_B2)
    meta = train[train.panel == "primary"]
    mu, sd = meta[cols].mean(), meta[cols].std().replace(0, 1)
    tree = cKDTree(((meta[cols] - mu) / sd).fillna(0).to_numpy())
    groups = query.drop_duplicates(["mdp_id", "policy_family"])
    dist, _ = tree.query(((groups[cols] - mu) / sd).fillna(0).to_numpy(), k=k)
    d = pd.Series(dist.mean(1), index=pd.MultiIndex.from_frame(groups[["mdp_id", "policy_family"]]))
    return d.reindex(pd.MultiIndex.from_frame(query[["mdp_id", "policy_family"]])).to_numpy()


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #

def single_head(train: pd.DataFrame, query: pd.DataFrame, fields, component: str) -> np.ndarray:
    """One component of matched_forecast, fitted per (panel, learner) stratum on all training rows."""
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis.develop_eh_oracle_three_way import aft_probabilities
    from scripts.analysis.develop_oeh_difficulty import fit_thresholds
    from scripts.analysis.develop_tree_capacity import VARIANTS, multiclass
    from scripts.analysis.compare_difficulty import METHODS

    out = np.full((len(query), 7), np.nan)
    for panel in ("primary", "control"):
        for learner in METHODS:
            tr = train[(train.panel == panel) & (train.algorithm == learner)]
            qi = np.flatnonzero(((query.panel == panel) & (query.algorithm == learner)).to_numpy())
            if not len(qi):
                continue
            x, y, q = F._matrix(tr, fields), tr.event.to_numpy(int), F._matrix(query.iloc[qi], fields)
            if component == "multiclass":
                out[qi] = multiclass(x, y, q, VARIANTS["depth3_200"])
            elif component == "aft":
                out[qi] = aft_probabilities(x, y, q)
            else:
                out[qi] = fit_thresholds(x, y, q)
    return out


_CACHE: dict = {}


def _fit(task):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis.colosseum_posthoc_20261014 import xgboost_cf

    for key in ("train", "query"):
        if task[key] not in _CACHE:
            _CACHE[task[key]] = pd.read_pickle(task[key])
    train, query = _CACHE[task["train"]], _CACHE[task["query"]]
    if task.get("finite_only"):
        train = train[train.event < 6]
    fields, kind = tuple(task["fields"]), task["kind"]
    tick = time.perf_counter()
    if kind == "ensemble":
        p, _ = F.matched_forecast(train, query, fields)
    elif kind == "xgboost_cf":
        p, _ = xgboost_cf(train, query, fields)
    else:
        p = single_head(train, query, fields, kind)
    if task.get("finite_only"):
        p = np.column_stack([p[:, :6] / p[:, :6].sum(1, keepdims=True), np.zeros(len(p))])
    return task["name"], p, time.perf_counter() - tick


M3_COHORT = A / "oeh_v2_behavior_cohort_2026-10-14"


def augmented_training(train: pd.DataFrame) -> pd.DataFrame:
    """Meta panel plus the M3 behavior cohort (rules_m2_m3.json)."""
    rows = pd.read_csv(M3_COHORT / "inputs/query.csv.gz").merge(pd.read_csv(M3_COHORT / "labels.csv")[KEYS + ["event"]],
                                                                 on=KEYS, validate="one_to_one")
    rows["cohort"], rows["panel"] = "v2_behavior", "primary"
    from scripts.analysis.develop_cohort8_candidates_20261014 import field_sets
    needed = {c for fields in field_sets().values() for c in fields} | set(KEYS) | {"event", "panel"}
    missing = sorted(needed - set(rows.columns))
    if missing:
        raise RuntimeError(f"M3 rows lack fitted columns: {missing[:5]}")
    return pd.concat([train, rows[[c for c in train.columns if c in rows.columns]]], ignore_index=True)


def run_fit(output: Path, workers: int, augment: bool = False) -> None:
    from scripts.analysis.develop_cohort8_candidates_20261014 import field_sets, meta_panel

    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.csv.gz").exists():
        raise FileExistsError("fit outputs exist")
    if augment:
        (output / "rules.json").write_bytes((OUTPUT / "rules.json").read_bytes())  # identical rules
    else:
        (output / "rules.json").write_text(json.dumps({"frozen_at_utc": now(), "rules": RULES,
                                                       "source_sha256": sha256(Path(__file__))}, indent=2), encoding="utf-8")
    train = meta_panel(sort=False)
    if augment:
        train = augmented_training(train)
    query = union_query()
    sets = field_sets()
    oeh, legacy = sets["oeh_final"], sets["prior_oeh_matched"]
    tpath, qpath = output / "_train.pkl", output / "_query.pkl"
    train.to_pickle(tpath)
    query.to_pickle(qpath)
    specs = [("oeh_ensemble", oeh, "ensemble", False), ("legacy_ensemble", legacy, "ensemble", False),
             ("cond_oeh_ensemble", oeh, "ensemble", True), ("cond_legacy_ensemble", legacy, "ensemble", True),
             ("oeh_multiclass", oeh, "multiclass", False), ("oeh_aft", oeh, "aft", False),
             ("oeh_thresholds", oeh, "thresholds", False), ("oeh_xgboost_cf", oeh, "xgboost_cf", False),
             ("cond_oeh_multiclass", oeh, "multiclass", True), ("cond_legacy_multiclass", legacy, "multiclass", True),
             ("legacy_multiclass", legacy, "multiclass", False)]
    tasks = [{"name": n, "fields": list(f), "kind": k, "finite_only": c, "train": str(tpath), "query": str(qpath)}
             for n, f, k, c in specs]
    frames, runtime = [], {}
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        for name, p, seconds in pool.map(_fit, tasks):
            frame = query[KEYS + ["source_panel"]].copy()
            frame["method"] = name
            frame[P] = p
            frames.append(frame)
            runtime[name] = seconds
            print(f"fit {name}: {seconds:.0f}s", flush=True)
    tpath.unlink()
    qpath.unlink()
    pred = pd.concat(frames, ignore_index=True)
    pred.to_csv(output / "predictions.csv.gz", index=False)
    d = pd.DataFrame({"mdp_id": query.mdp_id, "policy_family": query.policy_family, "algorithm": query.algorithm,
                      "source_panel": query.source_panel, "distance": distances(train, query)})
    d.to_csv(output / "distances.csv", index=False)
    prior = {learner: (np.bincount(g.event.to_numpy(int), minlength=7) / len(g)).tolist()
             for learner, g in train[train.panel == "primary"].groupby("algorithm")}
    (output / "fit_manifest.json").write_text(json.dumps({
        "fitted_at_utc": now(), "runtime_seconds": runtime, "class_prior": prior,
        "predictions_sha256": sha256(output / "predictions.csv.gz")}, indent=2), encoding="utf-8")
    # reproduction of the sealed OEH current forecasts on every panel
    sealed = {"cohort7": A / "oeh_confirmatory_unseen_120_2026-10-04/predictions_preoutcome.csv",
              "cohort8": A / "oeh_eighth_cohort_2026-10-14/internal/predictions_preoutcome.csv",
              "colosseum2027": A / "week8/colosseum_confirm_2026-10-14/colosseum_predictions_preoutcome.csv",
              "colosseum2028": A / "oeh_eighth_cohort_2026-10-14/colosseum/predictions_preoutcome.csv",
              "spibb": A / "oeh_spibb_panel_2026-10-14/spibb/predictions_preoutcome.csv"}
    mine = pred[pred.method == "oeh_ensemble"]
    check = {}
    for name, path in sealed.items():
        s = pd.read_csv(path)
        s = s[s.method == "oeh_final"][KEYS + P]
        j = mine[mine.source_panel == name].merge(s, on=KEYS, suffixes=("", "_s"))
        check[name] = float(np.abs(j[P].to_numpy() - j[[f"{c}_s" for c in P]].to_numpy()).max())
    (output / "reproduction.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    print("reproduction of sealed OEH:", check, flush=True)
    if not augment and max(check.values()) > 1e-6:
        raise RuntimeError("the refit does not reproduce the sealed OEH forecasts")


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def panel_scores(p: np.ndarray, frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    y = frame.event.to_numpy(int)
    ibs = ((np.cumsum(p, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)
    ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    s = pd.DataFrame({"panel": frame.source_panel.to_numpy(), "mdp": frame.mdp_id.to_numpy(), "ibs": ibs, "ll": ll})
    per_mdp = s.groupby(["panel", "mdp"])[["ibs", "ll"]].mean().reset_index()
    out = {k: (g.ibs.mean(), g.ll.mean()) for k, g in per_mdp.groupby("panel")}
    internal = per_mdp[per_mdp.panel.isin(INTERNAL)]
    out["internal"] = (internal.ibs.mean(), internal.ll.mean())
    return out


CONDITIONAL = {"oeh_ensemble": ("cond_oeh_ensemble", "legacy_ensemble", "cond_legacy_ensemble"),
               "oeh_multiclass": ("cond_oeh_multiclass", "legacy_multiclass", "cond_legacy_multiclass")}


def run_select(output: Path) -> None:
    rules = json.loads((output / "rules.json").read_text(encoding="utf-8"))
    pred = pd.read_csv(output / "predictions.csv.gz")
    lab = labels()
    dist = pd.read_csv(output / "distances.csv")
    prior = json.loads((output / "fit_manifest.json").read_text(encoding="utf-8"))["class_prior"]
    frame = pred[pred.method == "oeh_ensemble"][KEYS + ["source_panel"]].merge(lab, on=KEYS + ["source_panel"], validate="one_to_one")
    frame = frame.merge(dist, on=KEYS + ["source_panel"], validate="one_to_one").reset_index(drop=True)
    index = pd.MultiIndex.from_frame(frame[KEYS])
    laws = {m: g.set_index(KEYS).loc[index][P].to_numpy(float) for m, g in pred.groupby("method")}
    pi = np.stack([np.asarray(prior[a]) for a in frame.algorithm])
    d = frame.distance.to_numpy()

    def relative(scores, ref):
        return {k: (scores[k][0] / ref[k][0] - 1, scores[k][1] / ref[k][1] - 1) for k in scores}

    ref = panel_scores(laws["oeh_ensemble"], frame)
    report = {"reference_oeh_current": ref}
    # M6
    m6 = {}
    for head in ("oeh_multiclass", "oeh_aft", "oeh_thresholds", "oeh_xgboost_cf"):
        r = relative(panel_scores(laws[head], frame), ref)
        ext = float(np.mean([(r[p][0] + r[p][1]) / 2 for p in EXTERNAL]))
        m6[head] = {"internal_rel": r["internal"], "external_score": ext, "by_panel": r}
    ok = {h: v for h, v in m6.items() if v["internal_rel"][0] <= 0.01 and v["internal_rel"][1] <= 0.01 and v["external_score"] <= 0.01}
    base = min(ok, key=lambda h: ok[h]["external_score"]) if ok else "oeh_ensemble"
    report["m6"] = {"candidates": m6, "chosen_base": base}
    cond_oeh_name, legacy_name, cond_legacy_name = CONDITIONAL.get(base, CONDITIONAL["oeh_ensemble"])
    base_law = laws[base]
    p6 = base_law[:, 6:7]

    def finite(law):
        return law[:, :6] / np.clip(law[:, :6].sum(1, keepdims=True), 1e-300, None)

    q_oeh = {"base_finite": finite(base_law), "finite_only_head": finite(laws[cond_oeh_name])}
    q_rob = {"legacy_finite": finite(laws[legacy_name]), "legacy_finite_only_head": finite(laws[cond_legacy_name])}

    def weight(maxv, d0):
        return np.full_like(d, maxv) if d0 == "const" else maxv * np.minimum(1.0, d / float(d0))

    rows = []
    for (qo, qr, lam, eps, d0) in product(q_oeh, q_rob, rules["rules"]["m1"]["lambda_max"], rules["rules"]["m4"]["eps_max"], D0):
        lw, ew = weight(lam, d0)[:, None], weight(eps, d0)[:, None]
        q = (1 - lw) * q_oeh[qo] + lw * q_rob[qr]
        law = np.column_stack([q * (1 - p6), p6])
        law = (1 - ew) * law + ew * pi
        sc = panel_scores(law, frame)
        r = relative(sc, ref)
        rows.append({"q_oeh": qo, "q_robust": qr, "lambda_max": lam, "eps_max": eps, "d0": str(d0),
                     "internal_rel_ibs": r["internal"][0], "internal_rel_ll": r["internal"][1],
                     **{f"{p}_rel_ibs": r[p][0] for p in EXTERNAL}, **{f"{p}_rel_ll": r[p][1] for p in EXTERNAL},
                     **{f"{p}_ibs": sc[p][0] for p in (*INTERNAL, *EXTERNAL)}, **{f"{p}_ll": sc[p][1] for p in (*INTERNAL, *EXTERNAL)}})
    grid = pd.DataFrame(rows)
    grid["external_score"] = np.mean([(grid[f"{p}_rel_ibs"] + grid[f"{p}_rel_ll"]) / 2 for p in EXTERNAL], axis=0)
    grid["feasible"] = (grid.internal_rel_ibs <= 0.02) & (grid.internal_rel_ll <= 0.02)
    grid.to_csv(output / "grid.csv", index=False)
    d0_rank = {"const": 0, "4.0": 1, "3.0": 2, "2.0": 3, "1.5": 4}

    def choose(frame_grid, panels):
        g = frame_grid[frame_grid.feasible].copy()
        g["score"] = np.mean([(g[f"{p}_rel_ibs"] + g[f"{p}_rel_ll"]) / 2 for p in panels], axis=0)
        best = g.score.min()
        near = g[g.score <= best + 0.002].copy()
        near["rank"] = near.d0.map(d0_rank)
        near = near.sort_values(["lambda_max", "eps_max", "rank", "score"])
        return near.iloc[0]

    chosen = choose(grid, EXTERNAL)
    lopo = []
    for held in EXTERNAL:
        others = [p for p in EXTERNAL if p != held]
        pick = choose(grid, others)
        lopo.append({"held_out": held, "chosen": {k: pick[k] for k in ("q_oeh", "q_robust", "lambda_max", "eps_max", "d0")},
                     "held_rel_ibs": float(pick[f"{held}_rel_ibs"]), "held_rel_ll": float(pick[f"{held}_rel_ll"])})
    # success check vs B2+legacy (unconditional legacy ensemble)
    leg = panel_scores(laws["legacy_ensemble"], frame)
    fin = {p: (float(chosen[f"{p}_ibs"]), float(chosen[f"{p}_ll"])) for p in (*INTERNAL, *EXTERNAL)}
    report.update({"chosen": {k: (v.item() if hasattr(v, "item") else v) for k, v in chosen.items()},
                   "lopo": lopo, "legacy_reference": leg, "chosen_scores": fin,
                   "beats_legacy_external": {p: bool(fin[p][0] < leg[p][0] and fin[p][1] < leg[p][1]) for p in EXTERNAL},
                   "selected_at_utc": now(), "grid_sha256": sha256(output / "grid.csv")})
    (output / "selection.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 250)
    print("reference", {k: tuple(round(x, 4) for x in v) for k, v in ref.items()})
    print("M6", {h: (tuple(round(x, 4) for x in v["internal_rel"]), round(v["external_score"], 4)) for h, v in m6.items()}, "->", base)
    print("chosen", {k: chosen[k] for k in ("q_oeh", "q_robust", "lambda_max", "eps_max", "d0", "external_score", "internal_rel_ibs", "internal_rel_ll")})
    print("chosen scores", {k: tuple(round(x, 4) for x in v) for k, v in fin.items()})
    print("legacy", {k: tuple(round(x, 4) for x in v) for k, v in leg.items()})
    print("LOPO", json.dumps(lopo, default=float))


# --------------------------------------------------------------------------- #
# Frozen specification (deterministic from the selections and the M3 rule)
# --------------------------------------------------------------------------- #

SPEC = OUTPUT.parent / "oeh_v2_spec.json"
M3_OUTPUT = OUTPUT.parent / "heads_m3"


def chosen_law_scores(output: Path) -> dict[str, tuple[float, float]]:
    """Panel scores of the v2 law chosen in ``output`` (recomputed from its predictions)."""
    from scripts.analysis import oeh_v2_forecast as V

    report = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    sel, base = report["chosen"], report["m6"]["chosen_base"]
    cond_name, legacy_name, cond_legacy_name = CONDITIONAL[base]
    pred = pd.read_csv(output / "predictions.csv.gz")
    frame = pred[pred.method == "oeh_ensemble"][KEYS + ["source_panel"]].merge(labels(), on=KEYS + ["source_panel"], validate="one_to_one")
    frame = frame.merge(pd.read_csv(output / "distances.csv"), on=KEYS + ["source_panel"], validate="one_to_one").reset_index(drop=True)
    index = pd.MultiIndex.from_frame(frame[KEYS])
    laws = {m: g.set_index(KEYS).loc[index][P].to_numpy(float) for m, g in pred.groupby("method")}
    prior = json.loads((output / "fit_manifest.json").read_text(encoding="utf-8"))["class_prior"]
    pi = np.stack([np.asarray(prior[a]) for a in frame.algorithm])
    cond = laws[cond_name] if sel["q_oeh"] == "finite_only_head" else laws[base]
    robust = laws[cond_legacy_name] if sel["q_robust"] == "legacy_finite_only_head" else laws[legacy_name]
    d0 = sel["d0"] if sel["d0"] == "const" else float(sel["d0"])
    law = V.v2_law(laws[base], cond, robust, frame.distance.to_numpy(), pi,
                   lambda_max=sel["lambda_max"], eps_max=sel["eps_max"], d0=d0)
    return panel_scores(law, frame)


def run_spec() -> None:
    if SPEC.exists():
        raise FileExistsError("OEH v2 is already frozen")
    rules = json.loads((OUTPUT.parent / "rules_m2_m3.json").read_text(encoding="utf-8"))
    ref = {k: tuple(v) for k, v in json.loads((OUTPUT / "selection.json").read_text(encoding="utf-8"))["reference_oeh_current"].items()}

    def against_sealed(scores):  # relative to the sealed OEH current, the reference of every rule
        rel = {k: (float(scores[k][0] / ref[k][0] - 1), float(scores[k][1] / ref[k][1] - 1)) for k in ref}
        return {"by_panel": rel, "external_score": float(np.mean([(rel[p][0] + rel[p][1]) / 2 for p in EXTERNAL])),
                "spibb_score": float((rel["spibb"][0] + rel["spibb"][1]) / 2)}

    without = against_sealed(chosen_law_scores(OUTPUT))
    m3 = {"rule": rules["m3"]["adopt_if"], "without": without}
    if (M3_OUTPUT / "selection.json").exists():
        with_m3 = against_sealed(chosen_law_scores(M3_OUTPUT))
        checks = {"external_lower": bool(with_m3["external_score"] < without["external_score"]),
                  "spibb_lower": bool(with_m3["spibb_score"] < without["spibb_score"]),
                  "internal_within_2pct": bool(max(with_m3["by_panel"]["internal"]) <= 0.02)}
        refit = {k: tuple(v) for k, v in json.loads((M3_OUTPUT / "selection.json").read_text(encoding="utf-8"))["reference_oeh_current"].items()}
        m3.update({"with": with_m3, "checks": checks, "adopted": all(checks.values()),
                   "oeh_current_refit_on_augmented_panel": against_sealed(refit)})
    else:
        m3.update({"adopted": False, "reason": "M3 cohort not available"})
    source = M3_OUTPUT if m3["adopted"] else OUTPUT
    chosen = json.loads((source / "selection.json").read_text(encoding="utf-8"))["chosen"]
    files = [OUTPUT / "rules.json", OUTPUT.parent / "rules_m2_m3.json", source / "selection.json", source / "grid.csv",
             ROOT / "scripts/analysis/oeh_v2_forecast.py", Path(__file__)]
    spec = {
        "schema": "oeh-v2-spec-v1", "status": "frozen", "frozen_at_utc": now(),
        "definition": "scripts/analysis/oeh_v2_forecast.py:v2_law",
        "base_head": json.loads((source / "selection.json").read_text(encoding="utf-8"))["m6"]["chosen_base"],
        "parameters": {"q_oeh": chosen["q_oeh"], "q_robust": chosen["q_robust"], "lambda_max": float(chosen["lambda_max"]),
                       "eps_max": float(chosen["eps_max"]), "d0": chosen["d0"] if chosen["d0"] == "const" else float(chosen["d0"])},
        "distance": RULES["d"], "selected_in": str(source.relative_to(ROOT)).replace("\\", "/"),
        "m2": "stopped at step 0 (prior_diagnosis/): the continuation rule was not met",
        "m3": m3,
        "extra_training_cohorts": ([{"name": "v2_behavior", "query": "artifacts/oeh_v2_behavior_cohort_2026-10-14/inputs/query.csv.gz",
                                     "labels": "artifacts/oeh_v2_behavior_cohort_2026-10-14/labels.csv",
                                     "query_sha256": sha256(M3_COHORT / "inputs/query.csv.gz"),
                                     "labels_sha256": sha256(M3_COHORT / "labels.csv")}] if m3["adopted"] else []),
        "source_sha256": {str(f.relative_to(ROOT)).replace("\\", "/"): sha256(f) for f in files},
    }
    SPEC.write_text(json.dumps(spec, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"parameters": spec["parameters"], "m3_adopted": m3["adopted"], "checks": m3.get("checks")}, default=float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "select", "spec"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--augment", action="store_true", help="meta panel + M3 behavior cohort")
    args = parser.parse_args()
    if args.phase == "fit":
        run_fit(args.output, args.workers, args.augment)
    elif args.phase == "select":
        run_select(args.output)
    else:
        run_spec()


if __name__ == "__main__":
    main()

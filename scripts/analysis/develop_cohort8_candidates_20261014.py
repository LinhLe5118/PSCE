"""Development of the eighth-cohort candidates and comparators (14/10/2026).

Every choice here is made on the opened 720-MDP meta-training panel only;
cohort 7, Colosseum and BRIDGE outcomes are never read. Phases:

``oof``     out-of-fold forecasts on the meta panel under two schemes: five
            MDP-hash folds (``mdp5``) and leave-one-primary-family-out
            (``lofo``), for OEH current, B2, B2 + ERI, B2 + legacy, the new
            learning-curve extrapolation (LCE) and concentrability (CB) field
            sets, XGBoost-CF/18 and the conditional heads of OEH-2T;
``select``  apply the rules frozen in ``RULES`` to choose the OEH-S
            temperature/mixture, the OEH-2T variant and report A5 (family
            transfer); writes ``selection.json`` used by the cohort-8 protocol.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/week8/cohort8_development_2026-10-14"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]
BUDGETS = (1, 2, 5, 10, 20, 50)
CURVE_BUDGETS = (1, 2, 5, 10)
PRIMARY_FAMILIES = ("bottleneck", "forward_layered", "uniform_garnet")
OUTER_SALT = "cohort8-dev-outer|"
LCE_FLOOR, LCE_MIX = 0.005, 0.01  # fixed a priori, never fitted
CB_INFINITE_CAP = 25.0            # log-concentrability used when the optimal pair is unsupported
# Frozen before ``oof`` was run (14/10/2026).
RULES = {
    "oeh_s": {"grid_T": [1.0, 1.25, 1.5, 2.0], "grid_eps": [0.0, 0.01, 0.02, 0.05],
              "criterion": "pooled lofo out-of-fold seven-class log loss of OEH current, primary panel",
              "prior": "class distribution of the learner in the primary meta-training rows",
              "enter_cohort8": "only if the chosen (T, eps) differs from (1, 0)",
              "predeclared_fallback": "probability floor 0.005 for every class, renormalized (not fitted)"},
    "oeh_2t": {"p_censored": "p_6 of OEH current",
               "conditional": ["B2 + legacy (prior_oeh_matched fields) on finite rows",
                               "OEH 108 fields on finite rows"],
               "blend": "convex blend w * 2T + (1 - w) * OEH current, w in {0.25, 0.5, 0.75}",
               "criterion": "mdp5 out-of-fold log loss; the candidate must also not raise integrated Brier",
               "enter_cohort8": "best of {2T variants, blends} if it beats OEH current on mdp5 log loss and "
                                "does not raise its IBS; otherwise no 2T candidate"},
    "lce": {"fit": "logistic s(b) = sigmoid(a + c log b), c >= 0, Bernoulli loss on the four posterior "
                   "prefix success rates (x1, x2, x5, x10)",
            "law": f"CDF_k = {LCE_FLOOR} + (1 - 2*{LCE_FLOOR}) s(x_k); mixed {1 - LCE_MIX}/{LCE_MIX} with uniform",
            "variants": ["lce_raw (no meta-training)", "b2_plus_lce (B2 + LCE quantities, matched head)"]},
    "cb": {"fields": "log n_hat = log SA + log clipped C*; log C*; log clipped C*; log intrinsic bound; "
                     "unsupported flag (posterior-mean model, the G4 producer)",
           "variants": ["cb_only (matched head)", "b2_plus_cb"], "infinite_cap": CB_INFINITE_CAP},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def family_of(mdp_id: pd.Series) -> pd.Series:
    text = mdp_id.astype(str)
    out = pd.Series("other", index=mdp_id.index)
    for name in (*PRIMARY_FAMILIES, "lock_structured"):
        out[text.str.contains(name)] = name
    return out


def outer_fold(mdp_id: str) -> int:
    return int.from_bytes(hashlib.sha256((OUTER_SALT + mdp_id).encode()).digest()[:4], "little") % 5


# --------------------------------------------------------------------------- #
# New field sets
# --------------------------------------------------------------------------- #

def lce_fit(success: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized constrained logistic fit; rows of ``success`` match CURVE_BUDGETS."""
    s = np.clip(np.asarray(success, float), 1e-3, 1 - 1e-3)
    x = np.log(np.asarray(CURVE_BUDGETS, float))
    a = np.log(s.mean(1) / (1 - s.mean(1)))
    c = np.zeros(len(s))
    for _ in range(100):
        z = 1 / (1 + np.exp(-(a[:, None] + c[:, None] * x[None])))
        r = z - s
        w = np.maximum(z * (1 - z), 1e-9)
        g = np.stack([r.sum(1), (r * x).sum(1)], 1)
        h00, h01, h11 = w.sum(1), (w * x).sum(1), (w * x * x).sum(1)
        det = np.maximum(h00 * h11 - h01 ** 2, 1e-12)
        da = (h11 * g[:, 0] - h01 * g[:, 1]) / det
        dc = (-h01 * g[:, 0] + h00 * g[:, 1]) / det
        a, c = a - np.clip(da, -5, 5), c - np.clip(dc, -5, 5)
        negative = c < 0
        if negative.any():  # project to c = 0: best constant fit
            c[negative] = 0.0
            m = s[negative].mean(1)
            a[negative] = np.log(m / (1 - m))
    return a, c


def lce_law(frame: pd.DataFrame) -> np.ndarray:
    a, c = lce_fit(frame[[f"oehc_success_x{b}" for b in CURVE_BUDGETS]].to_numpy(float))
    cdf = 1 / (1 + np.exp(-(a[:, None] + c[:, None] * np.log(np.asarray(BUDGETS, float))[None])))
    cdf = LCE_FLOOR + (1 - 2 * LCE_FLOOR) * np.maximum.accumulate(cdf, axis=1)
    p = np.column_stack([cdf[:, 0], np.diff(cdf, axis=1), 1 - cdf[:, -1]])
    p = (1 - LCE_MIX) * p + LCE_MIX / 7
    return p / p.sum(1, keepdims=True)


def add_fields(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    a, c = lce_fit(frame[[f"oehc_success_x{b}" for b in CURVE_BUDGETS]].to_numpy(float))
    frame["lce_intercept"], frame["lce_slope"] = a, c
    for b in BUDGETS:
        frame[f"lce_s_x{b}"] = 1 / (1 + np.exp(-(a + c * np.log(b))))

    def cap(name):
        value = frame[name].astype(float)
        return value.where(np.isfinite(value), CB_INFINITE_CAP)

    unsupported = ~np.isfinite(frame["oehh_log_clipped_concentrability"].astype(float))
    frame["cb_log_c_star"] = cap("oehh_log_concentrability")
    frame["cb_log_c_clipped"] = cap("oehh_log_clipped_concentrability")
    frame["cb_log_intrinsic"] = cap("oehh_log_intrinsic_bound")
    frame["cb_log_n_hat"] = frame.log_state_action_count + frame.cb_log_c_clipped
    frame["cb_unsupported"] = unsupported.astype(float)
    return frame


LCE_FIELDS = ("lce_intercept", "lce_slope", *(f"lce_s_x{b}" for b in BUDGETS))
CB_FIELDS = ("cb_log_n_hat", "cb_log_c_star", "cb_log_c_clipped", "cb_log_intrinsic", "cb_unsupported")


def field_sets() -> dict[str, tuple[str, ...]]:
    from scripts.analysis import run_oeh_final_120 as R
    from scripts.analysis.compare_difficulty import FEATURES_B2

    b2 = tuple(FEATURES_B2)
    return {"oeh_final": tuple(R.CANDIDATE_VARIANTS["plus_prior_profile"]), "b2_matched": b2,
            "b2_plus_eri": tuple(R.FIELD_SETS["b2_plus_eri"]), "prior_oeh_matched": tuple(R.FIELD_SETS["prior_oeh_matched"]),
            "b2_plus_lce": (*b2, *LCE_FIELDS), "b2_plus_cb": (*b2, *CB_FIELDS), "cb_only": CB_FIELDS}


# --------------------------------------------------------------------------- #
# Phase: oof
# --------------------------------------------------------------------------- #

def meta_panel(sort: bool = True) -> pd.DataFrame:
    """Meta panel with the new fields. ``sort=True`` is the order used by the development OOF;
    ``sort=False`` keeps ``R.meta_training`` row order, which the sealed forecasts need: the
    xgboost heads are not invariant to row order (max |diff| 0.12 on Colosseum)."""
    from scripts.analysis import run_oeh_confirmatory_20261004 as C
    from scripts.analysis import run_oeh_final_120 as R

    C.configure()
    tag = json.loads((ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04/protocol.json")
                     .read_text(encoding="utf-8"))["oeh_final"]["structure_tag"]
    train = R.meta_training(R.OPENED, tag)
    if len(train) != 10800 or train.mdp_id.nunique() != 720:
        raise RuntimeError("Meta-training panel changed")
    train = add_fields(train)
    train["family"] = family_of(train.mdp_id)
    train["outer"] = train.mdp_id.map(outer_fold)
    return train.sort_values(KEYS).reset_index(drop=True) if sort else train


_PANEL: dict = {}


def _fit(task: dict) -> pd.DataFrame:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis.colosseum_posthoc_20261014 import xgboost_cf

    if "panel" not in _PANEL:
        _PANEL["panel"] = pd.read_pickle(task["panel_path"])
    panel = _PANEL["panel"]
    if task["scheme"] == "mdp5":
        query_mask = panel.outer == task["fold"]
    else:
        query_mask = (panel.panel == "primary") & (panel.family == task["fold"])
    train, query = panel[~query_mask], panel[query_mask]
    method, fields = task["method"], tuple(task["fields"])
    tick = time.perf_counter()
    if method == "xgboost_cf_18":
        probability, _ = xgboost_cf(train, query, fields)
    elif method.startswith("cond_"):
        finite = train[train.event < 6]
        probability, _ = F.matched_forecast(finite, query, fields)
        probability = probability[:, :6] / probability[:, :6].sum(1, keepdims=True)
        probability = np.column_stack([probability, np.zeros(len(probability))])
    else:
        probability, _ = F.matched_forecast(train, query, fields)
    out = query[[*KEYS, "panel", "family", "event"]].copy()
    out["scheme"], out["fold"], out["method"] = task["scheme"], str(task["fold"]), method
    out[P] = probability
    out["fit_seconds"] = time.perf_counter() - tick
    return out


def run_oof(output: Path, workers: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    target = output / "oof_predictions.csv.gz"
    if target.exists():
        raise FileExistsError(target)
    write_json(output / "rules.json", {"frozen_at_utc": now(), "rules": RULES, "outer_salt": OUTER_SALT,
                                       "source_sha256": sha256(Path(__file__))})
    panel = meta_panel()
    panel_path = output / "_meta_panel.pkl"
    panel.to_pickle(panel_path)
    sets = field_sets()
    methods = {**sets, "xgboost_cf_18": sets["b2_matched"],
               "cond_prior": sets["prior_oeh_matched"], "cond_full": sets["oeh_final"]}
    tasks = [{"scheme": scheme, "fold": fold, "method": method, "fields": list(fields), "panel_path": str(panel_path)}
             for scheme, folds in (("mdp5", range(5)), ("lofo", PRIMARY_FAMILIES))
             for fold in folds for method, fields in methods.items()]
    tasks.sort(key=lambda t: -len(t["fields"]))  # longest first
    tick = time.perf_counter()
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, frame in enumerate(pool.map(_fit, tasks), 1):
            frames.append(frame)
            print(f"oof {done}/{len(tasks)} {frame.method.iloc[0]} {frame.scheme.iloc[0]}/{frame.fold.iloc[0]} "
                  f"{frame.fit_seconds.iloc[0]:.0f}s ({time.perf_counter() - tick:.0f}s)", flush=True)
    result = pd.concat(frames, ignore_index=True)
    # LCE raw needs no fit: evaluate it on the same rows for both schemes.
    for scheme in ("mdp5", "lofo"):
        rows = result[(result.scheme == scheme) & (result.method == "oeh_final")][[*KEYS, "panel", "family", "event", "fold"]]
        joined = rows.merge(panel[KEYS + [f"oehc_success_x{b}" for b in CURVE_BUDGETS]], on=KEYS, validate="one_to_one")
        lce = joined[[*KEYS, "panel", "family", "event", "fold"]].copy()
        lce["scheme"], lce["method"], lce["fit_seconds"] = scheme, "lce_raw", 0.0
        lce[P] = lce_law(joined)
        result = pd.concat([result, lce], ignore_index=True)
    result.to_csv(target, index=False)
    panel_path.unlink()
    write_json(output / "oof_manifest.json", {"written_at_utc": now(), "rows": len(result), "tasks": len(tasks),
                                              "seconds": time.perf_counter() - tick, "sha256": sha256(target),
                                              "fields": {k: list(v) for k, v in methods.items()}})


# --------------------------------------------------------------------------- #
# Phase: select
# --------------------------------------------------------------------------- #

def scores(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ibs = ((np.cumsum(p, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)
    ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    return ibs, ll


def mdp_mean(frame: pd.DataFrame, values: np.ndarray) -> float:
    return float(pd.Series(values, index=frame.index).groupby(frame.mdp_id.to_numpy()).mean().mean())


def soften(p: np.ndarray, prior: np.ndarray, T: float, eps: float) -> np.ndarray:
    q = np.power(np.clip(p, 1e-300, 1), 1.0 / T)
    q /= q.sum(1, keepdims=True)
    return (1 - eps) * q + eps * prior


def class_prior(panel_rows: pd.DataFrame) -> dict[str, np.ndarray]:
    return {learner: np.bincount(g.event.to_numpy(int), minlength=7) / len(g)
            for learner, g in panel_rows.groupby("algorithm")}


def two_stage(oof: pd.DataFrame, conditional: str) -> pd.DataFrame:
    base = oof[oof.method == "oeh_final"].set_index(KEYS)
    cond = oof[oof.method == conditional].set_index(KEYS).loc[base.index]
    p6 = base["p_6"].to_numpy()
    p = cond[P[:6]].to_numpy() * (1 - p6)[:, None]
    out = base.copy()
    out[P] = np.column_stack([p, p6])
    return out.reset_index()


def run_select(output: Path) -> None:
    oof = pd.read_csv(output / "oof_predictions.csv.gz")
    rows, report = [], {}
    primary = oof[oof.panel == "primary"]

    def summarize(frame: pd.DataFrame, name: str, scheme: str) -> dict:
        ibs, ll = scores(frame[P].to_numpy(float), frame.event.to_numpy(int))
        item = {"scheme": scheme, "method": name, "rows": len(frame), "ibs": mdp_mean(frame, ibs), "ll": mdp_mean(frame, ll)}
        for fam, g in frame.groupby("family"):
            gi, gl = scores(g[P].to_numpy(float), g.event.to_numpy(int))
            item[f"ibs_{fam}"], item[f"ll_{fam}"] = mdp_mean(g, gi), mdp_mean(g, gl)
        return item

    for scheme in ("mdp5", "lofo"):
        part = primary[primary.scheme == scheme]
        for method, frame in part.groupby("method"):
            rows.append(summarize(frame, method, scheme))
    # OEH-S on lofo
    lofo = primary[(primary.scheme == "lofo")]
    base = lofo[lofo.method == "oeh_final"].sort_values(KEYS).reset_index(drop=True)
    train_rows = primary[(primary.scheme == "mdp5") & (primary.method == "oeh_final")]
    prior = class_prior(train_rows)
    prior_rows = np.stack([prior[a] for a in base.algorithm])
    grid = []
    for T in RULES["oeh_s"]["grid_T"]:
        for eps in RULES["oeh_s"]["grid_eps"]:
            p = soften(base[P].to_numpy(float), prior_rows, T, eps)
            ibs, ll = scores(p, base.event.to_numpy(int))
            grid.append({"T": T, "eps": eps, "lofo_ll": mdp_mean(base, ll), "lofo_ibs": mdp_mean(base, ibs)})
    grid = pd.DataFrame(grid)
    best = grid.loc[grid.lofo_ll.idxmin()]
    report["oeh_s"] = {"grid": grid.to_dict("records"), "chosen": {"T": float(best["T"]), "eps": float(best.eps)},
                       "enters_cohort8": bool((best["T"], best.eps) != (1.0, 0.0)),
                       "class_prior": {k: v.tolist() for k, v in prior.items()}}
    # OEH-S on mdp5 for the record
    mdp5 = primary[(primary.scheme == "mdp5") & (primary.method == "oeh_final")].sort_values(KEYS).reset_index(drop=True)
    p = soften(mdp5[P].to_numpy(float), np.stack([prior[a] for a in mdp5.algorithm]), best["T"], best.eps)
    ibs, ll = scores(p, mdp5.event.to_numpy(int))
    report["oeh_s"]["mdp5_with_chosen"] = {"ll": mdp_mean(mdp5, ll), "ibs": mdp_mean(mdp5, ibs)}
    # OEH-2T on mdp5
    mdp5_all = primary[primary.scheme == "mdp5"]
    reference = summarize(mdp5_all[mdp5_all.method == "oeh_final"], "oeh_final", "mdp5")
    options = []
    for cond in ("cond_prior", "cond_full"):
        tt = two_stage(mdp5_all, cond)
        options.append({**summarize(tt, f"two_stage_{cond}", "mdp5"), "w": 1.0, "conditional": cond})
        base_p = mdp5_all[mdp5_all.method == "oeh_final"].set_index(KEYS).loc[tt.set_index(KEYS).index][P].to_numpy()
        for w in (0.25, 0.5, 0.75):
            blend = tt.copy()
            blend[P] = w * tt[P].to_numpy() + (1 - w) * base_p
            options.append({**summarize(blend, f"blend_{cond}_{w}", "mdp5"), "w": w, "conditional": cond})
    options = pd.DataFrame(options)
    eligible = options[(options.ll < reference["ll"]) & (options.ibs <= reference["ibs"])]
    chosen = eligible.loc[eligible.ll.idxmin()].to_dict() if len(eligible) else None
    lofo_check = None
    if chosen is not None:
        lofo_all = primary[primary.scheme == "lofo"]
        tt = two_stage(lofo_all, chosen["conditional"])
        base_p = lofo_all[lofo_all.method == "oeh_final"].set_index(KEYS).loc[tt.set_index(KEYS).index][P].to_numpy()
        tt[P] = chosen["w"] * tt[P].to_numpy() + (1 - chosen["w"]) * base_p
        lofo_check = summarize(tt, chosen["method"], "lofo")
    report["oeh_2t"] = {"reference_mdp5": reference, "options": options.to_dict("records"),
                        "chosen": chosen, "enters_cohort8": chosen is not None, "lofo_of_chosen": lofo_check}
    table = pd.DataFrame(rows)
    table.to_csv(output / "oof_summary.csv", index=False)
    report["a5_lofo"] = table[table.scheme == "lofo"].set_index("method")[
        [c for c in table.columns if c.startswith(("ibs", "ll"))]].to_dict("index")
    write_json(output / "selection.json", {"selected_at_utc": now(), "status": "frozen", **report,
                                           "oof_sha256": sha256(output / "oof_predictions.csv.gz"),
                                           "rules_sha256": sha256(output / "rules.json")})
    pd.set_option("display.width", 220)
    print(table.round(4).to_string(index=False))
    print(grid.round(4).to_string(index=False))
    print(options.round(4)[["method", "ibs", "ll"]].to_string(index=False), "\nreference", reference)
    print("chosen OEH-S", report["oeh_s"]["chosen"], "| 2T", chosen and chosen["method"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("oof", "select"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    run_oof(args.output, args.workers) if args.phase == "oof" else run_select(args.output)


if __name__ == "__main__":
    main()

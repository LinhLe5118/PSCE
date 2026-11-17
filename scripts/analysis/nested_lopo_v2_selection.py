"""Nested leave-one-panel-out estimate of the whole v2 selection procedure (task card T5; post hoc, descriptive).

The recorded procedure (``develop_oeh_v2_heads_20261014.run_select``, rules in ``heads/rules.json``) is
re-implemented as one function of the panels it may look at:

1. M6: adopt the simple head with the best external score if its internal relative IBS and LL are both
   <= +1% and its external score <= +1%; otherwise keep the ensemble;
2. M1 x M4 grid: two q_oeh options x two q_robust options x lambda_max x eps_max x d0;
3. feasibility: pooled internal relative IBS and LL <= +2%; objective: mean over external panels of
   (rel IBS + rel LL) / 2; parsimony: within 0.002 of the best, smallest (lambda_max, eps_max), then d0 rank.

With all five panels it must reproduce ``heads/selection.json``. The outer loop holds out each of the five
selection panels in turn, re-runs the whole procedure on the other four, and scores the chosen configuration
on the held-out panel. M2 and M3 were rejected by their own rules and are not re-run.

Usage: ``python -m scripts.analysis.nested_lopo_v2_selection``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import develop_oeh_v2_heads_20261014 as H

ROOT = H.ROOT
HEADS = H.OUTPUT
OUTPUT = ROOT / "artifacts/week8/nested_lopo_v2_selection_2026-10-15"
INTERNAL, EXTERNAL = H.INTERNAL, H.EXTERNAL
D0_RANK = {"const": 0, "4.0": 1, "3.0": 2, "2.0": 3, "1.5": 4}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    pred = pd.read_csv(HEADS / "predictions.csv.gz")
    frame = pred[pred.method == "oeh_ensemble"][H.KEYS + ["source_panel"]].merge(H.labels(), on=H.KEYS + ["source_panel"], validate="one_to_one")
    frame = frame.merge(pd.read_csv(HEADS / "distances.csv"), on=H.KEYS + ["source_panel"], validate="one_to_one").reset_index(drop=True)
    index = pd.MultiIndex.from_frame(frame[H.KEYS])
    laws = {m: g.set_index(H.KEYS).loc[index][H.P].to_numpy(float) for m, g in pred.groupby("method")}
    prior = json.loads((HEADS / "fit_manifest.json").read_text(encoding="utf-8"))["class_prior"]
    pi = np.stack([np.asarray(prior[a]) for a in frame.algorithm])
    return frame, laws, pi, frame.distance.to_numpy()


def scores(law: np.ndarray, frame: pd.DataFrame, internal) -> dict[str, tuple[float, float]]:
    y = frame.event.to_numpy(int)
    ibs = ((np.cumsum(law, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)
    ll = -np.log(np.clip(law[np.arange(len(y)), y], 1e-12, 1))
    per = pd.DataFrame({"panel": frame.source_panel.to_numpy(), "mdp": frame.mdp_id.to_numpy(), "ibs": ibs, "ll": ll}
                       ).groupby(["panel", "mdp"])[["ibs", "ll"]].mean().reset_index()
    out = {k: (g.ibs.mean(), g.ll.mean()) for k, g in per.groupby("panel")}
    pooled = per[per.panel.isin(internal)]
    out["internal"] = (pooled.ibs.mean(), pooled.ll.mean())
    return out


def relative(sc: dict, ref: dict) -> dict:
    return {k: (sc[k][0] / ref[k][0] - 1, sc[k][1] / ref[k][1] - 1) for k in sc}


def candidate_laws(laws, base, pi, d):
    """Every grid configuration's law for a given M6 base head."""
    cond, legacy, cond_legacy = H.CONDITIONAL.get(base, H.CONDITIONAL["oeh_ensemble"])
    p6 = laws[base][:, 6:7]

    def finite(law):
        return law[:, :6] / np.clip(law[:, :6].sum(1, keepdims=True), 1e-300, None)

    q_oeh = {"base_finite": finite(laws[base]), "finite_only_head": finite(laws[cond])}
    q_rob = {"legacy_finite": finite(laws[legacy]), "legacy_finite_only_head": finite(laws[cond_legacy])}
    rules = json.loads((HEADS / "rules.json").read_text(encoding="utf-8"))["rules"]
    for qo, qr, lam, eps, d0 in product(q_oeh, q_rob, rules["m1"]["lambda_max"], rules["m4"]["eps_max"], H.D0):
        lw = (np.full_like(d, lam) if d0 == "const" else lam * np.minimum(1.0, d / float(d0)))[:, None]
        ew = (np.full_like(d, eps) if d0 == "const" else eps * np.minimum(1.0, d / float(d0)))[:, None]
        q = (1 - lw) * q_oeh[qo] + lw * q_rob[qr]
        law = (1 - ew) * np.column_stack([q * (1 - p6), p6]) + ew * pi
        yield {"q_oeh": qo, "q_robust": qr, "lambda_max": lam, "eps_max": eps, "d0": str(d0)}, law


def select(frame, laws, pi, d, internal, external) -> dict:
    """The recorded M6 + grid procedure, restricted to the given panels."""
    ref = scores(laws["oeh_ensemble"], frame, internal)
    m6 = {}
    for head in ("oeh_multiclass", "oeh_aft", "oeh_thresholds", "oeh_xgboost_cf"):
        r = relative(scores(laws[head], frame, internal), ref)
        m6[head] = (r["internal"], float(np.mean([(r[p][0] + r[p][1]) / 2 for p in external])))
    ok = {h: v for h, v in m6.items() if v[0][0] <= 0.01 and v[0][1] <= 0.01 and v[1] <= 0.01}
    base = min(ok, key=lambda h: ok[h][1]) if ok else "oeh_ensemble"
    rows = []
    for config, law in candidate_laws(laws, base, pi, d):
        sc = scores(law, frame, internal)
        r = relative(sc, ref)
        rows.append({**config, "feasible": r["internal"][0] <= 0.02 and r["internal"][1] <= 0.02,
                     "score": float(np.mean([(r[p][0] + r[p][1]) / 2 for p in external])), "law": law})
    grid = pd.DataFrame(rows)
    g = grid[grid.feasible]
    near = g[g.score <= g.score.min() + 0.002].copy()
    near["rank"] = near.d0.map(D0_RANK)
    chosen = near.sort_values(["lambda_max", "eps_max", "rank", "score"]).iloc[0]
    return {"base": base, "config": {k: chosen[k] for k in ("q_oeh", "q_robust", "lambda_max", "eps_max", "d0")},
            "inner_score": float(chosen.score), "law": chosen.law}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, laws, pi, d = load()
    full = select(frame, laws, pi, d, INTERNAL, EXTERNAL)
    recorded = json.loads((HEADS / "selection.json").read_text(encoding="utf-8"))
    rec = {k: recorded["chosen"][k] for k in ("q_oeh", "q_robust", "lambda_max", "eps_max", "d0")}
    if full["config"] != rec or full["base"] != recorded["m6"]["chosen_base"]:
        raise RuntimeError(f"the re-implementation does not reproduce the recorded selection: {full['config']} vs {rec}")
    ref_all = scores(laws["oeh_ensemble"], frame, INTERNAL)
    full_rel = relative(scores(full["law"], frame, INTERNAL), ref_all)
    log, folds = [], []
    for held in (*INTERNAL, *EXTERNAL):
        internal = tuple(p for p in INTERNAL if p != held)
        external = tuple(p for p in EXTERNAL if p != held)
        inner = select(frame, laws, pi, d, internal, external)
        outer_rel = relative(scores(inner["law"], frame, INTERNAL), ref_all)[held]
        folds.append({"held_out": held, "kind": "internal" if held in INTERNAL else "external", "m6_base": inner["base"],
                      **{f"chosen_{k}": v for k, v in inner["config"].items()},
                      "inner_external_score": inner["inner_score"],
                      "outer_rel_ibs": outer_rel[0], "outer_rel_ll": outer_rel[1], "outer_score": (outer_rel[0] + outer_rel[1]) / 2,
                      "single_fit_rel_ibs": full_rel[held][0], "single_fit_rel_ll": full_rel[held][1],
                      "single_fit_score": (full_rel[held][0] + full_rel[held][1]) / 2})
        log.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "candidate": "v2 selection procedure (M6 + grid)",
                    "fold": f"outer held-out {held}", "settings": json.dumps({"base": inner["base"], **inner["config"]}),
                    "inner_score": inner["inner_score"], "outer_score": (outer_rel[0] + outer_rel[1]) / 2})
    folds = pd.DataFrame(folds)
    folds.to_csv(OUTPUT / "outer_folds.csv", index=False)
    pd.DataFrame(log).to_csv(OUTPUT / "development_log.csv", index=False)
    ext = folds[folds.kind == "external"]
    summary = {
        "reproduces_recorded_selection": True, "recorded": rec,
        "external_outer_score_mean": float(ext.outer_score.mean()),
        "external_single_fit_score_mean": float(ext.single_fit_score.mean()),
        "external_inner_score_mean": float(ext.inner_external_score.mean()),
        "optimism_inner_minus_outer": float(ext.inner_external_score.mean() - ext.outer_score.mean()),
        "optimism_single_fit_minus_outer": float(ext.single_fit_score.mean() - ext.outer_score.mean()),
        "internal_outer_rel": {r.held_out: [r.outer_rel_ibs, r.outer_rel_ll] for r in folds[folds.kind == "internal"].itertuples()},
    }
    manifest = {
        "status": "posthoc_descriptive", "task": "T5 nested LOPO of the v2 selection procedure",
        "created_utc": datetime.now(timezone.utc).isoformat(), "summary": summary,
        "note": "scores are relative to OEH current (the full head), negative = better; M2/M3 rejected by their own rules, not re-run",
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in (HEADS / "predictions.csv.gz", HEADS / "distances.csv",
                                                                         HEADS / "fit_manifest.json", HEADS / "rules.json", HEADS / "selection.json")},
        "outputs_sha256": {p.name: sha(p) for p in (OUTPUT / "outer_folds.csv", OUTPUT / "development_log.csv")},
        "source_sha256": {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__)),
                          "scripts/analysis/develop_oeh_v2_heads_20261014.py": sha(ROOT / "scripts/analysis/develop_oeh_v2_heads_20261014.py")},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(folds.drop(columns=[c for c in folds if c.startswith("chosen_q")]).round(4).to_string(index=False))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()

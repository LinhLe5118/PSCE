"""Simplified OEH candidates S1-S4 (task card T4; post hoc on opened panels; spec written before the run).

Spec and selection rule: ``docs/week8/T4_SIMPLIFIED_CANDIDATES_SPEC_2026-10-15.md``. Every head is fitted on the
720-MDP meta-training panel; the eleven opened panels are only scored. No candidate is registered (checkpoint H1).

Usage: ``python -m scripts.analysis.simplified_candidates_t4 [--workers 6]``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import develop_oeh_v2_heads_20261014 as H
from scripts.analysis import oeh_pipeline as O
from scripts.analysis import oeh_v2_forecast as V

ROOT = H.ROOT
A = ROOT / "artifacts"
OUTPUT = A / "week8/simplified_candidates_t4_2026-10-15"
SPEC = ROOT / "docs/week8/T4_SIMPLIFIED_CANDIDATES_SPEC_2026-10-15.md"
KEYS, P = H.KEYS, H.P
INTERNAL = ("cohort7", "cohort8", "internal")
EXTERNAL = ("colosseum2027", "colosseum2028", "spibb", "colosseum2029", "spibb2", "forest", "frozenlake", "gambler")
ORDER = ("S1", "S2", "S3", "S4")
GATE = {"lambda_max": 1.0, "eps_max": 0.02, "d0": 3.0}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field_sets() -> dict[str, tuple[str, ...]]:
    from scripts.analysis.colosseum_posthoc_20261014 import factorial_arms
    arms = factorial_arms()
    return {"B2": arms["b2_matched"], "B2+P": arms["b2_plus_P"], "B2+P+H": arms["oeh_without_legacy_diagnostics"],
            "B2+P+D": arms["oeh_without_oehh_risk"], "B2+D": arms["b2_plus_L"]}


def panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = H.union_query()
    frames = [dev]
    labs = [H.labels()]
    for base, names in ((A / "oeh_v2_sealed_test_2026-10-14", ("internal", "colosseum2029", "spibb2", "forest")),
                        (A / "oeh_fifth_panel_2026-10-15", ("frozenlake", "gambler"))):
        for p in names:
            frames.append(pd.read_csv(base / p / "query_preoutcome.csv.gz").assign(source_panel=p))
            labs.append(pd.read_csv(base / p / "labels.csv")[KEYS + ["event"]].assign(source_panel=p))
    query = pd.concat(frames, ignore_index=True)
    query["panel"] = "primary"
    return query.reset_index(drop=True), pd.concat(labs, ignore_index=True)


def fit_single_heads(train: pd.DataFrame, query: pd.DataFrame, workers: int) -> dict[str, np.ndarray]:
    sets = field_sets()
    work = OUTPUT / "_work"
    work.mkdir(parents=True, exist_ok=True)
    tpath, qpath = work / "_train.pkl", work / "_query.pkl"
    train.to_pickle(tpath)
    query.drop(columns=["source_panel"]).to_pickle(qpath)
    specs = [("B2+P", False), ("B2+P", True), ("B2+P+H", False), ("B2+P+D", False), ("B2+P+D", True), ("B2", False), ("B2+D", False)]
    tasks = [{"name": f"{name}{'|fin' if fin else ''}", "fields": list(sets[name]), "kind": "multiclass", "finite_only": fin,
              "train": str(tpath), "query": str(qpath)} for name, fin in specs]
    out = {}
    try:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            for name, p, _ in pool.map(H._fit, tasks):
                out[name] = p
    finally:
        tpath.unlink(missing_ok=True)
        qpath.unlink(missing_ok=True)
    return out


def scores(law: np.ndarray, frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    y = frame.event.to_numpy(int)
    ibs = ((np.cumsum(law, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)
    ll = -np.log(np.clip(law[np.arange(len(y)), y], 1e-12, 1))
    per = pd.DataFrame({"panel": frame.source_panel.to_numpy(), "mdp": frame.mdp_id.to_numpy(), "ibs": ibs, "ll": ll}
                       ).groupby(["panel", "mdp"])[["ibs", "ll"]].mean().reset_index()
    return per


def rule(table: dict[str, pd.DataFrame], internal, external) -> list[str]:
    """Selection rule of the spec, section 3, on the given panels."""
    ref = table["OEH"]
    ok, ext_score = [], {}
    for c in ORDER:
        cand = table[c]
        ri = cand[cand.panel.isin(internal)][["ibs", "ll"]].mean() / ref[ref.panel.isin(internal)][["ibs", "ll"]].mean() - 1
        rel = []
        for k in external:
            a, b = cand[cand.panel == k][["ibs", "ll"]].mean(), ref[ref.panel == k][["ibs", "ll"]].mean()
            rel.append(((a.ibs / b.ibs - 1) + (a.ll / b.ll - 1)) / 2)
        ext_score[c] = float(np.mean(rel))
        if ri.ibs <= 0.01 and ri.ll <= 0.01 and ext_score[c] <= 0.01:
            ok.append(c)
    if not ok:
        return []
    picks = [ok[0]]
    best = min(ok, key=lambda c: ext_score[c])
    if best != ok[0]:
        picks.append(best)
    return picks


def main() -> None:
    from scripts.analysis.develop_cohort8_candidates_20261014 import meta_panel

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6)
    workers = parser.parse_args().workers
    OUTPUT.mkdir(parents=True, exist_ok=True)
    query, labels = panels()
    frame = query[KEYS + ["source_panel"]].merge(labels, on=KEYS + ["source_panel"], validate="one_to_one")
    if len(frame) != len(query):
        raise RuntimeError("labels missing for some query rows")
    train = meta_panel(sort=False)
    spec = O.load_spec()
    ens = O.fit_heads(train, query.drop(columns=["source_panel"]), workers, OUTPUT / "_work")
    oeh, d = O.final_law(ens, train, query, spec)
    single = fit_single_heads(train, query, workers)
    pi = np.stack([V.class_prior(train)[a] for a in query.algorithm])
    laws = {
        "OEH": oeh, "OEH-F": ens["full"],
        "S1": single["B2+P"], "S2": single["B2+P+H"],
        "S3": V.v2_law(single["B2+P"], single["B2+P|fin"], single["B2"], d, pi, **GATE),
        "S4": V.v2_law(single["B2+P+D"], single["B2+P+D|fin"], single["B2+D"], d, pi, **GATE),
    }
    for name, law in laws.items():
        if not np.isfinite(law).all() or not np.allclose(law.sum(1), 1, atol=1e-9):
            raise RuntimeError(f"invalid law {name}")
    table = {name: scores(law, frame) for name, law in laws.items()}
    # reproduction of the sealed OEH forecasts on the final panels (the fits do not depend on the query)
    sealed = []
    for p in ("internal", "colosseum2029", "spibb2", "forest"):
        s = pd.read_csv(A / "oeh_v2_sealed_test_2026-10-14" / p / "predictions_preoutcome.csv")
        s = s[s.method == "v2"][KEYS + P]
        idx = (query.source_panel == p).to_numpy()
        mine = query.loc[idx, KEYS].assign(**{c: oeh[idx][:, i] for i, c in enumerate(P)})
        j = mine.merge(s, on=KEYS, suffixes=("", "_s"))
        sealed.append(float(np.abs(j[P].to_numpy() - j[[f"{c}_s" for c in P]].to_numpy()).max()))
    rows, log = [], []
    now = datetime.now(timezone.utc).isoformat()
    for name, per in table.items():
        for panel, g in per.groupby("panel"):
            rows.append({"candidate": name, "panel": panel, "mdps": len(g), "ibs": g.ibs.mean(), "ll": g.ll.mean()})
            log.append({"timestamp_utc": now, "candidate": name, "settings": json.dumps(GATE if name in ("S3", "S4") else {}),
                        "panel": panel, "ibs": g.ibs.mean(), "ll": g.ll.mean()})
    summary = pd.DataFrame(rows)
    ref = summary[summary.candidate == "OEH"].set_index("panel")
    summary["rel_ibs"] = summary.apply(lambda r: r.ibs / ref.loc[r.panel, "ibs"] - 1, axis=1)
    summary["rel_ll"] = summary.apply(lambda r: r.ll / ref.loc[r.panel, "ll"] - 1, axis=1)
    summary.to_csv(OUTPUT / "scores_by_panel.csv", index=False)
    pd.DataFrame(log).to_csv(OUTPUT / "development_log.csv", index=False)
    recommendation = rule(table, INTERNAL, EXTERNAL)
    outer = []
    for held in (*INTERNAL, *EXTERNAL):
        pick = rule(table, tuple(p for p in INTERNAL if p != held), tuple(p for p in EXTERNAL if p != held))
        chosen = pick[0] if pick else "OEH"
        r = summary[(summary.candidate == chosen) & (summary.panel == held)].iloc[0]
        outer.append({"held_out": held, "selected": chosen, "held_rel_score": (r.rel_ibs + r.rel_ll) / 2})
    pd.DataFrame(outer).to_csv(OUTPUT / "nested_lopo.csv", index=False)
    result = {"recommendation": recommendation or ["keep OEH (no acceptable candidate)"],
              "sealed_oeh_reproduction_max_abs_diff": max(sealed),
              "pooled": {c: {"internal_rel_ibs": float(table[c][table[c].panel.isin(INTERNAL)].ibs.mean()
                                                       / table["OEH"][table["OEH"].panel.isin(INTERNAL)].ibs.mean() - 1),
                             "internal_rel_ll": float(table[c][table[c].panel.isin(INTERNAL)].ll.mean()
                                                      / table["OEH"][table["OEH"].panel.isin(INTERNAL)].ll.mean() - 1),
                             "external_score": float(summary[(summary.candidate == c) & summary.panel.isin(EXTERNAL)]
                                                     .eval("(rel_ibs + rel_ll) / 2").mean())} for c in ("OEH-F", *ORDER)}}
    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest = {"status": "posthoc_descriptive", "task": "T4 simplified candidates (not registered; stop at H1)",
                "created_utc": now, "spec": {"path": SPEC.relative_to(ROOT).as_posix(), "sha256": sha(SPEC)},
                "outputs_sha256": {p.name: sha(p) for p in (OUTPUT / "scores_by_panel.csv", OUTPUT / "development_log.csv",
                                                           OUTPUT / "nested_lopo.csv", OUTPUT / "result.json")},
                "source_sha256": {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__))}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(json.dumps(result, indent=1))
    print(summary.pivot(index="panel", columns="candidate", values="rel_ll").round(3).to_string())
    print(pd.DataFrame(outer).to_string(index=False))


if __name__ == "__main__":
    main()

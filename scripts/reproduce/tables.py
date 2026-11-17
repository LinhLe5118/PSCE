"""Step 4: Tables 11-18 of the thesis, computed from the rescored final tests, and a cell-by-cell check.

Inputs: the scored files written by step 3 (regenerated OEH, full head, robust head and B2; the other comparators
as sealed), the TabPFN forecasts kept as artifacts, and the forecasts without one block from step 2.
Every printed number of ``thesis_tables.json`` (Tables 11-18 of the thesis of 11/11/2026) must be matched to its
printed precision. Table 14 applies the registered decision rule (descriptive on Tests 1 and 2); Tables 15-18 are
post hoc, as the thesis says.

The bootstrap draws, seeds and log-loss floors are those that produced the printed numbers, 10,000 whole-MDP draws
each: Table 16 seed 20261028 with floor 1e-15; Table 17 seed 20261101 with floor 1e-12, except the gate rows in the
four all-budget columns, which were printed from the draws of seed 20261028.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reproduce.common import ART, KEYS, LEARNERS, P, PARTS, TESTS, load_json, part_of, row_scores, write_json

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "thesis_tables.json"
DRAWS = 10_000
BUDGETS = np.array([1, 2, 5, 10, 20, 50])
NAMES = {  # thesis row name -> method code in the scored files (Tests 2 and 3; Test 1 is translated below)
    "OEH": "oeh", "Full head": "oeh_full_head", "Robust head": "robust", "B2": "b2_matched",
    "TabPFN v2 on B2": "tabpfn_v2", "B2 + LCE": "b2_plus_lce", "B2 + CB": "b2_plus_cb", "XGBoost-CF/18": "xgboost_cf_18",
    "XGBoost-CF/108": "xgboost_cf_108", "LCE": "lce_raw", "CB": "cb_only", "k-NN analog on B2": "knn_analog",
    "B2 + prefix/FQE": "b2_plus_prefix_fqe"}
TEST1 = {"v2": "oeh", "oeh_final": "oeh_full_head", "legacy": "robust"}
REMOVED = {"Learner simulation": "oeh_no_simulation", "Planning risk": "oeh_no_planning_risk",
           "Pilot diagnostics": "oeh_no_pilot_diagnostics"}


# ----------------------------------------------------------------------------------------------- data
def load_cases(out: Path) -> pd.DataFrame:
    frames = []
    for test, t in TESTS.items():
        s = pd.read_csv(out / "3_registered" / t["folder"] / "scored.csv.gz")
        if "panel" in s:
            s = s[s.panel == "primary"]
        s = s.replace({"method": TEST1}) if test == "Test 1" else s
        frames.append(s[["source_panel", *KEYS, "method", *P, "event"]].assign(test=test))
    d = pd.concat(frames, ignore_index=True)
    labels = d[d.method == "oeh"][["test", "source_panel", *KEYS, "event"]]
    tab = pd.read_csv(ART / "week8/baseline_tabpfn_t3_2026-10-15/predictions.csv.gz")
    tab = tab.merge(labels[labels.test != "Test 3"], on=["source_panel", *KEYS], validate="one_to_one")
    t3 = [pd.read_csv(ART / TESTS["Test 3"]["folder"] / panel / "tabpfn_predictions_preoutcome.csv").assign(source_panel=panel)
          for panel in TESTS["Test 3"]["panels"]]
    t3 = pd.concat(t3).merge(labels[labels.test == "Test 3"], on=["source_panel", *KEYS], validate="one_to_one")
    parts = [d, tab.assign(method="tabpfn_v2"), t3.assign(method="tabpfn_v2")]
    if (out / "2_forecasts.csv.gz").exists():                    # Table 16 needs the forecasts without one block
        removed = pd.read_csv(out / "2_forecasts.csv.gz")
        removed = removed[removed.method.str.startswith("oeh_no_")]
        parts.append(removed.merge(labels, on=["test", "source_panel", *KEYS], validate="many_to_one")[d.columns])
    d = pd.concat(parts, ignore_index=True)
    d["part"] = [part_of(t, p) for t, p in zip(d.test, d.source_panel)]
    y = d.event.to_numpy(int)
    for k, v in row_scores(d[P].to_numpy(float), y).items():
        d[k] = v
    d["ll_15"] = -np.log(np.clip(d[P].to_numpy(float)[np.arange(len(y)), y], 1e-15, 1))   # floor of Table 16
    return d


def mdp_scores(frame: pd.DataFrame, method: str, col: str) -> pd.Series:
    return frame[frame.method == method].groupby("mdp_id")[col].mean()


def score(frame: pd.DataFrame, method: str, col: str) -> float:
    return float(mdp_scores(frame, method, col).mean())


# ----------------------------------------------------------------------------------------------- tables
def scores_table(d: pd.DataFrame, rows, columns) -> list:
    out = []
    for name in rows:
        cells = [[score(f, NAMES[name], a), score(f, NAMES[name], b)] for f, (a, b) in columns]
        out.append([name, *cells])
    return out


def table_11_13(d):
    src = lambda p: (d[d.source_panel == p], ("ibs", "ll"))
    part = lambda p, cols=("ibs", "ll"): (d[d.part == p], cols)
    t11 = scores_table(d, ["OEH", "Robust head", "TabPFN v2 on B2", "B2 + LCE", "B2 + CB", "B2", "XGBoost-CF/18", "LCE", "CB"],
                       [part("Test 1 internal"), src("colosseum2029"), src("spibb2"), src("forest"), part("Test 1 external")])
    t12 = scores_table(d, ["OEH", "Robust head", "TabPFN v2 on B2", "B2 + LCE", "B2 + CB", "B2", "XGBoost-CF/18", "LCE",
                           "k-NN analog on B2"], [src("frozenlake"), src("gambler"), part("Test 2")])
    t13 = scores_table(d, ["OEH", "Robust head", "B2 + prefix/FQE", "B2 + LCE", "B2 + CB", "B2", "TabPFN v2 on B2", "XGBoost-CF/18",
                           "XGBoost-CF/108", "LCE", "k-NN analog on B2"],
                       [src("mdpp"), src("labmaze"), part("Test 3"), part("Test 3", ("ibs_ge", "ll_ext"))])
    return t11, t12, t13


def table_14(d):
    from scripts.analysis.colosseum_posthoc_20261014 import decision_costs

    groups = [d.source_panel == "internal", d.source_panel == "colosseum2029", d.source_panel == "spibb2",
              d.source_panel == "forest", d.part == "Test 2", d.part == "Test 3"]
    rows = []
    for name in ("OEH", "Robust head", "B2"):
        cells = []
        for mask in groups:
            f = d[mask & (d.method == NAMES[name])]
            parts = pd.DataFrame(decision_costs(np.cumsum(f[P].to_numpy(float), 1)[:, :6], f.event.to_numpy(int), 0.8, 100))
            cells.append([float(parts.regret.groupby(f.mdp_id.to_numpy()).mean().mean())])
        rows.append([name, *cells])
    return rows


def _rule(x: pd.DataFrame, kind: str) -> list:
    """Budget rule on one forecast, pilot counted: [transitions in multiples of |S||A|, % of reachable targets met, net cost]."""
    E = x.event.to_numpy(int)
    finite = E <= 5
    if kind == "registered":                       # smallest budget with forecast success >= 0.8, else skip
        F = np.cumsum(x[P].to_numpy(float), 1)[:, :6]
        ok = F >= .8
        select = ok.any(1)
        j = np.where(select, ok.argmax(1), -1)
        D = np.where(select, np.maximum(10, BUDGETS[np.maximum(j, 0)]), 10).astype(float)
        met, fail = select & (E <= j), select & (E > j)
    elif kind == "cautious":                       # collect x50 unless the forecast is at least 90% sure of >x50
        collect = x.p_6.to_numpy() < .9
        D = np.where(collect, 50., 10.)
        met, fail = collect & finite, collect & ~finite
    elif kind == "always x50":                     # no pilot
        D, met, fail = np.full(len(E), 50.), finite, ~finite
    elif kind == "always x10":                     # train on the pilot
        D, met, fail = np.full(len(E), 10.), E <= 3, E > 3
    elif kind == "always skip":
        D, met, fail = np.zeros(len(E)), np.zeros(len(E), bool), np.zeros(len(E), bool)
    miss = finite & ~met & ~fail
    m = pd.DataFrame({"D": D, "met": met, "finite": finite, "bad": fail | miss}, dtype=float).groupby(x.mdp_id.to_numpy()).mean()
    return [float(m.D.mean()), float(100 * m.met.mean() / m.finite.mean()), float((m.D + 100 * m.bad).mean())]


def table_15(d):
    rows = []
    for label, name, kind in (("OEH, registered", "OEH", "registered"), ("OEH, cautious", "OEH", "cautious"),
                              ("B2, registered", "B2", "registered"), ("B2, cautious", "B2", "cautious"),
                              ("Always x50", "OEH", "always x50"), ("Always x10", "OEH", "always x10"),
                              ("Always skip", "OEH", "always skip")):
        rows.append([label, *[_rule(d[(d.part == p) & (d.method == NAMES[name])], kind) for p in PARTS]])
    return rows


def table_16(d):
    rng = np.random.default_rng(20261028)
    rows = []
    for p in PARTS:
        f = d[d.part == p]
        per = {m: f[f.method == m].groupby("mdp_id")[["ibs", "ll_15"]].mean() for m in ("oeh", *REMOVED.values())}
        mdps = per["oeh"].index
        index = rng.integers(0, len(mdps), (DRAWS, len(mdps)))
        first = True
        for block, m in REMOVED.items():
            cells = []
            for col in ("ibs", "ll_15"):
                diff = per[m][col].reindex(mdps).to_numpy() - per["oeh"][col].to_numpy()
                lo, hi = np.percentile(diff[index].mean(1), [2.5, 97.5])
                cells.append([float(diff.mean()), float(lo), float(hi)])
            oeh = [float(per["oeh"].ibs.mean()), float(per["oeh"].ll_15.mean())] if first else None
            rows.append([p if first else "", oeh, block, *cells])
            first = False
    return rows


def table_17(d):
    """Ladder B2 -> robust head -> full head -> OEH. The intervals of the first two steps and of the last column come
    from the draws of 01/11 (seed 20261101, one index per part); the gate rows in the four all-budget columns were
    printed from the run of 28/10 (seed 20261028, one index per part in the same order), so they use those draws."""
    keep = d[d.method.isin(["b2_matched", "robust", "oeh_full_head", "oeh"])]
    frames = {p: keep[keep.part == p] for p in PARTS}
    draws = {seed: (lambda rng: {p: rng.integers(0, f.mdp_id.nunique(), (DRAWS, f.mdp_id.nunique())) for p, f in frames.items()})(
        np.random.default_rng(seed)) for seed in (20261101, 20261028)}
    columns = [(p, "ibs", "ll") for p in PARTS] + [("Test 3", "ibs_ge", "ll_ext")]
    rows = [[name, *[[score(frames[p], NAMES[name], a), score(frames[p], NAMES[name], b)] for p, a, b in columns]]
            for name in ("B2", "Robust head", "Full head", "OEH")]
    steps = (("Pilot diagnostics", "b2_matched", "robust"), ("Simulation and planning risk", "robust", "oeh_full_head"),
             ("Gate and the other two heads", "oeh_full_head", "oeh"))
    for label, before, after in steps:
        for metric in ("IBS", "LL"):
            cells = []
            for k, (p, a, b) in enumerate(columns):
                col = a if metric == "IBS" else b
                diff = (mdp_scores(frames[p], before, col) - mdp_scores(frames[p], after, col)).to_numpy()
                seed = 20261028 if label.startswith("Gate") and k < 4 else 20261101
                boots = diff[draws[seed][p]].mean(1)
                lo, hi = (np.percentile(boots, [2.5, 97.5]) if seed == 20261028 else np.quantile(boots, [.025, .975]))
                cells.append([float(diff.mean()), float(lo), float(hi)])
            rows.append([f"{label}, {metric}", *cells])
    return rows


def _order(frame: pd.DataFrame, method: str) -> dict:
    """Pairs of learners of one pilot ordered by the expected class of the forecast, and the best-learner pick."""
    x = frame[frame.method == method].copy()
    x["exp"] = (x[P].to_numpy() * np.arange(7)).sum(1)
    conc = pairs = picked = differ = fixed_ok = 0.0
    fixed = frame[frame.method == "oeh"].groupby("algorithm").event.mean().idxmin()
    for _, h in x.groupby(["mdp_id", "policy_family"]):
        ev, ex = h.event.to_numpy(), h.exp.to_numpy()
        for i in range(len(ev)):
            for j in range(i + 1, len(ev)):
                if ev[i] != ev[j]:
                    pairs += 1
                    s = (ex[i] - ex[j]) * (ev[i] - ev[j])
                    conc += 1 if s > 0 else (.5 if s == 0 else 0)
        if len(set(ev.tolist())) > 1:
            differ += 1
            picked += ev[np.argmin(ex)] == ev.min()
            fixed_ok += ev[h.algorithm.to_numpy() == fixed][0] == ev.min()
    return {"pairs": conc / pairs, "picked": picked / differ, "fixed": LEARNERS[fixed], "fixed_picked": fixed_ok / differ}


def table_18(d):
    frames = {p: d[(d.part == p) & d.method.isin(["oeh", "b2_matched"])] for p in PARTS}
    rows = []
    for col, metric in (("ibs", "IBS"), ("ll", "LL")):
        for alg, short in LEARNERS.items():
            rows.append([f"{short}, {metric}", *[[score(f[f.algorithm == alg], "oeh", col), score(f[f.algorithm == alg], "b2_matched", col)]
                                                  for f in frames.values()]])
    order = {p: (_order(f, "oeh"), _order(f, "b2_matched")) for p, f in frames.items()}
    rows.append(["Learner pairs in the right order", *[[o["pairs"], b["pairs"]] for o, b in order.values()]])
    rows.append(["Best learner picked", *[[o["picked"], b["picked"]] for o, b in order.values()]])
    rows.append(["Best fixed learner", *[[o["fixed"], o["fixed_picked"]] for o, _ in order.values()]])
    return rows


def calibration(d) -> dict:
    """Expected calibration error of the probability that no budget is enough (ten equal-width bins)."""
    res = {}
    for p in PARTS:
        for name in ("OEH", "B2"):
            x = d[(d.part == p) & (d.method == NAMES[name])]
            q, y = x.p_6.to_numpy(), (x.event.to_numpy() == 6).astype(float)
            b = np.minimum((q * 10).astype(int), 9)
            res[f"{p} / {name}"] = float(sum(np.mean(b == k) * abs(q[b == k].mean() - y[b == k].mean()) for k in range(10) if (b == k).any()))
    return res


# ----------------------------------------------------------------------------------------------- check
NUMBER = re.compile(r"[+\-−]?\d*\.?\d+%?")


def _tokens(text: str) -> list[str]:
    return NUMBER.findall(text)


def _value(token: str) -> tuple[float, int]:
    t = token.replace("−", "-").rstrip("%")
    decimals = len(t.split(".")[1]) if "." in t else 0
    return float(t), decimals


def _format(value: float, like: str) -> str:
    _, decimals = _value(like)
    s = f"{abs(value):.{decimals}f}"
    if like.lstrip("+-−").startswith(".") and s.startswith("0."):
        s = s[1:]
    sign = "−" if value < 0 and round(abs(value), decimals) > 0 else ("+" if like[0] in "+-−" else "")
    return sign + s + ("%" if like.endswith("%") else "")


def _flat(cell) -> list:
    if cell is None or isinstance(cell, str):
        return [] if cell is None else [cell]
    out = []
    for v in cell:
        out.extend(_flat(v) if isinstance(v, (list, tuple)) else [v])
    return out


def numeric_cells(expected: dict, key: str):
    """(row, column, text) of every printed cell that holds numbers; row labels and Table 16's block names are text."""
    rows = expected[key]["rows"]
    for i, row in enumerate(rows[1:]):
        for j, text in enumerate(row):
            if j > 0 and not (key == "16" and j == 2):
                yield i, j, text


def compare(computed: dict, expected: dict) -> tuple[list, dict]:
    """Compare every printed number; returns the mismatches and the computed tables formatted like the thesis."""
    bad, rendered = [], {}
    for key, rows in computed.items():
        printed = expected[key]["rows"]
        header, body = printed[0], printed[1:]
        if len(body) != len(rows):
            raise RuntimeError(f"Table {key}: {len(rows)} computed rows against {len(body)} printed")
        lines = [header]
        for got, want in zip(rows, body):
            line = []
            for j, (g, w) in enumerate(zip(got, want)):
                values = _flat(g)
                labels = [v for v in values if isinstance(v, str)]
                numbers = [v for v in values if not isinstance(v, str)]
                if j == 0 or (key == "16" and j == 2):
                    line.append(str(g))
                    if str(g) != w:
                        bad.append(f"Table {key}: row label {g!r} printed as {w!r}")
                    continue
                tokens = _tokens(w)
                if len(tokens) != len(numbers):
                    bad.append(f"Table {key}, {want[0] or got[0]} / {header[j]}: printed {w!r}, computed {numbers}")
                    line.append(" / ".join(map(str, numbers)))
                    continue
                for tok, v in zip(tokens, numbers):
                    p, decimals = _value(tok)
                    if abs(v - p) > 0.5 * 10 ** -decimals + 1e-9:
                        bad.append(f"Table {key}, {want[0] or got[0]} / {header[j]}: printed {tok}, computed {v:.6f}")
                formatted = iter([_format(v, tok) for tok, v in zip(tokens, numbers)])
                text = NUMBER.sub(lambda m: next(formatted), w)
                for lab in labels:
                    if lab not in w:
                        bad.append(f"Table {key}, {want[0]} / {header[j]}: printed {w!r}, computed {lab}")
                line.append(text)
            lines.append(line)
        rendered[key] = lines
    return bad, rendered


LABELS = {"v2": "OEH", "oeh": "OEH", "legacy": "robust head", "robust": "robust head", "oeh_final": "full head",
          "b2_matched": "B2", "xgboost_cf_18": "XGBoost-CF/18", "lce_raw": "LCE", "cb_only": "CB", "b2_plus_lce": "B2 + LCE",
          "b2_plus_cb": "B2 + CB", "b2_plus_prefix_fqe": "B2 + prefix/FQE"}
METRICS = {"brier": "IBS", "log_loss": "LL", "ibs_ge": "IBS≥", "ll_ext": "LLext", "regret": "regret"}


def registered_rows(results: dict) -> list[tuple]:
    """(test, family, comparison, score, mean improvement, lower bound, verdict) from the result.json of each test."""
    rows = []
    for name, step in results["Test 1"]["steps"].items():
        for c in step["contrasts"]:
            rows.append(("Test 1", name.replace("step", "step "), f"OEH vs {LABELS[c['baseline']]}, {c['group'].replace('_', ' ')}",
                         METRICS[c["metric"]], c["mean_improvement"], c["simultaneous_lower_95"], step["verdict"]))
    gate = results["Test 2"]["gate"]
    rows += [("Test 2", "single family", f"OEH vs {LABELS[c['baseline']]}", METRICS[c["metric"]], c["mean_improvement"],
              c["simultaneous_lower_95"], gate["verdict"]) for c in gate["contrasts"]]
    t3 = results["Test 3"]
    for name, g in [*t3["gates"].items(), *t3["registered_estimates"].items()]:
        rows += [("Test 3", name, f"OEH vs {LABELS[c['comparator']]}", METRICS[c["metric"]], c["mean_improvement"],
                  c["simultaneous_lower_95"], g.get("verdict", "estimate, no pass rule")) for c in g["contrasts"]]
    return rows


def registered_table(results: dict) -> list[str]:
    lines = ["| Test | Family | Comparison | Score | Mean improvement | Lower bound | Verdict |", "|---|---|---|---|---|---|---|"]
    return lines + [f"| {t} | {f} | {c} | {m} | {mean:+.4f} | {low:+.4f} | {v} |" for t, f, c, m, mean, low, v in registered_rows(results)]


def registered_lines(out: Path) -> list[str]:
    """The registered comparisons of Table 10 as rerun in step 3 (positive = OEH better)."""
    results = {test: r["result"] for test, r in load_json(out / "3_registered_check.json")["tests"].items()}
    return ["## Registered comparisons (Table 10), rerun in step 3", "",
            "Mean improvement is the score of the comparator minus the score of OEH, averaged over MDPs; the bound is the "
            "simultaneous one-sided 95% lower bound of the registered rule.", "", *registered_table(results), ""]


def run(out: Path, log) -> dict:
    d = load_cases(out)
    t11, t12, t13 = table_11_13(d)
    computed = {"11": t11, "12": t12, "13": t13, "14": table_14(d), "15": table_15(d), "16": table_16(d),
                "17": table_17(d), "18": table_18(d)}
    doc = load_json(EXPECTED)
    expected = doc["tables"]
    differences, rendered = compare(computed, expected)
    known = {f"Table {k['table']}, {k['row']} / {k['column']}: printed {k['printed']}": k["note"] for k in doc["known_differences"]}
    documented = [b for b in differences if any(b.startswith(prefix) for prefix in known)]
    bad = [b for b in differences if b not in documented]
    cells = sum(len(_tokens(text)) for key in computed for _, _, text in numeric_cells(expected, key))
    md = ["# Tables 11-18 of the thesis, recomputed", "",
          "Computed by `python -m scripts.reproduce`; numbers are formatted like the printed tables. "
          f"{cells} printed numbers were compared: {cells - len(differences)} match, {len(documented)} differ as "
          f"documented below, {len(bad)} differ without explanation.", ""]
    if (out / "3_registered_check.json").exists():
        md += registered_lines(out)
    for key, lines in rendered.items():
        md += [f"## {expected[key]['caption']}", "", "| " + " | ".join(lines[0]) + " |", "|" + "---|" * len(lines[0])]
        md += ["| " + " | ".join(line) + " |" for line in lines[1:]] + [""]
    ece = calibration(d)
    md += ["## Expected calibration error of the forecast that no budget is enough (Figure 8)", "",
           "| Part / forecast | ECE |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in ece.items()] + [""]
    if documented:
        md += ["## Documented differences", ""] + [f"- {b}. {known[next(p for p in known if b.startswith(p))]}" for b in documented] + [""]
    if bad:
        md += ["## Unexplained differences", ""] + [f"- {b}" for b in bad] + [""]
    (out / "4_tables.md").write_text("\n".join(md), encoding="utf-8")
    write_json(out / "4_tables.json", {"printed_numbers": cells, "documented_differences": documented, "mismatches": bad,
                                       "tables": computed, "ece": ece})
    log(f"  {cells} printed numbers of Tables 11-18 compared: {cells - len(differences)} match, "
        f"{len(documented)} documented differences, {len(bad)} unexplained")
    for b in documented:
        log("    documented: " + b)
    for b in bad:
        log("    UNEXPLAINED: " + b)
    return {"printed_numbers": cells, "documented": documented, "mismatches": bad}

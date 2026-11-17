"""Tables of the OEH v2 sealed test (markdown for the record, LaTeX for the paper), from its artifacts only."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/oeh_v2_sealed_test_2026-10-14"
PANELS = ("internal", "colosseum2029", "spibb2", "forest")
NAMES = {"internal": "Internal", "colosseum2029": "Colosseum 2029", "spibb2": "SPIBB-2", "forest": "Forest",
         "pooled_external": "Pooled external"}
METHODS = {"v2": "OEH v2", "oeh_final": "OEH", "legacy": "B2 + legacy", "b2_matched": "B2", "xgboost_cf_18": "XGBoost-CF/18",
           "b2_plus_lce": "B2 + LCE", "b2_plus_cb": "B2 + CB", "legacy_shift": "B2 + legacy, shrunk", "b2_shift": "B2, shrunk"}


def md_table(frame: pd.DataFrame, index: bool = True) -> str:
    """GitHub markdown without the optional tabulate dependency."""
    frame = frame.reset_index() if index else frame
    cols = [str(c) for c in frame.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(rows)


def panel_table() -> pd.DataFrame:
    s = pd.read_csv(OUT / "summary.csv")
    s = s[s.scope.isin(["panel", "pooled_external"])].copy()
    s["where"] = s.source_panel.where(s.scope == "panel", "pooled_external")
    s = s[s.method.isin(METHODS)]
    return s.pivot_table(index="method", columns="where", values=["brier", "log_loss"]).reindex(list(METHODS))


def gate_rows() -> list[dict]:
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    rows = []
    for name, step in result["steps"].items():
        for c in step["contrasts"]:
            rows.append({"step": name, "verdict": step["verdict"], "critical": step["critical"], **c})
    return rows


def markdown() -> str:
    t = panel_table()
    cols = [c for c in (*PANELS, "pooled_external") if ("brier", c) in t.columns]
    lines = ["| Method | " + " | ".join(NAMES[c] for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for m in t.index:
        lines.append(f"| {METHODS[m]} | " + " | ".join(f"{t.loc[m, ('brier', c)]:.4f} / {t.loc[m, ('log_loss', c)]:.3f}" for c in cols) + " |")
    lines += ["", "| Step | Baseline | Group | Metric | Mean improvement | Simultaneous LB | Verdict |", "|---|---|---|---|---:|---:|---|"]
    for r in gate_rows():
        lines.append(f"| {r['step']} | {METHODS.get(r['baseline'], r['baseline'])} | {r['group']} | {r['metric']} | "
                     f"{r['mean_improvement']:+.4f} | {r['simultaneous_lower_95']:+.4f} | {r['verdict']} |")
    reg = pd.read_csv(OUT / "decision_regret.csv")
    head = reg[(reg.tau == 0.8) & (reg["lambda"] == 100)].pivot(index="method", columns="panel", values="mean_regret")
    lines += ["", "Decision regret (tau 0.8, lambda 100), multiples of |S||A|:", "", md_table(head.round(2))]
    lines += ["", "Extended endpoint (rows labeled >x50 by v2's primary label):", "",
              md_table(pd.read_csv(OUT / "extended_endpoint.csv"), index=False)]
    lines += ["", "Shapley (mean per-MDP loss improvement over B2):", "", md_table(pd.read_csv(OUT / "shapley.csv").round(5), index=False)]
    return "\n".join(lines)


def latex() -> str:
    t = panel_table()
    cols = [c for c in (*PANELS, "pooled_external") if ("brier", c) in t.columns]
    keep = ["v2", "oeh_final", "legacy", "b2_matched", "xgboost_cf_18", "legacy_shift"]
    best = {(metric, c): t.loc[keep, (metric, c)].idxmin() for metric in ("brier", "log_loss") for c in cols}
    body = []
    for m in keep:
        cells = []
        for c in cols:
            for metric, fmt in (("brier", "{:.4f}"), ("log_loss", "{:.3f}")):
                v = fmt.format(t.loc[m, (metric, c)])
                cells.append(f"\\textbf{{{v}}}" if best[(metric, c)] == m else v)
        body.append(f"{METHODS[m]} & " + " & ".join(cells) + r" \\")
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{NAMES[c]}}}" for c in cols)
    sub = " & ".join(["IBS & LL"] * len(cols))
    return "\n".join([r"\begin{tabular}{l" + "rr" * len(cols) + "}", r"\toprule", f"Method & {head} \\\\", f" & {sub} \\\\",
                      r"\midrule", *body, r"\bottomrule", r"\end{tabular}"])


def main() -> None:
    md = markdown()
    (OUT / "report.md").write_text(md + "\n", encoding="utf-8")
    (OUT / "table_v2.tex").write_text(latex() + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

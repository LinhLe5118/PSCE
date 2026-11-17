"""H6 helper: compare the hand-typed result tables of the paper with the artifacts.

Reads paper/sections/06_results.tex of the Devoted repository and checks every
number of tab:v2test (Test 1), tab:fifth (Test 2) and tab:panel6 (Test 3) against
the committed summary files, after rounding the artifact value the way the paper
prints it. Read-only; nothing in the repository is written.

Usage (repository root):  python -m scripts.analysis.check_paper_tables
Test: tests/test_check_paper_tables.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main(root: Path = ROOT, verbose: bool = True, tex: str | None = None) -> bool:
    """Return True when every checked number matches its artifact (``tex`` overrides the results section)."""
    TEX = tex if tex is not None else (root / "paper/sections/06_results.tex").read_text(encoding="utf-8")
    A = root / "artifacts"

    def table_rows(label):
        """Return {row name: [numbers as printed]} of the table with this label."""
        block = TEX[TEX.index(r"\label{" + label + "}"):]
        block = block[block.index(r"\midrule"):block.index(r"\bottomrule")]
        rows = {}
        for line in block.split(r"\\"):
            line = line.replace(r"\midrule", "").strip()
            if "&" not in line:
                continue
            name, *cells = [c.strip() for c in line.split("&")]
            nums = re.findall(r"\d*\.\d+", " ".join(cells).replace(r"\textbf", ""))
            rows[re.sub(r"\\OEH\{\}", "OEH", name)] = nums
        return rows

    def same(printed, value):
        decimals = len(printed.split(".")[1])
        return abs(float(printed) - value) <= 0.5 * 10 ** -decimals + 1e-12


    def check(label, expected):
        rows, bad, n = table_rows(label), [], 0
        for name, values in expected.items():
            printed = rows.get(name)
            if printed is None:
                bad.append(f"{name}: row not found")
                continue
            if len(printed) != len(values):
                bad.append(f"{name}: {len(printed)} numbers printed, {len(values)} expected")
                continue
            for p, v in zip(printed, values):
                n += 1
                if not same(p, v):
                    bad.append(f"{name}: printed {p}, artifact {v:.5f}")
        if verbose:
            print(f"{label}: {n} numbers checked, {len(bad)} mismatches")
            for b in bad:
                print("   ", b)
        return not bad


    # Test 1 -----------------------------------------------------------------------
    s1 = pd.read_csv(A / "oeh_v2_sealed_test_2026-10-14/summary.csv")


    def t1(method):
        out = []
        for panel in ["internal", "colosseum2029", "spibb2", "forest", None]:
            r = s1[(s1.scope == "pooled_external") & (s1.method == method)] if panel is None else \
                s1[(s1.scope == "panel") & (s1.source_panel == panel) & (s1.method == method)]
            out += [r.brier.item(), r.log_loss.item()]
        return out


    ok = check("tab:v2test", {
        "OEH": t1("v2"), "OEH-F (no gate)": t1("oeh_final"), "Robust head (B2 + D)": t1("legacy"),
        "B2 + LCE": t1("b2_plus_lce"), "B2 matched": t1("b2_matched"), "XGBoost-CF/18": t1("xgboost_cf_18")})

    # Test 2 -----------------------------------------------------------------------
    s2 = pd.read_csv(A / "oeh_fifth_panel_2026-10-15/summary.csv")


    def t2(method):
        return [x for sc in ["frozenlake", "gambler", "pooled"]
                for x in s2[(s2.scope == sc) & (s2.method == method)][["brier", "log_loss"]].iloc[0]]


    ok &= check("tab:fifth", {
        "OEH": t2("oeh"), "OEH-F (no gate)": t2("oeh_full_head"), "Robust head (B2 + D)": t2("robust"),
        "B2 matched": t2("b2_matched"), "LCE raw": t2("lce_raw"), "$k$-NN analog on B2": t2("knn_analog")})

    # Test 3 -----------------------------------------------------------------------
    s3 = pd.read_csv(A / "oeh_panel6_2026-10-15/summary.csv")
    tp = pd.read_csv(A / "oeh_panel6_2026-10-15/tabpfn_descriptive.csv")


    def t3(method, frame=s3):
        g = lambda sc, c: frame[(frame.scope == sc) & (frame.method == method)][c].item()
        return [g("mdpp", "ibs"), g("mdpp", "ll"), g("labmaze", "ibs"), g("labmaze", "ll"),
                g("pooled", "ibs_ge"), g("pooled", "ll_ext")]


    ok &= check("tab:panel6", {
        "OEH": t3("oeh"), "OEH-F (no gate)": t3("oeh_full_head"), "Robust head (B2 + D)": t3("robust"),
        "B2 + prefix/FQE": t3("b2_plus_prefix_fqe"), "B2 + LCE": t3("b2_plus_lce"), "LCE raw": t3("lce_raw"),
        "TabPFN v2 on B2": t3("tabpfn_v2", tp), "B2 matched": t3("b2_matched")})

    if verbose:
        print("ALL TABLES MATCH" if ok else "MISMATCHES FOUND")

    return ok


if __name__ == "__main__":
    sys.exit(0 if main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT) else 1)

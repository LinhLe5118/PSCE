"""Descriptive TabPFN v2 scores for panel 6, after the label step (not part of the registered analysis).

The forecasts were sealed by ``scripts/analysis/panel6_tabpfn.py`` before any outcome; this script only scores them
with the T1 scorer next to OEH and the main comparators. Built with PriorLabs-TabPFN (Hollmann et al., 2025).

Usage: ``python -m scripts.analysis.panel6_tabpfn_describe``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis import run_oeh_panel6_20261015 as S6

E = S6.E
KEYS, P = S6.KEYS, S6.P


def describe() -> pd.DataFrame:
    """Descriptive only (the registered analysis does not include TabPFN): TabPFN against OEH and the main comparators
    on the full and extrapolation endpoints, per-MDP means, after the label step."""
    import numpy as np
    from scripts.analysis import extrapolation_endpoint as X

    rows, inputs = [], {}
    for panel in S6.PANELS:
        target = S6.panel_dir(panel)
        tab, run = E.load_json(target / "tabpfn_seal.json"), E.load_json(target / "outcome_run.json")
        if not run["completed_at_utc"] > tab["sealed_at_utc"]:
            raise RuntimeError(f"{panel}: TabPFN was not sealed before the outcomes")
        labels = pd.read_csv(target / "labels.csv")[KEYS + ["event"]]
        pred = pd.concat([pd.read_csv(target / "predictions_preoutcome.csv"),
                          pd.read_csv(target / "tabpfn_predictions_preoutcome.csv")], ignore_index=True)
        m = pred[pred.method.isin(["tabpfn_v2", "oeh", "b2_matched", "b2_plus_prefix_fqe", "robust", "oeh_full_head"])].merge(
            labels, on=KEYS, validate="many_to_one")
        for k, v in X.row_scores(m[P].to_numpy(float), m.event.to_numpy(int)).items():
            m[k] = v
        rows.append(m.assign(source_panel=panel))
        inputs.update({E.rel(target / n): E.sha256(target / n) for n in ("labels.csv", "tabpfn_predictions_preoutcome.csv",
                                                                       "predictions_preoutcome.csv")})
    s = pd.concat(rows, ignore_index=True)
    metrics = ["ibs", "ll", "ibs_ge", "ll_ext"]
    per = s.groupby(["source_panel", "method", "mdp_id"])[metrics].mean().reset_index()
    out = []
    for scope, panels in [(p, (p,)) for p in S6.PANELS] + [("pooled", tuple(S6.PANELS))]:
        w = per[per.source_panel.isin(panels)].pivot_table(index=["source_panel", "mdp_id"], columns="method", values=metrics)
        for method in sorted(set(per.method)):
            row = {"scope": scope, "method": method, **{m: float(w[(m, method)].mean()) for m in metrics}}
            for m in metrics:  # gain of OEH over the method (positive = OEH better), descriptive 95% interval
                g = (w[(m, method)] - w[(m, "oeh")]).to_numpy()
                half = 1.96 * g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else float("nan")
                row[f"{m}_gain_of_oeh"], row[f"{m}_low95"], row[f"{m}_high95"] = g.mean(), g.mean() - half, g.mean() + half
            out.append(row)
    table = pd.DataFrame(out)
    path = S6.OUTPUT / "tabpfn_descriptive.csv"
    table.to_csv(path, index=False)
    E.write_json(S6.OUTPUT / "tabpfn_descriptive_manifest.json", {
        "status": "descriptive_after_labels (not part of the registered analysis)", "created_utc": E.now(),
        "attribution": "Built with PriorLabs-TabPFN (Hollmann et al., 2025)", "inputs_sha256": inputs,
        "outputs_sha256": {"tabpfn_descriptive.csv": E.sha256(path)}, "source_sha256": {E.rel(Path(__file__)): E.sha256(Path(__file__))}})
    return table



if __name__ == "__main__":
    print(describe().round(4).to_string(index=False))

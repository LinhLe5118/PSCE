"""Apply the seventh-cohort registered lower-bound-only gates after all jobs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import run_oeh_confirmatory_20261004 as C
from scripts.analysis.run_unseen_validation import load_json, write_json


def main() -> None:
    C.configure()
    R = C.R
    R.base.analyze()
    from scripts.analysis import oeh_final_analysis as A

    A.OUTPUT = R.OUTPUT
    doc, _ = R.base.verified_preoutcome()
    if doc.get("confirmatory_schema") != "oeh-seventh-one-shot-v1":
        raise RuntimeError("Wrong confirmatory registration")
    selection = A.selection_study(doc)
    horizon = A.horizon_study()
    contrasts = pd.read_csv(R.OUTPUT / "contrasts.csv")
    if len(contrasts) != 2 * len(R.COMPARATORS):
        raise RuntimeError("Incomplete difficulty contrast panel")

    def verdict(names):
        panel = contrasts[contrasts.baseline.isin(names)]
        if len(panel) != 2 * len(names):
            raise RuntimeError("Missing registered baseline")
        return "GO" if bool((panel.simultaneous_one_sided_lower_95 > 0).all()) else "NO-GO"

    groups = pd.read_csv(R.OUTPUT / "selection_group_outcomes.csv")
    effects = groups[["mdp_id", "oeh_final_selection|suc", "fixed:cfqi_tabular|suc",
                      "oeh_final_selection|ret", "fixed:cfqi_tabular|ret"]].copy()
    effects["success"] = effects["oeh_final_selection|suc"] - effects["fixed:cfqi_tabular|suc"]
    effects["return"] = effects["oeh_final_selection|ret"] - effects["fixed:cfqi_tabular|ret"]
    paired = effects.groupby("mdp_id")[["success", "return"]].mean()
    if len(paired) != 84:
        raise RuntimeError("Expected 84 independent MDP clusters")
    lower, critical = A.simultaneous_lower(paired.to_numpy(), 10000, 20261005)
    gate = "GO" if bool((lower > 0).all()) else "NO-GO"
    result = {
        "status": "complete",
        "protocol_sha256": R.sha256(R.OUTPUT / "protocol.json"),
        "preoutcome_seal_sha256": R.sha256(R.OUTPUT / "preoutcome_seal.json"),
        "external_benchmark_result_sha256": R.sha256(C.ROOT / "artifacts/published_fourrooms_calibration_2026-10-04/result.json"),
        "difficulty_lower_only": {"tier1": verdict(R.TIER1), "tier2_privileged": verdict(R.TIER2),
                                  "all_registered": verdict(R.COMPARATORS)},
        "selection_vs_fixed_cfqi": {"gate": gate, "success_gain": float(paired.success.mean()),
                                     "success_lower_95": float(lower[0]),
                                     "return_gain": float(paired["return"].mean()),
                                     "return_lower_95": float(lower[1]),
                                     "simultaneous_critical": float(critical),
                                     "mdps": len(paired)},
        "selection_vs_best_fixed_and_v5_descriptive": selection,
        "horizon_study_rows": len(horizon["rows"]),
        "note": "positive lower bound is required; no 5-percent point-improvement cutoff",
    }
    write_json(R.OUTPUT / "confirmatory_verdict.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

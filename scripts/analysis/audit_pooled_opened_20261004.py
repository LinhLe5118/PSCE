"""Pool existing out-of-sample development forecasts; descriptive only."""
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.oeh_final_analysis import simultaneous_lower

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "artifacts/oeh_final_development_2026-10-01/gate_A2_plus_prior_profile/scored.csv"
FINAL = ROOT / "artifacts/oeh_final_unseen_120_2026-10-02/scored_predictions.csv"


def main() -> None:
    a = pd.read_csv(DEV)
    b = pd.read_csv(FINAL)
    a["cohort"] = a.held_out
    b["cohort"] = "fourth120"
    frame = pd.concat([a, b], ignore_index=True)
    frame = frame[frame.panel == "primary"]
    mdp = frame.groupby(["cohort", "mdp_id", "method"], as_index=False)[["integrated_brier", "seven_class_log_loss"]].mean()
    pivot = mdp.pivot(index=["cohort", "mdp_id"], columns="method", values=["integrated_brier", "seven_class_log_loss"])
    comparators = [c for c in pivot["integrated_brier"] if c != "oeh_final" and c not in ("oeh_b_raw", "prior_oeh_matched")
                   and pivot[("integrated_brier", c)].notna().all()]
    contrasts, effects = [], []
    for baseline in comparators:
        for metric in ("integrated_brier", "seven_class_log_loss"):
            effect = (pivot[(metric, baseline)] - pivot[(metric, "oeh_final")]).to_numpy()
            effects.append(effect)
            contrasts.append((baseline, metric, effect.mean()))
    lower, critical = simultaneous_lower(np.column_stack(effects), 10000, 20261004)
    print("MDPs", len(pivot), "critical", critical)
    for (baseline, metric, point), bound in zip(contrasts, lower):
        print(baseline, metric, "gain", round(float(point), 5), "lower", round(float(bound), 5))


if __name__ == "__main__":
    main()

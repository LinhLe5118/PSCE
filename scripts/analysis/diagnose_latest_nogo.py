"""Exploratory audit of the opened fourth cohort; never used as confirmation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/oeh_final_unseen_120_2026-10-02"


def main() -> None:
    groups = pd.read_csv(OUT / "selection_group_outcomes.csv")
    oeh = json.loads((OUT / "oeh_choices_preoutcome.json").read_text())
    v5 = json.loads((OUT / "v5_choices_preoutcome.json").read_text())
    groups["oeh_choice"] = [oeh[f"{m}|{b}"] for m, b in zip(groups.mdp_id, groups.policy_family)]
    groups["v5_choice"] = [v5[f"{m}|{b}"] for m, b in zip(groups.mdp_id, groups.policy_family)]
    groups["family"] = groups.mdp_id.str.extract(r"oehfinal120_(.+)_\d{3}$")[0]
    methods = [c.removeprefix("fixed:").removesuffix("|suc") for c in groups if c.startswith("fixed:") and c.endswith("|suc")]
    for family, part in groups.groupby("family"):
        print("FAMILY", family, "groups", len(part))
        for rule in ("oeh_final_selection", "oeh_v5_router", "oracle_group", "fixed:cfqi_tabular"):
            print(rule, "ret", round(part[f"{rule}|ret"].mean(), 4), "suc", round(part[f"{rule}|suc"].mean(), 4))
        print("OEH choices", part.oeh_choice.value_counts().to_dict())
        print("v5 choices", part.v5_choice.value_counts().to_dict())
    print("DIFFERENT", int((groups.oeh_choice != groups.v5_choice).sum()))
    for choice, part in groups.groupby("v5_choice"):
        print("V5", choice, "groups", len(part), "suc advantage over OEH", round((part["oeh_v5_router|suc"] - part["oeh_final_selection|suc"]).mean(), 4))
    print("oracle composition", pd.Series([max(methods, key=lambda x: (row[f"fixed:{x}|suc"], row[f"fixed:{x}|ret"])) for _, row in groups.iterrows()]).value_counts().to_dict())


if __name__ == "__main__":
    main()

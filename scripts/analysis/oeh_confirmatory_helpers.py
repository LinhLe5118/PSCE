"""Read the opened fourth cohort as meta-training data for new experiments."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis import run_oeh_final_120 as R

ROOT = Path(__file__).resolve().parents[2]
FOURTH = ROOT / "artifacts/oeh_final_unseen_120_2026-10-02"
FOURTH_DATASET = ROOT / "datasets/oeh_final_unseen_120_2026-10-02"


def fourth_frame() -> pd.DataFrame:
    """Construct the same feature panel as ``R.opened_frame`` without copying artifacts."""
    frame = pd.read_csv(FOURTH / "labels.csv").merge(
        pd.read_csv(FOURTH / "pilot_features.csv"), on=R.KEYS, validate="one_to_one")
    oracle = pd.read_csv(FOURTH / "oracle_descriptors_preoutcome.csv")
    frame = frame.merge(oracle[["mdp_id", *R.EH]], on="mdp_id", validate="many_to_one")
    oeh = R.F.oeh_feature_rows(FOURTH / "oeh_b_groups", FOURTH / "horizon_preoutcome.csv")
    frame = frame.merge(oeh, on=R.KEYS, validate="one_to_one")
    published = pd.read_csv(FOURTH / "published_preoutcome.csv")
    frame = frame.merge(published, on=["mdp_id", "policy_family"], validate="many_to_one")
    original = pd.read_csv(FOURTH / "tqsaco_original_preoutcome.csv")
    frame = frame.merge(original[["mdp_id", "policy_family", "tq_original", "saco_original"]],
                        on=["mdp_id", "policy_family"], validate="many_to_one")
    curve = R.curve_features(pd.read_csv(FOURTH / "curve_preoutcome.csv"))
    frame = frame.merge(curve, on=R.KEYS, validate="one_to_one")
    ceiling = pd.read_csv(FOURTH / "ceiling_preoutcome.csv")
    frame = frame.merge(ceiling, on=["mdp_id", "policy_family"], validate="many_to_one")
    if len(frame) != 1800 or frame.mdp_id.nunique() != 120:
        raise RuntimeError("Incomplete opened fourth meta-training cohort")
    frame["cohort"] = "fourth120"
    return R.with_context(frame)


def expanded_meta_training(tag: str = "A2") -> pd.DataFrame:
    historical = R.meta_training(R.OPENED, tag)
    fourth = fourth_frame()
    if set(historical.mdp_id) & set(fourth.mdp_id):
        raise RuntimeError("Meta-training cohorts overlap")
    return pd.concat([historical, fourth], ignore_index=True)

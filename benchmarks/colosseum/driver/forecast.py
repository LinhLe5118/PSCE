"""Fit the frozen matched forecaster on the team's sealed meta-training panel.

The training panel is exactly the one the sealed cohort-7 run used: the
runner's own ``expanded_meta`` over e1, first60, second60, first120,
second120, third120 and the opened fourth cohort, from the published A2
OEH-B groups, horizon, curve, ceiling and published-metric files.

Modes:
  reproduce  rebuild the cohort-7 query from its sealed files, refit, and
             compare with the sealed cohort-7 predictions and scores.
  colosseum  predict the 32 Colosseum MDPs with the same fit; write the
             predictions plus a SHA-256 seal before any learner outcome.

Comparators share one fitting procedure (``matched_forecast``):
  b2_matched         18 generic pilot features (the paper's primary control)
  oeh_final          the frozen 108-column OEH field list
  b2_plus_colosseum  B2 + Colosseum's three hardness measures (true MDP)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import COHORTS, DATA, OUT, RES, RUN, RUN_BUNDLE_NAME, load_json, sha256, use_workspace, write_json

use_workspace()
from scripts.analysis import run_oeh_confirmatory_20261004 as C  # noqa: E402
from scripts.analysis import run_oeh_final_120 as R  # noqa: E402
from scripts.analysis import oeh_final_forecast as F  # noqa: E402
from scripts.analysis.compare_difficulty import FEATURES_B2  # noqa: E402

C.configure()
DOC = load_json(COHORTS["c7"] / "protocol.json")
R.use_candidate(DOC["oeh_final"]["candidate_variant"], DOC["oeh_final"]["candidate_law"])
F.SELECT_CAPACITY = DOC["oeh_final"]["select_capacity"]
OEH_FIELDS = tuple(DOC["oeh_final"]["fields"])
assert tuple(R.FIELD_SETS[R.CANDIDATE]) == OEH_FIELDS, "candidate fields differ from the frozen protocol"
HARD = ("log_diameter", "value_norm", "log_suboptimal_gaps")
FIELD_SETS = {"b2_matched": tuple(FEATURES_B2), "oeh_final": OEH_FIELDS,
              "b2_plus_colosseum": (*FEATURES_B2, *HARD)}
HARD_FILE = {"e1": "e1", "first60": "first60", "second60": "second60", "first120": "first120",
             "second120": "second120", "third120": "third120", "fourth120": "c4"}


def hardness(name: str) -> pd.DataFrame:
    h = pd.read_csv((RES if name == "colosseum" else OUT) / f"hardness_{name}.csv")
    return pd.DataFrame({"mdp_id": h.mdp_id.astype(str), "log_diameter": np.log(h.diameter.clip(lower=1e-9)),
                         "value_norm": h.value_norm,
                         "log_suboptimal_gaps": np.log(h.suboptimal_gaps.clip(lower=1e-9))})


def add_hardness(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for cohort, part in frame.groupby("cohort", sort=False):
        part = part.copy()
        part["_raw_id"] = part.mdp_id.astype(str).str.replace("opened_fourth120/", "", regex=False)
        h = hardness(HARD_FILE.get(cohort, cohort)).rename(columns={"mdp_id": "_raw_id"})
        merged = part.merge(h, on="_raw_id", how="left", validate="many_to_one")
        if merged[list(HARD)].isna().any().any():
            raise RuntimeError(f"Missing hardness for cohort {cohort}")
        parts.append(merged.drop(columns="_raw_id"))
    return pd.concat(parts, ignore_index=True)


def training_panel() -> pd.DataFrame:
    train = C.expanded_meta(R.OPENED, DOC["oeh_final"]["structure_tag"])
    if train.mdp_id.nunique() != 720 or len(train) != 10_800:
        raise RuntimeError(f"Unexpected meta-training panel: {train.mdp_id.nunique()} MDPs, {len(train)} rows")
    return add_hardness(train)


def seventh_query() -> pd.DataFrame:
    """Cohort-7 query rows rebuilt from its sealed pre-outcome files."""
    d = COHORTS["c7"]
    q = pd.read_csv(d / "pilot_features.csv")
    q = q.merge(pd.read_csv(d / "oracle_descriptors_preoutcome.csv")[["mdp_id", *R.EH]], on="mdp_id",
                validate="many_to_one")
    q = q.merge(F.oeh_feature_rows(d / "oeh_b_groups", d / "horizon_preoutcome.csv"), on=R.KEYS,
                validate="one_to_one")
    q = q.merge(R.curve_features(pd.read_csv(d / "curve_preoutcome.csv")), on=R.KEYS, validate="one_to_one")
    q = q.merge(pd.read_csv(d / "ceiling_preoutcome.csv"), on=["mdp_id", "policy_family"], validate="many_to_one")
    published = pd.read_csv(d / "published_preoutcome.csv").merge(
        pd.read_csv(d / "tqsaco_original_preoutcome.csv")[["mdp_id", "policy_family", "tq_original", "saco_original"]],
        on=["mdp_id", "policy_family"], validate="one_to_one")
    q = q.merge(published, on=["mdp_id", "policy_family"], validate="many_to_one")
    labels = pd.read_csv(d / "labels.csv")[R.KEYS + ["panel"]]
    q = R.with_context(q.merge(labels, on=R.KEYS, validate="one_to_one"))
    q["cohort"] = "c7"
    return add_hardness(q)


def colosseum_query() -> pd.DataFrame:
    q = pd.read_csv(RES / "colosseum_pilot_features.csv")
    q = q.merge(F.oeh_feature_rows(OUT / "oehb" / RUN_BUNDLE_NAME, RES / "colosseum_horizon.csv"),
                on=R.KEYS, validate="one_to_one")
    q = R.with_context(q)
    q["panel"], q["cohort"] = "primary", "colosseum"
    return add_hardness(q)


def forecast(train: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for method, fields in FIELD_SETS.items():
        probability, _ = F.matched_forecast(train, query, fields)
        table = query[R.KEYS + ["panel"]].copy()
        table["method"] = method
        table[[f"p_{i}" for i in range(7)]] = probability
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def mdp_scores(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    m = predictions.merge(labels[R.KEYS + ["event"]], on=R.KEYS, validate="many_to_one")
    P = m[[f"p_{i}" for i in range(7)]].to_numpy(float)
    e = m.event.to_numpy(int)
    m["ibs"] = ((np.cumsum(P, 1)[:, :6] - (e[:, None] <= np.arange(6)).astype(float)) ** 2).mean(1)
    m["ll"] = -np.log(np.clip(P[np.arange(len(e)), e], 1e-15, 1))
    return m.groupby(["panel", "method", "mdp_id"])[["ibs", "ll"]].mean().reset_index()


def main() -> None:
    mode = sys.argv[1]
    train = training_panel()
    print(f"training panel: {len(train)} rows, {train.mdp_id.nunique()} MDPs, cohorts {sorted(train.cohort.unique())}",
          flush=True)
    if mode == "reproduce":
        query = seventh_query()
        predictions = forecast(train, query)
        sealed = pd.read_csv(COHORTS["c7"] / "predictions_preoutcome.csv")
        joined = predictions[predictions.method != "b2_plus_colosseum"].merge(
            sealed, on=R.KEYS + ["method"], suffixes=("", "_sealed"), validate="one_to_one")
        diff = np.abs(joined[[f"p_{i}" for i in range(7)]].to_numpy()
                      - joined[[f"p_{i}_sealed" for i in range(7)]].to_numpy()).max(1)
        scores = mdp_scores(predictions, pd.read_csv(COHORTS["c7"] / "labels.csv"))
        mine = scores.groupby(["panel", "method"])[["ibs", "ll"]].mean()
        ref = pd.read_csv(COHORTS["c7"] / "summary.csv").set_index(["panel", "method"])
        rows = [{"panel": p, "method": m, "brier": v.ibs, "logloss": v.ll,
                 "brier_sealed": ref.integrated_brier.get((p, m)), "logloss_sealed": ref.seven_class_log_loss.get((p, m))}
                for (p, m), v in mine.iterrows()]
        result = pd.DataFrame(rows)
        result.to_csv(OUT / "reproduce_cohort7.csv", index=False)
        scores.to_csv(OUT / "reproduce_cohort7_mdp_scores.csv", index=False)
        write_json(OUT / "reproduce_cohort7.json", {
            "rows_compared": int(len(joined)), "max_abs_probability_diff": float(diff.max()),
            "rows_within_1e-6": int((diff < 1e-6).sum())})
        print(result.round(6).to_string(index=False))
        print(f"probability agreement with sealed predictions: max |diff| {diff.max():.2e}, "
              f"{(diff < 1e-6).sum()}/{len(diff)} rows within 1e-6")
    elif mode == "colosseum":
        target = RES / "colosseum_predictions_preoutcome.csv"
        if target.exists():
            raise SystemExit("Colosseum predictions already sealed; refusing to overwrite")
        query = colosseum_query()
        expected = load_json(DATA / f"{RUN}_selection.json")["selected"] * 15
        if len(query) != expected or query[R.KEYS].duplicated().any():
            raise SystemExit(f"Incomplete Colosseum query: {len(query)} rows")
        predictions = forecast(train, query)
        predictions.to_csv(target, index=False)
        write_json(RES / "colosseum_preoutcome_seal.json", {
            "learner_outcomes_observed": 0, "training_rows": len(train),
            "training_mdps": int(train.mdp_id.nunique()), "training_cohorts": sorted(train.cohort.unique()),
            "query_rows": len(query), "fields": FIELD_SETS, "predictions_sha256": sha256(target),
            "protocol_sha256": sha256(RES / "protocol_colosseum.json")})
        print(f"sealed {len(predictions)} predictions; train {len(train)} rows / {train.mdp_id.nunique()} MDPs")
    else:
        raise SystemExit("mode: reproduce | colosseum")


if __name__ == "__main__":
    main()

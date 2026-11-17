"""Matched requirement forecaster and the final OEH feature block.

Every feature set (generic B2, the previous OEH candidate and final OEH) is fit
by one procedure: the team's frozen AFT, direct-threshold and depth-3
multiclass components, MDP-grouped five-fold out-of-fold predictions on the
meta-training rows, convex component weights chosen by out-of-fold seven-class
log loss, and a final fit on all meta-training rows. Weights are learned from
data by the same rule for every comparator; no value is set by hand for OEH.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS
from scripts.analysis.develop_eh_oracle_three_way import aft_probabilities
from scripts.analysis.develop_oeh_difficulty import fit_thresholds
from scripts.analysis.develop_three_way_ensemble import choose_weights
from scripts.analysis.develop_tree_capacity import VARIANTS, multiclass

KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]
CAPACITY = VARIANTS["depth3_200"]
BUDGETS = (1, 2, 5, 10, 20, 50)
OWN = tuple([f"oehb_cdf_{j}" for j in range(6)] + [f"oehb_success_x{b}" for b in BUDGETS]
            + [f"oehb_return_x{b}" for b in BUDGETS])
HORIZON = ("oehh_value", "oehh_identified", "oehh_trap", "oehh_statistical", "oehh_coverage",
           "oehh_depth_fraction", "oehh_log_concentrability", "oehh_log_clipped_concentrability",
           "oehh_log_intrinsic_bound", "oehh_behavior_ceiling", "oehh_support_ceiling",
           "oehb_log_transition_concentration",
           "oehb_log_behavior_concentration")
CONTEXT = tuple(f"other_{m}_oehb_cdf_{j}" for m in METHODS for j in range(6))
PROFILE = tuple([f"oehh_risk_x{b}" for b in BUDGETS] + [f"oehh_miss_x{b}" for b in BUDGETS] + ["oehh_risk_slack"])
FINAL_FIELDS = (*FEATURES_B2, *OWN, *HORIZON, *CONTEXT)


def fold_of(mdp_id: str) -> int:
    """Deterministic MDP-grouped fold, independent of features and outcomes."""
    return int.from_bytes(hashlib.sha256(mdp_id.encode()).digest()[:4], "little") % 5


def _finite(value, fallback=0.0):
    return fallback if value is None or not math.isfinite(float(value)) else float(value)


def oeh_feature_rows(group_dir: Path, horizon_csv: Path | None) -> pd.DataFrame:
    """Final OEH block per (MDP, behavior, learner) from OEH-B and OEH-H outputs."""
    rows = []
    for path in sorted(group_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        posterior = document["posterior"]
        for learner, value in document["learners"].items():
            cdf = np.cumsum(value["probabilities"])[:6]
            success = value["success_rate"] or [0.0] * 6
            returns = value["mean_normalized_return"] or [0.0] * 6
            rows.append({
                "mdp_id": document["mdp_id"], "policy_family": document["policy_family"],
                "algorithm": learner,
                **{f"oehb_cdf_{j}": float(cdf[j]) for j in range(6)},
                **{f"oehb_p_{c}": float(value["probabilities"][c]) for c in range(7)},
                **{f"oehb_success_x{b}": float(success[j]) for j, b in enumerate(BUDGETS)},
                **{f"oehb_return_x{b}": float(returns[j]) for j, b in enumerate(BUDGETS)},
                "oehb_log_transition_concentration": _finite(
                    math.log(posterior["transition_concentration"])
                    if posterior["transition_concentration"] and math.isfinite(posterior["transition_concentration"])
                    else None),
                "oehb_log_behavior_concentration": _finite(
                    math.log(posterior["behavior_concentration"])
                    if posterior["behavior_concentration"] and math.isfinite(posterior["behavior_concentration"])
                    else None),
            })
    frame = pd.DataFrame(rows)
    if frame.empty or frame[KEYS].duplicated().any():
        raise RuntimeError(f"Invalid OEH-B group directory: {group_dir}")
    for method in METHODS:
        context = frame[frame.algorithm == method][["mdp_id", "policy_family",
                                                    *[f"oehb_cdf_{j}" for j in range(6)]]]
        context = context.rename(columns={f"oehb_cdf_{j}": f"other_{method}_oehb_cdf_{j}" for j in range(6)})
        frame = frame.merge(context, on=["mdp_id", "policy_family"], validate="many_to_one")
    if horizon_csv is not None:
        h = pd.read_csv(horizon_csv)
        h = pd.DataFrame({
            "mdp_id": h.mdp_id, "policy_family": h.policy_family,
            "oehh_value": h.data_oeh_h.fillna(0.0),
            "oehh_identified": (h.data_oeh_h_status == "identified").astype(float),
            "oehh_trap": (h.data_oeh_h_status == "coverage_trap").astype(float),
            "oehh_statistical": h.data_oeh_h_statistical.fillna(0.0),
            "oehh_coverage": h.data_oeh_h_coverage.fillna(0.0),
            "oehh_depth_fraction": h.data_oeh_h_depth.fillna(101.0) / 100.0,
            "oehh_log_concentrability": h.data_log_concentrability.fillna(np.inf),
            "oehh_log_clipped_concentrability": h.data_log_clipped_concentrability.fillna(np.inf),
            "oehh_log_intrinsic_bound": h.data_log_intrinsic_bound.fillna(np.inf),
            "oehh_behavior_ceiling": h.data_behavior_ceiling.fillna(0.0),
            "oehh_support_ceiling": h.data_support_ceiling.fillna(0.0),
            **{f"oehh_{kind}_x{b}": h.get(f"data_{kind}_x{b}", pd.Series(np.nan, index=h.index)).fillna(1.0)
               for kind in ("risk", "miss") for b in BUDGETS},
            "oehh_risk_slack": h.get("data_risk_slack", pd.Series(np.nan, index=h.index)).fillna(-1.0),
        })
        frame = frame.merge(h, on=["mdp_id", "policy_family"], validate="many_to_one")
    else:
        for name in HORIZON[:11]:
            frame[name] = 0.0
    return frame


def _matrix(frame: pd.DataFrame, fields) -> np.ndarray:
    return np.nan_to_num(frame[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)


SELECT_CAPACITY = False  # A6: choose multiclass capacity per stratum by OOF log loss
LAW = tuple(f"oehb_p_{c}" for c in range(7))


def recalibrated_law(train_law: np.ndarray, y: np.ndarray, query_law: np.ndarray) -> np.ndarray:
    """Unpenalized multinomial logistic recalibration of a seven-category law."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(penalty=None, max_iter=10_000)
    model.fit(np.log(np.clip(train_law, 1e-12, 1.0)), y)
    probability = np.zeros((len(query_law), 7))
    probability[:, model.classes_] = model.predict_proba(np.log(np.clip(query_law, 1e-12, 1.0)))
    return probability


def convex_weights(components: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Convex mixture weights minimizing log loss (the team's rule, any count)."""
    from scipy.optimize import minimize

    event = components[:, np.arange(len(y)), y]
    objective = lambda w: float(-np.log(np.maximum(w @ event, 1e-12)).mean())  # noqa: E731
    k = len(components)
    fitted = minimize(objective, np.ones(k) / k, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                      constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
                      options={"ftol": 1e-12, "maxiter": 1000})
    candidates = [np.eye(k)[i] for i in range(k)]
    if fitted.success and np.isfinite(fitted.x).all():
        candidates.append(np.clip(fitted.x, 0, 1) / np.clip(fitted.x, 0, 1).sum())
    best = min(candidates, key=objective)
    return best, objective(best)


def matched_forecast(train: pd.DataFrame, query: pd.DataFrame, fields,
                     law: bool = False) -> tuple[np.ndarray, dict]:
    """Seven-category forecasts for ``query`` with weights learned on ``train``.

    ``law=True`` adds the recalibrated OEH-B posterior-predictive law as a
    fourth ensemble component (OEH outputs only).
    """
    out = np.full((len(query), 7), np.nan)
    weights = {}
    for panel in ("primary", "control"):
        for learner in METHODS:
            tr = train[(train.panel == panel) & (train.algorithm == learner)].reset_index(drop=True)
            qi = np.flatnonzero(((query.panel == panel) & (query.algorithm == learner)).to_numpy())
            if not len(qi):
                continue
            if tr.mdp_id.nunique() < 10:
                raise RuntimeError("Too few meta-training MDPs for grouped folds")
            x, y = _matrix(tr, fields), tr.event.to_numpy(dtype=int)
            folds = tr.mdp_id.map(fold_of).to_numpy()
            capacities = list(VARIANTS) if SELECT_CAPACITY else ["depth3_200"]
            oof = np.full((2 + len(capacities), len(tr), 7), np.nan)
            law_oof = np.full((len(tr), 7), np.nan)
            law_train = tr[list(LAW)].to_numpy(float) if law else None
            for fold in range(5):
                fit, val = folds != fold, folds == fold
                if not val.any():
                    continue
                oof[0, val] = aft_probabilities(x[fit], y[fit], x[val])
                oof[1, val] = fit_thresholds(x[fit], y[fit], x[val])
                for c, name in enumerate(capacities):
                    oof[2 + c, val] = multiclass(x[fit], y[fit], x[val], VARIANTS[name])
                if law:
                    law_oof[val] = recalibrated_law(law_train[fit], y[fit], law_train[val])
            event_loss = [float(-np.log(np.maximum(oof[2 + c, np.arange(len(y)), y], 1e-12)).mean())
                          for c in range(len(capacities))]
            best = int(np.argmin(event_loss))
            capacity = capacities[best]
            q = _matrix(query.iloc[qi], fields)
            parts = [aft_probabilities(x, y, q), fit_thresholds(x, y, q), multiclass(x, y, q, VARIANTS[capacity])]
            if law:
                choice, loss = convex_weights(np.concatenate([oof[[0, 1, 2 + best]], law_oof[None]]), y)
                parts.append(recalibrated_law(law_train, y, query.iloc[qi][list(LAW)].to_numpy(float)))
            else:
                choice, loss = choose_weights(oof[[0, 1, 2 + best]], y)
            parts = np.asarray(parts)
            out[qi] = np.einsum("i,ijk->jk", choice, parts)
            weights[f"{panel}/{learner}"] = {"aft_direct_multi": choice.tolist(), "oof_logloss": loss,
                                             "capacity": capacity}
    if not np.isfinite(out).all() or not np.allclose(out.sum(1), 1.0):
        raise RuntimeError("Incomplete matched forecast")
    return out, weights

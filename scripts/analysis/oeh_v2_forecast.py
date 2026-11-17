"""OEH v2: shift-aware two-stage combination of sealed-procedure heads (single definition).

Given heads fitted with ``oeh_final_forecast.matched_forecast`` on the training panel:

* ``p_oeh``     OEH current (108 fields), all rows;
* ``cond_oeh``  the same fields fitted on finite rows only (its p_6 is zero);
* ``p_robust``  B2 + legacy diagnostics (``prior_oeh_matched`` fields), all rows;

and ``d`` (mean distance to the 10 nearest training primary rows in standardized
B2 statistics) and the learner's training class distribution ``prior``:

    P(E = 6)          = p_oeh[6]
    q(k | finite)     = (1 - lambda(d)) * cond_oeh(k) + lambda(d) * p_robust(k | finite)
    law               = [(1 - P6) * q, P6]
    OEH v2            = (1 - eps(d)) * law + eps(d) * prior,
    lambda(d) = lambda_max * min(1, d / d0),  eps(d) = eps_max * min(1, d / d0).

Parameters are chosen on opened data by ``develop_oeh_v2_heads_20261014``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P = [f"p_{i}" for i in range(7)]


def finite_part(law: np.ndarray) -> np.ndarray:
    return law[:, :6] / np.clip(law[:, :6].sum(1, keepdims=True), 1e-300, None)


def shift_weight(maximum: float, d: np.ndarray, d0) -> np.ndarray:
    return np.full_like(d, maximum, dtype=float) if d0 == "const" else maximum * np.minimum(1.0, d / float(d0))


def v2_law(p_oeh: np.ndarray, cond_oeh: np.ndarray, p_robust: np.ndarray, d: np.ndarray, prior: np.ndarray,
           *, lambda_max: float, eps_max: float, d0) -> np.ndarray:
    lam = shift_weight(lambda_max, d, d0)[:, None]
    eps = shift_weight(eps_max, d, d0)[:, None]
    q = (1 - lam) * finite_part(cond_oeh) + lam * finite_part(p_robust)
    p6 = p_oeh[:, 6:7]
    law = np.column_stack([q * (1 - p6), p6])
    law = (1 - eps) * law + eps * prior
    return law / law.sum(1, keepdims=True)


def knn_distance(train: pd.DataFrame, query: pd.DataFrame, k: int = 10) -> np.ndarray:
    """Per-row distance of each query MDP-behavior group to the training primary rows."""
    from scipy.spatial import cKDTree
    from scripts.analysis.compare_difficulty import FEATURES_B2

    cols = list(FEATURES_B2)
    meta = train[train.panel == "primary"]
    mu, sd = meta[cols].mean(), meta[cols].std().replace(0, 1)
    tree = cKDTree(((meta[cols] - mu) / sd).fillna(0).to_numpy())
    groups = query.drop_duplicates(["mdp_id", "policy_family"])
    dist, _ = tree.query(((groups[cols] - mu) / sd).fillna(0).to_numpy(), k=k)
    series = pd.Series(dist.mean(1), index=pd.MultiIndex.from_frame(groups[["mdp_id", "policy_family"]]))
    return series.reindex(pd.MultiIndex.from_frame(query[["mdp_id", "policy_family"]])).to_numpy()


def class_prior(train: pd.DataFrame) -> dict[str, np.ndarray]:
    primary = train[train.panel == "primary"]
    return {learner: np.bincount(g.event.to_numpy(int), minlength=7) / len(g) for learner, g in primary.groupby("algorithm")}

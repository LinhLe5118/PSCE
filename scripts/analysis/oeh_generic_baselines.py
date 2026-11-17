"""Generic pilot-only baselines on the 18 B2 statistics (no OEH information).

* ``knn_analog``: the requirement law of the k nearest meta-training rows of the same
  learner in standardized B2 space (analog forecasting), with add-1/2 smoothing;
* ``logistic``: an L2-regularized multinomial logistic regression per learner on the
  standardized B2 statistics, floored at 1e-4 and renormalized.

Both use only the meta-training panel and the query's pilot statistics, like B2 matched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_NEIGHBORS = 25
LOGISTIC_C = 1.0


def _standardize(train: pd.DataFrame, query: pd.DataFrame):
    from scripts.analysis.compare_difficulty import FEATURES_B2
    cols = list(FEATURES_B2)
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    return ((train[cols] - mu) / sd).fillna(0).to_numpy(), ((query[cols] - mu) / sd).fillna(0).to_numpy()


def knn_analog(train: pd.DataFrame, query: pd.DataFrame, k: int = K_NEIGHBORS) -> np.ndarray:
    from scipy.spatial import cKDTree
    out = np.full((len(query), 7), np.nan)
    primary = train[train.panel == "primary"]
    for learner, rows in primary.groupby("algorithm"):
        qi = np.flatnonzero((query.algorithm == learner).to_numpy())
        if not len(qi):
            continue
        x, q = _standardize(rows, query.iloc[qi])
        _, idx = cKDTree(x).query(q, k=k)
        y = rows.event.to_numpy(int)
        counts = np.stack([np.bincount(y[i], minlength=7) for i in idx])
        out[qi] = (counts + 0.5) / (k + 3.5)
    return out


def logistic(train: pd.DataFrame, query: pd.DataFrame, c: float = LOGISTIC_C) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    out = np.full((len(query), 7), np.nan)
    primary = train[train.panel == "primary"]
    for learner, rows in primary.groupby("algorithm"):
        qi = np.flatnonzero((query.algorithm == learner).to_numpy())
        if not len(qi):
            continue
        x, q = _standardize(rows, query.iloc[qi])
        y = rows.event.to_numpy(int)
        model = LogisticRegression(C=c, max_iter=5000).fit(x, y)
        p = np.zeros((len(qi), 7))
        p[:, model.classes_] = model.predict_proba(q)
        p = np.maximum(p, 1e-4)
        out[qi] = p / p.sum(1, keepdims=True)
    return out

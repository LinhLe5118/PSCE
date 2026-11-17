"""Frozen E1 meta-training and label-free routing for an unseen MDP batch.

The forecasting primitives, ridge grid, feature blocks and joint route are
the existing OEH definitions. E1 outcomes serve only as meta-training labels;
query input contains pilot features and no outcomes. This is an explicitly
declared deployment protocol, not a rewrite of the historical E1 analysis.
"""
from __future__ import annotations

import numpy as np

from . import core
from .reference import FEATURES, METHODS as ANCHORS


def fit_unseen_routes(train_panel, query_features, query_panels):
    """Fit on all E1 MDPs, then choose methods for an entirely unlabeled batch.

    ``query_features`` maps (mdp, behavior, core-anchor) to the two frozen
    feature blocks; ``query_panels`` maps new MDP IDs to the existing primary
    or control panel. The query has no outcome field by construction.
    """
    methods = tuple(train_panel["candidate_methods"])
    if methods[:len(ANCHORS)] != ANCHORS or len(set(methods)) != len(methods):
        raise ValueError("Invalid frozen learner library")
    training_ids = set(train_panel["outer"])
    if not query_panels or training_ids & set(query_panels):
        raise ValueError("Query MDPs must be unseen and nonempty")
    if set(query_panels.values()) - set(train_panel["panels"]):
        raise ValueError("Unknown query panel")
    if not query_features:
        raise ValueError("Empty query feature table")
    groups = sorted({key[:2] for key in query_features})
    if any(key[0] not in query_panels or key[2] not in ANCHORS for key in query_features):
        raise ValueError("Unregistered query feature identity")
    expected = {(*group, anchor) for group in groups for anchor in ANCHORS}
    if set(query_features) != expected:
        raise ValueError("Incomplete core-anchor pilot features")
    for group in groups:
        if not group[0] or not group[1]:
            raise ValueError("Invalid query group")
    for key, blocks in query_features.items():
        if set(blocks) != set(FEATURES):
            raise ValueError("Invalid query feature blocks")
        for name in FEATURES:
            array = np.asarray(blocks[name], dtype=float)
            if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
                raise ValueError("Invalid query feature values")

    choices, forecasts, tuning = {}, {}, []
    for panel_name, train_mdps in train_panel["panels"].items():
        query_groups = [g for g in groups if query_panels[g[0]] == panel_name]
        if not query_groups:
            continue
        train_groups = sorted(g for g in train_panel["runs"] if g[0] in train_mdps)
        if len(set(g[0] for g in train_groups)) < 4 or not train_groups:
            raise ValueError("Insufficient MDP-grouped E1 meta-training data")
        weights = np.ones(len(train_groups), dtype=int)
        panel_forecasts = {g: {ANCHORS[0]: [0., 0.]} for g in query_groups}
        for method in methods[1:]:
            targets = np.asarray([
                core.paired_targets(train_panel["runs"][g][ANCHORS[0]],
                                    train_panel["runs"][g][method])
                for g in train_groups
            ])
            block_predictions = []
            for feature in FEATURES:
                def row(table, group):
                    if method in ANCHORS:
                        baseline = np.asarray(table[(*group, ANCHORS[0])][feature], dtype=float)
                        candidate = np.asarray(table[(*group, method)][feature], dtype=float)
                        return np.concatenate((baseline, candidate - baseline))
                    return np.concatenate([
                        np.asarray(table[(*group, anchor)][feature], dtype=float)
                        for anchor in ANCHORS
                    ])
                x = np.asarray([row(train_panel["features"], g) for g in train_groups])
                query = np.asarray([row(query_features, g) for g in query_groups])
                if x.ndim != 2 or query.ndim != 2 or x.shape[1] != query.shape[1]:
                    raise ValueError("Query/train feature dimension mismatch")
                values, model = core.forecast_block(
                    x, query, targets, weights, [g[0] for g in train_groups],
                    train_panel["return_ridges"],
                    f"paired-quadratic|0|{method}|{feature}",
                )
                block_predictions.append(values)
                tuning.append({"panel": panel_name, "method": method,
                               "feature": feature, **model})
            combined = np.mean(block_predictions, axis=0)
            for group, value in zip(query_groups, combined):
                panel_forecasts[group][method] = value.tolist()
        matrix = np.asarray([[panel_forecasts[g][m] for m in methods] for g in query_groups])
        route, evidence = core.joint_routes(matrix[:, :, 0], matrix[:, :, 1])
        tuning.append({"panel": panel_name, "library": list(methods), **evidence})
        for group, index in zip(query_groups, route):
            choices[group] = methods[int(index)]
            forecasts[group] = panel_forecasts[group]
    if set(choices) != set(groups):
        raise RuntimeError("Incomplete unseen batch routing")
    return choices, forecasts, tuning

"""An unseen batch must be routed using only frozen E1 meta-training labels."""
import copy

import pytest

from oeh.reference import FEATURES, METHODS as ANCHORS
from oeh.unseen import fit_unseen_routes


def fixture():
    methods = (*ANCHORS, "external_fixture")
    train = dict(runs={}, features={}, outer={}, panels={"primary": ()},
                 candidate_methods=methods, return_ridges=(.1, 1.))
    mdps = []
    for i in range(8):
        mdp = f"train_{i}"; mdps.append(mdp); train["outer"][mdp] = i % 5
        for behavior in ("a", "b"):
            group = (mdp, behavior)
            train["runs"][group] = {}
            for method_index, method in enumerate(methods):
                value = .5 + .02 * i + .08 * method_index
                train["runs"][group][method] = {seed: (value, int(value >= .95)) for seed in range(10)}
            for method_index, method in enumerate(ANCHORS):
                train["features"][(*group, method)] = {
                    name: (i / 8, float(method_index), float(behavior == "b")) for name in FEATURES
                }
    train["panels"]["primary"] = tuple(mdps)
    query = {}
    for mdp in ("new_0", "new_1"):
        for behavior in ("a", "b"):
            for method_index, method in enumerate(ANCHORS):
                query[(mdp, behavior, method)] = {
                    name: (.3, float(method_index), float(behavior == "b")) for name in FEATURES
                }
    panels = {"new_0": "primary", "new_1": "primary"}
    return train, query, panels


def test_unseen_route_has_no_query_outcome_dependency():
    train, query, panels = fixture()
    first, forecasts, tuning = fit_unseen_routes(train, query, panels)
    assert len(first) == 4 and len(forecasts) == 4 and tuning
    assert set(first.values()) <= set(train["candidate_methods"])
    # The interface contains query pilot coordinates only. Reordering feature
    # records cannot change a decision or expose query labels.
    second, other, _ = fit_unseen_routes(train, dict(reversed(list(query.items()))), panels)
    assert first == second and forecasts == other
    changed = copy.deepcopy(query)
    del changed[("new_0", "a", ANCHORS[0])]
    with pytest.raises(ValueError, match="Incomplete"):
        fit_unseen_routes(train, changed, panels)
    with pytest.raises(ValueError, match="unseen"):
        fit_unseen_routes(train, query, {"new_0": "primary", "train_0": "primary"})

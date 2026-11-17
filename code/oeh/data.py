"""Strict interchange for precomputed pilot features and matched learner labels."""
import json
from pathlib import Path

import numpy as np

from .reference import FEATURES, METHODS

RIDGE_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def load_panel(path):
    """Read grouped software/data inputs; never generate MDPs or policy outcomes.

    A group's outcomes are read only when its MDP belongs to the training side
    of the current outer fold. The router itself implements that separation.
    """
    document = json.loads(Path(path).read_text('utf-8'))
    methods = tuple(document['methods'])
    ridges = tuple(document['return_ridges'])
    if (document.get('schema_version') != 1 or methods[:3] != METHODS
            or len(methods) < 4 or len(set(methods)) != len(methods)
            or not all(isinstance(m, str) and m for m in methods)
            or ridges != RIDGE_GRID or any(type(r) not in (int, float) for r in ridges)):
        raise ValueError('Invalid panel schema, registered library or ridge grid')
    panel = dict(runs={}, features={}, outer={}, v4e_decisions={}, return_ridges=ridges,
                 candidate_methods=methods, panels={})
    dimensions = {}
    for row in document['groups']:
        mdp, behavior = row['mdp_id'], row['policy_family']
        g = mdp, behavior
        fold, name = row['outer_fold'], row['panel']
        if (not isinstance(mdp, str) or not mdp or not isinstance(behavior, str) or not behavior
                or g in panel['runs'] or type(fold) is not int or fold not in range(5)
                or not isinstance(name, str) or not name):
            raise ValueError('Invalid/duplicate group or outer fold')
        if mdp in panel['outer'] and (panel['outer'][mdp] != fold or mdp not in panel['panels'].get(name, ())):
            raise ValueError('A whole MDP must remain in one panel and outer fold')
        panel['outer'][mdp] = fold
        panel['panels'].setdefault(name, set()).add(mdp)
        reference = row.get('v4e_algorithm', METHODS[0])
        if reference not in METHODS or set(row['outcomes']) != set(methods):
            raise ValueError('Unregistered reference or incomplete method library')
        panel['v4e_decisions'][g] = reference
        panel['runs'][g] = {}
        for method in methods:
            labels = {}
            for outcome in row['outcomes'][method]:
                seed, value, success = outcome['seed'], outcome['normalized_return'], outcome['success']
                if (type(seed) is not int or seed not in range(10) or seed in labels
                        or type(value) not in (float, int) or not np.isfinite(value)
                        or type(success) is not bool or success != (value >= .95)):
                    raise ValueError('Invalid/duplicate/nonfinite seed outcome or success threshold')
                labels[seed] = float(value), int(success)
            if set(labels) != set(range(10)):
                raise ValueError('Incomplete ten-seed labels')
            panel['runs'][g][method] = labels
        if set(row['features']) != set(METHODS):
            raise ValueError('Pilot features must contain the three registered core anchors')
        for method in METHODS:
            features = row['features'][method]
            if set(features) != set(FEATURES):
                raise ValueError('Unregistered pilot feature blocks')
            panel['features'][(*g, method)] = {}
            for feature in FEATURES:
                values = np.asarray(features[feature], float)
                key = feature
                if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
                        or (key in dimensions and len(values) != dimensions[key])):
                    raise ValueError('Invalid/nonfinite/inconsistent pilot feature dimensions')
                dimensions[key] = len(values)
                panel['features'][(*g, method)][feature] = tuple(values)
    if not panel['runs']:
        raise ValueError('Empty input panel')
    for name, mdps in panel['panels'].items():
        if {panel['outer'][m] for m in mdps} != set(range(5)):
            raise ValueError('Each panel needs all five original outer folds')
    panel['panels'] = {name:tuple(sorted(values)) for name, values in panel['panels'].items()}
    return panel

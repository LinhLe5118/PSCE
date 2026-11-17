"""Fixed quadratic paired forecasts and joint gain routing used by OEH."""
import numpy as np
from .reference import FEATURES, METHODS, grouped_split, original_routes, score_routes
from .numerics import deterministic_balanced_folds, fit_standardizer, transform_standardized

def simplex(values):
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError('Invalid categorical forecasts')
    ordered = np.sort(values, axis=1)[:, ::-1]
    offsets = (ordered.cumsum(axis=1)-1)/np.arange(1, 4)
    rho = (ordered > offsets).sum(axis=1)-1
    return np.maximum(values-offsets[np.arange(len(values)), rho, None], 0)

def kernel_predictions(x, query, targets, weights, ridges):
    """Weighted centered kernel ridge, one eigensolve for all targets/ridges.

    Weighted centering leaves the intercept unpenalized. Standardization is
    fitted on training rows only, with whole-MDP bootstrap multiplicities.
    """
    x, query, y = np.asarray(x, float), np.asarray(query, float), np.asarray(targets, float)
    w = np.asarray(weights)
    grid = np.asarray(ridges, float)
    if (x.ndim != 2 or query.ndim != 2 or not len(x) or not x.shape[1]
            or query.shape[1] != x.shape[1] or y.ndim != 2 or len(y) != len(x)
            or w.shape != (len(x),) or not np.issubdtype(w.dtype, np.integer) or np.any(w <= 0)
            or grid.ndim != 1 or not len(grid) or np.any(grid <= 0)
            or not all(np.isfinite(a).all() for a in (x, query, y, grid))):
        raise ValueError('Invalid kernel training/query/targets/multiplicities/ridges')
    mean, scale = fit_standardizer(np.repeat(x, w, axis=0))
    z = transform_standardized(x, mean, scale)
    q = transform_standardized(query, mean, scale)
    dimension = x.shape[1]
    gram = (1+z@z.T/dimension)**2
    cross = (1+q@z.T/dimension)**2
    p = w/w.sum()
    column = p@gram; grand = float(column@p)
    centered = gram-(gram@p)[:, None]-column[None, :]+grand
    centered_cross = cross-(cross@p)[:, None]-column[None, :]+grand
    root_w = np.sqrt(w)
    system = root_w[:, None]*centered*root_w[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh((system+system.T)/2)
    # Polynomial kernels are PSD; remove negative roundoff eigenvalues only.
    eigenvalues = np.maximum(eigenvalues, 0)
    outcome_mean = p@y
    coordinates = eigenvectors.T@(root_w[:, None]*(y-outcome_mean))
    left = centered_cross@(root_w[:, None]*eigenvectors)
    predictions = np.array([outcome_mean+left@(coordinates/(eigenvalues[:, None]+ridge)) for ridge in grid])
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite quadratic endpoint forecasts')
    return predictions

def paired_targets(reference, alternative):
    if not reference or set(reference) != set(alternative):
        raise ValueError('Unpaired seed labels')
    differences, counts = [], np.zeros(3)
    for seed in sorted(reference):
        a, b = reference[seed], alternative[seed]
        if a[1] not in (0, 1) or b[1] not in (0, 1) or not np.isfinite([a[0], b[0]]).all():
            raise ValueError('Invalid paired labels')
        differences.append(b[0]-a[0])
        counts[0 if b[1] > a[1] else 1 if b[1] < a[1] else 2] += 1
    return np.concatenate(([np.mean(differences)], counts/len(reference)))

def forecast_block(x, test, y, weights, mdps, ridges, salt):
    folds = deterministic_balanced_folds(mdps, num_folds=4, salt=salt)
    errors = np.zeros((len(ridges), 2)); total = 0
    for fold in range(4):
        a = np.array([i for i, mdp in enumerate(mdps) if folds[mdp] != fold])
        b = np.array([i for i, mdp in enumerate(mdps) if folds[mdp] == fold])
        forecast = kernel_predictions(x[a], x[b], y[a], weights[a], ridges)
        for index in range(len(ridges)):
            errors[index, 0] += np.dot(weights[b], (forecast[index, :, 0]-y[b, 0])**2)
            probability = simplex(forecast[index, :, 1:])
            errors[index, 1] += np.dot(weights[b], np.sum((probability-y[b, 1:])**2, axis=1))
        total += weights[b].sum()
    errors /= total
    chosen = [min(range(len(ridges)), key=lambda i:(errors[i, endpoint], -ridges[i])) for endpoint in range(2)]
    predicted = kernel_predictions(x, test, y, weights, ridges)
    probability = simplex(predicted[chosen[1], :, 1:])
    values = np.column_stack((predicted[chosen[0], :, 0], probability[:, 0]-probability[:, 1]))
    return values, dict(return_ridge=float(ridges[chosen[0]]), success_ridge=float(ridges[chosen[1]]),
        inner_return_mse=float(errors[chosen[0], 0]), inner_categorical_frequency_loss=float(errors[chosen[1], 1]))

def joint_routes(returns, success, weights=None):
    r, s = np.asarray(returns, float), np.asarray(success, float)
    w = np.ones(len(r)) if weights is None else np.asarray(weights, float)
    if (r.ndim != 2 or r.shape != s.shape or not r.shape[1] or w.shape != (len(r),)
            or not all(np.isfinite(a).all() for a in (r, s, w)) or np.any(w <= 0)
            or np.any(r[:, 0] != 0) or np.any(s[:, 0] != 0)):
        raise ValueError('Invalid forecast/reference/weights')
    boundaries = {0.}
    for a in range(r.shape[1]):
        for b in range(a+1, r.shape[1]):
            delta = s[:, b]-s[:, a]; valid = delta != 0
            crossing = (r[valid, a]-r[valid, b])/delta[valid]
            boundaries.update(float(v) for v in crossing if np.isfinite(v) and v > 0)
    rows = np.arange(len(r)); best = np.zeros(len(r), dtype=int); product = 0.; multiplier = None
    gains = [0., 0.]
    for value in sorted(boundaries):
        utility = r+value*s; maximum = utility.max(axis=1)
        tolerance = 8*np.finfo(float).eps*np.maximum(1, np.abs(maximum))
        tied = np.abs(utility-maximum[:, None]) <= tolerance[:, None]
        route = np.argmax(np.where(tied, s, -np.inf), axis=1)
        nr, sr = float(w@r[rows, route]), float(w@s[rows, route])
        score = nr*sr if nr > 0 and sr > 0 else 0.
        if score > product:
            best, product, multiplier, gains = route, score, value, [nr, sr]
    return best, dict(lagrange_breakpoint=multiplier, predicted_joint_gains=gains)

def fit_routes(panel, mdps, *, multiplicities=None, original_v4e=False):
    if original_v4e:
        return original_routes(panel, mdps, multiplicities)
    weights, groups = grouped_split(panel, mdps, multiplicities)
    choices, predictions, tuning = {}, {}, []
    for fold in range(5):
        train = [g for g in groups if panel['outer'][g[0]] != fold]
        test = [g for g in groups if panel['outer'][g[0]] == fold]
        if not test:
            continue
        forecasts = {g:{METHODS[0]:[0., 0.]} for g in test}
        train_weights = np.array([weights[g[0]] for g in train], dtype=int)
        for method in METHODS[1:]:
            y = np.array([paired_targets(panel['runs'][g][METHODS[0]], panel['runs'][g][method]) for g in train])
            blocks = []
            for feature in FEATURES:
                def row(g):
                    a, b = panel['features'][(*g, METHODS[0])][feature], panel['features'][(*g, method)][feature]
                    return tuple(a)+tuple(np.asarray(b)-a)
                x, query = np.array([row(g) for g in train]), np.array([row(g) for g in test])
                values, model = forecast_block(x, query, y, train_weights, [g[0] for g in train],
                    panel['return_ridges'], f'paired-quadratic|{fold}|{method}|{feature}')
                blocks.append(values); tuning.append(dict(fold=fold, method=method, feature=feature, **model))
            for g, values in zip(test, np.mean(blocks, axis=0)):
                forecasts[g][method] = values.tolist()
        values = np.array([[forecasts[g][m] for m in METHODS] for g in test])
        route, evidence = joint_routes(values[:, :, 0], values[:, :, 1], [weights[g[0]] for g in test])
        tuning.append(dict(fold=fold, **evidence))
        for g, index in zip(test, route):
            choices[g] = METHODS[index]; predictions[g] = forecasts[g]
    return choices, predictions, tuning

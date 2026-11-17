"""Vectorized equivalent of OEH box-simplex water filling; no approximation."""
import numpy as np


def expectation(values, lower, upper, *, maximize):
    order = np.argsort(values, kind='stable')
    if maximize:
        order = order[::-1]
    width = (upper-lower)[..., order]
    remaining = np.maximum(0, 1-lower.sum(-1))[..., None]
    # Capacity before each sorted coordinate, exclusive cumulative sum.
    before = np.concatenate([np.zeros_like(width[..., :1]),
                             np.cumsum(width[..., :-1], axis=-1)], axis=-1)
    extra = np.minimum(width, np.maximum(0, remaining-before))
    if np.any(remaining[..., 0]-width.sum(-1) > 1e-10):
        raise ValueError('Empty transition confidence set.')
    return np.sum(lower*values, axis=-1) + np.sum(extra*values[order], axis=-1)

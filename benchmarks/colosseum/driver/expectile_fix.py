"""Apply the team's documented expectile fix to the archived IQL at runtime.

The archived ``empirical_requirement._expectile`` raises when rounding puts
every split candidate outside its interval (tied action values of magnitude
above ~4). ``code/oeh/tabular.py`` fixes it by keeping the registered split
order and 1e-15 tolerance (bitwise-identical whenever the archived code
succeeds) and otherwise returning the split with the smallest interval
violation. This wrapper reproduces exactly that rule for the scalar version
and counts how often the fallback is used. No file is modified.
"""
from __future__ import annotations

import numpy as np

CALLS = {"fallback": 0}


def _fixed(original):
    def expectile(values: np.ndarray, weights: np.ndarray, expectile: float) -> float:
        try:
            return original(values, weights, expectile)
        except RuntimeError:
            CALLS["fallback"] += 1
            order = np.argsort(values, kind="stable")
            v, w = values[order], weights[order]
            best, violation = None, np.inf
            for split in range(len(v) + 1):
                c = np.empty_like(w)
                c[:split], c[split:] = 1.0 - expectile, expectile
                eff = w * c
                candidate = float(np.dot(eff, v) / eff.sum())
                lower = float(v[split - 1]) if split else -np.inf
                upper = float(v[split]) if split < len(v) else np.inf
                miss = max(lower - candidate, candidate - upper)
                if miss < violation:
                    best, violation = candidate, miss
            return best
    return expectile


def install() -> None:
    import empirical_requirement as er
    if not getattr(er._expectile, "_trial_fixed", False):
        fixed = _fixed(er._expectile)
        fixed._trial_fixed = True
        er._expectile = fixed

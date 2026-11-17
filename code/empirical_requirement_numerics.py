"""Numerical recovery for the frozen IQL objective, without changing its code.

The original expectile computes weighted means of uncentered values and uses
an absolute interval tolerance. At an almost constant positive value, division
roundoff can lie beyond every interval. Expectiles are translation equivariant:
solve after subtracting the minimum, then restore the offset. The IQL function
below uses the original bytecode/defaults with only this numerical primitive
replaced in a private globals dictionary; no shared-module monkeypatch occurs.
"""
from types import FunctionType

import numpy as np

from empirical_requirement import _expectile, iql_tabular as original_iql


def centered_expectile(values, weights, expectile):
    values = np.asarray(values,float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite nonempty expectile values required')
    offset = float(values.min())
    result = offset+_expectile(values-offset,weights,expectile)
    return float(np.clip(result,values.min(),values.max()))


iql_tabular = FunctionType(original_iql.__code__,
    {**original_iql.__globals__,'_expectile':centered_expectile},
    name='iql_tabular_centered',argdefs=original_iql.__defaults__,closure=original_iql.__closure__)
iql_tabular.__kwdefaults__ = original_iql.__kwdefaults__.copy()

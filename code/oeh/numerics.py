"""Original training-only scaling and deterministic MDP grouping."""
import hashlib
from typing import Sequence
import numpy as np

def deterministic_balanced_folds(
    groups: Sequence[str], *, num_folds: int, salt: str
) -> dict[str, int]:
    """Assign unique groups to deterministic folds with size imbalance <= 1."""

    unique = sorted(set(groups))
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2")
    if len(unique) < num_folds:
        raise ValueError("fewer groups than folds")
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{salt}|{value}".encode("utf-8")).digest(),
    )
    return {group: position % num_folds for position, group in enumerate(ordered)}

def fit_standardizer(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit column scaling; constant columns remain finite and map to zero."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("matrix must be nonempty and two-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix contains non-finite values")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    return mean, scale

def transform_standardized(
    matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    result = (np.asarray(matrix, dtype=float) - mean) / scale
    if not np.all(np.isfinite(result)):
        raise ValueError("standardized matrix contains non-finite values")
    return result

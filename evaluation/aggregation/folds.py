"""5-fold cross-validation split, by base MDP (``PLAN_Final.md`` sec 7).

Every behavior policy, pilot size, and training size of one MDP instance
lands in the same fold. The split is a deterministic hash, never a random
draw that could vary run to run or be re-rolled after looking at results
(``PLAN_Final.md`` sec 11, rule 2).
"""

from __future__ import annotations

import hashlib
from typing import Iterable


def assign_folds(mdp_ids: Iterable[str], *, num_folds: int = 5) -> dict[str, int]:
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2.")
    assignment: dict[str, int] = {}
    for mdp_id in mdp_ids:
        digest = hashlib.sha256(mdp_id.encode("utf-8")).hexdigest()
        assignment[mdp_id] = int(digest, 16) % num_folds
    return assignment


__all__ = ["assign_folds"]

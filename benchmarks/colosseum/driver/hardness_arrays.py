"""Colosseum's three hardness measures from (T, R) arrays (Python 3.9 env).

Same functions the Colosseum MDP properties call in the continuous setting:
diameter = get_diameter(T, False); (Q, V) = discounted_value_iteration(T, R);
value norm = calculate_norm_discounted(T, V); gaps = sum of reciprocals.
Usage: python driver/hardness_arrays.py <kernel_dir> <out_csv>
"""
from __future__ import annotations

import csv
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
if not hasattr(os, "sysconf"):
    os.sysconf = lambda key: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 4 * 2**30}[key]
TRIAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRIAL / "colosseum" / "src"))
import numpy as np  # noqa: E402
import colosseum.mdp  # noqa: E402,F401  (breaks a circular import in the package)
from colosseum.dynamic_programming import discounted_value_iteration  # noqa: E402
from colosseum.hardness.measures import (calculate_norm_discounted, get_diameter,  # noqa: E402
                                         get_sum_reciprocals_suboptimality_gaps)

from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

CAP = 1.0e4
source, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for path in sorted(source.glob("*.npz")):
    tick = time.time()
    with np.load(path) as z:
        T, R = z["T"].astype(np.float32), z["R"].astype(np.float32)
    T = T / T.sum(-1, keepdims=True)
    Q, V = discounted_value_iteration(T, R)
    # The diameter is infinite iff the MDP is not communicating, i.e. the
    # graph "some action reaches s'" is not strongly connected. Encode that
    # case as CAP (flagged); otherwise use Colosseum's own computation.
    support = csr_matrix((T > 0).any(axis=1))
    communicating = connected_components(support, directed=True, connection="strong")[0] == 1
    capped = not communicating
    diameter = CAP if capped else get_diameter(T, False)
    rows.append({"mdp_id": path.stem, "diameter": float(diameter), "diameter_capped": capped,
                 "value_norm": float(calculate_norm_discounted(T, V)),
                 "suboptimal_gaps": float(get_sum_reciprocals_suboptimality_gaps(Q, V)),
                 "seconds": round(time.time() - tick, 2)})
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"{len(rows)} MDPs -> {out}")

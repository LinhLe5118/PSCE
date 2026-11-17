"""Export Colosseum MDP candidates (run with the Python 3.9 env).

Selection rule (fixed before any learner outcome): for each of the seven
Colosseum families used in the authors' benchmark (MG-DoorKey excluded, as
the authors did), draw parameters with the authors' own sampler
``sample_mdp_parameters(n, is_episodic=False, seed=2026)`` and keep, in
sampler order, instances that are stochastic and have 16 <= |S||A| <= 256.
Up to MAX_PER_FAMILY candidates are exported; the project's dynamic-range
filter and the first-5-per-family cut are applied afterwards in Python 3.12.

Also exports Colosseum's three hardness measures (diameter, discounted value
norm, sum of reciprocal sub-optimality gaps) using the authors' code.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
if not hasattr(os, "sysconf"):  # Windows shim for a Linux-only RAM query
    os.sysconf = lambda key: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 4 * 2**30}[key]
TRIAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRIAL / "colosseum" / "src"))
import numpy as np  # noqa: E402
from colosseum import config  # noqa: E402

config.set_up_hardness_measures_cache_folder = lambda: None  # Windows MAX_PATH workaround
config.disable_multiprocessing()
config.set_hardness_measures_cache_folder(tempfile.mkdtemp(dir=TRIAL / ".tmp") + os.sep)
from colosseum.mdp.deep_sea import DeepSeaContinuous  # noqa: E402
from colosseum.mdp.frozen_lake import FrozenLakeContinuous  # noqa: E402
from colosseum.mdp.minigrid_empty import MiniGridEmptyContinuous  # noqa: E402
from colosseum.mdp.minigrid_rooms import MiniGridRoomsContinuous  # noqa: E402
from colosseum.mdp.river_swim import RiverSwimContinuous  # noqa: E402
from colosseum.mdp.simple_grid import SimpleGridContinuous  # noqa: E402
from colosseum.mdp.taxi import TaxiContinuous  # noqa: E402

FAMILIES = {"riverswim": RiverSwimContinuous, "deepsea": DeepSeaContinuous,
            "frozenlake": FrozenLakeContinuous, "simplegrid": SimpleGridContinuous,
            "mg_empty": MiniGridEmptyContinuous, "mg_rooms": MiniGridRoomsContinuous,
            "taxi": TaxiContinuous}
RUN = os.environ.get("COLOSSEUM_RUN", "colosseum")
SAMPLER_SEED = int(os.environ.get("SAMPLER_SEED", 2026))
DRAWS = int(os.environ.get("DRAWS", 80))
MAX_PER_FAMILY = int(os.environ.get("MAX_PER_FAMILY", 9))
PREFIX = "colosseum" if RUN == "colosseum" else f"{RUN}_colosseum"
OUT = TRIAL / "data" / RUN


def kernel_hash(T, R):
    return hashlib.sha256(np.asarray(T, np.float64).tobytes() + np.asarray(R, np.float64).tobytes()).hexdigest()


# Kernels used by any earlier run must not reappear (fresh MDPs only).
SEEN = set()
if RUN != "colosseum":
    for path in (TRIAL / "data" / "colosseum").glob("*.npz"):
        with np.load(path) as z:
            SEEN.add(kernel_hash(z["T"], z["R"]))
EXCLUDED_EARLIER = len(SEEN)
OUT.mkdir(parents=True, exist_ok=True)


def clean(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


index = []
for family, cls in FAMILIES.items():
    kept = 0
    for draw, params in enumerate(cls.sample_mdp_parameters(DRAWS, False, seed=SAMPLER_SEED)):
        if kept >= MAX_PER_FAMILY:
            break
        tick = time.time()
        try:
            mdp = cls(seed=draw, **params)
            T, R = np.asarray(mdp.T, float), np.asarray(mdp.R, float)
        except Exception as error:  # noqa: BLE001  (record and skip invalid draws)
            index.append({"family": family, "draw": draw, "status": f"error:{type(error).__name__}"})
            continue
        S, A = R.shape
        stochastic = bool(((T > 1e-12) & (T < 1 - 1e-12)).any())
        if not stochastic or not 16 <= S * A <= 256:
            index.append({"family": family, "draw": draw, "status": "filtered", "S": S, "A": A,
                          "stochastic": stochastic})
            continue
        kernel = kernel_hash(T, R)
        if kernel in SEEN:
            index.append({"family": family, "draw": draw, "status": "duplicate_kernel"})
            continue
        SEEN.add(kernel)
        name = f"{PREFIX}_{family}_{draw:03d}"
        hardness = {"diameter": float(mdp.diameter),
                    "value_norm": float(mdp.discounted_value_norm),
                    "suboptimal_gaps": float(mdp.sum_reciprocals_suboptimality_gaps)}
        np.savez(OUT / f"{name}.npz", T=T, R=R, rho=np.asarray(mdp.starting_state_distribution, float))
        record = {"family": family, "draw": draw, "status": "candidate", "mdp_id": name, "S": S, "A": A, "kernel_sha256": kernel,
                  "params": {k: clean(v) for k, v in params.items()}, **hardness,
                  "seconds": round(time.time() - tick, 2)}
        index.append(record)
        kept += 1
        print(json.dumps({k: record[k] for k in ("mdp_id", "S", "A", "diameter", "value_norm",
                                                  "suboptimal_gaps", "seconds")}), flush=True)
(OUT / "candidates.json").write_text(json.dumps(
    {"run": RUN, "sampler_seed": SAMPLER_SEED, "draws_per_family": DRAWS, "max_per_family": MAX_PER_FAMILY,
     "earlier_kernels_excluded": EXCLUDED_EARLIER,
     "rule": "authors' sampler order; stochastic; 16<=S*A<=256", "index": index}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print("candidates:", sum(r["status"] == "candidate" for r in index))

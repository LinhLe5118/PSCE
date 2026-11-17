"""Fast Colosseum kernel export (Python 3.9 env), same selection rule as colosseum_export.py.

Differences from ``colosseum_export.py``: the four families that ever produced a
candidate (RiverSwim, DeepSea, FrozenLake, SimpleGrid; the MiniGrid/Taxi
families never satisfy |S||A| <= 256) and no hardness properties during the
export (``hardness_arrays.py`` computes them afterwards from the arrays). The
kernel construction ``cls(seed=draw, **params)`` and the filters are unchanged.

Usage:
  python colosseum_export_fast.py --src <Colosseum/src> --out <dir> --prefix <p> --seed S
         --draws N --cap K [--exclude <dir> ...] [--only <ids.txt>]
``--only`` regenerates exactly the listed MDP ids (``<prefix>_<family>_<draw>``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
if not hasattr(os, "sysconf"):  # Windows shim for a Linux-only RAM query
    os.sysconf = lambda key: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 4 * 2**30}[key]

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--prefix", required=True)
parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--draws", type=int, default=600)
parser.add_argument("--cap", type=int, default=30)
parser.add_argument("--exclude", nargs="*", default=[])
parser.add_argument("--only", type=Path)
args = parser.parse_args()
sys.path.insert(0, args.src)
import numpy as np  # noqa: E402
from colosseum import config  # noqa: E402

config.set_up_hardness_measures_cache_folder = lambda: None
config.disable_multiprocessing()
args.out.mkdir(parents=True, exist_ok=True)
config.set_hardness_measures_cache_folder(tempfile.mkdtemp(dir=args.out) + os.sep)
from colosseum.mdp.deep_sea import DeepSeaContinuous  # noqa: E402
from colosseum.mdp.frozen_lake import FrozenLakeContinuous  # noqa: E402
from colosseum.mdp.river_swim import RiverSwimContinuous  # noqa: E402
from colosseum.mdp.simple_grid import SimpleGridContinuous  # noqa: E402

FAMILIES = {"riverswim": RiverSwimContinuous, "deepsea": DeepSeaContinuous,
            "frozenlake": FrozenLakeContinuous, "simplegrid": SimpleGridContinuous}


def kernel_hash(T, R):
    return hashlib.sha256(np.asarray(T, np.float64).tobytes() + np.asarray(R, np.float64).tobytes()).hexdigest()


def clean(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


seen = set()
for folder in args.exclude:
    for path in Path(folder).glob("*.npz"):
        with np.load(path) as z:
            seen.add(kernel_hash(z["T"], z["R"]))
only = None
if args.only:
    only = {}
    for line in args.only.read_text().split():
        family, draw = line[len(args.prefix) + 1:].rsplit("_", 1)
        only.setdefault(family, set()).add(int(draw))
index = []
for family, cls in FAMILIES.items():
    kept = 0
    for draw, params in enumerate(cls.sample_mdp_parameters(args.draws, False, seed=args.seed)):
        if only is not None and draw not in only.get(family, ()):
            continue
        if only is None and kept >= args.cap:
            break
        try:
            mdp = cls(seed=draw, **params)
            T, R = np.asarray(mdp.T, float), np.asarray(mdp.R, float)
        except Exception as error:  # noqa: BLE001
            index.append({"family": family, "draw": draw, "status": f"error:{type(error).__name__}"})
            continue
        S, A = R.shape
        stochastic = bool(((T > 1e-12) & (T < 1 - 1e-12)).any())
        if only is None and (not stochastic or not 16 <= S * A <= 256):
            index.append({"family": family, "draw": draw, "status": "filtered", "S": S, "A": A})
            continue
        kernel = kernel_hash(T, R)
        if only is None and kernel in seen:
            index.append({"family": family, "draw": draw, "status": "duplicate_kernel"})
            continue
        seen.add(kernel)
        name = f"{args.prefix}_{family}_{draw:03d}"
        np.savez(args.out / f"{name}.npz", T=T, R=R, rho=np.asarray(mdp.starting_state_distribution, float))
        index.append({"family": family, "draw": draw, "status": "candidate", "mdp_id": name, "S": S, "A": A,
                      "kernel_sha256": kernel, "params": {k: clean(v) for k, v in params.items()}})
        kept += 1
(args.out / "candidates.json").write_text(json.dumps(
    {"prefix": args.prefix, "sampler_seed": args.seed, "draws_per_family": args.draws, "max_per_family": args.cap,
     "earlier_kernels_excluded": len(seen) - sum(r["status"] == "candidate" for r in index),
     "regenerated_ids_only": bool(only), "rule": "authors' sampler order; stochastic; 16<=S*A<=256",
     "index": index}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print("candidates:", sum(r["status"] == "candidate" for r in index))

"""Turn exported Colosseum kernels into the project's dataset format.

The project's own generator, behavior-policy suite, pilot/training streams,
budgets, horizon and dynamic-range filter are used unchanged; only the MDP
source is replaced. Selection: every exported candidate (<= 9 per family, in
the authors' sampler order) that passes the project's dynamic-range filter.
"""
from __future__ import annotations

import time

import numpy as np

import os

from common import DATA, RUN, RUN_BUNDLE, RUN_SRC, load_json, sha256, use_workspace, write_json

use_workspace()
from scripts.analysis.run_unseen_validation import archive_imports  # noqa: E402

_, _, generator, _ = archive_imports()
from environments.benchmark_environments import BenchmarkSpec, DYNAMIC_RANGE_THRESHOLD  # noqa: E402
from environments.tabular_environments import TabularMDP  # noqa: E402
from exact_solver import dynamic_range, passes_dynamic_range_filter  # noqa: E402

SOURCE = RUN_SRC
TARGET = RUN_BUNDLE
MAX_SELECT = int(os.environ.get("MAX_SELECT", 10**6))  # first N passing per family, sampler order
SEED_BASE = int(os.environ.get("SEED_BASE", 9_100_000))


def to_mdp(record: dict) -> TabularMDP:
    with np.load(SOURCE / f"{record['mdp_id']}.npz") as z:
        T, R, rho = z["T"], z["R"], z["rho"]
    T = T / T.sum(-1, keepdims=True)
    S, A = R.shape
    mdp = TabularMDP(environment_id=record["mdp_id"], P=T.astype(np.float64),
                     R=np.repeat(R[:, :, None], S, axis=2).astype(np.float64),
                     initial_state_distribution=(rho / rho.sum()).astype(np.float64),
                     terminal_states=(), gamma=0.95,
                     metadata={"source": "colosseum", "family": record["family"], "draw": record["draw"],
                               "params": record["params"], "num_states": S, "num_actions": A})
    mdp.validate()
    return mdp


candidates = [r for r in load_json(SOURCE / "candidates.json")["index"] if r["status"] == "candidate"]
registry, specs, selection = {}, [], []
counts: dict[str, int] = {}
for record in candidates:
    mdp = to_mdp(record)
    keep = passes_dynamic_range_filter(mdp, threshold=DYNAMIC_RANGE_THRESHOLD)
    rng, opt, rand = dynamic_range(mdp)
    room = counts.get(record["family"], 0) < MAX_SELECT
    selection.append({**{k: record[k] for k in ("mdp_id", "family", "draw", "S", "A")},
                      "dynamic_range": rng, "passes_filter": bool(keep), "kept": bool(keep and room)})
    if not (keep and room):
        continue
    k = counts.get(record["family"], 0)
    counts[record["family"]] = k + 1
    spec = BenchmarkSpec(instance_id=record["mdp_id"], split="evaluation", family=f"colosseum_{record['family']}",
                         seed=SEED_BASE + 1_000 * len(specs), ordinal=len(specs), family_ordinal=k)
    registry[spec.instance_id] = mdp
    specs.append(spec)
write_json(DATA / f"{RUN}_selection.json", {
    "rule": "exported candidates passing the project's dynamic-range filter; first MAX_SELECT per family in sampler order",
    "max_select": MAX_SELECT,
    "threshold": DYNAMIC_RANGE_THRESHOLD, "family_counts": counts, "selected": len(specs),
    "candidates": selection, "candidates_sha256": sha256(SOURCE / "candidates.json")})
print("selected", len(specs), counts, flush=True)

generator.resolve_dynamic_range = lambda spec: (spec, registry[spec.instance_id], 0)
tick = time.time()
manifest = generator.generate_benchmark_bundle(
    TARGET, specs=specs, pilot_seeds=(0,), training_seeds=tuple(range(10)), max_size=None,
    horizon=100, run_mode="test", deviation_reason="external trial: Colosseum kernels",
    pilot_size_multipliers=(2, 5, 10))
print(f"bundle: {manifest['instance_count']} MDPs, {manifest['physical_buffer_count']} buffers, "
      f"{time.time() - tick:.0f}s")

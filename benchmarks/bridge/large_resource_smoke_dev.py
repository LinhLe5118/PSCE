"""Metadata-chosen largest BRIDGE development task: resource preflight only."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from benchmarks.bridge.adapter import load_sticky_mdp
from environments.behavior_policies import structured_random_policy
from scripts.data.generate_offline_datasets import collect_trajectories

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from oeh import predictive  # noqa: E402

TASK = "MiniGrid-SimpleCrossingS9N1-v0"
OUT = ROOT / "artifacts/week8/bridge_large_resource_smoke_dev_2026-10-07.json"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    frozen = json.loads((ROOT / "artifacts/week8/bridge_transfer_preflight_freeze_2026-10-07.json")
                        .read_text(encoding="utf-8"))
    if frozen["status"] != "PREFLIGHT_SPLIT_FROZEN_NOT_BENCHMARK_GO" or TASK not in frozen["development_tasks"]:
        raise RuntimeError("Large smoke task was not frozen as development")
    start = time.perf_counter()
    mdp = load_sticky_mdp(TASK)
    adapter_seconds = time.perf_counter() - start
    beta = structured_random_policy(mdp, epsilon=.2, seed=20261007)
    pilot = collect_trajectories(mdp, beta, seed=2026100701,
                                 size=10 * mdp.num_states * mdp.num_actions, horizon=100)
    collect_seconds = time.perf_counter() - start - adapter_seconds
    structure = json.loads((ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04/protocol.json")
                           .read_text(encoding="utf-8"))["oeh_final"]["structure"]
    tick = time.perf_counter()
    outcomes = predictive.posterior_outcomes(
        pilot, mdp.initial_state_distribution, num_states=mdp.num_states,
        num_actions=mdp.num_actions, gamma=mdp.gamma, draws=1, seeds=1,
        budgets=(10,), seed=[20261007], **structure)
    primitive_seconds = time.perf_counter() - tick
    import psutil
    memory_gib = psutil.Process().memory_info().rss / 2**30
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"status": "LARGE_DEVELOPMENT_PRIMITIVE_SMOKE_ONLY",
        "task": TASK, "exclude_from_independent_test": True,
        "source_npz_sha256": mdp.metadata["source_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "states": mdp.num_states, "actions": mdp.num_actions,
        "pilot_transitions": len(pilot["state"]), "pilot_nonzero_reward": int((pilot["reward"] != 0).sum()),
        "adapter_seconds": adapter_seconds, "pilot_collect_seconds": collect_seconds,
        "oehb_primitive_seconds": primitive_seconds, "rss_after_gib": memory_gib,
        "posterior_sampling": outcomes.sampling,
        "warning": "1 draw, 1 seed, one x10 budget is only a resource primitive; not the 64-draw/10-seed/6-budget forecast or a test outcome."},
        indent=2) + "\n", encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()

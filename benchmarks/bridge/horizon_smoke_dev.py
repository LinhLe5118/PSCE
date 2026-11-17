"""Development-only OEH-H/risk feature and runtime smoke on BRIDGE."""
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
from oeh import horizon as H  # noqa: E402
from scripts.analysis.develop_oeh_horizon import posterior_mean_model  # noqa: E402

OUTPUT = ROOT / "artifacts/week8/bridge_horizon_smoke_dev_2026-10-07.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    mdp = load_sticky_mdp("MiniGrid-LavaGapS5-v0")
    beta = structured_random_policy(mdp, epsilon=.2, seed=20261007)
    pilot = collect_trajectories(mdp, beta, seed=2026100701,
                                 size=10 * mdp.num_states * mdp.num_actions, horizon=100)
    frozen = json.loads((ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04/protocol.json")
                        .read_text(encoding="utf-8"))["oeh_final"]
    start = time.perf_counter()
    P, R, behavior, terminal = posterior_mean_model(
        pilot, mdp.initial_state_distribution, mdp.num_states, mdp.num_actions,
        mdp.gamma, **frozen["structure"])
    posterior_seconds = time.perf_counter() - start
    result = H.offline_effective_horizon(P, R, behavior, mdp.initial_state_distribution,
                                         mdp.gamma, terminal=terminal, tau=.95,
                                         horizon=100, form="risk")
    oehh_seconds = time.perf_counter() - start - posterior_seconds
    profile = H.risk_profile(P, R, behavior, mdp.initial_state_distribution,
                             mdp.gamma, terminal=terminal, tau=.95,
                             horizon=100,
                             budgets=[b * mdp.num_states * mdp.num_actions for b in (1, 2, 5, 10, 20, 50)])
    profile_seconds = time.perf_counter() - start - posterior_seconds - oehh_seconds
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"status": "DEVELOPMENT_OEHH_SMOKE_ONLY_NO_TEST_LABEL",
        "task": mdp.environment_id, "exclude_from_independent_test": True,
        "source_npz_sha256": mdp.metadata["source_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "states": mdp.num_states, "pilot_transitions": len(pilot["state"]),
        "posterior_seconds": posterior_seconds, "oehh_seconds": oehh_seconds,
        "risk_profile_seconds": profile_seconds, "total_seconds": time.perf_counter() - start,
        "oehh": result.as_dict(),
        "profile_shapes": {key: list(value.shape) if hasattr(value, "shape") else "scalar"
                           for key, value in profile.items()},
        "warning": "Only one development pilot; no full 108-feature panel or test prediction."},
        indent=2, default=lambda x: float(x)) + "\n", encoding="utf-8")
    print(f"OEH-H BRIDGE dev smoke: {time.perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()

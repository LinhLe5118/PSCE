"""Development-only legacy B2/OEH3/LCOEH4 feature lineage on BRIDGE."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

from benchmarks.bridge.adapter import load_sticky_mdp
from environments.behavior_policies import structured_random_policy
from scripts.data.generate_offline_datasets import collect_trajectories

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from lcoeh_v4 import LC_OEH_V4_FEATURES, estimate_lcoeh_v4  # noqa: E402
from oeh_risk_v3 import estimate_oeh_risk_v3  # noqa: E402
from stochastic_features import aggregate_oeh_risk_v3_profile, extract_stochastic_features  # noqa: E402

OUT = ROOT / "artifacts/week8/bridge_feature_smoke_dev_2026-10-07.json"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    mdp = load_sticky_mdp("MiniGrid-LavaGapS5-v0")
    beta = structured_random_policy(mdp, epsilon=.2, seed=20261007)
    pilot = collect_trajectories(mdp, beta, seed=2026100701,
                                 size=10 * mdp.num_states * mdp.num_actions, horizon=100)
    prefixes = {b: {key: value[:b * mdp.num_states * mdp.num_actions]
                    for key, value in pilot.items()} for b in (2, 5, 10)}
    names = json.loads((ROOT / "docs/protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0.json")
                       .read_text(encoding="utf-8"))["feature_sets"]
    start = time.perf_counter()
    base = extract_stochastic_features(pilot, num_states=mdp.num_states,
                                       num_actions=mdp.num_actions, gamma=mdp.gamma,
                                       bootstrap_replicates=200,
                                       seed_key="bridge-dev-LavaGapS5-structured_random")
    base_seconds = time.perf_counter() - start
    risk = estimate_oeh_risk_v3(pilot, mdp.initial_state_distribution,
                                num_states=mdp.num_states, num_actions=mdp.num_actions,
                                gamma=mdp.gamma, k_max=100)
    by_depth = {int(item["depth"]): item for item in risk["records"]}
    profile = [{**item, "risk_at_pilot": by_depth[int(item["depth"])]["risk_at_pilot"],
                "policy_instability": by_depth[int(item["depth"])]["policy_instability"]}
               for item in risk["profile"]]
    oeh = aggregate_oeh_risk_v3_profile(profile)
    risk_seconds = time.perf_counter() - start - base_seconds
    learners = estimate_lcoeh_v4(prefixes, mdp.initial_state_distribution,
                                 num_states=mdp.num_states, num_actions=mdp.num_actions,
                                 gamma=mdp.gamma, seed_key="bridge-dev-LavaGapS5-structured_random")
    learner_seconds = time.perf_counter() - start - base_seconds - risk_seconds
    expected = names["B2_OEH3_LCOEH4"]
    rows = {}
    for algorithm in ("cfqi_tabular", "iql_tabular", "behavior_cloning_tabular"):
        values = {**base, **oeh,
                  **{name: learners[algorithm][name] for name in LC_OEH_V4_FEATURES}}
        missing = set(expected) - set(values)
        if missing or any(not math.isfinite(float(values[name])) for name in expected):
            raise RuntimeError(f"Missing/nonfinite development feature: {algorithm}, {sorted(missing)}")
        rows[algorithm] = {name: float(values[name]) for name in expected}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"status": "DEVELOPMENT_FEATURE_SMOKE_ONLY",
        "task": mdp.environment_id, "exclude_from_independent_test": True,
        "source_npz_sha256": mdp.metadata["source_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "states": mdp.num_states, "pilot_transitions": len(pilot["state"]),
        "feature_columns": len(expected),
        "base_seconds": base_seconds, "legacy_risk_seconds": risk_seconds,
        "learner_context_seconds": learner_seconds, "total_seconds": time.perf_counter() - start,
        "features": rows,
        "warning": "One excluded development group only; no full 108-feature panel or test prediction."},
        indent=2) + "\n", encoding="utf-8")
    print(f"BRIDGE legacy feature smoke: {len(expected)} columns, "
          f"{time.perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()

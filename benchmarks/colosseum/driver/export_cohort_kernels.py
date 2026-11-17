"""Export cohort 4/7 kernels as (T, expected R, rho) for Colosseum's hardness code.

Terminal states restart from rho, matching how the project's logs are
collected (episodes restart), so the measures are finite and comparable.
"""
from __future__ import annotations

import numpy as np

from common import DATA, load_json, use_workspace

use_workspace()
from scripts.analysis.run_unseen_validation import archive_imports  # noqa: E402

archive_imports()
from environments.benchmark_environments import make_benchmark_mdp  # noqa: E402

for name in ("first60", "second60", "first120", "second120", "third120", "c4", "c7"):
    target = DATA / "kernels" / name
    target.mkdir(parents=True, exist_ok=True)
    for item in load_json(DATA / name / "manifest.json")["instances"]:
        mdp = make_benchmark_mdp(type("Spec", (), item)())
        T = np.array(mdp.P, dtype=float)
        rho = np.asarray(mdp.initial_state_distribution, dtype=float)
        for s in mdp.terminal_states:
            T[s] = rho
        np.savez(target / f"{item['instance_id']}.npz", T=T, R=np.asarray(mdp.expected_rewards, float), rho=rho)
    print(name, "exported")

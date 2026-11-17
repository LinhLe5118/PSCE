"""Development-only OEH-H panel: oracle and data-only offline effective horizon.

For each MDP-behavior group of an opened cohort, compute OEH-H on the true MDP
with the true behavior probabilities (oracle diagnostic, comparable to oracle
EH) and on the posterior-mean model of the x10 pilot (data-only). Rank
agreement with the empirical requirement is descriptive development evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/oeh_final_development_2026-10-01"


def _posterior_mean_rows(counts, mixture, concentration, base):
    """Posterior mean of Dirichlet rows under a single or mixture prior."""
    from oeh import posterior as post

    if mixture is None:
        rows = counts + post._prior(concentration, base)
        return rows / rows.sum(-1, keepdims=True)
    mean = np.zeros(counts.shape)
    for k, log_alpha in enumerate(mixture["log_alpha"]):
        alpha = float(np.exp(log_alpha)) if np.isfinite(log_alpha) else (1e12 if log_alpha > 0 else 0.0)
        rows = counts + alpha * base
        total = rows.sum(-1, keepdims=True)
        component = np.divide(rows, total, out=np.tile(base, (len(rows), 1)), where=total > 0)
        mean += mixture["responsibilities"][:, k:k + 1] * component
    return mean


def posterior_mean_model(arrays: dict, rho, S: int, A: int, gamma: float, **structure):
    """Posterior-mean transition, reward and behavior of the OEH-B posterior."""
    from oeh import posterior as post

    stats = post.pilot_statistics(arrays, rho, num_states=S, num_actions=A, gamma=gamma)
    model = post.fit_posterior(stats, **structure)
    P = _posterior_mean_rows(stats.transition_counts.reshape(-1, S), model.transition_mixture,
                             model.transition_concentration, model.transition_base).reshape(S, A, S)
    beta = _posterior_mean_rows(stats.action_counts, model.behavior_mixture,
                                model.behavior_concentration, np.full(A, 1.0 / A))
    P[stats.terminal] = 0.0
    P[stats.terminal, :, stats.terminal] = 1.0
    # Observed rewards are deterministic; unobserved keys take their prior mean.
    rate = 0.0 if model.reward_prior == "unseen_zero" else model.reward_rate[0] / sum(model.reward_rate)
    R = np.full((S, A, S), rate * float(np.mean(model.reward_values)))
    for s, a, n, value in zip(arrays["state"], arrays["action"], arrays["next_state"], arrays["reward"]):
        R[s, a, n] = value
    if stats.reward_key == "next_state":
        for (n,), value in stats.reward_table.items():
            R[:, :, n] = value
    elif stats.reward_key == "state_action":
        for (s, a), value in stats.reward_table.items():
            R[s, a, :] = value
    R[stats.terminal] = 0.0
    return P, R, beta, stats.terminal


def run_group(task: dict) -> dict:
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    from environments.behavior_policies import behavior_policy_suite
    from environments.benchmark_environments import make_benchmark_mdp
    from oeh import horizon as H

    instance, row = task["instance"], task["row"]
    mdp = make_benchmark_mdp(type("Spec", (), instance)())
    terminal = np.zeros(mdp.num_states, dtype=bool)
    terminal[list(mdp.terminal_states)] = True
    policies = behavior_policy_suite(mdp, seed=int(instance["seed"]) + 3_000_000,
                                     variant_index=int(instance["ordinal"]))
    beta = next(item.probabilities for item in policies if item.family == row["policy_family"])
    oracle = H.offline_effective_horizon(mdp.P, mdp.R, beta, mdp.initial_state_distribution, mdp.gamma,
                                         terminal=terminal, tau=0.95, horizon=100)
    with np.load(Path(task["dataset"]) / row["file"], allow_pickle=False) as loaded:
        arrays = {name: loaded[name][:int(row["size"])] for name in loaded.files}
    P, R, b, term = posterior_mean_model(arrays, mdp.initial_state_distribution, mdp.num_states,
                                         mdp.num_actions, mdp.gamma, **task["structure"])
    data = H.offline_effective_horizon(P, R, b, mdp.initial_state_distribution, mdp.gamma,
                                       terminal=term, tau=0.95, horizon=100)
    eh, eh_k = H.stationary_eh(mdp.P, mdp.R, mdp.gamma, max_depth=100, terminal=terminal)
    clean = lambda x: None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x  # noqa: E731
    return {"mdp_id": instance["instance_id"], "policy_family": row["policy_family"],
            **{f"oracle_{k}": clean(v) for k, v in oracle.as_dict().items()},
            **{f"data_{k}": clean(v) for k, v in data.as_dict().items()},
            "oracle_stationary_eh": clean(eh), "oracle_stationary_eh_k": eh_k}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("name")
    parser.add_argument("--reward-prior", default="exchangeable")
    parser.add_argument("--transition-base", default="uniform")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    dataset = ROOT / args.dataset
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    rows = sorted((row for row in manifest["datasets"]
                   if row.get("stream") == "pilot" and int(row.get("size_multiplier", -1)) == 10),
                  key=lambda row: (row["instance_id"], row["policy_family"]))
    structure = {"reward_prior": args.reward_prior, "transition_base": args.transition_base}
    tasks = [{"instance": instances[row["instance_id"]], "row": row, "dataset": str(dataset),
              "structure": structure} for row in rows]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_group, tasks, chunksize=4))
    import pandas as pd
    out = OUTPUT / f"horizon_{args.name}.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"Wrote {len(results)} OEH-H rows to {out}", flush=True)


if __name__ == "__main__":
    main()

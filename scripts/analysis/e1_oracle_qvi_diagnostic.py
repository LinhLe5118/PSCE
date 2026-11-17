"""Reconstruct and audit the sealed E1 MDPs; compute oracle diagnostics.

This does not regenerate trajectories or rerun training. The quantities below
are explicitly discounted infinite-horizon QVI/coverage *adaptations*, not the
published finite-horizon EH/EPW/covering-length bounds. Original E1 has no
stored transition-kernel hash, so source and manifest agreement are the
strongest available identity check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from scripts.analysis.compare_difficulty import sha256


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "tmp/oeh_development_2026-09-30"
DATASET = ROOT / "datasets/week6_e1_oehv4e_confirmatory_v1_1"
OUTPUT = ROOT / "artifacts/e1_oracle_qvi_posthoc_2026-09-30"


def archived_source_hash(archive: Path, files: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update((archive / "protocol_config.json").read_bytes())
    for name in files:
        content = (archive / name).read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(content).to_bytes(8, byteorder="little"))
        digest.update(content)
    return digest.hexdigest()


def validate_record(actual: dict, expected: dict) -> None:
    import numpy as np

    if set(actual) != set(expected):
        raise RuntimeError(f"Manifest fields differ: {expected['instance_id']}")
    for key in actual:
        left, right = actual[key], expected[key]
        if key in ("dynamic_range", "optimal_return", "random_return", "reward_min", "reward_max", "gamma"):
            if not np.isclose(left, right, atol=1e-10, rtol=0):
                raise RuntimeError(f"E1 record mismatch: {expected['instance_id']} {key}")
        elif left != right:
            raise RuntimeError(f"E1 record mismatch: {expected['instance_id']} {key}")


def diagnostics(mdp) -> dict[str, float | int]:
    import numpy as np
    from exact_solver import evaluate_policy, value_iteration

    s, a = mdp.num_states, mdp.num_actions
    qstar = value_iteration(mdp).q_values
    optimal = qstar.max(axis=1)
    nonterminal = np.ones(s, dtype=bool)
    nonterminal[list(mdp.terminal_states)] = False
    # Reachability is structural, not pilot-data inferred.
    reachable = mdp.initial_state_distribution > 0
    for _ in range(s):
        expanded = reachable | (mdp.P[reachable].sum(axis=(0, 1)) > 0)
        if np.array_equal(expanded, reachable):
            break
        reachable = expanded
    relevant = reachable & nonterminal
    if not relevant.any():
        raise RuntimeError("No reachable nonterminal state")

    # Q^1 is Q of the uniform random policy; then Bellman optimality backup.
    random_policy = np.full((s, a), 1 / a)
    q = evaluate_policy(mdp, random_policy).q_values
    depth = 101
    for k in range(1, 101):
        greedy = q >= q.max(axis=1, keepdims=True) - 1e-10
        bad = greedy & (qstar < optimal[:, None] - 1e-10)
        if not bad[relevant].any():
            depth = k
            break
        q = mdp.expected_rewards + mdp.gamma * np.einsum("san,n->sa", mdp.P, q.max(axis=1))

    positive_gaps = (optimal[:, None] - qstar)[relevant]
    positive_gaps = positive_gaps[positive_gaps > 1e-10]
    min_gap = float(positive_gaps.min()) if len(positive_gaps) else float("nan")

    # Random expected visitation over the protocol's 100-step collection
    # horizon. This is an occupancy diagnostic, not Liu--Brunskill's L.
    mass = mdp.initial_state_distribution.copy()
    visits = np.zeros(s)
    transition = mdp.P.mean(axis=1)
    for _ in range(100):
        visits += mass
        mass = mass @ transition
    sa_visits = (visits[relevant] / a)
    min_visits = float(sa_visits.min())
    return {
        "oracle_qvi_depth_capped_100": int(depth),
        "oracle_qvi_censored": int(depth > 100),
        "oracle_min_positive_qstar_gap": min_gap,
        "random_min_expected_sa_visits_h100": min_visits,
        "random_log_inverse_expected_sa_visits_h100": float(-np.log(max(min_visits, 1e-300))),
        "reachable_nonterminal_states": int(relevant.sum()),
    }


def finite_horizon_online_bounds(mdp, horizon: int = 100) -> dict[str, float | int]:
    """E1-discounted versions of published finite-horizon oracle definitions.

    QVI/EPW definitions follow Laidlaw et al. (2023, 2024), with E1's gamma
    included in each backup. Covering length is their occupancy *upper bound*,
    not the exact episode-count quantile L. All states and timesteps count.
    """
    import numpy as np

    s, a = mdp.num_states, mdp.num_actions
    reward = mdp.expected_rewards
    qstar = np.empty((horizon, s, a))
    qrand = np.empty_like(qstar)
    continuation_opt = np.zeros(s)
    continuation_rand = np.zeros(s)
    for t in range(horizon - 1, -1, -1):
        qstar[t] = reward + mdp.gamma * np.einsum("san,n->sa", mdp.P, continuation_opt)
        qrand[t] = reward + mdp.gamma * np.einsum("san,n->sa", mdp.P, continuation_rand)
        continuation_opt = qstar[t].max(axis=1)
        continuation_rand = qrand[t].mean(axis=1)
    optimal_actions = qstar >= qstar.max(axis=2, keepdims=True) - 1e-10

    def backup(values):
        next_values = np.zeros((horizon, s))
        next_values[:-1] = values[1:].max(axis=2)
        return reward[None, :, :] + mdp.gamma * np.einsum("san,tn->tsa", mdp.P, next_values)

    def solvable(values):
        greedy = values >= values.max(axis=2, keepdims=True) - 1e-10
        return not np.any(greedy & ~optimal_actions), greedy

    q_eh = qrand
    q_epw = np.broadcast_to(reward, (horizon, s, a)).copy()
    min_qvi = horizon + 1
    epw = horizon + 1
    best_h = float("inf")
    best_k = horizon + 1
    best_gap = float("nan")
    max_reward_range = float((reward.max() - reward.min()) / (1 - mdp.gamma))
    penalty_floor = -2 * np.log(max(max_reward_range, 1e-300)) / np.log(a)
    eh_finished = False
    for k in range(1, horizon + 1):
        if epw > horizon and solvable(q_epw)[0]:
            epw = k
        if not eh_finished:
            eh_solvable, greedy = solvable(q_eh)
            if eh_solvable:
                if min_qvi > horizon:
                    min_qvi = k
                # Delta_k is the minimum greedy/non-greedy action separation.
                second = np.where(~greedy, q_eh, -np.inf).max(axis=2)
                gaps = (q_eh.max(axis=2) - second)[np.isfinite(second)]
                if len(gaps):
                    gap = float(gaps.min())
                    if gap > 0:
                        h = float(k - 2 * np.log(gap) / np.log(a))
                        if h < best_h:
                            best_h, best_k, best_gap = h, k, gap
        # Once even the largest possible future gap cannot improve H, stop
        # EH backups; EPW still continues until its own solvability depth.
        eh_finished = eh_finished or (best_h < float("inf") and k + 1 + penalty_floor >= best_h)
        if epw <= horizon and eh_finished:
            break
        if not eh_finished:
            q_eh = backup(q_eh)
        if epw > horizon:
            q_epw = backup(q_epw)

    state_mass = mdp.initial_state_distribution.copy()
    max_state_mass = np.zeros(s)
    uniform_transition = mdp.P.mean(axis=1)
    for _ in range(horizon):
        max_state_mass = np.maximum(max_state_mass, state_mass)
        state_mass = state_mass @ uniform_transition
    min_max_sa_occupancy = float(max_state_mass.min() / a)
    log_cover_bound = (
        float(np.log(np.ceil(np.log(2 * s * a) / min_max_sa_occupancy)))
        if min_max_sa_occupancy > 0 else float("inf")
    )
    return {
        "stochastic_eh_min_qvi_k_h100": int(min_qvi),
        "stochastic_eh_hbar_h100": float(best_h),
        "stochastic_eh_best_k_h100": int(best_k),
        "stochastic_eh_delta_h100": float(best_gap),
        "epw_window_h100": int(epw),
        "log_eh_online_bound_t2_aH": float(2 * np.log(horizon) + best_h * np.log(a)),
        "log_epw_online_bound_t2_aW": float(2 * np.log(horizon) + epw * np.log(a)),
        "log_covering_length_upper_bound": log_cover_bound,
        "log_cover_online_bound_tL": float(np.log(horizon) + log_cover_bound),
        "log_ucb_sat_reference": float(np.log(s * a * horizon)),
        "log_worst_case_reference": float(np.log(horizon) + np.log(float((a ** horizon + 1) // 2))),
        "min_max_random_sa_occupancy_h100": min_max_sa_occupancy,
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    manifest_path = DATASET / "manifest.json"
    seal_path = ROOT / "artifacts/week6/E1_INPUT_SEAL_2026-09-26.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(manifest_path) != seal["manifest_sha256"] or len(manifest["instances"]) != 120:
        raise RuntimeError("Original E1 manifest seal failed")
    source_hash = archived_source_hash(ARCHIVE, manifest["generator_provenance"]["source_files"])
    if source_hash != manifest["generator_provenance"]["source_sha256"]:
        raise RuntimeError("Archived generator source does not match E1 seal")
    generator_file = ARCHIVE / "scripts/data/generate_week6_e1_bundle.py"
    if sha256(generator_file) != manifest["e1_generator_sha256"]:
        raise RuntimeError("E1 generator file does not match manifest")

    sys.path.insert(0, str(ARCHIVE))
    import scripts
    scripts.__path__.insert(0, str(ARCHIVE / "scripts"))
    from scripts.data.generate_week6_e1_bundle import registered_evaluation_specs
    from scripts.data import generate_offline_datasets as original

    expected = {record["instance_id"]: record for record in manifest["instances"]}
    rows = []
    for requested in registered_evaluation_specs():
        spec, mdp, rejections = original.resolve_dynamic_range(requested)
        dynamic, best, random = original.dynamic_range(mdp)
        reconstructed = {
            **spec.to_dict(), "requested_seed": requested.seed,
            "dynamic_range_rejections": rejections,
            "num_states": mdp.num_states, "num_actions": mdp.num_actions,
            "gamma": mdp.gamma, "terminal_states": list(mdp.terminal_states),
            "reward_min": float(mdp.R.min()), "reward_max": float(mdp.R.max()),
            "dynamic_range": dynamic, "optimal_return": best,
            "random_return": random, "metadata": original._json_ready(mdp.metadata),
            "dynamics_audit": original.audit_dynamics(mdp, spec.family),
        }
        validate_record(reconstructed, expected[spec.instance_id])
        fingerprint = hashlib.sha256()
        for array in (mdp.P, mdp.R, mdp.initial_state_distribution):
            fingerprint.update(array.tobytes())
        fingerprint.update(json.dumps(list(mdp.terminal_states)).encode("ascii"))
        rows.append({"mdp_id": spec.instance_id, "family": spec.family,
                     "panel": "control" if spec.family == "lock_structured" else "primary",
                     "kernel_fingerprint_reconstructed": fingerprint.hexdigest(),
                     **diagnostics(mdp), **finite_horizon_online_bounds(mdp)})
        if len(rows) % 20 == 0:
            print(f"Audited {len(rows)}/120 original E1 MDP recipes", flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 120 or frame.mdp_id.nunique() != 120 or frame.kernel_fingerprint_reconstructed.duplicated().any():
        raise RuntimeError("Incomplete or aliased E1 cohort")

    labels_path = ROOT / "artifacts/difficulty_e1_posthoc_2026-09-30/labels.csv"
    label_manifest = json.loads((labels_path.parent / "manifest.json").read_text(encoding="utf-8"))
    if sha256(labels_path) != label_manifest["output_hashes"]["labels.csv"]:
        raise RuntimeError("E1 label seal mismatch")
    labels = pd.read_csv(labels_path)
    if len(labels) != 1800 or labels.mdp_id.nunique() != 120:
        raise RuntimeError("Incomplete E1 labels")
    outcome = labels.groupby("mdp_id").event.apply(lambda col: float((col < 6).mean())).rename("sustained_success_fraction_by_x50")
    frame = frame.merge(outcome, on="mdp_id", validate="one_to_one")
    scores = []
    for panel, part in frame.groupby("panel"):
        for metric, polarity in (("stochastic_eh_hbar_h100", -1),
                                 ("stochastic_eh_min_qvi_k_h100", -1),
                                 ("epw_window_h100", -1),
                                 ("log_cover_online_bound_tL", -1),
                                 ("log_ucb_sat_reference", -1),
                                 ("log_worst_case_reference", -1),
                                 ("oracle_qvi_depth_capped_100", -1),
                                 ("oracle_min_positive_qstar_gap", 1),
                                 ("random_log_inverse_expected_sa_visits_h100", -1)):
            clean = part[[metric, "sustained_success_fraction_by_x50"]].dropna()
            # Infinite covering bounds are genuine E1 outcomes (unreachable
            # cells), so retain them as tied hardest ranks. Only NaN is
            # undefined and excluded.
            clean = clean[~np.isnan(clean[metric])]
            if clean[metric].nunique() < 2 or clean.sustained_success_fraction_by_x50.nunique() < 2:
                statistic, pvalue = float("nan"), float("nan")
            else:
                correlation = spearmanr(polarity * clean[metric], clean.sustained_success_fraction_by_x50)
                statistic, pvalue = float(correlation.statistic), float(correlation.pvalue)
            scores.append({"panel": panel, "metric": metric, "n_mdps": len(clean),
                           "spearman_with_success_fraction": statistic,
                           "p_value_descriptive_only": pvalue})
            if metric == "log_cover_online_bound_tL" and len(clean) > np.isfinite(clean[metric]).sum():
                finite = clean[np.isfinite(clean[metric])]
                sensitivity = spearmanr(-finite[metric], finite.sustained_success_fraction_by_x50)
                scores.append({"panel": panel, "metric": "log_cover_online_bound_tL_finite_only_sensitivity",
                               "n_mdps": len(finite),
                               "spearman_with_success_fraction": float(sensitivity.statistic),
                               "p_value_descriptive_only": float(sensitivity.pvalue)})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "oracle_diagnostics.csv", index=False)
    pd.DataFrame(scores).to_csv(OUTPUT / "rank_comparison.csv", index=False)
    report = {
        "status": "posthoc_original_E1_adaptively_reused_not_confirmation",
        "identity": "120 regenerated in memory; sealed manifest, all instance metadata and oracle audits match; original kernel byte hashes unavailable",
        "source_hash": source_hash, "manifest_sha256": sha256(manifest_path),
        "e1_generator_sha256": sha256(generator_file), "labels_sha256": sha256(labels_path),
        "mdps": 120, "stochastic_primary": 84, "control": 36,
        "information_access": "true P,R,initial distribution, terminal states (oracle only)",
        "metric_status": "Stochastic EH and EPW use the published finite-horizon QVI definitions with E1 gamma=0.95 and T=100; covering length is the published occupancy upper bound, not exact L. All are oracle online-RL descriptors, not calibrated offline CFQI/IQL/BC requirements. Additional discounted-infinite-horizon QVI/occupancy adaptations are labeled separately.",
        "stochastic_eh_definition": "finite T=100; Q^1=Q_random; every Q^k-greedy action must be Q*-optimal at all states/timesteps; min over k of k+log_A(1/Delta_k^2); gamma=0.95; ties at 1e-10",
        "infinite_qvi_adaptation": "min k<=100 such that every Q^k-greedy action is Q*-optimal at structurally reachable nonterminal states; Q^1=Q_random; gamma=0.95",
        "occupancy_adaptation": "minimum 100-step expected visits of a state-action pair under uniform random policy across structurally reachable nonterminal states",
        "covering_bound_rank_policy": "infinite bounds retained as tied hardest ranks in the full 120-MDP comparison; finite-only rows are separately labeled sensitivity analyses",
        "outcome": "fraction of 15 behavior-policy/learner groups per MDP with sustained success by x50",
        "published_reference": "https://github.com/cassidylaidlaw/effective-horizon",
        "script_sha256": sha256(Path(__file__)),
        "output_hashes": {name: sha256(OUTPUT / name) for name in ("oracle_diagnostics.csv", "rank_comparison.csv")},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(scores).to_string(index=False))


if __name__ == "__main__":
    main()

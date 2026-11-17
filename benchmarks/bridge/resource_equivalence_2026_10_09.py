"""Audit BRIDGE task coverage and compute proxies against the internal 120 MDPs.

Reads only frozen protocol/pilot metadata, never learner outcomes. A state-action
budget proxy is more relevant here than matching the raw number of MDP names.
"""

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "bridge" / "resource_equivalence_2026-10-09.json"
INTERNAL = ROOT / "artifacts" / "oeh_confirmatory_unseen_120_2026-10-04" / "pilot_features.csv"
PANELS = {
    "opened_minigrid_14_exploratory": ROOT / "datasets/bridge_v1_1_1/offline_pilot_preoutcome_v1/manifest.json",
    "fresh_atari_5_completed": ROOT / "datasets/bridge_v1_1_1/offline_pilot_fresh_2026-10-08/manifest.json",
    "expanded_5_running": ROOT / "datasets/bridge_v1_1_1/offline_pilot_expanded_2026-10-08/manifest.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


internal_sa = {}
with INTERNAL.open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        sa = round(math.exp(float(row["log_state_action_count"])))
        if not math.isclose(math.log(sa), float(row["log_state_action_count"]), abs_tol=1e-10):
            raise ValueError(f"Noninteger state-action count: {row['mdp_id']}")
        previous = internal_sa.setdefault(row["mdp_id"], sa)
        if previous != sa:
            raise ValueError(f"Inconsistent state-action count: {row['mdp_id']}")
if len(internal_sa) != 120:
    raise ValueError("Internal panel must contain 120 distinct MDPs")

summary = {}
all_bridge = set()
for name, path in PANELS.items():
    manifest = json.loads(path.read_text(encoding="utf-8"))
    groups = manifest["pilot_groups"]
    task_sa = {}
    policies = {}
    for row in groups:
        task = row["mdp_id"]
        sa = int(row["num_states"]) * int(row["num_actions"])
        if task in task_sa and task_sa[task] != sa:
            raise ValueError(f"Inconsistent BRIDGE dimensions: {task}")
        task_sa[task] = sa
        policies.setdefault(task, set()).add(row["policy_family"])
    if len(groups) != len(task_sa) * 5 or any(len(p) != 5 for p in policies.values()):
        raise ValueError(f"Incomplete five-behavior BRIDGE panel: {name}")
    if all_bridge.intersection(task_sa):
        raise ValueError(f"Overlapping BRIDGE task names in {name}")
    all_bridge.update(task_sa)
    total_sa = sum(task_sa.values())
    summary[name] = {
        "mdps": len(task_sa), "pilot_groups": len(groups),
        "state_action_sum": total_sa,
        "pilot_transitions_x10_all_behaviors": 50 * total_sa,
        "training_stream_transitions_x50_10_seeds_all_behaviors": 2500 * total_sa,
        "learner_budget_seed_jobs": len(task_sa) * 5 * 3 * 6 * 10,
        "task_state_action": task_sa,
        "pilot_manifest_sha256": sha256(path),
    }

bridge_sa = sum(row["state_action_sum"] for row in summary.values())
internal_total = sum(internal_sa.values())
report = {
    "status": "METADATA_RESOURCE_EQUIVALENCE_ONLY",
    "metric": "sum of S*A across MDPs; all panels use five behaviors, ten label seeds and an x50 maximum prefix",
    "internal_120": {
        "mdps": 120, "state_action_sum": internal_total,
        "pilot_transitions_x10_all_behaviors": 50 * internal_total,
        "training_stream_transitions_x50_10_seeds_all_behaviors": 2500 * internal_total,
        "learner_budget_seed_jobs": 120 * 5 * 3 * 6 * 10,
        "feature_csv_sha256": sha256(INTERNAL),
    },
    "bridge_panels": summary,
    "bridge_distinct_mdps": len(all_bridge),
    "bridge_state_action_sum": bridge_sa,
    "bridge_to_internal_state_action_budget_ratio": bridge_sa / internal_total,
    "interpretation": (
        "This is a transition-budget proxy, not measured CPU time, statistical equivalence, "
        "or a claim that opened labels are independent confirmation. Published BRIDGE MDPs "
        "supply environments; this project collected pilot logs and learner labels."
    ),
    "no_outcome_file_read": True,
}
OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({key: report[key] for key in (
    "bridge_distinct_mdps", "bridge_state_action_sum",
    "bridge_to_internal_state_action_budget_ratio")}, indent=2))

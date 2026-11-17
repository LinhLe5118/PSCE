"""Freeze the new BRIDGE transfer protocol before any learner test labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PANEL = Path(__file__).resolve().parent / "fresh_panel_2026-10-08"
ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = PANEL / "protocol.json"
    if output.exists():
        raise FileExistsError(output)
    gate = json.loads((PANEL / "reward_gate.json").read_text(encoding="utf-8"))
    adapter = json.loads((PANEL / "adapter_preflight.json").read_text(encoding="utf-8"))
    parity = json.loads((ROOT / "artifacts/week8/bridge_fast_backend_parity_dev_2026-10-08.json")
                        .read_text(encoding="utf-8"))
    if (len(gate["eligible_tasks"]) != 5 or adapter["status"] != "PRELABEL_ADAPTER_PASS"
            or parity["status"] != "PASS_DEVELOPMENT_PARITY_NO_TEST_LABELS"):
        raise RuntimeError("Prelabel feasibility gates incomplete")
    structure = json.loads((ROOT / "artifacts/week8/bridge_transfer_range_addendum_2026-10-07.json")
                           .read_text(encoding="utf-8"))["posterior_structure"]
    document = {
        "status": "FRESH_BRIDGE_PROTOCOL_FROZEN_BEFORE_LABELS",
        "evaluation_tasks": gate["eligible_tasks"],
        "excluded_candidates": [r["mdp_name"] for r in gate["tasks"] if r["status"] != "ELIGIBLE_NONNEGATIVE"],
        "environment_source": "Published BRIDGE v1.1.1 base Atari MDPs, sticky probability 0.25; group-created offline trajectories and labels",
        "state": "BFS-reachable (base_state, previous_executed_action), terminal absorbing; reset sentinel",
        "episode_horizon": {r["mdp_name"]: r["horizon"] for r in gate["tasks"] if r["status"] == "ELIGIBLE_NONNEGATIVE"},
        "reward": "Divide raw public MDP rewards by each task's positive published maximum; no shift; same normalized-return endpoint",
        "behavior_families_per_task": 5,
        "pilot": "Independent x10*S*A transition stream per behavior; fixed hash-derived seed; generated before labels",
        "label_training": "Independent x50*S*A natural streams with nested x1,x2,x5,x10,x20,x50 prefixes, ten seeds, BC/CFQI/IQL; 8/10 sustained suffix",
        "learner_names": ["BC", "CFQI", "IQL"],
        "requirement_budgets": [1, 2, 5, 10, 20, 50],
        "requirement_categories": ["x1", "x2", "x5", "x10", "x20", "x50", ">x50"],
        "tau": 0.95, "gamma": 0.99,
        "posterior": {"draws": 64, "seeds": 10, "minimum_dynamic_range": 0.2,
                      "structure": structure, "simulated_training_horizon": "task-specific published H",
                      "exact_return_horizon": "task-specific published H"},
        "methods_primary": ["oeh_full_108"],
        "controlled_internal_diagnostic": ["B2_matched"],
        "published_original_secondary_rank_only": ["EH_true_MDP_privileged", "ERI_when_applicable", "TQ", "SACo"],
        "comparison_rule": "Same task-behavior-learner rows, no new tuning from these labels; original published scores rank against E_l only and do not output its seven-class law",
        "primary_metrics": ["Brier", "log_loss", "paired_task_delta", "task_cluster_bootstrap_95pct_CI", "censoring", "runtime"],
        "bootstrap": {"unit": "task", "replicates": 10000, "seed": 20261008},
        "seal_gate": "All 75 full OEH forecasts (5 tasks x 5 behaviors x 3 learners), internal diagnostic and original scores must be hashed before any label run",
        "test_labels_read": False,
        "source_sha256": {path.name: sha256(path) for path in (
            PANEL / "manifest.json", PANEL / "candidate_registry.csv", PANEL / "download_manifest.json",
            PANEL / "reward_gate.json", PANEL / "adapter_preflight.json",
            ROOT / "artifacts/week8/bridge_fast_backend_parity_dev_2026-10-08.json",
            ROOT / "artifacts/week8/bridge_transfer_range_addendum_2026-10-07.json")},
        "limits": ["Five reward-compatible tasks only; small cross-family panel.",
                   "These are published MDPs, not published offline logs/requirement labels.",
                   "Original EH/ERI/TQ/SACo do not yield the seven-class learner-specific requirement distribution.",
                   "A numerical optimization changes only the fresh-run implementation after development parity tests; it does not revise historical sealed outputs."],
    }
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print("Frozen fresh BRIDGE protocol for five eligible tasks; zero learner labels read", flush=True)


if __name__ == "__main__":
    main()

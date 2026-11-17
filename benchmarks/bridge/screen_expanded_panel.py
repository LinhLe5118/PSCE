"""Freeze every resource-eligible, previously unlabelled BRIDGE sticky MDP.

The screen reads only published structural metadata and the published reward
minimum. It does not read author performance, OEH scores, or learner labels.
The output is a new namespace; the five-task sealed panel is never modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "expanded_panel_2026-10-08"
MAX_STICKY_STATES = 3000
MAX_DENSE_MIB = 512.0
MAX_HORIZON = 100
FIELDS = ("mdp_name", "origin", "base_states", "sticky_states", "actions",
          "horizon", "dense_p_r_mib", "pilot_x10_transitions",
          "x50_transitions_per_seed")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_metadata(filename: str, expected: dict) -> list[dict[str, str]]:
    path = HERE / "metadata" / filename
    if hashlib.md5(path.read_bytes()).hexdigest() != expected[filename]:
        raise ValueError(f"Published metadata checksum changed: {filename}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to replace frozen screen: {OUT}")
    source_file = HERE / "source_manifest.json"
    source = json.loads(source_file.read_text(encoding="utf-8"))
    base = {r["mdp_name"]: r for r in verified_metadata("summary.csv", source["source_csv_md5"])
            if r["type"] == "base"}
    sticky = verified_metadata("summary_sticky.csv", source["source_csv_md5"])
    if len(base) != 155 or len(sticky) != 155:
        raise ValueError("Expected all 155 published BRIDGE MDPs")
    old_file = ROOT / "artifacts/week8/bridge_transfer_preflight_freeze_2026-10-07.json"
    old = json.loads(old_file.read_text(encoding="utf-8"))
    fresh_file = HERE / "fresh_panel_2026-10-08/protocol.json"
    fresh = json.loads(fresh_file.read_text(encoding="utf-8"))
    opened = set(old["development_tasks"]) | set(old["candidate_test_tasks"]) | set(fresh["evaluation_tasks"])
    inventory, selected = [], []
    for r in sorted(sticky, key=lambda item: item["mdp_name"]):
        name = r["mdp_name"]
        b = base.get(name)
        if b is None or int(r["num_actions"]) != int(b["num_actions"]) or int(r["horizon"]) != int(b["horizon"]):
            raise ValueError(f"Missing or conflicting base metadata for {name}")
        states, actions, horizon = int(r["num_states"]), int(r["num_actions"]), int(r["horizon"])
        dense_mib = 16 * states * states * actions / (1024 * 1024)
        reward_min = float(b["reward_min"])
        if name in opened:
            reason = "PRIOR_OUTCOME_OPENED"
        elif states > MAX_STICKY_STATES or dense_mib > MAX_DENSE_MIB or horizon > MAX_HORIZON:
            reason = "RESOURCE_CAP"
        elif reward_min < 0:
            reason = "NEGATIVE_REWARD_OUTSIDE_CURRENT_POSTERIOR"
        else:
            reason = "SELECTED_PRELABEL"
        inventory.append({"mdp_name": name, "origin": r["origin"],
                          "sticky_states": states, "actions": actions,
                          "horizon": horizon, "dense_p_r_mib": round(dense_mib, 2),
                          "published_reward_min": reward_min,
                          "screen_decision": reason})
        if reason == "SELECTED_PRELABEL":
            selected.append({"mdp_name": name, "origin": r["origin"],
                             "base_states": int(b["num_states_consolidated"]),
                             "sticky_states": states, "actions": actions,
                             "horizon": horizon, "dense_p_r_mib": round(dense_mib, 2),
                             "pilot_x10_transitions": 10 * states * actions,
                             "x50_transitions_per_seed": 50 * states * actions})
    if not selected or len({r["mdp_name"] for r in selected}) != len(selected):
        raise ValueError("No unique resource-eligible unseen MDPs")
    OUT.mkdir(parents=True)
    for filename, rows, fields in (
        ("inventory_155.csv", inventory, tuple(inventory[0])),
        ("candidate_registry.csv", selected, FIELDS),
    ):
        with (OUT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "status": "EXPANDED_BRIDGE_METADATA_SCREEN_FROZEN_BEFORE_NEW_LABELS",
        "rule": (f"all 155 published sticky MDPs; exclude prior outcome-opened tasks; "
                 f"S<={MAX_STICKY_STATES}; dense P/R<={MAX_DENSE_MIB:g} MiB; "
                 f"H<={MAX_HORIZON}; published reward minimum >= 0"),
        "source": source["source_record"],
        "task_names": [r["mdp_name"] for r in selected],
        "task_count": len(selected),
        "counts_by_decision": {reason: sum(row["screen_decision"] == reason for row in inventory)
                               for reason in sorted({row["screen_decision"] for row in inventory})},
        "source_sha256": {"summary.csv": sha256(HERE / "metadata/summary.csv"),
                          "summary_sticky.csv": sha256(HERE / "metadata/summary_sticky.csv"),
                          "old_split": sha256(old_file), "fresh_protocol": sha256(fresh_file),
                          "screen_script": sha256(Path(__file__)),
                          "inventory": sha256(OUT / "inventory_155.csv"),
                          "candidate_registry": sha256(OUT / "candidate_registry.csv")},
        "test_labels_read": False,
        "note": "Metadata screen only; NPZ/reward/adaptor/runtime preflight may still exclude a task."
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": manifest["task_names"], "counts": manifest["counts_by_decision"]}, indent=2))


if __name__ == "__main__":
    main()

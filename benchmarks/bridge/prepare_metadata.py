"""Verify published BRIDGE metadata and screen MiniGrid tasks without outcomes.

This script does not read reward, EH, or learner-performance columns. Its output
is a resource shortlist, not an accepted offline benchmark or a test result.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
METADATA = HERE / "metadata"
EXPECTED_MD5 = {
    "summary.csv": "79259199613118464d36aad8d9b5867d",
    "summary_sticky.csv": "eecc01108e0509c43b76c906883c7399",
}
MAX_STICKY_STATES = 1200
FIELDS = (
    "mdp_name", "base_states", "sticky_states", "actions", "horizon",
    "dense_P_R_mib", "pilot_x10_transitions", "x50_transitions_per_seed",
    "resource_screen", "adapter_status", "offline_log_status",
)


def verify_source() -> None:
    for name, expected in EXPECTED_MD5.items():
        actual = hashlib.md5((METADATA / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{name}: MD5 {actual} differs from Zenodo {expected}")


def read_csv(name: str) -> list[dict[str, str]]:
    with (METADATA / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_registry() -> list[dict[str, object]]:
    base = {
        row["mdp_name"]: row for row in read_csv("summary.csv")
        if row["origin"] == "minigrid" and row["type"] == "base"
    }
    sticky = {
        row["mdp_name"]: row for row in read_csv("summary_sticky.csv")
        if row["origin"] == "minigrid"
    }
    if len(base) != 33 or set(base) != set(sticky):
        raise ValueError("Expected the 33 published, unshaped MiniGrid MDPs in both tables")

    registry = []
    for name in sorted(base):
        old, new = base[name], sticky[name]
        states, actions, horizon = (int(new[key]) for key in
                                    ("num_states", "num_actions", "horizon"))
        if actions != int(old["num_actions"]) or horizon != int(old["horizon"]):
            raise ValueError(f"Inconsistent action space or horizon: {name}")
        if min(states, actions, horizon) <= 0:
            raise ValueError(f"Invalid metadata: {name}")
        registry.append({
            "mdp_name": name,
            "base_states": int(old["num_states_consolidated"]),
            "sticky_states": states,
            "actions": actions,
            "horizon": horizon,
            # Current TabularMDP keeps both P and R as dense float64 S x A x S.
            "dense_P_R_mib": round(2 * states * states * actions * 8 / 2**20, 2),
            "pilot_x10_transitions": 10 * states * actions,
            "x50_transitions_per_seed": 50 * states * actions,
            "resource_screen": (
                "CANDIDATE_METADATA_ONLY" if states <= MAX_STICKY_STATES
                else "EXCLUDE_DENSE_RESOURCE_CAP"
            ),
            "adapter_status": "NOT_TESTED",
            "offline_log_status": "GROUP_COLLECTION_REQUIRED",
        })
    return registry


def main() -> None:
    verify_source()
    registry = build_registry()
    output = HERE / "candidate_registry.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(registry)
    manifest = {
        "source_record": "https://zenodo.org/records/12556298",
        "source_doi": "10.5281/zenodo.12556298",
        "source_repo": "https://github.com/cassidylaidlaw/effective-horizon",
        "source_dataset_readme": "https://github.com/cassidylaidlaw/effective-horizon/blob/main/data/bridge_dataset/README.md",
        "source_dataset_readme_version": "v1.1.1",
        "source_license": "CC-BY-4.0",
        "source_zip": "bridge_dataset_v1.1.1.zip",
        "source_zip_bytes": 28708843161,
        "source_zip_md5": "7b67fe0bd15ab987114b1cb131178214",
        "zip_index_tail_sha256": "3266c288b7a76c007e35abb289764ddf71e480264f43bdecf18dea000c063187",
        "source_csv_md5": EXPECTED_MD5,
        "screen": "unshaped MiniGrid, sticky 0.25, S <= 1200, no outcome fields",
        "max_sticky_states": MAX_STICKY_STATES,
        "base_minigrid_count": len(registry),
        "candidate_count": sum(r["resource_screen"] == "CANDIDATE_METADATA_ONLY"
                               for r in registry),
        "status": "published third-party MDP benchmark; metadata screen only; offline trajectories and adapter not yet prepared",
    }
    (HERE / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Verified BRIDGE metadata; wrote {len(registry)} rows to candidate_registry.csv")


if __name__ == "__main__":
    main()

"""Check a published BRIDGE base MDP archive before any offline rollout.

Usage: python benchmarks/bridge/preflight_npz.py NAME PATH_TO_consolidated.npz
This checks file structure only. It does not create sticky transitions or labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REGISTRY = Path(__file__).resolve().parent / "candidate_registry.csv"


def inspect(name: str, path: Path) -> dict[str, object]:
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        rows = {row["mdp_name"]: row for row in csv.DictReader(handle)}
    if name not in rows:
        raise ValueError(f"Name absent from frozen metadata registry: {name}")
    row = rows[name]
    if row["resource_screen"] != "CANDIDATE_METADATA_ONLY":
        raise ValueError(f"Name failed the prespecified resource screen: {name}")
    if path.name != "consolidated.npz":
        raise ValueError("Expected the original consolidated.npz filename")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as archive:
        if not {"transitions", "rewards"}.issubset(archive.files):
            raise ValueError("BRIDGE MDP needs transitions and rewards arrays")
        transitions = archive["transitions"]
        rewards = archive["rewards"]
    # BRIDGE's summary count includes one terminal state. In consolidated.npz
    # terminal successors are encoded as -1, so no terminal row is stored.
    shape = (int(row["base_states"]) - 1, int(row["actions"]))
    if transitions.shape != shape or rewards.shape != shape:
        raise ValueError(f"Expected both arrays to have shape {shape}")
    if not np.issubdtype(transitions.dtype, np.integer):
        raise ValueError("Transition IDs must be integers")
    if not np.all((transitions >= -1) & (transitions < shape[0])):
        raise ValueError("Transition ID outside [-1, S-1]")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("Non-finite reward")
    return {
        "mdp_name": name,
        "file_sha256": digest,
        "base_nonterminal_states": shape[0],
        "metadata_states_including_terminal": int(row["base_states"]),
        "actions": shape[1],
        "terminal_edges": int(np.count_nonzero(transitions == -1)),
        "status": "BASE_ARRAY_SCHEMA_PASS_ONLY",
        "remaining": ["sticky-state construction", "time/horizon Markov audit",
                      "reward/terminal equivalence", "gamma and oracle audit",
                      "pilot/training independence", "full learner feasibility"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.name, args.path), indent=2))


if __name__ == "__main__":
    main()

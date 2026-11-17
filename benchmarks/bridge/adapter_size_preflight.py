"""Audit the state-count offset between Markov reset and author Julia layouts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from benchmarks.bridge.adapter import load_sticky_mdp

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/week8/bridge_adapter_size_preflight_v3_2026-10-07.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    registry = list(csv.DictReader((ROOT / "benchmarks/bridge/candidate_registry.csv").open(
        encoding="utf-8", newline="")))
    rows = []
    for item in registry:
        if item["resource_screen"] != "CANDIDATE_METADATA_ONLY":
            continue
        mdp = load_sticky_mdp(item["mdp_name"])
        reported = int(item["sticky_states"])
        actions = int(item["actions"])
        # Author Julia initializes a previous-action slot and special-cases
        # timestep 1; it also counts absorbing terminal/action slots. Our
        # stationary adapter adds one reset sentinel and collapses terminal
        # variants not directly reachable before an episode ends. For these
        # candidates one executed action reaches terminal, yielding A-2.
        expected_offset = actions - 2
        rows.append({"mdp_name": item["mdp_name"], "adapter_states": mdp.num_states,
                     "published_sticky_states": reported,
                     "published_minus_adapter": reported - mdp.num_states,
                     "expected_representation_offset": expected_offset,
                     "matches_expected_offset": reported - mdp.num_states == expected_offset})
        print(f"{item['mdp_name']}: adapter={mdp.num_states}, published={reported}, "
              f"offset={reported-mdp.num_states}", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"status": "EXPECTED_REPRESENTATION_OFFSET"
                                  if all(r["matches_expected_offset"] for r in rows)
                                  else "UNEXPLAINED_STATE_COUNT_OFFSET",
        "explanation": "Author Julia uses a first-step special case and counts absorbing terminal/action slots; adapter uses a reset sentinel and only directly reachable terminal slots. Pilot budget uses adapter num_states consistently.",
        "adapter_source_sha256": hashlib.sha256((ROOT / "benchmarks/bridge/adapter.py").read_bytes()).hexdigest(),
        "mdps": rows}, indent=2) + "\n", encoding="utf-8")
    if len(rows) != 16 or not all(r["matches_expected_offset"] for r in rows):
        raise RuntimeError("Unexpected sticky-state representation offset")


if __name__ == "__main__":
    main()

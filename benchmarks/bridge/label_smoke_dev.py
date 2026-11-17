"""Development-only complete six-budget/ten-seed label protocol smoke."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.bridge.adapter import load_sticky_mdp
from benchmarks.bridge.labels import one_group
from environments.behavior_policies import structured_random_policy

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/week8/bridge_label_smoke_dev_2026-10-07.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    mdp = load_sticky_mdp("MiniGrid-LavaGapS5-v0")
    behavior = structured_random_policy(mdp, epsilon=.2, seed=20261007)
    result = one_group(mdp, behavior)
    result["status"] = "DEVELOPMENT_LABEL_SMOKE_ONLY"
    result["exclude_from_independent_test"] = True
    result["label_source_sha256"] = hashlib.sha256((ROOT / "benchmarks/bridge/labels.py").read_bytes()).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "task": result["task"],
                      "behavior": result["behavior"], "events": result["event"],
                      "success_counts": result["success_counts"],
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

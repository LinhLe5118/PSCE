"""Prelabel exact-reference and positive-scale invariance check."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .finite_horizon import FiniteHorizonEvaluator
from .fresh_adapter import PANEL, load_fresh_sticky_mdp
from .fresh_reward_gate import load_eligible_mdp

OUT = PANEL / "reference_preflight.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    protocol_file = PANEL / "protocol.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    rows = []
    for name in protocol["evaluation_tasks"]:
        raw, normalized = load_fresh_sticky_mdp(name), load_eligible_mdp(name)
        horizon = int(protocol["episode_horizon"][name])
        raw_random, raw_optimal = FiniteHorizonEvaluator(raw, horizon=horizon).reference()
        random, optimal = FiniteHorizonEvaluator(normalized, horizon=horizon).reference()
        scale = float(normalized.metadata["reward_scale_positive"])
        error = max(abs(raw_random / scale - random), abs(raw_optimal / scale - optimal))
        if optimal - random <= 1e-8 or error > 1e-10:
            raise RuntimeError(f"Degenerate or scale-inconsistent reference: {name}")
        rows.append({"mdp_id": name, "horizon": horizon, "reward_scale_positive": scale,
                     "raw_random": raw_random, "raw_optimal": raw_optimal,
                     "normalized_random": random, "normalized_optimal": optimal,
                     "positive_span": optimal - random, "max_scale_error": error})
    OUT.write_text(json.dumps({"status": "PASS_PRELABEL_REFERENCE_NOT_LEARNER_LABELS",
                               "protocol_sha256": sha256(protocol_file),
                               "rows": rows, "test_labels_read": False},
                              indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(rows)} positive task-specific reference spans; max scale error "
          f"{max(r['max_scale_error'] for r in rows):.3g}", flush=True)


if __name__ == "__main__":
    main()

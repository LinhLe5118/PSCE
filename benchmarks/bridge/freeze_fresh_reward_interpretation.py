"""Clarify the scope of positive reward scaling before learner labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PANEL = Path(__file__).resolve().parent / "fresh_panel_2026-10-08"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = PANEL / "reward_interpretation_addendum.json"
    if output.exists():
        raise FileExistsError(output)
    protocol = PANEL / "protocol.json"
    reward_gate = PANEL / "reward_gate.json"
    document = {
        "status": "PRELABEL_REWARD_SCALE_INTERPRETATION_FROZEN",
        "protocol_sha256": sha256(protocol),
        "reward_gate_sha256": sha256(reward_gate),
        "mathematical_invariance": "For any fixed policy pi, multiplying all rewards in one MDP by c>0 multiplies J(pi), J_random and J_opt by c, so (J(pi)-J_random)/(J_opt-J_random) is unchanged.",
        "non_invariance": "A reward-sensitive learner with fixed hyperparameters (notably IQL's inverse temperature) can learn a different policy after scaling. The E_l labels and posterior forecast in this panel therefore target the explicitly reward-normalized BRIDGE adapter, not raw-reward author training results.",
        "comparison_scope": "Every tested offline learner and forecaster uses the same normalized adapter; author TQ/SACo rank scores are scalar secondary analyses. Do not call this a verbatim reproduction of author raw-reward performance.",
        "test_labels_read": False,
    }
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print("Frozen reward-scale interpretation before labels", flush=True)


if __name__ == "__main__":
    main()

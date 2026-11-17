"""Generate Workstream 1's independent positive-control MDP handover.

The buffers deliberately use very few trajectories and stochastic Bernoulli
rewards.  They are separate benchmark-owned candidates, not the verifier's
built-in reference fixture.  Output is deterministic and immediately readable
by ``adaptive_pilot.transitions_from_arrays`` through ``load_candidate``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from positive_control import stochastic_chain_control_buffer  # noqa: E402


DESIGNS = (
    {
        "label": "benchmark_pc_short_chain",
        "chain_length": 3,
        "repeats": 2,
        "advance_probability": 0.55,
        "buffer_seed": 1001,
    },
    {
        "label": "benchmark_pc_long_chain",
        "chain_length": 4,
        "repeats": 2,
        "advance_probability": 0.45,
        "buffer_seed": 1103,
    },
)

TRANSITION_FIELDS = (
    "state",
    "action",
    "reward",
    "next_state",
    "terminal",
    "truncated",
    "trajectory_id",
    "timestep",
)


def generate(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    manifest = {
        "owner": "Viet Tran — Benchmark & Oracle Validation",
        "estimator_seed": 20260903,
        "format": "adaptive_pilot.transitions_from_arrays",
        "design_reason": (
            "Bernoulli {0,1} rewards maximize reward variance within [0,1]; "
            "four trajectories keep trajectory-bootstrap dispersion high; "
            "multi-step stochastic chains permit an interior depth optimum, "
            "which a one-state bandit cannot provide."
        ),
        "instances": [],
    }
    for design in DESIGNS:
        arrays, initial, num_states, num_actions = stochastic_chain_control_buffer(
            chain_length=int(design["chain_length"]),
            repeats=int(design["repeats"]),
            advance_probability=float(design["advance_probability"]),
            seed=int(design["buffer_seed"]),
        )
        path = output_dir / f"{design['label']}.npz"
        np.savez(
            path,
            **arrays,
            num_states=np.asarray(num_states),
            num_actions=np.asarray(num_actions),
            initial_state_distribution=initial,
        )
        written.append(path)
        manifest["instances"].append(
            {
                **design,
                "num_states": num_states,
                "num_actions": num_actions,
                "num_trajectories": len(set(arrays["trajectory_id"].tolist())),
                "num_transitions": len(arrays["state"]),
                "transition_fields": list(TRANSITION_FIELDS),
            }
        )
    (output_dir / "design_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty; refusing to overwrite it.")
    for path in generate(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()

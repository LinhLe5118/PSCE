"""Baseline 6 -- random negative control.

Exists to catch spurious correlation in the statistical pipeline itself.
Beating this baseline is reported but never presented as evidence of
contribution (``PLAN_Final.md`` sec 6.1, last line;
``WEEK2_EVALUATION_PROTOCOL.md`` sec 6.6).

The RNG is seeded from a namespace disjoint from both the pilot and training
seed offsets (``protocol_config.json:collection.pilot_seed_offset`` /
``training_seed_offset``), so a random-control feature can never coincide
with a real data seed.
"""

from __future__ import annotations

import numpy as np

RANDOM_CONTROL_SEED_OFFSET = 30_000_000
FEATURE_SET: tuple[str, ...] = ("random_feature",)


def extract(mdp_id: str, *, seed: int) -> dict[str, object]:
    generator = np.random.default_rng([RANDOM_CONTROL_SEED_OFFSET, seed])
    return {"mdp_id": mdp_id, "random_feature": float(generator.standard_normal())}


__all__ = ["RANDOM_CONTROL_SEED_OFFSET", "FEATURE_SET", "extract"]

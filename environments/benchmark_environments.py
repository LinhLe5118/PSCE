"""Reproducible finite-MDP benchmark used by the DAORL pilot pipeline.

The benchmark contains 120 evaluation instances, 15 disjoint development
instances, and three interpretable OOD illustrations.  The original 14
hand-written environments in :mod:`tabular_environments` remain smoke-test
fixtures; they are not counted as benchmark instances.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from exact_solver import passes_dynamic_range_filter  # noqa: E402
from config.protocol_config import PROTOCOL_CONFIG
from environments.tabular_environments import (
    GRID_WALLS,
    TabularMDP,
    make_combination_lock,
    make_gridworld,
)


_BENCHMARK_CONFIG = PROTOCOL_CONFIG["benchmark"]
BENCHMARK_SEED = int(_BENCHMARK_CONFIG["master_seed"])
BENCHMARK_GAMMA = float(_BENCHMARK_CONFIG["gamma"])
STRUCTURAL_FAMILIES = tuple(_BENCHMARK_CONFIG["structural_families"])
OOD_FAMILIES = tuple(_BENCHMARK_CONFIG["ood_representatives"])
EVALUATION_COUNT = int(_BENCHMARK_CONFIG["evaluation_count"])
DEVELOPMENT_COUNT = int(_BENCHMARK_CONFIG["development_count"])
DYNAMIC_RANGE_SEED_STRIDE = int(_BENCHMARK_CONFIG["dynamic_range_seed_stride"])
MAX_REGENERATION_ATTEMPTS = int(_BENCHMARK_CONFIG["max_regeneration_attempts"])
DYNAMIC_RANGE_THRESHOLD = 0.01 / (1.0 - BENCHMARK_GAMMA)
EVALUATION_FAMILY_COUNTS = {
    family: int(count)
    for family, count in _BENCHMARK_CONFIG["evaluation_family_counts"].items()
}
DEVELOPMENT_FAMILY_COUNTS = {
    family: int(count)
    for family, count in _BENCHMARK_CONFIG["development_family_counts"].items()
}
ENVIRONMENT_RECIPES = _BENCHMARK_CONFIG["environment_recipes"]


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


@dataclass(frozen=True)
class BenchmarkSpec:
    """Compact recipe from which one benchmark MDP can be regenerated."""

    instance_id: str
    split: str
    family: str
    seed: int
    ordinal: int
    family_ordinal: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_specs(
    *,
    evaluation_count: int = EVALUATION_COUNT,
    development_count: int = DEVELOPMENT_COUNT,
    master_seed: int = BENCHMARK_SEED,
) -> list[BenchmarkSpec]:
    """Return the frozen 120 + 15 + 3 benchmark recipes.

    Structural families follow the registered unequal quotas and are interleaved
    deterministically. The three OOD entries remain one representative per family.
    """

    if evaluation_count <= 0 or development_count <= 0:
        raise ValueError("evaluation_count and development_count must be positive.")

    def apportioned_counts(
        total: int, registered: dict[str, int]
    ) -> dict[str, int]:
        registered_total = sum(registered.values())
        raw = {
            family: total * registered[family] / registered_total
            for family in STRUCTURAL_FAMILIES
        }
        counts = {family: int(raw[family]) for family in STRUCTURAL_FAMILIES}
        remainder = total - sum(counts.values())
        priority = sorted(
            STRUCTURAL_FAMILIES,
            key=lambda family: (
                -(raw[family] - counts[family]),
                STRUCTURAL_FAMILIES.index(family),
            ),
        )
        for family in priority[:remainder]:
            counts[family] += 1
        return counts

    def interleaved_schedule(counts: dict[str, int]) -> list[str]:
        positions = [
            ((index + 0.5) / count, STRUCTURAL_FAMILIES.index(family), family)
            for family, count in counts.items()
            for index in range(count)
        ]
        return [family for _, _, family in sorted(positions)]

    specs: list[BenchmarkSpec] = []
    split_offsets = {"evaluation": 0, "development": 1_000_000}
    for split, count, registered_counts in (
        ("evaluation", evaluation_count, EVALUATION_FAMILY_COUNTS),
        ("development", development_count, DEVELOPMENT_FAMILY_COUNTS),
    ):
        counts = apportioned_counts(count, registered_counts)
        family_ordinals = {family: 0 for family in STRUCTURAL_FAMILIES}
        for ordinal, family in enumerate(interleaved_schedule(counts)):
            seed = master_seed + split_offsets[split] + ordinal
            specs.append(
                BenchmarkSpec(
                    instance_id=f"{split}_{family}_{ordinal:03d}",
                    split=split,
                    family=family,
                    seed=seed,
                    ordinal=ordinal,
                    family_ordinal=family_ordinals[family],
                )
            )
            family_ordinals[family] += 1

    for ordinal, family in enumerate(OOD_FAMILIES):
        specs.append(
            BenchmarkSpec(
                instance_id=f"ood_{family}",
                split="ood",
                family=family,
                seed=master_seed + 2_000_000 + ordinal,
                ordinal=ordinal,
                family_ordinal=0,
            )
        )
    return specs


def _stratified_choice(spec: BenchmarkSpec, axis: str) -> Any:
    """Choose a recipe level with balanced marginals within each family/split."""

    values = tuple(ENVIRONMENT_RECIPES[spec.family][axis])
    block, slot = divmod(spec.family_ordinal, len(values))
    material = (
        f"{BENCHMARK_SEED}|{spec.split}|{spec.family}|{axis}|{block}"
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], byteorder="little")
    permutation = _rng(seed).permutation(len(values))
    return values[int(permutation[slot])]


def _empty_mdp(num_states: int, num_actions: int) -> tuple[np.ndarray, np.ndarray]:
    shape = (num_states, num_actions, num_states)
    return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64)


def _finalize(
    spec: BenchmarkSpec,
    transitions: np.ndarray,
    rewards: np.ndarray,
    initial: np.ndarray,
    terminal_states: tuple[int, ...],
    metadata: dict[str, Any],
) -> TabularMDP:
    mdp = TabularMDP(
        environment_id=spec.instance_id,
        P=transitions,
        R=rewards,
        initial_state_distribution=initial,
        terminal_states=terminal_states,
        gamma=BENCHMARK_GAMMA,
        metadata={
            **metadata,
            "benchmark_split": spec.split,
            "benchmark_seed": spec.seed,
            "benchmark_ordinal": spec.ordinal,
            "benchmark_family_ordinal": spec.family_ordinal,
            "stratified_recipe": spec.split != "ood",
            "num_states": int(transitions.shape[0]),
            "num_actions": int(transitions.shape[1]),
        },
    )
    mdp.validate()
    return mdp


def _draw_sparse_rewards(
    rng: np.random.Generator,
    num_states: int,
    num_actions: int,
    sparsity: float,
    reward_range: tuple[float, float],
) -> np.ndarray:
    rewards = np.zeros((num_states, num_actions), dtype=np.float64)
    count = max(1, round(sparsity * num_states * num_actions))
    chosen = rng.choice(num_states * num_actions, size=count, replace=False)
    rewards.flat[chosen] = rng.uniform(*reward_range, size=count)
    return rewards


def _make_uniform_garnet(spec: BenchmarkSpec, rng: np.random.Generator) -> TabularMDP:
    recipe = ENVIRONMENT_RECIPES[spec.family]
    num_states = int(_stratified_choice(spec, "num_states"))
    num_actions = int(_stratified_choice(spec, "num_actions"))
    branching = min(int(_stratified_choice(spec, "branching_factor")), num_states)
    concentration = float(_stratified_choice(spec, "dirichlet_concentration"))
    reward_sparsity = float(_stratified_choice(spec, "reward_sparsity"))
    nonzero_reward_range = tuple(float(value) for value in recipe["nonzero_reward_range"])
    P, R = _empty_mdp(num_states, num_actions)
    expected_rewards = _draw_sparse_rewards(
        rng,
        num_states,
        num_actions,
        reward_sparsity,
        nonzero_reward_range,
    )
    for state in range(num_states):
        for action in range(num_actions):
            successors = rng.choice(num_states, size=branching, replace=False)
            probabilities = rng.dirichlet(np.full(branching, concentration))
            P[state, action, successors] = probabilities
            R[state, action, successors] = expected_rewards[state, action]

    initial = np.zeros(num_states)
    initial[int(rng.integers(num_states))] = 1.0
    return _finalize(
        spec,
        P,
        R,
        initial,
        (),
        {
            "family": "uniform_garnet",
            "branching_factor": branching,
            "dirichlet_concentration": concentration,
            "reward_sparsity": reward_sparsity,
            "transition_type": "stochastic",
        },
    )


def _make_forward_layered(spec: BenchmarkSpec, rng: np.random.Generator) -> TabularMDP:
    recipe = ENVIRONMENT_RECIPES[spec.family]
    depth = int(_stratified_choice(spec, "depth"))
    width = int(_stratified_choice(spec, "width"))
    num_states = depth * width
    num_actions = int(_stratified_choice(spec, "num_actions"))
    branching = min(int(_stratified_choice(spec, "branching_factor")), width)
    concentration = float(_stratified_choice(spec, "dirichlet_concentration"))
    P, R = _empty_mdp(num_states, num_actions)
    terminal_states = tuple(range((depth - 1) * width, depth * width))
    goal = int(rng.choice(terminal_states))
    backbone_states = [
        int(rng.choice(np.arange(layer * width, (layer + 1) * width)))
        for layer in range(1, depth - 1)
    ] + [goal]
    correct_actions = rng.integers(
        0, num_actions, size=(depth - 1) * width
    )
    backbone_probability = float(recipe["backbone_probability"])

    for layer in range(depth - 1):
        next_layer = np.arange((layer + 1) * width, (layer + 2) * width)
        for state in range(layer * width, (layer + 1) * width):
            for action in range(num_actions):
                successors = rng.choice(next_layer, size=branching, replace=False)
                P[state, action, successors] = rng.dirichlet(
                    np.full(branching, concentration)
                )
            correct_action = int(correct_actions[state])
            P[state, correct_action] *= 1.0 - backbone_probability
            P[state, correct_action, backbone_states[layer]] += backbone_probability
    R[:, :, goal] = 1.0
    for terminal in terminal_states:
        P[terminal, :, terminal] = 1.0
        R[terminal] = 0.0

    initial = np.zeros(num_states)
    initial[:width] = 1.0 / width
    return _finalize(
        spec,
        P,
        R,
        initial,
        terminal_states,
        {
            "family": "forward_layered",
            "depth": depth,
            "width": width,
            "branching_factor": branching,
            "dirichlet_concentration": concentration,
            "goal_state": goal,
            "backbone_states": backbone_states,
            "backbone_probability": backbone_probability,
            "correct_actions": correct_actions.tolist(),
            "reward_sparsity": 1.0 / width,
            "transition_type": "stochastic_forward_only",
        },
    )


def _make_lock_structured(spec: BenchmarkSpec, rng: np.random.Generator) -> TabularMDP:
    length = int(_stratified_choice(spec, "length"))
    base = make_combination_lock(
        length, seed=int(rng.integers(2**31 - 1)), gamma=BENCHMARK_GAMMA
    )
    return _finalize(
        spec,
        base.P,
        base.R,
        base.initial_state_distribution,
        base.terminal_states,
        {
            **base.metadata,
            "family": "lock_structured",
            "source_family": "combination_lock",
            "reward_sparsity": 1.0 / (length * 2),
        },
    )


def _make_bottleneck(spec: BenchmarkSpec, rng: np.random.Generator) -> TabularMDP:
    recipe = ENVIRONMENT_RECIPES[spec.family]
    community_size = int(_stratified_choice(spec, "community_size"))
    num_states = 2 * community_size
    num_actions = int(_stratified_choice(spec, "num_actions"))
    branching = min(int(_stratified_choice(spec, "branching_factor")), community_size)
    gateway, target = community_size - 1, num_states - 1
    P, R = _empty_mdp(num_states, num_actions)

    for state in range(num_states - 1):
        in_left = state < community_size
        community = np.arange(0, community_size) if in_left else np.arange(
            community_size, num_states
        )
        for action in range(num_actions):
            successors = rng.choice(community, size=branching, replace=False)
            P[state, action, successors] = rng.dirichlet(np.ones(branching))

    # A rare, identifiable bridge plus a forward backbone make the target
    # reachable without turning the topology into a simple chain.
    bridge_probability = float(_stratified_choice(spec, "bridge_probability"))
    P[gateway, 0] *= 1.0 - bridge_probability
    P[gateway, 0, community_size] += bridge_probability
    for state in range(num_states - 1):
        if state in (gateway, target):
            continue
        forward = state + 1
        backbone_probability = float(recipe["backbone_probability"])
        P[state, 0] *= 1.0 - backbone_probability
        P[state, 0, forward] += backbone_probability

    R[:, :, target] = 1.0
    P[target] = 0.0
    P[target, :, target] = 1.0
    R[target] = 0.0
    initial = np.zeros(num_states)
    initial[0] = 1.0
    return _finalize(
        spec,
        P,
        R,
        initial,
        (target,),
        {
            "family": "bottleneck",
            "community_size": community_size,
            "branching_factor": branching,
            "gateway_state": gateway,
            "bridge_probability": bridge_probability,
            "goal_state": target,
            "reward_sparsity": 1.0 / (num_states * num_actions),
            "transition_type": "stochastic_bottleneck",
        },
    )


def make_riverswim(
    *, num_states: int = 8, gamma: float = BENCHMARK_GAMMA, seed: int = 0
) -> TabularMDP:
    """Create the classic hard-exploration RiverSwim topology."""

    if num_states < 4:
        raise ValueError("RiverSwim needs at least four states.")
    P, R = _empty_mdp(num_states, 2)
    for state in range(num_states):
        left = max(0, state - 1)
        right = min(num_states - 1, state + 1)
        P[state, 0, left] = 1.0
        P[state, 1, left] += 0.05
        P[state, 1, state] += 0.60
        P[state, 1, right] += 0.35
    R[0, 0, 0] = 0.005
    R[-1, 1, -1] = 1.0
    initial = np.zeros(num_states)
    initial[0] = 1.0
    mdp = TabularMDP(
        environment_id=f"riverswim_{num_states}",
        P=P,
        R=R,
        initial_state_distribution=initial,
        terminal_states=(),
        gamma=gamma,
        metadata={
            "family": "riverswim",
            "num_states": num_states,
            "transition_type": "stochastic",
            "seed": seed,
        },
    )
    mdp.validate()
    return mdp


def _make_ood(spec: BenchmarkSpec) -> TabularMDP:
    recipe = ENVIRONMENT_RECIPES["ood"]
    if spec.family == "combination_lock":
        base = make_combination_lock(
            int(recipe["combination_lock_length"]),
            seed=spec.seed,
            gamma=BENCHMARK_GAMMA,
        )
    elif spec.family == "grid_maze":
        grid_size = int(recipe["grid_maze_size"])
        base = make_gridworld(
            grid_size,
            walls=GRID_WALLS[grid_size],
            seed=spec.seed,
            gamma=BENCHMARK_GAMMA,
            step_cost=0.0,
        )
    elif spec.family == "riverswim":
        base = make_riverswim(
            num_states=int(recipe["riverswim_num_states"]),
            gamma=BENCHMARK_GAMMA,
            seed=spec.seed,
        )
    else:
        raise KeyError(f"Unknown OOD family: {spec.family}")
    return _finalize(
        spec,
        base.P,
        base.R,
        base.initial_state_distribution,
        base.terminal_states,
        {**base.metadata, "family": spec.family, "ood_illustration": True},
    )


def make_benchmark_mdp(
    spec: BenchmarkSpec, *, enforce_dynamic_range: bool = False
) -> TabularMDP:
    """Materialize one recipe; optionally enforce the registered range filter."""

    rng = _rng(spec.seed)
    if spec.split == "ood":
        mdp = _make_ood(spec)
    elif spec.family == "uniform_garnet":
        mdp = _make_uniform_garnet(spec, rng)
    elif spec.family == "forward_layered":
        mdp = _make_forward_layered(spec, rng)
    elif spec.family == "lock_structured":
        mdp = _make_lock_structured(spec, rng)
    elif spec.family == "bottleneck":
        mdp = _make_bottleneck(spec, rng)
    else:
        raise KeyError(f"Unknown structural family: {spec.family}")

    if enforce_dynamic_range and not passes_dynamic_range_filter(
        mdp, threshold=DYNAMIC_RANGE_THRESHOLD
    ):
        raise ValueError(
            f"{spec.instance_id} fails the registered dynamic-range filter; "
            "replace its seed before freezing the benchmark."
        )
    return mdp


def resolve_dynamic_range(
    spec: BenchmarkSpec, *, max_attempts: int = MAX_REGENERATION_ATTEMPTS
) -> tuple[BenchmarkSpec, TabularMDP, int]:
    """Deterministically regenerate a recipe until it passes the range filter.

    The returned rejection count must be recorded in the benchmark manifest.
    OOD illustrations are kept as designed and are not part of superiority
    testing, so the filter is applied only to development/evaluation MDPs.
    """

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    if spec.split == "ood":
        return spec, make_benchmark_mdp(spec), 0
    for attempt in range(max_attempts):
        candidate = replace(spec, seed=spec.seed + attempt * DYNAMIC_RANGE_SEED_STRIDE)
        mdp = make_benchmark_mdp(candidate)
        if passes_dynamic_range_filter(mdp, threshold=DYNAMIC_RANGE_THRESHOLD):
            return candidate, mdp, attempt
    raise RuntimeError(
        f"Could not generate a non-degenerate {spec.family} instance after "
        f"{max_attempts} attempts."
    )


def materialize_benchmark(
    specs: Iterable[BenchmarkSpec] | None = None,
    *,
    enforce_dynamic_range: bool = False,
) -> list[TabularMDP]:
    recipes = benchmark_specs() if specs is None else list(specs)
    return [
        make_benchmark_mdp(spec, enforce_dynamic_range=enforce_dynamic_range)
        for spec in recipes
    ]


__all__ = [
    "BENCHMARK_GAMMA",
    "BENCHMARK_SEED",
    "DEVELOPMENT_COUNT",
    "DEVELOPMENT_FAMILY_COUNTS",
    "DYNAMIC_RANGE_THRESHOLD",
    "ENVIRONMENT_RECIPES",
    "EVALUATION_COUNT",
    "EVALUATION_FAMILY_COUNTS",
    "BenchmarkSpec",
    "OOD_FAMILIES",
    "STRUCTURAL_FAMILIES",
    "benchmark_specs",
    "make_benchmark_mdp",
    "make_riverswim",
    "materialize_benchmark",
    "resolve_dynamic_range",
]

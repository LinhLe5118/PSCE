"""Load the single machine-readable protocol configuration."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_CONFIG_PATH = Path(
    os.environ.get("DAORL_PROTOCOL_CONFIG", str(PROJECT_ROOT / "protocol_config.json"))
).resolve()


def _require_mapping(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Protocol section {name!r} must be an object.")
    return value


def _require_positive_int(section: dict[str, Any], name: str) -> int:
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Protocol field {name!r} must be a positive integer.")
    return value


def _require_unique_strings(section: dict[str, Any], name: str) -> tuple[str, ...]:
    value = section.get(name)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"Protocol field {name!r} must contain unique strings.")
    return tuple(value)


def _finite_numbers(value: Any, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Protocol field {name!r} must be a non-empty list.")
    numbers = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in numbers):
        raise ValueError(f"Protocol field {name!r} must contain finite numbers.")
    return numbers


def _require_family_counts(
    benchmark: dict[str, Any],
    name: str,
    families: tuple[str, ...],
    expected_total: int,
) -> dict[str, int]:
    value = benchmark.get(name)
    if not isinstance(value, dict) or set(value) != set(families):
        raise ValueError(f"benchmark.{name} must define every structural family once.")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
        for count in value.values()
    ):
        raise ValueError(f"benchmark.{name} counts must be positive integers.")
    if sum(value.values()) != expected_total:
        raise ValueError(f"benchmark.{name} must sum to {expected_total}.")
    return value


def _require_recipe_levels(
    recipe: dict[str, Any], name: str, *, positive: bool = True
) -> tuple[float, ...]:
    values = _finite_numbers(recipe.get(name), name=f"recipe.{name}")
    if len(set(values)) != len(values):
        raise ValueError(f"recipe.{name} levels must be unique.")
    if positive and any(value <= 0.0 for value in values):
        raise ValueError(f"recipe.{name} levels must be positive.")
    return values


def validate_protocol_config(config: dict[str, Any]) -> None:
    """Validate every runtime assumption encoded by the benchmark protocol."""

    required = {
        "schema_version",
        "protocol_status",
        "pending_validations",
        "benchmark",
        "behavior_policies",
        "estimator",
        "learning",
        "collection",
        "coverage",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Protocol config is missing sections: {sorted(missing)}")
    if not isinstance(config.get("schema_version"), str) or not config["schema_version"]:
        raise ValueError("Protocol schema_version must be a non-empty string.")
    status = config.get("protocol_status")
    if status not in {"candidate", "frozen"}:
        raise ValueError("protocol_status must be candidate or frozen.")
    pending = config.get("pending_validations")
    if (
        not isinstance(pending, list)
        or any(not isinstance(item, str) or not item for item in pending)
        or len(set(pending)) != len(pending)
    ):
        raise ValueError("pending_validations must contain unique non-empty strings.")
    if status == "frozen" and pending:
        raise ValueError("A frozen protocol cannot have pending validations.")

    benchmark = _require_mapping(config, "benchmark")
    evaluation_count = _require_positive_int(benchmark, "evaluation_count")
    development_count = _require_positive_int(benchmark, "development_count")
    structural_families = _require_unique_strings(benchmark, "structural_families")
    if structural_families != (
        "uniform_garnet",
        "forward_layered",
        "lock_structured",
        "bottleneck",
    ):
        raise ValueError("structural_families does not match implemented generators.")
    if _require_unique_strings(benchmark, "ood_representatives") != (
        "combination_lock",
        "grid_maze",
        "riverswim",
    ):
        raise ValueError("ood_representatives does not match implemented generators.")
    _require_family_counts(
        benchmark,
        "evaluation_family_counts",
        structural_families,
        evaluation_count,
    )
    _require_family_counts(
        benchmark,
        "development_family_counts",
        structural_families,
        development_count,
    )
    _require_positive_int(benchmark, "master_seed")
    _require_positive_int(benchmark, "dynamic_range_seed_stride")
    _require_positive_int(benchmark, "max_regeneration_attempts")
    gamma = float(benchmark.get("gamma", math.nan))
    if not math.isfinite(gamma) or not 0.0 <= gamma < 1.0:
        raise ValueError("benchmark.gamma must lie in [0,1).")
    reward_range = _finite_numbers(benchmark.get("reward_range"), name="reward_range")
    if len(reward_range) != 2 or reward_range[0] >= reward_range[1]:
        raise ValueError("benchmark.reward_range must be [minimum, maximum].")
    if reward_range != (0.0, 1.0):
        raise ValueError("The implemented benchmark requires reward_range=[0,1].")
    if benchmark.get("dynamic_range_threshold") != "0.01/(1-gamma)":
        raise ValueError(
            "The prototype supports dynamic_range_threshold='0.01/(1-gamma)' only."
        )

    recipes = benchmark.get("environment_recipes")
    if not isinstance(recipes, dict) or set(recipes) != set(structural_families) | {
        "ood"
    }:
        raise ValueError("benchmark.environment_recipes does not match benchmark families.")
    uniform = recipes["uniform_garnet"]
    forward = recipes["forward_layered"]
    lock = recipes["lock_structured"]
    bottleneck = recipes["bottleneck"]
    ood = recipes["ood"]
    if not all(isinstance(recipe, dict) for recipe in recipes.values()):
        raise ValueError("Every environment recipe must be an object.")
    for recipe, fields in (
        (
            uniform,
            (
                "num_states",
                "num_actions",
                "branching_factor",
                "dirichlet_concentration",
                "reward_sparsity",
            ),
        ),
        (
            forward,
            (
                "depth",
                "width",
                "num_actions",
                "branching_factor",
                "dirichlet_concentration",
            ),
        ),
        (
            bottleneck,
            (
                "community_size",
                "num_actions",
                "branching_factor",
                "bridge_probability",
            ),
        ),
    ):
        for field in fields:
            _require_recipe_levels(recipe, field)
    reward_levels = _require_recipe_levels(uniform, "nonzero_reward_range")
    if len(reward_levels) != 2 or reward_levels[0] >= reward_levels[1]:
        raise ValueError("uniform_garnet.nonzero_reward_range must be [min,max].")
    if reward_levels[0] < reward_range[0] or reward_levels[1] > reward_range[1]:
        raise ValueError("uniform_garnet nonzero rewards must lie in reward_range.")
    if any(not 0.0 < value <= 1.0 for value in uniform["reward_sparsity"]):
        raise ValueError("uniform_garnet.reward_sparsity must lie in (0,1].")
    if any(value > 1.0 for value in bottleneck["bridge_probability"]):
        raise ValueError("bottleneck.bridge_probability must lie in (0,1].")
    backbone_probability = float(bottleneck.get("backbone_probability", math.nan))
    if not math.isfinite(backbone_probability) or not 0.0 < backbone_probability < 1.0:
        raise ValueError("bottleneck.backbone_probability must lie in (0,1).")
    forward_backbone = float(forward.get("backbone_probability", math.nan))
    if not math.isfinite(forward_backbone) or not 0.0 < forward_backbone < 1.0:
        raise ValueError("forward_layered.backbone_probability must lie in (0,1).")
    lengths = _require_recipe_levels(lock, "length")
    if any(not value.is_integer() or value < 2 for value in lengths):
        raise ValueError("lock_structured.length must contain integers >= 2.")
    if lock.get("num_actions") != 2 or lock.get("transition_type") != "deterministic":
        raise ValueError("lock_structured recipe does not match its implementation.")
    if lock.get("wrong_action") != "reset_to_start":
        raise ValueError("Unsupported lock_structured wrong-action semantics.")
    for field in (
        "combination_lock_length",
        "grid_maze_size",
        "riverswim_num_states",
    ):
        if (
            isinstance(ood.get(field), bool)
            or not isinstance(ood.get(field), int)
            or ood[field] <= 0
        ):
            raise ValueError(f"ood.{field} must be a positive integer.")

    policies = _require_mapping(config, "behavior_policies")
    if _require_unique_strings(policies, "families") != (
        "q_star_softmax",
        "perturbed_reward",
        "structured_random",
        "two_policy_mixture",
        "shifted_dynamics",
    ):
        raise ValueError("behavior_policies.families does not match implemented policies.")
    grids = {
        name: _finite_numbers(policies.get(name), name=name)
        for name in (
            "temperature_grid",
            "reward_noise_grid",
            "epsilon_grid",
            "mixture_weight_grid",
            "dynamics_shift_grid",
        )
    }
    if len({len(grid) for grid in grids.values()}) != 1:
        raise ValueError("All behavior-policy parameter grids must have equal length.")
    if any(value <= 0.0 for value in grids["temperature_grid"]):
        raise ValueError("temperature_grid values must be positive.")
    if any(value < 0.0 for value in grids["reward_noise_grid"]):
        raise ValueError("reward_noise_grid values must be non-negative.")
    for name in ("epsilon_grid", "dynamics_shift_grid"):
        if any(not 0.0 <= value <= 1.0 for value in grids[name]):
            raise ValueError(f"{name} values must lie in [0,1].")
    if any(not 0.0 < value < 1.0 for value in grids["mixture_weight_grid"]):
        raise ValueError("mixture_weight_grid values must lie strictly inside (0,1).")
    _require_positive_int(policies, "structured_score_rank")

    estimator = _require_mapping(config, "estimator")
    estimator_gamma = float(estimator.get("gamma", math.nan))
    if estimator_gamma != gamma:
        raise ValueError("estimator.gamma must match benchmark.gamma.")
    tau_coefficient = float(estimator.get("tau_coefficient", math.nan))
    if not math.isfinite(tau_coefficient) or not 0.0 < tau_coefficient <= 1.0:
        raise ValueError("estimator.tau_coefficient must lie in (0,1].")
    for name in (
        "n_min",
        "k_max",
        "num_bootstrap",
        "num_exact_replicates",
        "n0_coefficient",
        "horizon",
        "min_trajectory_support",
    ):
        _require_positive_int(estimator, name)
    if estimator["min_trajectory_support"] < 2:
        raise ValueError("estimator.min_trajectory_support must be >= 2.")
    if estimator.get("bootstrap_scheme") != "bayesian_trajectory_dirichlet":
        raise ValueError(
            "estimator.bootstrap_scheme must be "
            "'bayesian_trajectory_dirichlet'."
        )
    bootstrap_alpha = float(
        estimator.get("bootstrap_dirichlet_alpha", math.nan)
    )
    if bootstrap_alpha != 1.0:
        raise ValueError(
            "estimator.bootstrap_dirichlet_alpha must be 1.0 for the "
            "registered Bayesian bootstrap."
        )
    n0_candidates = _finite_numbers(
        estimator.get("n0_candidate_coefficients"),
        name="estimator.n0_candidate_coefficients",
    )
    if (
        any(not value.is_integer() or value <= 0 for value in n0_candidates)
        or tuple(sorted(set(n0_candidates))) != n0_candidates
        or estimator["n0_coefficient"] not in n0_candidates
        or n0_candidates != (2.0, 5.0, 10.0)
    ):
        raise ValueError(
            "estimator.n0_candidate_coefficients must be [2,5,10] and include "
            "the reference n0_coefficient."
        )
    if estimator.get("n0_selection_rule") != "first_usable_data_only_prefix":
        raise ValueError("Unsupported estimator.n0_selection_rule.")
    if estimator.get("n0_cap_outcome") != "abstain_or_censor":
        raise ValueError("Unsupported estimator.n0_cap_outcome.")
    epsilon = float(estimator.get("epsilon", math.nan))
    valid_bootstrap = float(
        estimator.get("min_valid_bootstrap_fraction", math.nan)
    )
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("estimator.epsilon must lie in (0,1).")
    if not math.isfinite(valid_bootstrap) or not 0.0 < valid_bootstrap <= 1.0:
        raise ValueError(
            "estimator.min_valid_bootstrap_fraction must lie in (0,1]."
        )

    collection = _require_mapping(config, "collection")
    if collection.get("rng") != "numpy.random.PCG64":
        raise ValueError("The registered collection RNG must be numpy.random.PCG64.")
    collection_horizon = _require_positive_int(collection, "horizon")
    if collection_horizon != estimator["horizon"]:
        raise ValueError("collection.horizon must match estimator.horizon.")
    if estimator["k_max"] > collection_horizon:
        raise ValueError("estimator.k_max cannot exceed collection.horizon.")
    pilot_multipliers = _finite_numbers(
        collection.get("pilot_size_multipliers"),
        name="pilot_size_multipliers",
    )
    if (
        any(not value.is_integer() or value <= 0 for value in pilot_multipliers)
        or tuple(sorted(set(pilot_multipliers))) != pilot_multipliers
    ):
        raise ValueError(
            "pilot_size_multipliers must contain unique increasing positive integers."
        )
    if pilot_multipliers != n0_candidates:
        raise ValueError(
            "pilot_size_multipliers must match estimator N0 candidates."
        )
    multipliers = _finite_numbers(collection.get("size_multipliers"), name="size_multipliers")
    if any(not value.is_integer() or value <= 0 for value in multipliers):
        raise ValueError("size_multipliers must contain positive integers.")
    if tuple(sorted(set(multipliers))) != multipliers:
        raise ValueError("size_multipliers must be unique and strictly increasing.")
    for name in ("pilot_seed_indices", "training_seed_indices"):
        values = collection.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in values
            )
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"collection.{name} must contain unique non-negative integers.")
    pilot_offset = _require_positive_int(collection, "pilot_seed_offset")
    training_offset = _require_positive_int(collection, "training_seed_offset")
    if pilot_offset == training_offset:
        raise ValueError("Pilot and training seed offsets must be distinct.")
    pilot_namespace = {
        (pilot_offset + index) % (2**32 - 1)
        for index in collection["pilot_seed_indices"]
    }
    training_namespace = {
        (training_offset + index) % (2**32 - 1)
        for index in collection["training_seed_indices"]
    }
    if pilot_namespace & training_namespace:
        raise ValueError("Pilot and training seed namespaces overlap.")

    learning = _require_mapping(config, "learning")
    if _require_unique_strings(learning, "algorithms") != (
        "cfqi_tabular",
        "iql_tabular",
    ):
        raise ValueError("learning.algorithms must match the registered tabular learners.")
    for name in ("iterations", "success_runs_required", "success_runs_total"):
        _require_positive_int(learning, name)
    if learning["iterations"] != 100:
        raise ValueError("learning.iterations must be the registered value 100.")
    if learning["success_runs_required"] != 8:
        raise ValueError("learning.success_runs_required must be 8.")
    if learning["success_runs_total"] != len(collection["training_seed_indices"]):
        raise ValueError("learning.success_runs_total must match training seed count.")
    tolerance = float(learning.get("tolerance", math.nan))
    expectile = float(learning.get("iql_expectile", math.nan))
    inverse_temperature = float(learning.get("iql_inverse_temperature", math.nan))
    success_return = float(learning.get("success_normalized_return", math.nan))
    if tolerance != 1e-10 or not 0.0 < expectile < 1.0 or inverse_temperature != 3.0:
        raise ValueError("Unregistered tabular learner settings.")
    if success_return != 0.95:
        raise ValueError("learning.success_normalized_return must be 0.95.")
    if learning.get("unknown_state_policy") != "uniform":
        raise ValueError("Unsupported unknown-state policy.")
    if learning.get("unsupported_action_policy") != "excluded_from_greedy_max":
        raise ValueError("Unsupported unsupported-action policy.")

    coverage = _require_mapping(config, "coverage")
    if coverage.get("rule") != "generate_then_measure_then_stratify":
        raise ValueError("coverage.rule must preserve generate-then-measure semantics.")
    if coverage.get("primary_metric") != "observed_reachable_sa_fraction":
        raise ValueError("Unsupported primary coverage metric.")
    thresholds = _finite_numbers(coverage.get("thresholds"), name="coverage.thresholds")
    labels = _require_unique_strings(coverage, "labels")
    if len(labels) != len(thresholds) + 1:
        raise ValueError("coverage.labels must have exactly one more item than thresholds.")
    if any(not 0.0 < value < 1.0 for value in thresholds) or tuple(
        sorted(set(thresholds))
    ) != thresholds:
        raise ValueError("coverage.thresholds must be unique, increasing, and inside (0,1).")
    coverage_n_min = _require_positive_int(coverage, "n_min")
    if coverage_n_min != estimator["n_min"]:
        raise ValueError("coverage.n_min must match estimator.n_min.")
    if coverage.get("post_hoc_transition_filtering") is not False:
        raise ValueError("post_hoc_transition_filtering must remain false.")


def load_protocol_config(path: Path = PROTOCOL_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Protocol config root must be an object.")
    validate_protocol_config(config)
    if path.resolve() != (PROJECT_ROOT / "protocol_config.json").resolve():
        if config["protocol_status"] != "candidate":
            raise ValueError("Alternate profiles must remain candidate (review-only).")
    return config


PROTOCOL_CONFIG = load_protocol_config()


__all__ = [
    "PROJECT_ROOT",
    "PROTOCOL_CONFIG",
    "PROTOCOL_CONFIG_PATH",
    "load_protocol_config",
    "validate_protocol_config",
]

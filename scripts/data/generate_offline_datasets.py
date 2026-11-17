"""Generate auditable offline trajectories for the DAORL benchmark.

Coverage follows the frozen three-stage rule: generate unmodified trajectories,
measure realized coverage, then assign a stratum.  No transition is inserted,
removed, duplicated, or filtered to force a requested support level.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from copy import deepcopy
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np

from environments.behavior_policies import BehaviorPolicy, behavior_policy_suite
from environments.benchmark_panels import audit_dynamics, PRIMARY_FAMILIES, CONTROL_FAMILIES
from environments.benchmark_environments import (
    BENCHMARK_SEED,
    BenchmarkSpec,
    benchmark_specs,
    resolve_dynamic_range,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from exact_solver import dynamic_range, evaluate_policy  # noqa: E402
from config.protocol_config import PROTOCOL_CONFIG, PROTOCOL_CONFIG_PATH
from environments.tabular_environments import ENVIRONMENT_REGISTRY, TabularMDP, make_env


SCHEMA_VERSION = "2.6-candidate"
_COLLECTION_CONFIG = PROTOCOL_CONFIG["collection"]
_COVERAGE_CONFIG = PROTOCOL_CONFIG["coverage"]
_ESTIMATOR_CONFIG = PROTOCOL_CONFIG["estimator"]
SIZE_MULTIPLIERS = tuple(int(value) for value in _COLLECTION_CONFIG["size_multipliers"])
PILOT_SIZE_MULTIPLIERS = tuple(
    int(value) for value in _COLLECTION_CONFIG["pilot_size_multipliers"]
)
DEFAULT_N0_COEFFICIENT = int(_ESTIMATOR_CONFIG["n0_coefficient"])
DEFAULT_HORIZON = int(_COLLECTION_CONFIG["horizon"])
DEFAULT_N_MIN = int(_COVERAGE_CONFIG["n_min"])
DEFAULT_TRAINING_SEEDS = tuple(int(value) for value in _COLLECTION_CONFIG["training_seed_indices"])
DEFAULT_PILOT_SEEDS = tuple(int(value) for value in _COLLECTION_CONFIG["pilot_seed_indices"])
DEFAULT_COVERAGE_THRESHOLDS = tuple(float(value) for value in _COVERAGE_CONFIG["thresholds"])
DEFAULT_COVERAGE_LABELS = tuple(str(value) for value in _COVERAGE_CONFIG["labels"])
STREAM_SEED_OFFSET = {
    "pilot": int(_COLLECTION_CONFIG["pilot_seed_offset"]),
    "training": int(_COLLECTION_CONFIG["training_seed_offset"]),
}
TRANSITION_FIELDS = (
    "state",
    "action",
    "reward",
    "next_state",
    "done",
    "terminal",
    "truncated",
    "trajectory_id",
    "timestep",
)
GENERATOR_SOURCE_FILES = (
    "protocol_config.json",
    "config/protocol_config.py",
    "environments/benchmark_environments.py",
    "environments/benchmark_panels.py",
    "environments/behavior_policies.py",
    "scripts/data/generate_offline_datasets.py",
    "environments/tabular_environments.py",
    "code/exact_solver.py",
    "code/adaptive_pilot.py",
    "code/oeh_estimator.py",
    "requirements.txt",
)


@dataclass(frozen=True)
class CollectionConfig:
    instance_id: str
    benchmark_split: str
    mdp_seed: int
    policy_family: str
    policy_seed: int
    stream: str
    stream_seed: int
    max_size: int
    horizon: int


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _stable_identifier(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)
    return digest.hexdigest()


def policy_sha256(policy: BehaviorPolicy) -> str:
    canonical = np.ascontiguousarray(policy.probabilities, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def dataset_content_sha256(data: dict[str, np.ndarray]) -> str:
    """Hash logical array contents independently of NPZ compression details."""

    digest = hashlib.sha256()
    for field in TRANSITION_FIELDS:
        values = np.ascontiguousarray(data[field])
        digest.update(field.encode("utf-8"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _generator_provenance() -> dict[str, Any]:
    source_hash = hashlib.sha256()
    source_hash.update(PROTOCOL_CONFIG_PATH.read_bytes())
    for name in GENERATOR_SOURCE_FILES:
        path = PROJECT_ROOT / name
        content = path.read_bytes()
        source_hash.update(name.encode("utf-8"))
        source_hash.update(b"\0")
        source_hash.update(len(content).to_bytes(8, byteorder="little"))
        source_hash.update(content)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, dirty = None, None
    try:
        scipy_version = version("scipy")
    except PackageNotFoundError:
        scipy_version = None
    return {
        "git_commit": commit,
        "git_worktree_dirty": dirty,
        "source_sha256": source_hash.hexdigest(),
        "source_files": list(GENERATOR_SOURCE_FILES),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
    }


def collect_trajectories(
    mdp: TabularMDP,
    policy: BehaviorPolicy,
    *,
    seed: int,
    size: int,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, np.ndarray]:
    """Collect exactly ``size`` natural transitions under ``policy``."""

    if size <= 0 or horizon <= 0:
        raise ValueError("size and horizon must be positive.")
    policy.validate(mdp)
    rng = np.random.Generator(np.random.PCG64(seed))
    # Match Generator.choice(p=...) inverse-CDF sampling, including its
    # normalization, while caching the unchanged distributions per buffer.
    initial_cdf = np.cumsum(mdp.initial_state_distribution)
    initial_cdf /= initial_cdf[-1]
    action_cdf = np.cumsum(policy.probabilities, axis=-1)
    action_cdf /= action_cdf[:, -1:]
    transition_cdf = np.cumsum(mdp.P, axis=-1)
    transition_cdf /= transition_cdf[:, :, -1:]

    # Sparse successor lists avoid Python/NumPy dispatch for every step.
    # Keep cumulative sums from the full row to preserve choice's rounding.
    successors = [[np.flatnonzero(mdp.P[s, a]).tolist()
                   for a in range(mdp.num_actions)] for s in range(mdp.num_states)]
    transition_thresholds = [[transition_cdf[s, a, successors[s][a]].tolist()
                             for a in range(mdp.num_actions)] for s in range(mdp.num_states)]
    initial_thresholds = initial_cdf.tolist()
    action_thresholds = action_cdf.tolist()

    def draw(cdf: list[float]) -> int:
        return bisect_right(cdf, rng.random())

    rows: list[tuple[int, int, float, int, bool, bool, bool, int, int]] = []
    trajectory_id = 0
    timestep = 0
    state = draw(initial_thresholds)
    terminals = set(mdp.terminal_states)

    while len(rows) < size:
        action = draw(action_thresholds[state])
        next_state = successors[state][action][draw(transition_thresholds[state][action])]
        terminal = next_state in terminals
        truncated = (timestep + 1 >= horizon) and not terminal
        reward = float(mdp.R[state, action, next_state])
        rows.append(
            (
                state,
                action,
                reward,
                next_state,
                terminal,
                terminal,
                truncated,
                trajectory_id,
                timestep,
            )
        )
        if terminal or truncated:
            trajectory_id += 1
            timestep = 0
            state = draw(initial_thresholds)
        else:
            state = next_state
            timestep += 1

    dtypes = (
        np.int64,
        np.int64,
        np.float64,
        np.int64,
        np.bool_,
        np.bool_,
        np.bool_,
        np.int64,
        np.int64,
    )
    return {
        name: np.asarray([row[index] for row in rows], dtype=dtypes[index])
        for index, name in enumerate(TRANSITION_FIELDS)
    }


def validate_buffer(
    mdp: TabularMDP,
    data: dict[str, np.ndarray],
    *,
    horizon: int | None = None,
) -> None:
    """Validate schema, transition semantics, and trajectory lifecycle."""

    if set(data) != set(TRANSITION_FIELDS):
        raise AssertionError(f"Incorrect transition schema: {sorted(data)}")
    if any(not isinstance(values, np.ndarray) or values.ndim != 1 for values in data.values()):
        raise AssertionError("Every transition field must be a one-dimensional NumPy array.")
    size = len(data["state"])
    if size == 0 or any(len(values) != size for values in data.values()):
        raise AssertionError("Transition fields are empty or have unequal lengths.")
    integer_fields = ("state", "action", "next_state", "trajectory_id", "timestep")
    if any(not np.issubdtype(data[name].dtype, np.integer) for name in integer_fields):
        raise AssertionError("State, action, trajectory, and timestep fields must be integers.")
    if any(data[name].dtype != np.bool_ for name in ("done", "terminal", "truncated")):
        raise AssertionError("done, terminal, and truncated fields must be boolean arrays.")
    if not np.issubdtype(data["reward"].dtype, np.floating) or not np.all(
        np.isfinite(data["reward"])
    ):
        raise AssertionError("Rewards must be finite floating-point values.")
    state = data["state"]
    action = data["action"]
    next_state = data["next_state"]
    if (
        np.any(state < 0)
        or np.any(state >= mdp.num_states)
        or np.any(next_state < 0)
        or np.any(next_state >= mdp.num_states)
        or np.any(action < 0)
        or np.any(action >= mdp.num_actions)
    ):
        raise AssertionError("State or action index lies outside the MDP.")
    if np.any(np.isin(state, mdp.terminal_states)):
        raise AssertionError("A natural trajectory cannot act from a terminal state.")
    if np.any(mdp.P[state, action, next_state] <= 0.0):
        raise AssertionError("A sampled transition is inconsistent with P.")
    if not np.allclose(data["reward"], mdp.R[state, action, next_state]):
        raise AssertionError("A sampled reward is inconsistent with R.")
    expected_terminal = np.isin(next_state, mdp.terminal_states)
    if not np.array_equal(data["terminal"], expected_terminal):
        raise AssertionError("terminal is inconsistent with terminal_states.")
    if not np.array_equal(data["done"], data["terminal"]):
        raise AssertionError("done denotes environment termination, not truncation.")
    if np.any(data["terminal"] & data["truncated"]):
        raise AssertionError("A transition cannot be terminal and truncated together.")
    trajectory_id = data["trajectory_id"]
    timestep = data["timestep"]
    if trajectory_id[0] != 0 or timestep[0] != 0:
        raise AssertionError("The first trajectory and timestep must both start at zero.")
    if np.any(trajectory_id < 0) or np.any(timestep < 0):
        raise AssertionError("Trajectory IDs and timesteps must be non-negative.")
    starts = np.concatenate(([True], trajectory_id[1:] != trajectory_id[:-1]))
    if np.any(mdp.initial_state_distribution[state[starts]] <= 0.0):
        raise AssertionError("Every trajectory must start in the support of rho0.")
    if horizon is not None:
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        if np.any(timestep >= horizon):
            raise AssertionError("A timestep reaches or exceeds the collection horizon.")
        expected_truncated = (timestep == horizon - 1) & ~data["terminal"]
        if np.any(data["truncated"] != expected_truncated):
            raise AssertionError("truncated is inconsistent with the collection horizon.")
    for index in range(1, size):
        same_trajectory = data["trajectory_id"][index] == data["trajectory_id"][index - 1]
        if same_trajectory:
            if data["terminal"][index - 1] or data["truncated"][index - 1]:
                raise AssertionError("A trajectory continues after an end transition.")
            if data["timestep"][index] != data["timestep"][index - 1] + 1:
                raise AssertionError("Timestep is not consecutive within a trajectory.")
            if data["state"][index] != data["next_state"][index - 1]:
                raise AssertionError("Transitions are not contiguous within a trajectory.")
        else:
            if data["trajectory_id"][index] != data["trajectory_id"][index - 1] + 1:
                raise AssertionError("Trajectory IDs must increase consecutively.")
            if not (data["terminal"][index - 1] or data["truncated"][index - 1]):
                raise AssertionError("A new trajectory starts before the previous one ended.")
            if data["timestep"][index] != 0:
                raise AssertionError("A new trajectory must start at timestep zero.")


def _reachable_states(mdp: TabularMDP, action_mask: np.ndarray) -> np.ndarray:
    reachable = np.asarray(mdp.initial_state_distribution > 0.0, dtype=np.bool_)
    pending = np.flatnonzero(reachable).tolist()
    while pending:
        state = pending.pop()
        actions = np.flatnonzero(action_mask[state])
        if len(actions) == 0:
            continue
        successors = np.any(mdp.P[state, actions] > 0.0, axis=0)
        new_states = successors & ~reachable
        reachable |= new_states
        pending.extend(np.flatnonzero(new_states).tolist())
    return reachable


def coverage_metrics(
    mdp: TabularMDP,
    policy: BehaviorPolicy,
    data: dict[str, np.ndarray],
    *,
    n_min: int = DEFAULT_N_MIN,
    _reachability: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Measure support after collection without modifying the buffer."""

    if n_min <= 0:
        raise ValueError("n_min must be positive.")
    counts = np.zeros((mdp.num_states, mdp.num_actions), dtype=np.int64)
    np.add.at(counts, (data["state"], data["action"]), 1)
    observed = counts > 0
    all_actions = np.ones_like(observed, dtype=np.bool_)
    policy_action_support = policy.probabilities > 0.0
    if _reachability is None:
        environment_reachable_states = _reachable_states(mdp, all_actions)
        policy_reachable_states = _reachable_states(mdp, policy_action_support)
    else:
        environment_reachable_states, policy_reachable_states = _reachability
    # Terminal states are reachable destinations, but collection resets as soon
    # as one is entered. They are therefore not decision states and their
    # absorbing actions can never appear as source state-action pairs.
    terminal_mask = np.zeros(mdp.num_states, dtype=np.bool_)
    terminal_mask[np.asarray(mdp.terminal_states, dtype=np.int64)] = True
    environment_decision_states = environment_reachable_states & ~terminal_mask
    policy_decision_states = policy_reachable_states & ~terminal_mask
    reachable_pairs = environment_decision_states[:, None] & all_actions
    supported_pairs = policy_decision_states[:, None] & policy_action_support

    reachable_pair_count = int(reachable_pairs.sum())
    supported_pair_count = int(supported_pairs.sum())
    observed_supported = observed & supported_pairs
    supported_counts = counts[supported_pairs]
    positive_counts = counts[counts > 0]
    probabilities = positive_counts / positive_counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    entropy_denominator = np.log(max(reachable_pair_count, 2))

    return {
        "num_transitions": int(len(data["state"])),
        "num_trajectories_started": int(len(np.unique(data["trajectory_id"]))),
        "observed_state_count": int(len(np.unique(data["state"]))),
        "environment_reachable_state_count": int(environment_reachable_states.sum()),
        "environment_reachable_decision_state_count": int(
            environment_decision_states.sum()
        ),
        "reachable_terminal_state_count": int(
            (environment_reachable_states & terminal_mask).sum()
        ),
        "observed_sa_count": int(observed.sum()),
        "environment_reachable_sa_count": reachable_pair_count,
        "policy_supported_sa_count": supported_pair_count,
        "observed_sa_fraction_all": float(observed.mean()),
        "observed_reachable_sa_fraction": float(
            (observed & reachable_pairs).sum() / max(reachable_pair_count, 1)
        ),
        "observed_policy_support_fraction": float(
            observed_supported.sum() / max(supported_pair_count, 1)
        ),
        "fraction_policy_support_at_least_n_min": float(
            np.mean(supported_counts >= n_min) if supported_pair_count else 0.0
        ),
        "minimum_policy_support_count": int(
            supported_counts.min() if supported_pair_count else 0
        ),
        "normalized_visitation_entropy": float(
            entropy / entropy_denominator if entropy_denominator > 0.0 else 0.0
        ),
        "n_min": n_min,
    }


def _prefix(data: dict[str, np.ndarray], size: int) -> dict[str, np.ndarray]:
    return {name: values[:size] for name, values in data.items()}


def _normalized_coverage_thresholds(
    thresholds: Sequence[float] | None,
) -> tuple[list[float], str]:
    if thresholds is None:
        return list(DEFAULT_COVERAGE_THRESHOLDS), "protocol_config_thresholds"
    actual = [float(value) for value in thresholds]
    if (
        len(actual) != len(DEFAULT_COVERAGE_LABELS) - 1
        or len(set(actual)) != len(actual)
        or actual != sorted(actual)
        or any(not 0.0 < value < 1.0 for value in actual)
    ):
        raise ValueError(
            "coverage thresholds must be unique, increasing, and inside (0,1)."
        )
    return actual, "configured_threshold_table"


def assign_coverage_strata(
    records: list[dict[str, Any]],
    *,
    thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Assign low/medium/high/very-high strata after coverage is measured.

    The default thresholds come from ``protocol_config.json``.  Strata are
    attached to every nested prefix using the maximum-prefix measurement so a
    single collection does not change stratum merely because a shorter prefix
    is selected.
    """

    if not records:
        return {
            "metric": _COVERAGE_CONFIG["primary_metric"],
            "mode": "none",
            "thresholds": [],
            "labels": list(DEFAULT_COVERAGE_LABELS),
        }
    max_records: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for record in records:
        key = (
            record["instance_id"],
            record["policy_family"],
            record["stream"],
            record["stream_seed"],
        )
        previous = max_records.get(key)
        if previous is None or record["size"] > previous["size"]:
            max_records[key] = record
    actual_thresholds, mode = _normalized_coverage_thresholds(thresholds)
    names = DEFAULT_COVERAGE_LABELS
    max_strata = {
        key: names[
            int(
                np.digitize(
                    record["coverage"]["observed_reachable_sa_fraction"],
                    actual_thresholds,
                    right=True,
                )
            )
        ]
        for key, record in max_records.items()
    }
    for record in records:
        key = (
            record["instance_id"],
            record["policy_family"],
            record["stream"],
            record["stream_seed"],
        )
        record["coverage_stratum"] = max_strata[key]
        record["coverage_stratum_basis"] = "maximum_nested_prefix"
    return {
        "metric": _COVERAGE_CONFIG["primary_metric"],
        "mode": mode,
        "thresholds": actual_thresholds,
        "labels": list(names),
    }


def _logical_sizes(
    mdp: TabularMDP,
    max_size: int | None,
    *,
    stream: str,
    multipliers: Sequence[int] | None = None,
) -> tuple[int, ...]:
    if stream not in {"pilot", "training"}:
        raise ValueError("stream must be pilot or training.")
    effective_multipliers = (
        tuple(multipliers)
        if multipliers is not None
        else (PILOT_SIZE_MULTIPLIERS if stream == "pilot" else SIZE_MULTIPLIERS)
    )
    planned = tuple(
        multiplier * mdp.num_states * mdp.num_actions
        for multiplier in effective_multipliers
    )
    if max_size is None:
        return planned
    selected = tuple(size for size in planned if size <= max_size)
    return selected or (max_size,)


def _stream_seed(mdp_seed: int, stream: str, seed_index: int) -> int:
    return int((mdp_seed + STREAM_SEED_OFFSET[stream] + seed_index) % (2**32 - 1))


def _write_buffer(path: Path, data: dict[str, np.ndarray]) -> str:
    np.savez_compressed(path, **data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collection_record(
    *,
    spec: BenchmarkSpec,
    mdp: TabularMDP,
    policy: BehaviorPolicy,
    config: CollectionConfig,
    data: dict[str, np.ndarray],
    size: int,
    file_name: str,
    file_sha256: str,
    policy_expected_return: float,
    policy_normalized_return: float,
    reachability: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    prefix = _prefix(data, size)
    state_action_count = mdp.num_states * mdp.num_actions
    size_multiplier = (
        size // state_action_count if size % state_action_count == 0 else None
    )
    if config.stream == "pilot" and size_multiplier in PILOT_SIZE_MULTIPLIERS:
        budget_role = "adaptive_pilot_candidate"
    elif config.stream == "training" and size_multiplier in SIZE_MULTIPLIERS:
        budget_role = "training_grid"
    else:
        budget_role = "review_deviation"
    descriptive_id = (
        f"{spec.instance_id}__{policy.family}__{config.stream}-"
        f"{config.stream_seed}__n-{size}"
    )
    return {
        "dataset_id": descriptive_id,
        "data_only_dataset_id": _stable_identifier("dataset", descriptive_id),
        "data_only_group_id": _stable_identifier("instance", spec.instance_id),
        **asdict(config),
        "num_states": mdp.num_states,
        "num_actions": mdp.num_actions,
        "initial_state_distribution": _json_ready(mdp.initial_state_distribution),
        "size": size,
        "size_multiplier": size_multiplier,
        "budget_role": budget_role,
        "is_reference_n0": bool(
            config.stream == "pilot"
            and size_multiplier == DEFAULT_N0_COEFFICIENT
        ),
        "slice": [0, size],
        "file": file_name,
        "sha256": file_sha256,
        "content_sha256": dataset_content_sha256(prefix),
        "policy_parameters": _json_ready(policy.parameters),
        "policy_uses_q_star": policy.uses_q_star,
        "policy_sha256": policy_sha256(policy),
        "policy_expected_return": policy_expected_return,
        "policy_normalized_return": policy_normalized_return,
        "coverage": coverage_metrics(mdp, policy, prefix, _reachability=reachability),
    }


def _validate_seed_indices(name: str, values: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(values)
    if (
        not normalized
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in normalized
        )
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError(f"{name} must contain unique non-negative integers.")
    return normalized


def _generation_deviations(
    specs: Sequence[BenchmarkSpec],
    *,
    pilot_seeds: Sequence[int],
    training_seeds: Sequence[int],
    max_size: int | None,
    horizon: int,
    coverage_thresholds: Sequence[float],
    pilot_size_multipliers: Sequence[int],
    streams: Sequence[str],
) -> list[dict[str, Any]]:
    deviations: list[dict[str, Any]] = []

    def record(setting: str, registered: Any, effective: Any) -> None:
        if registered != effective:
            deviations.append(
                {
                    "setting": setting,
                    "registered": _json_ready(registered),
                    "effective": _json_ready(effective),
                }
            )

    registered_specs = benchmark_specs()
    record(
        "benchmark.instances",
        [spec.to_dict() for spec in registered_specs],
        [spec.to_dict() for spec in specs],
    )
    record("collection.pilot_seed_indices", DEFAULT_PILOT_SEEDS, tuple(pilot_seeds))
    record(
        "collection.training_seed_indices",
        DEFAULT_TRAINING_SEEDS,
        tuple(training_seeds),
    )
    record("collection.horizon", DEFAULT_HORIZON, horizon)
    record("collection.max_size_cap", None, max_size)
    record(
        "collection.pilot_size_multipliers",
        PILOT_SIZE_MULTIPLIERS,
        tuple(pilot_size_multipliers),
    )
    record("collection.streams", ("pilot", "training"), tuple(streams))
    record(
        "coverage.thresholds",
        DEFAULT_COVERAGE_THRESHOLDS,
        tuple(coverage_thresholds),
    )
    return deviations


def _data_only_index(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build an estimator-facing index with oracle and coverage labels removed."""

    allowed_record_fields = (
        "stream",
        "max_size",
        "horizon",
        "num_states",
        "num_actions",
        "initial_state_distribution",
        "size",
        "size_multiplier",
        "budget_role",
        "is_reference_n0",
        "slice",
        "file",
        "sha256",
        "content_sha256",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "data-only estimator input index; manifest.json is oracle audit metadata",
        "transition_fields": list(TRANSITION_FIELDS),
        "adaptive_pilot": {
            "entrypoint": "code/adaptive_pilot.py::select_effective_pilot_with_oeh",
            "candidate_multipliers": list(PILOT_SIZE_MULTIPLIERS),
            "n_min": int(_ESTIMATOR_CONFIG["n_min"]),
            "min_trajectory_support": int(
                _ESTIMATOR_CONFIG["min_trajectory_support"]
            ),
            "bootstrap_scheme": _ESTIMATOR_CONFIG["bootstrap_scheme"],
            "bootstrap_dirichlet_alpha": float(
                _ESTIMATOR_CONFIG["bootstrap_dirichlet_alpha"]
            ),
            "min_valid_bootstrap_fraction": float(
                _ESTIMATOR_CONFIG["min_valid_bootstrap_fraction"]
            ),
            "selection": (
                "first prefix that does not abstain, is not right-censored, "
                "and meets min_valid_bootstrap_fraction"
            ),
            "cap_outcome": "abstain/censor without extending or altering data",
        },
        "datasets": [
            {
                "dataset_id": record["data_only_dataset_id"],
                "group_id": record["data_only_group_id"],
                **{field: record[field] for field in allowed_record_fields},
            }
            for record in records
        ],
    }


def generate_benchmark_bundle(
    output: Path,
    *,
    specs: Iterable[BenchmarkSpec] | None = None,
    pilot_seeds: Sequence[int] = DEFAULT_PILOT_SEEDS,
    training_seeds: Sequence[int] = DEFAULT_TRAINING_SEEDS,
    max_size: int | None = None,
    horizon: int = DEFAULT_HORIZON,
    coverage_thresholds: Sequence[float] | None = None,
    run_mode: str = "protocol",
    deviation_reason: str | None = None,
    pilot_size_multipliers: Sequence[int] = PILOT_SIZE_MULTIPLIERS,
    streams: Sequence[str] = ("pilot", "training"),
) -> dict[str, Any]:
    """Generate benchmark trajectories and return the written manifest."""

    requested_specs = benchmark_specs() if specs is None else list(specs)
    if not requested_specs:
        raise ValueError("At least one benchmark specification is required.")
    if len({spec.instance_id for spec in requested_specs}) != len(requested_specs):
        raise ValueError("Benchmark instance IDs must be unique.")
    normalized_pilot_seeds = _validate_seed_indices("pilot_seeds", pilot_seeds)
    normalized_training_seeds = _validate_seed_indices("training_seeds", training_seeds)
    normalized_pilot_multipliers = tuple(pilot_size_multipliers)
    if (
        not normalized_pilot_multipliers
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in normalized_pilot_multipliers
        )
        or tuple(sorted(set(normalized_pilot_multipliers)))
        != normalized_pilot_multipliers
    ):
        raise ValueError(
            "pilot_size_multipliers must be unique increasing positive integers."
        )
    normalized_streams = tuple(streams)
    if (
        not normalized_streams
        or len(set(normalized_streams)) != len(normalized_streams)
        or any(stream not in {"pilot", "training"} for stream in normalized_streams)
    ):
        raise ValueError(
            "streams must be a non-empty unique subset of pilot/training."
        )
    pilot_namespace = {
        (STREAM_SEED_OFFSET["pilot"] + index) % (2**32 - 1)
        for index in normalized_pilot_seeds
    }
    training_namespace = {
        (STREAM_SEED_OFFSET["training"] + index) % (2**32 - 1)
        for index in normalized_training_seeds
    }
    if pilot_namespace & training_namespace:
        raise ValueError("Pilot and training seed namespaces overlap.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if max_size is not None and max_size <= 0:
        raise ValueError("max_size must be positive when provided.")
    actual_thresholds = tuple(
        _normalized_coverage_thresholds(coverage_thresholds)[0]
    )
    if run_mode not in {"protocol", "review", "test"}:
        raise ValueError("run_mode must be one of: protocol, review, test.")
    if run_mode == "protocol" and PROTOCOL_CONFIG["protocol_status"] != "frozen":
        pending = ", ".join(PROTOCOL_CONFIG["pending_validations"])
        raise ValueError(
            "Full protocol generation is blocked while protocol_status is candidate. "
            f"Pending validations: {pending}."
        )
    if run_mode == "review" and any(
        spec.split == "evaluation" for spec in requested_specs
    ):
        raise ValueError(
            "Review runs cannot materialize evaluation instances; use the development split."
        )
    deviations = _generation_deviations(
        requested_specs,
        pilot_seeds=normalized_pilot_seeds,
        training_seeds=normalized_training_seeds,
        max_size=max_size,
        horizon=horizon,
        coverage_thresholds=actual_thresholds,
        pilot_size_multipliers=normalized_pilot_multipliers,
        streams=normalized_streams,
    )
    if deviations and run_mode == "protocol":
        settings = ", ".join(item["setting"] for item in deviations)
        raise ValueError(
            "A protocol run cannot override registered settings "
            f"({settings}). Use run_mode='review' with a deviation reason."
        )
    if deviations and (not deviation_reason or not deviation_reason.strip()):
        raise ValueError("Every non-protocol run with overrides needs a deviation reason.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new directory to avoid mixing runs."
        )
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    instance_records: list[dict[str, Any]] = []
    total_rejections = 0

    for requested_spec in requested_specs:
        spec, mdp, rejections = resolve_dynamic_range(requested_spec)
        total_rejections += rejections
        range_value, optimal_return, random_return = dynamic_range(mdp)
        # OOD illustrations are outside the registered primary/control panels.
        dynamics_audit = (audit_dynamics(mdp, spec.family)
                          if spec.family in PRIMARY_FAMILIES + CONTROL_FAMILIES
                          else None)
        instance_records.append(
            {
                **spec.to_dict(),
                "requested_seed": requested_spec.seed,
                "dynamic_range_rejections": rejections,
                "num_states": mdp.num_states,
                "num_actions": mdp.num_actions,
                "gamma": mdp.gamma,
                "terminal_states": list(mdp.terminal_states),
                "reward_min": float(mdp.R.min()),
                "reward_max": float(mdp.R.max()),
                "dynamic_range": range_value,
                "optimal_return": optimal_return,
                "random_return": random_return,
                "metadata": _json_ready(mdp.metadata),
                "dynamics_audit": dynamics_audit,
            }
        )
        policies = behavior_policy_suite(
            mdp, seed=spec.seed + 3_000_000, variant_index=spec.ordinal
        )
        for policy in policies:
            # Reachability depends on the MDP/policy, never on a prefix or seed.
            reachability = (
                _reachable_states(mdp, np.ones_like(policy.probabilities, dtype=bool)),
                _reachable_states(mdp, policy.probabilities > 0.0),
            )
            # Exact policy quality is oracle metadata for dataset auditing; it
            # is never exposed to the data-only OEH estimator.
            policy_return = evaluate_policy(
                mdp, policy.probabilities
            ).expected_return
            policy_normalized_return = (
                policy_return - random_return
            ) / range_value
            stream_seeds = {
                "pilot": normalized_pilot_seeds,
                "training": normalized_training_seeds,
            }
            for stream in normalized_streams:
                seed_indices = stream_seeds[stream]
                sizes = _logical_sizes(
                    mdp,
                    max_size,
                    stream=stream,
                    multipliers=(
                        normalized_pilot_multipliers
                        if stream == "pilot"
                        else SIZE_MULTIPLIERS
                    ),
                )
                collection_size = max(sizes)
                for seed_index in seed_indices:
                    stream_seed = _stream_seed(spec.seed, stream, int(seed_index))
                    config = CollectionConfig(
                        instance_id=spec.instance_id,
                        benchmark_split=spec.split,
                        mdp_seed=spec.seed,
                        policy_family=policy.family,
                        policy_seed=policy.seed,
                        stream=stream,
                        stream_seed=stream_seed,
                        max_size=collection_size,
                        horizon=horizon,
                    )
                    data = collect_trajectories(
                        mdp,
                        policy,
                        seed=stream_seed,
                        size=collection_size,
                        horizon=horizon,
                    )
                    validate_buffer(mdp, data, horizon=horizon)
                    buffer_id = _stable_identifier(
                        "buffer",
                        spec.instance_id,
                        spec.seed,
                        policy.family,
                        stream,
                        int(seed_index),
                    )
                    path = output / f"buffer-{buffer_id}.npz"
                    sha256 = _write_buffer(path, data)
                    for size in sizes:
                        records.append(
                            _collection_record(
                                spec=spec,
                                mdp=mdp,
                                policy=policy,
                                config=config,
                                data=data,
                                size=size,
                                file_name=path.name,
                                file_sha256=sha256,
                                policy_expected_return=policy_return,
                                policy_normalized_return=policy_normalized_return,
                                reachability=reachability,
                            )
                        )

    stratification = assign_coverage_strata(records, thresholds=coverage_thresholds)
    effective_protocol = deepcopy(PROTOCOL_CONFIG)
    effective_protocol["collection"]["pilot_seed_indices"] = list(
        normalized_pilot_seeds
    )
    effective_protocol["collection"]["training_seed_indices"] = list(
        normalized_training_seeds
    )
    effective_protocol["collection"]["horizon"] = horizon
    effective_protocol["collection"]["pilot_size_multipliers"] = list(
        normalized_pilot_multipliers
    )
    effective_protocol["coverage"]["thresholds"] = list(actual_thresholds)
    effective_protocol["run_scope"] = {
        "mode": run_mode,
        "instances": [spec.to_dict() for spec in requested_specs],
        "max_size_cap": max_size,
        "streams": list(normalized_streams),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_config": _json_ready(PROTOCOL_CONFIG),
        "protocol_status": PROTOCOL_CONFIG["protocol_status"],
        "pending_validations": list(PROTOCOL_CONFIG["pending_validations"]),
        "effective_protocol": _json_ready(effective_protocol),
        "run_mode": run_mode,
        "protocol_deviations": deviations,
        "deviation_reason": deviation_reason.strip() if deviation_reason else None,
        "estimator_input_index": "data_only_index.json",
        "access_policy": {
            "data_only_estimator": "NPZ transitions plus data_only_index.json",
            "oracle_audit_only": (
                "manifest.json fields for exact MDP, policy quality, and coverage labels"
            ),
        },
        "generator_provenance": _generator_provenance(),
        "benchmark_design": {
            "evaluation_count": int(PROTOCOL_CONFIG["benchmark"]["evaluation_count"]),
            "development_count": int(PROTOCOL_CONFIG["benchmark"]["development_count"]),
            "ood_count": len(PROTOCOL_CONFIG["benchmark"]["ood_representatives"]),
        },
        "benchmark_seed": BENCHMARK_SEED,
        "storage": "natural trajectory buffers with nested prefixes",
        "transition_fields": list(TRANSITION_FIELDS),
        "size_multipliers": list(SIZE_MULTIPLIERS),
        "training_size_multipliers": list(SIZE_MULTIPLIERS),
        "pilot_size_multipliers": list(normalized_pilot_multipliers),
        "reference_pilot_n0_coefficient": DEFAULT_N0_COEFFICIENT,
        "pilot_training_independence": "disjoint deterministic seed namespaces",
        "coverage_rule": "generate_then_measure_then_stratify; no post-hoc filtering",
        "coverage_stratification": stratification,
        "dynamic_range_rejection_count": total_rejections,
        "instance_count": len(instance_records),
        "physical_buffer_count": len(
            {(record["file"], record["sha256"]) for record in records}
        ),
        "logical_dataset_count": len(records),
        "instances": instance_records,
        "datasets": records,
    }
    data_only_path = output / "data_only_index.json"
    data_only_path.write_text(
        json.dumps(_json_ready(_data_only_index(records)), indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["estimator_input_index_sha256"] = hashlib.sha256(
        data_only_path.read_bytes()
    ).hexdigest()
    # Write the audit manifest last; its presence indicates the run reached the
    # final indexing and provenance stage.
    (output / "manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def smoke_test() -> None:
    """Exercise all five policies on the original 14 path-test environments."""

    for environment_index, environment_id in enumerate(sorted(ENVIRONMENT_REGISTRY)):
        mdp = make_env(environment_id, seed=42)
        policies = behavior_policy_suite(
            mdp, seed=70_000 + environment_index * 10, variant_index=environment_index
        )
        for policy_index, policy in enumerate(policies):
            seed = 80_000 + environment_index * 100 + policy_index
            first = collect_trajectories(mdp, policy, seed=seed, size=120, horizon=50)
            second = collect_trajectories(mdp, policy, seed=seed, size=120, horizon=50)
            validate_buffer(mdp, first, horizon=50)
            if any(not np.array_equal(first[key], second[key]) for key in first):
                raise AssertionError("Dataset collection is not reproducible by seed.")
            coverage_metrics(mdp, policy, first)
    print(
        "PASS: 14/14 path-test environments x 5 behavior families; "
        "natural trajectories, schema, coverage and reproducibility valid."
    )


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return values


def _parse_floats(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated thresholds.")
    return values


def _parse_streams(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError(
            "Expected pilot, training, or pilot,training."
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets/tabular_v2"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--limit-instances", type=int)
    parser.add_argument(
        "--split",
        choices=("development", "ood"),
        help="Subset for review runs; defaults to development.",
    )
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--pilot-seeds", type=_parse_ints, default=DEFAULT_PILOT_SEEDS)
    parser.add_argument(
        "--training-seeds", type=_parse_ints, default=DEFAULT_TRAINING_SEEDS
    )
    parser.add_argument("--coverage-thresholds", type=_parse_floats)
    parser.add_argument(
        "--pilot-size-multipliers",
        type=_parse_ints,
        default=PILOT_SIZE_MULTIPLIERS,
        help="Review-only diagnostic pilot grid; protocol default stays 2,5,10.",
    )
    parser.add_argument(
        "--streams",
        type=_parse_streams,
        default=("pilot", "training"),
        help="Review-only subset of pilot,training.",
    )
    parser.add_argument(
        "--run-mode",
        choices=("protocol", "review"),
        default="protocol",
        help="Use review for an intentionally reduced or overridden diagnostic run.",
    )
    parser.add_argument(
        "--deviation-reason",
        help="Required when a review run differs from protocol_config.json.",
    )
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    specs = benchmark_specs()
    selected_split = args.split
    if args.run_mode == "review" and selected_split is None:
        selected_split = "development"
    if selected_split is not None:
        specs = [spec for spec in specs if spec.split == selected_split]
    if args.limit_instances is not None:
        if args.limit_instances <= 0:
            parser.error("--limit-instances must be positive.")
        specs = specs[: args.limit_instances]
    manifest = generate_benchmark_bundle(
        args.output,
        specs=specs,
        pilot_seeds=args.pilot_seeds,
        training_seeds=args.training_seeds,
        max_size=args.max_size,
        horizon=args.horizon,
        coverage_thresholds=args.coverage_thresholds,
        run_mode=args.run_mode,
        deviation_reason=args.deviation_reason,
        pilot_size_multipliers=args.pilot_size_multipliers,
        streams=args.streams,
    )
    print(
        f"PASS: wrote {manifest['instance_count']} instances, "
        f"{manifest['physical_buffer_count']} buffers and "
        f"{manifest['logical_dataset_count']} logical datasets to {args.output}."
    )


if __name__ == "__main__":
    main()

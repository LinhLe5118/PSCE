from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from environments.behavior_policies import BehaviorPolicy, POLICY_FAMILIES, behavior_policy_suite
from environments.benchmark_environments import benchmark_specs, make_benchmark_mdp
from scripts.data.generate_offline_datasets import (
    TRANSITION_FIELDS,
    assign_coverage_strata,
    collect_trajectories,
    coverage_metrics,
    generate_benchmark_bundle,
    validate_buffer,
)
from environments.tabular_environments import make_combination_lock


class DatasetGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = benchmark_specs()[0]
        cls.mdp = make_benchmark_mdp(cls.spec)
        cls.policies = behavior_policy_suite(
            cls.mdp, seed=1234, variant_index=cls.spec.ordinal
        )

    def test_five_families_and_only_reference_uses_q_star(self) -> None:
        self.assertEqual(tuple(policy.family for policy in self.policies), POLICY_FAMILIES)
        self.assertEqual(sum(policy.uses_q_star for policy in self.policies), 1)
        for policy in self.policies:
            policy.validate(self.mdp)
            np.testing.assert_allclose(policy.probabilities.sum(axis=1), 1.0)

    def test_natural_trajectory_schema_and_reproducibility(self) -> None:
        policy = self.policies[2]
        first = collect_trajectories(
            self.mdp, policy, seed=99, size=250, horizon=30
        )
        second = collect_trajectories(
            self.mdp, policy, seed=99, size=250, horizon=30
        )
        self.assertEqual(set(first), set(TRANSITION_FIELDS))
        validate_buffer(self.mdp, first)
        for field in TRANSITION_FIELDS:
            np.testing.assert_array_equal(first[field], second[field])

    def test_pilot_and_training_seed_namespaces_produce_different_streams(self) -> None:
        policy = self.policies[0]
        pilot = collect_trajectories(
            self.mdp, policy, seed=10_000_123, size=200, horizon=50
        )
        training = collect_trajectories(
            self.mdp, policy, seed=20_000_123, size=200, horizon=50
        )
        self.assertFalse(
            np.array_equal(pilot["action"], training["action"])
            and np.array_equal(pilot["next_state"], training["next_state"])
        )

    def test_coverage_metrics_are_bounded(self) -> None:
        policy = self.policies[4]
        data = collect_trajectories(
            self.mdp, policy, seed=123, size=300, horizon=40
        )
        metrics = coverage_metrics(self.mdp, policy, data)
        for key in (
            "observed_sa_fraction_all",
            "observed_reachable_sa_fraction",
            "observed_policy_support_fraction",
            "fraction_policy_support_at_least_n_min",
            "normalized_visitation_entropy",
        ):
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_terminal_actions_are_excluded_from_coverage_denominator(self) -> None:
        mdp = make_combination_lock(2, seed=7, gamma=0.95)
        policy = BehaviorPolicy(
            family="structured_random",
            probabilities=np.full((mdp.num_states, mdp.num_actions), 0.5),
            parameters={},
            seed=1,
            uses_q_star=False,
        )
        data = {
            "state": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "action": np.asarray([0, 1, 0, 1], dtype=np.int64),
            "trajectory_id": np.arange(4, dtype=np.int64),
        }
        metrics = coverage_metrics(mdp, policy, data)
        self.assertEqual(metrics["environment_reachable_state_count"], 3)
        self.assertEqual(metrics["environment_reachable_decision_state_count"], 2)
        self.assertEqual(metrics["reachable_terminal_state_count"], 1)
        self.assertEqual(metrics["environment_reachable_sa_count"], 4)
        self.assertEqual(metrics["policy_supported_sa_count"], 4)
        self.assertEqual(metrics["observed_reachable_sa_fraction"], 1.0)

    def test_buffer_validator_rejects_invalid_lifecycle_and_indices(self) -> None:
        policy = self.policies[2]
        data = collect_trajectories(
            self.mdp, policy, seed=314, size=20, horizon=3
        )
        validate_buffer(self.mdp, data, horizon=3)

        invalid_lifecycle = {name: values.copy() for name, values in data.items()}
        invalid_lifecycle["trajectory_id"][1] = 1
        with self.assertRaises(AssertionError):
            validate_buffer(self.mdp, invalid_lifecycle, horizon=3)

        invalid_index = {name: values.copy() for name, values in data.items()}
        invalid_index["state"][0] = self.mdp.num_states
        with self.assertRaises(AssertionError):
            validate_buffer(self.mdp, invalid_index, horizon=3)

    def test_coverage_thresholds_must_be_strict_and_bounded(self) -> None:
        record = {
            "instance_id": "x",
            "policy_family": "x",
            "stream": "pilot",
            "stream_seed": 1,
            "size": 1,
            "coverage": {"observed_reachable_sa_fraction": 0.5},
        }
        for thresholds in ((0.5, 0.5, 0.75), (-0.1, 0.5, 0.75), (0.5, 0.25, 0.75)):
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                assign_coverage_strata([record.copy()], thresholds=thresholds)

    def test_small_bundle_writes_auditable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = generate_benchmark_bundle(
                output,
                specs=[self.spec],
                pilot_seeds=(0,),
                training_seeds=(0,),
                max_size=80,
                horizon=30,
                run_mode="test",
                deviation_reason="unit-test reduced bundle",
            )
            stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            data_only = json.loads(
                (output / "data_only_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["instance_count"], 1)
            self.assertEqual(
                manifest["instances"][0]["dynamics_audit"]["panel"],
                "stochastic_primary",
            )
            self.assertTrue(manifest["instances"][0]["dynamics_audit"]["oracle_audit_only"])
            self.assertEqual(stored["physical_buffer_count"], 10)
            self.assertEqual(
                stored["generator_provenance"]["source_files"],
                [
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
                ],
            )
            self.assertEqual(len(stored["generator_provenance"]["source_sha256"]), 64)
            self.assertEqual(
                stored["coverage_stratification"]["thresholds"],
                [0.25, 0.5, 0.75],
            )
            self.assertTrue(
                all(len(record["content_sha256"]) == 64 for record in stored["datasets"])
            )
            self.assertTrue(
                all(np.isfinite(record["policy_expected_return"]) for record in stored["datasets"])
            )
            self.assertTrue(
                all(
                    np.isfinite(record["policy_normalized_return"])
                    for record in stored["datasets"]
                )
            )
            self.assertEqual(
                stored["coverage_rule"],
                "generate_then_measure_then_stratify; no post-hoc filtering",
            )
            self.assertTrue(all(record["sha256"] for record in stored["datasets"]))
            self.assertEqual(stored["run_mode"], "test")
            self.assertTrue(stored["protocol_deviations"])
            self.assertEqual(
                stored["deviation_reason"], "unit-test reduced bundle"
            )
            self.assertEqual(len(data_only["datasets"]), len(stored["datasets"]))
            self.assertEqual(
                data_only["adaptive_pilot"]["bootstrap_scheme"],
                "bayesian_trajectory_dirichlet",
            )
            self.assertEqual(
                data_only["adaptive_pilot"]["bootstrap_dirichlet_alpha"],
                1.0,
            )
            self.assertEqual(
                data_only["adaptive_pilot"]["min_trajectory_support"],
                2,
            )
            self.assertTrue(
                all(
                    record["num_states"] == self.mdp.num_states
                    and record["num_actions"] == self.mdp.num_actions
                    and np.isclose(sum(record["initial_state_distribution"]), 1.0)
                    for record in data_only["datasets"]
                )
            )
            self.assertEqual(
                stored["estimator_input_index_sha256"],
                hashlib.sha256(
                    (output / "data_only_index.json").read_bytes()
                ).hexdigest(),
            )
            forbidden = {
                "dynamics_audit",
                "panel",
                "coverage",
                "coverage_stratum",
                "policy_expected_return",
                "policy_normalized_return",
                "policy_parameters",
                "policy_uses_q_star",
                "instance_id",
                "benchmark_split",
                "policy_family",
                "stream_seed",
            }
            self.assertTrue(
                all(
                    forbidden.isdisjoint(record)
                    for record in data_only["datasets"]
                )
            )
            self.assertTrue(
                all(
                    record["file"].startswith("buffer-")
                    and "uniform_garnet" not in record["file"]
                    for record in data_only["datasets"]
                )
            )

    def test_adaptive_n0_candidates_and_reference_prefix_are_explicit(self) -> None:
        state_action_count = self.mdp.num_states * self.mdp.num_actions
        with tempfile.TemporaryDirectory() as temporary:
            manifest = generate_benchmark_bundle(
                Path(temporary),
                specs=[self.spec],
                pilot_seeds=(0,),
                training_seeds=(0,),
                max_size=10 * state_action_count,
                run_mode="test",
                deviation_reason="unit-test registered pilot budgets",
            )
        pilot = [record for record in manifest["datasets"] if record["stream"] == "pilot"]
        self.assertEqual({record["size_multiplier"] for record in pilot}, {2, 5, 10})
        self.assertTrue(
            all(
                record["budget_role"] == "adaptive_pilot_candidate"
                for record in pilot
            )
        )
        reference = [record for record in pilot if record["is_reference_n0"]]
        self.assertEqual(len(reference), 5)
        self.assertTrue(
            all(
                record["size"] == 5 * state_action_count
                for record in reference
            )
        )

    def test_review_can_generate_pilot_only_twenty_fold_k2_leverage(self) -> None:
        state_action_count = self.mdp.num_states * self.mdp.num_actions
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = generate_benchmark_bundle(
                output,
                specs=[self.spec],
                pilot_seeds=(0,),
                training_seeds=(0,),
                run_mode="test",
                deviation_reason="unit-test Week 4 K2 leverage",
                pilot_size_multipliers=(2, 5, 10, 20, 40),
                streams=("pilot",),
            )
            self.assertEqual(manifest["pilot_size_multipliers"], [2, 5, 10, 20, 40])
            self.assertEqual({row["stream"] for row in manifest["datasets"]}, {"pilot"})
            self.assertEqual(
                {row["size_multiplier"] for row in manifest["datasets"]},
                {2, 5, 10, 20, 40},
            )
            self.assertEqual(manifest["physical_buffer_count"], 5)
            self.assertEqual(manifest["logical_dataset_count"], 25)
            for file_name in {row["file"] for row in manifest["datasets"]}:
                with np.load(output / file_name, allow_pickle=False) as loaded:
                    self.assertEqual(len(loaded["state"]), 40 * state_action_count)

    def test_invalid_diagnostic_grid_or_stream_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for multipliers, streams in (
                ((5, 2), ("pilot",)),
                ((2, 2), ("pilot",)),
                ((2, 5), ("evaluation",)),
            ):
                with self.subTest(multipliers=multipliers, streams=streams), self.assertRaises(ValueError):
                    generate_benchmark_bundle(
                        Path(temporary) / f"case-{len(multipliers)}-{streams[0]}",
                        specs=[self.spec],
                        pilot_seeds=(0,),
                        training_seeds=(0,),
                        run_mode="test",
                        deviation_reason="invalid diagnostic request",
                        pilot_size_multipliers=multipliers,
                        streams=streams,
                    )

    def test_protocol_run_rejects_unrecorded_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                generate_benchmark_bundle(
                    Path(temporary),
                    specs=[self.spec],
                    pilot_seeds=(0,),
                    training_seeds=(0,),
                    max_size=20,
                )

    def test_candidate_protocol_blocks_full_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "full"
            with self.assertRaisesRegex(ValueError, "protocol_status is candidate"):
                generate_benchmark_bundle(output)
            self.assertFalse(output.exists())

    def test_review_run_is_confined_to_development(self) -> None:
        development_spec = next(
            spec for spec in benchmark_specs() if spec.split == "development"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                generate_benchmark_bundle(
                    root / "evaluation",
                    specs=[self.spec],
                    pilot_seeds=(0,),
                    training_seeds=(0,),
                    max_size=20,
                    run_mode="review",
                    deviation_reason="must not inspect evaluation",
                )
            manifest = generate_benchmark_bundle(
                root / "development",
                specs=[development_spec],
                pilot_seeds=(0,),
                training_seeds=(0,),
                max_size=20,
                run_mode="review",
                deviation_reason="development-only probe",
            )
            self.assertEqual(manifest["instance_count"], 1)
            self.assertTrue(
                all(instance["split"] == "development" for instance in manifest["instances"])
            )

    def test_nonempty_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "unrelated.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate_benchmark_bundle(
                    output,
                    specs=[self.spec],
                    pilot_seeds=(0,),
                    training_seeds=(0,),
                    max_size=20,
                    run_mode="test",
                    deviation_reason="output safety test",
                )
            self.assertEqual(
                (output / "unrelated.txt").read_text(encoding="utf-8"), "keep"
            )

    def test_small_bundle_content_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = []
            for name in ("first", "second"):
                manifests.append(
                    generate_benchmark_bundle(
                        root / name,
                        specs=[self.spec],
                        pilot_seeds=(0,),
                        training_seeds=(0,),
                        max_size=40,
                        horizon=10,
                        run_mode="test",
                        deviation_reason="reproducibility test",
                    )
                )
            first_records = manifests[0]["datasets"]
            second_records = manifests[1]["datasets"]
            self.assertEqual(first_records, second_records)
            self.assertEqual(
                manifests[0]["generator_provenance"],
                manifests[1]["generator_provenance"],
            )


if __name__ == "__main__":
    unittest.main()

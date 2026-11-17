from __future__ import annotations

from copy import deepcopy
import unittest

from environments.behavior_policies import POLICY_FAMILIES
from environments.benchmark_environments import (
    DEVELOPMENT_FAMILY_COUNTS,
    DEVELOPMENT_COUNT,
    DYNAMIC_RANGE_THRESHOLD,
    EVALUATION_COUNT,
    EVALUATION_FAMILY_COUNTS,
    OOD_FAMILIES,
    STRUCTURAL_FAMILIES,
)
from scripts.data.generate_offline_datasets import (
    DEFAULT_COVERAGE_THRESHOLDS,
    DEFAULT_HORIZON,
    DEFAULT_N0_COEFFICIENT,
    PILOT_SIZE_MULTIPLIERS,
    SIZE_MULTIPLIERS,
)
from config.protocol_config import PROTOCOL_CONFIG, validate_protocol_config


class ProtocolConfigTests(unittest.TestCase):
    def test_machine_readable_config_drives_runtime_constants(self) -> None:
        benchmark = PROTOCOL_CONFIG["benchmark"]
        policies = PROTOCOL_CONFIG["behavior_policies"]
        collection = PROTOCOL_CONFIG["collection"]
        coverage = PROTOCOL_CONFIG["coverage"]
        estimator = PROTOCOL_CONFIG["estimator"]
        learning = PROTOCOL_CONFIG["learning"]
        self.assertEqual(EVALUATION_COUNT, benchmark["evaluation_count"])
        self.assertEqual(DEVELOPMENT_COUNT, benchmark["development_count"])
        self.assertEqual(STRUCTURAL_FAMILIES, tuple(benchmark["structural_families"]))
        self.assertEqual(OOD_FAMILIES, tuple(benchmark["ood_representatives"]))
        self.assertEqual(POLICY_FAMILIES, tuple(policies["families"]))
        self.assertEqual(DEFAULT_HORIZON, collection["horizon"])
        self.assertEqual(SIZE_MULTIPLIERS, tuple(collection["size_multipliers"]))
        self.assertEqual(
            PILOT_SIZE_MULTIPLIERS,
            tuple(collection["pilot_size_multipliers"]),
        )
        self.assertEqual(DEFAULT_N0_COEFFICIENT, estimator["n0_coefficient"])
        self.assertEqual(
            PILOT_SIZE_MULTIPLIERS,
            tuple(estimator["n0_candidate_coefficients"]),
        )
        self.assertEqual(
            estimator["n0_selection_rule"],
            "first_usable_data_only_prefix",
        )
        self.assertEqual(DEFAULT_HORIZON, estimator["horizon"])
        self.assertEqual(coverage["n_min"], estimator["n_min"])
        self.assertEqual(
            estimator["bootstrap_scheme"],
            "bayesian_trajectory_dirichlet",
        )
        self.assertEqual(estimator["bootstrap_dirichlet_alpha"], 1.0)
        self.assertEqual(estimator["min_trajectory_support"], 2)
        self.assertGreaterEqual(DEFAULT_HORIZON, estimator["k_max"])
        self.assertEqual(
            DEFAULT_COVERAGE_THRESHOLDS, tuple(coverage["thresholds"])
        )
        self.assertFalse(coverage["post_hoc_transition_filtering"])
        self.assertEqual(learning["algorithms"], ["cfqi_tabular", "iql_tabular"])
        self.assertEqual(learning["success_runs_total"], len(collection["training_seed_indices"]))
        self.assertEqual(
            EVALUATION_FAMILY_COUNTS, benchmark["evaluation_family_counts"]
        )
        self.assertEqual(
            DEVELOPMENT_FAMILY_COUNTS, benchmark["development_family_counts"]
        )
        self.assertEqual(PROTOCOL_CONFIG["protocol_status"], "candidate")
        # Every registered validation is closed (2026-09-05), so the list is
        # empty and the status is 'candidate' by decision rather than by
        # backlog -- see protocol_status_note. Both facts are asserted so that
        # emptying the list without recording why, or flipping the status
        # without the team overriding WEEK2_DECISION_GATE.md sec 11, turns the
        # suite red.
        self.assertEqual(PROTOCOL_CONFIG["pending_validations"], [])
        self.assertTrue(PROTOCOL_CONFIG["protocol_status_note"].strip())
        for name, record in PROTOCOL_CONFIG["resolved_validations"].items():
            with self.subTest(validation=name):
                self.assertIn(
                    record["outcome"],
                    {"closed_positive", "closed_negative", "closed_with_recorded_limitation"},
                )
                self.assertTrue(record["evidence"].strip())
                self.assertTrue(record["resolved_date"].strip())
        self.assertEqual(
            DYNAMIC_RANGE_THRESHOLD,
            0.01 / (1.0 - float(benchmark["gamma"])),
        )

    def test_invalid_coverage_thresholds_are_rejected(self) -> None:
        for thresholds in ([0.5, 0.5, 0.75], [-0.1, 0.5, 0.75], [0.5, 0.25, 0.75]):
            with self.subTest(thresholds=thresholds):
                config = deepcopy(PROTOCOL_CONFIG)
                config["coverage"]["thresholds"] = thresholds
                with self.assertRaises(ValueError):
                    validate_protocol_config(config)

    def test_policy_grids_must_have_equal_length(self) -> None:
        config = deepcopy(PROTOCOL_CONFIG)
        config["behavior_policies"]["epsilon_grid"] = [0.1]
        with self.assertRaises(ValueError):
            validate_protocol_config(config)

    def test_family_quotas_must_match_registered_totals(self) -> None:
        config = deepcopy(PROTOCOL_CONFIG)
        config["benchmark"]["evaluation_family_counts"]["bottleneck"] += 1
        with self.assertRaises(ValueError):
            validate_protocol_config(config)

    def test_recipe_levels_must_be_unique(self) -> None:
        config = deepcopy(PROTOCOL_CONFIG)
        config["benchmark"]["environment_recipes"]["uniform_garnet"][
            "num_actions"
        ] = [2, 2, 4]
        with self.assertRaises(ValueError):
            validate_protocol_config(config)

    def test_frozen_protocol_cannot_have_pending_validations(self) -> None:
        # The pending item is planted rather than inherited from the live
        # config: the rule under test is the validator's, and it must keep
        # holding now that the shipped config has closed every validation.
        config = deepcopy(PROTOCOL_CONFIG)
        config["protocol_status"] = "frozen"
        config["pending_validations"] = ["some_unfinished_check"]
        with self.assertRaises(ValueError):
            validate_protocol_config(config)

    def test_frozen_protocol_is_accepted_once_nothing_is_pending(self) -> None:
        config = deepcopy(PROTOCOL_CONFIG)
        config["protocol_status"] = "frozen"
        config["pending_validations"] = []
        validate_protocol_config(config)  # must not raise

    def test_estimator_and_collection_constants_cannot_conflict(self) -> None:
        mutations = (
            ("collection_horizon", lambda c: c["collection"].update(horizon=500)),
            ("k_max", lambda c: c["estimator"].update(k_max=101)),
            ("coverage_n_min", lambda c: c["coverage"].update(n_min=5)),
            (
                "bootstrap_scheme",
                lambda c: c["estimator"].update(
                    bootstrap_scheme="trajectory_resample_with_replacement"
                ),
            ),
            (
                "bootstrap_alpha",
                lambda c: c["estimator"].update(
                    bootstrap_dirichlet_alpha=0.5
                ),
            ),
            (
                "trajectory_support",
                lambda c: c["estimator"].update(min_trajectory_support=1),
            ),
            (
                "pilot_n0",
                lambda c: c["collection"].update(
                    pilot_size_multipliers=[2, 10]
                ),
            ),
            (
                "adaptive_grid",
                lambda c: (
                    c["estimator"].update(n0_candidate_coefficients=[2, 5, 20]),
                    c["collection"].update(pilot_size_multipliers=[2, 5, 20]),
                ),
            ),
            (
                "learning_seed_count",
                lambda c: c["learning"].update(success_runs_total=9),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                config = deepcopy(PROTOCOL_CONFIG)
                mutate(config)
                with self.assertRaises(ValueError):
                    validate_protocol_config(config)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
import sys

import numpy as np

from environments.benchmark_environments import (
    DEVELOPMENT_FAMILY_COUNTS,
    ENVIRONMENT_RECIPES,
    EVALUATION_FAMILY_COUNTS,
    OOD_FAMILIES,
    STRUCTURAL_FAMILIES,
    benchmark_specs,
    make_benchmark_mdp,
    resolve_dynamic_range,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from exact_solver import passes_dynamic_range_filter  # noqa: E402


class BenchmarkEnvironmentTests(unittest.TestCase):
    def test_candidate_120_plus_15_plus_3_topology(self) -> None:
        specs = benchmark_specs()
        split_counts = Counter(spec.split for spec in specs)
        self.assertEqual(
            split_counts, {"evaluation": 120, "development": 15, "ood": 3}
        )
        evaluation_families = Counter(
            spec.family for spec in specs if spec.split == "evaluation"
        )
        development_families = Counter(
            spec.family for spec in specs if spec.split == "development"
        )
        self.assertEqual(
            EVALUATION_FAMILY_COUNTS,
            {
                "uniform_garnet": 30,
                "forward_layered": 36,
                "lock_structured": 36,
                "bottleneck": 18,
            },
        )
        self.assertEqual(evaluation_families, EVALUATION_FAMILY_COUNTS)
        self.assertEqual(development_families, DEVELOPMENT_FAMILY_COUNTS)
        self.assertEqual(
            tuple(spec.family for spec in specs if spec.split == "ood"),
            OOD_FAMILIES,
        )

    def test_evaluation_recipe_axes_have_balanced_marginals(self) -> None:
        stratified_axes = {
            "uniform_garnet": (
                "num_states",
                "num_actions",
                "branching_factor",
                "dirichlet_concentration",
                "reward_sparsity",
            ),
            "forward_layered": (
                "depth",
                "width",
                "num_actions",
                "branching_factor",
                "dirichlet_concentration",
            ),
            "lock_structured": ("chain_length",),
            "bottleneck": (
                "community_size",
                "num_actions",
                "branching_factor",
                "bridge_probability",
            ),
        }
        recipe_key = {"lock_structured": {"chain_length": "length"}}
        mdps_by_family: dict[str, list] = {family: [] for family in STRUCTURAL_FAMILIES}
        specs_by_family: dict[str, list] = {family: [] for family in STRUCTURAL_FAMILIES}
        for spec in benchmark_specs():
            if spec.split != "evaluation":
                continue
            specs_by_family[spec.family].append(spec)
            mdps_by_family[spec.family].append(make_benchmark_mdp(spec))

        for family in STRUCTURAL_FAMILIES:
            self.assertEqual(
                [spec.family_ordinal for spec in specs_by_family[family]],
                list(range(EVALUATION_FAMILY_COUNTS[family])),
            )
            for metadata_axis in stratified_axes[family]:
                config_axis = recipe_key.get(family, {}).get(
                    metadata_axis, metadata_axis
                )
                allowed = set(ENVIRONMENT_RECIPES[family][config_axis])
                counts = Counter(
                    mdp.metadata[metadata_axis] for mdp in mdps_by_family[family]
                )
                self.assertEqual(set(counts), allowed)
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_one_instance_per_family_is_a_valid_scaled_mdp(self) -> None:
        seen: set[str] = set()
        for spec in benchmark_specs():
            if spec.family in seen:
                continue
            mdp = make_benchmark_mdp(spec)
            mdp.validate()
            self.assertGreaterEqual(float(mdp.R.min()), 0.0)
            self.assertLessEqual(float(mdp.R.max()), 1.0)
            self.assertTrue(np.allclose(mdp.P.sum(axis=2), 1.0))
            seen.add(spec.family)
        self.assertEqual(
            seen, set(STRUCTURAL_FAMILIES) | set(OOD_FAMILIES)
        )

    def test_dynamic_range_resolution_is_reproducible(self) -> None:
        source = benchmark_specs()[13]
        first_spec, first_mdp, first_rejections = resolve_dynamic_range(source)
        second_spec, second_mdp, second_rejections = resolve_dynamic_range(source)
        self.assertEqual(first_spec, second_spec)
        self.assertEqual(first_rejections, second_rejections)
        np.testing.assert_array_equal(first_mdp.P, second_mdp.P)
        np.testing.assert_array_equal(first_mdp.R, second_mdp.R)
        self.assertTrue(passes_dynamic_range_filter(first_mdp))

    def test_dynamic_range_regeneration_preserves_recipe_stratum(self) -> None:
        source = next(
            spec
            for spec in benchmark_specs()
            if spec.instance_id == "evaluation_forward_layered_054"
        )
        original = make_benchmark_mdp(source)
        resolved_spec, resolved, rejections = resolve_dynamic_range(source)
        self.assertGreater(rejections, 0)
        self.assertNotEqual(source.seed, resolved_spec.seed)
        for field in (
            "depth",
            "width",
            "num_actions",
            "branching_factor",
            "dirichlet_concentration",
            "backbone_probability",
        ):
            self.assertEqual(original.metadata[field], resolved.metadata[field])


if __name__ == "__main__":
    unittest.main()

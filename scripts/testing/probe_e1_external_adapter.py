"""Run real tabular paper-method cores on one generated fixture buffer."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from environments.behavior_policies import behavior_policy_suite
from environments.benchmark_environments import BenchmarkSpec, resolve_dynamic_range
from evaluation.e1_external import build_external_train_score_callback, verify_external_components
from scripts.data.generate_offline_datasets import collect_trajectories, dynamic_range, _write_buffer


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = BenchmarkSpec(instance_id="fixture_external_lock_000", split="fixture",
                        family="lock_structured", seed=42, ordinal=0,
                        family_ordinal=0)


class E1ExternalAdapterTests(unittest.TestCase):
    def test_spibb_author_and_vi_lcb_paper_spec_fit_and_score(self):
        external = ROOT / "tmp"
        if not all((external / name).is_dir() for name in ("external_spibb", "external_vi_lcb")):
            self.skipTest("Pinned external checkouts are absent")
        resolved, mdp, _ = resolve_dynamic_range(FIXTURE)
        _, optimal, random = dynamic_range(mdp)
        behavior = behavior_policy_suite(mdp, seed=resolved.seed + 3_000_000,
                                         variant_index=resolved.ordinal)[0]
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            file_name = "one_x10_buffer.npz"
            digest = _write_buffer(
                dataset / file_name,
                collect_trajectories(mdp, behavior, seed=resolved.seed + 20_000_000,
                                     size=10 * mdp.num_states * mdp.num_actions),
            )
            record = {
                "instance_id": resolved.instance_id, "benchmark_split": "evaluation",
                "policy_family": behavior.family, "stream": "training",
                "mdp_seed": resolved.seed, "stream_seed": resolved.seed + 20_000_000,
                "size_multiplier": 10, "file": file_name, "sha256": digest,
                "size": 10 * mdp.num_states * mdp.num_actions,
            }
            instance = {**resolved.to_dict(), "split": "evaluation", "random_return": random,
                        "dynamic_range": optimal - random}
            callback = build_external_train_score_callback(
                dataset, {"instances": [instance], "datasets": [record]}, external)
            for method in ("pib_spibb_n10_author", "vi_lcb_cb005_author_empirical"):
                job = {
                    "method": method, "instance_id": record["instance_id"],
                    "policy_family": record["policy_family"],
                    "size_multiplier": 10, "training_seed": 0,
                    "buffer_file": file_name, "buffer_sha256": digest,
                }
                result = callback(job)
                self.assertEqual(result["status"], "observed")
                self.assertTrue(math.isfinite(result["normalized_return"]))
                self.assertEqual(result["success"], result["normalized_return"] >= .95)

    def test_bcq_and_cql_500_update_capacity_probe(self):
        verify_external_components(ROOT / "tmp")
        resolved, mdp, _ = resolve_dynamic_range(FIXTURE)
        _, optimal, random = dynamic_range(mdp)
        behavior = behavior_policy_suite(mdp, seed=resolved.seed + 3_000_000,
                                         variant_index=resolved.ordinal)[0]
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            file_name = "one_x10_buffer.npz"
            digest = _write_buffer(
                dataset / file_name,
                collect_trajectories(mdp, behavior, seed=resolved.seed + 20_000_000,
                                     size=10 * mdp.num_states * mdp.num_actions),
            )
            record = {
                "instance_id": resolved.instance_id, "benchmark_split": "evaluation",
                "policy_family": behavior.family, "stream": "training",
                "mdp_seed": resolved.seed, "stream_seed": resolved.seed + 20_000_000,
                "size_multiplier": 10, "file": file_name, "sha256": digest,
                "size": 10 * mdp.num_states * mdp.num_actions,
            }
            instance = {**resolved.to_dict(), "split": "evaluation", "random_return": random,
                        "dynamic_range": optimal - random}
            callback = build_external_train_score_callback(
                dataset, {"instances": [instance], "datasets": [record]}, ROOT / "tmp")
            for method in ("discrete_bcq_author", "discrete_cql_d3rlpy"):
                job = {
                    "method": method, "instance_id": record["instance_id"],
                    "policy_family": record["policy_family"],
                    "size_multiplier": 10, "training_seed": 0,
                    "buffer_file": file_name, "buffer_sha256": digest,
                }
                result = callback(job)
                print(f"{method}: fit_seconds={result['fit_seconds']:.3f}, peak_vram_mib={result['peak_vram_mib']:.1f}")
                self.assertEqual(result["status"], "observed")
                self.assertTrue(math.isfinite(result["normalized_return"]))


if __name__ == "__main__":
    unittest.main()

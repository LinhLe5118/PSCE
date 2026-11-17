"""Local pinned-source and CUDA probe for the E1 paper-method panel."""

from pathlib import Path
import unittest

from evaluation.e1_external import verify_external_components
from evaluation.confirmatory import json_sha256
from evaluation.e1_pipeline import EXTERNAL_METHODS


ROOT = Path(__file__).resolve().parents[2]


class E1ExternalComponentsTests(unittest.TestCase):
    def test_real_pinned_implementations_are_available(self):
        external = ROOT / "tmp"
        if not all((external / name).is_dir() for name in
                   ("external_spibb", "external_vi_lcb", "external_bcq", "external_d3rlpy")):
            self.skipTest("Author checkouts are not present in this environment")
        result = verify_external_components(external)
        self.assertEqual(tuple(result["methods"]), EXTERNAL_METHODS)
        self.assertEqual(set(result["commits"]), set(EXTERNAL_METHODS))
        self.assertEqual(result["source_type"]["vi_lcb_cb005_author_empirical"],
                         "discounted_paper_spec_port")
        self.assertEqual(len(json_sha256(result["runtime"])), 64)


if __name__ == "__main__":
    unittest.main()

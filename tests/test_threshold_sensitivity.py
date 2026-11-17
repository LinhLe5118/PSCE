"""Threshold-sensitivity relabelling (task card T6)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.analysis import threshold_sensitivity as T
from scripts.analysis.compare_difficulty import suffix_event


def test_vectorized_suffix_rule_matches_the_registered_rule():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 11, (500, 6))
    np.testing.assert_array_equal(T.suffix_events(counts, 8), [suffix_event(tuple(int(c) for c in row)) for row in counts])


def test_more_required_seeds_never_lowers_the_requirement():
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 11, (300, 6))
    assert (T.suffix_events(counts, 9) >= T.suffix_events(counts, 8)).all()
    assert (T.suffix_events(counts, 8) >= T.suffix_events(counts, 7)).all()


def test_recorded_run_reproduced_the_official_labels():
    out = T.OUTPUT
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    checks = manifest["official_label_reproduction"]
    assert len(checks) == 11 and all(v["label_mismatches"] == 0 and v["success_flag_mismatches"] == 0 for v in checks.values())
    sens = pd.read_csv(out / "sensitivity.csv")
    assert len(sens) == 5 * 2 * 9
    for name, digest in manifest["outputs_sha256"].items():
        assert T.X.sha(out / name) == digest

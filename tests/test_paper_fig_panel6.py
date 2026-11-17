"""The sixth-panel paper figure is a faithful view of the committed panel-6 results."""
import hashlib
import json

import pandas as pd
import pytest

from scripts.analysis import paper_fig_panel6 as F


def test_relative_gains_match_registered_gates_and_committed_csv():
    gains = F.relative_gains().set_index("method")
    # registered gates F1 and F2 of the sixth panel (result.json / T10 report)
    assert gains.loc["b2_plus_prefix_fqe", "ibs_ge_gain_pct"] == pytest.approx(25.59, abs=0.01)
    assert gains.loc["b2_plus_prefix_fqe", "ll_ext_gain_pct"] == pytest.approx(17.86, abs=0.01)
    assert gains.loc["b2_matched", "ibs_ge_gain_pct"] == pytest.approx(43.84, abs=0.01)
    assert gains.loc["oeh_full_head", "ibs_ge_gain_pct"] < 0  # the gate hurt on this panel
    assert (gains.filter(like="_low_pct").to_numpy() <= gains.filter(like="_gain_pct").to_numpy()).all()
    committed = pd.read_csv(F.OUT / "relative_gains.csv").set_index("method")
    pd.testing.assert_frame_equal(committed[gains.columns], gains, atol=1e-6, check_exact=False)


def test_manifest_records_the_committed_sources():
    manifest = json.loads((F.OUT / "manifest.json").read_text(encoding="utf-8"))
    for path, digest in manifest["sources"].items():
        assert hashlib.sha256((F.ROOT / path).read_bytes()).hexdigest() == digest
    assert (F.ROOT / "paper/figures/fig_panel6.pdf").is_file()

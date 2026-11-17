"""Check the finite-horizon oracle definitions on a tiny known MDP."""
from types import SimpleNamespace
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.analysis.e1_oracle_qvi_diagnostic import finite_horizon_online_bounds


ROOT = Path(__file__).resolve().parents[1]


def test_qvi_and_epw_separate_random_tail_from_zero_tail():
    # Action 0 at the start reaches a delayed reward; action 1 pays 0.2 now.
    # Random-tail Q already sees the delayed reward, whereas one-step EPW does not.
    p = np.zeros((3, 2, 3))
    r = np.zeros_like(p)
    p[0, 0, 1] = 1
    p[0, 1, 2] = 1
    r[0, 1, 2] = .2
    p[1, :, 2] = 1
    r[1, :, 2] = 1
    p[2, :, 2] = 1
    mdp = SimpleNamespace(
        P=p, R=r, expected_rewards=(p * r).sum(axis=2),
        gamma=.95, num_states=3, num_actions=2,
        initial_state_distribution=np.array([1., 0., 0.]),
    )
    result = finite_horizon_online_bounds(mdp)
    assert result["stochastic_eh_min_qvi_k_h100"] == 1
    assert result["epw_window_h100"] == 2
    assert np.isfinite(result["stochastic_eh_hbar_h100"])
    assert np.isfinite(result["log_covering_length_upper_bound"])


def test_e1_comparison_commands_start_in_fresh_process():
    for module in (
        "scripts.analysis.compare_e1_dataset_signals",
        "scripts.analysis.e1_oracle_qvi_diagnostic",
    ):
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout


def test_published_e1_posthoc_panels_keep_all_original_mdps():
    for directory, script, expected_files in (
        ("e1_dataset_signals_posthoc_2026-09-30", "compare_e1_dataset_signals.py", 9000),
        ("e1_oracle_qvi_posthoc_2026-09-30", "e1_oracle_qvi_diagnostic.py", 120),
    ):
        artifact = ROOT / "artifacts" / directory
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "posthoc_original_E1_adaptively_reused_not_confirmation"
        source = ROOT / "scripts/analysis" / script
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["script_sha256"]
        for name, digest in manifest["output_hashes"].items():
            assert hashlib.sha256((artifact / name).read_bytes()).hexdigest() == digest
        row_file = "predictions.csv" if "dataset_signals" in directory else "oracle_diagnostics.csv"
        with (artifact / row_file).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == expected_files
        assert len({row["mdp_id"] for row in rows}) == 120

    oracle = ROOT / "artifacts/e1_oracle_qvi_posthoc_2026-09-30"
    with (oracle / "rank_comparison.csv").open(encoding="utf-8", newline="") as stream:
        ranks = list(csv.DictReader(stream))
    covering = {row["panel"]: row for row in ranks if row["metric"] == "log_cover_online_bound_tL"}
    assert int(covering["primary"]["n_mdps"]) == 84
    assert int(covering["control"]["n_mdps"]) == 36
    sensitivity = next(row for row in ranks if row["metric"].endswith("finite_only_sensitivity"))
    assert int(sensitivity["n_mdps"]) == 79

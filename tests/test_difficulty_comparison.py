"""Check the published post-hoc requirement scores from row-level predictions."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.analysis.compare_difficulty import GRID, aft_interval, suffix_event
from scripts.analysis.export_difficulty_table import render


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/difficulty_e1_posthoc_2026-09-30"


def test_requirement_suffix_and_censor_intervals():
    assert suffix_event((8, 8, 7, 8, 8, 8)) == 3
    assert suffix_event((10, 10, 10, 10, 10, 10)) == 0
    assert suffix_event((0, 0, 0, 0, 0, 7)) == 6
    assert aft_interval(0) == (0.0, 1.0)
    assert aft_interval(3) == (GRID[2], GRID[3])
    assert aft_interval(6) == (50.0, float("inf"))
    with pytest.raises(ValueError):
        suffix_event((8, 8))
    with pytest.raises(ValueError):
        aft_interval(7)


def test_difficulty_cli_starts_from_fresh_process():
    for module, arguments in (
        ("scripts.analysis.compare_difficulty", ("--help",)),
        ("scripts.analysis.export_difficulty_table", ("--help",)),
        ("scripts.analysis.export_difficulty_table", ("--check",)),
    ):
        run = subprocess.run(
            [sys.executable, "-m", module, *arguments], cwd=ROOT, capture_output=True,
        )
        assert run.returncode == 0, run.stderr.decode(errors="replace")


def test_published_scores_recompute_from_out_of_fold_predictions():
    manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "posthoc_original_E1_adaptively_reused_not_confirmation"
    for name, digest in manifest["output_hashes"].items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((ROOT / "scripts/analysis/compare_difficulty.py").read_bytes()).hexdigest() == manifest["script_sha256"]
    with (ARTIFACT / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        detail = list(csv.DictReader(stream))
    with (ARTIFACT / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))
    assert (ROOT / "tests/fixtures/paper_history/difficulty_table.tex").read_text(encoding="utf-8") == render(summary)
    assert len(detail) == 1800 * 5 and len(summary) == 10
    assert len({(r["mdp_id"], r["policy_family"], r["algorithm"], r["method"]) for r in detail}) == len(detail)
    for row in detail:
        event = int(row["event"])
        probabilities = [float(row[f"p_{i}"]) for i in range(7)]
        assert 0 <= event <= 6
        assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-10)
        assert all(0 <= value <= 1 for value in probabilities)
        brier = sum(
            (sum(probabilities[:i + 1]) - float(event <= i)) ** 2
            for i in range(6)
        ) / 6
        assert math.isclose(brier, float(row["brier"]), abs_tol=1e-10)
        assert int(row["correct_category"]) == int(max(range(7), key=probabilities.__getitem__) == event)
        if "raw_proxy" not in row["method"]:
            assert math.isclose(
                -math.log(max(probabilities[event], 1e-12)),
                float(row["log_loss"]), abs_tol=1e-10,
            )
    for row in summary:
        cases = [r for r in detail if r["panel"] == row["panel"] and r["method"] == row["method"]]
        assert len(cases) == int(row["n"])
        assert math.isclose(sum(float(r["brier"]) for r in cases) / len(cases),
                            float(row["integrated_brier"]), abs_tol=1e-10)
        assert math.isclose(sum(int(r["correct_category"]) for r in cases) / len(cases),
                            float(row["event_category_accuracy"]), abs_tol=1e-10)
        if "raw_proxy" not in row["method"]:
            assert math.isclose(sum(float(r["log_loss"]) for r in cases) / len(cases),
                                float(row["seven_class_log_loss"]), abs_tol=1e-10)

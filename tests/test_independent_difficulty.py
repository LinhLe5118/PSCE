"""Operational guards for the separate, outcome-sealed difficulty runner."""
from pathlib import Path
import subprocess
import sys

from scripts.analysis.independent_difficulty import _job_file


ROOT = Path(__file__).resolve().parents[1]


def test_new_validation_clis_start_in_fresh_process():
    for module in ("scripts.analysis.independent_difficulty",
                   "scripts.analysis.run_unseen_validation",
                   "scripts.analysis.prepare_unseen_validation"):
        completed = subprocess.run([sys.executable, "-m", module, "--help"],
                                   cwd=ROOT, capture_output=True)
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")


def test_different_training_budgets_cannot_overwrite_a_job():
    job = {"instance_id": "new_mdp", "policy_family": "behavior",
           "size_multiplier": 1, "training_seed": 0, "method": "cfqi_tabular"}
    another_budget = {**job, "size_multiplier": 50}
    assert _job_file(job) != _job_file(another_budget)

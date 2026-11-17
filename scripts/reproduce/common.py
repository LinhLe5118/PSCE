"""Paths, the three final tests and the scoring rules shared by the reproduction steps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]
FLOOR = 1e-12  # log-loss floor of the registered analyses
LEARNERS = {"behavior_cloning_tabular": "BC", "cfqi_tabular": "CFQI", "iql_tabular": "IQL"}

# The three final tests of the thesis. "codes" maps the four regenerated forecasts to the method names used in
# each test's sealed files (Test 1 predates the final names: v2 = OEH, oeh_final = full head, legacy = robust head).
TESTS = {
    "Test 1": {"folder": "oeh_v2_sealed_test_2026-10-14", "panels": ["internal", "colosseum2029", "spibb2", "forest"],
               "runner": "scripts.analysis.run_oeh_v2_sealed_test_20261014",
               "codes": {"oeh": "v2", "full": "oeh_final", "robust": "legacy", "b2": "b2_matched"}},
    "Test 2": {"folder": "oeh_fifth_panel_2026-10-15", "panels": ["frozenlake", "gambler"],
               "runner": "scripts.analysis.run_oeh_fifth_panel_20261015",
               "codes": {"oeh": "oeh", "full": "oeh_full_head", "robust": "robust", "b2": "b2_matched"}},
    "Test 3": {"folder": "oeh_panel6_2026-10-15", "panels": ["mdpp", "labmaze"],
               "runner": "scripts.analysis.run_oeh_panel6_20261015",
               "codes": {"oeh": "oeh", "full": "oeh_full_head", "robust": "robust", "b2": "b2_matched"}},
}
# OEH with one evidence block removed from the full and finite-only heads (field sets of the sealed ablation arms).
BLOCKS = {"no_simulation": "oeh_without_direct_oehb", "no_planning_risk": "oeh_without_oehh_risk",
          "no_pilot_diagnostics": "oeh_without_legacy_diagnostics"}
# The four parts of the final tests used by the thesis tables.
PARTS = ("Test 1 internal", "Test 1 external", "Test 2", "Test 3")


def part_of(test: str, panel: str) -> str:
    if test == "Test 1":
        return "Test 1 internal" if panel == "internal" else "Test 1 external"
    return test


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, doc) -> None:
    Path(path).write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")


def row_scores(p: np.ndarray, y: np.ndarray) -> dict:
    """Per-case scores: IBS over the six budgets, seven-class log loss, and the extrapolation scores of Test 3."""
    F = np.cumsum(p, 1)[:, :6]
    sq = (F - (y[:, None] <= np.arange(6))) ** 2
    group = np.minimum(np.maximum(y, 3), 6) - 3                     # x1..x10 | x20 | x50 | >x50
    grouped = np.column_stack((p[:, :4].sum(1), p[:, 4], p[:, 5], p[:, 6]))
    rows = np.arange(len(y))
    return {"ibs": sq.mean(1), "ll": -np.log(np.maximum(p[rows, y], FLOOR)), "ibs_ge": sq[:, 3:6].mean(1),
            "ll_ext": -np.log(np.maximum(grouped[rows, group], FLOOR))}

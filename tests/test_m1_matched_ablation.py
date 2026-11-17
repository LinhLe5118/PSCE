"""M1 acceptance: column lists, no outcome leakage, seven classes, namespace, hashes."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import m1_feature_lineage as L
from scripts.analysis import run_m1_matched_ablation as M
from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS

ROOT = Path(__file__).resolve().parents[1]
SEALED = L.PROTOCOL.exists()
needs_cohort = pytest.mark.skipif(not SEALED, reason="sealed cohort-7 artifacts are not available")


@pytest.fixture(scope="module")
def rows():
    return L.lineage()


@pytest.fixture(scope="module")
def arms(rows):
    return L.arms(rows)


@needs_cohort
def test_lineage_assigns_every_sealed_column_once(rows):
    fields = json.loads(L.PROTOCOL.read_text(encoding="utf-8"))["oeh_final"]["fields"]
    assert [row["column"] for row in rows] == fields
    assert [row["position"] for row in rows] == list(range(108))
    counts = pd.Series([row["block"] for row in rows]).value_counts().to_dict()
    assert counts == {"G1": 18, "G2": 18, "G3": 2, "G4": 11, "G5": 13, "G6": 18, "G7": 10, "G8": 18}
    for row in rows:
        assert row["producer"] and row["depends_on"] and row["meaning"] and row["source_field"]
        assert row["component"] in L.COMPONENTS
        assert row["source_artifact"] in L.PREOUTCOME_SOURCES
        assert row["learner_scope"] in ("shared", "own", "cross_learner")
        assert row["outcome_access"] is False and row["privileged_access"] is False


@needs_cohort
def test_arm_column_lists(arms):
    full = arms["oeh_full_108"]
    assert {name: len(fields) for name, fields in arms.items()} == {
        "b2_matched": 18, "oeh_full_108": 108, "oeh_without_direct_oehb": 70,
        "oeh_without_oehh_risk": 84, "oeh_without_legacy_diagnostics": 80}
    assert arms["b2_matched"] == tuple(FEATURES_B2)
    for fields in arms.values():
        assert len(set(fields)) == len(fields)
        assert set(FEATURES_B2) <= set(fields)
        assert list(fields) == [column for column in full if column in fields]  # sealed order
    direct = arms["oeh_without_direct_oehb"]
    assert not any(c.startswith("oehb_") or "_oehb_" in c for c in direct)
    assert {c for c in full if c.startswith("oehh_")} <= set(direct)  # OEH-H kept
    assert not any(c.startswith("oehh_") for c in arms["oeh_without_oehh_risk"])
    legacy = arms["oeh_without_legacy_diagnostics"]
    assert not any(c.startswith(("oeh3_", "lcoeh4_")) or "_lcoeh4_" in c for c in legacy)
    assert {c for c in full if "_oehb_cdf_" in c} <= set(legacy)  # OEH-B context kept


# Digest of the five arm column tuples; equal to the 2026-10-07 exploratory
# ablation definitions, which are compared directly when that folder is present.
ARMS_SHA256 = "30a3d1137924de43a28d5a6f28f6efce52908d30d7e9efab9c2006ed1c912281"


@needs_cohort
def test_arms_match_the_week8_definitions(arms):
    assert L.digest(arms) == ARMS_SHA256
    previous = M.PREVIOUS / "feature_lineage.json"
    if previous.exists():  # local evidence folder; not required in a clean clone
        old = json.loads(previous.read_text(encoding="utf-8"))
        for name, fields in arms.items():
            assert tuple(old[name]["columns"]) == fields


@needs_cohort
def test_dependencies_follow_compute_components(arms):
    assert L.components_of(arms["b2_matched"]) == ("b2_statistics",)
    direct = L.components_of(arms["oeh_without_direct_oehb"])
    assert "oehb_simulation" not in direct and {"posterior_fit", "oehh_horizon"} <= set(direct)
    legacy = L.components_of(arms["oeh_without_legacy_diagnostics"])
    assert not {"oeh3_risk", "lcoeh4_crossfit"} & set(legacy)
    assert "oehh_horizon" not in L.components_of(arms["oeh_without_oehh_risk"])
    assert set(L.components_of(arms["oeh_full_108"])) == set(L.COMPONENTS)


@needs_cohort
def test_no_outcome_or_privileged_column_in_any_arm(arms):
    from scripts.analysis import run_oeh_final_120 as R

    with (L.COHORT / "labels.csv").open(newline="", encoding="utf-8") as handle:
        label_header = set(next(csv.reader(handle))) - set(M.KEYS)
    privileged = set(R.EH) | set(R.PUBLISHED)
    for fields in arms.values():
        assert not set(fields) & label_header
        assert not set(fields) & privileged
        assert not any(c.startswith(L.PRIVILEGED_PREFIXES) for c in fields)


def test_outcome_guard_blocks_outcome_files(tmp_path):
    """Fresh process: audit hooks persist for the life of the interpreter."""
    code = f"""
import pandas as pd
from pathlib import Path
from scripts.analysis import run_m1_matched_ablation as M
cohort = Path({str(tmp_path)!r})
(cohort / 'execution').mkdir()
for name in ('labels.csv', 'pilot_features.csv', 'execution/job.csv'):
    (cohort / name).write_text('a' + chr(10) + '1' + chr(10), encoding='utf-8')
M.FORBIDDEN_FILES = frozenset(M._norm(cohort / n) for n in M.L.OUTCOME_FILES)
M.FORBIDDEN_DIRS = tuple(M._norm(cohort / n) + M.os.sep for n in M.L.OUTCOME_DIRS)
with M.outcome_guard() as opened:
    assert len(pd.read_csv(cohort / 'pilot_features.csv')) == 1
    for name in ('labels.csv', 'execution/job.csv'):
        try:
            pd.read_csv(cohort / name)
        except M.OutcomeAccessError:
            continue
        raise SystemExit('guard did not block ' + name)
assert M._norm(cohort / 'pilot_features.csv') in opened
pd.read_csv(cohort / 'labels.csv')  # allowed again once the guard is closed
print('OK')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")


def test_real_query_is_built_without_outcome_files():
    """Uses the sealed cohort when its local OEH-B groups exist (they are not in Git)."""
    if not (L.COHORT / "oeh_b_groups").is_dir():
        assert not (L.COHORT / "oeh_b_groups").exists()
        return
    code = """
from scripts.analysis import run_m1_matched_ablation as M
with M.outcome_guard() as opened:
    query = M.query_features()
assert len(query) == 1800 and 'event' not in query.columns
assert not any(c.startswith('successes_') for c in query.columns)
inputs = M.input_inventory(opened)
assert not inputs.path.str.endswith('labels.csv').any()
assert inputs.path.str.contains('oeh_b_groups').sum() == 600
print('OK')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def synthetic(rng, mdps, rows_per_mdp=2, prefix="m"):
    records = []
    for m in range(mdps):
        for b in range(rows_per_mdp):
            for learner in METHODS:
                x = rng.normal(size=3)
                event = int(np.clip(np.round(3 + 2 * x[0] + rng.normal()), 0, 6))
                records.append({"mdp_id": f"{prefix}{m}", "policy_family": f"beh{b}", "algorithm": learner,
                                "panel": "primary", "event": event, "f0": x[0], "f1": x[1], "f2": x[2]})
    return pd.DataFrame(records)


def test_forecast_is_seven_class_and_rejects_outcomes_in_query():
    rng = np.random.default_rng(7)
    train = synthetic(rng, 12)
    train.loc[train.event == 1, "event"] = 2  # one class never observed in training
    query = synthetic(rng, 3, prefix="q").drop(columns="event")
    variants = {"small": ("f0",), "large": ("f0", "f1", "f2")}
    first, tuning, runtime = M.forecast_arms(train, query, variants)
    second, _, _ = M.forecast_arms(train, query, variants)
    assert list(first.columns) == [*M.KEYS, "panel", "arm", *M.P]
    assert len(first) == 2 * len(query) and set(first.arm) == set(variants)
    for _, frame in first.groupby("arm"):
        M.check_seven_class(frame, len(query))
    np.testing.assert_array_equal(first[M.P].to_numpy(), second[M.P].to_numpy())
    assert set(tuning) == set(variants) and len(runtime) == 2
    with pytest.raises(RuntimeError, match="outcome"):
        M.forecast_arms(train, query.assign(event=0), variants)
    with pytest.raises(RuntimeError, match="outcome or privileged"):
        M.forecast_arms(train.assign(oracle_eh=1.0), query.assign(oracle_eh=1.0), {"bad": ("f0", "oracle_eh")})


def test_seven_class_check_fails_closed():
    frame = pd.DataFrame({"mdp_id": ["a"], "policy_family": ["b"], "algorithm": ["c"],
                          **{p: [1 / 7] for p in M.P}})
    M.check_seven_class(frame, 1)
    with pytest.raises(RuntimeError):
        M.check_seven_class(frame.assign(p_6=0.5), 1)
    with pytest.raises(RuntimeError):
        M.check_seven_class(frame.drop(columns="p_6").assign(p_5=2 / 7), 1)
    with pytest.raises(RuntimeError):
        M.check_seven_class(frame, 2)


def test_output_namespace_is_new_and_never_overwritten(tmp_path):
    with pytest.raises(RuntimeError, match="new namespace"):
        M.prepare_output(M.COHORT, "lineage")
    with pytest.raises(RuntimeError, match="new namespace"):
        M.prepare_output(M.PREVIOUS / "sub", "forecast")
    assert M.OUTPUT != M.PREVIOUS and M.COHORT not in M.OUTPUT.parents
    out = M.prepare_output(tmp_path / "m1", "forecast")
    (out / "prediction_seal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        M.prepare_output(out, "forecast")
    M.prepare_output(out, "score")  # other phases may still write


@needs_cohort
def test_lineage_is_frozen_with_hashes(tmp_path):
    M.run_lineage(tmp_path)
    document = json.loads((tmp_path / "feature_lineage.json").read_text(encoding="utf-8"))
    assert document["sealed_protocol_sha256"] == M.sha256(L.PROTOCOL)
    assert set(document["source_sha256"]) == {"scripts/analysis/m1_feature_lineage.py",
                                              "scripts/analysis/run_m1_matched_ablation.py"}
    M.frozen_lineage(tmp_path)
    with (tmp_path / "feature_lineage.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(RuntimeError, match="changed"):
        M.frozen_lineage(tmp_path)
    with pytest.raises(FileExistsError):
        M.run_lineage(tmp_path)


def test_sealed_m1_run_hashes_and_reproduction():
    seal = json.loads((M.OUTPUT / "prediction_seal.json").read_text(encoding="utf-8"))
    assert seal["labels_read"] is False and seal["namespace"] == M.NAMESPACE
    for name, key in (("predictions_before_label_read.csv", "prediction_sha256"),
                      ("input_inventory.csv", "input_inventory_sha256"),
                      ("source_inventory.csv", "source_inventory_sha256"),
                      ("feature_lineage.json", "lineage_json_sha256")):
        assert M.sha256(M.OUTPUT / name) == seal[key]
    manifest = json.loads((M.OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["output_sha256"].items():
        assert M.sha256(M.OUTPUT / name) == digest, name
    assert all(item["max_abs_diff"] <= M.REPRODUCE_TOLERANCE for item in seal["reproduces_sealed"].values())
    assert seal["meta_training"]["rows"] == 10800 and seal["meta_training"]["mdps"] == 720
    for name, digest in seal["cohort_sha256"].items():
        assert M.sha256(L.COHORT / name) == digest, name
    inputs = pd.read_csv(M.OUTPUT / "input_inventory.csv")
    assert len(inputs) == seal["input_files"]
    assert not inputs.path.str.endswith("oeh_confirmatory_unseen_120_2026-10-04/labels.csv").any()
    present = inputs[[(ROOT / path).exists() for path in inputs.path]]
    assert len(present) > 0  # large local inputs (pilot buffers, OEH-B groups) are not in Git
    for row in present.iloc[:: max(1, len(present) // 25)].itertuples():
        assert M.sha256(ROOT / row.path) == row.sha256
    sources = pd.read_csv(M.OUTPUT / "source_inventory.csv")
    assert {"scripts/analysis/oeh_final_forecast.py", "scripts/analysis/run_m1_matched_ablation.py",
            "scripts/analysis/m1_feature_lineage.py"} <= set(sources.path)
    predictions = pd.read_csv(M.OUTPUT / "predictions_before_label_read.csv")
    for _, frame in predictions.groupby("arm"):
        M.check_seven_class(frame, 1800)

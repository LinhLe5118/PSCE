"""Published bytes, relocation hashes, semantic equivalence and failure controls."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from oeh.evidence import (
    CONTRASTS, REFERENCE_BUNDLE, FINAL_BUNDLE, sha256_file, verify_hashes, validate_report,
    verify_definitions, verify_repository, read_json,
)

ROOT = Path(__file__).resolve().parents[1]


def test_published_sources_raw_bytes_reference_seals_and_complete_bootstrap():
    summary = verify_repository(ROOT)
    assert summary['published_evidence_verified'] is True
    assert summary['confirmation'] is False and summary['bootstrap_draws'] == 2000
    assert summary['scientific_changes'] is False


def test_portable_verification_does_not_require_unpublished_original_inputs(tmp_path):
    migration = read_json(ROOT/'artifacts/provenance/migration.json')
    for name in ('artifacts/provenance', REFERENCE_BUNDLE, FINAL_BUNDLE, 'code/oeh', 'scripts'):
        shutil.copytree(ROOT/name, tmp_path/name, ignore=shutil.ignore_patterns('__pycache__'))
    summary = verify_repository(tmp_path)
    assert summary['original_inputs_available'] == 1  # Archived original T1 contract only.
    assert summary['full_original_inputs_verified'] is False
    with pytest.raises(FileNotFoundError, match='Required local E1 input'):
        verify_repository(tmp_path, full_inputs=True)
    p = tmp_path/FINAL_BUNDLE/'comparison_x10.csv'; p.write_text('tampered\n', 'utf-8')
    with pytest.raises(ValueError, match='Changed sealed evidence'):
        verify_repository(tmp_path)


def test_relocation_requires_identical_scientific_definitions(tmp_path):
    migration = read_json(ROOT/'artifacts/provenance/migration.json')
    for name in ('artifacts/provenance', 'code/oeh', 'scripts'):
        shutil.copytree(ROOT/name, tmp_path/name, ignore=shutil.ignore_patterns('__pycache__'))
    target = tmp_path/'code/oeh/core.py'
    target.write_text(target.read_text('utf-8').replace('dimension = x.shape[1]', 'dimension = 1'), 'utf-8')
    with pytest.raises(ValueError, match='Scientific definition changed'):
        verify_definitions(tmp_path, migration)


def test_public_figure_names_cannot_allow_arithmetic_changes(tmp_path):
    migration = read_json(ROOT/'artifacts/provenance/migration.json')
    for name in ('artifacts/provenance', 'code/oeh', 'scripts'):
        shutil.copytree(ROOT/name, tmp_path/name, ignore=shutil.ignore_patterns('__pycache__'))
    target = tmp_path/'scripts/analysis/render_oeh.py'
    target.write_text(target.read_text('utf-8').replace('(1, 100)', '(1, 99)'), 'utf-8')
    with pytest.raises(ValueError, match='Scientific definition changed'):
        verify_definitions(tmp_path, migration)


def test_presentation_aliases_cannot_bypass_core_integrity():
    migration = read_json(ROOT/'artifacts/provenance/migration.json')
    migration['unchanged_definitions'][0]['presentation_only'] = True
    with pytest.raises(ValueError, match='only to the original figure renderer'):
        verify_definitions(ROOT, migration)


def test_hash_and_path_tamper_rejected(tmp_path):
    p = tmp_path/'data.json'; p.write_text('{}', 'utf-8')
    digest = sha256_file(p); verify_hashes(tmp_path, {'data.json':digest})
    p.write_text('changed', 'utf-8')
    with pytest.raises(ValueError): verify_hashes(tmp_path, {'data.json':digest})
    with pytest.raises(ValueError): verify_hashes(tmp_path, {'../escape':digest})


def test_report_rejects_incomplete_failed_or_fabricated_claims():
    observed = np.arange(1, 7.)/10; draws = np.tile(observed, (2000, 1))
    report = dict(screen_pass=True, confirmation=False, new_mdps_generated=0, bootstrap_completed=2000,
                  contrast_order=list(CONTRASTS), paired_effects=observed.tolist(), basic_one_sided_lower=observed.tolist())
    validate_report(report, dict(draws=draws, observed=observed))
    with pytest.raises(ValueError): validate_report(dict(report, confirmation=True), dict(draws=draws, observed=observed))
    with pytest.raises(ValueError): validate_report(report, dict(draws=draws[:10], observed=observed))
    with pytest.raises(AssertionError): validate_report(dict(report, basic_one_sided_lower=[.01]*6), dict(draws=draws, observed=observed))
    negative = -observed
    with pytest.raises(ValueError):
        validate_report(dict(report, paired_effects=negative.tolist(), basic_one_sided_lower=negative.tolist()),
                        dict(draws=np.tile(negative, (2000, 1)), observed=negative))

"""D1 export contract: immutable-input audits, grouped scores and startup."""
import math
import subprocess
import sys

import pytest

from scripts.analysis import export_week8_d1 as D


def test_scores_handle_censoring_boundaries_and_log_floor():
    p = [.1, .1, .1, .1, .1, .1, .4]
    brier, loss = D.forecast_score(p, 6)
    assert brier == pytest.approx(sum((j / 10) ** 2 for j in range(1, 7)) / 6)
    assert loss == pytest.approx(-math.log(.4))
    assert D.forecast_score([1., 0, 0, 0, 0, 0, 0], 6)[1] == pytest.approx(-math.log(1e-12))
    with pytest.raises(ValueError):
        D.forecast_score([.1] * 7, 0)
    with pytest.raises(ValueError):
        D.forecast_score([float('nan')] + [0] * 6, 0)


def test_duplicate_keys_are_not_silently_collapsed():
    with pytest.raises(ValueError, match='Duplicate'):
        D.unique([{'id': 'a'}, {'id': 'a'}], ('id',))


@pytest.fixture(scope='module')
def audited():
    return D.audit()


def test_real_forecasts_are_recomputed_at_mdp_level(audited):
    info, summary, selection, cost, report = audited
    assert report['forecast_rows'] == 1800 * 14
    assert report['mdp_score_rows'] == 120 * 14
    assert report['registered_contrasts'] == 22
    assert report['selection_groups'] == 420
    assert all(row['jobs'] == 4200 for row in cost)
    rows = {r['method']: r for r in info['forecasts']}
    assert rows['b2_generic']['head'] == 'AFT-only'
    assert rows['published_metrics_only']['group'] == 'Privileged'
    assert len(info['learners']) == 7
    assert report['bootstrap_recomputed'] is False
    assert all('\\' not in path for path in report['source_sha256'])


def test_removed_predictions_and_duplicate_labels_fail(monkeypatch):
    original = D.read_csv
    def missing_predictions(path):
        rows = original(path)
        return rows[:-1] if path.name == 'predictions_preoutcome.csv' else rows
    monkeypatch.setattr(D, 'read_csv', missing_predictions)
    with pytest.raises(ValueError, match='cartesian product'):
        D.audit()
    def duplicate_labels(path):
        rows = original(path)
        return rows + [rows[0]] if path.name == 'labels.csv' else rows
    monkeypatch.setattr(D, 'read_csv', duplicate_labels)
    with pytest.raises(ValueError, match='Duplicate'):
        D.audit()


def test_privileged_rows_stay_out_of_matched_pilot_block():
    info = D.read_json(D.MATRIX)
    pilot = {r['method'] for r in info['forecasts'] if r['group'] == 'Matched pilot'}
    assert pilot == {'oeh_final', 'b2_matched', 'b2_plus_eri', 'b2_plus_bwd'}
    spibb = next(r for r in info['learners'] if r['name'] == 'SPIBB')
    assert 'true behavior' in spibb['fit_access']
    assert all(r['requirement'].startswith('no') for r in info['learners'][3:])


def test_wrong_summary_and_bad_probabilities_fail(monkeypatch):
    original = D.read_csv
    def wrong_summary(path):
        rows = original(path)
        if path.name == 'summary.csv':
            rows[0]['integrated_brier'] = '0.5'
        return rows
    monkeypatch.setattr(D, 'read_csv', wrong_summary)
    with pytest.raises(ValueError, match='mismatch'):
        D.audit()
    def bad_probabilities(path):
        rows = original(path)
        if path.name == 'predictions_preoutcome.csv':
            rows[0]['p_6'] = '2'
        return rows
    monkeypatch.setattr(D, 'read_csv', bad_probabilities)
    with pytest.raises(ValueError, match='probability'):
        D.audit()


def test_cli_fresh_process_and_checked_exports():
    for args in (('--help',), ('--check',)):
        run = subprocess.run([sys.executable, '-m', 'scripts.analysis.export_week8_d1', *args],
                             cwd=D.ROOT, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
    for name in ('learner_matrix.tex', 'primary_forecast.tex'):
        text = (D.OUTPUT / name).read_text(encoding='utf-8')
        assert text.count('{') == text.count('}')

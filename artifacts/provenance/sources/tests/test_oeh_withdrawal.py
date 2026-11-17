"""Deletion guards, evidence retention, and startup of the withdrawal CLI."""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
from evaluation.io import sha256_file
from evaluation import withdrawal

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path, monkeypatch):
    names = ['code/oeh_fixture.py', 'tests/test_oeh_fixture.py', 'docs/protocol/fixture.json']
    for name in names:
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}', 'utf-8')
    (tmp_path/names[-1]).write_text(json.dumps(dict(attempt_id='r99', candidate_module='oeh_fixture')), 'utf-8')
    ledger = tmp_path/'docs/reassessment/withdrawal_ledger/r99.json'
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps(dict(attempt_id='r99', status='withdrawn', reason_codes=['SUCCESS_UNCERTAINTY'],
        source_hashes={name:sha256_file(tmp_path/name) for name in names}, consumed_seeds=[1234])), 'utf-8')
    bundle = tmp_path/'tmp/oeh_prospective/r99'
    bundle.mkdir(parents=True); (bundle/'rows.csv').write_text('raw failed result', 'utf-8')
    monkeypatch.setattr(withdrawal.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1))
    return names, ledger, bundle


def test_failed_attempt_removes_only_owned_sources_and_bundle(tmp_path, monkeypatch):
    names, ledger, bundle = fixture(tmp_path, monkeypatch)
    original = tmp_path/'code/original_v4e.py'; original.write_text('frozen', 'utf-8')
    other = tmp_path/'tmp/oeh_prospective/r98'; other.mkdir(); (other/'raw').write_text('other', 'utf-8')
    diagnosis = tmp_path/'docs/reassessment/OEHV4E_RESTART_2026-09-28.md'
    diagnosis.write_text('root diagnosis\n<!-- active-candidate:r99:start -->\nwithdraw this direction\n<!-- active-candidate:r99:end -->\nother content\n', 'utf-8')
    before = ledger.read_bytes()
    result = withdrawal.withdraw_attempt(tmp_path, 'r99')
    assert result['prototype_files_removed'] == 3 and result['ledger_preserved']
    assert not bundle.exists() and all(not (tmp_path/name).exists() for name in names)
    assert ledger.read_bytes() == before and original.exists() and other.exists()
    assert diagnosis.read_text('utf-8') == 'root diagnosis\n\nother content\n'
    assert withdrawal.withdraw_attempt(tmp_path, 'r99')['prototype_files_removed'] == 0


@pytest.mark.parametrize('status, reasons', [('screen_pass', []), ('withdrawn', [])])
def test_passing_or_unsubstantiated_attempt_cannot_be_deleted(tmp_path, monkeypatch, status, reasons):
    names, ledger, bundle = fixture(tmp_path, monkeypatch)
    record = json.loads(ledger.read_text('utf-8')); record.update(status=status, reason_codes=reasons)
    ledger.write_text(json.dumps(record), 'utf-8')
    with pytest.raises(ValueError, match='Only a failed'):
        withdrawal.withdraw_attempt(tmp_path, 'r99')
    assert bundle.exists() and all((tmp_path/name).exists() for name in names)


def test_user_edit_is_preserved_and_prevents_partial_cleanup(tmp_path, monkeypatch):
    names, _, bundle = fixture(tmp_path, monkeypatch)
    (tmp_path/names[0]).write_text('user edit', 'utf-8')
    with pytest.raises(ValueError, match='changed after'):
        withdrawal.withdraw_attempt(tmp_path, 'r99')
    assert bundle.exists() and all((tmp_path/name).exists() for name in names)


def test_tracked_source_and_unreadable_git_index_are_protected(tmp_path, monkeypatch):
    names, _, bundle = fixture(tmp_path, monkeypatch)
    for code in (0, 128):
        monkeypatch.setattr(withdrawal.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=code))
        with pytest.raises(ValueError, match='Tracked source'):
            withdrawal.withdraw_attempt(tmp_path, 'r99')
    assert bundle.exists() and all((tmp_path/name).exists() for name in names)


def test_protocol_escape_and_invalid_attempt_are_rejected(tmp_path, monkeypatch):
    _, ledger, bundle = fixture(tmp_path, monkeypatch)
    record = json.loads(ledger.read_text('utf-8'))
    record['source_hashes'] = {'docs/protocol/../../../outside.json':'invalid'}
    ledger.write_text(json.dumps(record), 'utf-8')
    with pytest.raises(ValueError, match='escapes'):
        withdrawal.withdraw_attempt(tmp_path, 'r99')
    with pytest.raises(ValueError, match='Invalid attempt'):
        withdrawal.withdraw_attempt(tmp_path, '../r99')
    assert bundle.exists()


def test_malformed_scope_marker_prevents_partial_deletion(tmp_path, monkeypatch):
    names, _, bundle = fixture(tmp_path, monkeypatch)
    diagnosis = tmp_path/'docs/reassessment/OEHV4E_RESTART_2026-09-28.md'
    diagnosis.write_text('<!-- active-candidate:r99:start -->\nunclosed', 'utf-8')
    with pytest.raises(ValueError, match='markers'):
        withdrawal.withdraw_attempt(tmp_path, 'r99')
    assert bundle.exists() and all((tmp_path/name).exists() for name in names)


def test_e1_reuse_cleanup_does_not_touch_original_e1(tmp_path, monkeypatch):
    _, ledger, old_bundle = fixture(tmp_path, monkeypatch)
    record=json.loads(ledger.read_text('utf-8')); record['evaluation_reuse']=True
    ledger.write_text(json.dumps(record),'utf-8')
    reuse=tmp_path/'tmp/oeh_e1_reuse/r99'; reuse.mkdir(parents=True)
    (reuse/'rows.csv').write_text('candidate','utf-8')
    original=tmp_path/'artifacts/week6/oehv4e_confirmatory_v1_1'; original.mkdir(parents=True)
    (original/'raw').write_text('frozen','utf-8')
    withdrawal.withdraw_attempt(tmp_path,'r99')
    assert not reuse.exists() and old_bundle.exists() and (original/'raw').read_text('utf-8')=='frozen'


def test_withdrawal_cli_starts_in_fresh_process():
    result = subprocess.run([sys.executable, '-m', 'scripts.analysis.withdraw_oeh_attempt', '--help'],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--watch' in result.stdout

"""Delete failed development bundles and their untracked, sealed prototype.

The minimal attempt/seed/hash ledger is preserved. Passing studies, tracked
sources, edited sources and paths outside the repository are never deleted.
This operation does not modify frozen E1/S1 evidence or Git history.
"""
import json
from pathlib import Path
import re
import shutil
import subprocess

from evaluation.io import sha256_file


def _inside(root, relative):
    candidate = root/relative
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or resolved != candidate.absolute():
        raise ValueError('Withdrawal path escapes repository or traverses a link')
    return resolved


def withdraw_attempt(repo, attempt):
    if not re.fullmatch(r'r[1-9][0-9]*', attempt):
        raise ValueError('Invalid attempt ID')
    root = Path(repo).resolve()
    ledger = _inside(root, f'docs/reassessment/withdrawal_ledger/{attempt}.json')
    record = json.loads(ledger.read_text('utf-8'))
    if record.get('attempt_id') != attempt or record.get('status') != 'withdrawn' or not record.get('reason_codes'):
        raise ValueError('Only a failed, recorded development attempt may be withdrawn')
    hashes = {name.replace('\\', '/'): digest for name, digest in record['source_hashes'].items()}
    protocols = [name for name in hashes if name.startswith('docs/protocol/') and name.endswith('.json')]
    if len(protocols) != 1:
        raise ValueError('Withdrawal requires exactly one sealed protocol')
    protocol = _inside(root, protocols[0])
    owned = []
    if protocol.exists():
        config = json.loads(protocol.read_text('utf-8'))
        module = config.get('candidate_module', '')
        if config.get('attempt_id') != attempt or not re.fullmatch(r'oeh_[a-z0-9_]+', module):
            raise ValueError('Invalid prototype ownership')
        names = [f'code/{module}.py', f'tests/test_{module}.py', protocols[0]]
        # Check every source BEFORE any mutation, including bundle deletion.
        for name in names:
            path = _inside(root, name)
            if name not in hashes:
                raise ValueError('Prototype is absent from the source seal')
            if path.exists():
                if sha256_file(path) != hashes[name]:
                    raise ValueError('Prototype changed after the source seal')
                tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', '--', name],
                                         cwd=root, capture_output=True, check=False)
                if tracked.returncode != 1:
                    raise ValueError('Tracked source or unreadable Git index cannot be removed')
                owned.append(path)
    study_root = 'oeh_e1_reuse' if record.get('evaluation_reuse') is True else 'oeh_prospective'
    bundle = _inside(root, f'tmp/{study_root}/{attempt}')
    diagnosis = _inside(root, 'docs/reassessment/OEHV4E_RESTART_2026-09-28.md')
    revised_diagnosis = None
    if diagnosis.exists():
        contents = diagnosis.read_text('utf-8')
        opening = f'<!-- active-candidate:{attempt}:start -->'
        closing = f'<!-- active-candidate:{attempt}:end -->'
        if opening in contents or closing in contents:
            if contents.count(opening) != 1 or contents.count(closing) != 1:
                raise ValueError('Invalid active-candidate document markers')
            start = contents.index(opening); end = contents.index(closing)
            if end < start:
                raise ValueError('Invalid active-candidate document marker order')
            revised_diagnosis = (contents[:start]+contents[end+len(closing):]).rstrip()+'\n'
    # The resolved absolute bundle has been verified against the exact fixed
    # attempt path above. rmtree does not follow directory symlinks inside it.
    if bundle.exists():
        shutil.rmtree(bundle)
    for path in owned:
        path.unlink()
    if revised_diagnosis is not None:
        diagnosis.write_text(revised_diagnosis, 'utf-8')
    return dict(attempt_id=attempt, bundle_removed=not bundle.exists(),
                prototype_files_removed=len(owned), ledger_preserved=ledger.exists())

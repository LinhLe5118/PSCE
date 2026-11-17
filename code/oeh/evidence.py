"""Portable verification of unchanged scientific evidence and source relocation."""
import ast
import csv
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

REFERENCE_BUNDLE = 'artifacts/week7/oeh_v5_e1_reuse_r34_2026-09-29'
FINAL_BUNDLE = 'artifacts/week7/oeh_v5_e1_reuse_r44_2026-09-29'
CONTRASTS = ('NR_VS_CFQI', 'SR_VS_CFQI', 'NR_VS_V4E', 'SR_VS_V4E', 'NR_VS_R34', 'SR_VS_R34')


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text('utf-8'))


def read_csv_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def confined_path(root, name):
    root = Path(root).resolve()
    path = (root/name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'Evidence path escapes repository: {name}')
    return path


def verify_hashes(root, hashes):
    for name, digest in hashes.items():
        if sha256_file(confined_path(root, name)) != digest:
            raise ValueError(f'Changed sealed evidence: {name}')


def _public_figure_labels(node):
    """Allow three fixed display-only aliases; never transform numeric logic."""
    changes = {
        "'OEH-v5 R34'": "'Archived reference'",
        "identifier.replace('oeh_v5_', 'OEH-v5 ').upper().replace('OEH-V5', 'OEH-v5')": "'OEH'",
        "'vs R34'": "'vs archived reference'",
    }
    replacements = {ast.dump(ast.parse(before, mode='eval').body):ast.parse(after, mode='eval').body
                    for before, after in changes.items()}
    class Labels(ast.NodeTransformer):
        def visit(self, current):
            replacement = replacements.get(ast.dump(current))
            return copy.deepcopy(replacement) if replacement is not None else super().visit(current)
    return Labels().visit(copy.deepcopy(node))


def verify_definitions(root, migration):
    """Each operational scientific function/class has exactly its frozen AST.

    Imports are rewired into the final package. Scientific bodies/signatures
    remain exact. The renderer permits three fixed method-label aliases only.
    """
    for item in migration['unchanged_definitions']:
        original = migration['original_sources'][item['original']]['archived_path']
        old_tree = ast.parse(confined_path(root, original).read_text('utf-8'))
        new_tree = ast.parse(confined_path(root, item['destination']).read_text('utf-8'))
        old = {n.name:n for n in old_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        new = {n.name:n for n in new_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in item['definitions']:
            expected = old[name]
            if item.get('presentation_only'):
                if item['original'] != 'scripts/analysis/export_oeh_upgrade.py' or name != 'figures':
                    raise ValueError('Presentation aliases apply only to the original figure renderer')
                expected = _public_figure_labels(expected)
            if ast.dump(expected, include_attributes=False) != ast.dump(new[name], include_attributes=False):
                raise ValueError(f'Scientific definition changed during migration: {name}')


def validate_report(report, archive):
    if (report.get('screen_pass') is not True or report.get('confirmation') is not False
            or report.get('new_mdps_generated') != 0 or report.get('bootstrap_completed') != 2000
            or report.get('contrast_order') != list(CONTRASTS)
            or np.asarray(report.get('paired_effects')).shape != (6,)
            or np.asarray(report.get('basic_one_sided_lower')).shape != (6,)):
        raise ValueError('Incomplete or incorrectly scoped OEH evidence')
    draws, observed = archive['draws'], archive['observed']
    if draws.shape != (2000, 6) or not np.isfinite(draws).all():
        raise ValueError('Incomplete bootstrap evidence')
    np.testing.assert_allclose(observed, report['paired_effects'], rtol=0, atol=1e-12)
    lower = 2*observed-np.quantile(draws, .95, axis=0)
    np.testing.assert_allclose(lower, report['basic_one_sided_lower'], rtol=0, atol=1e-12)
    if not np.all(observed > 0) or not np.all(lower > 0):
        raise ValueError('OEH did not pass every registered contrast')


def verify_repository(root, *, full_inputs=False):
    """Verify public evidence; optionally require all 17 local original inputs."""
    root = Path(root).resolve()
    migration = read_json(root/'artifacts/provenance/migration.json')
    if migration['scientific_changes'] is not False or migration['original_e1_inputs_published'] is not False:
        raise ValueError('Incorrect migration scope')
    verify_hashes(root, migration['original_bundle_manifests'])
    for item in migration['original_sources'].values():
        verify_hashes(root, {item['archived_path']:item['sha256']})
    verify_hashes(root, migration['selection_history_sha256'])
    verify_definitions(root, migration)
    old = read_json(root/REFERENCE_BUNDLE/'manifest.json')
    final = read_json(root/FINAL_BUNDLE/'manifest.json')
    for directory, manifest in ((REFERENCE_BUNDLE, old), (FINAL_BUNDLE, final)):
        for name, digest in manifest['scientific_source_hashes'].items():
            archived = migration['original_sources'][name]
            if archived['sha256'] != digest:
                raise ValueError(f'Incorrect original source seal: {name}')
        verify_hashes(root/directory, manifest['artifact_sha256'])
        raw = root/directory/'raw'
        verify_hashes(raw, read_json(raw/'artifact_manifest.json')['artifact_sha256'])
        for panel in ('stochastic_primary', 'deterministic_control'):
            seal = read_json(raw/f'{panel}_choice_seal.json')
            verify_hashes(raw, {f'{panel}_choices.json':seal['sha256']})
    expected = {name.replace('\\', '/'):digest for field in ('scientific_source_hashes', 'original_input_hashes')
                for name, digest in old[field].items()}
    expected.update({f'{REFERENCE_BUNDLE}/{name}':digest for name, digest in old['artifact_sha256'].items()})
    expected[f'{REFERENCE_BUNDLE}/manifest.json'] = sha256_file(root/REFERENCE_BUNDLE/'manifest.json')
    if final['original_input_hashes'] != expected or len(old['original_input_hashes']) != 17:
        raise ValueError('Original E1/reference seals changed')
    exporter = migration['original_sources']['scripts/analysis/export_oeh_upgrade.py']
    if final['export_source_sha256'] != exporter['sha256']:
        raise ValueError('Incorrect frozen exporter seal')
    if final['confirmation'] is not False or final['posthoc_consumed_e1'] is not True:
        raise ValueError('Incorrect research scope')
    if final['new_mdps_generated'] != 0 or final['export_refits'] != 0:
        raise ValueError('Export must not create data or fit models')
    report = read_json(root/FINAL_BUNDLE/'raw/report.json')
    with np.load(root/FINAL_BUNDLE/'raw/bootstrap.npz', allow_pickle=False) as archive:
        validate_report(report, archive)
    available = 0
    for name, digest in old['original_input_hashes'].items():
        name = name.replace('\\', '/')
        path = migration['original_sources'].get(name, {}).get('archived_path', name)
        source = confined_path(root, path)
        if source.is_file():
            verify_hashes(root, {path:digest})
            available += 1
        elif full_inputs:
            raise FileNotFoundError(f'Required local E1 input missing: {name}')
    return dict(method='OEH', scientific_revision='OEH', provenance_id='oeh_v5_r44', scientific_changes=False,
                published_evidence_verified=True, original_input_count=17,
                original_inputs_available=available, full_original_inputs_verified=available == 17,
                bootstrap_draws=2000, confirmation=False)

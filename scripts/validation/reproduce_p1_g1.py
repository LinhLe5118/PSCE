"""Reproduce P1/G1 tables from repository-contained evidence, without training."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def csv_equal(actual: Path, expected: Path) -> bool:
    def read(path):
        with path.open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    return read(actual) == read(expected)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    bundle = ROOT/'artifacts/week5/source_bundles/full_pipeline_results_2026-09-14.zip'
    expected = (bundle.with_suffix('.sha256')).read_text().split()[0]
    if sha(bundle) != expected:
        raise ValueError('Source bundle checksum mismatch')
    unpack = out/'source'
    unpack.mkdir()
    with zipfile.ZipFile(bundle) as z:
        for info in z.infolist():
            target = (unpack/info.filename).resolve()
            if not target.is_relative_to(unpack):
                raise ValueError('Unsafe archive member')
        z.extractall(unpack)
    candidates = list(unpack.rglob('full_development_pipeline_2026-09-14_181522'))
    if len(candidates) != 1:
        raise ValueError('Expected one expanded source run')
    expanded = candidates[0]
    commands = []
    def invoke(module, options, profile='protocol_config.json'):
        env = os.environ.copy()
        env['DAORL_PROTOCOL_CONFIG'] = str(ROOT/profile)
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONPATH'] = str(ROOT)
        cmd = [sys.executable, '-m', module, *map(str,options)]
        run = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
        log = out/f'step_{len(commands)+1}.txt'
        log.write_text(run.stdout+'\n'+run.stderr, encoding='utf-8')
        commands.append({'module':module,'returncode':run.returncode,'log':log.name})
        run.check_returncode()
    invoke('scripts.validation.validate_full_pipeline_results', ['--run-dir',expanded,'--output-dir',out/'integrity'])
    invoke('scripts.analysis.analyze_learner_results', ['--input-dir',expanded/'training','--output-dir',out/'d2'])
    comparisons = {}
    for name, train, oracle, alt, profile in [
        ('historical', ROOT/'artifacts/week2/development_offline_rl_2026-08-22',
         ROOT/'artifacts/week2/benchmark_oracle_handoff_2026-08-23',
         ROOT/'artifacts/week2/alternative_definitions_2026-09-01','protocol_config.json'),
        ('expanded',expanded/'training',expanded/'oracle',expanded/'alternatives','config/profiles/expanded_development_v2.json')]:
        invoke('scripts.analysis.run_calibrated_comparison',[
            '--n-emp',train/'empirical_sample_complexity.csv',
            '--behavior-metadata',train/'behavior_policy_metadata.csv',
            '--benchmark-metadata',oracle/'benchmark_metadata.csv',
            '--exact-oracle',oracle/'exact_oracle_results.csv',
            '--alternatives',alt/'alternative_definitions.csv','--output-dir',out/name],profile)
        for file in ['calibrated_results.csv','calibrated_comparison.csv']:
            comparisons[f'{name}/{file}'] = csv_equal(out/name/file,
                ROOT/'artifacts/week5/statistical_correction_2026-09-17'/name/file)
    for file in ['learner_summary.csv','learning_curves.csv','n_emp_by_family.csv']:
        comparisons[f'd2/{file}'] = csv_equal(out/'d2'/file,ROOT/'artifacts/week5/d2_learner_analysis_2026-09-15'/file)
    report = {'status':'pass' if all(comparisons.values()) else 'fail',
              'scope':'automated clean-snapshot reproduction of tables; not human sign-off or retraining',
              'python':sys.version,'source_bundle_sha256':sha(bundle),'comparisons':comparisons,'commands':commands}
    (out/'reproduction.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if report['status'] != 'pass': raise SystemExit(1)

if __name__ == '__main__':
    main()

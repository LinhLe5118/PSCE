"""Paper tables must agree with the sealed comparison and report."""
import csv
import json
from pathlib import Path
import subprocess
import sys

from oeh.evidence import FINAL_BUNDLE
from scripts.analysis.export_paper_tables import render_tables

ROOT = Path(__file__).resolve().parents[1]


def test_paper_tables_match_sealed_values_and_reject_missing_method():
    bundle = ROOT / FINAL_BUNDLE
    with (bundle / 'comparison_x10.csv').open(encoding='utf-8', newline='') as stream:
        comparison = list(csv.DictReader(stream))
    report = json.loads((bundle / 'raw/report.json').read_text('utf-8'))
    table = render_tables(comparison, report)
    assert (ROOT / 'tests/fixtures/paper_history/tables.tex').read_text('utf-8') == table
    assert '0.8068' in table and '45.76' in table and '0.0203' in table
    try:
        render_tables(comparison[:-1], report)
    except ValueError:
        pass
    else:
        raise AssertionError('An incomplete method table was accepted')


def test_paper_table_cli_starts_and_checks_from_fresh_process():
    for args in (('--help',), ('--check',)):
        result = subprocess.run([sys.executable, '-m', 'scripts.analysis.export_paper_tables', *args],
                                cwd=ROOT, capture_output=True)
        assert result.returncode == 0, result.stderr.decode(errors='replace')

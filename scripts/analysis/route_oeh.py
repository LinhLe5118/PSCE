"""Fit the final OEH router on precomputed grouped inputs with frozen definitions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from oeh import fit_routes
from oeh.data import load_panel
from oeh.evidence import sha256_file


def run(input_path, output):
    panel = load_panel(input_path)
    rows, models = [], {}
    for name, mdps in panel['panels'].items():
        choices, forecasts, tuning = fit_routes(panel, mdps)
        models[name] = tuning
        for (mdp, behavior), method in sorted(choices.items()):
            rows.append(dict(mdp_id=mdp, policy_family=behavior, panel=name,
                             outer_fold=panel['outer'][mdp], selected_algorithm=method,
                             forecasts=forecasts[mdp, behavior]))
    document = dict(method='OEH', scientific_revision='OEH', provenance_id='oeh_v5_r44', input_sha256=sha256_file(input_path),
                    new_mdps_generated=0, policy_training_performed=False, confirmation=False,
                    decisions=rows, training=models)
    # A result can never overwrite sealed research evidence or an earlier run.
    output = Path(output).resolve()
    if output.is_relative_to(ROOT/'artifacts'):
        raise ValueError('Routing outputs must stay outside immutable artifacts/')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == '__main__':
    main()

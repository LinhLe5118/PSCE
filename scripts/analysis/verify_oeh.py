"""Verify published OEH evidence without fitting or rerunning evaluation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from oeh.evidence import verify_repository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-inputs', action='store_true', help='Require the 17 local original E1 inputs')
    args = parser.parse_args()
    print(json.dumps(verify_repository(ROOT, full_inputs=args.full_inputs), indent=2))


if __name__ == '__main__':
    main()

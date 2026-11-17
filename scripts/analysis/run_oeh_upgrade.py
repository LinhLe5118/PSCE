"""Compare a frozen candidate against the original E1 and R34."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from evaluation.e1_upgrade import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--workers', default=18, type=int)
    args = parser.parse_args()
    run(ROOT, args.protocol, args.workers)


if __name__ == '__main__':
    main()

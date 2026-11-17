"""Withdraw a failed prospective attempt; optionally wait for its final ledger."""
import argparse
import json
from pathlib import Path
import re
import time

from evaluation.withdrawal import withdraw_attempt

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--watch', action='store_true', help='Wait for the runner final ledger, without reading scores')
    args = parser.parse_args()
    if not re.fullmatch(r'r[1-9][0-9]*', args.attempt):
        parser.error('Invalid attempt ID')
    ledger = ROOT/'docs/reassessment/withdrawal_ledger'/f'{args.attempt}.json'
    while args.watch and not ledger.exists():
        time.sleep(2)
    record = json.loads(ledger.read_text('utf-8'))
    if args.watch and record.get('status') == 'screen_pass':
        print('Screen passed; preserve sealed evidence pending OEH-v4E comparison and confirmation', flush=True)
        return
    print(json.dumps(withdraw_attempt(ROOT, args.attempt), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

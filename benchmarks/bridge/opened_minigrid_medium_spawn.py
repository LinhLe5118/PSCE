"""Spawn a checkpointed exploratory MiniGrid worker with a recorded child PID."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
import sys


def run_child(max_groups: int) -> None:
    from . import opened_minigrid_medium_precompute

    sys.argv = ["opened_minigrid_medium_precompute", "--max-groups", str(max_groups)]
    opened_minigrid_medium_precompute.main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.max_groups <= 5:
        raise ValueError("One to five groups per invocation")
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(target=run_child, args=(args.max_groups,))
    child.start()
    record = {"status": "EXPLORATORY_CHILD_STARTED", "child_pid": child.pid,
              "max_groups": args.max_groups, "test_labels_read": False}
    temporary = args.pid_file.with_suffix(".json.part")
    temporary.write_text(json.dumps(record) + "\n", encoding="utf-8")
    temporary.replace(args.pid_file)
    child.join()
    if child.exitcode:
        raise SystemExit(child.exitcode)


if __name__ == "__main__":
    main()

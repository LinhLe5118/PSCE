#!/usr/bin/env python
"""Run the complete pytest suite and reject silent skips.

The repository contains both ``unittest.TestCase`` tests and native pytest test
functions.  Pytest collects both styles; ``unittest.discover`` silently misses
the latter.  A missing artifact or dependency must make CI red instead of
quietly reducing the tested surface.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


class SkipTracker:
    """Collect skipped test reports so CI can reject incomplete execution."""

    def __init__(self) -> None:
        self.skipped: list[tuple[str, str]] = []

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.skipped:
            reason = str(report.longrepr)
            self.skipped.append((report.nodeid, reason))

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.skipped:
            reason = str(report.longrepr)
            self.skipped.append((report.nodeid, reason))


def result_exit_code(result: object) -> int:
    """Return a process code that treats every skipped test as incomplete."""

    if not result.wasSuccessful():
        return 1
    if result.skipped:
        return 2
    return 0


def main() -> int:
    tracker = SkipTracker()
    pytest_status = int(pytest.main(["tests", "-q"], plugins=[tracker]))
    result = SimpleNamespace(
        wasSuccessful=lambda: pytest_status == int(pytest.ExitCode.OK),
        skipped=tracker.skipped,
    )
    if tracker.skipped:
        print("\nERROR: full-suite contract forbids skipped tests:", file=sys.stderr)
        for test, reason in tracker.skipped:
            print(f"  - {test}: {reason}", file=sys.stderr)
    return result_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())

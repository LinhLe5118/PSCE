"""CI must fail on test failures, runtime skips and collection skips."""
from types import SimpleNamespace

from scripts.testing.run_full_tests import SkipTracker, result_exit_code


def test_full_suite_forbids_failure_or_skips():
    assert result_exit_code(SimpleNamespace(wasSuccessful=lambda:True, skipped=[])) == 0
    assert result_exit_code(SimpleNamespace(wasSuccessful=lambda:False, skipped=[])) == 1
    tracker = SkipTracker()
    for method in (tracker.pytest_runtest_logreport, tracker.pytest_collectreport):
        method(SimpleNamespace(skipped=True, longrepr='missing', nodeid='example'))
    assert len(tracker.skipped) == 2
    assert result_exit_code(SimpleNamespace(wasSuccessful=lambda:True, skipped=tracker.skipped)) == 2

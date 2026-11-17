"""T7 theory drafts: status header and numerical checks (theorems are drafts until checkpoint H5)."""
from __future__ import annotations

import json

import numpy as np

from scripts.analysis import theory_checks_t7 as T


def test_theory_file_is_checked_and_has_no_open_gaps():
    text = (T.ROOT / "paper/appendix_theory.tex").read_text(encoding="utf-8")
    assert text.splitlines()[0].strip() == "% status: checked"  # checked by Linh at H5 (2026-10-15)
    assert "\\TODO{" not in text and "[TODO(proof)" not in text  # every gap is now proved or stated as an assumption
    assert "\\label{prop:tau}" in text and "\\label{cor:ll}" in text


def test_plugin_bayes_regret_obeys_the_row_bound():
    rng = np.random.default_rng(0)
    ref = np.cumsum(rng.dirichlet(np.ones(7), 2000), 1)[:, :6]
    f = np.cumsum(rng.dirichlet(np.ones(7), 2000), 1)[:, :6]
    regret = T.bayes_regret(f, ref)
    assert (regret >= -1e-9).all() and (regret <= 2 * T.LAMBDA * np.abs(f - ref).max(1) + 1e-9).all()
    assert np.allclose(T.bayes_regret(ref, ref), 0)


def test_selection_bound_is_vacuous_at_this_scale():
    t2 = T.t2_numbers()
    assert t2["slack_in_units_of_B"] > 1 and t2["panels_for_slack_0.1B"] == 1981


def test_recorded_checks_hold():
    checks = json.loads((T.OUTPUT / "checks.json").read_text(encoding="utf-8"))
    assert checks["t3_panels_all_hold"] and all(v["row_bound_holds"] for v in checks["synthetic_t3"].values())


def test_tau_rule_and_log_loss_bounds_hold_row_by_row():
    rng = np.random.default_rng(3)
    ref = np.cumsum(rng.dirichlet(np.ones(7), 5000), 1)[:, :6]
    f = np.cumsum(rng.dirichlet(np.ones(7), 5000), 1)[:, :6]
    terms = T.tau_terms(f, ref)
    differ = T.tau_choice(f) != T.tau_choice(ref)
    scale = T.BUDGETS[-1] + T.LAMBDA
    assert (np.abs(terms["tau_regret"]) <= scale * differ + 1e-9).all()
    for m in (0.05, 0.2):
        assert (~differ | (terms["linf"] >= m) | (terms["margin"] < m)).all()
    assert (T.kl(ref, ref) < 1e-12).all() and (T.kl(ref, f) >= -1e-12).all()
    tv = 0.5 * np.abs(np.diff(np.column_stack([np.zeros(len(f)), f, np.ones(len(f))]), axis=1)
                      - np.diff(np.column_stack([np.zeros(len(f)), ref, np.ones(len(f))]), axis=1)).sum(1)
    assert (terms["linf"] <= tv + 1e-12).all() and (tv <= np.sqrt(T.kl(ref, f) / 2) + 1e-9).all()


def test_recorded_new_bounds_hold():
    checks = json.loads((T.OUTPUT / "checks.json").read_text(encoding="utf-8"))
    assert checks["t3_ll_panels_all_hold"] and checks["t3_tau_panels_all_hold"]
    assert all(v["tau_bound_holds"] and v["linf_le_tv"] for v in checks["synthetic_t3"].values())

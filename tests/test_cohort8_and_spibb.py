"""Eighth cohort, its development rules and the SPIBB panel: field sets, laws, loaders, protocols."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import develop_cohort8_candidates_20261014 as D
from scripts.analysis import run_oeh_eighth_cohort_20261014 as E
from scripts.analysis import run_spibb_panel_20261014 as S
from scripts.analysis.compare_difficulty import FEATURES_B2

needs_cohort = pytest.mark.skipif(not (E.COHORT7 / "protocol.json").exists(), reason="sealed cohort-7 artifacts missing")


def test_lce_fit_is_monotone_and_recovers_a_logistic_curve():
    x = np.log(np.asarray(D.CURVE_BUDGETS, float))
    truth = 1 / (1 + np.exp(-(-2.0 + 1.5 * x)))
    a, c = D.lce_fit(np.vstack([truth, truth[::-1], np.zeros(4)]))
    assert c[0] == pytest.approx(1.5, abs=1e-3) and a[0] == pytest.approx(-2.0, abs=1e-3)
    assert c[1] == 0.0  # decreasing data is projected to a flat curve
    frame = pd.DataFrame(np.vstack([truth, truth[::-1], np.zeros(4)]), columns=[f"oehc_success_x{b}" for b in D.CURVE_BUDGETS])
    law = D.lce_law(frame)
    assert np.allclose(law.sum(1), 1) and (law > 0).all()
    assert law[2, 6] > 0.95  # never succeeding in the posterior -> censored


def test_soften_is_identity_at_t1_eps0_and_mixes_the_prior():
    p = np.array([[0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01]])
    prior = np.full((1, 7), 1 / 7)
    assert np.allclose(D.soften(p, prior, 1.0, 0.0), p)
    q = D.soften(p, prior, 2.0, 0.1)
    assert np.isclose(q.sum(), 1) and q[0, 0] < p[0, 0] and q.min() >= 0.1 / 7 - 1e-12


def test_cb_fields_flag_unsupported_pairs():
    frame = pd.DataFrame({"log_state_action_count": [np.log(100.0)] * 2, "oehh_log_concentrability": [1.0, np.inf],
                          "oehh_log_clipped_concentrability": [0.5, np.inf], "oehh_log_intrinsic_bound": [0.0, np.inf],
                          **{f"oehc_success_x{b}": [0.1, 0.0] for b in D.CURVE_BUDGETS}})
    out = D.add_fields(frame)
    assert out.cb_unsupported.tolist() == [0.0, 1.0]
    assert out.cb_log_c_clipped.iloc[1] == D.CB_INFINITE_CAP
    assert out.cb_log_n_hat.iloc[0] == pytest.approx(np.log(100.0) + 0.5)


@needs_cohort
def test_field_sets_extend_b2_and_match_the_factorial_arms():
    sets = E.forecast_sets()
    assert sets["b2_matched"] == tuple(FEATURES_B2)
    assert sets["b2_plus_lce"] == (*FEATURES_B2, *D.LCE_FIELDS)
    assert sets["b2_plus_cb"] == (*FEATURES_B2, *D.CB_FIELDS)
    assert len(sets["oeh_final"]) == 108 and sets["xgboost_cf_108"] == sets["oeh_final"] == sets["cond_full"]


@needs_cohort
def test_frozen_protocols_match_their_hashes_and_development():
    for folder in (E.OUTPUT, S.OUTPUT):
        if not (folder / "protocol.json").exists():
            pytest.skip("protocol not frozen in this checkout")
        assert E.sha256(folder / "protocol.json") == (folder / "protocol.sha256").read_text(encoding="ascii").strip()
    doc = json.loads((E.OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    selection = json.loads((E.DEVELOPMENT / "selection.json").read_text(encoding="utf-8"))
    assert doc["candidates"]["oeh_s"]["T"] == selection["oeh_s"]["chosen"]["T"]
    assert doc["candidates"]["oeh_2t"]["w"] == selection["oeh_2t"]["chosen"]["w"]
    assert doc["panels"]["internal"]["mdps"] == 168 and len(doc["panels"]["internal"]["requested_specs"]) == 168
    assert len(set(doc["panels"]["internal"]["kernel_fingerprints"].values())) == 168
    assert doc["gate"]["steps"]["step1"]["baselines"] == ["b2_matched", "xgboost_cf_18"]


def test_kernel_loaders(tmp_path):
    """Fresh process: archive_imports refuses a process that already imported the root runtime."""
    code = f"""
import numpy as np
from pathlib import Path
from scripts.analysis.run_unseen_validation import archive_imports
archive_imports()
from environments.tabular_environments import TabularMDP
from scripts.analysis import run_oeh_eighth_cohort_20261014 as E
from scripts.analysis import run_spibb_panel_20261014 as S
tmp = Path({str(tmp_path)!r})
T = np.full((3, 2, 3), 1 / 3)
np.savez(tmp / 'col.npz', T=T, R=np.ones((3, 2)), rho=np.array([1.0, 0, 0]))
mdp = E.colosseum_mdp(str(tmp), 'col', TabularMDP)
assert mdp.R.shape == (3, 2, 3) and mdp.terminal_states == () and mdp.gamma == 0.95
P = T.copy(); P[2] = 0; P[2, :, 2] = 1
R = np.zeros((3, 2, 3)); R[:, :, 2] = 1; R[2] = 0
np.savez(tmp / 'sp.npz', P=P, R=R, rho=np.array([1.0, 0, 0]), terminal=np.array([2]))
mdp = S.spibb_mdp(str(tmp), 'sp', TabularMDP)
assert mdp.terminal_states == (2,) and mdp.R[0, 0, 2] == 1
print('OK')
"""
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")


def test_spibb_kernels_match_the_seal_and_regenerate(tmp_path):
    """Committed kernels equal the dataset seal; with the author code present they regenerate exactly."""
    seal_path = S.OUTPUT / "dataset_seal.json"
    if not seal_path.exists():  # a checkout without the SPIBB panel has nothing to verify
        assert not S.KERNELS.exists()
        return
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    kernels = sorted(S.KERNELS.glob("*.npz"))
    assert len(kernels) == len(seal["kernels_sha256"]) == 108
    for path in kernels:
        assert E.sha256(path) == seal["kernels_sha256"][path.stem]
    with np.load(kernels[0]) as z:
        P, final = z["P"], int(z["terminal"][0])
        assert P.shape == (50, 4, 50) and np.allclose(P.sum(-1), 1) and np.all(P[final, :, final] == 1)
        assert z["rho"][0] == 1 and np.isclose(z["pi_b"].sum(1), 1).all()
    if (S.SPIBB / "garnets.py").exists():  # author code is local only (tmp/ is not versioned)
        selection = json.loads((S.OUTPUT / "spibb_selection.json").read_text(encoding="utf-8"))["records"][0]
        task = {"mdp_id": selection["mdp_id"], "ratio": selection["ratio"], "seed": selection["seed"], "out": str(tmp_path)}
        assert S._garnet(task)["final_state"] == selection["final_state"]
        with np.load(tmp_path / f"{task['mdp_id']}.npz") as fresh, np.load(S.KERNELS / f"{task['mdp_id']}.npz") as sealed:
            for name in ("P", "R", "rho", "terminal", "pi_b"):
                assert np.array_equal(fresh[name], sealed[name])

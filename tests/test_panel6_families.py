"""Panel-6 candidate families (task card T8): kernel builders, conversion and fingerprints."""
from __future__ import annotations

import importlib.util
import json

import numpy as np

from scripts.analysis import panel6_families as F

_spec = importlib.util.spec_from_file_location("export_kernels", F.EXPORTER)
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)

MAZE = np.array([list(r) for r in ("*****", "*P  *", "* *G*", "*   *", "*****")])


def _random_kernel(rng, S=7, A=3):
    T = rng.dirichlet(np.ones(S) * 0.5, (S, A))
    return T, rng.uniform(0, 1, (S, A)), rng.dirichlet(np.ones(S))


def test_candidates_are_seeded_distinct_and_within_the_size_cap():
    for family in F.FAMILIES:
        a = F.candidates(family, 40, 5, "x_")
        assert a == F.candidates(family, 40, 5, "x_") and a != F.candidates(family, 40, 6, "x_")
        assert len({tuple(sorted(c.items())) for c in a}) == 40
    for c in F.candidates("mdpp", 200, 7, "x_"):
        S = c["actions"] * c["diameter"]
        assert 8 <= S <= 64 and S * c["actions"] <= F.SA_CAP and 0 <= c["transition_noise"] <= 0.3


def test_successor_kernel_matches_the_playground_noise_model():
    rng = np.random.default_rng(0)
    succ = rng.integers(0, 12, (12, 4))
    T = K.successor_kernel(succ, 0.2)
    assert np.allclose(T.sum(-1), 1)
    assert np.allclose(T[np.arange(12)[:, None], np.arange(4)[None, :], succ], 0.8)
    assert np.isclose(T[0, 0][np.arange(12) != succ[0, 0]], 0.2 / 11).all()
    assert (K.successor_kernel(succ, 0.0).max(-1) == 1).all()


def test_grid_kernel_walls_goal_and_spawn():
    T, R, rho, terminal = K.grid_kernel(MAZE, 0.8)
    S = int((MAZE != "*").sum())
    assert T.shape == (S, 4, S) and np.allclose(T.sum(-1), 1) and np.isclose(rho.sum(), 1)
    cells = [tuple(x) for x in np.argwhere(MAZE != "*")]
    start, goal = cells.index((1, 1)), cells.index((2, 3))
    assert rho[start] == 1 and list(terminal) == [goal]
    # from the start cell, north and west are walls: moving north stays put with probability 0.8 + 0.2 / 3
    assert np.isclose(T[start, 0, start], 0.8 + 0.2 / 3)
    above = cells.index((1, 3))
    assert np.isclose(R[above, 2], 0.8) and (R >= 0).all() and (R <= 1).all()


def test_continuing_conversion_restarts_terminal_states():
    rng = np.random.default_rng(1)
    T, R, rho = _random_kernel(rng)
    T2, R2, _ = F.continuing(T, R, rho, [2, 5])
    assert np.allclose(T2[[2, 5]], rho) and (R2[[2, 5]] == 0).all() and np.allclose(T2[0], T[0])


def test_signature_is_invariant_to_relabelling_and_detects_changes():
    rng = np.random.default_rng(2)
    T, R, rho = _random_kernel(rng)
    perm = rng.permutation(7)
    inv = np.argsort(perm)
    Tp, Rp, rhop = T[perm][:, :, perm], R[perm], rho[perm]
    acts = rng.permutation(3)
    Tp, Rp = Tp[:, acts], Rp[:, acts]
    assert F.signature(T, R, rho) == F.signature(Tp, Rp, rhop)
    assert np.allclose(Tp[inv][:, :, inv][:, np.argsort(acts)], T)
    R2 = R.copy()
    R2[0, 0] += 1e-3
    assert F.signature(T, R, rho) != F.signature(T, R2, rho)
    assert F.fingerprint(T, R, rho) != F.fingerprint(Tp, Rp, rhop)


def test_fingerprint_follows_the_project_convention(tmp_path):
    """Compared in a fresh process: the frozen runtime refuses to load after root modules (archive_imports)."""
    import subprocess
    import sys

    T, R, rho = _random_kernel(np.random.default_rng(3))
    np.savez(tmp_path / "k.npz", T=T, R=R, rho=rho)
    code = ("import sys; from scripts.analysis.run_unseen_validation import archive_imports; archive_imports(); "
            "from scripts.analysis import run_oeh_eighth_cohort_20261014 as E; "
            "from scripts.analysis.run_oeh_120_unseen import fingerprint; "
            "from environments.tabular_environments import TabularMDP; "
            "print(fingerprint(E.colosseum_mdp(sys.argv[1], 'k', TabularMDP)))")
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=F.ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == F.fingerprint(T, R, rho)


def test_recorded_kernels_rebuild_from_the_source_tables():
    """Every development kernel exported by the source libraries (isolated venv) is rebuilt here from its recorded
    source tables and must give the recorded signature; the recorded step checks of MDP Playground must pass."""
    import pandas as pd

    table = pd.read_csv(F.OUTPUT / "feasibility.csv")
    sources = json.loads((F.OUTPUT / "source_tables.json").read_text(encoding="utf-8"))
    assert set(sources) == set(table.mdp_id)
    for row in table.itertuples(index=False):
        T, R, rho = F.rebuild(row.family, sources[row.mdp_id])
        assert F.signature(T, R, rho) == row.signature and F.fingerprint(T, R, rho) == row.fingerprint
    mdpp = table[table.family == "mdpp"]
    assert (mdpp.verify_bonferroni_p > 1e-3).all() and mdpp.verify_support_ok.all() and (mdpp.verify_reward_error == 0).all()
    meta = json.loads((F.OUTPUT / "manifest.json").read_text(encoding="utf-8"))["versions"]
    assert {v["mdp_playground"] for v in meta.values()} == {"1.0.0"} and {v["labmaze"] for v in meta.values()} == {"1.0.6"}


def test_recorded_feasibility_run():
    manifest = json.loads((F.OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "development_draws_only" and manifest["learner_outcomes_observed"] == 0
    for name, digest in manifest["outputs_sha256"].items():
        assert F.sha(F.OUTPUT / name) == digest
    import pandas as pd
    table = pd.read_csv(F.OUTPUT / "feasibility.csv")
    assert set(table.family) == set(F.FAMILIES)
    assert not table.seen_exact.any() and table.matches_earlier_panel.isna().all()
    assert (table[table.within_cap].sa <= F.SA_CAP).all()


def test_protocol_draft_is_a_draft_and_tests_only_new_questions():
    from scripts.analysis import panel6_protocol_draft as D

    doc = json.loads((D.OUTPUT / "protocol_draft.json").read_text(encoding="utf-8"))
    assert doc["status"].startswith("DRAFT") and not (D.OUTPUT / "protocol.json").exists() and not (D.OUTPUT / "protocol.sha256").exists()
    assert set(doc["families"]) == set(F.FAMILIES) and doc["sample_size"]["mdps_per_family"] >= 36
    gated = [c for fam in ("F1", "F2", "F3") for c in doc["gates"][fam]["contrasts"]]
    assert all(c[1] == "oeh" and c[0] != "robust" for c in gated)
    assert {c[2] for c in gated} == {"ibs_ge", "ll_ext", "regret"}
    for rel, digest in doc["inputs_sha256"].items():
        if rel.startswith("artifacts/"):
            assert F.sha(F.ROOT / rel) == digest

"""Practitioner baseline: learner on pilot prefixes + tabular FQE + extrapolation (task card T2).

Implements ``docs/week8/T2_PREFIX_FQE_SPEC_2026-10-15.md`` exactly (written before any run):
five trajectory folds x two pseudo-seeds (original order, trajectory bootstrap) = ten pseudo-seeds;
prefixes of 1, 2, 4, 8 x |S||A| transitions of the four training folds; the registered tabular
learners; FQE on the held-out fold; J* and J_random estimated from the pilot; a monotone logistic
curve in log budget; the seven-class law from P(Binomial(10, p) >= 8).

Phases::

    python -m scripts.analysis.baseline_prefix_fqe features [--workers 10]   # pilot-only features, every dataset
    python -m scripts.analysis.baseline_prefix_fqe evaluate                   # raw and calibrated laws, scores
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/week8/baseline_prefix_fqe_2026-10-15"
SPEC = ROOT / "docs/week8/T2_PREFIX_FQE_SPEC_2026-10-15.md"
GAMMA = 0.95
FOLDS = 5
PSEUDO = 2
PREFIXES = (1, 2, 4, 8)
BUDGETS = (1, 2, 5, 10, 20, 50)
THRESHOLD = 0.95
PRIOR_SD = 3.0
FLOOR = 1e-4
LEARNERS = ("behavior_cloning_tabular", "cfqi_tabular", "iql_tabular")
FEATURES = tuple([f"pfqe_success_b{b}" for b in PREFIXES] + [f"pfqe_nr_b{b}" for b in PREFIXES] + ["pfqe_alpha", "pfqe_beta"])
P = [f"p_{i}" for i in range(7)]
KEYS = ["mdp_id", "policy_family", "algorithm"]


# --------------------------------------------------------------------------- #
# Tabular FQE and helpers
# --------------------------------------------------------------------------- #

def empirical(s, a, r, n, t, S: int, A: int):
    """Maximum-likelihood model of a transition set: p_hat over non-terminal next states, r_hat, visited mask."""
    counts = np.zeros((S, A))
    np.add.at(counts, (s, a), 1.0)
    rewards = np.zeros((S, A))
    np.add.at(rewards, (s, a), r.astype(float))
    nxt = np.zeros((S, A, S))
    live = ~t.astype(bool)
    np.add.at(nxt, (s[live], a[live], n[live]), 1.0)
    visited = counts > 0
    p_hat = np.divide(nxt, counts[..., None], out=np.zeros_like(nxt), where=visited[..., None])
    r_hat = np.divide(rewards, counts, out=np.zeros_like(rewards), where=visited)
    return p_hat, r_hat, visited


def fqe(p_hat, r_hat, visited, policy, gamma: float = GAMMA, tol: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
    """Tabular fitted Q evaluation of a (stochastic) policy; unvisited pairs keep Q = 0. Returns V."""
    q = np.zeros_like(r_hat)
    for _ in range(max_iter):
        v = (policy * q).sum(-1)
        updated = np.where(visited, r_hat + gamma * np.einsum("san,n->sa", p_hat, v), 0.0)
        if np.abs(updated - q).max() < tol:
            q = updated
            break
        q = updated
    return (policy * q).sum(-1)


def greedy_optimal(p_hat, r_hat, visited, gamma: float = GAMMA) -> np.ndarray:
    """Optimal deterministic policy of the empirical model (unvisited pairs worth 0)."""
    sys.path[:0] = [str(ROOT / "code")]
    from oeh.tabular import optimal_values
    _, index = optimal_values(np.where(visited[..., None], p_hat, 0.0), np.where(visited, r_hat, 0.0), gamma)
    return np.eye(r_hat.shape[1])[index]


def fit_curve(budgets: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
    """Penalized logistic MLE of success on log budget with beta >= 0 (bounded = refit at beta 0)."""
    from scipy.optimize import minimize
    x = np.log(budgets.astype(float))

    def objective(theta):
        alpha, beta = theta
        z = alpha + beta * x
        ll = outcomes * -np.logaddexp(0, -z) + (1 - outcomes) * -np.logaddexp(0, z)
        return -(ll.sum() - 0.5 * (alpha ** 2 + beta ** 2) / PRIOR_SD ** 2)

    res = minimize(objective, np.array([0.0, 0.5]), method="L-BFGS-B", bounds=[(None, None), (0.0, None)])
    return float(res.x[0]), float(res.x[1])


def raw_law(alpha: float, beta: float) -> np.ndarray:
    from scipy.stats import binom
    p = 1.0 / (1.0 + np.exp(-(alpha + beta * np.log(np.asarray(BUDGETS, float)))))
    cdf = np.maximum.accumulate(binom.sf(7, 10, p))  # P(>= 8 of 10 seeds succeed), non-decreasing
    law = np.diff(np.concatenate([[0.0], cdf, [1.0]]))
    law = np.maximum(law, FLOOR)
    return law / law.sum()


def _seed(mdp: str, behavior: str) -> int:
    return int.from_bytes(hashlib.sha256(f"prefix-fqe-v1|{mdp}|{behavior}".encode()).digest()[:8], "little")


# --------------------------------------------------------------------------- #
# One pilot
# --------------------------------------------------------------------------- #

def pilot_features(arrays: dict, S: int, A: int, mdp: str, behavior: str) -> list[dict]:
    sys.path[:0] = [str(ROOT / "code")]
    from oeh.tabular import CORE_LEARNERS, count_model

    s, a, r, n, t = (np.asarray(arrays[k]) for k in ("state", "action", "reward", "next_state", "terminal"))
    traj, step = np.asarray(arrays["trajectory_id"]), np.asarray(arrays["timestep"])
    rng = np.random.default_rng(_seed(mdp, behavior))
    ids = np.unique(traj)
    fold_of = dict(zip(rng.permutation(ids), np.arange(len(ids)) % FOLDS))
    fold = np.array([fold_of[i] for i in traj])
    pfull, rfull, vfull = empirical(s, a, r, n, t, S, A)
    star = greedy_optimal(pfull, rfull, vfull)
    uniform = np.full((S, A), 1.0 / A)
    nr = {name: np.zeros((len(PREFIXES), FOLDS * PSEUDO)) for name in LEARNERS}
    for k in range(FOLDS):
        ev = fold == k
        pe, re_, ve = empirical(s[ev], a[ev], r[ev], n[ev], t[ev], S, A)
        starts = s[ev & (step == 0)]
        rho = np.bincount(starts, minlength=S).astype(float)
        rho = rho / rho.sum() if rho.sum() else np.full(S, 1.0 / S)
        train_ids = [i for i in ids if fold_of[i] != k]
        pools = [np.flatnonzero(~ev)]
        boot = rng.choice(train_ids, size=len(train_ids), replace=True)
        pools.append(np.concatenate([np.flatnonzero(traj == i) for i in boot]) if len(boot) else np.flatnonzero(~ev))
        j_rand = rho @ fqe(pe, re_, ve, uniform)
        values = {}
        for ri, pool in enumerate(pools):
            for bi, b in enumerate(PREFIXES):
                idx = pool[:b * S * A]
                model = count_model(s[idx], a[idx], r[idx], n[idx], t[idx], num_states=S, num_actions=A)
                for name in LEARNERS:
                    policy = CORE_LEARNERS[name](model, GAMMA)[0]
                    values[(name, bi, ri)] = rho @ fqe(pe, re_, ve, policy)
        j_star = max(rho @ fqe(pe, re_, ve, star), max(values.values()))
        span = j_star - j_rand
        for (name, bi, ri), j in values.items():
            nr[name][bi, k * PSEUDO + ri] = (j - j_rand) / span if span > 1e-9 else 0.0
    out = []
    for name in LEARNERS:
        success = (nr[name] >= THRESHOLD).astype(float)
        budgets = np.repeat(np.asarray(PREFIXES, float), FOLDS * PSEUDO)
        alpha, beta = fit_curve(budgets, success.ravel())
        law = raw_law(alpha, beta)
        row = {"mdp_id": mdp, "policy_family": behavior, "algorithm": name, "pfqe_alpha": alpha, "pfqe_beta": beta}
        for bi, b in enumerate(PREFIXES):
            row[f"pfqe_success_b{b}"] = float(success[bi].mean())
            row[f"pfqe_nr_b{b}"] = float(nr[name][bi].mean())
        row.update({f"raw_p_{i}": float(law[i]) for i in range(7)})
        out.append(row)
    return out


def _task(task: dict) -> list[dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    row = task["row"]
    with np.load(Path(task["dataset"]) / row["file"], allow_pickle=False) as z:
        arrays = {k: z[k][:int(row["size"])] for k in z.files}
    return pilot_features(arrays, int(row["num_states"]), int(row["num_actions"]), row["instance_id"], row["policy_family"])


# --------------------------------------------------------------------------- #
# Datasets, panels
# --------------------------------------------------------------------------- #

def datasets() -> dict[str, Path]:
    from scripts.analysis import run_oeh_confirmatory_20261004 as C
    from scripts.analysis import run_oeh_final_120 as R
    C.configure()
    out = {f"meta_{k}": Path(v["dataset"]) for k, v in R.OPENED.items()}
    out.update({
        "cohort7": ROOT / "datasets/oeh_confirmatory_unseen_120_2026-10-04",
        "cohort8": ROOT / "datasets/oeh_eighth_cohort_2026-10-14",
        "colosseum2027": Path("D:/Tools/colosseum_work/trial/data/confirm_bundle"),
        "colosseum2028": Path("D:/Tools/colosseum_work/trial/data/c8tier_bundle"),
        "spibb": ROOT / "datasets/oeh_spibb_panel_2026-10-14",
        **{p: ROOT / f"datasets/oeh_v2_sealed_test_2026-10-14/{p}" for p in ("internal", "colosseum2029", "spibb2", "forest")},
        **{p: ROOT / f"datasets/oeh_fifth_panel_2026-10-15/{p}" for p in ("frozenlake", "gambler")},
    })
    return out


def run_features(workers: int) -> None:
    from scripts.analysis import run_oeh_final_120 as R
    (OUTPUT / "features").mkdir(parents=True, exist_ok=True)
    for name, dataset in datasets().items():
        target = OUTPUT / "features" / f"{name}.csv.gz"
        if target.exists():
            continue
        rows = R.pilot_rows(dataset)
        tasks = [{"row": row, "dataset": str(dataset)} for row in rows]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            frame = pd.DataFrame([r for part in pool.map(_task, tasks, chunksize=8) for r in part])
        frame.to_csv(target, index=False)
        print(f"{name}: {len(rows)} pilots", flush=True)


def run_evaluate() -> None:
    from scripts.analysis import extrapolation_endpoint as X
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis.compare_difficulty import FEATURES_B2
    from scripts.analysis.develop_cohort8_candidates_20261014 import meta_panel

    feats = {p.name.removesuffix(".csv.gz"): pd.read_csv(p) for p in sorted((OUTPUT / "features").glob("*.csv.gz"))}
    # the meta panel namespaces the fourth cohort (run_oeh_confirmatory_20261004.configure)
    feats["meta_fourth120"]["mdp_id"] = "opened_fourth120/" + feats["meta_fourth120"]["mdp_id"].astype(str)
    meta = pd.concat([f for k, f in feats.items() if k.startswith("meta_")], ignore_index=True)
    train = meta_panel(sort=False)
    t = train.merge(meta[KEYS + list(FEATURES)], on=KEYS, how="left", validate="many_to_one")
    coverage = float(t[FEATURES[0]].notna().mean())
    if coverage < 0.999:
        raise RuntimeError(f"meta-training coverage {coverage:.4f}")
    rows, log = [], []
    for panel in X.PANELS:
        f = feats[panel]
        scored = X.load(panel)
        query = scored[scored.canonical == "b2"][KEYS + ["source_panel", "event"]].copy()
        q = query.merge(f, on=KEYS, validate="one_to_one")
        if len(q) != len(query):
            raise RuntimeError(f"{panel}: {len(q)} of {len(query)} rows have features")
        qtab = _query_table(panel)
        qq = q.merge(qtab[KEYS + list(FEATURES_B2)], on=KEYS, validate="one_to_one")
        qq["panel"] = "primary"
        calibrated, _ = F.matched_forecast(t, qq, tuple(FEATURES_B2) + FEATURES)
        laws = {"prefix_fqe_raw": qq[[f"raw_p_{i}" for i in range(7)]].to_numpy(float), "b2_plus_prefix_fqe": calibrated}
        y = qq.event.to_numpy(int)
        for method, law in laws.items():
            if not np.isfinite(law).all() or not np.allclose(law.sum(1), 1, atol=1e-9):
                raise RuntimeError(f"invalid law {panel} {method}")
            sc = X.row_scores(law, y)
            frame = qq[KEYS + ["source_panel"]].copy()
            for k, v in sc.items():
                frame[k] = v
            frame["canonical"] = method
            rows.append(frame)
            per = frame.groupby("mdp_id")[["ibs", "ll", "ibs_ge", "ll_ext"]].mean().mean()
            log.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "candidate": method,
                        "settings": "spec T2_PREFIX_FQE_SPEC_2026-10-15", "panel": panel, **per.to_dict()})
    ours = pd.concat(rows, ignore_index=True)
    ours.to_csv(OUTPUT / "scores_rows.csv.gz", index=False)
    pd.DataFrame(log).to_csv(OUTPUT / "development_log.csv", index=False)
    summarize(ours, coverage)


def _query_table(panel: str) -> pd.DataFrame:
    A = ROOT / "artifacts"
    paths = {"cohort7": None, "cohort8": A / "oeh_eighth_cohort_2026-10-14/internal/query_preoutcome.csv.gz",
             "colosseum2028": A / "oeh_eighth_cohort_2026-10-14/colosseum/query_preoutcome.csv.gz",
             "spibb": A / "oeh_spibb_panel_2026-10-14/spibb/query_preoutcome.csv.gz",
             **{p: A / f"oeh_v2_sealed_test_2026-10-14/{p}/query_preoutcome.csv.gz" for p in ("internal", "colosseum2029", "spibb2", "forest")},
             **{p: A / f"oeh_fifth_panel_2026-10-15/{p}/query_preoutcome.csv.gz" for p in ("frozenlake", "gambler")}}
    if paths.get(panel) is not None:
        return pd.read_csv(paths[panel])
    from scripts.analysis.develop_oeh_v2_heads_20261014 import union_query
    q = union_query()
    return q[q.source_panel == panel]


def summarize(ours: pd.DataFrame, coverage: float) -> None:
    from scripts.analysis import extrapolation_endpoint as X
    others = pd.concat([X.load(p) for p in X.PANELS], ignore_index=True)
    lce = []
    for panel in X.PANELS:
        pred = pd.concat([pd.read_csv(f) for f in X.PANELS[panel]["pred"]], ignore_index=True)
        pred = pred[pred.method.isin(["lce_raw", "b2_plus_lce"])]
        if pred.empty:
            continue
        lab = pd.read_csv(X.PANELS[panel]["labels"])
        if "panel" in lab:
            lab = lab[lab.panel == "primary"]
        if "panel" in pred:
            pred = pred[pred.panel == "primary"]
        m = pred.merge(lab[KEYS + ["event"]], on=KEYS, validate="many_to_one")
        sc = X.row_scores(m[P].to_numpy(float), m.event.to_numpy(int))
        for k, v in sc.items():
            m[k] = v
        m["canonical"], m["source_panel"] = m.method, panel
        lce.append(m)
    allrows = pd.concat([ours, others, *lce], ignore_index=True)
    pm = allrows.groupby(["source_panel", "canonical", "mdp_id"])[["ibs", "ll", "ibs_ge", "ibs_gt", "ll_ext"]].mean().reset_index()
    table = pm.groupby(["source_panel", "canonical"])[["ibs", "ll", "ibs_ge", "ll_ext"]].mean().reset_index()
    table.to_csv(OUTPUT / "scores_by_panel.csv", index=False)
    gains = []
    for scope, panels in [(p, (p,)) for p in X.PANELS] + list(X.GROUPS.items()):
        for cand in ("oeh", "oeh_f"):
            for base in ("prefix_fqe_raw", "b2_plus_prefix_fqe"):
                g = X.gains(pm, panels, base, cand)
                if g:
                    gains.append({"scope": scope, "candidate": cand, "baseline": base, **g})
    pd.DataFrame(gains).to_csv(OUTPUT / "gains.csv", index=False)
    files = [OUTPUT / "scores_rows.csv.gz", OUTPUT / "development_log.csv", OUTPUT / "scores_by_panel.csv", OUTPUT / "gains.csv"]
    manifest = {
        "status": "posthoc_descriptive", "task": "T2 prefix + FQE baseline", "created_utc": datetime.now(timezone.utc).isoformat(),
        "spec": {"path": SPEC.relative_to(ROOT).as_posix(), "sha256": X.sha(SPEC)},
        "meta_training_feature_coverage": coverage,
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): X.sha(p) for p in sorted((OUTPUT / "features").glob("*.csv.gz"))},
        "outputs_sha256": {p.name: X.sha(p) for p in files},
        "source_sha256": {Path(__file__).relative_to(ROOT).as_posix(): X.sha(Path(__file__))},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    show = table[table.canonical.isin(["b2", "prefix_fqe_raw", "b2_plus_prefix_fqe", "lce_raw", "b2_plus_lce", "oeh_f", "oeh"])]
    print(show.pivot(index="source_panel", columns="canonical", values="ll").round(3).to_string())
    print(show.pivot(index="source_panel", columns="canonical", values="ibs").round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("features", "evaluate"))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_features(args.workers) if args.phase == "features" else run_evaluate()


if __name__ == "__main__":
    main()

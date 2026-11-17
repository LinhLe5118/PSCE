"""Numerical sanity checks of the T7 theory drafts (task card T7; paper/appendix_theory.tex, status draft).

1. Theorem T3 on synthetic laws: the row bound and both mean bounds hold.
2. Remark T2: the LOPO selection slack for K = 500 candidates and P = 3 panels, and the P needed for 0.1 B.
3. Proposition T3-tau (margin bound of the registered tau-rule) and Corollary T3-LL (log-loss form) on synthetic laws
   and on the panels.
4. Theorem T3 on the four final panels: the reference law F_o is the plug-in law of the requirement class from
   200 bootstrap resamples of the ten training seeds (given the MDP and behavior, the pilot is independent of the
   training seeds, so F_o = P(E <= j | MDP, behavior, pilot)); the plug-in Bayes regret of OEH, OEH-F, the robust
   head and B2 under F_o is compared with 2 lambda E max|F - F_o| and 2 lambda sqrt(6 Delta).

Usage: ``python -m scripts.analysis.theory_checks_t7``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/week8/theory_checks_2026-10-15"
BUDGETS = np.array([1, 2, 5, 10, 20, 50], float)
LAMBDA = 100.0
TAU = 0.8
MARGINS = np.linspace(0.01, 0.5, 50)
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]
PANELS = ("internal", "colosseum2029", "spibb2", "forest")
METHODS = {"v2": "OEH", "oeh_final": "OEH-F", "legacy": "robust head", "b2_matched": "B2"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def costs(cdf: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    """Expected cost of each action under a cumulative law: columns 0..5 collect b_a, column 6 skips."""
    collect = BUDGETS[None, :] + lam * (1 - cdf)
    skip = lam * cdf[:, 5:6]
    return np.column_stack([collect, skip])


def bayes_regret(cdf: np.ndarray, ref: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    chosen = costs(cdf, lam).argmin(1)
    c_ref = costs(ref, lam)
    return c_ref[np.arange(len(cdf)), chosen] - c_ref.min(1)


def tau_choice(cdf: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Registered tau-rule: the smallest a with F_a >= tau, else skip (index 6)."""
    reach = cdf >= tau
    return np.where(reach.any(1), reach.argmax(1), 6)


def tau_terms(cdf: np.ndarray, ref: np.ndarray, tau: float = TAU, lam: float = LAMBDA) -> dict[str, np.ndarray]:
    """Row terms of Proposition T3-tau: regret of the tau-rule of F against the tau-rule of F_o, both under F_o."""
    c_ref = costs(ref, lam)
    rows = np.arange(len(cdf))
    return {"tau_regret": c_ref[rows, tau_choice(cdf, tau)] - c_ref[rows, tau_choice(ref, tau)],
            "linf": np.abs(cdf - ref).max(1), "margin": np.abs(ref - tau).min(1), "sq": ((cdf - ref) ** 2).sum(1)}


def tau_bounds(terms: dict[str, np.ndarray], weights: np.ndarray, lam: float = LAMBDA) -> dict[str, float]:
    """(b_5 + lam) [Pr(||F - F_o|| >= m) + Pr(margin < m)] and its Chebyshev form (b_5 + lam) [E||F - F_o||_2^2 / m^2 +
    Pr(margin < m)], both minimized over the margin grid, with row weights ``weights`` (summing to one)."""
    scale = BUDGETS[-1] + lam
    w = weights / weights.sum()
    freq = [scale * (w @ (terms["linf"] >= m) + w @ (terms["margin"] < m)) for m in MARGINS]
    cheb = [scale * (min(1.0, w @ terms["sq"] / m ** 2) + w @ (terms["margin"] < m)) for m in MARGINS]
    return {"tau_regret": float(w @ terms["tau_regret"]), "tau_bound_events": float(min(freq)), "tau_bound_chebyshev": float(min(cheb))}


def kl(ref_cdf: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """KL(p_o || p) of the seven-class laws given by two cumulative forecasts."""
    def law(c):
        return np.clip(np.diff(np.column_stack([np.zeros(len(c)), c, np.ones(len(c))]), axis=1), 0, 1)
    p_o, p = law(ref_cdf), law(cdf)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p_o > 0, p_o * (np.log(p_o) - np.log(np.maximum(p, 1e-300))), 0.0)
    return terms.sum(1)


def synthetic(rng, rows: int = 20000) -> dict:
    ref = np.cumsum(rng.dirichlet(np.full(7, 0.6), rows), 1)[:, :6]
    noise = np.cumsum(rng.dirichlet(np.full(7, 0.6), rows), 1)[:, :6]
    out = {}
    for w in (0.05, 0.2, 0.5):
        f = (1 - w) * ref + w * noise
        regret = bayes_regret(f, ref)
        linf = np.abs(f - ref).max(1)
        delta = ((f - ref) ** 2).sum(1).mean() / 6
        terms = tau_terms(f, ref)
        tb = tau_bounds(terms, np.ones(rows))
        tv = 0.5 * np.abs(np.diff(np.column_stack([np.zeros(rows), f, np.ones(rows)]), axis=1)
                          - np.diff(np.column_stack([np.zeros(rows), ref, np.ones(rows)]), axis=1)).sum(1)
        mean_kl = float(kl(ref, f).mean())
        out[f"mix_{w}"] = {"row_bound_holds": bool((regret <= 2 * LAMBDA * linf + 1e-9).all()), "mean_regret": float(regret.mean()),
                           "bound_linf": float(2 * LAMBDA * linf.mean()), "bound_ibs": float(2 * LAMBDA * math.sqrt(6 * delta)),
                           "linf_le_tv": bool((linf <= tv + 1e-12).all()), "bound_ll": float(LAMBDA * math.sqrt(2 * mean_kl)),
                           **tb, "tau_bound_holds": bool(tb["tau_regret"] <= tb["tau_bound_events"] + 1e-9 <= tb["tau_bound_chebyshev"] + 2e-9)}
    return out


def t2_numbers(K: int = 500, P_: int = 3, delta: float = 0.05) -> dict:
    slack = 2 * math.sqrt(math.log(2 * K / delta) / (2 * P_))
    needed = math.ceil(math.log(2 * K / delta) / (2 * 0.05 ** 2))
    return {"K": K, "P": P_, "delta": delta, "slack_in_units_of_B": slack, "panels_for_slack_0.1B": needed}


def suffix_events(counts: np.ndarray, k: int = 8) -> np.ndarray:
    ok = counts >= k
    suffix = np.flip(np.logical_and.accumulate(np.flip(ok, -1), -1), -1)
    return np.where(suffix.any(-1), suffix.argmax(-1), counts.shape[-1])


def plugin_cdf(outcomes: pd.DataFrame, rng, boots: int = 200) -> pd.DataFrame:
    o = outcomes[outcomes.budget.isin(BUDGETS.astype(int))]
    cube = o.pivot_table(index=KEYS, columns=["budget", "training_seed"], values="normalized_return", aggfunc="first")
    cube = cube.reindex(columns=pd.MultiIndex.from_product([BUDGETS.astype(int), range(10)]))
    arr = (cube.to_numpy(float).reshape(len(cube), 6, 10) >= 0.95)
    law = np.zeros((len(cube), 7))
    for _ in range(boots):
        pick = rng.integers(0, 10, 10)
        events = suffix_events(arr[:, :, pick].sum(2))
        law[np.arange(len(cube)), events] += 1
    law /= boots
    out = pd.DataFrame(np.cumsum(law, 1)[:, :6], columns=[f"ref_F{j}" for j in range(6)], index=cube.index).reset_index()
    return out


def panels(rng) -> pd.DataFrame:
    rows, inputs = [], []
    for panel in PANELS:
        base = ROOT / "artifacts/oeh_v2_sealed_test_2026-10-14" / panel
        inputs += [base / "outcomes.csv.gz", base / "predictions_preoutcome.csv"]
        ref = plugin_cdf(pd.read_csv(base / "outcomes.csv.gz"), rng)
        pred = pd.read_csv(base / "predictions_preoutcome.csv")
        pred = pred[pred.method.isin(METHODS)].merge(ref, on=KEYS, validate="many_to_one")
        F = np.cumsum(pred[P].to_numpy(float), 1)[:, :6]
        R = pred[[f"ref_F{j}" for j in range(6)]].to_numpy(float)
        pred["regret"] = bayes_regret(F, R)
        pred["linf"] = np.abs(F - R).max(1)
        pred["sq"] = ((F - R) ** 2).sum(1)
        pred["kl"] = kl(R, F)
        for method, g in pred.groupby("method"):
            per = g.groupby("mdp_id")[["regret", "linf", "sq", "kl"]].mean().mean()
            Fg = np.cumsum(g[P].to_numpy(float), 1)[:, :6]
            Rg = g[[f"ref_F{j}" for j in range(6)]].to_numpy(float)
            weights = 1.0 / g.groupby("mdp_id").mdp_id.transform("size").to_numpy(float)  # equal weight per MDP
            tb = tau_bounds(tau_terms(Fg, Rg), weights)
            rows.append({"panel": panel, "method": METHODS[method], "mean_plugin_bayes_regret": per.regret,
                         "bound_linf": 2 * LAMBDA * per.linf, "delta_ibs": per.sq / 6,
                         "bound_ibs": 2 * LAMBDA * math.sqrt(per.sq), "holds": bool(per.regret <= 2 * LAMBDA * per.linf + 1e-9),
                         "mean_kl": per.kl, "bound_ll": LAMBDA * math.sqrt(2 * per.kl),
                         "holds_ll": bool(per.regret <= LAMBDA * math.sqrt(2 * per.kl) + 1e-9), **tb,
                         "holds_tau": bool(tb["tau_regret"] <= tb["tau_bound_events"] + 1e-9)})
    return pd.DataFrame(rows), inputs


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261015)
    syn = synthetic(rng)
    t2 = t2_numbers()
    table, inputs = panels(rng)
    table.to_csv(OUTPUT / "t3_panels.csv", index=False)
    result = {"synthetic_t3": syn, "t2_vacuity": t2,
              "t3_panels_all_hold": bool(table.holds.all()), "t3_ll_panels_all_hold": bool(table.holds_ll.all()),
              "t3_tau_panels_all_hold": bool(table.holds_tau.all()), "tau": TAU}
    (OUTPUT / "checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest = {"status": "posthoc_descriptive", "task": "T7 numerical checks (theory status: draft)",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "inputs_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
                "outputs_sha256": {p.name: sha(p) for p in (OUTPUT / "t3_panels.csv", OUTPUT / "checks.json")},
                "source_sha256": {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__)),
                                  "paper/appendix_theory.tex": sha(ROOT / "paper/appendix_theory.tex")}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(json.dumps(result, indent=1))
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Sample size for panel 6 from the opened panels (task card T8; post hoc on opened panels, no new data).

Per-MDP effects of the proposed panel-6 contrasts are computed on the opened panels where the final OEH forecasts
exist (the sealed final test and panel 5):

* extrapolation endpoint (task card T1): ``ibs_ge`` and ``ll_ext``, OEH against B2 + prefix/FQE (task card T2) and
  against B2;
* decision regret at tau = 0.8, lambda = 100 (``colosseum_posthoc_20261014.decision_costs``), OEH against the same two.

An effect is comparator score minus OEH score (positive favours OEH). Power of a gate family (one-sided max-t with the
original standard error, GO when every simultaneous lower bound is > 0) is simulated by drawing MDP effect vectors with
replacement from the opened external panels, with the mean effect scaled by ``shrink``.

The calibrated B2 + prefix/FQE laws are not stored by T2; they are refitted here with the same head and checked
against the recorded T2 row scores.

Usage: ``python -m scripts.analysis.panel6_power``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import baseline_prefix_fqe as B
from scripts.analysis import extrapolation_endpoint as X

ROOT = B.ROOT
OUTPUT = ROOT / "artifacts/week8/panel6_power_2026-10-15"
KEYS, P = B.KEYS, B.P
PANELS = ("internal", "colosseum2029", "spibb2", "forest", "frozenlake", "gambler")
EXTERNAL = ("colosseum2029", "spibb2", "forest", "frozenlake", "gambler")
TAU, LAMBDA = 0.8, 100.0
FAMILIES = {
    "F1_vs_prefix_fqe": [("b2_plus_prefix_fqe", "ibs_ge"), ("b2_plus_prefix_fqe", "ll_ext")],
    "F2_vs_b2": [("b2", "ibs_ge"), ("b2", "ll_ext")],
    "F3_regret": [("b2_plus_prefix_fqe", "regret"), ("b2", "regret")],
}
TOTALS = (48, 72, 96, 120, 144, 192)
SHRINK = (1.0, 0.75, 0.5)
SIMS, DRAWS, SEED = 1000, 1000, 20261016


def calibrated_laws() -> pd.DataFrame:
    """Refit ``b2_plus_prefix_fqe`` on the six panels exactly as ``baseline_prefix_fqe.run_evaluate``."""
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis.compare_difficulty import FEATURES_B2
    from scripts.analysis.develop_cohort8_candidates_20261014 import meta_panel

    feats = {p.name.removesuffix(".csv.gz"): pd.read_csv(p) for p in sorted((B.OUTPUT / "features").glob("*.csv.gz"))}
    feats["meta_fourth120"]["mdp_id"] = "opened_fourth120/" + feats["meta_fourth120"]["mdp_id"].astype(str)
    meta = pd.concat([f for k, f in feats.items() if k.startswith("meta_")], ignore_index=True)
    t = meta_panel(sort=False).merge(meta[KEYS + list(B.FEATURES)], on=KEYS, how="left", validate="many_to_one")
    frames = []
    for panel in PANELS:
        scored = X.load(panel)
        query = scored[scored.canonical == "b2"][KEYS + ["source_panel", "event"]]
        q = query.merge(feats[panel], on=KEYS, validate="one_to_one")
        qq = q.merge(B._query_table(panel)[KEYS + list(FEATURES_B2)], on=KEYS, validate="one_to_one")
        qq["panel"] = "primary"
        law, _ = F.matched_forecast(t, qq, tuple(FEATURES_B2) + B.FEATURES)
        frame = qq[KEYS + ["source_panel", "event"]].copy()
        frame[P] = law
        frame["canonical"] = "b2_plus_prefix_fqe"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def per_mdp_effects() -> tuple[pd.DataFrame, float]:
    from scripts.analysis.colosseum_posthoc_20261014 import decision_costs

    own = calibrated_laws()
    others = pd.concat([X.load(p) for p in PANELS], ignore_index=True)
    others = others[others.canonical.isin(["oeh", "b2"])][KEYS + ["source_panel", "event", "canonical"] + P]
    rows = pd.concat([own, others], ignore_index=True)
    y = rows.event.to_numpy(int)
    law = rows[P].to_numpy(float)
    for k, v in X.row_scores(law, y).items():
        rows[k] = v
    rows["regret"] = decision_costs(np.cumsum(law, 1)[:, :6], y, TAU, LAMBDA)["regret"]
    recorded = pd.read_csv(B.OUTPUT / "scores_rows.csv.gz")
    recorded = recorded[(recorded.canonical == "b2_plus_prefix_fqe") & recorded.source_panel.isin(PANELS)]
    check = rows[rows.canonical == "b2_plus_prefix_fqe"].merge(recorded, on=KEYS + ["source_panel"], suffixes=("", "_t2"))
    reproduction = float(np.abs(check[["ibs", "ll"]].to_numpy() - check[["ibs_t2", "ll_t2"]].to_numpy()).max())
    per = rows.groupby(["source_panel", "canonical", "mdp_id"])[["ibs_ge", "ll_ext", "regret"]].mean().reset_index()
    wide = per.pivot_table(index=["source_panel", "mdp_id"], columns="canonical", values=["ibs_ge", "ll_ext", "regret"])
    eff = pd.DataFrame(index=wide.index)
    for comparator in ("b2_plus_prefix_fqe", "b2"):
        for metric in ("ibs_ge", "ll_ext", "regret"):
            eff[f"{comparator}|{metric}"] = wide[(metric, comparator)] - wide[(metric, "oeh")]
    return eff.reset_index(), reproduction


def gate_power(effects: np.ndarray, total: int, shrink: float, rng: np.random.Generator) -> float:
    """Fraction of simulated panels of ``total`` MDPs where every max-t lower bound is > 0."""
    mean = effects.mean(0)
    centred = effects - mean + shrink * mean
    go = 0
    for _ in range(SIMS):
        sample = centred[rng.integers(0, len(centred), total)]
        m, se = sample.mean(0), sample.std(0, ddof=1) / np.sqrt(total)
        boot = sample[rng.integers(0, total, (DRAWS, total))].mean(1)
        critical = np.quantile(np.max((boot - m) / se, axis=1), 0.95)
        go += bool(np.all(m - critical * se > 0))
    return go / SIMS


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    eff, reproduction = per_mdp_effects()
    if reproduction > 1e-9:
        raise RuntimeError(f"refitted B2 + prefix/FQE does not reproduce T2 ({reproduction:.2e})")
    eff.to_csv(OUTPUT / "effects_by_mdp.csv", index=False)
    cols = [c for c in eff.columns if "|" in c]
    sd = []
    for scope, panels in [(p, (p,)) for p in PANELS] + [("external_pooled", EXTERNAL)]:
        sub = eff[eff.source_panel.isin(panels)]
        for c in cols:
            sd.append({"scope": scope, "contrast": c, "mdps": len(sub), "mean": sub[c].mean(), "sd": sub[c].std(ddof=1),
                       "standardized": sub[c].mean() / sub[c].std(ddof=1)})
    pd.DataFrame(sd).to_csv(OUTPUT / "effect_sd.csv", index=False)
    rng = np.random.default_rng(SEED)
    power = []
    for scope, panels in [("external_pooled", EXTERNAL)] + [(f"without_{p}", tuple(q for q in EXTERNAL if q != p)) for p in EXTERNAL]:
        pool = eff[eff.source_panel.isin(panels)]
        for family, contrasts in FAMILIES.items():
            matrix = pool[[f"{c}|{m}" for c, m in contrasts]].to_numpy(float)
            for shrink in SHRINK:
                for total in TOTALS:
                    power.append({"proxy": scope, "family": family, "shrink": shrink, "total_mdps": total,
                                  "power": gate_power(matrix, total, shrink, rng)})
    table = pd.DataFrame(power)
    table.to_csv(OUTPUT / "power.csv", index=False)
    files = [OUTPUT / n for n in ("effects_by_mdp.csv", "effect_sd.csv", "power.csv")]
    manifest = {"status": "posthoc_descriptive", "task": "T8 panel-6 sample size from opened panels",
                "created_utc": datetime.now(timezone.utc).isoformat(), "tau": TAU, "lambda": LAMBDA,
                "simulation": {"sims": SIMS, "bootstrap_draws": DRAWS, "seed": SEED, "rule": "max-t with the original standard error"},
                "b2_plus_prefix_fqe_reproduction_max_abs_diff": reproduction,
                "inputs_sha256": {B.OUTPUT.relative_to(ROOT).as_posix() + "/" + n: X.sha(B.OUTPUT / n)
                                  for n in ("scores_rows.csv.gz", "manifest.json")},
                "outputs_sha256": {p.name: X.sha(p) for p in files},
                "source_sha256": {Path(__file__).relative_to(ROOT).as_posix(): X.sha(Path(__file__))}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(pd.DataFrame(sd).query("scope == 'external_pooled'").round(4).to_string(index=False))
    print(table[table.proxy == "external_pooled"].pivot_table(index=["family", "shrink"], columns="total_mdps", values="power").round(2).to_string())
    worst = table[table.proxy != "external_pooled"].groupby(["family", "shrink", "total_mdps"]).power.min().unstack()
    print("worst leave-one-panel-out proxy"); print(worst.round(2).to_string())


if __name__ == "__main__":
    main()

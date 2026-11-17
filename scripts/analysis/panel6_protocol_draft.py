"""Draft protocol for panel 6 (task card T8). Writes a DRAFT for checkpoint H1; it freezes nothing.

The draft records proposed choices and the decisions left to the human. At H1 the human approves or edits it. After
that, a panel-6 runner (task card T9) freezes the approved version under its own namespace, with ``protocol.json`` and
``protocol.sha256``, before any panel data exist.

Usage: ``python -m scripts.analysis.panel6_protocol_draft``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import pandas as pd

from scripts.analysis import panel6_families as F
from scripts.analysis import panel6_power as W

ROOT = F.ROOT
OUTPUT = ROOT / "artifacts/week8/panel6_protocol_draft_2026-10-15"
OEH_SPEC = ROOT / "artifacts/week8/oeh_v2_development_2026-10-14/oeh_v2_spec.json"
TARGET_POWER, DESIGN_SHRINK, MIN_PER_FAMILY = 0.90, 0.75, 36
CANDIDATE_SEEDS = {"mdpp": 20301115, "labmaze": 20301116}
DATA_SEED_BASE = {"mdpp": 30_000_000, "labmaze": 31_000_000}
BOOTSTRAP = {"draws": 10_000, "seed": 20261216, "rule": "max-t with the original standard error (one-sided)"}


def sample_size() -> dict:
    power = pd.read_csv(W.OUTPUT / "power.csv")
    worst = power[power.proxy != "external_pooled"].groupby(["family", "shrink", "total_mdps"]).power.min().reset_index()
    f1 = worst[(worst.family == "F1_vs_prefix_fqe") & (worst.shrink == DESIGN_SHRINK)]
    ok = f1[f1.power >= TARGET_POWER].total_mdps
    total = int(ok.min()) if len(ok) else int(f1.total_mdps.max())
    per_family = max(MIN_PER_FAMILY, math.ceil(total / 2))
    at = worst[worst.total_mdps == min(W.TOTALS, key=lambda t: abs(t - 2 * per_family))]
    return {"rule": f"smallest total in {list(W.TOTALS)} with power >= {TARGET_POWER} for gate family F1 when the opened-panel "
                    f"mean effects are scaled by {DESIGN_SHRINK}, under the worst leave-one-external-panel-out proxy; at least "
                    f"{MIN_PER_FAMILY} MDPs per family",
            "reached_target": bool(len(ok)), "mdps_per_family": per_family, "total_mdps": 2 * per_family,
            "simulated_power_worst_proxy": {f"{r.family}|shrink={r.shrink}": round(float(r.power), 3) for r in at.itertuples()},
            "source": "artifacts/week8/panel6_power_2026-10-15/power.csv"}


def candidates_needed(per_family: int) -> dict:
    summary = pd.read_csv(F.OUTPUT / "summary.csv").set_index("family")
    out = {}
    for family in F.FAMILIES:
        rate = summary.loc[family, "eligible"] / summary.loc[family, "candidates"]
        out[family] = {"development_eligible_rate": round(float(rate), 3),
                       "candidates": int(math.ceil(per_family / max(rate, 1e-3) * 1.5 / 10) * 10)}
    return out


def dynamic_range_note() -> str:
    summary = pd.read_csv(F.OUTPUT / "summary.csv").set_index("family")
    names = {"mdpp": "MDP Playground", "labmaze": "labmaze"}
    return ", ".join(f"{names[f]} {summary.loc[f, 'min_dynamic_range']:.1f}-{summary.loc[f, 'max_dynamic_range']:.1f}" for f in F.FAMILIES)


def draft() -> dict:
    size = sample_size()
    need = candidates_needed(size["mdps_per_family"])
    feas = json.loads((F.OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    families = {family: {**F.FAMILIES[family], "mdps": size["mdps_per_family"], "candidates": need[family]["candidates"],
                         "candidate_seed": CANDIDATE_SEEDS[family], "data_seed_base": DATA_SEED_BASE[family],
                         "development_eligible_rate": need[family]["development_eligible_rate"],
                         "library": {"mdpp": "mdp-playground==1.0.0 (Apache-2.0)", "labmaze": "labmaze==1.0.6 (Apache-2.0)"}[family]}
                for family in F.FAMILIES}
    return {
        "schema": "oeh-panel6-v1-draft",
        "status": "DRAFT for checkpoint H1: not frozen, no panel data generated, no learner run on any candidate family",
        "namespace": "oeh-panel6-v1-<freeze date> (set at freeze)",
        "method": "OEH final method = oeh_v2_spec.json via scripts/analysis/oeh_pipeline.py (unchanged since the four-panel test)",
        "oeh_spec_sha256": F.sha(OEH_SPEC),
        "families": families,
        "kernel_source": "isolated venv (scripts/panel6/export_kernels.py); the project environment keeps Gymnasium 1.0.0",
        "conversion": "episodic to continuing: terminal states restart from the initial distribution; reward kept on entry; rewards in [0, 1]",
        "selection": "first `mdps` candidates in list order with |S||A| <= 256 that pass the project's dynamic-range filter, whose "
                     "exact fingerprint is not in run_oeh_final_120.seen_kernels and whose relabelling-invariant signature matches "
                     "no earlier external kernel and no earlier candidate (scripts/analysis/panel6_families.py)",
        "data": {"behaviors": 5, "pilot_multipliers": [2, 5, 10], "budgets": [1, 2, 5, 10, 20, 50, 100, 200], "training_seeds": 10,
                 "learners": ["behavior_cloning_tabular", "cfqi_tabular", "iql_tabular"],
                 "primary_label": "first six budgets (sustained 8/10 seeds at NR >= 0.95), seven classes (unchanged)"},
        "questions": {
            "Q1": "Does OEH forecast better than the strongest pilot-extrapolation baseline (B2 + prefix/FQE, task card T2) on the "
                  "budgets the x10 pilot cannot reach (extrapolation endpoint, task card T1)?",
            "Q2": "Same endpoint against B2.",
            "Q3": "Does the OEH forecast lead to lower decision regret (tau = 0.8, lambda = 100) than either comparator?",
        },
        "endpoints": {"ibs_ge": "integrated Brier over the thresholds x10, x20, x50 (scripts/analysis/extrapolation_endpoint.py)",
                      "ll_ext": "log loss of the coarsened law {<= x10, x20, x50, > x50}, floor 1e-12",
                      "regret": "colosseum_posthoc_20261014.decision_costs(cdf, event, tau=0.8, lam=100).regret"},
        "gates": {
            "order": "fixed sequence F1 -> F2 -> F3; a family is tested only if every earlier family is GO",
            "unit": "MDP (pooled over both families, equal weight); per-MDP mean over the 5 behaviors x 3 learners",
            "bootstrap": BOOTSTRAP,
            "F1": {"contrasts": [["b2_plus_prefix_fqe", "oeh", "ibs_ge"], ["b2_plus_prefix_fqe", "oeh", "ll_ext"]],
                   "go": "both simultaneous lower bounds > 0"},
            "F2": {"contrasts": [["b2_matched", "oeh", "ibs_ge"], ["b2_matched", "oeh", "ll_ext"]], "go": "both simultaneous lower bounds > 0"},
            "F3": {"contrasts": [["b2_plus_prefix_fqe", "oeh", "regret"], ["b2_matched", "oeh", "regret"]],
                   "go": "both simultaneous lower bounds > 0"},
        },
        "not_tested": ["OEH vs B2 + diagnostics (robust head): the panel-5 contrast is not re-tested; reported only descriptively",
                       "full-endpoint IBS and LL: reported descriptively for every method"],
        "sample_size": size,
        "descriptive": {
            "methods": {"oeh": "OEH (final method)", "oeh_full_head": "OEH-F, full 108-field head", "robust": "B2 + diagnostics",
                        "b2_matched": "B2", "xgboost_cf_18": "XGBoost-CF on B2", "xgboost_cf_108": "XGBoost-CF on the 108 fields",
                        "prefix_fqe_raw": "prefix + FQE, zero-shot law (T2)", "b2_plus_prefix_fqe": "B2 + prefix/FQE (T2)",
                        "lce_raw": "LCE", "b2_plus_lce": "B2 + LCE", "knn_analog": "k-NN analog on B2", "logistic": "logistic regression on B2",
                        "s4": "T4 candidate S4 (B2 + P + D, one head, gated); descriptive only, no registration",
                        "oeh_without_horizon": "OEH without the OEH-H block"},
            "conditional_methods": {"lc_pfn": "only if the T3 licence decision is taken before freeze",
                                    "tabpfn": "only if the T3 licence decision is taken before freeze (token and terms are a human step)"},
            "also": ["full endpoint IBS/LL", "per family", "per learner", "gain by distance", "Murphy decomposition",
                     "label variant tau = 0.99 (task card T6), descriptive", "decision regret of every method"],
        },
        "disclosures": [
            "Feasibility only: development candidates drawn with numpy seed 1 (never the panel seeds) were exported, converted and "
            "passed through the size cap and the dynamic-range filter (artifacts/week8/panel6_feasibility_2026-10-15); no learner "
            "was run on any MDP Playground or labmaze MDP.",
            "The extrapolation endpoint (T1) and the prefix + FQE baseline (T2) were defined and run post hoc on opened panels; "
            "panel 6 registers them prospectively.",
            "The sample size uses per-MDP effects of opened panels (artifacts/week8/panel6_power_2026-10-15).",
            "Seed ranges are checked against every earlier dataset manifest at freeze.",
        ],
        "open_decisions_for_H1": [
            "Families: approve MDP Playground + labmaze (two new sources), or the one-source fallback (MDP Playground split by "
            "diameter: d = 1 vs d >= 2), or other families.",
            "labmaze dynamics: the generator gives the layout only; slip model and goal reward are ours (as for FrozenLake's "
            "conversion). Approve or change.",
            "Primary family F1 (vs B2 + prefix/FQE) and the order F1 -> F2 -> F3.",
            "Sample size rule (power target, shrink factor) and the resulting MDPs per family.",
            "F3 (decision regret) is underpowered (simulated 0.50 at 0.75 x the opened-panel effect, worst proxy): keep it as the "
            "last gate family or make it descriptive.",
            "Label degeneracy: development kernels have large dynamic ranges (" + dynamic_range_note() + "; threshold 0.2), so "
            "labels may pile up at x1, as FrozenLake piled up at > x50. Accept the risk, or allow a disclosed learner dry run on "
            "development draws (seed 1) before freeze.",
            "Whether LC-PFN/TabPFN enter (needs the T3 licence decision first).",
            "Whether S4 is listed (descriptive only; T4 recommends no registration).",
        ],
        "inputs_sha256": {**{p.relative_to(ROOT).as_posix(): F.sha(p) for p in (F.OUTPUT / "summary.csv", F.OUTPUT / "manifest.json",
                                                                              W.OUTPUT / "power.csv", W.OUTPUT / "manifest.json")},
                          "feasibility_versions": feas["versions"]},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    doc = draft()
    (OUTPUT / "protocol_draft.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ("sample_size",)}, indent=1))
    print({f: (v["mdps"], v["candidates"], v["development_eligible_rate"]) for f, v in doc["families"].items()})


if __name__ == "__main__":
    main()

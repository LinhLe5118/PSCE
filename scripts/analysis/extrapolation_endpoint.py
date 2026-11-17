"""Extrapolation endpoint (task card T1; post hoc, descriptive).

The pilot holds x10 transitions, so forecasts of the classes x1-x5 are partly retrospective
(a practitioner could train the learner on pilot prefixes). This scorer restricts the scores to
the budgets a pilot cannot reach:

* ``ibs_ge``: integrated Brier over the thresholds at or above the pilot budget,
  P(E <= x10), P(E <= x20), P(E <= x50); the last one carries the censored class (card T1);
* ``ibs_gt``: the strict variant, thresholds x20 and x50 only;
* ``ll_ext``: log loss of the coarsened law over {<= x10, x20, x50, > x50}, floor 1e-12.

With all six thresholds and the seven classes the scorer reproduces the sealed summaries
(checked in ``equivalence``). Rows are averaged within each MDP, then across MDPs.

Usage: ``python -m scripts.analysis.extrapolation_endpoint [--output DIR]``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "artifacts"
OUTPUT = A / "week8/extrapolation_endpoint_2026-10-15"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]
PILOT_INDEX = 3  # budgets x1, x2, x5, x10, x20, x50 -> the pilot is x10
FULL = tuple(range(6))
GE = tuple(range(PILOT_INDEX, 6))
GT = tuple(range(PILOT_INDEX + 1, 6))
FLOOR = 1e-12

# canonical method -> names used by the panel artifacts
METHODS = {"oeh": ("v2", "oeh"), "oeh_f": ("oeh_final", "oeh_full_head"),
           "robust": ("prior_oeh_matched", "legacy", "robust"), "b2": ("b2_matched",)}
LABEL = {"oeh": "OEH", "oeh_f": "OEH-F (full head)", "robust": "Robust head (B2 + D)", "b2": "B2 matched"}


def _panel(pred, labels, role, summary=None):
    return {"pred": [pred] if isinstance(pred, Path) else list(pred), "labels": labels, "role": role, "summary": summary}


V2 = A / "oeh_v2_sealed_test_2026-10-14"
P5 = A / "oeh_fifth_panel_2026-10-15"
PANELS = {
    "cohort7": _panel(A / "oeh_confirmatory_unseen_120_2026-10-04/predictions_preoutcome.csv",
                      A / "oeh_confirmatory_unseen_120_2026-10-04/labels.csv", "development",
                      ("cohort7", A / "oeh_confirmatory_unseen_120_2026-10-04/summary.csv")),
    "cohort8": _panel(A / "oeh_eighth_cohort_2026-10-14/internal/predictions_preoutcome.csv",
                      A / "oeh_eighth_cohort_2026-10-14/internal/labels.csv", "development",
                      ("pooled", A / "oeh_eighth_cohort_2026-10-14/internal/summary.csv")),
    "colosseum2027": _panel((A / "week8/colosseum_confirm_2026-10-14/colosseum_predictions_preoutcome.csv",
                             A / "week8/colosseum_confirm_2026-10-14/comparator_predictions.csv"),
                            A / "week8/colosseum_confirm_2026-10-14/colosseum_labels.csv", "development"),
    "colosseum2028": _panel(A / "oeh_eighth_cohort_2026-10-14/colosseum/predictions_preoutcome.csv",
                            A / "oeh_eighth_cohort_2026-10-14/colosseum/labels.csv", "development",
                            ("pooled", A / "oeh_eighth_cohort_2026-10-14/colosseum/summary.csv")),
    "spibb": _panel(A / "oeh_spibb_panel_2026-10-14/spibb/predictions_preoutcome.csv",
                    A / "oeh_spibb_panel_2026-10-14/spibb/labels.csv", "development",
                    ("pooled", A / "oeh_spibb_panel_2026-10-14/spibb/summary.csv")),
    **{p: _panel(V2 / p / "predictions_preoutcome.csv", V2 / p / "labels.csv", "final_internal" if p == "internal" else "final_external",
                 ("v2test", V2 / "summary.csv")) for p in ("internal", "colosseum2029", "spibb2", "forest")},
    **{p: _panel(P5 / p / "predictions_preoutcome.csv", P5 / p / "labels.csv", "panel5", ("panel5", P5 / "summary.csv"))
       for p in ("frozenlake", "gambler")},
}
GROUPS = {"final_external_pooled": ("colosseum2029", "spibb2", "forest"), "panel5_pooled": ("frozenlake", "gambler"),
          "development_external_pooled": ("colosseum2027", "colosseum2028", "spibb")}


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #

def ibs(p: np.ndarray, y: np.ndarray, thresholds=FULL) -> np.ndarray:
    """Row-wise integrated Brier over the given cumulative thresholds."""
    t = np.asarray(thresholds)
    cdf = np.cumsum(p, 1)[:, t]
    return ((cdf - (y[:, None] <= t[None, :])) ** 2).mean(1)


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], FLOOR, 1))


def coarsen(p: np.ndarray, y: np.ndarray, pilot: int = PILOT_INDEX) -> tuple[np.ndarray, np.ndarray]:
    """Merge the classes at or below the pilot budget into one class."""
    q = np.column_stack([p[:, :pilot + 1].sum(1), p[:, pilot + 1:]])
    return q, np.maximum(y - pilot, 0)


def row_scores(p: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    q, z = coarsen(p, y)
    return {"ibs": ibs(p, y), "ll": log_loss(p, y), "ibs_ge": ibs(p, y, GE), "ibs_gt": ibs(p, y, GT), "ll_ext": log_loss(q, z)}


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #

def load(panel: str) -> pd.DataFrame:
    spec = PANELS[panel]
    pred = pd.concat([pd.read_csv(f) for f in spec["pred"]], ignore_index=True)
    if "panel" in pred:
        pred = pred[pred.panel == "primary"]
    names = {name: canon for canon, alias in METHODS.items() for name in alias}
    pred = pred[pred.method.isin(names)].assign(canonical=lambda f: f.method.map(names))
    pred = pred.drop_duplicates(KEYS + ["canonical"])
    lab = pd.read_csv(spec["labels"])
    if "panel" in lab:
        lab = lab[lab.panel == "primary"]
    m = pred.merge(lab[KEYS + ["event"]], on=KEYS, validate="many_to_one")
    rows = row_scores(m[P].to_numpy(float), m.event.to_numpy(int))
    for k, v in rows.items():
        m[k] = v
    m["source_panel"] = panel
    return m


def per_mdp(scored: pd.DataFrame, column: str = "canonical") -> pd.DataFrame:
    return scored.groupby(["source_panel", column, "mdp_id"])[["ibs", "ll", "ibs_ge", "ibs_gt", "ll_ext"]].mean().reset_index()


def equivalence() -> list[dict]:
    """Full-endpoint scores of this scorer vs the sealed summaries (every method, raw names)."""
    out = []
    for panel, spec in PANELS.items():
        if spec["summary"] is None:
            continue
        kind, path = spec["summary"]
        pred = pd.concat([pd.read_csv(f) for f in spec["pred"]], ignore_index=True)
        if "panel" in pred:
            pred = pred[pred.panel == "primary"]
        lab = pd.read_csv(spec["labels"])
        if "panel" in lab:
            lab = lab[lab.panel == "primary"]
        m = pred.merge(lab[KEYS + ["event"]], on=KEYS, validate="many_to_one")
        p, y = m[P].to_numpy(float), m.event.to_numpy(int)
        m["ibs"], m["ll"] = ibs(p, y), log_loss(p, y)
        mine = m.groupby(["method", "mdp_id"])[["ibs", "ll"]].mean().groupby("method").mean()
        s = pd.read_csv(path)
        if kind == "cohort7":
            s = s[s.panel == "primary"].rename(columns={"integrated_brier": "brier", "seven_class_log_loss": "log_loss"})
        elif kind == "pooled":
            s = s[s.scope == "pooled"]
        elif kind == "v2test":
            s = s[(s.scope == "panel") & (s.source_panel == panel)]
        else:
            s = s[s.scope == panel]
        s = s.set_index("method")
        for method in mine.index.intersection(s.index):
            out.append({"panel": panel, "method": method, "ibs_diff": float(abs(mine.loc[method, "ibs"] - s.loc[method, "brier"])),
                        "ll_diff": float(abs(mine.loc[method, "ll"] - s.loc[method, "log_loss"]))})
    return out


def gains(pm: pd.DataFrame, panels, baseline: str, candidate: str) -> dict:
    w = pm[pm.source_panel.isin(panels)].pivot_table(index=["source_panel", "mdp_id"], columns="canonical",
                                                    values=["ibs", "ll", "ibs_ge", "ibs_gt", "ll_ext"])
    if (("ibs", baseline) not in w) or (("ibs", candidate) not in w):
        return {}
    out = {"mdps": int(len(w))}
    for metric in ("ibs", "ll", "ibs_ge", "ibs_gt", "ll_ext"):
        d = (w[(metric, baseline)] - w[(metric, candidate)]).dropna().to_numpy()
        half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        base = w[(metric, baseline)].mean()
        out.update({f"{metric}_gain": float(d.mean()), f"{metric}_low95": float(d.mean() - half), f"{metric}_high95": float(d.mean() + half),
                    f"{metric}_rel": float(d.mean() / base) if base else float("nan")})
    return out


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    eq = pd.DataFrame(equivalence())
    eq.to_csv(out / "equivalence.csv", index=False)
    worst = float(eq[["ibs_diff", "ll_diff"]].to_numpy().max())
    if worst > 1e-12:
        raise RuntimeError(f"the full endpoint does not reproduce the sealed summaries (max diff {worst})")
    scored = pd.concat([load(p) for p in PANELS], ignore_index=True)
    pm = per_mdp(scored)
    table = pm.groupby(["source_panel", "canonical"])[["ibs", "ll", "ibs_ge", "ibs_gt", "ll_ext"]].mean().reset_index()
    table["role"] = table.source_panel.map(lambda p: PANELS[p]["role"])
    table.to_csv(out / "scores_by_panel.csv", index=False)
    rows = []
    for scope, panels in [(p, (p,)) for p in PANELS] + list(GROUPS.items()):
        for candidate in ("oeh", "oeh_f"):
            for baseline in ("b2", "robust", "oeh_f"):
                if candidate == baseline:
                    continue
                g = gains(pm, panels, baseline, candidate)
                if g:
                    rows.append({"scope": scope, "candidate": candidate, "baseline": baseline, **g})
    gain = pd.DataFrame(rows)
    gain.to_csv(out / "gains.csv", index=False)
    inputs = sorted({f for spec in PANELS.values() for f in spec["pred"]} | {spec["labels"] for spec in PANELS.values()}
                    | {spec["summary"][1] for spec in PANELS.values() if spec["summary"]})
    manifest = {
        "status": "posthoc_descriptive", "task": "T1 extrapolation endpoint", "created_utc": datetime.now(timezone.utc).isoformat(),
        "definitions": {"pilot": "x10 (index 3)", "ibs_ge": "thresholds x10, x20, x50 (at or above the pilot; card T1)",
                        "ibs_gt": "thresholds x20, x50 (strictly above the pilot)",
                        "ll_ext": "log loss of the coarsened law {<= x10, x20, x50, > x50}, floor 1e-12",
                        "averaging": "rows within MDP, then MDPs", "intervals": "95% t intervals of the mean per-MDP gain (descriptive)"},
        "equivalence_max_abs_diff": worst,
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        "outputs_sha256": {p.name: sha(p) for p in (out / "equivalence.csv", out / "scores_by_panel.csv", out / "gains.csv")},
        "source_sha256": {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print("equivalence max diff", worst, "over", len(eq), "panel-method pairs")
    show = gain[gain.baseline.isin(["b2", "robust"])][["scope", "candidate", "baseline", "mdps", "ibs_rel", "ibs_ge_rel", "ibs_gt_rel",
                                                        "ll_rel", "ll_ext_rel", "ll_ext_low95", "ll_ext_high95"]]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

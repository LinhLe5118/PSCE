"""Descriptive re-analysis of every sealed full-OEH panel (14/10/2026).

Reads sealed predictions and opened labels only; fits nothing that could change
a forecast and gates nothing. Panels: cohorts 4, 7, 8 (internal generator) and
Colosseum 2027/2028, SPIBB, BRIDGE Atari-5 and Expanded-5 (external). Outputs:

``verification.csv``      every panel/method mean IBS and LL from one common scorer;
``paired.csv``            OEH versus each comparator per panel (bootstrap, influence);
``decomposition.csv``     share of the OEH-B2 gain from censored versus finite rows;
``swap.csv``              hybrids that exchange P(>x50) and the finite-budget law;
``murphy.csv``            reliability / resolution / uncertainty per threshold;
``floor_sensitivity.csv`` log loss under several probability floors;
``meta_analysis.csv``     random-effects pooling of log(B2/OEH) across panels;
``shift.csv``             per-MDP distance to meta-training versus gain;
``noise_floor.csv``       label-noise floor from seed bootstraps;
``by_learner.csv``        OEH-B2 by learner and panel; ``holm.csv`` multiplicity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "artifacts"
OUTPUT = A / "week8/cross_panel_reanalysis_2026-10-14"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]
BUDGETS = (1, 2, 5, 10, 20, 50)
SEED, DRAWS = 20261016, 10_000
RENAME = {"oeh_full_108": "oeh_final", "b2_matched_internal_diagnostic": "b2_matched",
          "published_xgboost_cf_2026_adapted_b2": "xgboost_cf_18",
          "xgboost_cf_2026_full108_architecture_control": "xgboost_cf_108"}
METHODS = ("oeh_final", "b2_matched", "xgboost_cf_18", "prior_oeh_matched", "b2_plus_eri", "b2_plus_tq_saco_original",
           "xgboost_cf_108", "oeh_s", "oeh_2t")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #

def _internal(folder: Path, xgb: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(folder / "predictions_preoutcome.csv")
    if xgb is not None:
        extra = pd.read_csv(xgb)
        pred = pd.concat([pred, extra.assign(method="xgboost_cf_18")], ignore_index=True)
    labels = pd.read_csv(folder / "labels.csv")
    labels = labels[labels.panel == "primary"] if "panel" in labels else labels
    pred = pred[pred.mdp_id.isin(labels.mdp_id)]
    return pred, labels


def panels() -> dict[str, dict]:
    c4, c7 = A / "oeh_final_unseen_120_2026-10-02", A / "oeh_confirmatory_unseen_120_2026-10-04"
    c8 = A / "oeh_eighth_cohort_2026-10-14"
    col = A / "week8/colosseum_confirm_2026-10-14"
    out = {}
    out["cohort4"] = dict(kind="internal", data=_internal(c4), outcomes=c4 / "all_core_budget_seed_outcomes.csv",
                          features=c4 / "pilot_features.csv")
    out["cohort7"] = dict(kind="internal", data=_internal(c7, A / "week8/published_xgboost_cf_120_exploratory_2026-10-08/predictions.csv"),
                          outcomes=c7 / "all_core_budget_seed_outcomes.csv", features=c7 / "pilot_features.csv")
    p8 = pd.read_csv(c8 / "internal/predictions_preoutcome.csv")
    out["cohort8"] = dict(kind="internal", data=(p8, pd.read_csv(c8 / "internal/labels.csv")),
                          outcomes=c8 / "internal/outcomes.csv.gz", features=c8 / "internal/query_preoutcome.csv.gz")
    part2 = pd.read_csv(A / "week8/colosseum_posthoc_part2_2026-10-14/predictions_before_label_read.csv")
    first = pd.read_csv(A / "week8/colosseum_posthoc_2026-10-14/predictions_before_label_read.csv")
    p27 = pd.concat([part2, first[first.method == "xgboost_cf_18"]], ignore_index=True)
    out["colosseum2027"] = dict(kind="external", data=(p27, pd.read_csv(col / "colosseum_labels.csv")),
                                outcomes=col / "colosseum_outcomes.csv", features=col / "colosseum_pilot_features.csv")
    out["colosseum2028"] = dict(kind="external", data=(pd.read_csv(c8 / "colosseum/predictions_preoutcome.csv"),
                                                       pd.read_csv(c8 / "colosseum/labels.csv")),
                                outcomes=c8 / "colosseum/outcomes.csv.gz", features=c8 / "colosseum/query_preoutcome.csv.gz")
    sp = A / "oeh_spibb_panel_2026-10-14/spibb"
    out["spibb"] = dict(kind="external", data=(pd.read_csv(sp / "predictions_preoutcome.csv"), pd.read_csv(sp / "labels.csv")),
                        outcomes=sp / "outcomes.csv.gz", features=sp / "query_preoutcome.csv.gz")
    for name, folder, feats in (("bridge_atari5", "bridge_fresh_full_test_analysis_2026-10-08", "bridge_fresh_pilot_features_preoutcome_2026-10-08"),
                                ("bridge_expanded5", "bridge_expanded_full_test_analysis_2026-10-08", "bridge_expanded_pilot_features_preoutcome_2026-10-08")):
        scored = pd.read_csv(A / "week8" / folder / "scored.csv").replace({"method": RENAME})
        out[name] = dict(kind="external", data=(scored[[*KEYS, "method", *P]], scored[KEYS + ["event"]].drop_duplicates(KEYS)),
                         outcomes=None, features=A / "week8" / feats / "features.csv")
    return out


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #

def score(frame: pd.DataFrame, floor: float = 1e-12) -> pd.DataFrame:
    p, y = frame[P].to_numpy(float), frame.event.to_numpy(int)
    frame = frame.copy()
    frame["ibs"] = ((np.cumsum(p, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)
    frame["ll"] = -np.log(np.maximum(p[np.arange(len(y)), y], floor))
    return frame


def joined(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    pred = pred.replace({"method": RENAME})
    pred = pred[pred.method.isin(METHODS)][[*KEYS, "method", *P]]
    return pred.merge(labels[KEYS + ["event"]], on=KEYS, validate="many_to_one")


def per_mdp(scored: pd.DataFrame, value: str) -> pd.DataFrame:
    return scored.groupby(["method", "mdp_id"])[value].mean().unstack("method")


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    index = rng.integers(0, len(x), size=(DRAWS, len(x)))
    means = x[index].mean(1)
    p_one_sided = float((means <= 0).mean())  # H0: no improvement
    return float(np.quantile(means, .025)), float(np.quantile(means, .975)), p_one_sided


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #

def swap_hybrids(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Exchange P(E=6) and the finite-budget conditional law between OEH and B2."""
    oeh = frame[frame.method == "oeh_final"].set_index(KEYS).sort_index()
    b2 = frame[frame.method == "b2_matched"].set_index(KEYS).sort_index()
    out = {}
    for name, (censor, finite) in {"oeh_p6_b2_finite": (oeh, b2), "b2_p6_oeh_finite": (b2, oeh)}.items():
        p6 = censor["p_6"].to_numpy()
        cond = finite[P[:6]].to_numpy()
        cond = cond / np.clip(cond.sum(1, keepdims=True), 1e-300, None)
        h = oeh[["event"]].copy()
        h[P] = np.column_stack([cond * (1 - p6)[:, None], p6])
        h["method"] = name
        out[name] = h.reset_index()
    return out


def murphy(frame: pd.DataFrame, bins: int = 10) -> list[dict]:
    rows = []
    for method, g in frame.groupby("method"):
        cdf = np.cumsum(g[P].to_numpy(float), 1)[:, :6]
        y = g.event.to_numpy(int)
        for j, b in enumerate(BUDGETS):
            f, o = cdf[:, j], (y <= j).astype(float)
            edges = np.minimum((f * bins).astype(int), bins - 1)
            rel = res = 0.0
            for k in range(bins):
                m = edges == k
                if m.any():
                    rel += m.mean() * (f[m].mean() - o[m].mean()) ** 2
                    res += m.mean() * (o[m].mean() - o.mean()) ** 2
            rows.append({"method": method, "budget": b, "brier": float(((f - o) ** 2).mean()), "reliability": rel,
                         "resolution": res, "uncertainty": float(o.mean() * (1 - o.mean())),
                         "mean_forecast": float(f.mean()), "observed_rate": float(o.mean())})
    return rows


def noise_floor(outcomes: Path, labels: pd.DataFrame, rng: np.random.Generator, boots: int = 200) -> pd.DataFrame:
    """Plug-in label law from seed bootstraps (an optimistic lower bound on attainable error)."""
    from scripts.analysis.compare_difficulty import suffix_event

    out = pd.read_csv(outcomes, usecols=["mdp_id", "policy_family", "algorithm", "budget", "training_seed", "success"])
    out = out[out.mdp_id.isin(labels.mdp_id)]
    cube = out.pivot_table(index=KEYS, columns=["budget", "training_seed"], values="success", aggfunc="first")
    cube = cube.reindex(columns=pd.MultiIndex.from_product([BUDGETS, range(10)]))
    arr = cube.to_numpy(float).reshape(len(cube), 6, 10)
    laws = np.zeros((len(cube), 7))
    for _ in range(boots):
        pick = rng.integers(0, 10, 10)
        counts = arr[:, :, pick].sum(2)
        events = np.array([suffix_event(tuple(int(c) for c in row)) for row in counts])
        laws[np.arange(len(cube)), events] += 1
    laws /= boots
    frame = cube.index.to_frame(index=False)
    frame[P] = laws
    frame["method"] = "seed_bootstrap_plugin"
    return frame.merge(labels[KEYS + ["event"]], on=KEYS, validate="one_to_one")


def shift_distance(meta: pd.DataFrame, features: pd.DataFrame, k: int = 10) -> pd.Series:
    from scipy.spatial import cKDTree
    from scripts.analysis.compare_difficulty import FEATURES_B2

    cols = list(FEATURES_B2)
    mu, sd = meta[cols].mean(), meta[cols].std().replace(0, 1)
    tree = cKDTree(((meta[cols] - mu) / sd).fillna(0).to_numpy())
    q = ((features[cols] - mu) / sd).fillna(0).to_numpy()
    dist, _ = tree.query(q, k=k)
    return pd.Series(dist.mean(1), index=features.index).groupby(features.mdp_id.to_numpy()).mean()


def random_effects(estimates: np.ndarray, variances: np.ndarray) -> dict:
    w = 1 / variances
    fixed = float((w * estimates).sum() / w.sum())
    q = float((w * (estimates - fixed) ** 2).sum())
    df = len(estimates) - 1
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - df) / c) if df > 0 else 0.0
    wr = 1 / (variances + tau2)
    pooled = float((wr * estimates).sum() / wr.sum())
    se = float(np.sqrt(1 / wr.sum()))
    return {"pooled": pooled, "se": se, "ci_low": pooled - 1.96 * se, "ci_high": pooled + 1.96 * se,
            "tau2": tau2, "q": q, "i2": max(0.0, (q - df) / q) if q > 0 else 0.0, "panels": len(estimates)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    from scripts.analysis import run_oeh_confirmatory_20261004 as C
    from scripts.analysis import run_oeh_final_120 as R
    C.configure()
    tag = json.loads((A / "oeh_confirmatory_unseen_120_2026-10-04/protocol.json").read_text(encoding="utf-8"))["oeh_final"]["structure_tag"]
    meta = R.meta_training(R.OPENED, tag)
    meta = meta[meta.panel == "primary"]

    verification, paired, decomposition, swaps, murphy_rows, floors = [], [], [], [], [], []
    shift_rows, noise_rows, learner_rows, meta_rows = [], [], [], []
    for name, spec in panels().items():
        pred, labels = spec["data"]
        frame = joined(pred, labels)
        scored = score(frame)
        n_mdp = scored.mdp_id.nunique()
        for method, g in scored.groupby("method"):
            m = g.groupby("mdp_id")[["ibs", "ll"]].mean().mean()
            verification.append({"panel": name, "kind": spec["kind"], "method": method, "mdps": g.mdp_id.nunique(),
                                 "rows": len(g), "ibs": m.ibs, "ll": m.ll, "censored_share": float((g.event == 6).mean())})
        ibs, ll = per_mdp(scored, "ibs"), per_mdp(scored, "ll")
        for comp in [m for m in ibs.columns if m != "oeh_final"]:
            for metric, table in (("ibs", ibs), ("ll", ll)):
                d = (table[comp] - table["oeh_final"]).dropna().to_numpy()
                if not len(d):
                    continue
                low, high, pval = boot_ci(d, rng)
                order = np.sort(d)
                paired.append({"panel": name, "kind": spec["kind"], "comparator": comp, "metric": metric, "mdps": len(d),
                               "comparator_mean": table[comp].mean(), "oeh_mean": table["oeh_final"].mean(),
                               "mean_gain": d.mean(), "relative_gain": d.mean() / table[comp].mean(),
                               "median_gain": float(np.median(d)), "ci95_low": low, "ci95_high": high,
                               "p_one_sided": pval, "t": d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan,
                               "share_mdps_better": float((d > 0).mean()),
                               "gain_without_best5": float(order[:-5].mean()) if len(d) > 10 else np.nan,
                               "gain_without_worst5": float(order[5:].mean()) if len(d) > 10 else np.nan})
        # censored/finite decomposition of the OEH-B2 gain (MDP mean of row means)
        two = scored[scored.method == "oeh_final"][[*KEYS, "event", "ibs", "ll"]].merge(
            scored[scored.method == "b2_matched"][[*KEYS, "ibs", "ll"]], on=KEYS, suffixes=("_oeh", "_b2"),
            validate="one_to_one")
        rows_per = two.groupby("mdp_id").mdp_id.transform("size")
        for metric in ("ibs", "ll"):
            diff = (two[f"{metric}_b2"] - two[f"{metric}_oeh"]) / rows_per / n_mdp
            total = diff.sum()
            for part, mask in (("censored", two.event == 6), ("finite", two.event < 6)):
                decomposition.append({"panel": name, "metric": metric, "part": part, "rows": int(mask.sum()),
                                      "gain_contribution": float(diff[mask].sum()), "share": float(diff[mask].sum() / total)})
        # swap hybrids
        hyb = pd.concat(swap_hybrids(frame).values(), ignore_index=True)
        hs = score(hyb)
        for method, g in pd.concat([scored[scored.method.isin(["oeh_final", "b2_matched"])], hs]).groupby("method"):
            m = g.groupby("mdp_id")[["ibs", "ll"]].mean().mean()
            swaps.append({"panel": name, "method": method, "ibs": m.ibs, "ll": m.ll})
        murphy_rows += [{"panel": name, **r} for r in murphy(frame[frame.method.isin(["oeh_final", "b2_matched", "prior_oeh_matched"])])]
        for floor in (1e-15, 1e-12, 1e-6, 1e-3, 1e-2):
            s = score(frame[frame.method.isin(["oeh_final", "b2_matched", "xgboost_cf_18", "prior_oeh_matched"])], floor)
            for method, g in s.groupby("method"):
                floors.append({"panel": name, "floor": floor, "method": method,
                               "ll": g.groupby("mdp_id").ll.mean().mean()})
        # by learner
        for (learner, method), g in scored[scored.method.isin(["oeh_final", "b2_matched"])].groupby(["algorithm", "method"]):
            m = g.groupby("mdp_id")[["ibs", "ll"]].mean().mean()
            learner_rows.append({"panel": name, "algorithm": learner, "method": method, "ibs": m.ibs, "ll": m.ll,
                                 "censored_share": float((g.event == 6).mean())})
        # meta-analysis inputs: log ratio of MDP means, bootstrap variance
        for metric, table in (("ibs", ibs), ("ll", ll)):
            t = table[["b2_matched", "oeh_final"]].dropna()
            index = rng.integers(0, len(t), size=(2000, len(t)))
            ratios = np.log(t.b2_matched.to_numpy()[index].mean(1) / t.oeh_final.to_numpy()[index].mean(1))
            meta_rows.append({"panel": name, "kind": spec["kind"], "metric": metric, "mdps": len(t),
                              "log_ratio": float(np.log(t.b2_matched.mean() / t.oeh_final.mean())),
                              "variance": float(ratios.var(ddof=1))})
        # distribution shift versus gain
        if spec["features"] is not None and Path(spec["features"]).exists():
            feats = pd.read_csv(spec["features"])
            if "algorithm" in feats:
                feats = feats.drop_duplicates(["mdp_id", "policy_family"])
            feats = feats[feats.mdp_id.isin(ibs.index)]
            try:
                dist = shift_distance(meta, feats)
                g_ibs = (ibs.b2_matched - ibs.oeh_final).reindex(dist.index)
                g_ll = (ll.b2_matched - ll.oeh_final).reindex(dist.index)
                for mdp in dist.index:
                    shift_rows.append({"panel": name, "kind": spec["kind"], "mdp_id": mdp, "knn_distance": dist[mdp],
                                       "gain_ibs": g_ibs[mdp], "gain_ll": g_ll[mdp]})
            except KeyError as error:
                print(f"shift skipped for {name}: {error}")
        # label-noise floor
        if spec["outcomes"] is not None:
            plug = score(noise_floor(spec["outcomes"], labels, rng))
            m = plug.groupby("mdp_id")[["ibs", "ll"]].mean().mean()
            noise_rows.append({"panel": name, "plugin_ibs": m.ibs, "plugin_ll": m.ll,
                               "oeh_ibs": float(ibs.oeh_final.mean()), "b2_ibs": float(ibs.b2_matched.mean())})
        print(f"{name}: {n_mdp} MDPs done", flush=True)

    pd.DataFrame(verification).to_csv(out / "verification.csv", index=False)
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "paired.csv", index=False)
    pd.DataFrame(decomposition).to_csv(out / "decomposition.csv", index=False)
    pd.DataFrame(swaps).to_csv(out / "swap.csv", index=False)
    pd.DataFrame(murphy_rows).to_csv(out / "murphy.csv", index=False)
    pd.DataFrame(floors).to_csv(out / "floor_sensitivity.csv", index=False)
    pd.DataFrame(learner_rows).to_csv(out / "by_learner.csv", index=False)
    shift = pd.DataFrame(shift_rows)
    shift.to_csv(out / "shift.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(out / "noise_floor.csv", index=False)
    meta_in = pd.DataFrame(meta_rows)
    pooled = []
    for metric in ("ibs", "ll"):
        for group in ("internal", "external", "all"):
            g = meta_in[(meta_in.metric == metric) & ((meta_in.kind == group) | (group == "all"))]
            pooled.append({"metric": metric, "group": group, **random_effects(g.log_ratio.to_numpy(), g.variance.to_numpy())})
    pd.concat([meta_in.assign(row="panel"), pd.DataFrame(pooled).assign(row="pooled")]).to_csv(out / "meta_analysis.csv", index=False)
    primary = paired[paired.comparator == "b2_matched"].copy()
    m = len(primary)
    primary = primary.sort_values("p_one_sided")
    primary["holm_adjusted_p"] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * primary.p_one_sided.to_numpy()))
    primary.to_csv(out / "holm.csv", index=False)
    stats = []
    for kind in ("internal", "external"):
        s = shift[shift.kind == kind]
        if len(s) > 5:
            stats.append({"kind": kind, "mdps": len(s), "median_distance": s.knn_distance.median(),
                          "spearman_ibs": spearmanr(s.knn_distance, s.gain_ibs).statistic,
                          "spearman_ll": spearmanr(s.knn_distance, s.gain_ll).statistic})
    pd.DataFrame(stats).to_csv(out / "shift_summary.csv", index=False)
    names = sorted(p.name for p in out.glob("*.csv"))
    (out / "manifest.json").write_text(json.dumps({
        "status": "descriptive re-analysis of sealed predictions and opened labels; no gate",
        "bootstrap": f"{DRAWS} MDP draws, seed {SEED}", "ll_floor": 1e-12,
        "source_sha256": sha256(Path(__file__)), "output_sha256": {n: sha256(out / n) for n in names}}, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()

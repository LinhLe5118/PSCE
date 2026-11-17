"""Post-hoc analyses of the sealed four-panel test for the final OEH method (exploratory: labels are opened).

* gain over B2 by shift distance d for the final method, its full head alone and the B2 + diagnostics head;
* OEH built without the OEH-H block (G4, G5) - the horizon ablation of the final method;
* two generic baselines on the B2 statistics (k-NN analog, multinomial logistic regression).

Sealed predictions are reused where they exist; the extra heads are fitted once on the
meta panel for the union of the four query tables (a fit does not depend on the query).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.analysis import oeh_pipeline as O

ROOT = O.ROOT
SEALED = ROOT / "artifacts/oeh_v2_sealed_test_2026-10-14"
OUT = ROOT / "artifacts/week8/oeh_final_posthoc_2026-10-15"
PANELS = ("internal", "colosseum2029", "spibb2", "forest")
EXTERNAL = ("colosseum2029", "spibb2", "forest")
EDGES = (0, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, np.inf)
NAMES = {"v2": "OEH", "oeh_final": "Full head alone", "legacy": "B2 + diagnostics head"}
COLORS = {"v2": "#2a78d6", "oeh_final": "#1baf7a", "legacy": "#eb6834"}  # reference palette slots 1, 3, 2 (validated)
MARKERS = {"v2": "o", "oeh_final": "s", "legacy": "^"}


def scores(frame: pd.DataFrame) -> pd.DataFrame:
    p, y = frame[O.P].to_numpy(float), frame.event.to_numpy(int)
    frame = frame.copy()
    frame["brier"] = ((np.cumsum(p, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)
    frame["log_loss"] = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    return frame


def extra_forecasts() -> pd.DataFrame:
    from scripts.analysis import run_oeh_eighth_cohort_20261014 as E
    from scripts.analysis import oeh_generic_baselines as G

    cache = OUT / "extra_predictions.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    query = pd.concat([pd.read_csv(SEALED / p / "query_preoutcome.csv.gz").assign(source_panel=p) for p in PANELS], ignore_index=True)
    spec = O.load_spec()
    train = O.training_panel(spec)
    sets = E.forecast_sets()
    tables = O.fit_heads(train, query.drop(columns="source_panel"), 6, OUT,
                         {"no_h": (sets["oeh_without_oehh_risk"], False), "no_h_finite": (sets["oeh_without_oehh_risk"], True)})
    no_h, d = O.final_law(tables, train, query, spec, full="no_h", full_finite="no_h_finite")
    final, _ = O.final_law(tables, train, query, spec)
    laws = {"oeh_without_horizon": no_h, "knn_analog": G.knn_analog(train, query), "logistic": G.logistic(train, query),
            "_oeh_refit_check": final}
    frames = []
    for method, law in laws.items():
        f = query[["source_panel", *O.KEYS]].copy()
        f["method"], f["distance"] = method, d
        f[O.P] = law
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(cache, index=False)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sealed = pd.read_csv(SEALED / "scored.csv.gz")
    extra = extra_forecasts()
    labels = sealed[sealed.method == "v2"][["source_panel", *O.KEYS, "event"]]
    extra = scores(extra.merge(labels, on=["source_panel", *O.KEYS], validate="many_to_one"))
    check = extra[extra.method == "_oeh_refit_check"].merge(sealed[sealed.method == "v2"], on=["source_panel", *O.KEYS], suffixes=("", "_s"))
    refit_diff = float(np.abs(check[O.P].to_numpy() - check[[f"{p}_s" for p in O.P]].to_numpy()).max())
    rows = pd.concat([sealed[["source_panel", "method", "mdp_id", "policy_family", "algorithm", "distance", "brier", "log_loss"]],
                      extra[extra.method != "_oeh_refit_check"][["source_panel", "method", "mdp_id", "policy_family", "algorithm",
                                                                  "distance", "brier", "log_loss"]]], ignore_index=True)

    # ---- methods table with per-panel intervals relative to the final method
    per = rows.groupby(["source_panel", "method", "mdp_id"])[["brier", "log_loss"]].mean().reset_index()
    keep = ["v2", "oeh_final", "legacy", "oeh_without_horizon", "v2_no_direct_oehb", "knn_analog", "logistic", "b2_matched",
            "xgboost_cf_18", "b2_plus_lce"]
    table = []
    for scope, panels in [(p, (p,)) for p in PANELS] + [("pooled_external", EXTERNAL)]:
        w = per[per.source_panel.isin(panels)].pivot_table(index=["source_panel", "mdp_id"], columns="method", values=["brier", "log_loss"])
        for m in keep:
            row = {"scope": scope, "method": m}
            for metric in ("brier", "log_loss"):
                g = (w[(metric, m)] - w[(metric, "v2")]).to_numpy()
                half = 1.96 * g.std(ddof=1) / np.sqrt(len(g))
                row[metric] = float(w[(metric, m)].mean())
                row[f"{metric}_minus_oeh"], row[f"{metric}_low95"], row[f"{metric}_high95"] = float(g.mean()), float(g.mean() - half), float(g.mean() + half)
            table.append(row)
    pd.DataFrame(table).to_csv(OUT / "methods.csv", index=False)

    # ---- gain over B2 by distance (row level, all four panels pooled)
    w = rows[rows.method.isin(["b2_matched", *NAMES])].pivot_table(
        index=["source_panel", "mdp_id", "policy_family", "algorithm"], columns="method", values=["brier", "log_loss", "distance"]).reset_index()
    d = w[("distance", "v2")].to_numpy()
    w["bin"] = pd.cut(d, EDGES, right=True)
    gain = []
    for b, g in w.groupby("bin", observed=True):
        row = {"bin": str(b), "rows": len(g), "d_median": float(np.median(g[("distance", "v2")])),
               **{f"share_{p}": float((g.source_panel == p).mean()) for p in PANELS}}
        for metric in ("brier", "log_loss"):
            for m in NAMES:
                x = (g[(metric, "b2_matched")] - g[(metric, m)]).to_numpy()
                row[f"{metric}_gain_{m}"] = float(x.mean())
                row[f"{metric}_se_{m}"] = float(x.std(ddof=1) / np.sqrt(len(x)))
        gain.append(row)
    gain = pd.DataFrame(gain)
    gain.to_csv(OUT / "gain_by_distance.csv", index=False)
    figure(gain)
    (OUT / "manifest.json").write_text(json.dumps({
        "status": "exploratory (labels of the four panels are opened)", "refit_reproduces_sealed_oeh_max_abs_diff": refit_diff,
        "sources": {"scored": str((SEALED / "scored.csv.gz").relative_to(ROOT)).replace("\\", "/")}}, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print("refit check", refit_diff)
    print(pd.DataFrame(table)[lambda t: t.scope.isin(["internal", "pooled_external"])].round(4).to_string(index=False))
    print(gain[["bin", "rows", "d_median", "log_loss_gain_v2", "log_loss_gain_oeh_final", "log_loss_gain_legacy"]].round(4).to_string(index=False))


def figure(gain: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    x = gain.d_median.to_numpy()
    for ax, metric, title in zip(axes, ("brier", "log_loss"), ("Integrated Brier", "Log loss")):
        for m, label in NAMES.items():
            y, se = gain[f"{metric}_gain_{m}"].to_numpy(), gain[f"{metric}_se_{m}"].to_numpy()
            ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=COLORS[m], alpha=0.12, linewidth=0)
            ax.plot(x, y, color=COLORS[m], lw=2, marker=MARKERS[m], ms=4.5, label=label)
        ax.axhline(0, color="0.55", lw=0.8)
        ax.axvline(3.0, color="0.55", lw=0.8, ls="--")
        ax.text(3.05, ax.get_ylim()[1], " $d_0$", va="top", fontsize=7, color="0.35")
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks([0.75, 1, 1.5, 2, 3, 5])
        ax.set_xticklabels(["0.75", "1", "1.5", "2", "3", "5"])
        ax.set_xlabel("shift distance $d$ (bin median)", fontsize=8)
        ax.set_ylabel(f"{title} gain over B2", fontsize=8)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="0.92", lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for target in (OUT / "fig_gain_by_distance", ROOT / "paper/figures/fig_gain_by_distance"):
        fig.savefig(target.with_suffix(".pdf"))
        fig.savefig(target.with_suffix(".png"), dpi=200)


if __name__ == "__main__":
    main()

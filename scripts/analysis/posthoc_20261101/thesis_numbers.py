"""Post hoc numbers of the thesis round of 01/11/2026. No head is refitted and no learner is run.

    python scripts/analysis/posthoc_20261101/thesis_numbers.py

Reads the sealed forecasts and recorded labels of the three final tests, checks that it reproduces the numbers already
printed in Tables 11-15, and writes out/thesis_numbers.json with its manifest. Everything here is post hoc: the protocols
list per-learner, Murphy, regret and Shapley summaries as descriptive, and they register no interval used below.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ART = ROOT / "artifacts"
OUT = HERE / "out"
DRAWS, SEED, FLOOR = 10_000, 20261101, 1e-12
P = [f"p_{i}" for i in range(7)]
PARTS = ("T1 internal", "T1 external", "T2", "T3")
BUDGETS = np.array([1, 2, 5, 10, 20, 50])
SOURCES = {
    "t1": ART / "oeh_v2_sealed_test_2026-10-14/scored.csv.gz",
    "t2": ART / "oeh_fifth_panel_2026-10-15/scored.csv.gz",
    "t3": ART / "oeh_panel6_2026-10-15/scored.csv.gz",
}
NAMES = {"v2": "oeh", "oeh_final": "oeh_full_head", "legacy": "robust"}
KEEP = ("b2_matched", "robust", "oeh_full_head", "oeh", "b2_plus_prefix_fqe", "lce_raw", "xgboost_cf_108")


def load():
    t1 = pd.read_csv(SOURCES["t1"])
    t1 = t1[t1.panel == "primary"].replace({"method": NAMES})
    t1["part"] = np.where(t1.source_panel == "internal", "T1 internal", "T1 external")
    t2 = pd.read_csv(SOURCES["t2"]).assign(part="T2")
    t3 = pd.read_csv(SOURCES["t3"]).rename(columns={"ibs": "brier", "ll": "log_loss"}).assign(part="T3")
    d = pd.concat([t1, t2, t3], ignore_index=True)
    d = d[d.method.isin(KEEP)].copy()
    p = d[P].to_numpy()
    y = d.event.to_numpy().astype(int)
    F = np.cumsum(p, 1)[:, :6]
    ind = (y[:, None] <= np.arange(6)[None]).astype(float)
    d["ibs"] = ((F - ind) ** 2).mean(1)
    d["ll"] = -np.log(np.maximum(p[np.arange(len(y)), y], FLOOR))
    d["ibs_ge"] = ((F - ind) ** 2)[:, 3:6].mean(1)
    group = np.minimum(np.maximum(y, 3), 6) - 3
    grouped = np.column_stack((p[:, :4].sum(1), p[:, 4], p[:, 5], p[:, 6]))
    d["ll_ext"] = -np.log(np.maximum(grouped[np.arange(len(y)), group], FLOOR))
    assert np.abs(d.ibs - d.brier).max() < 1e-9 and np.abs(d.ll - d.log_loss).max() < 1e-9
    return d


def mdp_scores(frame, method, col):
    return frame[frame.method == method].groupby("mdp_id")[col].mean()


def mean_score(frame, method, col):
    return float(mdp_scores(frame, method, col).mean())


def contrast(frame, before, after, col, index):
    """Score of `before` minus score of `after` with a percentile whole-MDP interval; positive means `after` is better."""
    a, b = mdp_scores(frame, before, col), mdp_scores(frame, after, col)
    assert a.index.equals(b.index) and len(a) == index.shape[1]
    diff = (a - b).to_numpy()
    boots = diff[index].mean(1)
    return {"diff": float(diff.mean()), "lo": float(np.quantile(boots, .025)), "hi": float(np.quantile(boots, .975))}


def by_method(frame, method, cols=("ibs", "ll")):
    return [mean_score(frame, method, c) for c in cols]


def ladder(frame, index):
    steps = (("pilot_diagnostics", "b2_matched", "robust"), ("simulation_planning", "robust", "oeh_full_head"),
             ("shift_gate", "oeh_full_head", "oeh"))
    res = {"scores": {m: {c: mean_score(frame, m, c) for c in ("ibs", "ll", "ibs_ge", "ll_ext")}
                      for m in ("b2_matched", "robust", "oeh_full_head", "oeh")}}
    for name, a, b in steps:
        res[name] = {c: contrast(frame, a, b, c, index) for c in ("ibs", "ll", "ibs_ge", "ll_ext")}
    return res


def learners(frame):
    res = {}
    for alg, g in frame.groupby("algorithm"):
        o, b = by_method(g, "oeh"), by_method(g, "b2_matched")
        res[alg.split("_")[0]] = {"oeh": o, "b2": b, "gain_ibs": 1 - o[0] / b[0], "gain_ll": 1 - o[1] / b[1]}
    return res


def order(frame, method):
    """Concordance of expected class with the label over learner pairs of one pilot, and the best-learner pick."""
    x = frame[frame.method == method].copy()
    x["exp"] = (x[P].to_numpy() * np.arange(7)).sum(1)
    conc = pairs = picked = differ = 0.0
    fixed = x.groupby("algorithm").event.mean().idxmin()
    fixed_ok = 0
    for _, h in x.groupby(["mdp_id", "policy_family"]):
        ev, ex = h.event.to_numpy(), h.exp.to_numpy()
        for i in range(len(ev)):
            for j in range(i + 1, len(ev)):
                if ev[i] != ev[j]:
                    pairs += 1
                    s = (ex[i] - ex[j]) * (ev[i] - ev[j])
                    conc += 1 if s > 0 else (.5 if s == 0 else 0)
        if len(set(ev.tolist())) > 1:
            differ += 1
            picked += ev[np.argmin(ex)] == ev.min()
            fixed_ok += ev[h.algorithm.to_numpy() == fixed][0] == ev.min()
    return {"pairs": int(pairs), "concordance": conc / pairs, "pilots_with_different_labels": int(differ),
            "best_picked": picked / differ, "fixed": fixed.split("_")[0], "fixed_best_picked": fixed_ok / differ}


def calibration(frame, method):
    x = frame[frame.method == method]
    p, y = x.p_6.to_numpy(), (x.event.to_numpy() == 6).astype(float)
    b = np.minimum((p * 10).astype(int), 9)
    bins = [{"mean_forecast": float(p[b == k].mean()), "observed": float(y[b == k].mean()), "n": int((b == k).sum())}
            for k in range(10) if (b == k).any()]
    ece = float(sum(np.mean(b == k) * abs(p[b == k].mean() - y[b == k].mean()) for k in range(10) if (b == k).any()))
    return {"ece": ece, "bins": bins, "mean_forecast": float(p.mean()), "observed": float(y.mean()),
            "mean_p6_on_finite": float(p[y == 0].mean()), "mean_p6_on_censored": float(p[y == 1].mean())}


def rule_rows(x, kind):
    E = x.event.to_numpy().astype(int)
    finite = E <= 5
    if kind == "registered":
        F = np.cumsum(x[P].to_numpy(), 1)[:, :6]
        ok = F >= .8
        select = ok.any(1)
        j = np.where(select, ok.argmax(1), -1)
        D = np.where(select, np.maximum(10, BUDGETS[np.maximum(j, 0)]), 10).astype(float)
        met, fail = select & (E <= j), select & (E > j)
        extra = D - 10
    elif kind == "cautious":
        collect = x.p_6.to_numpy() < .9
        D = np.where(collect, 50., 10.)
        met, fail = collect & finite, collect & ~finite
        extra = np.where(collect, 50., 0.)
    miss = finite & ~met & ~fail
    return pd.DataFrame({"mdp_id": x.mdp_id.to_numpy(), "D": D, "D_no_reuse": 10 + extra, "met": met.astype(float), "finite": finite.astype(float),
                         "fail": fail.astype(float), "miss": miss.astype(float), "failed_target": (fail | miss).astype(float)})


def rule(frame, method, kind, index):
    r = rule_rows(frame[frame.method == method], kind)
    m = r.groupby("mdp_id").mean()
    cost = m.D + 100 * (m.fail + m.miss)
    lost = (m.finite - m.met).to_numpy()
    boots_lost, boots_fin = lost[index].mean(1), m.finite.to_numpy()[index].mean(1)
    share = np.where(boots_fin > 0, boots_lost / boots_fin, 0.)
    return {"data": float(m.D.mean()), "data_no_reuse": float(m.D_no_reuse.mean()), "share_met": float(m.met.mean() / m.finite.mean()),
            "net_cost": float(cost.mean()), "lost_share": float(lost.mean() / m.finite.mean()),
            "lost_share_lo": float(np.quantile(share, .025)), "lost_share_hi": float(np.quantile(share, .975))}


def cautious_breakdown(frame):
    """Figure 7: mean data per case split by whether some budget can solve the case, against always collecting x50."""
    r = rule_rows(frame[frame.method == "oeh"], "cautious")
    m = r.assign(solvable=r.D * r.finite, unsolvable=r.D * (1 - r.finite), a_solvable=50 * r.finite, a_unsolvable=50 * (1 - r.finite)).groupby("mdp_id").mean()
    return {"solvable": float(m.solvable.mean()), "unsolvable": float(m.unsolvable.mean()), "always_solvable": float(m.a_solvable.mean()), "always_unsolvable": float(m.a_unsolvable.mean()),
            "change_solvable": float(m.solvable.mean() / m.a_solvable.mean() - 1), "change_unsolvable": float(m.unsolvable.mean() / m.a_unsolvable.mean() - 1)}


def always(frame):
    x = frame[frame.method == "oeh"]
    finite = (x.event <= 5).groupby(x.mdp_id).mean().mean()
    return {"reachable": float(finite)}


def horizon_alone(d):
    """Best case for one value per pilot: class frequencies in five quantile bins of h_H, taken in-sample from the test."""
    res = {}
    for part in PARTS[:2]:
        x = d[(d.part == part) & (d.method == "oeh")].copy()
        panels = ["internal"] if part == "T1 internal" else ["colosseum2029", "spibb2", "forest"]
        h = pd.concat([pd.read_csv(ART / f"oeh_v2_sealed_test_2026-10-14/{p}/horizon_preoutcome.csv")[["mdp_id", "policy_family", "data_oeh_h"]] for p in panels])
        x = x.merge(h, on=["mdp_id", "policy_family"], validate="many_to_one")
        v = x.data_oeh_h.replace([np.inf], 1e9).fillna(1e9)
        rho = {a.split("_")[0]: float(pd.Series(v[x.algorithm == a]).rank().corr(x.event[x.algorithm == a].rank())) for a in sorted(x.algorithm.unique())}
        x["bin"] = pd.qcut(v.rank(method="first"), 5, labels=False)
        F_bin, F_const = np.zeros((len(x), 6)), np.zeros((len(x), 6))
        for a, g in x.groupby("algorithm"):
            F_const[x.algorithm.to_numpy() == a] = [(g.event <= j).mean() for j in range(6)]
            for b, gb in g.groupby("bin"):
                F_bin[(x.algorithm.to_numpy() == a) & (x.bin.to_numpy() == b)] = [(gb.event <= j).mean() for j in range(6)]
        ind = (x.event.to_numpy()[:, None] <= np.arange(6)[None]).astype(float)
        score = lambda F: float(pd.Series(((F - ind) ** 2).mean(1)).groupby(x.mdp_id.to_numpy()).mean().mean())
        res[part] = {"spearman": rho, "ibs_constant": score(F_const), "ibs_five_bins": score(F_bin), "ibs_oeh": float(mean_score(d[d.part == part], "oeh", "ibs"))}
    return res


def regret_sensitivity():
    r = pd.read_csv(ART / "oeh_v2_sealed_test_2026-10-14/decision_regret.csv")
    w = r[r.method.isin(["v2", "b2_matched"])].pivot_table(index=["panel", "tau", "lambda"], columns="method", values="mean_regret")
    return {"settings": int(len(w)), "oeh_lower": int((w.v2 < w.b2_matched).sum()), "worst": w.assign(d=w.b2_matched - w.v2).reset_index().sort_values("d").head(2).to_dict("records")}


def statuses():
    from collections import Counter
    n = Counter()
    for f in ART.glob("oeh_v2_sealed_test_2026-10-14/*/horizon_preoutcome.csv"):
        n.update(pd.read_csv(f).data_oeh_h_status.tolist())
    for folder in ("oeh_fifth_panel_2026-10-15", "oeh_panel6_2026-10-15"):
        for f in (ART / folder).rglob("horizon.csv"):
            n.update(pd.read_csv(f).data_oeh_h_status.tolist())
    return dict(n)


def history():
    """Scores of the five sealed 120-MDP tests on their 84 stochastic MDPs, read from each test's own summary."""
    tests = (("Cohort 3", "oeh_difficulty_unseen_120_2026-10-01", "oeh_three_way", "b2_three_way"),
             ("Cohort 4", "oeh_pooled_successor_unseen_120_2026-10-01", "oeh_three_way", "b2_three_way"),
             ("Cohort 5", "oeh_cross_capacity_unseen_120_2026-10-01", "oeh_cross_capacity", "b2_three_way"),
             ("Cohort 6", "oeh_final_unseen_120_2026-10-02", "oeh_final", "b2_matched"),
             ("Cohort 7", "oeh_confirmatory_unseen_120_2026-10-04", "oeh_final", "b2_matched"))
    res = {}
    for name, folder, oeh, b2 in tests:
        s = pd.read_csv(ART / folder / "summary.csv").query("panel == 'primary'").set_index("method")
        res[name] = {"oeh": [float(s.loc[oeh, "integrated_brier"]), float(s.loc[oeh, "seven_class_log_loss"])],
                     "b2": [float(s.loc[b2, "integrated_brier"]), float(s.loc[b2, "seven_class_log_loss"])], "mdps": int(s.loc[oeh, "mdps"])}
    return res


def saturation():
    """The first scalar measure on its development data, from the run of 23/08/2026 kept under the main checkout."""
    folder = ROOT.parent / "Devoted/tmp/oeh_development_2026-09-30/artifacts/week2/oeh_validation_2026-08-23"
    c = pd.read_csv(folder / "evidence_curves.csv")
    k = pd.read_csv(folder / "k_dagger_estimates.csv")
    keys = ["mdp_id", "policy_family", "pilot_multiplier"]
    return {"points": int(len(c)), "points_at_floor": int((c.multiplier == 1).sum()), "max_loss_at_one": float(c.loss_at_one.max()),
            "mdps": int(c.mdp_id.nunique()), "curves": int(c.groupby(keys).ngroups), "pilots": int(len(k)), "pilots_with_depth": int(k.k_dagger_oeh.notna().sum()),
            "pilots_depth_zero": int((k.k_dagger_oeh == 0).sum()), "files": [str(folder / "evidence_curves.csv"), str(folder / "k_dagger_estimates.csv")]}


def worked_case(d):
    mdp, beh = "v2t_bottleneck_004", "shifted_dynamics"
    t1 = ART / "oeh_v2_sealed_test_2026-10-14/internal"
    sim = None
    for f in sorted((t1 / "oeh_b_groups").glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        if doc["mdp_id"] == mdp and doc["policy_family"] == beh:
            sim = doc
            break
    h = pd.read_csv(t1 / "horizon_preoutcome.csv").query("mdp_id == @mdp and policy_family == @beh").iloc[0]
    c = pd.read_csv(t1 / "curve_preoutcome.csv").query("mdp_id == @mdp and policy_family == @beh").iloc[0]
    x = d[(d.mdp_id == mdp) & (d.policy_family == beh)]
    res = {"distance": float(x.distance.iloc[0]), "h": float(h.data_oeh_h), "requirement": float(h.data_oeh_h_requirement), "depth": int(h.data_oeh_h_depth),
           "sim_draws": sim["sampling"], "sim": {}, "held_out_x10": {}, "forecast": {}}
    for alg, v in sim["learners"].items():
        short = alg.split("_")[0]
        res["sim"][short] = {"p_censored": v["probabilities"][6], "return_x50": v["mean_normalized_return"][5]}
        res["held_out_x10"][short] = float(c[f"{alg}|curve_mean_x10"])
        for m in ("oeh", "oeh_full_head", "robust", "b2_matched"):
            row = x[(x.method == m) & (x.algorithm == alg)].iloc[0]
            res["forecast"].setdefault(short, {})[m] = {"p": [float(row[k]) for k in P], "label": int(row.event)}
    return res


def main():
    d = load()
    rng = np.random.default_rng(SEED)
    parts = {p: d[d.part == p] for p in PARTS}
    index = {p: rng.integers(0, f.mdp_id.nunique(), (DRAWS, f.mdp_id.nunique())) for p, f in parts.items()}
    res = {"status": "post hoc", "draws": DRAWS, "seed": SEED, "parts": {}}
    for p, f in parts.items():
        res["parts"][p] = {
            "mdps": int(f.mdp_id.nunique()), "cases": int(len(f[f.method == "oeh"])),
            "ladder": ladder(f, index[p]),
            "oeh_vs_b2": {c: contrast(f, "b2_matched", "oeh", c, index[p]) for c in ("ibs", "ll", "ibs_ge", "ll_ext")},
            "learners": learners(f),
            "order": {m: order(f, m) for m in ("oeh", "b2_matched")},
            "calibration": {m: calibration(f, m) for m in ("oeh", "b2_matched")},
            "rules": {f"{m}_{k}": rule(f, m, k, index[p]) for m in ("oeh", "b2_matched") for k in ("registered", "cautious")},
            "always": always(f),
            "cautious_breakdown": cautious_breakdown(f),
            "labels": {a.split("_")[0]: [float((g.event == c).mean()) for c in range(7)] for a, g in f[f.method == "oeh"].groupby("algorithm")},
        }
    full = {}
    for (part, source), f in d.groupby(["part", "source_panel"]):
        full[source] = by_method(f, "oeh_full_head") + by_method(f, "oeh")
    for part, f in parts.items():
        full[part] = by_method(f, "oeh_full_head") + by_method(f, "oeh")
    full["T3 extrapolation"] = by_method(parts["T3"], "oeh_full_head", ("ibs_ge", "ll_ext")) + by_method(parts["T3"], "oeh", ("ibs_ge", "ll_ext"))
    res["full_head_table_rows"] = full
    u = d[d.method == "oeh"]
    res["labels_all"] = {"cases": int(len(u)), "up_to_x10": float((u.event <= 3).mean()), "x20_x50": float(u.event.isin([4, 5]).mean()), "censored": float((u.event == 6).mean())}
    res["labels_t1_by_learner"] = {a.split("_")[0]: float((g.event == 6).mean()) for a, g in u[u.part.str.startswith("T1")].groupby("algorithm")}
    res["horizon_alone"] = horizon_alone(d)
    res["regret_sensitivity_test1"] = regret_sensitivity()
    res["planning_risk_status"] = statuses()
    res["worked_case"] = worked_case(d)
    res["saturation"] = saturation()
    col = rule_rows(parts["T1 external"][(parts["T1 external"].method == "oeh") & (parts["T1 external"].source_panel == "colosseum2029")], "cautious")
    collected = col.D == 50
    res["colosseum_cautious_counts"] = {"correct_skips": int((~collected & (col.finite == 0)).sum()), "correct_collections": int((collected & (col.finite == 1)).sum()),
                                        "failed_collections": int((collected & (col.finite == 0)).sum()), "missed_targets": int((~collected & (col.finite == 1)).sum())}
    assert res["colosseum_cautious_counts"] == {"correct_skips": 543, "correct_collections": 275, "failed_collections": 245, "missed_targets": 17}, res["colosseum_cautious_counts"]
    for key in ("T1 internal", "T1 external", "T3"):
        b = res["parts"][key]["cautious_breakdown"]
        assert abs(b["change_solvable"]) <= .0425 and .615 <= -b["change_unsolvable"] <= .685, (key, b)
    res["history"] = history()
    res["shapley_external"] = {m: dict(zip(("simulation", "planning_risk", "pilot_diagnostics"), (float(r.P), float(r.H), float(r.L))))
                               for m, r in pd.read_csv(ART / "oeh_v2_sealed_test_2026-10-14/shapley.csv").query("scope == 'pooled_external'").set_index("metric").iterrows()}

    # the numbers already printed in the thesis must come back
    printed = {"T1 internal": (.0476, .562, .0565, .613, .0497, .579), "T1 external": (.0662, .599, .1063, .782, .0832, .659),
               "T2": (.0486, .417, .0540, .475, .0468, .428), "T3": (.0971, 1.039, .1533, 1.297, .1140, 1.122)}
    for p, v in printed.items():
        s = res["parts"][p]["ladder"]["scores"]
        got = (s["oeh"]["ibs"], s["oeh"]["ll"], s["b2_matched"]["ibs"], s["b2_matched"]["ll"], s["robust"]["ibs"], s["robust"]["ll"])
        assert all(abs(a - b) < (6e-4 if i % 2 else 6e-5) for i, (a, b) in enumerate(zip(got, v))), (p, got, v)
    table15 = {"T1 internal": ((14.0, 59, 27.1), (27.5, 98, 44.4), (14.0, 46, 30.6), (29.7, 99, 51.6)),
               "T1 external": ((17.4, 76, 28.3), (28.2, 98, 43.1), (15.8, 54, 34.0), (31.2, 98, 53.9)),
               "T2": ((10.4, 7, 20.7), (18.0, 63, 35.4), (11.4, 18, 22.1), (23.9, 79, 52.4)),
               "T3": ((15.3, 62, 32.8), (30.2, 95, 40.7), (13.4, 48, 41.1), (39.2, 95, 71.8))}
    keys = ("oeh_registered", "oeh_cautious", "b2_matched_registered", "b2_matched_cautious")
    for p, rows in table15.items():
        for k, (data, share, cost) in zip(keys, rows):
            r = res["parts"][p]["rules"][k]
            assert abs(r["data"] - data) < .051 and abs(100 * r["share_met"] - share) < .51 and abs(r["net_cost"] - cost) < .051, (p, k, r, data, share, cost)
    print("OK printed scores of Tables 11-13 and all of Table 15 reproduced")

    OUT.mkdir(exist_ok=True)
    target = OUT / "thesis_numbers.json"
    target.write_text(json.dumps(res, indent=1), encoding="utf-8")
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    inputs = [Path(f) for f in res["saturation"].pop("files")] + list(SOURCES.values()) + [ART / "oeh_v2_sealed_test_2026-10-14/decision_regret.csv"]
    inputs += [ART / "oeh_v2_sealed_test_2026-10-14/shapley.csv"]
    inputs += [ART / f / "summary.csv" for f in ("oeh_difficulty_unseen_120_2026-10-01", "oeh_pooled_successor_unseen_120_2026-10-01", "oeh_cross_capacity_unseen_120_2026-10-01", "oeh_final_unseen_120_2026-10-02", "oeh_confirmatory_unseen_120_2026-10-04")]
    manifest = {"status": "post hoc, descriptive; run after every label was opened", "created_utc": pd.Timestamp.now("UTC").isoformat(),
                "bootstrap": {"draws": DRAWS, "seed": SEED, "kind": "percentile, whole MDPs, one index matrix per part"},
                "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
                "inputs_sha256": {str(f.relative_to(ROOT.parent)).replace("\\", "/"): sha(f) for f in inputs},
                "source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"): sha(__file__)},
                "outputs_sha256": {target.name: sha(target)}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("written", target)


if __name__ == "__main__":
    main()

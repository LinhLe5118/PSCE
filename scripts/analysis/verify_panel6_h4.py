"""H4 for panel 6: independent re-scoring on a second machine.

Run from the root of a fresh clone of Devoted (the analyze step rewrites result.json
and scored.csv.gz, so do not run it in the working repository):

    git clone --no-hardlinks <repo> h4_clone && cd h4_clone
    python -m scripts.analysis.run_oeh_panel6_20261015 analyze
    python <this file>

Checks: (1) every analysis output equals the committed one; (2) labels recomputed
from seed outcomes with code independent of the team's; (3) seals; (4) full-endpoint
IBS/LL recomputed with independent code against summary.csv.
"""
import hashlib
import io
import json
import subprocess

import numpy as np
import pandas as pd

A = "artifacts/oeh_panel6_2026-10-15/"
K = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]


def committed(path):
    return subprocess.run(["git", "show", "HEAD:" + path], capture_output=True, check=True).stdout


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def maxdiff(a, b):
    if isinstance(a, dict):
        return max([maxdiff(a[k], b[k]) for k in a if k != "completed_at_utc"] + [0])
    if isinstance(a, list):
        return max([maxdiff(x, y) for x, y in zip(a, b)] + [0])
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return abs(a - b)
    return 0 if a == b else float("inf")


new, old = json.load(open(A + "result.json")), json.loads(committed(A + "result.json"))
print("result.json max abs diff (timestamp excluded):", maxdiff(new, old))
for gate, v in new["gates"].items():
    print(" ", gate, v.get("verdict"))
s_old = pd.read_csv(io.BytesIO(committed(A + "scored.csv.gz")), compression="gzip")
print("scored.csv.gz content identical:", s_old.equals(pd.read_csv(A + "scored.csv.gz")))
for f in ["summary.csv", "by_learner.csv", "murphy.csv", "gain_by_distance.csv", "label_variant_tau099.csv"]:
    a, b = pd.read_csv(A + f), pd.read_csv(io.BytesIO(committed(A + f)))
    num = a.select_dtypes("number").columns
    print(f, "max abs diff", float((a[num] - b[num]).abs().max().max()), "text equal", a.drop(columns=num).equals(b.drop(columns=num)))

print("protocol sha256 matches protocol.sha256:", sha(A + "protocol.json") == open(A + "protocol.sha256").read().split()[0])
frames = []
for panel in ["mdpp", "labmaze"]:
    d = A + panel + "/"
    seal, run = json.load(open(d + "preoutcome_seal.json")), json.load(open(d + "outcome_run.json"))
    print(panel, "| outcome run points to seal:", run["preoutcome_seal_sha256"] == sha(d + "preoutcome_seal.json"),
          "| predictions hash in seal:", seal["artifact_sha256"]["predictions_preoutcome.csv"] == sha(d + "predictions_preoutcome.csv"),
          "| outcomes observed at seal:", seal["learner_outcomes_observed"])
    o, lab = pd.read_csv(d + "outcomes.csv.gz"), pd.read_csv(d + "labels.csv")
    succ = o.normalized_return >= 0.95
    c = o.assign(s=succ).groupby(K + ["budget"]).s.sum().unstack("budget")

    def event(row, budgets):
        v = row[budgets].to_numpy()
        return next((k for k in range(len(v)) if (v[k:] >= 8).all()), len(v))

    prim = [1, 2, 5, 10, 20, 50]
    mine = pd.DataFrame({"e": c.apply(lambda r: event(r, prim), axis=1),
                         "ex": c.apply(lambda r: event(r, prim + [100, 200]), axis=1)}).reset_index()
    m = lab.merge(mine, on=K, validate="one_to_one")
    print(f"  success flag = NR>=0.95: {(succ == o.success).all()}; jobs {len(o)}; 10 seeds everywhere: "
          f"{(o.groupby(K + ['budget']).size() == 10).all()}; labels match: {(m.event == m.e).all()}; "
          f"extended labels match: {(m.event_extended == m.ex).all()}")
    m = pd.read_csv(d + "predictions_preoutcome.csv").merge(lab[K + ["event"]], on=K)
    p, y = m[P].to_numpy(float), m.event.to_numpy(int)
    m["ibs"] = ((np.cumsum(p, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)
    m["ll"] = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    frames.append(m.assign(panel=panel))
w = pd.concat(frames).groupby(["method", "panel", "mdp_id"])[["ibs", "ll"]].mean().groupby("method").mean()
summ = pd.read_csv(A + "summary.csv").query("scope == 'pooled'").set_index("method")[["ibs", "ll"]]
j = w.join(summ, rsuffix="_committed", how="inner")
print("independent full-endpoint IBS/LL vs summary.csv, max abs diff:",
      float(((j.ibs - j.ibs_committed).abs() + (j.ll - j.ll_committed).abs()).max()))
print(j.loc[["oeh", "oeh_full_head", "robust", "b2_plus_prefix_fqe", "b2_matched"]].round(4).to_string())

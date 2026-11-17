"""OEH v2, step 0 of M2: does the reward prior explain the external failures?

Re-runs OEH-B on opened panels with one structural change, ``reward_prior`` from
``unseen_zero`` (sealed A2) to ``exchangeable``, keeping the transition base,
behavior prior, draw count **and the sealed per-group seed**, so differences are
the prior's, not Monte Carlo noise. Then compares, per panel, family and learner:
the raw OEH-B law's log loss and Brier against the labels, and the calibration of
OEH-B's per-budget success probability against the observed share of the ten
training seeds that succeed. Opened data only: a diagnosis, not a test.

Phases: ``run`` (compute the alternative groups) and ``score``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "artifacts"
OUTPUT = A / "week8/oeh_v2_development_2026-10-14/prior_diagnosis"
ALT = {"reward_prior": "exchangeable", "transition_base": "uniform", "behavior_prior": "mixture"}
BUDGETS = (1, 2, 5, 10, 20, 50)
PANELS = {
    "cohort8": dict(dataset=ROOT / "datasets/oeh_eighth_cohort_2026-10-14", groups=A / "oeh_eighth_cohort_2026-10-14/internal/oeh_b_groups",
                    labels=A / "oeh_eighth_cohort_2026-10-14/internal/labels.csv", outcomes=A / "oeh_eighth_cohort_2026-10-14/internal/outcomes.csv.gz"),
    "colosseum2027": dict(dataset=Path("D:/Tools/colosseum_work/trial/data/confirm_bundle"), groups=A / "week8/colosseum_confirm_2026-10-14/oeh_b_groups",
                          labels=A / "week8/colosseum_confirm_2026-10-14/colosseum_labels.csv", outcomes=A / "week8/colosseum_confirm_2026-10-14/colosseum_outcomes.csv"),
    "colosseum2028": dict(dataset=Path("D:/Tools/colosseum_work/trial/data/c8tier_bundle"), groups=A / "oeh_eighth_cohort_2026-10-14/colosseum/oeh_b_groups",
                          labels=A / "oeh_eighth_cohort_2026-10-14/colosseum/labels.csv", outcomes=A / "oeh_eighth_cohort_2026-10-14/colosseum/outcomes.csv.gz"),
    "spibb": dict(dataset=ROOT / "datasets/oeh_spibb_panel_2026-10-14", groups=A / "oeh_spibb_panel_2026-10-14/spibb/oeh_b_groups",
                  labels=A / "oeh_spibb_panel_2026-10-14/spibb/labels.csv", outcomes=A / "oeh_spibb_panel_2026-10-14/spibb/outcomes.csv.gz"),
}


def run(workers: int, panels: list[str]) -> None:
    from scripts.analysis import run_oeh_final_120 as R

    tasks = []
    for name in panels:
        spec = PANELS[name]
        target = OUTPUT / name
        target.mkdir(parents=True, exist_ok=True)
        rows = {(r["instance_id"], r["policy_family"]): r for r in R.pilot_rows(spec["dataset"])}
        for path in sorted(spec["groups"].glob("*.json")):
            sealed = json.loads(path.read_text(encoding="utf-8"))
            row = rows[(sealed["mdp_id"], sealed["policy_family"])]
            if row["sha256"] != sealed["pilot_sha256"]:
                raise RuntimeError(f"pilot differs from the sealed group: {path.name}")
            out = target / path.name
            if out.exists():
                continue
            out.with_suffix(".claim").unlink(missing_ok=True)
            tasks.append({"row": row, "path": str(out), "dataset": str(spec["dataset"]), "structure": ALT,
                          "draws": sealed["draws"], "seed": sealed["seed"], "backend": "cpu"})
    print(f"prior diagnosis: {len(tasks)} groups", flush=True)
    R._repo_first()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, _ in enumerate(pool.map(R.oeh_b_group, tasks, chunksize=2), 1):
            if done % 100 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}", flush=True)


def _groups(folder: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(folder.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        for learner, v in d["learners"].items():
            rows.append({"mdp_id": d["mdp_id"], "policy_family": d["policy_family"], "algorithm": learner,
                         "draws_returned": d["sampling"]["draws_returned"],
                         **{f"p_{i}": p for i, p in enumerate(v["probabilities"])},
                         **{f"s_{b}": (v["success_rate"][j] if v["success_rate"] else 0.0) for j, b in enumerate(BUDGETS)}})
    return pd.DataFrame(rows)


def score() -> None:
    keys = ["mdp_id", "policy_family", "algorithm"]
    P = [f"p_{i}" for i in range(7)]
    S = [f"s_{b}" for b in BUDGETS]
    law_rows, cal_rows = [], []
    for name, spec in PANELS.items():
        labels = pd.read_csv(spec["labels"])[keys + ["event"]]
        out = pd.read_csv(spec["outcomes"], usecols=keys + ["budget", "success"])
        observed = out.groupby(keys + ["budget"]).success.mean().unstack("budget")[list(BUDGETS)]
        observed.columns = [f"o_{b}" for b in BUDGETS]
        if not any((OUTPUT / name).glob("*.json")):
            continue
        alt_keys = _groups(OUTPUT / name)[["mdp_id", "policy_family"]].drop_duplicates()  # paired subset
        for arm, folder in (("A2_unseen_zero", spec["groups"]), ("exchangeable", OUTPUT / name)):
            g = _groups(folder).merge(alt_keys, on=["mdp_id", "policy_family"])
            g = g.merge(labels, on=keys).merge(observed.reset_index(), on=keys)
            g["family"] = g.mdp_id.str.replace(r"^(c8_colosseum_|confirm_colosseum_|c8_|spibb_r\d_)", "", regex=True).str.replace(r"_\d+$", "", regex=True)
            if name == "spibb":
                g["family"] = "spibb"
            p, y = g[P].to_numpy(float), g.event.to_numpy(int)
            g["ll"] = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
            g["ibs"] = ((np.cumsum(p, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)
            gap = g[S].to_numpy() - g[[f"o_{b}" for b in BUDGETS]].to_numpy()
            g["success_bias"] = gap.mean(1)
            g["success_abs_error"] = np.abs(gap).mean(1)
            for (fam, learner), h in g.groupby(["family", "algorithm"]):
                law_rows.append({"panel": name, "arm": arm, "family": fam, "algorithm": learner, "rows": len(h),
                                 "zero_draw_share": float((h.draws_returned == 0).mean()),
                                 "raw_ll": h.groupby("mdp_id").ll.mean().mean(), "raw_ibs": h.groupby("mdp_id").ibs.mean().mean(),
                                 "success_bias": h.success_bias.mean(), "success_abs_error": h.success_abs_error.mean()})
            cal_rows.append({"panel": name, "arm": arm, "raw_ll": g.groupby("mdp_id").ll.mean().mean(),
                             "raw_ibs": g.groupby("mdp_id").ibs.mean().mean(), "success_bias": g.success_bias.mean(),
                             "success_abs_error": g.success_abs_error.mean(), "zero_draw_share": float((g.draws_returned == 0).mean())})
    pd.DataFrame(law_rows).to_csv(OUTPUT / "by_family_learner.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(OUTPUT / "by_panel.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(cal_rows).round(4).to_string(index=False))
    t = pd.DataFrame(law_rows)
    print(t.pivot_table(index=["panel", "family"], columns="arm", values=["raw_ll", "success_bias"]).round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "score"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--panels", nargs="*", default=list(PANELS))
    args = parser.parse_args()
    run(args.workers, args.panels) if args.phase == "run" else score()


if __name__ == "__main__":
    main()

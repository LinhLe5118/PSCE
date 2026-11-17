"""Registered leave-one-cohort-out development gate for one OEH arm.

Each of the three opened 120-MDP cohorts is held out in turn; every fitted
method is trained on the other five opened cohorts by the same matched
procedure. The pooled 252 stochastic MDPs carry the registered rule: >=5%
relative gain in integrated Brier and seven-class log loss versus every
comparator, all simultaneous one-sided 95% MDP-cluster lower bounds > 0, and a
positive point gain versus every comparator inside each held-out cohort.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from scripts.analysis import run_oeh_final_120 as R
from scripts.analysis.develop_oeh_difficulty import score_probabilities
from scripts.analysis.oeh_final_analysis import simultaneous_lower

HELD_OUT = ("first120", "second120", "third120")
METRICS = ("integrated_brier", "seven_class_log_loss")


def evaluate(tag: str, variant: str = "base", suffix: str = "", law: bool = False) -> dict:
    R.use_candidate(variant, law)
    out = R.DEVELOPMENT / ((f"gate_{tag}" if variant == "base" else f"gate_{tag}_{variant}") + suffix)
    out.mkdir(exist_ok=True)
    frames = []
    for held in HELD_OUT:
        train = R.meta_training(tuple(c for c in R.OPENED if c != held), tag)
        query = R.opened_frame(held, tag)
        predictions, tuning = R.all_forecasts(train, query, R.raw_oeh_b(R.DEVELOPMENT / "groups" / held / tag))
        (out / f"tuning_{held}.json").write_text(json.dumps(tuning, indent=1), encoding="utf-8")
        scored = predictions.merge(query[R.KEYS + ["event"]], on=R.KEYS, validate="many_to_one")
        scored["held_out"] = held
        frames.append(scored)
        print(f"gate {tag}: scored {held}", flush=True)
    scored = pd.concat(frames, ignore_index=True)
    brier, logloss = score_probabilities(scored[R.P].to_numpy(float), scored.event.to_numpy(int))
    scored["integrated_brier"], scored["seven_class_log_loss"] = brier, logloss
    scored.to_csv(out / "scored.csv", index=False)
    mdp = scored.groupby(["held_out", "mdp_id", "panel", "method"], as_index=False)[list(METRICS)].mean()
    summary = mdp.groupby(["panel", "method"], as_index=False)[list(METRICS)].mean()
    per_cohort = mdp.groupby(["held_out", "panel", "method"], as_index=False)[list(METRICS)].mean()
    summary.to_csv(out / "summary.csv", index=False)
    per_cohort.to_csv(out / "summary_by_cohort.csv", index=False)
    primary = mdp[mdp.panel == "primary"].pivot_table(index="mdp_id", columns="method", values=list(METRICS))
    rows, effects = [], []
    for baseline in R.COMPARATORS:
        for metric in METRICS:
            before, after = primary[(metric, baseline)], primary[(metric, R.CANDIDATE)]
            effects.append((before - after).to_numpy())
            cohort_gain = {
                held: float(per_cohort[(per_cohort.held_out == held) & (per_cohort.panel == "primary")
                                       & (per_cohort.method == baseline)][metric].iloc[0]
                            - per_cohort[(per_cohort.held_out == held) & (per_cohort.panel == "primary")
                                         & (per_cohort.method == R.CANDIDATE)][metric].iloc[0])
                for held in HELD_OUT}
            rows.append({"baseline": baseline, "metric": metric, "baseline_mean": float(before.mean()),
                         "candidate_mean": float(after.mean()),
                         "relative_gain": float(1 - after.mean() / before.mean()),
                         "mean_absolute_improvement": float((before - after).mean()),
                         **{f"gain_{held}": value for held, value in cohort_gain.items()}})
    lower, critical = simultaneous_lower(np.column_stack(effects), 10000, 20261007)
    for row, bound in zip(rows, lower):
        row["simultaneous_one_sided_lower_95"] = float(bound)
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(out / "contrasts.csv", index=False)
    passed = bool(all(row["relative_gain"] >= 0.05 and row["simultaneous_one_sided_lower_95"] > 0
                      and all(row[f"gain_{held}"] > 0 for held in HELD_OUT) for row in rows))
    result = {"tag": tag, "variant": variant + suffix, "mdps": int(len(primary)), "critical": critical, "passed": passed,
              "weakest_relative_gain": float(contrasts.relative_gain.min()),
              "weakest_lower_bound": float(contrasts.simultaneous_one_sided_lower_95.min())}
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    with open(R.DEVELOPMENT / "registry.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"record": "gate_scored", **result}) + "\n")
    print(contrasts[["baseline", "metric", "relative_gain", "simultaneous_one_sided_lower_95",
                     *[f"gain_{h}" for h in HELD_OUT]]].round(4).to_string(index=False))
    print(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--variant", default="base", choices=sorted(R.CANDIDATE_VARIANTS))
    parser.add_argument("--select-capacity", action="store_true")
    parser.add_argument("--law", action="store_true")
    args = parser.parse_args()
    if args.select_capacity:
        R.F.SELECT_CAPACITY = True
    suffix = ("_capacity" if args.select_capacity else "") + ("_law" if args.law else "")
    evaluate(args.tag, args.variant, suffix=suffix, law=args.law)


if __name__ == "__main__":
    main()

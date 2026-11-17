"""Post-hoc part 2 on the opened 72-MDP Colosseum panel (14/10/2026).

The confirmatory bundle was regenerated from the authors' sampler on the main
machine (``colosseum_export_fast.py`` + ``build_colosseum.py``); all 360 pilot
SHA-256 values equal the ones recorded in the sealed OEH-B groups. That makes
two missing inputs computable:

* the authors' TQ/SACo (ml-jku/OfflineRL @ 8cc8eec), completing the 22-contrast
  family of amendment 1 (``b2_plus_tq_saco_original``, ``published_metrics_only``);
* posterior prefix curves, for the LCE comparators fixed by the cohort-8
  development (``lce_raw``, ``b2_plus_lce``), next to ``b2_plus_cb``, ``cb_only``,
  OEH-S and OEH-2T.

Phases: ``freeze`` (plan), ``forecast`` (inputs and forecasts under the outcome
guard of ``colosseum_posthoc_20261014``), ``score``. Labels were opened before
this file existed: every number is post hoc.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from scripts.analysis import colosseum_posthoc_20261014 as H

ROOT = H.ROOT
CONFIRM = H.CONFIRM
OUTPUT = ROOT / "artifacts/week8/colosseum_posthoc_part2_2026-10-14"
BUNDLE = Path(os.environ.get("COLOSSEUM_CONFIRM_BUNDLE", "D:/Tools/colosseum_work/trial/data/confirm_bundle"))
KERNELS = Path(os.environ.get("COLOSSEUM_CONFIRM_KERNELS", "D:/Tools/colosseum_work/trial/data/confirm"))
DEVELOPMENT = ROOT / "artifacts/week8/cohort8_development_2026-10-14"
KEYS, P = H.KEYS, H.P
NEW_METHODS = ("lce_raw", "b2_plus_lce", "b2_plus_cb", "cb_only", "oeh_s", "oeh_2t")


def regeneration_audit() -> dict:
    manifest = H.json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    pilots = {(r["instance_id"], r["policy_family"]): r["sha256"] for r in manifest["datasets"]
              if r.get("stream") == "pilot" and int(r["size_multiplier"]) == 10}
    groups = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((CONFIRM / "oeh_b_groups").glob("*.json"))]
    match = sum(pilots.get((g["mdp_id"], g["policy_family"])) == g["pilot_sha256"] for g in groups)
    return {"groups": len(groups), "pilot_sha256_match": int(match), "bundle_manifest_sha256": H.sha256(BUNDLE / "manifest.json"),
            "kernels": len(list(KERNELS.glob("*.npz")))}


def run_freeze(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    audit = regeneration_audit()
    if audit["pilot_sha256_match"] != 360:
        raise RuntimeError(f"Regenerated bundle differs from the sealed pilots: {audit}")
    H.write_json(output / "analysis_plan.json", {
        "frozen_at_utc": H.now(), "post_hoc": True, "regeneration_audit": audit,
        "analyses": {
            "amendment1_complete": "R.all_forecasts with the TQ/SACo original inputs; 22 contrasts (11 comparators x 2 "
                                   "metrics), team max-t rule (fixed se), 10,000 draws, seed 20261004; tiers as in cohort 7",
            "cohort8_methods": "lce_raw, b2_plus_lce, b2_plus_cb, cb_only (matched head or none), oeh_s and oeh_2t with "
                               "the parameters of cohort8_development selection.json; paired MDP deltas vs OEH, "
                               "10,000-draw percentile intervals, seed 20261014 (descriptive)"},
        "selection_sha256": H.sha256(DEVELOPMENT / "selection.json")})


def _tq_task(task):
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    import environments.benchmark_environments as B
    from environments.tabular_environments import TabularMDP
    from scripts.analysis.run_oeh_eighth_cohort_20261014 import colosseum_mdp
    B.make_benchmark_mdp = lambda spec: colosseum_mdp(str(KERNELS), spec.instance_id, TabularMDP)
    from scripts.analysis import oeh_final_tqsaco_original as T
    return T._task(task)


def tqsaco(out: Path, workers: int) -> pd.DataFrame:
    if out.exists():
        return pd.read_csv(out)
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    by_mdp = {}
    for row in sorted((r for r in manifest["datasets"] if r.get("stream") == "pilot" and int(r["size_multiplier"]) == 10),
                      key=lambda r: (r["instance_id"], r["policy_family"])):
        by_mdp.setdefault(row["instance_id"], []).append(row)
    tasks = [{"instance": instances[m], "rows": r, "dataset": str(BUNDLE)} for m, r in by_mdp.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frame = pd.DataFrame([item for chunk in pool.map(_tq_task, tasks) for item in chunk])
    frame.to_csv(out, index=False)
    return frame


def run_forecast(output: Path, workers: int) -> None:
    from scripts.analysis import run_oeh_final_120 as R
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis import run_oeh_confirmatory_20261004 as C
    from scripts.analysis.develop_cohort8_candidates_20261014 import add_fields, field_sets, lce_law, meta_panel, soften

    if (output / "prediction_seal.json").exists():
        raise FileExistsError("Part-2 predictions already sealed")
    plan = json.loads((output / "analysis_plan.json").read_text(encoding="utf-8"))
    seven = json.loads((H.COHORT7 / "protocol.json").read_text(encoding="utf-8"))["oeh_final"]
    tick = time.perf_counter()
    with H.outcome_guard() as opened:
        R._repo_first()
        curve = R.run_curve(BUNDLE, output / "curve_preoutcome.csv", seven["structure"], seven["requirement_draws"], workers)
        tq = tqsaco(output / "tqsaco_original_preoutcome.csv", workers)
        print(f"inputs {time.perf_counter() - tick:.0f}s", flush=True)
        C.configure()
        R.use_candidate(seven["candidate_variant"], seven["candidate_law"])
        query = H.colosseum_query()
        query = query.merge(pd.read_csv(CONFIRM / "oracle_descriptors_preoutcome.csv"), on="mdp_id", validate="many_to_one")
        query = query.merge(pd.read_csv(CONFIRM / "published_preoutcome.csv"), on=["mdp_id", "policy_family"], validate="many_to_one")
        query = query.merge(tq[["mdp_id", "policy_family", "tq_original", "saco_original"]], on=["mdp_id", "policy_family"],
                            validate="many_to_one")
        query = query.merge(R.curve_features(curve), on=KEYS, validate="one_to_one")
        query = add_fields(query).sort_values(KEYS).reset_index(drop=True)
        train = meta_panel(sort=False)
        pending = [name for name, fields in R.FIELD_SETS.items() if not set(fields) <= set(query.columns)]
        if pending:
            raise RuntimeError(f"Comparator inputs still missing: {pending}")
        registered, _ = R.all_forecasts(train, query, R.raw_oeh_b(CONFIRM / "oeh_b_groups"))
        sealed = pd.read_csv(CONFIRM / "colosseum_predictions_preoutcome.csv")
        check = {}
        for method in ("oeh_final", "b2_matched"):
            j = registered[registered.method == method].merge(sealed[sealed.method == method], on=KEYS,
                                                               suffixes=("", "_s"), validate="one_to_one")
            check[method] = float(np.abs(j[P].to_numpy() - j[[f"{p}_s" for p in P]].to_numpy()).max())
        if max(check.values()) > 1e-6:
            raise RuntimeError(f"all_forecasts does not reproduce the sealed forecasts: {check}")
        sets = field_sets()
        frames = [registered]
        tables = {}
        for name in ("b2_plus_lce", "b2_plus_cb", "cb_only"):
            tables[name], _ = F.matched_forecast(train, query, sets[name])
        cond, _ = F.matched_forecast(train[train.event < 6], query, sets["oeh_final"])
        cond = cond[:, :6] / cond[:, :6].sum(1, keepdims=True)
        oeh = sealed[sealed.method == "oeh_final"].set_index(KEYS).loc[pd.MultiIndex.from_frame(query[KEYS])][P].to_numpy()
        selection = json.loads((DEVELOPMENT / "selection.json").read_text(encoding="utf-8"))
        prior = np.stack([np.asarray(selection["oeh_s"]["class_prior"][a]) for a in query.algorithm])
        tables["oeh_s"] = soften(oeh, prior, selection["oeh_s"]["chosen"]["T"], selection["oeh_s"]["chosen"]["eps"])
        w = selection["oeh_2t"]["chosen"]["w"]
        tables["oeh_2t"] = w * np.column_stack([cond * (1 - oeh[:, 6:7]), oeh[:, 6]]) + (1 - w) * oeh
        tables["lce_raw"] = lce_law(query)
        for name, probability in tables.items():
            frame = query[[*KEYS, "panel"]].copy()
            frame["method"] = name
            frame[P] = probability
            frames.append(frame)
        predictions = pd.concat(frames, ignore_index=True)
        inputs = sorted(opened)
    predictions.to_csv(output / "predictions_before_label_read.csv", index=False)
    query.to_csv(output / "query.csv.gz", index=False)
    H.write_json(output / "prediction_seal.json", {
        "sealed_at_utc": H.now(), "labels_read": False, "reproduces_sealed_max_abs_diff": check,
        "prediction_sha256": H.sha256(output / "predictions_before_label_read.csv"),
        "plan_sha256": H.sha256(output / "analysis_plan.json"), "inputs_opened": len(inputs),
        "seconds": time.perf_counter() - tick})
    print(f"sealed {len(predictions)} predictions; reproduction {check}", flush=True)


def run_score(output: Path) -> None:
    from scripts.analysis import run_oeh_final_120 as R

    seal = json.loads((output / "prediction_seal.json").read_text(encoding="utf-8"))
    if H.sha256(output / "predictions_before_label_read.csv") != seal["prediction_sha256"]:
        raise RuntimeError("Predictions changed after their seal")
    labels = pd.read_csv(CONFIRM / "colosseum_labels.csv")
    pred = pd.read_csv(output / "predictions_before_label_read.csv")
    scored = H.score_rows(pred.merge(labels[KEYS + ["event"]], on=KEYS, validate="many_to_one"))
    scored["log_loss"] = -np.log(np.maximum(scored[P].to_numpy(float)[np.arange(len(scored)), scored.event.to_numpy(int)], 1e-12))
    scored["family"] = H.family(scored.mdp_id)
    scored.to_csv(output / "scored.csv.gz", index=False)
    mdp = scored.groupby(["method", "mdp_id"])[["brier", "log_loss"]].mean().reset_index()
    summary = []
    for scope, by in (("pooled", []), ("learner", ["algorithm"]), ("family", ["family"])):
        t = scored.groupby([*by, "method", "mdp_id"])[["brier", "log_loss"]].mean().groupby([*by, "method"]).mean().reset_index()
        t.insert(0, "scope", scope)
        summary.append(t)
    pd.concat(summary).to_csv(output / "summary.csv", index=False)
    rows = H.contrast_family(mdp, list(R.COMPARATORS), "oeh_final", 20261004, "amendment1_complete_22", ("team", "handover"))
    table = pd.DataFrame(rows)
    table["tier"] = np.where(table.baseline.isin(R.TIER1), 1, 2)
    table.to_csv(output / "amendment1_complete_contrasts.csv", index=False)
    verdict = {rule: {"tier1": "GO" if (g[g.tier == 1].simultaneous_lower_95 > 0).all() else "NO-GO",
                      "tier2": "GO" if (g[g.tier == 2].simultaneous_lower_95 > 0).all() else "NO-GO",
                      "all": "GO" if (g.simultaneous_lower_95 > 0).all() else "NO-GO"}
               for rule, g in table.groupby("rule")}
    ids = sorted(mdp.mdp_id.unique())
    index = np.random.default_rng(H.SEED).integers(0, len(ids), size=(H.BOOTSTRAP, len(ids)))
    wide = {m: mdp.pivot(index="mdp_id", columns="method", values=m).reindex(ids) for m in ("brier", "log_loss")}
    deltas = []
    for name in (*NEW_METHODS, "b2_matched", "prior_oeh_matched"):
        for metric, w in wide.items():
            delta = (w[name] - w["oeh_final"]).to_numpy(float)
            low, high = H.percentile_ci(delta, index)
            deltas.append({"method": name, "metric": metric, "method_mean": w[name].mean(), "oeh_mean": w["oeh_final"].mean(),
                           "method_minus_oeh": delta.mean(), "ci95_low": low, "ci95_high": high})
    pd.DataFrame(deltas).to_csv(output / "cohort8_methods_vs_oeh.csv", index=False)
    H.write_json(output / "manifest.json", {"completed_at_utc": H.now(), "verdict_amendment1_complete": verdict,
                                            "labels_sha256": H.sha256(CONFIRM / "colosseum_labels.csv"),
                                            "output_sha256": {n: H.sha256(output / n) for n in (
                                                "summary.csv", "amendment1_complete_contrasts.csv",
                                                "cohort8_methods_vs_oeh.csv", "scored.csv.gz")}})
    pd.set_option("display.width", 220)
    print(verdict)
    print(table[table.rule == "team"][["tier", "baseline", "metric", "relative_gain", "simultaneous_lower_95"]].round(4).to_string(index=False))
    print(pd.DataFrame(deltas).round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "forecast", "score"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    {"freeze": lambda: run_freeze(args.output), "forecast": lambda: run_forecast(args.output, args.workers),
     "score": lambda: run_score(args.output)}[args.phase]()


if __name__ == "__main__":
    main()

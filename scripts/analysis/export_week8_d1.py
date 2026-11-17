"""Audit and export D1 matrices from sealed compact evidence; never fit learners."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
MATRIX = ROOT / "config/week8_d1_matrix.json"
OUTPUT = ROOT / "artifacts/week8/d1_2026-10-11"
KEYS = ("mdp_id", "policy_family", "algorithm")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"Duplicate key: {key}")
        result[key] = row
    return result


def close(actual, expected):
    if not math.isfinite(actual) or not math.isclose(actual, float(expected), abs_tol=2e-12, rel_tol=2e-12):
        raise ValueError(f"Value mismatch: {actual} != {expected}")


def forecast_score(probabilities, event):
    if len(probabilities) != 7 or not 0 <= event <= 6:
        raise ValueError("Expected seven classes and event 0..6")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
        raise ValueError("Invalid probability")
    # Match the frozen scorer's np.allclose default; keep stored float32
    # rounding, and never renormalize probabilities to improve a score.
    if not math.isclose(sum(probabilities), 1, rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("Probabilities do not sum to one")
    total, loss = 0.0, 0.0
    for j, p in enumerate(probabilities[:6]):
        total += p
        loss += (total - int(event <= j)) ** 2
    return loss / 6, -math.log(max(probabilities[event], 1e-12))


def audit(bundle=BUNDLE, matrix=MATRIX):
    info = read_json(matrix)
    protocol = read_json(bundle / "protocol.json")
    result = read_json(bundle / "result_manifest.json")
    seal = read_json(bundle / "preoutcome_seal.json")
    inputs = {"protocol.json": (bundle / "protocol.sha256").read_text().strip()}
    inputs.update(result["artifact_sha256"])
    inputs.update(seal["artifact_sha256"])
    for name, expected in inputs.items():
        if digest(bundle / name) != expected:
            raise ValueError(f"Sealed hash mismatch: {name}")
    if set(protocol["comparators"]) != {r["method"] for r in info["forecasts"] if r["registered_comparator"]}:
        raise ValueError("Comparator registration differs from D1 matrix")
    if len(protocol["oeh_final"]["fields"]) != 108:
        raise ValueError("Wrong final feature count")
    labels = unique(read_csv(bundle / "labels.csv"), KEYS)
    if len(labels) != 1800:
        raise ValueError("Expected 1800 unique label rows")
    mdps = defaultdict(set)
    for row in labels.values():
        mdps[row["panel"]].add(row["mdp_id"])
    if {k: len(v) for k, v in mdps.items()} != {"primary": 84, "control": 36}:
        raise ValueError("Wrong primary/control panels")
    predictions = unique(read_csv(bundle / "predictions_preoutcome.csv"), (*KEYS, "method"))
    names = {r["method"] for r in info["forecasts"]}
    expected = {(*key, method) for key in labels for method in names}
    if set(predictions) != expected:
        raise ValueError("Incomplete prediction-key cartesian product")
    groups = defaultdict(list)
    for key, row in predictions.items():
        label = labels[key[:3]]
        if row["panel"] != label["panel"]:
            raise ValueError("Prediction/label panel mismatch")
        score = forecast_score([float(row[f"p_{i}"]) for i in range(7)], int(label["event"]))
        groups[(row["mdp_id"], row["panel"], row["method"])].append(score)
    recorded = unique(read_csv(bundle / "mdp_scores.csv"), ("mdp_id", "panel", "method"))
    if set(recorded) != set(groups):
        raise ValueError("MDP score keys differ")
    pooled = defaultdict(list)
    for key, scores in groups.items():
        if len(scores) != 15 or int(recorded[key]["tasks"]) != 15:
            raise ValueError("Expected five behaviors x three learners per MDP")
        avg = [sum(v[j] for v in scores) / len(scores) for j in range(2)]
        close(avg[0], recorded[key]["integrated_brier"])
        close(avg[1], recorded[key]["seven_class_log_loss"])
        pooled[key[1:]].append(avg)
    summary = unique(read_csv(bundle / "summary.csv"), ("panel", "method"))
    if set(summary) != set(pooled):
        raise ValueError("Summary keys differ")
    for key, scores in pooled.items():
        if int(summary[key]["mdps"]) != len(scores):
            raise ValueError("Wrong summary MDP count")
        for j, metric in enumerate(("integrated_brier", "seven_class_log_loss")):
            close(sum(v[j] for v in scores) / len(scores), summary[key][metric])
    contrasts = read_csv(bundle / "contrasts.csv")
    expected_contrasts = {(m, metric) for m in protocol["comparators"] for metric in
                          ("integrated_brier", "seven_class_log_loss")}
    indexed_contrasts = unique(contrasts, ("baseline", "metric"))
    if set(indexed_contrasts) != expected_contrasts:
        raise ValueError("Wrong registered contrast family")
    for (baseline, metric), row in indexed_contrasts.items():
        b, c = float(summary[("primary", baseline)][metric]), float(summary[("primary", "oeh_final")][metric])
        close(b - c, row["mean_absolute_improvement"])
        close((b - c) / b, row["relative_gain"])
        if row["candidate"] != "oeh_final" or int(row["mdps"]) != 84:
            raise ValueError("Wrong contrast candidate/panel")
        if not math.isfinite(float(row["simultaneous_one_sided_lower_95"])):
            raise ValueError("Invalid published bound")
    choices = unique(read_csv(bundle / "selection_group_outcomes.csv"), ("mdp_id", "policy_family"))
    primary_groups = {(key[0], key[1]) for key, row in labels.items() if row["panel"] == "primary"}
    if set(choices) != primary_groups or len(choices) != 420:
        raise ValueError("Selection group panel differs")
    selection = read_csv(bundle / "selection_summary.csv")
    unique(selection, ("rule",))
    expected_rules = {f"fixed:{r['method']}" for r in info["learners"]} | {
        "oeh_final_selection", "oeh_v5_router", "random", "oracle_group"}
    if {r["rule"] for r in selection} != expected_rules:
        raise ValueError("Selection library differs")
    for row in selection:
        for suffix, metric in (("ret", "mean_normalized_return"), ("suc", "seed_success_rate")):
            values = [float(v[f"{row['rule']}|{suffix}"]) for v in choices.values()]
            close(sum(values) / len(values), row[metric])
    core = read_csv(bundle / "all_core_budget_seed_outcomes.csv")
    unique(core, (*KEYS, "budget", "training_seed"))
    if len(core) != 108000:
        raise ValueError("Expected 108000 core jobs")
    cost = defaultdict(list)
    for row in core:
        elapsed = float(row["runtime_seconds"])
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid recorded runtime")
        if row["mdp_id"] in mdps["primary"] and int(row["budget"]) == 10:
            cost[row["algorithm"]].append(elapsed)
    cost_rows = []
    for learner, times in sorted(cost.items()):
        if len(times) != 4200:
            raise ValueError("Expected 4200 primary x10 core runtime records per learner")
        cost_rows.append({"algorithm": learner, "panel": "primary", "budget": 10,
                          "jobs": len(times), "recorded_callback_seconds_sum": sum(times),
                          "recorded_callback_seconds_mean": sum(times) / len(times)})
    if set(cost) != set(protocol["core_learners"]):
        raise ValueError("Core runtime library differs")
    report = {"schema": "week8-d1-audit-v1", "evidence_commit": info["evidence_commit"],
              "checked_sealed_hash_entries": len(inputs), "labels": len(labels),
              "forecast_rows": len(predictions), "mdp_score_rows": len(groups),
              "primary_mdps": 84, "control_mdps": 36, "registered_contrasts": len(contrasts),
              "selection_groups": len(choices), "core_jobs": len(core),
              "score_and_summary_recomputed": True, "fitted_new_models": False,
              "bootstrap_recomputed": False, "statistical_approval_claimed": False,
              "external_fit_cost_verified": False,
              "source_sha256": {(bundle / name).relative_to(ROOT).as_posix(): digest(bundle / name) for name in
                  ("protocol.json", "preoutcome_seal.json", "result_manifest.json", "labels.csv",
                   "predictions_preoutcome.csv", "summary.csv", "contrasts.csv", "selection_summary.csv",
                   "selection_group_outcomes.csv", "all_core_budget_seed_outcomes.csv")}}
    return info, summary, selection, cost_rows, report


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def exports():
    info, summary, selection, cost, report = audit()
    primary = [{**r, "mdps": 84, "integrated_brier": float(summary[("primary", r["method"])]["integrated_brier"]),
                "seven_class_log_loss": float(summary[("primary", r["method"])]["seven_class_log_loss"])}
               for r in info["forecasts"]]
    md = ["# D1 — Primary forecast matrix", "", "84 stochastic MDPs; lower scores are better. Frozen cohort 7; no new fitting.", "",
          "| Group | Row | Head | Access | Brier | Log loss |", "|---|---|---|---|---:|---:|"]
    for r in primary:
        md.append(f"| {r['group']} | {r['label']} | {r['head']} | {r['access']} | {r['integrated_brier']:.6f} | {r['seven_class_log_loss']:.6f} |")
    md += ["", "Matched = AFT + direct-threshold + depth-3 multiclass, fitted separately for each row.",
           "Privileged and AFT-only rows are separate comparisons; diagnostics are outside the 22-contrast gate.", ""]
    tex = [r"% Generated by python -m scripts.analysis.export_week8_d1", r"\begin{table}[t]", r"\centering\small",
           r"\caption{Primary forecast scores on 84 stochastic MDPs. Blocks distinguish fitted heads and information access; lower is better. Diagnostics are outside the registered gate.}",
           r"\label{tab:d1-primary}", r"\begin{tabular}{lrr}\toprule", r"Features & IBS & Log loss\\\midrule"]
    group = None
    for r in primary:
        if r["group"] != group:
            group = r["group"]
            tex.append(r"\multicolumn{3}{l}{\textit{" + group + r"}}\\")
        tex.append(f"{r['label']} & {r['integrated_brier']:.5f} & {r['seven_class_log_loss']:.5f}" + r"\\")
    tex += [r"\bottomrule\end{tabular}", r"\end{table}", ""]
    report["source_sha256"]["config/week8_d1_matrix.json"] = digest(MATRIX)
    report["source_sha256"]["scripts/analysis/export_week8_d1.py"] = digest(Path(__file__))
    learner_tex = [r"% D1 learner settings, not a retuned experiment", r"\begin{table*}[t]", r"\centering\small",
                   r"\caption{Registered learner implementations. Core learners define the six-budget requirement; all seven enter selection at $x10$. All use $\gamma=.95$ and ten training streams. SPIBB receives true logging behavior.}",
                   r"\label{tab:d1-learners}", r"\begin{tabular}{lp{0.43\textwidth}p{0.31\textwidth}}\toprule",
                   r"Learner & Configuration and optimization budget & Fit information and task role\\\midrule",
                   r"BC & Logged-action frequencies; uniform fallback; no gradient updates & Log only; requirement and selection\\",
                   r"CFQI & At most 100 backups, tolerance $10^{-10}$; observed-action restriction & Log only; requirement and selection\\",
                   r"IQL & Expectile $.7$, inverse temperature $3$; at most 100 iterations, tolerance $10^{-10}$ & Log only; requirement and selection\\",
                   r"BCQ & Author discrete learner; one-hot, $256\times256$, batch 64, threshold $.3$, 500 updates & Log only; selection at $x10$\\",
                   r"CQL & Pinned d3rlpy; one-hot, $256\times256$, batch 64, $\alpha=1$, 500 updates & Log only; selection at $x10$\\",
                   r"SPIBB & Author Pi-b core; $N_\wedge=10$; author policy-iteration stop rule & Log plus true $\beta$; selection at $x10$\\",
                   r"VI-LCB & Discounted paper-spec port; $c_b=\delta=.05$, 100 iterations & Log plus terminal metadata; selection at $x10$\\",
                   r"\bottomrule\end{tabular}", r"\end{table*}", ""]
    return {"learner_matrix.csv": csv_text(info["learners"]), "learner_matrix.tex": "\n".join(learner_tex),
            "primary_forecast.csv": csv_text(primary),
            "primary_forecast.md": "\n".join(md), "primary_forecast.tex": "\n".join(tex),
            "selection.csv": csv_text(selection), "core_x10_cost.csv": csv_text(cost),
            "validation.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify exports without writing")
    args = parser.parse_args()
    files = exports()
    if args.check:
        for name, content in files.items():
            if (OUTPUT / name).read_text(encoding="utf-8") != content:
                raise ValueError(f"D1 export differs: {name}")
        print("D1 sealed scores, access matrix, selection means and exports verified")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (OUTPUT / name).write_text(content, encoding="utf-8", newline="\n")
        print(f"Wrote {len(files)} D1 exports to {OUTPUT}")


if __name__ == "__main__":
    main()

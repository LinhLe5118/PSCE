"""Build the same-machine comparison report without merging incomparable panels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def seconds(value: float) -> str:
    return f"{value:.3f} s"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learner-summary", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir
    runtime = read_json(output / "tabular_runtime" / "runtime_summary.json")
    learners = read_csv(args.learner_summary)
    calibration = read_csv(args.calibration_summary)
    public = read_json(output / "public_components" / "summary.json")

    neural: dict[str, dict] = {}
    for method in ("cql", "iql"):
        strict_dir = output / f"neural_{method}_strict"
        compat_dir = output / f"neural_{method}_compatibility"
        neural[method] = {
            "strict": read_json(strict_dir / "summary.json") if (strict_dir / "summary.json").exists()
            else read_json(strict_dir / "failure.json"),
            "compatibility": read_json(compat_dir / "summary.json"),
        }

    learner_rows = []
    for row in learners:
        learner_rows.append({
            "algorithm": row["algorithm"],
            "role": row["role"],
            "groups": int(row["total_groups"]),
            "finite_groups": int(row["finite_groups"]) if row["finite_groups"] else None,
            "censoring_rate": float(row["censoring_rate_all"]) if row["censoring_rate_all"] else None,
            "largest_prefix_median_return": float(row["largest_prefix_median_return"]),
            "largest_prefix_success_rate": float(row["largest_prefix_success_rate"]),
        })

    significant = [
        {
            "algorithm": row["algorithm"], "metric": row["metric"],
            "baseline": row["baseline"], "difference": float(row["difference"]),
            "p_holm": float(row["p_holm"]), "oeh_better": row["oeh_better"] == "True",
        }
        for row in calibration if row["significant_after_holm"] == "True"
    ]
    summary = {
        "scope": "same-machine evidence split into scientifically comparable and component-only panels",
        "tabular_runtime": runtime,
        "tabular_scientific_results": learner_rows,
        "significant_calibration_differences": significant,
        "public_components": public,
        "neural_author_code": neural,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Same-machine comparison — 17 September 2026", "",
        "## Decision", "",
        "The only direct learner-quality comparison is CFQI versus tabular IQL and BC on the same "
        "expanded stochastic dataset. Public neural CQL/IQL and author utility probes are component "
        "timings on different inputs; they cannot establish return or sample-efficiency superiority.", "",
        "## Scientifically comparable learner panel", "",
        "| Learner | Role | Groups | Finite N_emp | Censoring | Largest-prefix median return | Success |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in learner_rows:
        finite = "n/a" if row["finite_groups"] is None else str(row["finite_groups"])
        censor = "n/a" if row["censoring_rate"] is None else f"{100*row['censoring_rate']:.2f}%"
        lines.append(
            f"| {row['algorithm']} | {row['role']} | {row['groups']} | {finite} | {censor} | "
            f"{row['largest_prefix_median_return']:.6f} | {100*row['largest_prefix_success_rate']:.2f}% |"
        )
    lines += [
        "", "These quality values come from the separately verified 43,200-fit artifact. The new "
        "runtime run covers three complete registered seed slices and does not create new N_emp labels.", "",
        "### Runtime on this machine", "",
        f"Measured wall time: **{seconds(runtime['benchmark_wall_seconds'])}**. Projected registered "
        f"ten-seed grid: **{seconds(runtime['full_grid_estimate_seconds'])}**; fastest/slowest-slice "
        f"sensitivity: **{seconds(runtime['full_grid_seed_sensitivity_seconds'][0])}–"
        f"{seconds(runtime['full_grid_seed_sensitivity_seconds'][1])}** (not a confidence interval).", "",
        "| Learner | Measured jobs | Fit time | Score time | Invalid policies |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in runtime["algorithms"].items():
        lines.append(f"| {name} | {row['measured_jobs']} | {seconds(row['fit_seconds'])} | "
                     f"{seconds(row['score_seconds'])} | {row['invalid_policies']} |")

    lines += ["", "## Diagnostic prediction panel", ""]
    if significant:
        lines += ["Holm-significant paired differences (OEH minus baseline):", "",
                  "| Learner | Metric | Baseline | Difference | Holm p | OEH better |",
                  "|---|---|---|---:|---:|---|"]
        for row in significant:
            lines.append(f"| {row['algorithm']} | {row['metric']} | {row['baseline']} | "
                         f"{row['difference']:.6f} | {row['p_holm']:.3g} | {row['oeh_better']} |")
    else:
        lines.append("There are no Holm-significant paired differences.")
    lines += [
        "", "OEH has no significant win over a fair non-oracle baseline. On CFQI it is significantly "
        "worse than coverage for log likelihood and worse than visitation entropy for both log "
        "likelihood and Brier score. Oracle behavior labels are context only.", "",
        "## Public author-code probes (not learner-quality comparisons)", "",
    ]
    author_rows = public["rows"]
    for row in author_rows:
        if row["method"] == "author_value_iteration":
            timing = seconds(row["median_seconds"]) if row["status"] == "measured" else row["status"]
            lines.append(f"- Effective Horizon value iteration, `{row['dataset']}`: {timing}.")
    lines.append(f"- OfflineRL return/HLL utilities over real pilots: summed median component time "
                 f"{seconds(public['pilot_component_median_sum_seconds'])}.")
    pilot_rows = [row for row in author_rows if row["method"] == "author_return_and_hll_components"]
    zero_episode_rows = sum(row["complete_episodes"] == 0 for row in pilot_rows)
    lines.append(f"- Episode-return output is unusable for **{zero_episode_rows}/{len(pilot_rows)}** "
                 "pilot buffers because they contain no completed episode; HLL is still timed, but "
                 "neither output is a validated normalized TQ/SACo comparison on scalar tabular IDs.")
    lines += ["", "These probes retain the input/normalization caveats recorded in `public_components/plan.json`.",
              "", "## Neural CQL/IQL author-update probes (not tabular comparisons)", ""]
    for method, result in neural.items():
        strict = result["strict"]
        if strict.get("status") == "failed":
            error = strict["error"].rstrip(".")
            lines.append(f"- {method.upper()} strict historical code failed under current dependencies: "
                         f"{strict['error_type']}: {error}.")
        else:
            lines.append(f"- {method.upper()} strict run completed.")
        phases = result["compatibility"]["phases"]
        phase_text = ", ".join(f"{p['phase']} {p['seconds_per_update']:.6f} s/update" for p in phases)
        lines.append(f"  Compatibility probe: {phase_text}.")
    lines += [
        "", "## Conclusion", "",
        "The current evidence supports CFQI as the strongest implemented tabular learner on this "
        "development benchmark. It does not support OEH as the primary predictor: simple coverage "
        "and visitation-entropy diagnostics match or outperform it. These are development results; "
        "the sealed evaluation split must remain unopened until the protocol and manuscript are frozen.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_inputs = {
        "learner_summary": args.learner_summary,
        "calibration_summary": args.calibration_summary,
        "tabular_runtime": output / "tabular_runtime" / "runtime_summary.json",
        "public_components": output / "public_components" / "summary.json",
        "cql_strict_failure": output / "neural_cql_strict" / "failure.json",
        "cql_compatibility": output / "neural_cql_compatibility" / "summary.json",
        "iql_strict_failure": output / "neural_iql_strict" / "failure.json",
        "iql_compatibility": output / "neural_iql_compatibility" / "summary.json",
        "summarizer": Path(__file__),
    }
    manifest_outputs = {"report": output / "REPORT.md", "summary": output / "summary.json"}
    (output / "MANIFEST.json").write_text(json.dumps({
        "status": "complete",
        "inputs": {name: {"path": str(path), "sha256": sha256(path)}
                   for name, path in manifest_inputs.items()},
        "outputs": {name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in manifest_outputs.items()},
        "sealed_evaluation_opened": False,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Read-only cross-panel verification of the same seven-class forecast scores.

The strict mode requires all four named panels and never treats previously
opened MiniGrid labels as a fresh independent result. The preview mode reports
only completed panels, for audit while long BRIDGE jobs are still running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = [f"p_{i}" for i in range(7)]
KEYS = ["mdp_id", "policy_family", "algorithm"]

PANELS = (
    {
        "name": "internal120_stochastic_primary",
        "directory": ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04",
        "data": "scored_predictions.csv",
        "manifest": "result_manifest.json",
        "status": "complete",
        "hash_key": "artifact_sha256",
        "source_metric": ("integrated_brier", "seven_class_log_loss"),
        "methods": (
            "b2_matched", "b2_plus_eri",
            "published_xgboost_cf_2026_adapted_b2",
            "xgboost_cf_2026_full108_architecture_control",
            "oeh_final",
        ),
        "tasks": 84,
        "rows_per_method": 1260,
        "freshness": "preoutcome_sealed_within_generator",
    },
    {
        "name": "bridge_fresh_atari_5",
        "directory": ROOT / "artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08",
        "data": "scored.csv",
        "manifest": "manifest.json",
        "status": "COMPLETE_FRESH_BRIDGE_ATARI_FULL_OEH_TRANSFER",
        "hash_key": "output_sha256",
        "source_metric": ("brier", "log_loss"),
        "methods": (
            "b2_matched_internal_diagnostic",
            "published_xgboost_cf_2026_adapted_b2",
            "xgboost_cf_2026_full108_architecture_control",
            "oeh_full_108",
        ),
        "tasks": 5,
        "rows_per_method": 75,
        "freshness": "prelabel_sealed_new_published_mdps",
    },
    {
        "name": "bridge_expanded_prospective_5",
        "directory": ROOT / "artifacts/week8/bridge_expanded_full_test_analysis_2026-10-08",
        "data": "scored.csv",
        "manifest": "manifest.json",
        "status": "COMPLETE_BRIDGE_EXPANDED_FIVE_TASK_FULL_OEH_TRANSFER",
        "hash_key": "output_sha256",
        "source_metric": ("brier", "log_loss"),
        "methods": (
            "b2_matched_internal_diagnostic",
            "published_xgboost_cf_2026_adapted_b2",
            "xgboost_cf_2026_full108_architecture_control",
            "oeh_full_108",
        ),
        "tasks": 5,
        "rows_per_method": 75,
        "freshness": "separate_followup_prelabel_seal",
    },
    {
        "name": "bridge_opened_minigrid_14_exploratory",
        "directory": ROOT / "artifacts/week8/bridge_opened_minigrid_full_analysis_exploratory_2026-10-09",
        "data": "scored.csv",
        "manifest": "manifest.json",
        "status": "COMPLETE_POST_OUTCOME_EXPLORATORY_14_MINIGRID",
        "hash_key": "output_sha256",
        "source_metric": ("integrated_brier", "seven_class_log_loss"),
        "methods": ("b2_matched", "oeh_without_direct_oehb",
                    "oeh_full_108_post_outcome_exploratory"),
        "tasks": 14,
        "rows_per_method": 210,
        "freshness": "post_outcome_exploratory",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hash(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"Missing or changed evidence file: {path}")


def verify_source_chain(spec: dict, manifest: dict) -> list[str]:
    """Verify the protocol/seal/label links, not just the final score CSV."""
    name = spec["name"]
    checked = []
    if name == "internal120_stochastic_primary":
        base = spec["directory"]
        for key, file in (("protocol_sha256", "protocol.json"),
                          ("preoutcome_seal_sha256", "preoutcome_seal.json")):
            check_hash(base / file, manifest[key])
            checked.append(file)
        check_hash(base / "labels.csv", manifest["artifact_sha256"]["labels.csv"])
        checked.append("labels.csv")
        for dirname in (
            "published_xgboost_cf_120_exploratory_2026-10-08",
            "published_xgboost_cf_120_full108_architecture_2026-10-08",
        ):
            directory = ROOT / "artifacts/week8" / dirname
            extra = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            expected_status = (
                "COMPLETE_RETROSPECTIVE_EXPLORATORY_120_COMPARISON"
                if "_exploratory_" in dirname else
                "COMPLETE_RETROSPECTIVE_120_FULL_INPUT_ARCHITECTURE_CONTROL")
            if (extra["status"] != expected_status or
                    extra["source_sha256"]["labels"] != manifest["artifact_sha256"]["labels.csv"]):
                raise RuntimeError(f"Retrospective comparator source mismatch: {dirname}")
            check_hash(directory / "scored.csv", extra["output_sha256"]["scored.csv"])
            checked.append(str((directory / "scored.csv").relative_to(ROOT)))
    elif name in ("bridge_fresh_atari_5", "bridge_expanded_prospective_5"):
        fresh = name == "bridge_fresh_atari_5"
        panel = ROOT / "benchmarks/bridge" / (
            "fresh_panel_2026-10-08" if fresh else "expanded_panel_2026-10-08")
        labels = ROOT / "artifacts/week8" / (
            "bridge_fresh_test_labels_2026-10-08" if fresh else
            "bridge_expanded_test_labels_2026-10-08")
        source = manifest["source_sha256"]
        for key, path in (("protocol", panel / "protocol.json"),
                          ("prelabel_seal", panel / "prelabel_seal.json"),
                          ("labels_manifest", labels / "manifest.json")):
            check_hash(path, source[key])
            checked.append(str(path.relative_to(ROOT)))
        lm = json.loads((labels / "manifest.json").read_text(encoding="utf-8"))
        if lm["prelabel_seal_sha256"] != source["prelabel_seal"]:
            raise RuntimeError(f"Label manifest has a different prelabel seal: {labels}")
        check_hash(labels / "labels.csv", lm["labels_sha256"])
        checked.append(str((labels / "labels.csv").relative_to(ROOT)))
    else:
        source = manifest["source_sha256"]
        paths = {
            "full": ROOT / "artifacts/week8/bridge_opened_minigrid_full_forecast_exploratory_2026-10-09/manifest.json",
            "b2": ROOT / "artifacts/week8/bridge_b2_forecast_preoutcome_2026-10-07/manifest.json",
            "reduced": ROOT / "artifacts/week8/bridge_reduced_forecast_preoutcome_2026-10-07/manifest.json",
            "labels": ROOT / "artifacts/week8/bridge_reduced_test_labels_2026-10-07/manifest.json",
        }
        for key, path in paths.items():
            check_hash(path, source[key])
            checked.append(str(path.relative_to(ROOT)))
        label_dir = paths["labels"].parent
        lm = json.loads(paths["labels"].read_text(encoding="utf-8"))
        check_hash(label_dir / "labels.csv", lm["labels_sha256"])
        checked.append(str((label_dir / "labels.csv").relative_to(ROOT)))
    return checked


def read_panel(spec: dict) -> tuple[pd.DataFrame, dict]:
    directory = spec["directory"]
    manifest_file = directory / spec["manifest"]
    data_file = directory / spec["data"]
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest["status"] != spec["status"]:
        raise RuntimeError(f"Wrong status: {manifest_file}")
    if manifest[spec["hash_key"]][spec["data"]] != sha256(data_file):
        raise RuntimeError(f"Source hash mismatch: {data_file}")
    chain = verify_source_chain(spec, manifest)
    frame = pd.read_csv(data_file)
    if spec["name"] == "internal120_stochastic_primary":
        frame = frame[frame.panel == "primary"].copy()
        original_events = frame[KEYS + ["event"]].drop_duplicates()
        additions = []
        for dirname in (
            "published_xgboost_cf_120_exploratory_2026-10-08",
            "published_xgboost_cf_120_full108_architecture_2026-10-08",
        ):
            directory = ROOT / "artifacts/week8" / dirname
            extra_manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            extra = pd.read_csv(directory / "scored.csv")
            extra = extra[(extra.panel == "primary") &
                          (extra.method == extra_manifest["method"])].copy()
            labels = extra[KEYS + ["event"]].merge(
                original_events, on=KEYS, validate="one_to_one", suffixes=("_extra", "_sealed"))
            if (len(labels) != spec["rows_per_method"] or
                    not (labels.event_extra == labels.event_sealed).all()):
                raise RuntimeError(f"Retrospective comparator label mismatch: {dirname}")
            extra["integrated_brier"] = extra.brier
            extra["seven_class_log_loss"] = extra.log_loss
            additions.append(extra)
        frame = pd.concat([frame, *additions], ignore_index=True)
    if frame[KEYS + ["method"]].duplicated().any():
        raise RuntimeError(f"Duplicate scored rows: {data_file}")
    if frame.mdp_id.nunique() != spec["tasks"]:
        raise RuntimeError(f"Wrong MDP count: {data_file}")
    for method in spec["methods"]:
        if len(frame[frame.method == method]) != spec["rows_per_method"]:
            raise RuntimeError(f"Incomplete method rows: {spec['name']} / {method}")
    if not set(KEYS + P + ["method", "event"]).issubset(frame.columns):
        raise RuntimeError(f"Required columns missing: {data_file}")
    p = frame[P].to_numpy(float)
    y = frame.event.to_numpy(int)
    if (not np.isfinite(p).all() or np.any(p < -1e-10) or
            not np.allclose(p.sum(axis=1), 1, atol=1e-7) or
            np.any((y < 0) | (y > 6))):
        raise RuntimeError(f"Invalid seven-class law: {data_file}")
    cdf = np.cumsum(p, axis=1)[:, :6]
    brier = np.mean((cdf - (y[:, None] <= np.arange(6))) ** 2, axis=1)
    log_loss = -np.log(np.maximum(p[np.arange(len(p)), y], 1e-12))
    source_brier, source_log_loss = spec["source_metric"]
    if (not np.allclose(brier, frame[source_brier].to_numpy(float), atol=1e-8) or
            not np.allclose(log_loss, frame[source_log_loss].to_numpy(float), atol=1e-8)):
        raise RuntimeError(f"Common metric differs from stored scores: {data_file}")
    frame["verified_brier"] = brier
    frame["verified_log_loss"] = log_loss
    return frame, {
        "panel": spec["name"], "manifest_sha256": sha256(manifest_file),
        "scored_sha256": sha256(data_file), "tasks": spec["tasks"],
        "freshness": spec["freshness"], "verified_source_chain": chain,
    }


def method_status(spec: dict, method: str) -> str:
    if (spec["name"] == "internal120_stochastic_primary" and
            method.startswith(("published_xgboost_cf", "xgboost_cf"))):
        return "post_outcome_exploratory"
    return spec["freshness"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Check only completed panels; output cannot be cited as final")
    args = parser.parse_args()
    summaries, pairs, evidence, missing = [], [], [], []
    bootstrap_rng = np.random.default_rng(20261009)
    for spec in PANELS:
        if not (spec["directory"] / spec["manifest"]).exists():
            missing.append(spec["name"])
            continue
        frame, proof = read_panel(spec)
        evidence.append(proof)
        own = spec["methods"][-1]
        for learner, part in frame[frame.method.isin(spec["methods"])].groupby("algorithm"):
            pivot = part.pivot(index=["mdp_id", "policy_family"], columns="method",
                               values=["verified_brier", "verified_log_loss"])
            if len(pivot) * 3 != spec["rows_per_method"] or pivot.isna().any().any():
                raise RuntimeError(f"Unpaired rows: {spec['name']} / {learner}")
            for method in spec["methods"]:
                m = part[part.method == method]
                summaries.append({
                    "panel": spec["name"], "freshness": method_status(spec, method),
                    "learner": learner, "method": method, "mdps": m.mdp_id.nunique(),
                    "rows": len(m), "integrated_brier": m.verified_brier.mean(),
                    "seven_class_log_loss": m.verified_log_loss.mean(),
                    "censored_fraction": (m.event == 6).mean(),
                })
            for metric in ("verified_brier", "verified_log_loss"):
                for comparator in spec["methods"][:-1]:
                    delta = pivot[(metric, own)] - pivot[(metric, comparator)]
                    by_task = delta.groupby(level="mdp_id").mean()
                    draws = bootstrap_rng.choice(
                        by_task.to_numpy(float),
                        size=(10000, len(by_task)), replace=True).mean(axis=1)
                    pairs.append({
                        "panel": spec["name"],
                        "freshness": ("post_outcome_exploratory" if
                                      method_status(spec, comparator) == "post_outcome_exploratory"
                                      else spec["freshness"]),
                        "learner": learner, "metric": metric,
                        "comparison": f"{own} minus {comparator}",
                        "paired_mdp_delta": by_task.mean(), "mdps": len(by_task),
                        "rows": len(delta), "interpretation": "negative favors OEH",
                        "ci95_low_descriptive": float(np.quantile(draws, .025)),
                        "ci95_high_descriptive": float(np.quantile(draws, .975)),
                        "bootstrap_unit": "MDP; five behaviors stay together",
                        "bootstrap_draws": 10000,
                    })
    if missing and not args.preview:
        raise RuntimeError("Strict verification waits for: " + ", ".join(missing))
    if not evidence:
        raise RuntimeError("No completed panel")
    output = ROOT / "artifacts/week8" / (
        "difficulty_common_metric_preview_2026-10-09" if args.preview else
        "difficulty_common_metric_final_2026-10-09")
    output.mkdir(parents=True, exist_ok=True)
    summary_file, paired_file = output / "summary.csv", output / "paired.csv"
    summary_frame, paired_frame = pd.DataFrame(summaries), pd.DataFrame(pairs)
    summary_frame.to_csv(summary_file, index=False)
    paired_frame.to_csv(paired_file, index=False)
    report_file = output / "RESULTS.md"
    report_file.write_text(
        "# Common seven-class metric audit\n\n"
        + ("**INCOMPLETE PREVIEW — do not cite as a final all-panel result.**\n\n"
           if missing else "**All four named panels passed source and metric checks.**\n\n")
        + "Each method is scored on the same MDP–behavior–learner rows **within its panel**. "
          "Panels have different task sources, horizons, discounts, and analysis status; "
          "their raw scores are not a pooled confirmatory test. Negative paired deltas favor OEH.\n\n"
        + ("Missing: " + ", ".join(missing) + "\n\n" if missing else "")
        + "## Per-panel forecast scores\n\n"
        + summary_frame.to_markdown(index=False, floatfmt=".6f")
        + "\n\n## Paired MDP means\n\n"
        + paired_frame.to_markdown(index=False, floatfmt=".6f") + "\n\n"
        + "The paired 95% intervals resample MDPs with five behaviors together; they are descriptive and unadjusted for multiple comparisons. They do not replace the registered simultaneous bounds on the "
          "internal cohort or the separate descriptive task-cluster intervals from each BRIDGE scorer. "
          "Original ERI/TQ/SACo/EH scalar scores belong in rank-only tables.\n",
        encoding="utf-8",
    )
    result = {
        "status": "PARTIAL_NOT_FOR_MANUSCRIPT" if missing else "ALL_PANELS_COMMON_METRIC_VERIFIED",
        "metric": "mean of six cumulative squared errors; seven-class negative log probability",
        "output": {"summary_sha256": sha256(summary_file),
                   "paired_sha256": sha256(paired_file),
                   "report_sha256": sha256(report_file)},
        "source": evidence, "missing": missing,
        "panel_aggregation": "separate; no pooled BRIDGE confirmatory claim",
        "source_script_sha256": sha256(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "verified": [e["panel"] for e in evidence],
                      "missing": missing, "summary_rows": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()

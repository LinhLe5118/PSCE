"""Score the frozen BRIDGE seven-class forecasts after the one test-label run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
B2 = ROOT / "artifacts/week8/bridge_b2_forecast_preoutcome_2026-10-07"
FULL = ROOT / "artifacts/week8/bridge_full_forecast_preoutcome_2026-10-07"
REDUCED = ROOT / "artifacts/week8/bridge_reduced_forecast_preoutcome_2026-10-07"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "reduced"), default="full")
    args = parser.parse_args()
    comparison = FULL if args.mode == "full" else REDUCED
    method = "oeh_full_108" if args.mode == "full" else "oeh_without_direct_oehb"
    labels_dir = ROOT / "artifacts/week8" / f"bridge_{args.mode}_test_labels_2026-10-07"
    output = ROOT / "artifacts/week8" / f"bridge_{args.mode}_test_analysis_2026-10-07"
    if output.exists():
        raise FileExistsError(output)
    source = {}
    frames = []
    for directory, status in ((B2, "SEALED_B2_FORECAST_NO_TEST_LABELS"),
                              (comparison, "SEALED_FULL_108_FORECAST_NO_TEST_LABELS" if args.mode == "full"
                               else "SEALED_REDUCED_OEH_ABLATION_NO_TEST_LABELS")):
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        predictions = directory / "predictions_before_label_read.csv"
        if (manifest["status"] != status or manifest["test_labels_read"] or
                sha256(predictions) != manifest["output_sha256"]["predictions"]):
            raise RuntimeError(f"Invalid prediction seal: {directory}")
        source[directory.name] = sha256(manifest_path)
        frames.append(pd.read_csv(predictions))
    label_manifest_path = labels_dir / "manifest.json"
    label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
    label_path = labels_dir / "labels.csv"
    if (label_manifest["status"] != f"BRIDGE_{args.mode.upper()}_TEST_LABELS_COMPLETE" or
            sha256(label_path) != label_manifest["labels_sha256"]):
        raise RuntimeError("Invalid label seal")
    source["labels_manifest"] = sha256(label_manifest_path)
    predictions = pd.concat(frames, ignore_index=True)
    labels = pd.read_csv(label_path)
    # The adapter records a sticky-state environment_id in labels, whereas the
    # prelabel forecast seal uses the published BRIDGE task name. Keep both IDs
    # and verify an exact one-to-one task mapping before joining outcomes.
    labels["label_task_id"] = labels["mdp_id"]
    suffix = "_sticky25"
    label_tasks = set(labels["label_task_id"].unique())
    forecast_tasks = set(predictions["mdp_id"].unique())
    if (len(label_tasks) != 14 or not all(task.endswith(suffix) for task in label_tasks)
            or {task.removesuffix(suffix) for task in label_tasks} != forecast_tasks):
        raise RuntimeError("Sticky label task IDs do not map exactly to forecast tasks")
    labels["mdp_id"] = labels["label_task_id"].str.removesuffix(suffix)
    scored = predictions.merge(labels[KEYS + ["event", "label_task_id"]],
                               on=KEYS, validate="many_to_one")
    if (len(scored) != 420 or scored[KEYS + ["method"]].duplicated().any() or
            labels[KEYS].duplicated().any() or len(labels) != 210):
        raise RuntimeError("Prediction-label mismatch")
    p = scored[P].to_numpy(float)
    y = scored.event.to_numpy(int)
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1) or np.any((y < 0) | (y > 6)):
        raise RuntimeError("Invalid law/event")
    cdf = np.cumsum(p, axis=1)[:, :6]
    scored["brier"] = np.mean((cdf - (y[:, None] <= np.arange(6))) ** 2, axis=1)
    scored["log_loss"] = -np.log(np.maximum(p[np.arange(len(p)), y], 1e-12))
    # Before test outcomes, q=.8 and the allowed decision set x10/x20/x50 were frozen.
    choices = np.array([3, 4, 5])
    feasible = cdf[:, choices] >= .8
    decision = np.where(feasible.any(axis=1), choices[feasible.argmax(axis=1)], 5)
    scored["chosen_budget"] = np.array([1, 2, 5, 10, 20, 50])[decision]
    scored["target_met"] = (y <= decision).astype(int)
    scored["under_budgeted"] = ((y > decision) & (y < 6)).astype(int)
    summary = (scored.groupby(["algorithm", "method"], observed=True)
               .agg(rows=("event", "size"), mdps=("mdp_id", "nunique"),
                    brier=("brier", "mean"), log_loss=("log_loss", "mean"),
                    censored_fraction=("event", lambda v: (v == 6).mean()),
                    mean_chosen_budget=("chosen_budget", "mean"),
                    target_met=("target_met", "mean"),
                    under_budgeted=("under_budgeted", "mean"))
               .reset_index())
    rng = np.random.default_rng(20261007)
    task_ids = sorted(labels.mdp_id.unique())
    if len(task_ids) != 14:
        raise RuntimeError("Unexpected BRIDGE task count")
    paired = []
    for learner in ("behavior_cloning_tabular", "cfqi_tabular", "iql_tabular"):
        part = scored[scored.algorithm == learner]
        for metric in ("brier", "log_loss"):
            pivot = part.pivot(index=["mdp_id", "policy_family"], columns="method", values=metric)
            delta = pivot[method] - pivot["b2_matched"]
            by_task = delta.groupby(level="mdp_id").mean().reindex(task_ids)
            if by_task.isna().any() or len(delta) != 70:
                raise RuntimeError("Missing paired task rows")
            samples = rng.choice(by_task.to_numpy(), size=(10000, 14), replace=True).mean(axis=1)
            paired.append({"algorithm": learner, "metric": metric,
                           "comparison_minus_b2": float(by_task.mean()),
                           "ci95_low": float(np.quantile(samples, .025)),
                           "ci95_high": float(np.quantile(samples, .975)),
                           "bootstrap_unit": "MDP task (five behaviors kept together)"})
    output.mkdir(parents=True)
    scored.to_csv(output / "scored.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(output / "paired_task_bootstrap.csv", index=False)
    scope = ("OEH đầy đủ 108 cột" if args.mode == "full"
             else "ablation OEH bỏ OEH-B trực tiếp; không suy ra hiệu năng của OEH đầy đủ")
    lines = [f"# BRIDGE {args.mode}: kết quả kiểm định độc lập\n",
             "14 MDP BRIDGE do bên thứ ba công bố; 70 log offline và nhãn do nhóm sinh theo giao thức đã khóa.\n",
             f"Phương pháp đối chiếu: B2 matched với {scope}.\n",
             "Brier/log loss càng thấp càng tốt; chênh lệch âm nghĩa là nhánh OEH có lỗi thấp hơn B2.\n",
             "## Điểm dự báo và quyết định budget\n",
             "| Learner | Method | Brier | Log loss | >x50 | Budget chọn TB | Đạt mục tiêu |\n",
             "|---|---|---:|---:|---:|---:|---:|\n"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.algorithm} | {row.method} | {row.brier:.6f} | "
                     f"{row.log_loss:.6f} | {row.censored_fraction:.3f} | "
                     f"{row.mean_chosen_budget:.2f} | {row.target_met:.3f} |\n")
    lines += ["\n## Chênh lệch ghép cặp theo MDP\n",
              "CI bootstrap 95% lấy mẫu lại 14 MDP, giữ năm behavior của mỗi MDP cùng nhau.\n",
              "| Learner | Metric | OEH − B2 | CI 95% |\n",
              "|---|---|---:|---:|\n"]
    for row in paired_frame.itertuples(index=False):
        lines.append(f"| {row.algorithm} | {row.metric} | {row.comparison_minus_b2:+.6f} | "
                     f"[{row.ci95_low:+.6f}; {row.ci95_high:+.6f}] |\n")
    lines += ["\nCensoring `>x50`, chi phí tính và mọi kết quả âm được giữ nguyên. "
              "TQ/SACo dùng MDP thật thuộc bảng oracle riêng; không xem điểm gốc của chúng "
              "là forecaster bảy lớp cùng quyền truy cập.\n"]
    (output / "RESULTS.md").write_text("".join(lines), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "status": f"COMPLETE_BRIDGE_INDEPENDENT_TEST_B2_VS_{args.mode.upper()}_OEH",
        "role": "third-party published MDPs; team-generated offline logs and labels",
        "source_manifest_sha256": source, "script_sha256": sha256(Path(__file__)),
        "tasks": 14, "behaviors_per_task": 5, "learners": 3,
        "rows_per_method": 210, "comparison": "matched seven-class requirement forecast",
        "comparison_mode": args.mode,
        "label_task_id_mapping": "exact one-to-one removal of _sticky25 adapter suffix",
        "output_sha256": {name: sha256(output / name) for name in
                          ("scored.csv", "summary.csv", "paired_task_bootstrap.csv", "RESULTS.md")},
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(paired_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

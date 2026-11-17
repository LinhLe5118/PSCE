#!/usr/bin/env python
"""Audit learner jobs and export reproducible Week 5 D2 result tables.

The command consumes a complete training artifact directory. It verifies the
artifact manifest, checks every expected job key for omissions and duplicates,
recomputes learning curves from run-level rows, and keeps finite ``N_emp``
statistics separate from the full population containing right censoring.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.io import read_csv_rows, reserve_output_dir, sha256_file, write_csv_rows, write_json


REGISTERED = ("cfqi_tabular", "iql_tabular")
BC = "behavior_cloning_tabular"
ALL_ALGORITHMS = (*REGISTERED, BC)
RUN_FILES = {
    "cfqi_tabular": "rl_run_results.csv",
    "iql_tabular": "rl_run_results.csv",
    BC: "bc_sanity_results.csv",
}


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    points = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "count": int(array.size),
        **dict(zip(("min", "q25", "median", "q75", "max"), (float(v) for v in points))),
    }


def _run_key(row: Mapping[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(row["mdp_id"]), str(row["policy_family"]), str(row["algorithm"]),
        int(row["training_size_multiplier"]), int(row["training_seed"]),
    )


def _label_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["mdp_id"]), str(row["policy_family"]), str(row["algorithm"])


def _verify_manifest(input_dir: Path, required_files: Sequence[str]) -> dict[str, str]:
    path = input_dir / "artifact_manifest.json"
    if not path.is_file():
        raise ValueError("A complete D2 audit requires artifact_manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", {})
    hashes: dict[str, str] = {}
    for name in required_files:
        source = input_dir / name
        if not source.is_file():
            raise ValueError(f"Missing training artifact: {name}")
        record = outputs.get(name)
        if not isinstance(record, dict) or "sha256" not in record:
            raise ValueError(f"Artifact manifest does not inventory {name}")
        digest = sha256_file(source)
        if digest != record["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
        hashes[name] = digest
    hashes["artifact_manifest.json"] = sha256_file(path)
    return hashes


def _curve_row(rows: Sequence[Mapping[str, Any]], algorithm: str, multiplier: int) -> dict[str, Any]:
    valid = [row for row in rows if str(row["run_status"]) == "observed"]
    invalid = len(rows) - len(valid)
    returns = [float(row["normalized_return"]) for row in valid]
    return {
        "algorithm": algorithm,
        "training_size_multiplier": multiplier,
        "valid_runs": len(valid),
        "invalid_runs": invalid,
        "mean_normalized_return": "" if not returns else float(np.mean(returns)),
        "median_normalized_return": "" if not returns else float(np.median(returns)),
        "success_rate": "" if not valid else sum(_as_bool(row["success"]) for row in valid) / len(valid),
    }


def _assert_close(recorded: Mapping[str, Any], computed: Mapping[str, Any]) -> None:
    key = (computed["algorithm"], computed["training_size_multiplier"])
    for field in ("valid_runs", "invalid_runs"):
        if int(recorded[field]) != int(computed[field]):
            raise ValueError(f"Learning-curve {field} disagrees with run rows for {key}")
    for field in ("mean_normalized_return", "median_normalized_return", "success_rate"):
        if not np.isclose(float(recorded[field]), float(computed[field]), rtol=1e-12, atol=1e-12):
            raise ValueError(f"Learning-curve {field} disagrees with run rows for {key}")


def analyse(input_dir: Path) -> dict[str, Any]:
    required = (
        "rl_run_results.csv", "bc_sanity_results.csv", "empirical_sample_complexity.csv",
        "learning_curve_summary.csv", "behavior_policy_metadata.csv", "offline_rl_summary.json",
    )
    source_hashes = _verify_manifest(input_dir, required)
    registered_rows = read_csv_rows(input_dir / "rl_run_results.csv")
    bc_rows = read_csv_rows(input_dir / "bc_sanity_results.csv")
    label_rows = read_csv_rows(input_dir / "empirical_sample_complexity.csv")
    recorded_curves = read_csv_rows(input_dir / "learning_curve_summary.csv")
    metadata = read_csv_rows(input_dir / "behavior_policy_metadata.csv")
    summary = json.loads((input_dir / "offline_rl_summary.json").read_text(encoding="utf-8"))
    sizes = tuple(int(value) for value in summary["training_sizes"])
    seeds = tuple(int(value) for value in summary["training_seeds"])
    groups = {(str(row["mdp_id"]), str(row["policy_family"])) for row in metadata}
    if len(groups) != len(metadata):
        raise ValueError("Duplicate MDP-policy rows in behavior_policy_metadata.csv")

    rows_by_algorithm = {
        algorithm: [row for row in registered_rows if row["algorithm"] == algorithm]
        for algorithm in REGISTERED
    }
    rows_by_algorithm[BC] = [row for row in bc_rows if row["algorithm"] == BC]
    unexpected = [row for row in registered_rows if row["algorithm"] not in REGISTERED]
    unexpected += [row for row in bc_rows if row["algorithm"] != BC]
    if unexpected:
        raise ValueError("Training artifacts contain an unexpected algorithm")

    job_audit: list[dict[str, Any]] = []
    grouped_runs: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for algorithm, rows in rows_by_algorithm.items():
        expected = {
            (mdp, family, algorithm, size, seed)
            for mdp, family in groups for size in sizes for seed in seeds
        }
        counts = Counter(_run_key(row) for row in rows)
        actual = set(counts)
        missing = expected - actual
        extra = actual - expected
        duplicates = {key: count for key, count in counts.items() if count > 1}
        invalid = sum(str(row["run_status"]) != "observed" for row in rows)
        job_audit.append({
            "algorithm": algorithm,
            "expected_jobs": len(expected),
            "observed_rows": len(rows),
            "unique_job_keys": len(actual),
            "missing_jobs": len(missing),
            "duplicate_job_keys": len(duplicates),
            "duplicate_rows": sum(count - 1 for count in duplicates.values()),
            "unexpected_jobs": len(extra),
            "invalid_jobs": invalid,
            "status": "pass" if not (missing or duplicates or extra or invalid) else "fail",
        })
        if missing or duplicates or extra or invalid:
            raise ValueError(
                f"Incomplete job grid for {algorithm}: missing={len(missing)}, "
                f"duplicate_keys={len(duplicates)}, unexpected={len(extra)}, invalid={invalid}"
            )
        for row in rows:
            grouped_runs[(algorithm, int(row["training_size_multiplier"]))].append(row)

    curve_lookup: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in recorded_curves:
        key = (str(row["algorithm"]), int(row["training_size_multiplier"]))
        if key in curve_lookup:
            raise ValueError(f"Duplicate learning-curve row: {key}")
        curve_lookup[key] = row
    expected_curve_keys = {(algorithm, size) for algorithm in ALL_ALGORITHMS for size in sizes}
    if set(curve_lookup) != expected_curve_keys:
        raise ValueError("Learning-curve table has missing or unexpected algorithm-size cells")
    curves: list[dict[str, Any]] = []
    for algorithm in ALL_ALGORITHMS:
        for size in sizes:
            computed = _curve_row(grouped_runs[(algorithm, size)], algorithm, size)
            _assert_close(curve_lookup[(algorithm, size)], computed)
            curves.append(computed)

    label_counts = Counter(_label_key(row) for row in label_rows)
    expected_labels = {(mdp, family, algorithm) for mdp, family in groups for algorithm in REGISTERED}
    if set(label_counts) != expected_labels or any(count != 1 for count in label_counts.values()):
        raise ValueError("N_emp labels have missing, duplicate, or unexpected keys")
    learning = summary["training_plan"]["protocol"]["learning"]
    required_successes = int(learning["success_runs_required"])
    if int(learning["success_runs_total"]) != len(seeds):
        raise ValueError("N_emp success denominator disagrees with the training seed grid")
    success_by_cell: dict[tuple[str, str, str, int], int] = Counter()
    for algorithm in REGISTERED:
        for row in rows_by_algorithm[algorithm]:
            key = (
                str(row["mdp_id"]), str(row["policy_family"]), algorithm,
                int(row["training_size_multiplier"]),
            )
            success_by_cell[key] += int(_as_bool(row["success"]))
    for row in label_rows:
        key = _label_key(row)
        computed_counts = [success_by_cell[(*key, size)] for size in sizes]
        recorded_counts = [int(value) for value in json.loads(str(row["success_counts"]))]
        if recorded_counts != computed_counts:
            raise ValueError(f"N_emp success_counts disagree with run rows for {key}")
        recorded_rates = [float(value) for value in json.loads(str(row["success_rates"]))]
        computed_rates = [count / len(seeds) for count in computed_counts]
        if not np.allclose(recorded_rates, computed_rates, rtol=1e-12, atol=1e-12):
            raise ValueError(f"N_emp success_rates disagree with run rows for {key}")
        stable = next(
            (
                size for index, size in enumerate(sizes)
                if all(count >= required_successes for count in computed_counts[index:])
            ),
            None,
        )
        recorded = None if str(row["n_emp_multiplier"]).strip() == "" else int(row["n_emp_multiplier"])
        censored = _as_bool(row["n_emp_censored"])
        if stable != recorded or censored != (stable is None):
            raise ValueError(f"N_emp is not the first stable success suffix for {key}")

    distributions: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    learner_rows: list[dict[str, Any]] = []
    largest = max(sizes)
    bc_curve = next(row for row in curves if row["algorithm"] == BC and row["training_size_multiplier"] == largest)
    for algorithm in REGISTERED:
        scoped = [row for row in label_rows if row["algorithm"] == algorithm]
        finite = [row for row in scoped if not _as_bool(row["n_emp_censored"])]
        finite_values = [int(row["n_emp_transitions"]) for row in finite]
        finite_multipliers = Counter(int(row["n_emp_multiplier"]) for row in finite)
        censored = len(scoped) - len(finite)
        for size in sizes:
            distributions.append({
                "algorithm": algorithm, "outcome": "finite", "n_emp_multiplier": size,
                "groups": finite_multipliers[size], "denominator_all_groups": len(scoped),
                "share_of_all_groups": finite_multipliers[size] / len(scoped),
            })
        distributions.append({
            "algorithm": algorithm, "outcome": "right_censored", "n_emp_multiplier": "",
            "groups": censored, "denominator_all_groups": len(scoped),
            "share_of_all_groups": censored / len(scoped),
        })
        for family in sorted({str(row["policy_family"]) for row in scoped}):
            family = str(family)
            subset = [row for row in scoped if row["policy_family"] == family]
            family_finite = [row for row in subset if not _as_bool(row["n_emp_censored"])]
            values = [int(row["n_emp_transitions"]) for row in family_finite]
            family_rows.append({
                "algorithm": algorithm, "policy_family": family, "total_groups": len(subset),
                "finite_groups": len(family_finite), "right_censored_groups": len(subset) - len(family_finite),
                "solved_group_rate_all": len(family_finite) / len(subset),
                "finite_median_n_emp": "" if not values else float(np.median(values)),
                "finite_median_denominator": len(values),
            })
        q = _quantiles(finite_values)
        final_curve = next(row for row in curves if row["algorithm"] == algorithm and row["training_size_multiplier"] == largest)
        learner_rows.append({
            "algorithm": algorithm, "role": "registered", "total_groups": len(scoped),
            "finite_groups": len(finite), "right_censored_groups": censored,
            "solved_group_rate_all": len(finite) / len(scoped), "censoring_rate_all": censored / len(scoped),
            "finite_n_emp_min": q["min"], "finite_n_emp_q25": q["q25"],
            "finite_n_emp_median": q["median"], "finite_n_emp_q75": q["q75"],
            "finite_n_emp_max": q["max"], "finite_n_emp_denominator": q["count"],
            "largest_multiplier": largest,
            "largest_prefix_median_return": final_curve["median_normalized_return"],
            "largest_prefix_success_rate": final_curve["success_rate"],
            "median_return_minus_bc": float(final_curve["median_normalized_return"]) - float(bc_curve["median_normalized_return"]),
            "success_rate_minus_bc": float(final_curve["success_rate"]) - float(bc_curve["success_rate"]),
        })
    learner_rows.append({
        "algorithm": BC, "role": "sanity_baseline", "total_groups": len(groups),
        "finite_groups": "", "right_censored_groups": "", "solved_group_rate_all": "",
        "censoring_rate_all": "", "finite_n_emp_min": "", "finite_n_emp_q25": "",
        "finite_n_emp_median": "", "finite_n_emp_q75": "", "finite_n_emp_max": "",
        "finite_n_emp_denominator": 0, "largest_multiplier": largest,
        "largest_prefix_median_return": bc_curve["median_normalized_return"],
        "largest_prefix_success_rate": bc_curve["success_rate"],
        "median_return_minus_bc": 0.0, "success_rate_minus_bc": 0.0,
    })
    return {
        "scope": summary.get("scope", "development training artifacts"),
        "source_hashes": source_hashes,
        "job_audit": job_audit,
        "learner_summary": learner_rows,
        "learning_curves": curves,
        "n_emp_distribution": distributions,
        "n_emp_by_family": family_rows,
        "n_emp_labels_verified": len(label_rows),
        "observations": [
            "Số job và các giá trị learning curve được tái tính từ dữ liệu mức job.",
            "Tỷ lệ nhóm giải được dùng toàn bộ nhóm; các phân vị N_emp hữu hạn chỉ dùng nhóm không bị censor và luôn ghi mẫu số.",
            "BC là baseline kiểm tra, không có nhãn N_emp đã đăng ký.",
        ],
        "explanatory_hypotheses": [
            "Đường cong CFQI tăng phù hợp với giả thuyết learner hưởng lợi từ thêm dữ liệu, nhưng chưa cô lập được cơ chế nhân quả.",
            "IQL bão hòa sớm hoặc bị censor nhiều có thể liên quan đến ngân sách tối ưu, expectile, behavior support hoặc tương tác giữa chúng; các artifact này chưa phân biệt được các nguyên nhân.",
            "Chênh lệch mô tả với BC không phải khoảng bất định ghép cặp hay kết luận về ưu thế.",
        ],
    }


def _markdown(report: Mapping[str, Any]) -> str:
    learners = report["learner_summary"]
    jobs = report["job_audit"]
    lines = [
        "# D2 — Learner Results & Baseline Analysis", "",
        f"Phạm vi: `{report['scope']}`.", "",
        "## Kiểm toán job", "",
        "| Thuật toán | Kỳ vọng | Dòng | Thiếu | Khóa trùng | Job lỗi | Trạng thái |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in jobs:
        lines.append(
            f"| `{row['algorithm']}` | {row['expected_jobs']} | {row['observed_rows']} | "
            f"{row['missing_jobs']} | {row['duplicate_job_keys']} | {row['invalid_jobs']} | {row['status']} |"
        )
    lines += [
        "", "## Tổng hợp learner", "",
        "| Thuật toán | Vai trò | Nhóm hữu hạn / toàn bộ | Censoring | Median N_emp hữu hạn (n) | Return median ở mức lớn nhất | Tỷ lệ run thành công |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in learners:
        if row["role"] == "registered":
            finite = f"{row['finite_groups']}/{row['total_groups']}"
            censoring = f"{float(row['censoring_rate_all']):.2%}"
            median = f"{float(row['finite_n_emp_median']):g} (n={row['finite_n_emp_denominator']})"
        else:
            finite, censoring, median = "không áp dụng", "không áp dụng", "không áp dụng"
        lines.append(
            f"| `{row['algorithm']}` | {row['role']} | {finite} | {censoring} | {median} | "
            f"{float(row['largest_prefix_median_return']):.6f} | {float(row['largest_prefix_success_rate']):.2%} |"
        )
    lines += ["", "## Quan sát thực nghiệm", ""]
    lines += [f"- {value}" for value in report["observations"]]
    lines += ["", "## Giả thuyết giải thích", ""]
    lines += [f"- {value}" for value in report["explanatory_hypotheses"]]
    lines += [
        "", "## Tái tạo và truy nguyên", "",
        "Các bảng CSV trong thư mục này được tái tính từ dữ liệu mức job. `provenance.json` ghi SHA-256 của từng đầu vào.", "",
    ]
    if report.get("source_archive"):
        lines += [
            f"Gói nguồn: `{report['source_archive']['path']}`.", "",
            "```powershell",
            "Expand-Archive artifacts\\week5\\source_bundles\\full_pipeline_results_2026-09-14.zip -DestinationPath <temporary-dir>",
            "python -m scripts.analysis.analyze_learner_results --input-dir <temporary-dir>\\artifacts\\week5\\full_development_pipeline_2026-09-14_181522\\training --output-dir <new-output-dir>",
            "```", "",
        ]
    else:
        lines += [
            "```powershell",
            "python -m scripts.analysis.analyze_learner_results --input-dir <training-artifact-dir> --output-dir <new-output-dir>",
            "```", "",
        ]
    return "\n".join(lines)


def run(input_dir: Path, output_dir: Path, *, source_archive: Path | None = None) -> dict[str, Any]:
    report = analyse(input_dir)
    if source_archive is not None:
        report["source_archive"] = {
            "path": source_archive.as_posix(),
            "sha256": sha256_file(source_archive),
        }
    reserve_output_dir(output_dir)
    tables = {
        "job_audit.csv": report["job_audit"],
        "learner_summary.csv": report["learner_summary"],
        "learning_curves.csv": report["learning_curves"],
        "n_emp_distribution.csv": report["n_emp_distribution"],
        "n_emp_by_family.csv": report["n_emp_by_family"],
    }
    for name, rows in tables.items():
        write_csv_rows(output_dir / name, rows, tuple(rows[0]))
    write_json(output_dir / "analysis.json", report)
    provenance_inputs = dict(report["source_hashes"])
    if source_archive is not None:
        provenance_inputs[source_archive.as_posix()] = sha256_file(source_archive)
    provenance = {
        "inputs": provenance_inputs,
        "outputs": {name: sha256_file(output_dir / name) for name in (*tables, "analysis.json")},
        "generator": "scripts/analysis/analyze_learner_results.py",
        "generator_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / "provenance.json", provenance)
    (output_dir / "README.md").write_text(_markdown(report), encoding="utf-8")
    return report


def run_available_snapshot(analysis_path: Path, output_dir: Path) -> dict[str, Any]:
    """Export what can be checked when the committed raw run is unavailable."""
    source = json.loads(analysis_path.read_text(encoding="utf-8"))
    sizes = tuple(int(value) for value in source["learning_curves"]["cfqi_tabular"]["multipliers"])
    learners = source["training"]
    total_groups = {algorithm: int(values["count"]) for algorithm, values in learners.items()}
    seed_count = 10
    job_rows = [
        {
            "algorithm": algorithm,
            "expected_jobs": total_groups.get(algorithm, total_groups["cfqi_tabular"]) * len(sizes) * seed_count,
            "observed_rows": "",
            "missing_jobs": "",
            "duplicate_job_keys": "",
            "invalid_jobs": "",
            "status": "blocked_raw_bundle_missing",
        }
        for algorithm in ALL_ALGORITHMS
    ]
    curve_rows: list[dict[str, Any]] = []
    for algorithm, values in source["learning_curves"].items():
        for index, multiplier in enumerate(values["multipliers"]):
            curve_rows.append({
                "algorithm": algorithm,
                "training_size_multiplier": multiplier,
                "median_normalized_return": values["median_normalized_returns"][index],
                "success_rate": values["success_rates"][index],
                "verification": "committed_summary_only",
                "source_field": f"learning_curves.{algorithm}",
            })
    largest = max(sizes)
    final = {
        row["algorithm"]: row for row in curve_rows
        if int(row["training_size_multiplier"]) == largest
    }
    bc = final[BC]
    learner_rows: list[dict[str, Any]] = []
    for algorithm in REGISTERED:
        values = learners[algorithm]
        curve = final[algorithm]
        learner_rows.append({
            "algorithm": algorithm,
            "role": "registered",
            "total_groups": values["count"],
            "finite_groups": values["finite_count"],
            "right_censored_groups": values["right_censored"],
            "solved_group_rate_all": values["finite_count"] / values["count"],
            "censoring_rate_all": values["right_censored"] / values["count"],
            "finite_n_emp_min": values["n_emp_min"],
            "finite_n_emp_median": values["n_emp_median"],
            "finite_n_emp_max": values["n_emp_max"],
            "finite_n_emp_denominator": values["finite_count"],
            "largest_multiplier": largest,
            "largest_prefix_median_return": curve["median_normalized_return"],
            "largest_prefix_success_rate": curve["success_rate"],
            "median_return_minus_bc": float(curve["median_normalized_return"]) - float(bc["median_normalized_return"]),
            "success_rate_minus_bc": float(curve["success_rate"]) - float(bc["success_rate"]),
            "verification": "committed_summary_only",
        })
    learner_rows.append({
        "algorithm": BC,
        "role": "sanity_baseline",
        "total_groups": total_groups["cfqi_tabular"],
        "finite_groups": "",
        "right_censored_groups": "",
        "solved_group_rate_all": "",
        "censoring_rate_all": "",
        "finite_n_emp_min": "",
        "finite_n_emp_median": "",
        "finite_n_emp_max": "",
        "finite_n_emp_denominator": 0,
        "largest_multiplier": largest,
        "largest_prefix_median_return": bc["median_normalized_return"],
        "largest_prefix_success_rate": bc["success_rate"],
        "median_return_minus_bc": 0.0,
        "success_rate_minus_bc": 0.0,
        "verification": "committed_summary_only",
    })
    family_rows = []
    for algorithm in REGISTERED:
        values = learners[algorithm]
        family_total = int(values["count"]) // len(values["per_family_censored"])
        for family, censored in sorted(values["per_family_censored"].items()):
            family_rows.append({
                "algorithm": algorithm,
                "policy_family": family,
                "total_groups": family_total,
                "finite_groups": family_total - int(censored),
                "right_censored_groups": censored,
                "solved_group_rate_all": (family_total - int(censored)) / family_total,
                "verification": "committed_summary_only",
            })
    report = {
        "acceptance_status": "blocked_raw_bundle_missing",
        "scope": source["scope"],
        "source_analysis": str(analysis_path),
        "source_analysis_sha256": sha256_file(analysis_path),
        "raw_inputs_named_by_source": source.get("input_sha256", {}),
        "raw_inputs_present": False,
        "missing_checks": [
            "Observed, missing, duplicate and invalid job counts cannot be independently recomputed.",
            "The finite N_emp distribution by multiplier and its quartiles cannot be reconstructed.",
            "Learning-curve aggregates cannot be reconciled against run-level rows.",
        ],
        "learner_summary": learner_rows,
        "learning_curves": curve_rows,
        "n_emp_by_family": family_rows,
        "job_audit": job_rows,
    }
    reserve_output_dir(output_dir)
    tables = {
        "job_audit.csv": job_rows,
        "learner_summary.csv": learner_rows,
        "learning_curves.csv": curve_rows,
        "n_emp_by_family.csv": family_rows,
    }
    for name, rows in tables.items():
        write_csv_rows(output_dir / name, rows, tuple(rows[0]))
    write_json(output_dir / "availability.json", report)
    write_json(output_dir / "provenance.json", {
        "input": {analysis_path.as_posix(): sha256_file(analysis_path)},
        "outputs": {name: sha256_file(output_dir / name) for name in (*tables, "availability.json")},
        "generator": "scripts/analysis/analyze_learner_results.py",
        "generator_sha256": sha256_file(Path(__file__)),
    })
    (output_dir / "README.md").write_text(
        "# D2 — Learner Results & Baseline Analysis\n\n"
        "**Trạng thái nghiệm thu: bị chặn vì thiếu bộ artifact training mức job.**\n\n"
        "Các bảng hiện tại được tái tạo từ `analysis.json` đã commit. Chúng đủ để kiểm tra "
        "số nhóm hữu hạn/censored, đường cong tổng hợp và chênh lệch mô tả với BC. Chúng "
        "không đủ để kiểm toán độc lập số job thiếu/trùng hoặc tái tính phân phối `N_emp`.\n\n"
        "## Quan sát thực nghiệm\n\n"
        "- `learner_summary.csv` tách tỷ lệ giải được trên toàn bộ nhóm khỏi median `N_emp` chỉ trên nhóm hữu hạn.\n"
        "- `learning_curves.csv` giữ CFQI, IQL và BC ở từng mức dữ liệu.\n"
        "- `n_emp_by_family.csv` giữ số nhóm censored theo từng họ chính sách hành vi.\n\n"
        "## Giả thuyết giải thích\n\n"
        "- CFQI tăng theo lượng dữ liệu phù hợp với giả thuyết learner hưởng lợi từ thêm mẫu; bảng này không chứng minh cơ chế nhân quả.\n"
        "- IQL bão hòa sớm có thể do ngân sách cập nhật, expectile, behavior support hoặc tương tác giữa chúng; cần ablation để phân biệt.\n"
        "- Chênh lệch với BC chỉ là thống kê mô tả, không phải kiểm định ưu thế.\n\n"
        "## Đầu vào còn thiếu\n\n"
        "Cần thư mục `full_development_pipeline_2026-09-14_181522/training/` chứa "
        "`artifact_manifest.json` và sáu tệp được công cụ kiểm toán yêu cầu. Sau khi nhận, chạy:\n\n"
        "```powershell\n"
        "python -m scripts.analysis.analyze_learner_results --input-dir <training-dir> --output-dir <new-output-dir>\n"
        "```\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-dir", type=Path)
    inputs.add_argument("--analysis-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    if args.input_dir:
        report = run(args.input_dir, args.output_dir, source_archive=args.source_archive)
        status = "pass"
    else:
        if args.source_archive is not None:
            parser.error("--source-archive can only be used with --input-dir")
        report = run_available_snapshot(args.analysis_json, args.output_dir)
        status = report["acceptance_status"]
    print(json.dumps({"status": status, "job_audit": report["job_audit"]}, indent=2))


if __name__ == "__main__":
    main()

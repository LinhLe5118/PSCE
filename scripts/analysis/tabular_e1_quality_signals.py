"""Extract published dataset-quality ingredients and declared tabular adaptations.

This CLI uses only the original E1 x10 pilot buffers. It does not reproduce
SACo/TQ reference normalizations, continuous-action EAS, CNN, or BWD. Undefined
ratios remain missing rather than silently filled or tuned to E1 outcomes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets/week6_e1_oehv4e_confirmatory_v1_1"
OUTPUT = ROOT / "artifacts/tabular_e1_quality_signals_2026-10-01"
SCORING = ROOT / "artifacts/tabular_e1_quality_scoring_2026-10-01"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def pilot_signals(arrays: dict[str, np.ndarray], num_states: int,
                  num_actions: int, gamma: float = .95) -> dict:
    """Return data-only ingredients; incomplete prefix trajectories are excluded."""
    keys = ("state", "action", "reward", "trajectory_id", "timestep", "done")
    if any(key not in arrays for key in keys):
        raise ValueError("Missing pilot field")
    values = {key: np.asarray(arrays[key]) for key in keys}
    length = len(values["state"])
    if length == 0 or any(len(value) != length for value in values.values()):
        raise ValueError("Misaligned pilot")
    state = values["state"].astype(int)
    action = values["action"].astype(int)
    reward = values["reward"].astype(float)
    trajectory = values["trajectory_id"].astype(int)
    timestep = values["timestep"].astype(int)
    done = values["done"].astype(bool)
    if (num_states < 1 or num_actions < 2 or np.any(state < 0) or np.any(state >= num_states)
            or np.any(action < 0) or np.any(action >= num_actions) or
            np.any(timestep < 0) or not np.isfinite(reward).all()):
        raise ValueError("Invalid pilot values")
    counts = np.zeros((num_states, num_actions), dtype=int)
    np.add.at(counts, (state, action), 1)
    unique_sa = int(np.count_nonzero(counts))
    states = counts.sum(axis=1)
    probabilities = np.divide(counts, states[:, None],
                              out=np.zeros_like(counts, dtype=float),
                              where=states[:, None] > 0)
    positive = probabilities > 0
    entropy_by_state = -np.sum(np.where(positive, probabilities *
                                        np.log(np.maximum(probabilities, 1e-300)), 0.), axis=1)
    categorical_eas = float(np.dot(states / length, entropy_by_state) / math.log(num_actions))
    ids, inverse = np.unique(trajectory, return_inverse=True)
    returns = np.zeros(len(ids))
    completed = np.zeros(len(ids), dtype=bool)
    np.add.at(returns, inverse, reward * np.power(gamma, timestep))
    np.logical_or.at(completed, inverse, done)
    full = returns[completed]
    if len(full):
        mean = float(np.mean(full))
        highest = float(np.max(full))
        floor = float(np.min(full))
        shifted_mean = mean - floor
        eri = float((highest - mean) / shifted_mean) if shifted_mean > 0 else None
    else:
        mean = highest = floor = eri = None
    return {
        "observed_unique_sa": unique_sa,
        "sa_fraction_adaptation": unique_sa / (num_states * num_actions),
        "complete_trajectory_count": int(np.sum(completed)),
        "incomplete_trajectory_count": int(np.sum(~completed)),
        "mean_complete_discounted_return": mean,
        "max_complete_discounted_return": highest,
        "min_complete_discounted_return": floor,
        "eri_observed_floor_adaptation": eri,
        "categorical_eas_adaptation": categorical_eas,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(output: Path = OUTPUT) -> None:
    if output.exists():
        raise FileExistsError(f"Existing immutable output: {output}")
    manifest_path = DATASET / "manifest.json"
    seal = json.loads((ROOT / "artifacts/week6/E1_INPUT_SEAL_2026-09-26.json").read_text(encoding="utf-8"))
    if digest(manifest_path) != seal["manifest_sha256"]:
        raise RuntimeError("Original E1 manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [row for row in manifest["datasets"] if
               row.get("benchmark_split") == "evaluation" and
               row.get("stream") == "pilot" and int(row.get("size_multiplier", -1)) == 10]
    if len(records) != 600 or len({(row["instance_id"], row["policy_family"]) for row in records}) != 600:
        raise RuntimeError("Incomplete E1 x10 pilot grid")
    output.mkdir(parents=True)
    rows = []
    for index, record in enumerate(sorted(records, key=lambda row: (row["instance_id"], row["policy_family"])), 1):
        source = DATASET / record["file"]
        if digest(source) != record["sha256"]:
            raise RuntimeError(f"E1 pilot buffer hash mismatch: {record['file']}")
        with np.load(source, allow_pickle=False) as loaded:
            arrays = {key: loaded[key][:int(record["size"])] for key in loaded.files}
        item = pilot_signals(arrays, int(record["num_states"]), int(record["num_actions"]))
        rows.append({"mdp_id": record["instance_id"], "policy_family": record["policy_family"],
                     "pilot_buffer_sha256": record["sha256"], **item})
        if index % 100 == 0:
            print(f"Extracted {index}/600 existing E1 pilots", flush=True)
    # COI ranks five datasets jointly within each MDP, which is extra access.
    by_mdp = defaultdict(list)
    for row in rows:
        by_mdp[row["mdp_id"]].append(row)
    if len(by_mdp) != 120 or any(len(group) != 5 for group in by_mdp.values()):
        raise RuntimeError("Incomplete five-behavior blocks")
    from scipy.stats import rankdata
    for group in by_mdp.values():
        finite = all(row["eri_observed_floor_adaptation"] is not None for row in group)
        if finite:
            eri_rank = rankdata([row["eri_observed_floor_adaptation"] for row in group], method="average")
            eas_rank = rankdata([row["categorical_eas_adaptation"] for row in group], method="average")
            for row, eri, eas in zip(group, eri_rank, eas_rank):
                row["coi_tabular_within_mdp_adaptation"] = float((2 * eri + eas) / 3)
        else:
            for row in group:
                row["coi_tabular_within_mdp_adaptation"] = None
    output_file = output / "quality_signals.csv"
    _write_csv(output_file, rows)
    summary = {
        "status": "complete", "scope": "E1 pilot-only post-hoc tabular adaptations; no exact TQ/SACo/EAS/BWD/CNN claim",
        "row_count": len(rows), "mdp_count": len(by_mdp),
        "undefined_eri_count": sum(row["eri_observed_floor_adaptation"] is None for row in rows),
        "undefined_coi_count": sum(row["coi_tabular_within_mdp_adaptation"] is None for row in rows),
        "partial_trajectories_total": sum(row["incomplete_trajectory_count"] for row in rows),
        "input_manifest_sha256": digest(manifest_path),
        "source_sha256": digest(Path(__file__)), "output_sha256": digest(output_file),
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def score(signals_dir: Path = OUTPUT, output: Path = SCORING) -> None:
    """Score data-only adaptations on the original grouped E1 folds, post-hoc."""
    if output.exists():
        raise FileExistsError(f"Existing immutable score output: {output}")
    import pandas as pd
    from scipy.special import ndtr
    import xgboost as xgb
    from scripts.analysis.compare_difficulty import GRID, aft_interval

    signal_path = signals_dir / "quality_signals.csv"
    signal_manifest = json.loads((signals_dir / "manifest.json").read_text(encoding="utf-8"))
    if digest(signal_path) != signal_manifest["output_sha256"] or signal_manifest["row_count"] != 600:
        raise RuntimeError("Pilot-only quality signal seal changed")
    label_path = ROOT / "artifacts/difficulty_e1_posthoc_2026-09-30/labels.csv"
    label_manifest = json.loads((label_path.parent / "manifest.json").read_text(encoding="utf-8"))
    if digest(label_path) != label_manifest["output_hashes"]["labels.csv"]:
        raise RuntimeError("E1 requirement labels changed")
    labels = pd.read_csv(label_path)
    signals = pd.read_csv(signal_path)
    frame = labels.merge(signals, on=["mdp_id", "policy_family"], validate="many_to_one")
    if len(frame) != 1800 or frame.mdp_id.nunique() != 120 or frame.duplicated(["mdp_id", "policy_family", "algorithm"]).any():
        raise RuntimeError("Incomplete original E1 learner/behavior panel")
    methods = {
        "unique_sa_count": ("observed_unique_sa",),
        "sa_fraction_adaptation": ("sa_fraction_adaptation",),
        "complete_return_adaptation": ("mean_complete_discounted_return",),
        "eri_observed_floor_adaptation": ("eri_observed_floor_adaptation",),
        "categorical_eas_adaptation": ("categorical_eas_adaptation",),
        "coi_tabular_within_mdp_adaptation": ("coi_tabular_within_mdp_adaptation",),
        "combined_tabular_quality_adaptation": (
            "sa_fraction_adaptation", "mean_complete_discounted_return",
            "eri_observed_floor_adaptation", "categorical_eas_adaptation"),
    }
    params = {
        "objective": "survival:aft", "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.2,
        "tree_method": "hist", "learning_rate": .05, "max_depth": 2,
        "min_child_weight": 5, "lambda": 1, "seed": 20260930,
        "nthread": 1, "verbosity": 0,
    }
    outcomes = frame.event.to_numpy(dtype=int)
    predictions = {name: np.full((len(frame), 7), np.nan) for name in methods}
    for panel in ("primary", "control"):
        for learner in sorted(frame.algorithm.unique()):
            group = (frame.panel == panel) & (frame.algorithm == learner)
            for fold in range(5):
                train = np.flatnonzero(group & (frame.fold != fold))
                test = np.flatnonzero(group & (frame.fold == fold))
                if not len(train) or not len(test) or set(frame.iloc[train].mdp_id) & set(frame.iloc[test].mdp_id):
                    raise RuntimeError("Invalid MDP-grouped fold")
                lower, upper = np.asarray([aft_interval(int(value)) for value in outcomes[train]], dtype=float).T
                for method, columns in methods.items():
                    x = frame.iloc[train][list(columns)].to_numpy(dtype=float)
                    q = frame.iloc[test][list(columns)].to_numpy(dtype=float)
                    if np.isinf(x).any() or np.isinf(q).any():
                        raise RuntimeError(f"Infinite feature: {method}")
                    matrix = xgb.DMatrix(x, missing=np.nan)
                    matrix.set_float_info("label_lower_bound", lower)
                    matrix.set_float_info("label_upper_bound", upper)
                    fitted = xgb.train(params, matrix, num_boost_round=200)
                    median = fitted.predict(xgb.DMatrix(q, missing=np.nan))
                    if not np.isfinite(median).all() or (median <= 0).any():
                        raise RuntimeError(f"Invalid AFT prediction: {method}")
                    cdf = ndtr((np.log(np.asarray(GRID))[None, :] - np.log(median)[:, None]) / 1.2)
                    categories = np.maximum(np.column_stack((cdf[:, 0], np.diff(cdf, axis=1), 1 - cdf[:, -1])), 0.)
                    predictions[method][test] = categories / categories.sum(axis=1, keepdims=True)
    detail, summary = [], []
    for panel in ("primary", "control"):
        indices = np.flatnonzero(frame.panel == panel)
        for method, columns in methods.items():
            prob = predictions[method][indices]
            event = outcomes[indices]
            if not np.isfinite(prob).all() or not np.allclose(prob.sum(axis=1), 1):
                raise RuntimeError(f"Incomplete E1 probability grid: {method}")
            actual = (event[:, None] <= np.arange(6)[None, :]).astype(float)
            brier = ((prob.cumsum(axis=1)[:, :6] - actual) ** 2).mean(axis=1)
            logloss = -np.log(np.maximum(prob[np.arange(len(event)), event], 1e-12))
            summary.append({"panel": panel, "method": method, "n": len(indices),
                            "missing_feature_fraction": float(frame.iloc[indices][list(columns)].isna().to_numpy().mean()),
                            "integrated_brier": float(np.mean(brier)),
                            "seven_class_log_loss": float(np.mean(logloss)),
                            "event_category_accuracy": float(np.mean(prob.argmax(axis=1) == event))})
            for local, index in enumerate(indices):
                item = frame.iloc[index]
                detail.append({"mdp_id": item.mdp_id, "policy_family": item.policy_family,
                               "algorithm": item.algorithm, "panel": panel, "fold": int(item.fold),
                               "event": int(event[local]), "method": method,
                               "integrated_brier": float(brier[local]),
                               "seven_class_log_loss": float(logloss[local]),
                               **{f"p_{category}": float(prob[local, category]) for category in range(7)}})
    output.mkdir(parents=True)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "predictions.csv", detail)
    manifest = {
        "status": "posthoc_original_E1_adaptively_reused_not_confirmation",
        "scope": "Published quality ingredients and explicitly named tabular adaptations, not exact TQ/SACo/EAS/COI/CNN/BWD",
        "mdp_count": 120, "label_rows": 1800, "folds": "original MDP-grouped five folds",
        "features": {key: list(value) for key, value in methods.items()},
        "xgboost_params": params, "rounds": 200,
        "input_hashes": {"quality_signals.csv": digest(signal_path), "labels.csv": digest(label_path)},
        "source_sha256": digest(Path(__file__)),
        "output_hashes": {name: digest(output / name) for name in ("summary.csv", "predictions.csv")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output)
    elif args.output == OUTPUT:
        score()
    else:
        score(output=args.output)


if __name__ == "__main__":
    main()

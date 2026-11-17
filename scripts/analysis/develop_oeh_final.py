"""Development-only posterior-predictive OEH (OEH-B) on opened cohorts.

The declared arm grid lives in
``artifacts/oeh_final_development_2026-10-01/registry.jsonl``. Each arm changes
one structural axis of the hyperparameter-free posterior. Every group result is
written once and verified on resume; scoring reads only completed groups after
all groups of an arm exist. Cohorts used here are already opened and cannot
confirm anything.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/oeh_final_development_2026-10-01"
COHORTS = {
    "third120": ROOT / "datasets/oeh_cross_capacity_unseen_120_2026-10-01",
    "second120": ROOT / "datasets/oeh_pooled_successor_unseen_120_2026-10-01",
    "first120": ROOT / "datasets/oeh_difficulty_unseen_120_2026-10-01",
    "first60": ROOT / "datasets/independent_validation_2026-09-30",
    "second60": ROOT / "datasets/oeh_difficulty_confirmation_2026-10-01",
    "e1": ROOT / "datasets/week6_e1_oehv4e_confirmatory_v1_1",
}
ARMS = {
    "D1": {"reward_prior": "exchangeable", "transition_base": "uniform"},
    "D2": {"reward_prior": "unseen_zero", "transition_base": "uniform"},
    "D3": {"reward_prior": "exchangeable", "transition_base": "pooled"},
    "D4": {"reward_prior": "unseen_zero", "transition_base": "pooled"},
}
DRAWS = 64


def pilot_groups(dataset: Path) -> list[dict]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest["datasets"]
            if row.get("stream") == "pilot" and int(row.get("size_multiplier", -1)) == 10
            and row.get("benchmark_split", "evaluation") == "evaluation"]
    rows.sort(key=lambda row: (row["instance_id"], row["policy_family"]))
    keys = [(row["instance_id"], row["policy_family"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Duplicate pilot group")
    return rows


def group_seed(cohort: str, arm: str, row: dict) -> list[int]:
    digest = hashlib.sha256(f"oeh-final-dev|{cohort}|{row['instance_id']}|{row['policy_family']}".encode()).digest()
    return [int.from_bytes(digest[:8], "little"), int(arm[1:])]


def _claim(path: Path) -> bool:
    """Atomically reserve one group so CPU and GPU runs can share an arm."""
    try:
        os.close(os.open(str(path.with_suffix(".claim")), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def run_group(cohort: str, arm: str, row: dict, dataset: str, path: str, backend: str = "cpu") -> str:
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from oeh import predictive

    if Path(path).exists() or not _claim(Path(path)):
        return path
    batched = None
    if backend == "gpu":
        from scripts.analysis import oeh_final_gpu
        batched = oeh_final_gpu.learners()
    tick = time.perf_counter()
    with np.load(Path(dataset) / row["file"], allow_pickle=False) as loaded:
        arrays = {name: loaded[name][:int(row["size"])] for name in loaded.files}
    outcomes = predictive.posterior_outcomes(
        arrays, row["initial_state_distribution"], num_states=int(row["num_states"]),
        num_actions=int(row["num_actions"]), gamma=0.95, draws=DRAWS,
        seed=group_seed(cohort, arm, row), batched=batched, **ARMS[arm])
    forecast = predictive.requirement_forecast(outcomes)
    nr = outcomes.normalized_return
    document = {
        "cohort": cohort, "arm": arm, "mdp_id": row["instance_id"],
        "policy_family": row["policy_family"], "pilot_sha256": row["sha256"],
        "posterior": outcomes.posterior, "sampling": outcomes.sampling,
        "learners": {name: {
            "probabilities": value["probabilities"].tolist(),
            "events": value["events"].tolist(),
            "median_event": value["median_event"],
            "censored_fraction": value["censored_fraction"],
            "mean_normalized_return": (nr[:, :, :, index].mean((0, 1)).tolist() if len(nr) else None),
            "success_rate": ((nr[:, :, :, index] >= 0.95).mean((0, 1)).tolist() if len(nr) else None),
        } for index, (name, value) in enumerate(forecast.items())},
        "runtime_seconds": time.perf_counter() - tick,
        "backend": backend,
    }
    temporary = Path(path).with_suffix(".writing")
    temporary.write_text(json.dumps(document), encoding="utf-8")
    temporary.replace(path)
    Path(path).with_suffix(".claim").unlink(missing_ok=True)
    return path


def compute(cohort: str, arm: str, workers: int, backend: str = "cpu") -> None:
    dataset = COHORTS[cohort]
    target = OUTPUT / "groups" / cohort / arm
    target.mkdir(parents=True, exist_ok=True)
    rows = pilot_groups(dataset)
    pending = []
    for row in rows:
        name = hashlib.sha256(f"{row['instance_id']}|{row['policy_family']}".encode()).hexdigest()[:24]
        path = target / f"{name}.json"
        if path.with_suffix(".claim").exists():
            continue
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            if document["pilot_sha256"] != row["sha256"] or document["arm"] != arm:
                raise RuntimeError(f"Stale development result: {path}")
            continue
        pending.append((row, path))
    print(f"{cohort}/{arm}: {len(rows) - len(pending)} done, {len(pending)} pending", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_group, cohort, arm, row, str(dataset), str(path), backend)
                   for row, path in pending]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 25 == 0 or done == len(futures):
                print(f"{cohort}/{arm}: {done}/{len(futures)} in {time.perf_counter() - started:.0f}s", flush=True)


LABELS = {
    "third120": ROOT / "artifacts/oeh_cross_capacity_unseen_120_2026-10-01",
    "second120": ROOT / "artifacts/oeh_pooled_successor_unseen_120_2026-10-01",
    "first120": ROOT / "artifacts/oeh_difficulty_unseen_120_2026-10-01",
}


def predictions(cohort: str, arm: str):
    """Forecast rows of one completed arm in the protocol's column layout."""
    import pandas as pd

    rows = []
    for path in sorted((OUTPUT / "groups" / cohort / arm).glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for learner, value in document["learners"].items():
            rows.append({"mdp_id": document["mdp_id"], "policy_family": document["policy_family"],
                         "algorithm": learner, "method": f"oeh_b_{arm}",
                         **{f"p_{i}": p for i, p in enumerate(value["probabilities"])}})
    frame = pd.DataFrame(rows)
    if len(frame) != 1800:
        raise RuntimeError(f"{cohort}/{arm} incomplete: {len(frame)} of 1800 rows")
    return frame


def score(cohort: str, arms: list[str]) -> None:
    import pandas as pd
    from scripts.analysis.develop_oeh_difficulty import score_probabilities

    source = LABELS[cohort]
    labels = pd.read_csv(source / "labels.csv")
    frozen = pd.read_csv(source / "scored_predictions.csv")
    keys = ["mdp_id", "policy_family", "algorithm"]
    P = [f"p_{i}" for i in range(7)]
    frames = [frozen[keys + ["method", "panel", "event"] + P]]
    for arm in arms:
        frame = predictions(cohort, arm).merge(labels[keys + ["panel", "event"]], on=keys,
                                                validate="one_to_one")
        frames.append(frame)
    scored = pd.concat(frames, ignore_index=True)
    brier, logloss = score_probabilities(scored[P].to_numpy(float), scored.event.to_numpy(int))
    scored["integrated_brier"], scored["seven_class_log_loss"] = brier, logloss
    mdp = scored.groupby(["mdp_id", "panel", "method"], as_index=False)[
        ["integrated_brier", "seven_class_log_loss"]].mean()
    summary = mdp.groupby(["panel", "method"], as_index=False)[
        ["integrated_brier", "seven_class_log_loss"]].mean().sort_values(["panel", "seven_class_log_loss"])
    calibration = scored.assign(censored_forecast=scored.p_6, censored=(scored.event == 6)).groupby(
        ["panel", "method"], as_index=False)[["censored_forecast", "censored"]].mean()
    summary = summary.merge(calibration, on=["panel", "method"])
    out = OUTPUT / f"scores_{cohort}.csv"
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compute", "score"))
    parser.add_argument("cohort", choices=sorted(COHORTS))
    parser.add_argument("arms", nargs="+", choices=sorted(ARMS))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    if args.command == "score":
        score(args.cohort, args.arms)
        return
    for arm in args.arms:
        compute(args.cohort, arm, args.workers, args.backend)


if __name__ == "__main__":
    main()

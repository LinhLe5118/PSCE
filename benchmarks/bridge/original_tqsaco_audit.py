"""Checkpointed exploratory TQ/SACo audit using the authors' DQN reference.

The formulas and DQN implementation come from ml-jku/OfflineRL at commit
8cc8eec. The tabular one-hot BRIDGE environment is an adapter. References
read the published true MDP, so these scores have privileged access; they are
not seven-class offline data-requirement forecasts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "datasets/bridge_v1_1_1/offline_pilot_preoutcome_v1"
LABELS = ROOT / "artifacts/week8/bridge_reduced_test_labels_2026-10-07"
OUT = ROOT / "artifacts/week8/bridge_original_tqsaco_exploratory_2026-10-08"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def task_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"bridge-tqsaco-author|{name}".encode()).digest()[:4], "little") % (2**31 - 1)


def task_path(name: str) -> Path:
    return OUT / "checkpoints" / f"{name}.json"


def run_task(name: str, rows: list[dict]) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from benchmarks.bridge.adapter import load_sticky_mdp
    from scripts.analysis.oeh_final_published import episode_returns
    from scripts.analysis.oeh_final_tqsaco_original import (
        TabularEnv, get_returns, online_reference, unique_state_actions_exact,
    )

    t0 = time.perf_counter()
    mdp = load_sticky_mdp(name)
    n = int(rows[0]["max_size"])
    if len(rows) != 5 or any(int(row["max_size"]) != n for row in rows):
        raise RuntimeError(f"Incomplete pilot suite: {name}")
    seed = task_seed(name)
    online_max, online_pairs, trace = online_reference(mdp, n, seed)
    env = TabularEnv(mdp)
    env.seed(seed + 1)
    rewards, dones, done = [], [], True
    for _ in range(n):
        if done:
            env.reset()
        _, reward, done, _ = env.step(int(env.rng.integers(mdp.num_actions)))
        rewards.append(reward)
        dones.append(done)
    random_return = float(get_returns(np.asarray(rewards), np.asarray(dones)).mean())
    denominator = online_max - random_return
    scores = []
    for row in rows:
        path = PILOT / row["file"]
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"Pilot hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if len(arrays["state"]) != n:
            raise RuntimeError(f"Pilot length mismatch: {path}")
        mean_return = float(episode_returns(arrays).mean())
        one_hot = np.eye(mdp.num_states, dtype=np.float32)[arrays["state"]]
        pairs = unique_state_actions_exact(one_hot, arrays["action"])
        scores.append({"mdp_id": name, "policy_family": row["policy_family"],
                       "tq_original": (mean_return - random_return) / denominator if denominator > 0 else None,
                       "tq_defined": denominator > 0,
                       "saco_original": pairs / online_pairs if online_pairs > 0 else None,
                       "pilot_mean_return": mean_return, "pilot_pairs": pairs})
    result = {"status": "COMPLETE_EXPLORATORY_ORIGINAL_FORMULA_AUTHORS_DQN_REFERENCE",
              "task": name, "seed": seed, "states": mdp.num_states, "actions": mdp.num_actions,
              "transitions": n, "online_reference_max_return": online_max,
              "online_reference_unique_pairs": online_pairs,
              "online_reference_evaluations": len(trace),
              "random_reference_mean_return": random_return,
              "tq_denominator": denominator,
              "source_mdp_sha256": mdp.metadata["source_sha256"],
              "pilot_sha256": {row["file"]: row["sha256"] for row in rows},
              "runtime_seconds": time.perf_counter() - t0, "scores": scores}
    checkpoint = task_path(name)
    staging = checkpoint.with_suffix(".tmp")
    staging.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(checkpoint)
    return {"task": name, "runtime_seconds": result["runtime_seconds"],
            "tq_defined": denominator > 0, "checkpoint": str(checkpoint)}


def verify_checkpoint(name: str, rows: list[dict]) -> dict | None:
    path = task_path(name)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["task"] != name or data["transitions"] != int(rows[0]["max_size"]):
        raise RuntimeError(f"Stale task checkpoint: {name}")
    if data["pilot_sha256"] != {row["file"]: row["sha256"] for row in rows}:
        raise RuntimeError(f"Pilot changed after task checkpoint: {name}")
    if len(data["scores"]) != 5:
        raise RuntimeError(f"Incomplete checkpoint: {name}")
    return data


def summarize(tasks: list[str], by_task: dict[str, list[dict]]) -> None:
    from benchmarks.bridge.original_eri_eh_audit import rho
    checkpoints = [verify_checkpoint(name, by_task[name]) for name in tasks]
    if any(checkpoint is None for checkpoint in checkpoints):
        raise RuntimeError("Cannot summarize incomplete task grid")
    raw = pd.DataFrame([row for checkpoint in checkpoints for row in checkpoint["scores"]])
    if len(raw) != 70 or raw[["mdp_id", "policy_family"]].duplicated().any():
        raise RuntimeError("Unexpected score grid")
    label_manifest = json.loads((LABELS / "manifest.json").read_text(encoding="utf-8"))
    label_path = LABELS / "labels.csv"
    if label_manifest["status"] != "BRIDGE_REDUCED_TEST_LABELS_COMPLETE" or sha256(label_path) != label_manifest["labels_sha256"]:
        raise RuntimeError("BRIDGE labels have changed")
    labels = pd.read_csv(label_path)
    labels["mdp_id"] = labels.mdp_id.str.removesuffix("_sticky25")
    joined = labels.merge(raw, on=["mdp_id", "policy_family"], validate="many_to_one")
    if len(joined) != 210:
        raise RuntimeError("Score/label join incomplete")
    summary = []
    for algorithm, group in joined.groupby("algorithm"):
        for method, field in (("TQ original formula, authors' DQN reference", "tq_original"),
                              ("SACo original formula, authors' replay reference", "saco_original")):
            subset = group[np.isfinite(group[field])]
            summary.append({"algorithm": algorithm, "method": method,
                            "access": "privileged_true_MDP_online_reference",
                            "rows": len(subset), "mdps": subset.mdp_id.nunique(),
                            "defined_fraction": len(subset) / len(group),
                            "spearman_score_vs_harder_event": rho(subset[field].to_numpy(float), subset.event.to_numpy(int)),
                            "note": "exploratory; scalar score, not a seven-class forecast"})
    raw.to_csv(OUT / "raw_scores.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "rank_summary.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "status": "COMPLETE_EXPLORATORY_ORIGINAL_TQ_SACO_BRIDGE",
        "reference_code": "ml-jku/OfflineRL commit 8cc8eec (author DQN)",
        "test_outcomes_already_open": True,
        "source_sha256": {"pilot_manifest": sha256(PILOT / "manifest.json"),
                          "labels_manifest": sha256(LABELS / "manifest.json"),
                          "author_wrapper": sha256(ROOT / "scripts/analysis/oeh_final_tqsaco_original.py"),
                          "script": sha256(Path(__file__))},
        "task_checkpoints_sha256": {name: sha256(task_path(name)) for name in tasks},
        "output_sha256": {name: sha256(OUT / name) for name in ("raw_scores.csv", "rank_summary.csv")},
        "warning": "TQ/SACo require true-MDP online/random references and have a different endpoint than E_l; no Brier comparison.",
    }, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task", action="append", help="Run only named task(s); resumable")
    args = parser.parse_args()
    manifest = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "SEALED_PILOT_ONLY_NO_TEST_LABELS" or len(manifest["pilot_groups"]) != 70:
        raise RuntimeError("Pilot suite not sealed")
    by_task: dict[str, list[dict]] = {}
    for row in manifest["pilot_groups"]:
        by_task.setdefault(row["mdp_id"], []).append(row)
    tasks = sorted(by_task)
    chosen = sorted(set(args.task)) if args.task else tasks
    if any(name not in by_task for name in chosen):
        raise ValueError("Requested task outside frozen panel")
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    pending = [name for name in chosen if verify_checkpoint(name, by_task[name]) is None]
    print(f"TQ/SACo author-reference: {len(pending)} pending of {len(chosen)} chosen; workers={args.workers}", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_task, name, by_task[name]): name for name in pending}
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
    if all(verify_checkpoint(name, by_task[name]) is not None for name in tasks):
        if (OUT / "manifest.json").exists():
            print("Final manifest already exists; leaving sealed results untouched", flush=True)
        else:
            summarize(tasks, by_task)


if __name__ == "__main__":
    main()

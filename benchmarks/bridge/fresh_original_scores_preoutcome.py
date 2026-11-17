"""Freeze published-method scalar scores before fresh BRIDGE learner labels.

ERI and the published EH summary are scalar difficulty signals. TQ/SACo use
the authors' formulas and DQN reference implementation on our tabular adapter;
that reference has true-MDP online access, so it is privileged. None outputs
OEH's seven-class learner-specific requirement law.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.oeh_final_published import episode_returns
from scripts.analysis import oeh_final_tqsaco_original as TQ
from .fresh_adapter import PANEL
from .fresh_prepare_pilots import OUTPUT as PILOT
from .fresh_reward_gate import load_eligible_mdp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/week8/bridge_fresh_original_scores_preoutcome_2026-10-08"
AUTHOR = ROOT / "benchmarks/bridge/metadata/summary_sticky.csv"
SOURCE = ROOT / "benchmarks/bridge/source_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"bridge-fresh-author-tqsaco|{name}".encode()).digest()[:4], "little") % (2**31 - 1)


def one_task(name: str, rows: list[dict], published: pd.Series, horizon: int) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    mdp = load_eligible_mdp(name)
    n = int(rows[0]["max_size"])
    if len(rows) != 5 or any(int(row["max_size"]) != n for row in rows):
        raise RuntimeError(f"Incomplete pilot suite: {name}")
    source_class = TQ.TabularEnv
    TQ.TabularEnv = lambda m: source_class(m, horizon=horizon)
    started = time.perf_counter()
    try:
        online_max, online_pairs, trace = TQ.online_reference(mdp, n, seed(name))
    finally:
        TQ.TabularEnv = source_class
    env = source_class(mdp, horizon=horizon)
    env.seed(seed(name) + 1)
    rewards, dones, done = [], [], True
    for _ in range(n):
        if done:
            env.reset()
        _, reward, done, _ = env.step(int(env.rng.integers(mdp.num_actions)))
        rewards.append(reward)
        dones.append(done)
    random_return = float(TQ.get_returns(np.asarray(rewards), np.asarray(dones)).mean())
    denominator = online_max - random_return
    scores = []
    for row in rows:
        path = PILOT / row["file"]
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"Pilot hash mismatch: {name}")
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        returns = episode_returns(arrays)
        mean = float(returns.mean())
        pairs = TQ.unique_state_actions_exact(
            np.eye(mdp.num_states, dtype=np.float32)[arrays["state"]], arrays["action"])
        scores.append({"mdp_id": name, "policy_family": row["policy_family"],
                       "eri_original": (float((returns.max() - mean) / mean)
                                        if np.isfinite(mean) and mean > 0 else None),
                       "eri_defined": bool(np.isfinite(mean) and mean > 0),
                       "tq_original": (float((mean - random_return) / denominator)
                                       if denominator > 0 else None),
                       "tq_defined": bool(denominator > 0),
                       "saco_original": float(pairs / online_pairs) if online_pairs > 0 else None,
                       "saco_defined": bool(online_pairs > 0),
                       "eh_author_epw": float(published.epw),
                       "eh_author_k_min": float(published.k_min),
                       "pilot_mean_return": mean, "pilot_unique_pairs": pairs,
                       "pilot_episodes": len(returns)})
    return {"status": "FRESH_ORIGINAL_SCALAR_SCORES_PREOUTCOME_ONLY", "mdp_id": name,
            "pilot_sha256": {row["file"]: row["sha256"] for row in rows},
            "source_npz_sha256": mdp.metadata["source_sha256"],
            "author_horizon": horizon, "online_reference_max_return": online_max,
            "online_reference_unique_pairs": online_pairs,
            "online_reference_evaluations": len(trace),
            "random_reference_mean_return": random_return,
            "tq_denominator": denominator,
            "runtime_seconds": time.perf_counter() - started,
            "scores": scores, "test_labels_read": False}


def main() -> None:
    protocol_file, pilot_file = PANEL / "protocol.json", PILOT / "manifest.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_file.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if (protocol["status"] != "FRESH_BRIDGE_PROTOCOL_FROZEN_BEFORE_LABELS" or
            pilot["status"] != "SEALED_FRESH_PILOT_ONLY_NO_TEST_LABELS" or
            pilot["protocol_sha256"] != sha256(protocol_file) or
            hashlib.md5(AUTHOR.read_bytes()).hexdigest() != source["source_csv_md5"]["summary_sticky.csv"]):
        raise RuntimeError("Fresh original-score preoutcome gate failed")
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Original scores already sealed")
    author = pd.read_csv(AUTHOR).set_index("mdp_name")
    by_task = {name: [row for row in pilot["pilot_groups"] if row["mdp_id"] == name]
               for name in protocol["evaluation_tasks"]}
    OUT.mkdir(parents=True, exist_ok=True)
    group_dir = OUT / "tasks"
    group_dir.mkdir(exist_ok=True)
    documents = []
    for index, name in enumerate(protocol["evaluation_tasks"], 1):
        path = group_dir / f"{name}.json"
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            if (document["status"] != "FRESH_ORIGINAL_SCALAR_SCORES_PREOUTCOME_ONLY" or
                    document["pilot_sha256"] != {row["file"]: row["sha256"] for row in by_task[name]}):
                raise RuntimeError(f"Stale original-score checkpoint: {name}")
        else:
            document = one_task(name, by_task[name], author.loc[name],
                                int(protocol["episode_horizon"][name]))
            temporary = path.with_suffix(".json.part")
            temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(path)
        documents.append(document)
        print(f"Fresh original scores {index}/5 {name}: {document['runtime_seconds']:.1f}s", flush=True)
    raw = pd.DataFrame([row for document in documents for row in document["scores"]])
    if len(raw) != 25 or raw[["mdp_id", "policy_family"]].duplicated().any():
        raise RuntimeError("Incomplete original scalar score table")
    raw.sort_values(["mdp_id", "policy_family"]).to_csv(OUT / "scores_before_label_read.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "status": "SEALED_FRESH_ORIGINAL_SCALAR_SCORES_NO_TEST_LABELS",
        "test_labels_read": False, "tasks": 5, "behavior_rows": 25,
        "protocol_sha256": sha256(protocol_file),
        "source_sha256": {"pilot_manifest": sha256(pilot_file),
                          "author_sticky_summary": sha256(AUTHOR),
                          "published_formula_adapter": sha256(Path(TQ.__file__)),
                          "script": sha256(Path(__file__))},
        "task_sha256": {name: sha256(group_dir / f"{name}.json") for name in by_task},
        "scores_sha256": sha256(OUT / "scores_before_label_read.csv"),
        "scope": "ERI pilot-only; EH author summary privileged true MDP; TQ/SACo original formulas with authors' online DQN reference on task-specific horizon and true MDP; all scalar rank comparisons only."},
        indent=2) + "\n", encoding="utf-8")
    print("Sealed 25 original scalar-score rows before labels", flush=True)


if __name__ == "__main__":
    main()

"""One-shot external component check on Farama's published FourRooms trajectories.

This tests the empirical-Bayes Dirichlet transition component on a specified
local visual abstraction. It does not test the full OEH learner-requirement
claim because Farama's observations are partial and the data contain one task.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

from scripts.analysis.oeh_final_analysis import simultaneous_lower
from scripts.analysis.run_unseen_validation import write_json
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from oeh.posterior import dirichlet_multinomial_concentration

DATA = ROOT / "datasets/published_minari/D4RL/minigrid/fourrooms-random-v0/data/main_data.hdf5"
META = DATA.with_name("metadata.json")
OUT = ROOT / "artifacts/published_fourrooms_calibration_2026-10-04"
SOURCE = "https://minari.farama.org/datasets/D4RL/minigrid/fourrooms-random-v0/"
CATEGORIES = 11


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_id(episode_id: int) -> str:
    value = int.from_bytes(hashlib.sha256(f"oeh-fourrooms-v1|{episode_id}".encode()).digest()[:8], "big") % 10
    return "train" if value == 0 else "test" if value in (8, 9) else "unused"


def context(image: np.ndarray, direction: int, action: int) -> bytes:
    patch = image[2:5, 3:6, :]
    return hashlib.blake2b(patch.tobytes() + bytes((int(direction), int(action))), digest_size=12).digest()


def freeze():
    if OUT.exists():
        raise FileExistsError("External benchmark namespace exists")
    OUT.mkdir(parents=True)
    protocol = {
        "dataset": "D4RL/minigrid/fourrooms-random-v0",
        "source": SOURCE,
        "hdf5_sha256": digest(DATA), "metadata_sha256": digest(META),
        "script_sha256": digest(Path(__file__)),
        "split": "sha256('oeh-fourrooms-v1|<episode_id>') first 8 bytes modulo 10; train 0, test 8 or 9, otherwise unused",
        "context": "3x3x3 image patch image[2:5,3:6,:] plus direction and action, Blake2b-96 hash",
        "target": "object-type integer at next_image[3,5,0], 11 categories",
        "methods": {"empirical_bayes": "uniform-base Dirichlet posterior mean, total concentration exact empirical Bayes from training rows",
                    "mle": "empirical categorical MLE, global training distribution for unseen contexts",
                    "kt": "Jeffreys Dirichlet with 0.5 pseudocount per category"},
        "metrics": ["multiclass Brier", "multiclass log loss"],
        "inference": "paired episode-cluster bootstrap, simultaneous one-sided 95% over all four EB contrasts; no minimum point gain",
        "scope": "component transfer only; partial-observation abstraction, no full OEH learner-requirement claim",
        "outcomes_observed_before_freeze": 0,
    }
    write_json(OUT / "protocol.json", protocol)
    (OUT / "protocol.sha256").write_text(digest(OUT / "protocol.json") + "\n", encoding="ascii")
    print("External protocol", digest(OUT / "protocol.json"), flush=True)


def verify():
    path = OUT / "protocol.json"
    if digest(path) != (OUT / "protocol.sha256").read_text().strip():
        raise RuntimeError("External protocol changed")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if digest(DATA) != doc["hdf5_sha256"] or digest(META) != doc["metadata_sha256"]:
        raise RuntimeError("Published dataset changed")
    if digest(Path(__file__)) != doc["script_sha256"]:
        raise RuntimeError("Frozen benchmark code changed")


def run():
    verify()
    if (OUT / "result.json").exists():
        raise FileExistsError("External benchmark already evaluated")
    training = defaultdict(lambda: np.zeros(CATEGORIES, dtype=np.int64))
    heldout = []
    global_counts = np.zeros(CATEGORIES, dtype=np.int64)
    episodes = {"train": 0, "test": 0, "unused": 0}
    steps = {"train": 0, "test": 0, "unused": 0}
    with DATA.open("rb") as handle, h5py.File(handle, "r") as hdf:
        for group_name in sorted(hdf):
            if not group_name.startswith("episode_"):
                continue
            episode_id = int(group_name.split("_")[-1])
            subset = split_id(episode_id)
            episodes[subset] += 1
            if subset == "unused":
                continue
            group = hdf[group_name]
            images = group["observations/image"][:]
            directions = group["observations/direction"][:]
            actions = group["actions"][:]
            if len(images) != len(actions) + 1:
                raise RuntimeError(f"Observation/action alignment: {group_name}")
            rows = []
            for step, action in enumerate(actions):
                key = context(images[step], int(directions[step]), int(action))
                label = int(images[step + 1, 3, 5, 0])
                if not (0 <= label < CATEGORIES):
                    raise RuntimeError("Object category exceeds frozen vocabulary")
                rows.append((key, label))
            steps[subset] += len(rows)
            if subset == "train":
                for key, label in rows:
                    training[key][label] += 1
                    global_counts[label] += 1
            else:
                heldout.append((episode_id, rows))
    if len(training) < 4 or not heldout:
        raise RuntimeError("Benchmark split incomplete")
    matrix = np.stack(list(training.values()))
    alpha = dirichlet_multinomial_concentration(matrix)
    if not np.isfinite(alpha):
        alpha = 1e12
    base = np.full(CATEGORIES, 1 / CATEGORIES)
    names = ("empirical_bayes", "mle", "kt")
    score = np.zeros((len(heldout), len(names), 2), dtype=float)
    for idx, (_, rows) in enumerate(heldout):
        for key, label in rows:
            count = training.get(key)
            if count is None:
                count = np.zeros(CATEGORIES, dtype=np.int64)
            total = count.sum()
            eb = (count + alpha * base) / (total + alpha)
            mle = count / total if total else global_counts / global_counts.sum()
            kt = (count + .5) / (total + .5 * CATEGORIES)
            for j, probs in enumerate((eb, mle, kt)):
                score[idx, j, 0] += (1 - probs[label]) ** 2 + (probs @ probs - probs[label] ** 2)
                score[idx, j, 1] -= np.log(np.clip(probs[label], 1e-12, 1))
        score[idx] /= len(rows)
    effects = np.column_stack([(score[:, j, metric] - score[:, 0, metric]) for j in (1, 2) for metric in (0, 1)])
    lower, critical = simultaneous_lower(effects, 10000, 20261004)
    result = {
        "protocol_sha256": digest(OUT / "protocol.json"),
        "episodes": episodes, "steps": steps,
        "training_contexts": len(training), "empirical_bayes_concentration": float(alpha),
        "scores": {name: {"brier": float(score[:, j, 0].mean()),
                           "log_loss": float(score[:, j, 1].mean())} for j, name in enumerate(names)},
        "effects_baseline_minus_eb": {f"{baseline}_{metric}": {"mean": float(effects[:, k].mean()),
                "simultaneous_one_sided_lower_95": float(lower[k])}
            for k, (baseline, metric) in enumerate((("mle", "brier"), ("mle", "log_loss"),
                                                       ("kt", "brier"), ("kt", "log_loss")))},
        "simultaneous_critical": float(critical),
        "gate": "GO" if bool((lower > 0).all()) else "NO-GO",
        "scope": "Dirichlet transition component on a visual abstraction of published partially observed data; not full OEH validation",
    }
    write_json(OUT / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    else:
        run()


if __name__ == "__main__":
    main()

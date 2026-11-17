"""TabPFN v2 seven-class forecasts on B2 (task card T3; spec docs/week8/T3_TABPFN_SPEC_2026-10-15.md).

Runs in the isolated venv (``D:/Tools/venvs/tabpfn``); only numpy, pandas and tabpfn are imported.

    D:/Tools/venvs/tabpfn/Scripts/python.exe -I scripts/t3/tabpfn_forecast.py --train <train.csv.gz> --query <query.csv.gz> --out <probs.csv.gz>

``train`` has ``algorithm``, ``event`` and the feature columns; ``query`` has ``mdp_id, policy_family, algorithm`` and the
same features. One classifier per learner (``algorithm``), as in the spec. Writes ``p_0..p_6`` per query row and
``<out>.meta.json`` with versions and the sha256 of the weights file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np

CLASSES = 7
KEYS = ["mdp_id", "policy_family", "algorithm"]


def expand(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Probabilities over the classes seen in training -> all seven classes (unseen classes get 0)."""
    out = np.zeros((len(proba), CLASSES))
    out[:, np.asarray(classes, int)] = proba
    return out / out.sum(1, keepdims=True)


def main() -> None:
    import pandas as pd
    import tabpfn
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    train, query = pd.read_csv(args.train), pd.read_csv(args.query)
    features = [c for c in train.columns if c not in ("algorithm", "event", "mdp_id", "policy_family")]
    frames, weights = [], None
    for learner, q in query.groupby("algorithm", sort=True):
        tr = train[train.algorithm == learner]
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)
        clf.fit(tr[features].to_numpy(float), tr.event.to_numpy(int))
        p = expand(clf.predict_proba(q[features].to_numpy(float)), clf.classes_)
        frame = q[KEYS].copy()
        frame[[f"p_{i}" for i in range(CLASSES)]] = p
        frames.append(frame)
        weights = Path(clf.model_path) if not isinstance(clf.model_path, (list, tuple)) else Path(clf.model_path[0])
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False)
    meta = {"tabpfn": tabpfn.__version__, "torch": torch.__version__, "python": platform.python_version(),
            "model_version": "v2", "weights_file": weights.name, "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "settings": "create_default_for_version(V2, device='cpu', random_state=0)", "features": features,
            "train_rows": int(len(train)), "query_rows": int(len(query))}
    Path(str(args.out) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

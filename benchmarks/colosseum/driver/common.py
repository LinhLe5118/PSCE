"""Shared paths for the external Colosseum trial.

Everything runs against an isolated copy of the Devoted repository
(``workspace/``); the original repository is never written.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

TRIAL = Path(__file__).resolve().parents[1]
# OEH_WORKSPACE points at a Devoted checkout; the default is the trial's copy.
WS = Path(os.environ.get("OEH_WORKSPACE", TRIAL / "workspace"))
DATA = TRIAL / "data"
OUT = TRIAL / "results"
os.environ.setdefault("TMP", str(TRIAL / ".tmp"))
os.environ.setdefault("TEMP", str(TRIAL / ".tmp"))

# External run selector: "colosseum" = exploratory run (results/), "confirm" =
# confirmatory run on fresh MDPs (results/confirm/). Paths derive from it.
RUN = os.environ.get("COLOSSEUM_RUN", "colosseum")
RUN_SRC = DATA / RUN
RUN_BUNDLE_NAME = f"{RUN}_bundle"
RUN_BUNDLE = DATA / RUN_BUNDLE_NAME
RES = OUT if RUN == "colosseum" else OUT / RUN

COHORTS = {
    "first60": WS / "artifacts/independent_validation_2026-09-30",
    "second60": WS / "artifacts/oeh_difficulty_confirmation_2026-10-01",
    "first120": WS / "artifacts/oeh_difficulty_unseen_120_2026-10-01",
    "second120": WS / "artifacts/oeh_pooled_successor_unseen_120_2026-10-01",
    "third120": WS / "artifacts/oeh_cross_capacity_unseen_120_2026-10-01",
    "c4": WS / "artifacts/oeh_final_unseen_120_2026-10-02",
    "c7": WS / "artifacts/oeh_confirmatory_unseen_120_2026-10-04",
}
# Sealed label/feature sources (some cohorts keep labels in a sibling artifact).
LABELS = {"first60": WS / "artifacts/independent_difficulty_2026-09-30/labels.csv"}


def use_workspace() -> None:
    for item in (str(WS / "code"), str(WS)):
        if item in sys.path:
            sys.path.remove(item)
        sys.path.insert(0, item)
    os.chdir(WS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str), encoding="utf-8")

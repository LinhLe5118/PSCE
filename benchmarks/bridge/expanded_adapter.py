"""Use the previously validated sticky adapter with a new frozen registry.

The five-task adapter has no algorithmic change. Its registry lookup is scoped
to this call so sealed five-task artifacts and files remain untouched.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from . import fresh_adapter

PANEL = Path(__file__).resolve().parent / "expanded_panel_2026-10-08"


def registry() -> dict[str, dict]:
    manifest = json.loads((PANEL / "manifest.json").read_text(encoding="utf-8"))
    path = PANEL / "candidate_registry.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["source_sha256"]["candidate_registry"]:
        raise ValueError("Expanded candidate registry changed")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [row["mdp_name"] for row in rows] != manifest["task_names"]:
        raise ValueError("Expanded candidate order changed")
    return {row["mdp_name"]: row for row in rows}


def load_expanded_sticky_mdp(name: str):
    """Call the proven sticky conversion with an explicit new task registry."""
    rows = registry()
    if name not in rows:
        raise ValueError("Task outside frozen expanded registry")
    original_lookup = fresh_adapter.registry
    try:
        fresh_adapter.registry = lambda: rows
        return fresh_adapter.load_fresh_sticky_mdp(name)
    finally:
        fresh_adapter.registry = original_lookup

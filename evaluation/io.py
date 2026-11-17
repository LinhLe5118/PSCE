"""Shared CSV/JSON helpers.

Deliberately stdlib-only (``csv`` + ``json``), matching the rest of the
repository (``scripts/data/generate_offline_datasets.py`` does not use pandas either, and
``requirements.txt`` locks only ``numpy``/``scipy``). Adding a new dependency
for Week 2 tooling is not this workstream's call to make unilaterally.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

Row = dict[str, Any]


@contextmanager
def _atomic_text(path: Path):
    """Publish a complete UTF-8 file with an atomic same-directory replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            yield handle
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_csv_rows(path: Path) -> list[Row]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _atomic_text(path) as handle:
        # The csv module defaults to CRLF even on non-Windows hosts.  Git
        # normalises text artifacts to LF in this repository, which used to
        # make a freshly recorded SHA-256 disagree with the committed file
        # immediately after checkout.  Emit the repository's canonical line
        # ending directly so artifact hashes are portable.
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_dataset_path(root: Path, record: Mapping[str, Any]) -> Path:
    """Resolve a manifest member and verify its bytes before loading it."""
    root = root.resolve()
    path = (root / str(record["file"])).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes bundle: {path}")
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Missing or invalid dataset SHA-256: {path}")
    if sha256_file(path) != expected.lower():
        raise ValueError(f"Dataset checksum mismatch: {path}")
    return path


def reserve_output_dir(path: Path) -> None:
    """Reserve a new bundle; an interrupted run remains available for audit.

    A final report is written last by the caller. Existing directories,
    including incomplete runs, must never be reused by a CLI invocation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _atomic_text(path) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = [
    "Row",
    "read_csv_rows",
    "write_csv_rows",
    "sha256_file",
    "sha256_text",
    "write_json",
    "verified_dataset_path",
    "reserve_output_dir",
]

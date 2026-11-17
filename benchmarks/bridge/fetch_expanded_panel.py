"""Fetch only the five frozen new BRIDGE MDP arrays from the author ZIP."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .fetch_fresh_panel import fetch_member
from .fetch_candidates import ARCHIVE_BYTES, SOURCE, TARGET
from .inspect_zip_index import entries_from_tail

PANEL = Path(__file__).resolve().parent / "expanded_panel_2026-10-08"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest_file = PANEL / "manifest.json"
    frozen = json.loads(manifest_file.read_text(encoding="utf-8"))
    registry_file = PANEL / "candidate_registry.csv"
    if (frozen["status"] != "EXPANDED_BRIDGE_METADATA_SCREEN_FROZEN_BEFORE_NEW_LABELS" or
            sha256(registry_file) != frozen["source_sha256"]["candidate_registry"] or
            SOURCE["source_zip_bytes"] != ARCHIVE_BYTES):
        raise ValueError("Frozen source/registry mismatch")
    with registry_file.open(newline="", encoding="utf-8") as handle:
        names = [row["mdp_name"] for row in csv.DictReader(handle)]
    if names != frozen["task_names"]:
        raise ValueError("Task order changed")
    tail = TARGET / "zip_index_tail.bin"
    if sha256(tail) != SOURCE["zip_index_tail_sha256"]:
        raise ValueError("Pinned ZIP index changed")
    entries = {entry["name"]: entry for entry in entries_from_tail(tail.read_bytes())}
    result = []
    for index, name in enumerate(names, 1):
        member = f"bridge_dataset/mdps/{name}/consolidated.npz"
        if member not in entries:
            raise ValueError(f"Missing source member: {member}")
        print(f"[{index}/{len(names)}] {name}: {entries[member]['compressed_bytes']:,} compressed bytes", flush=True)
        result.append(fetch_member(entries[member], name))
    output = PANEL / "download_manifest.json"
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if ([row["file_sha256"] for row in previous["members"]] !=
                [row["file_sha256"] for row in result]):
            raise ValueError("Existing expanded download manifest differs")
        return
    output.write_text(json.dumps({
        "status": "EXPANDED_BRIDGE_MEMBERS_VERIFIED_NO_LABELS",
        "source_record": SOURCE["source_record"],
        "source_zip_md5_expected_not_checked": SOURCE["source_zip_md5"],
        "zip_index_tail_sha256": sha256(tail),
        "frozen_registry_sha256": sha256(registry_file),
        "verification": "HTTP 206 range; per-member ZIP size and CRC32; file SHA-256. Whole ZIP MD5 not checked.",
        "test_labels_read": False,
        "members": result,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(result)} new BRIDGE MDP arrays", flush=True)


if __name__ == "__main__":
    main()

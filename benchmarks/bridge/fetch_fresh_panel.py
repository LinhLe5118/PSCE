"""Fetch the metadata-frozen, untouched BRIDGE panel from the author ZIP.

Only the eight exact consolidated.npz paths in candidate_registry.csv are
allowed. ZIP member CRC32 and size are checked; the entire 28.7 GB archive's
MD5 is not claimed to have been checked.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_candidates import ARCHIVE_BYTES, LOCAL_HEADER, SOURCE, TARGET, fetch_range
from .inspect_zip_index import entries_from_tail

ROOT = Path(__file__).resolve().parents[2]
PANEL = Path(__file__).resolve().parent / "fresh_panel_2026-10-08"


def fetch_member(entry: dict, task: str) -> dict:
    expected_name = f"bridge_dataset/mdps/{task}/consolidated.npz"
    if entry["name"] != expected_name or not re.fullmatch(r"[A-Za-z0-9_-]+", task):
        raise ValueError("Unexpected ZIP member name")
    output = TARGET / "mdps" / task / "consolidated.npz"
    if output.exists():
        data = output.read_bytes()
        if len(data) == entry["uncompressed_bytes"] and f"{zlib.crc32(data):08x}" == entry["crc32"]:
            return {**entry, "local_path": str(output), "file_sha256": hashlib.sha256(data).hexdigest(), "status": "VERIFIED_EXISTING"}
        raise ValueError(f"Existing file failed ZIP verification: {output}")
    offset = entry["local_header_offset"]
    header = fetch_range(offset, offset + 511)
    fields = LOCAL_HEADER.unpack_from(header)
    if fields[0] != b"PK\x03\x04" or fields[3] != entry["method"] or fields[2] & 1:
        raise ValueError(f"ZIP local header mismatch: {task}")
    name_len, extra_len = fields[9], fields[10]
    header_size = LOCAL_HEADER.size + name_len + extra_len
    if header_size > len(header):
        header = fetch_range(offset, offset + header_size - 1)
    local_name = header[LOCAL_HEADER.size:LOCAL_HEADER.size + name_len].decode(
        "utf-8" if fields[2] & 0x800 else "cp437")
    if local_name != expected_name:
        raise ValueError(f"ZIP local and central names differ: {task}")
    start = offset + header_size
    payload = fetch_range(start, start + entry["compressed_bytes"] - 1)
    if entry["method"] == 8:
        data = zlib.decompress(payload, -15)
    elif entry["method"] == 0:
        data = payload
    else:
        raise ValueError(f"Unsupported ZIP method: {entry['method']}")
    if len(data) != entry["uncompressed_bytes"] or f"{zlib.crc32(data):08x}" != entry["crc32"]:
        raise ValueError(f"ZIP member checksum mismatch: {task}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.part")
    temporary.write_bytes(data)
    temporary.replace(output)
    return {**entry, "local_path": str(output), "file_sha256": hashlib.sha256(data).hexdigest(), "status": "DOWNLOADED_VERIFIED"}


def main() -> None:
    frozen = json.loads((PANEL / "manifest.json").read_text(encoding="utf-8"))
    registry = PANEL / "candidate_registry.csv"
    if hashlib.sha256(registry.read_bytes()).hexdigest() != frozen["registry_sha256"]:
        raise ValueError("Frozen candidate registry changed")
    with registry.open(newline="", encoding="utf-8") as handle:
        tasks = [row["mdp_name"] for row in csv.DictReader(handle)]
    if tasks != frozen["task_names"] or len(tasks) != 8:
        raise ValueError("Frozen task list changed")
    if SOURCE["source_zip_bytes"] != ARCHIVE_BYTES:
        raise ValueError("Pinned ZIP size changed")
    tail = (TARGET / "zip_index_tail.bin").read_bytes()
    if hashlib.sha256(tail).hexdigest() != SOURCE["zip_index_tail_sha256"]:
        raise ValueError("Pinned ZIP index tail changed")
    entries = {entry["name"]: entry for entry in entries_from_tail(tail)}
    expected = [f"bridge_dataset/mdps/{task}/consolidated.npz" for task in tasks]
    if any(name not in entries for name in expected):
        raise ValueError("A frozen task is missing from the published ZIP")
    print(f"Fetching {len(tasks)} frozen BRIDGE members, {sum(entries[name]['compressed_bytes'] for name in expected):,} compressed bytes", flush=True)
    results = []
    for index, (task, name) in enumerate(zip(tasks, expected), 1):
        print(f"[{index}/{len(tasks)}] {task}", flush=True)
        results.append(fetch_member(entries[name], task))
    manifest = {
        "status": "EIGHT_PUBLISHED_MDP_MEMBERS_VERIFIED_NO_LABELS",
        "source_record": SOURCE["source_record"],
        "zip_index_tail_sha256": hashlib.sha256(tail).hexdigest(),
        "source_zip_md5_expected_not_checked": SOURCE["source_zip_md5"],
        "registry_sha256": frozen["registry_sha256"],
        "verification": "HTTP 206 exact Content-Range, ZIP member size and CRC32, file SHA256; whole ZIP MD5 not checked",
        "test_labels_read": False,
        "members": results,
    }
    (PANEL / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("PASS: all eight members verified; no outcome labels read", flush=True)


if __name__ == "__main__":
    main()

"""Fetch only the pinned BRIDGE MiniGrid candidate NPZs via ZIP byte ranges.

The official archive is 28.7 GB, but its ZIP64 central directory identifies
the 16 prescreened MDP members. Each downloaded member is checked against the
archive's uncompressed size and CRC32. This does not verify the whole ZIP MD5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import time
import urllib.request
import zlib
from pathlib import Path

from inspect_zip_index import ARCHIVE_BYTES, entries_from_tail, selected_entries


ROOT = Path(__file__).resolve().parents[2]
SOURCE = json.loads((ROOT / "benchmarks/bridge/source_manifest.json").read_text(encoding="utf-8"))
URL = "https://zenodo.org/api/records/12556298/files/bridge_dataset_v1.1.1.zip/content"
TARGET = ROOT / "datasets/bridge_v1_1_1"
LOCAL_HEADER = struct.Struct("<4s5H3L2H")


def fetch_range(first: int, last: int) -> bytes:
    if first < 0 or last < first or last >= ARCHIVE_BYTES:
        raise ValueError("Invalid archive byte range")
    expected = f"bytes {first}-{last}/{ARCHIVE_BYTES}"
    request = urllib.request.Request(URL, headers={"Range": f"bytes={first}-{last}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                content_range = response.headers.get("Content-Range")
                if response.status != 206 or content_range != expected:
                    raise ValueError(f"Unexpected range response: {response.status} {content_range}")
                data = response.read()
            if len(data) != last - first + 1:
                raise ValueError("Short range response")
            return data
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def fetch_member(entry: dict) -> dict:
    name = entry["name"]
    match = re.fullmatch(r"bridge_dataset/mdps/(MiniGrid-[^/]+)/consolidated\.npz", name)
    if match is None:
        raise ValueError(f"Unexpected candidate path: {name}")
    output = TARGET / "mdps" / match.group(1) / "consolidated.npz"
    if output.exists():
        data = output.read_bytes()
        if len(data) == entry["uncompressed_bytes"] and f"{zlib.crc32(data):08x}" == entry["crc32"]:
            return {**entry, "file_sha256": hashlib.sha256(data).hexdigest(), "local_path": str(output),
                    "status": "VERIFIED_EXISTING"}

    offset = entry["local_header_offset"]
    header = fetch_range(offset, offset + 511)
    fields = LOCAL_HEADER.unpack_from(header)
    if fields[0] != bytes([80, 75, 3, 4]) or fields[3] != entry["method"]:
        raise ValueError(f"Local header mismatch: {name}")
    if fields[2] & 1:
        raise ValueError(f"Encrypted member unsupported: {name}")
    name_len, extra_len = fields[9], fields[10]
    header_size = LOCAL_HEADER.size + name_len + extra_len
    if header_size > len(header):
        header = fetch_range(offset, offset + header_size - 1)
    local_name = header[LOCAL_HEADER.size:LOCAL_HEADER.size + name_len].decode(
        "utf-8" if fields[2] & 0x800 else "cp437"
    )
    if local_name != name:
        raise ValueError(f"Local and central names differ: {name}")
    start = offset + header_size
    compressed = fetch_range(start, start + entry["compressed_bytes"] - 1)
    if entry["method"] == 8:
        data = zlib.decompress(compressed, -15)
    elif entry["method"] == 0:
        data = compressed
    else:
        raise ValueError(f"Unsupported ZIP compression method: {entry['method']}")
    if len(data) != entry["uncompressed_bytes"] or f"{zlib.crc32(data):08x}" != entry["crc32"]:
        raise ValueError(f"Size or ZIP CRC32 mismatch: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.part")
    temporary.write_bytes(data)
    temporary.replace(output)
    return {**entry, "file_sha256": hashlib.sha256(data).hexdigest(), "local_path": str(output),
            "status": "DOWNLOADED_VERIFIED"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail", type=Path, default=TARGET / "zip_index_tail.bin")
    args = parser.parse_args()
    if SOURCE["source_zip_bytes"] != ARCHIVE_BYTES:
        raise ValueError("Pinned source size changed")
    if not args.tail.exists():
        args.tail.parent.mkdir(parents=True, exist_ok=True)
        args.tail.write_bytes(fetch_range(ARCHIVE_BYTES - 524_288, ARCHIVE_BYTES - 1))
    tail = args.tail.read_bytes()
    if hashlib.sha256(tail).hexdigest() != SOURCE["zip_index_tail_sha256"]:
        raise ValueError("ZIP central-directory tail differs from the pinned source")
    selected = selected_entries(entries_from_tail(tail))
    result = []
    for i, entry in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {entry['name']}", flush=True)
        result.append(fetch_member(entry))
    manifest = {
        "source_record": SOURCE["source_record"],
        "source_zip_bytes": ARCHIVE_BYTES,
        "source_zip_md5_expected_not_checked": SOURCE["source_zip_md5"],
        "zip_index_tail_sha256": hashlib.sha256(tail).hexdigest(),
        "verification": "HTTP 206 exact Content-Range, ZIP member size and CRC32, file SHA256; whole archive MD5 not checked",
        "members": result,
    }
    path = TARGET / "candidate_download_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(result)} candidate MDPs verified; manifest datasets/bridge_v1_1_1/candidate_download_manifest.json")


if __name__ == "__main__":
    main()

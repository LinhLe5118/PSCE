"""Inspect a byte-range copy of the ZIP tail without downloading the archive."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARCHIVE_BYTES = 28_708_843_161
CD_RECORD = struct.Struct("<4s6H3L5H2L")


def entries_from_tail(tail: bytes, total_bytes: int = ARCHIVE_BYTES) -> list[dict]:
    start = total_bytes - len(tail)
    locator = tail.rfind(bytes([80, 75, 6, 7]))
    if locator < 0:
        raise ValueError("ZIP64 locator absent from supplied tail")
    _, disk, record_abs, disks = struct.unpack_from("<4sLQL", tail, locator)
    if disk != 0 or disks != 1:
        raise ValueError("Multi-disk ZIP unsupported")
    record_pos = record_abs - start
    if record_pos < 0:
        raise ValueError("ZIP64 record outside supplied tail")
    record = struct.unpack_from("<4sQHHIIQQQQ", tail, record_pos)
    if record[0] != bytes([80, 75, 6, 6]):
        raise ValueError("ZIP64 record signature mismatch")
    count, cd_bytes, cd_abs = record[7], record[8], record[9]
    pos = cd_abs - start
    if pos < 0 or pos + cd_bytes > len(tail):
        raise ValueError("Central directory outside supplied tail")
    result = []
    for _ in range(count):
        fields = CD_RECORD.unpack_from(tail, pos)
        if fields[0] != bytes([80, 75, 1, 2]):
            raise ValueError(f"Central directory signature mismatch at {pos}")
        name_len, extra_len, comment_len = fields[10:13]
        raw_name = tail[pos + CD_RECORD.size:pos + CD_RECORD.size + name_len]
        name = raw_name.decode("utf-8" if fields[3] & 0x800 else "cp437")
        compressed, uncompressed, local_abs = fields[8], fields[9], fields[16]
        if 0xFFFFFFFF in (compressed, uncompressed, local_abs):
            extra = tail[pos + CD_RECORD.size + name_len:
                         pos + CD_RECORD.size + name_len + extra_len]
            cursor = 0
            while cursor + 4 <= len(extra):
                kind, length = struct.unpack_from("<HH", extra, cursor)
                block = extra[cursor + 4:cursor + 4 + length]
                if kind == 1:
                    q = 0
                    if uncompressed == 0xFFFFFFFF:
                        uncompressed = struct.unpack_from("<Q", block, q)[0]
                        q += 8
                    if compressed == 0xFFFFFFFF:
                        compressed = struct.unpack_from("<Q", block, q)[0]
                        q += 8
                    if local_abs == 0xFFFFFFFF:
                        local_abs = struct.unpack_from("<Q", block, q)[0]
                    break
                cursor += 4 + length
            if 0xFFFFFFFF in (compressed, uncompressed, local_abs):
                raise ValueError(f"Missing ZIP64 extra field: {name}")
        result.append({
            "name": name,
            "method": fields[4],
            "crc32": f"{fields[7]:08x}",
            "compressed_bytes": compressed,
            "uncompressed_bytes": uncompressed,
            "local_header_offset": local_abs,
        })
        pos += CD_RECORD.size + name_len + extra_len + comment_len
    if pos != cd_abs - start + cd_bytes:
        raise ValueError("Parsed central directory size mismatch")
    return result


def selected_entries(entries: list[dict]) -> list[dict]:
    with (HERE / "candidate_registry.csv").open(newline="", encoding="utf-8") as handle:
        names = {r["mdp_name"] for r in csv.DictReader(handle)
                 if r["resource_screen"] == "CANDIDATE_METADATA_ONLY"}
    selected = []
    for name in names:
        matches = [e for e in entries if e["name"].endswith(f"/{name}/consolidated.npz")]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one consolidated.npz for {name}: {len(matches)}")
        selected.append(matches[0])
    return sorted(selected, key=lambda x: x["name"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tail", type=Path)
    args = parser.parse_args()
    entries = entries_from_tail(args.tail.read_bytes())
    selected = selected_entries(entries)
    print(json.dumps({
        "archive_entries": len(entries),
        "selected_entries": len(selected),
        "compressed_bytes": sum(e["compressed_bytes"] for e in selected),
        "members": selected,
    }, indent=2))


if __name__ == "__main__":
    main()

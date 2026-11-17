"""Verify the T1 local checksum package without opening evaluation data.

This is an integrity check, not an H1/D1/B2 approval or a trusted timestamp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "docs/protocol/WEEK6_T1_FREEZE_v1.1.0-r1.sha256.json"
ALLOWED = {"code", "config", "environments", "evaluation", "scripts", "docs", "tests"}


def verify(root: Path, manifest_path: Path) -> dict:
    root = root.resolve()
    payload = manifest_path.read_bytes()
    expected = manifest_path.with_suffix(".json.sha256").read_text("ascii").strip()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Manifest checksum mismatch")
    manifest = json.loads(payload)
    if manifest["version"] not in {"1.0.0", "1.1.0", "1.1.0-r1"} or not manifest["files"]:
        raise ValueError("Unsupported or empty freeze")
    for name, checksum in manifest["files"].items():
        rel = PurePosixPath(name)
        if (rel.is_absolute() or ".." in rel.parts or "\\" in name
                or ":" in name or not rel.parts
                or (rel.parts[0] not in ALLOWED and name != "protocol_config.json")):
            raise ValueError(f"Disallowed freeze path: {name}")
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Freeze path escapes repository: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Frozen file changed: {name}")
    return {
        "status": "PASS", "version": manifest["version"],
        "files_verified": len(manifest["files"]),
        "manifest_sha256": expected,
        "evaluation_data_read": False,
        "execution_go_granted": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result = verify(ROOT, ROOT / MANIFEST)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify the H1 formula/theory package and T1 dependency; never opens datasets.

An integrity PASS is not independent B2 approval, an execution gate, or a
trusted timestamp. This command deliberately has no evaluation/run option.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from scripts.validation.verify_week6_t1_freeze import verify as verify_t1


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "docs/protocol/WEEK6_H1_FREEZE_v1.0.0.sha256.json"
CONTRACT = "docs/protocol/WEEK6_H1_FORMULA_v1.0.0.json"
T1_MANIFEST = "docs/protocol/WEEK6_T1_FREEZE_v1.1.0-r1.sha256.json"
ALLOWED_ROOTS = {"docs", "paper", "code", "scripts", "tests", "evaluation", "config"}
ALLOWED_SUFFIXES = {".md", ".tex", ".json", ".sha256", ".py"}


def checked_path(root: Path, name: str) -> Path:
    """Reject data paths and traversal before opening the target file."""
    rel = PurePosixPath(name)
    if (not rel.parts or rel.is_absolute() or ".." in rel.parts
            or "\\" in name or ":" in name or rel.parts[0] not in ALLOWED_ROOTS
            or rel.suffix not in ALLOWED_SUFFIXES
            or any(part.lower() in {"datasets", "data", "artifacts"} for part in rel.parts)):
        raise ValueError(f"Disallowed H1 path: {name}")
    path = root / name
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent):
        raise ValueError(f"Symlink in H1 path: {name}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"H1 path escapes repository: {name}")
    return resolved


def verify_files(root: Path, manifest_name: str) -> dict:
    root = root.resolve()
    manifest_path = checked_path(root, manifest_name)
    payload = manifest_path.read_bytes()
    expected = checked_path(root, manifest_name + ".sha256").read_text("ascii").strip()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("H1 manifest checksum mismatch")
    manifest = json.loads(payload)
    files = manifest.get("files")
    if manifest.get("version") != "1.0.0" or not isinstance(files, dict) or not files:
        raise ValueError("Unsupported or empty H1 freeze")
    for name, checksum in files.items():
        if hashlib.sha256(checked_path(root, name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Frozen H1 file changed: {name}")
    return {"files_verified": len(files), "manifest_sha256": expected, "files": files}


def verify(root: Path = ROOT) -> dict:
    root = root.resolve()
    checked = verify_files(root, MANIFEST)
    required = {CONTRACT, T1_MANIFEST, T1_MANIFEST + ".sha256",
                "docs/week6/H1_FORMULA_THEORY_FREEZE_v1.0.0.md",
                "paper/sections/h1_formula_freeze.tex",
                "scripts/validation/verify_week6_h1_freeze.py",
                "tests/test_h1_formula_freeze.py"}
    if not required <= checked["files"].keys():
        raise ValueError("H1 manifest is missing required handover files")
    contract = json.loads(checked_path(root, CONTRACT).read_text("utf-8"))
    if contract["execution_go"] or contract["evaluation_data_read"]:
        raise ValueError("H1 cannot grant execution GO or claim evaluation access")
    t1 = verify_t1(root, root / T1_MANIFEST)
    return {
        "status": "PASS", "version": contract["version"],
        "files_verified": checked["files_verified"],
        "manifest_sha256": checked["manifest_sha256"], "t1": t1,
        "formula_scope": "documented_frozen_implementation",
        "independent_b2_review_granted": False,
        "execution_go_granted": False, "evaluation_data_read": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        print(json.dumps(verify(), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc),
                          "execution_go_granted": False}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

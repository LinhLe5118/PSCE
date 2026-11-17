"""Sealed artifacts are immutable: every protocol hash and every resolvable file hash recorded in a seal or
manifest under artifacts/ must match the tracked file (sealed files never change).

A seal entry is "resolvable" when its key names a tracked file relative to the seal's directory or to the
repository root. Files stored through Git LFS are excluded, because a checkout without LFS holds pointers.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEX = re.compile(r"^[0-9a-f]{64}$")


def _git(*args: str, stdin: str | None = None) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, input=stdin, capture_output=True, text=True, encoding="utf-8",
                          check=True).stdout


def _tracked() -> set[str]:
    return set(_git("ls-files", "artifacts", "datasets").splitlines())


def _lfs(paths: list[str]) -> set[str]:
    out = _git("check-attr", "--stdin", "filter", stdin="\n".join(paths) + "\n")
    return {line.split(": filter: ")[0] for line in out.splitlines() if line.endswith(": filter: lfs")}


def _entries(obj, path=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and HEX.match(value):
                yield path + (key,), value
            else:
                yield from _entries(value, path + (key,))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _entries(value, path + (str(i),))


def _resolve(seal: Path, name: str, tracked: set[str]) -> str | None:
    for base in (seal.parent, ROOT):
        for candidate in (name, name.removesuffix("_sha256")):
            if "." not in candidate and "/" not in candidate and "\\" not in candidate:
                continue
            path = base / candidate.replace("\\", "/")
            if path.is_file():
                rel = path.resolve().relative_to(ROOT).as_posix()
                if rel in tracked:
                    return rel
    return None


def test_every_frozen_protocol_matches_its_recorded_hash():
    tracked = _tracked()
    checked = 0
    for rel in sorted(t for t in tracked if t.endswith("/protocol.sha256")):
        directory = ROOT / rel.rsplit("/", 1)[0]
        recorded = (ROOT / rel).read_text(encoding="ascii").strip()
        assert hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest() == recorded, rel
        checked += 1
    assert checked >= 14


def test_every_resolvable_seal_and_manifest_entry_matches():
    tracked = _tracked()
    seals = sorted(t for t in tracked if t.startswith("artifacts/") and t.endswith(".json")
                   and ("seal" in t.rsplit("/", 1)[1] or "manifest" in t.rsplit("/", 1)[1]) and "/oeh_b_groups/" not in t)
    references: list[tuple[str, str, str, str]] = []
    for rel in seals:
        seal = ROOT / rel
        doc = json.loads(seal.read_text(encoding="utf-8"))
        for keys, digest in _entries(doc):
            target = _resolve(seal, keys[-1], tracked)
            if target is not None:
                references.append((rel, "/".join(keys), target, digest))
    lfs = _lfs(sorted({r[2] for r in references}))
    mismatches, verified, cache = [], 0, {}
    for rel, key, target, digest in references:
        if target in lfs:
            continue
        if target not in cache:
            cache[target] = hashlib.sha256((ROOT / target).read_bytes()).hexdigest()
        verified += 1
        if cache[target] != digest:
            mismatches.append(f"{rel} [{key}] -> {target}")
    assert verified >= 600, verified
    assert not mismatches, "sealed files changed:\n" + "\n".join(mismatches)

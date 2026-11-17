"""Reconstruct the frozen development runtime from checked-in source bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config/restored_runtime_manifest.json"
DEFAULT_OUTPUT = ROOT / "tmp/oeh_development_2026-09-30"


def _relative_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Invalid runtime manifest path: {name}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Runtime path escapes its directory: {name}")
    return path


def restore_runtime(output: Path = DEFAULT_OUTPUT, *, check: bool = False) -> dict:
    """Verify every source first; create missing copies without overwriting files."""
    output = Path(output).resolve()
    if ROOT.resolve().is_relative_to(output):
        raise ValueError("Runtime output must not be the repository or its parent")
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if document["schema_version"] != 1:
        raise ValueError("Unsupported restored-runtime manifest")
    pending = []
    for record in document["runtime_files"]:
        source = _relative_path(ROOT, record["source"])
        target = _relative_path(output, record["destination"])
        content = source.read_bytes()
        digest = record["sha256"]
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Restored source checksum mismatch: {record['source']}")
        if target.exists():
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Existing runtime file differs; refusing to overwrite: {target}")
        elif check:
            raise FileNotFoundError(f"Frozen runtime file is missing: {target}")
        else:
            pending.append((target, content))
    for target, content in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(content)
    return {"source_commit": document["source_commit"], "output": str(output),
            "verified_files": len(document["runtime_files"]), "created_files": len(pending)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Verify an existing runtime without writing")
    args = parser.parse_args()
    print(json.dumps(restore_runtime(args.output, check=args.check), indent=2))


if __name__ == "__main__":
    main()

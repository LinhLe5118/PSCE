"""Store a byte-exact, compact copy of a sealed learner execution plan."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    source = args.artifact_dir / "execution_plan.json"
    target = args.artifact_dir / "execution_plan.json.gz"
    archive_manifest = args.artifact_dir / "execution_plan_archive.json"
    if target.exists() or archive_manifest.exists():
        raise FileExistsError("Execution-plan archive already exists")
    expected = None
    for name, key in (("result_manifest.json", "execution_plan_sha256"),
                      ("execution_schedule_amendment.json", "execution_plan_sha256")):
        path = args.artifact_dir / name
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8-sig"))
            if key in row:
                if expected is not None and expected != row[key]:
                    raise RuntimeError("Plan hash differs across seals")
                expected = row[key]
    raw_hash = digest(source)
    if expected is not None and expected != raw_hash:
        raise RuntimeError("Raw plan differs from existing seal")
    temporary = target.with_suffix(target.suffix + ".writing")
    with source.open("rb") as reader, temporary.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0,
                           compresslevel=9) as zipper:
            shutil.copyfileobj(reader, zipper, length=1 << 20)
    temporary.replace(target)
    uncompressed = hashlib.sha256()
    with gzip.open(target, "rb") as reader:
        for chunk in iter(lambda: reader.read(1 << 20), b""):
            uncompressed.update(chunk)
    if uncompressed.hexdigest() != raw_hash:
        raise RuntimeError("Compressed plan does not restore the frozen bytes")
    archive_manifest.write_text(json.dumps({
        "schema": "byte_exact_execution_plan_archive_v1",
        "raw_sha256": raw_hash,
        "gzip_sha256": digest(target),
        "raw_bytes": source.stat().st_size,
        "gzip_bytes": target.stat().st_size,
        "restored_sha256_verified": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Archived {source} as {target.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()

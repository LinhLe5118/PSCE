"""Fresh-checkout source availability and sealed generator compatibility."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.restore_frozen_runtime import ROOT, restore_runtime


def test_runtime_reconstructs_the_sources_sealed_by_the_current_cohort(tmp_path):
    output = tmp_path / "frozen"
    result = restore_runtime(output)
    assert result["created_files"] > 0
    protocol = json.loads((ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04/protocol.json").read_text("utf-8"))
    prefix = "tmp/oeh_development_2026-09-30/"
    for path, digest in protocol["source_sha256"].items():
        if path.startswith(prefix):
            assert hashlib.sha256((output / path[len(prefix):]).read_bytes()).hexdigest() == digest
    assert restore_runtime(output, check=True)["created_files"] == 0
    assert restore_runtime(output)["created_files"] == 0


def test_runtime_preserves_an_existing_modified_file(tmp_path):
    output = tmp_path / "frozen"
    restore_runtime(output)
    changed = output / "scripts/data/generate_offline_datasets.py"
    changed.write_text("local edit\n", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        restore_runtime(output)
    assert changed.read_text("utf-8") == "local edit\n"


def test_current_runner_imports_the_frozen_generator_in_a_fresh_process():
    source = (
        "import json; from scripts.analysis.run_unseen_validation import archive_imports; "
        "pipeline, external, generator, seed_index = archive_imports(); "
        "print(json.dumps({'sha256': generator._generator_provenance()['source_sha256'], "
        "'generator': generator.__file__}))"
    )
    completed = subprocess.run([sys.executable, "-c", source], cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    protocol = json.loads((ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04/protocol.json").read_text("utf-8"))
    assert result["sha256"] == protocol["original_generator_sha256"]
    assert Path(result["generator"]).is_relative_to(ROOT / "tmp/oeh_development_2026-09-30")

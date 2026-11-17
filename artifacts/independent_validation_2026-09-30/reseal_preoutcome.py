"""Record a pre-outcome implementation repair without replacing the cohort."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[2]
folder = Path(__file__).resolve().parent
original = folder / "protocol.json"
archived = folder / "protocol_v1_failed_preoutcome.json"
runner = root / "scripts/analysis/run_unseen_validation.py"
features = folder / "pilot_features.csv"
dataset_seal = folder / "dataset_seal.json"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if archived.exists() or (folder / "choices_seal.json").exists():
    raise RuntimeError("Protocol revision already recorded or choices already sealed")
if (folder / "execution_plan.json").exists() or (folder / "execution/jobs").exists():
    raise RuntimeError("Cannot repair after any new training outcome")
if digest(original) != (folder / "protocol.sha256").read_text("ascii").strip():
    raise RuntimeError("Original protocol hash mismatch")
old = json.loads(original.read_text("utf-8"))
if len(features.read_text("utf-8").splitlines()) != 901:
    raise RuntimeError("Incomplete pilot-only checkpoint")
seal = json.loads(dataset_seal.read_text("utf-8"))
if seal["protocol_sha256"] != digest(original):
    raise RuntimeError("Original dataset/protocol seal mismatch")
archived.write_bytes(original.read_bytes())
old.update({
    "status": "v2_implementation_repair_after_pilot_before_any_training_outcome",
    "previous_protocol_sha256": digest(archived),
    "dataset_generated_under_protocol_sha256": seal["protocol_sha256"],
    "pilot_feature_sha256": digest(features),
    "pilot_feature_checkpoint_origin": "complete 900-row pilot-only CSV from v1 prepare, before any choices or training outcomes",
    "revision_reason": "Restore missing exact T1 contract and discard extraneous B2-only block at router interface; no cohort, method, endpoint, or threshold changed",
    "runner_sha256": digest(runner),
})
old["input_source_sha256"]["scripts/analysis/run_unseen_validation.py"] = digest(runner)
original.write_text(json.dumps(old, indent=2, allow_nan=False) + "\n", encoding="utf-8")
(folder / "protocol.sha256").write_text(digest(original) + "\n", encoding="ascii")
print("Archived v1", digest(archived))
print("Frozen v2", digest(original))

"""First-access boundary for the frozen Week 6 evaluation cohort.

The manifest is the first sealed input.  This module records that access before
reading it, validates the registered cohort and logical grid, and only then
permits callers to verify or open trajectory buffers.  It does not fit models.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from config.protocol_config import PROTOCOL_CONFIG
from environments.benchmark_environments import (
    DYNAMIC_RANGE_SEED_STRIDE, MAX_REGENERATION_ATTEMPTS, benchmark_specs,
    make_benchmark_mdp,
)
from environments.benchmark_panels import audit_dynamics
from evaluation.confirmatory import CONFIRMATORY_OUTPUT_DIR_NAME, CONFIRMATORY_RUN_ID
from evaluation.io import sha256_file, verified_dataset_path, write_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(value: str) -> str:
    digest = value.lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Expected a 64-character SHA-256 digest")
    return digest


def validate_evaluation_manifest(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Validate the complete registered evaluation grid without opening buffers."""

    if manifest.get("run_mode") not in {"protocol", "e1_confirmatory"} or manifest.get("protocol_deviations"):
        raise ValueError("E1 requires an undeviated protocol bundle")
    if manifest.get("run_mode") == "e1_confirmatory" and (
        manifest.get("e1_source_review") != "docs/week6/E1_EXECUTION_IMPLEMENTATION_REVIEW.json"
        or not manifest.get("e1_authorization_reference")
    ):
        raise ValueError("E1-only bundle is missing its execution-review provenance")
    for section in ("benchmark", "behavior_policies", "collection", "coverage", "estimator"):
        if manifest.get("protocol_config", {}).get(section) != PROTOCOL_CONFIG[section]:
            raise ValueError(f"Evaluation protocol_config.{section} differs from the frozen recipe")
        if manifest.get("effective_protocol", {}).get(section) != PROTOCOL_CONFIG[section]:
            raise ValueError(f"Evaluation effective_protocol.{section} differs from the frozen recipe")
    if (int(manifest.get("instance_count", -1)) != len(manifest.get("instances", ()))
            or int(manifest.get("logical_dataset_count", -1)) != len(manifest.get("datasets", ()))):
        raise ValueError("Evaluation manifest declared counts disagree with its records")
    expected_specs = {spec.instance_id: spec for spec in benchmark_specs() if spec.split == "evaluation"}
    instances = [item for item in manifest.get("instances", ()) if item.get("split") == "evaluation"]
    if len(instances) != 120 or {str(item.get("instance_id")) for item in instances} != set(expected_specs):
        raise ValueError("Evaluation instance quota or identities differ from the frozen 120-MDP recipe")
    by_id = {str(item["instance_id"]): item for item in instances}
    families = Counter()
    topology_keys: set[tuple[str, int]] = set()
    for instance_id, item in by_id.items():
        spec = expected_specs[instance_id]
        for field in ("family", "ordinal", "family_ordinal"):
            if item.get(field) != getattr(spec, field):
                raise ValueError(f"Evaluation recipe mismatch: {instance_id}/{field}")
        rejections = int(item.get("dynamic_range_rejections", -1))
        if (int(item.get("requested_seed", -1)) != spec.seed
                or rejections < 0 or rejections >= MAX_REGENERATION_ATTEMPTS
                or int(item["seed"]) != spec.seed + rejections * DYNAMIC_RANGE_SEED_STRIDE):
            raise ValueError(f"Evaluation dynamic-range seed mismatch: {instance_id}")
        alias = (str(item["family"]), int(item["seed"]))
        if alias in topology_keys:
            raise ValueError(f"Base-MDP alias: {alias}")
        topology_keys.add(alias)
        families[item["family"]] += 1
    registered_families = PROTOCOL_CONFIG["benchmark"]["evaluation_family_counts"]
    if dict(families) != registered_families:
        raise ValueError(f"Evaluation family quota mismatch: {dict(families)}")
    other_ids = {str(item.get("instance_id")) for item in manifest.get("instances", ())
                 if item.get("split") != "evaluation"}
    if other_ids & set(by_id):
        raise ValueError("Development/evaluation instance identity overlap")

    collection = PROTOCOL_CONFIG["collection"]
    policy_families = set(PROTOCOL_CONFIG["behavior_policies"]["families"])
    expected_cells = {
        (stream, int(seed), int(size))
        for stream in ("pilot", "training")
        for seed in collection[f"{stream}_seed_indices"]
        for size in collection["pilot_size_multipliers" if stream == "pilot" else "size_multipliers"]
    }
    cells: dict[tuple[str, str], set[tuple[str, int, int]]] = defaultdict(set)
    physical: dict[tuple[str, str, str, int], set[tuple[str, str, int]]] = defaultdict(set)
    file_owners: dict[str, tuple[str, str, str, int]] = {}
    dataset_ids: set[str] = set()
    evaluation_records = 0
    for record in manifest.get("datasets", ()):
        if record.get("benchmark_split") != "evaluation":
            if str(record.get("instance_id")) in by_id:
                raise ValueError("Evaluation MDP record is marked as another split")
            continue
        evaluation_records += 1
        instance_id = str(record.get("instance_id"))
        policy_family = str(record.get("policy_family"))
        if instance_id not in by_id or policy_family not in policy_families:
            raise ValueError("Evaluation record has an unregistered MDP or behavior policy")
        stream = str(record.get("stream"))
        if stream not in ("pilot", "training"):
            raise ValueError("Evaluation record has an unregistered stream")
        seed_offset = int(collection[f"{stream}_seed_offset"])
        stream_seed = int(record["stream_seed"])
        seed_index = (stream_seed - int(by_id[instance_id]["seed"]) - seed_offset) % (2**32 - 1)
        size = int(record["size_multiplier"])
        cell = (stream, seed_index, size)
        if cell not in expected_cells or int(record["mdp_seed"]) != int(by_id[instance_id]["seed"]):
            raise ValueError(f"Evaluation stream seed or budget mismatch: {instance_id}/{cell}")
        if cell in cells[(instance_id, policy_family)]:
            raise ValueError(f"Duplicate evaluation logical cell: {instance_id}/{policy_family}/{cell}")
        cells[(instance_id, policy_family)].add(cell)
        states, actions = int(record["num_states"]), int(record["num_actions"])
        if (states != int(by_id[instance_id]["num_states"])
                or actions != int(by_id[instance_id]["num_actions"])
                or int(record["size"]) != size * states * actions
                or record.get("slice") != [0, int(record["size"])]):
            raise ValueError(f"Evaluation prefix/schema mismatch: {instance_id}/{cell}")
        max_size = max(collection["pilot_size_multipliers" if stream == "pilot" else "size_multipliers"])
        if int(record["max_size"]) != max_size * states * actions:
            raise ValueError(f"Evaluation physical buffer size mismatch: {instance_id}/{cell}")
        file_name = str(record.get("file", ""))
        if not file_name or Path(file_name).name != file_name:
            raise ValueError("Evaluation buffer name must be a local filename")
        digest = _sha256(str(record.get("sha256", "")))
        owner = (instance_id, policy_family, stream, seed_index)
        if file_name in file_owners and file_owners[file_name] != owner:
            raise ValueError("Evaluation buffer filename is shared across independent streams")
        file_owners[file_name] = owner
        physical[owner].add((file_name, digest, int(record["max_size"])))
        dataset_id = str(record.get("dataset_id", ""))
        if not dataset_id or dataset_id in dataset_ids:
            raise ValueError("Missing or duplicate evaluation dataset_id")
        dataset_ids.add(dataset_id)
    expected_groups = {(instance_id, family) for instance_id in by_id for family in policy_families}
    if set(cells) != expected_groups or any(actual != expected_cells for actual in cells.values()):
        raise ValueError("Evaluation behavior/seed/budget grid is incomplete")
    if any(len(identities) != 1 for identities in physical.values()):
        raise ValueError("Nested prefixes disagree on their physical buffer")
    if evaluation_records != 120 * len(policy_families) * len(expected_cells):
        raise ValueError("Evaluation logical record count mismatch")
    if len(physical) != 120 * len(policy_families) * (
        len(collection["pilot_seed_indices"]) + len(collection["training_seed_indices"])
    ):
        raise ValueError("Evaluation physical buffer count mismatch")
    if int(manifest.get("physical_buffer_count", -1)) != len(physical):
        raise ValueError("Evaluation manifest declared physical buffer count mismatch")
    return {
        "base_mdps": len(by_id), "behavior_groups": len(cells),
        "logical_records": evaluation_records, "physical_buffers": len(physical),
        "training_prefixes": sum(1 for record in manifest["datasets"]
                                 if record.get("benchmark_split") == "evaluation"
                                 and record.get("stream") == "training"),
    }


def validate_base_mdp_aliases(manifest: Mapping[str, Any]) -> int:
    """Reconstruct frozen recipes and reject duplicate exact MDPs before buffers."""

    seen: dict[str, str] = {}
    checked = 0
    for item in manifest["instances"]:
        if item.get("split") != "evaluation":
            continue
        mdp = make_benchmark_mdp(type("FrozenSpec", (), dict(item))())
        if (mdp.num_states != int(item["num_states"])
                or mdp.num_actions != int(item["num_actions"])):
            raise ValueError(f"Reconstructed E1 MDP dimensions disagree: {item['instance_id']}")
        digest = hashlib.sha256()
        for array in (mdp.P, mdp.R, mdp.initial_state_distribution):
            digest.update(array.tobytes())
        digest.update(json.dumps(list(mdp.terminal_states)).encode("ascii"))
        identity = digest.hexdigest()
        if identity in seen:
            raise ValueError(f"Exact base-MDP alias: {seen[identity]} and {item['instance_id']}")
        seen[identity] = str(item["instance_id"])
        checked += 1
    return checked


def write_or_verify_dynamics_audit(
    manifest: Mapping[str, Any], output_dir: Path, *, manifest_sha256: str,
    resume: bool,
) -> Path:
    """Verify every registered panel from its kernel and seal the oracle sidecar."""

    rows = []
    for item in manifest["instances"]:
        if item.get("split") != "evaluation":
            continue
        mdp = make_benchmark_mdp(type("FrozenSpec", (), dict(item))())
        audit = audit_dynamics(mdp, str(item["family"]))
        if item.get("dynamics_audit") is not None and item["dynamics_audit"] != audit:
            raise ValueError(f"Sealed dynamics audit differs from reconstructed kernel: {item['instance_id']}")
        rows.append({"instance_id": str(item["instance_id"]), **audit})
    rows.sort(key=lambda row: row["instance_id"])
    if len(rows) != 120:
        raise ValueError("E1 dynamics sidecar requires all 120 evaluation MDPs")
    payload = {"scope": "oracle_only_not_estimator_input",
               "manifest_sha256": _sha256(manifest_sha256), "instances": rows}
    path = output_dir / "dynamics_audit_sidecar.json"
    if path.exists():
        if not resume or json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("E1 dynamics sidecar differs on resume")
    else:
        write_json(path, payload)
    return path


def open_evaluation_bundle(
    dataset_dir: Path, output_dir: Path, *, expected_manifest_sha256: str,
    authorization_reference: str, source_commit: str,
    source_sha256: Mapping[str, str], resume: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Log first access, verify the seal and full grid, then verify every buffer."""

    expected = _sha256(expected_manifest_sha256)
    commit = source_commit.lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("E1 first access requires a full Git source commit")
    if not source_sha256 or any(_sha256(str(value)) != str(value).lower()
                                for value in source_sha256.values()):
        raise ValueError("E1 first access requires source SHA-256 checksums")
    if not authorization_reference.strip():
        raise ValueError("E1 requires a recorded explicit authorization reference")
    if output_dir.name != CONFIRMATORY_OUTPUT_DIR_NAME:
        raise ValueError("E1 output directory differs from the frozen run ID")
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Sealed evaluation manifest is absent: {manifest_path}")
    log_path = output_dir / "first_access_log.json"
    if resume and not (output_dir.exists() and any(output_dir.iterdir())):
        raise FileNotFoundError("Verified E1 resume requires an existing nonempty output")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume or not log_path.is_file():
            raise FileExistsError("E1 output exists; only verified resume is allowed")
        log = json.loads(log_path.read_text(encoding="utf-8"))
        retryable_first_access = (
            log.get("status") == "opened_pending_validation"
            or (log.get("status") == "rejected"
                and log.get("error_type") in {"OSError", "TimeoutError",
                                              "ConnectionError", "InterruptedError"})
        )
        if (log.get("run_id") != CONFIRMATORY_RUN_ID
                or log.get("expected_manifest_sha256") != expected
                or log.get("authorization_reference") != authorization_reference
                or log.get("dataset_dir") != str(dataset_dir.resolve())
                or log.get("source_commit") != commit
                or log.get("source_sha256") != dict(source_sha256)
                or (log.get("status") != "verified" and not retryable_first_access)):
            raise ValueError("E1 first-access log identity mismatch on resume")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        log = {
            "run_id": CONFIRMATORY_RUN_ID,
            "first_access_utc": _now(),
            "authorization_reference": authorization_reference,
            "dataset_dir": str(dataset_dir.resolve()),
            "expected_manifest_sha256": expected,
            "source_commit": commit,
            "source_sha256": dict(source_sha256),
            "status": "opened_pending_validation",
        }
        write_json(log_path, log)
    try:
        observed = sha256_file(manifest_path)
        if observed != expected:
            raise ValueError("Sealed evaluation manifest SHA-256 mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts = validate_evaluation_manifest(manifest)
        if manifest.get("run_mode") == "e1_confirmatory":
            generator_path = Path(__file__).resolve().parents[1] / "scripts/data/generate_week6_e1_bundle.py"
            if (manifest.get("e1_authorization_reference") != authorization_reference
                    or manifest.get("e1_generator_sha256") != sha256_file(generator_path)):
                raise ValueError("E1-only bundle generation authorization/source mismatch")
            generation_log = json.loads((dataset_dir / "generation_access_log.json").read_text(encoding="utf-8"))
            if (generation_log.get("status") != "sealed_bundle_generated"
                    or generation_log.get("manifest_sha256") != expected
                    or generation_log.get("authorization_reference") != authorization_reference
                    or generation_log.get("source_commit") != commit
                    or generation_log.get("source_sha256") != dict(source_sha256)):
                raise ValueError("E1 generation access log or seal mismatch")
        if validate_base_mdp_aliases(manifest) != counts["base_mdps"]:
            raise ValueError("Base-MDP reconstruction count mismatch")
        dynamics_path = write_or_verify_dynamics_audit(
            manifest, output_dir, manifest_sha256=expected, resume=resume,
        )
        index_name = str(manifest.get("estimator_input_index", ""))
        if Path(index_name).name != index_name or not index_name:
            raise ValueError("Data-only index path is invalid")
        index_path = (dataset_dir / index_name).resolve()
        if not index_path.is_relative_to(dataset_dir.resolve()):
            raise ValueError("Data-only index escapes the sealed bundle")
        if sha256_file(index_path) != _sha256(str(manifest.get("estimator_input_index_sha256", ""))):
            raise ValueError("Data-only index SHA-256 mismatch")
        verified: set[str] = set()
        for record in manifest["datasets"]:
            if record.get("benchmark_split") != "evaluation":
                continue
            name = str(record["file"])
            if name not in verified:
                verified_dataset_path(dataset_dir, record)
                verified.add(name)
        if len(verified) != counts["physical_buffers"]:
            raise ValueError("Verified buffer count mismatch")
    except Exception as exc:
        log.update(status="rejected", failed_at_utc=_now(), error_type=type(exc).__name__, error=str(exc))
        write_json(log_path, log)
        raise
    log.update(status="verified", verified_at_utc=_now(), observed_manifest_sha256=observed,
               data_only_index_sha256=sha256_file(index_path),
               dynamics_audit_sha256=sha256_file(dynamics_path), counts=counts)
    write_json(log_path, log)
    return manifest, counts


__all__ = ["validate_evaluation_manifest", "validate_base_mdp_aliases",
           "write_or_verify_dynamics_audit", "open_evaluation_bundle"]

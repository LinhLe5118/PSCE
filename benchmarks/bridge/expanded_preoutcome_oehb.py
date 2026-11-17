# Generated from the sealed fresh pipeline by materialize_expanded_pipeline.py.
"""Checkpointed full 64-draw OEH-B on the fresh, label-free BRIDGE panel."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from oeh import predictive  # noqa: E402
from .fast_backend import install  # noqa: E402

PANEL = ROOT / "benchmarks/bridge/expanded_panel_2026-10-08"
DATA = ROOT / "datasets/bridge_v1_1_1/offline_pilot_expanded_2026-10-08"
OUTPUT = ROOT / "artifacts/week8/bridge_expanded_full_oehb_preoutcome_2026-10-08"
MEMORY_ADDENDUM = PANEL / "prelabel_posterior_memory_addendum.json"
BUDGETS = (1, 2, 5, 10, 20, 50)
POSTERIOR_CHUNK_DRAWS = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_one(task: dict) -> dict:
    row, protocol = task["row"], task["protocol"]
    output = Path(task["output"])
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if (saved.get("pilot_sha256") != row["sha256"] or
                saved.get("protocol_sha256") != task["protocol_sha256"] or
                saved.get("source_script_sha256") != task["source_script_sha256"] or
                saved.get("backend_sha256") != task["backend_sha256"] or
                saved.get("memory_addendum_sha256") != task["memory_addendum_sha256"] or
                saved.get("posterior_chunk_draws") != POSTERIOR_CHUNK_DRAWS):
            raise RuntimeError(f"Stale OEH-B checkpoint: {output.name}")
        return {"status": saved["status"], "mdp_id": row["mdp_id"],
                "policy_family": row["policy_family"], "seconds": saved.get("runtime_seconds", 0)}
    started = time.perf_counter()
    try:
        pilot_path = DATA / row["file"]
        if sha256(pilot_path) != row["sha256"]:
            raise RuntimeError("Frozen pilot hash changed")
        with np.load(pilot_path, allow_pickle=False) as loaded:
            pilot = {key: loaded[key] for key in loaded.files}
        S, A, H = int(row["num_states"]), int(row["num_actions"]), int(row["horizon"])
        if H != int(protocol["episode_horizon"][row["mdp_id"]]) or len(pilot["state"]) != 10*S*A:
            raise RuntimeError("Frozen task horizon or pilot size changed")
        install()
        predictive.CHUNK_DRAWS = POSTERIOR_CHUNK_DRAWS
        seed_digest = hashlib.sha256(
            f"bridge-fresh-oehb-v1|{row['mdp_id']}|{row['policy_family']}".encode()).digest()
        options = protocol["posterior"]
        outcomes = predictive.posterior_outcomes(
            pilot, row["initial_state_distribution"], num_states=S, num_actions=A,
            gamma=float(protocol["gamma"]), draws=int(options["draws"]),
            seeds=int(options["seeds"]), budgets=BUDGETS,
            horizon=H, evaluation_horizon=H,
            seed=[int.from_bytes(seed_digest[:8], "little")],
            minimum_dynamic_range=float(options["minimum_dynamic_range"]),
            reward_bound=1.0, **options["structure"])
        forecast = predictive.requirement_forecast(outcomes)
        nr = outcomes.normalized_return
        result = {"status": "FRESH_FULL_OEHB_PREOUTCOME_ONLY", "mdp_id": row["mdp_id"],
                  "policy_family": row["policy_family"], "pilot_sha256": row["sha256"],
                  "source_npz_sha256": row["source_npz_sha256"],
                  "protocol_sha256": task["protocol_sha256"],
                  "source_script_sha256": task["source_script_sha256"],
                  "backend_sha256": task["backend_sha256"],
                  "memory_addendum_sha256": task["memory_addendum_sha256"],
                  "posterior_chunk_draws": POSTERIOR_CHUNK_DRAWS,
                  "states": S, "actions": A, "horizon": H,
                  "draws": int(options["draws"]), "posterior": outcomes.posterior,
                  "sampling": outcomes.sampling,
                  "learners": {name: {
                      "probabilities": value["probabilities"].tolist(),
                      "events": value["events"].tolist(),
                      "median_event": value["median_event"],
                      "censored_fraction": value["censored_fraction"],
                      "mean_normalized_return": nr[:, :, :, i].mean((0, 1)).tolist() if len(nr) else None,
                      "success_rate": (nr[:, :, :, i] >= .95).mean((0, 1)).tolist() if len(nr) else None,
                  } for i, (name, value) in enumerate(forecast.items())},
                  "runtime_seconds": time.perf_counter() - started,
                  "test_labels_read": False}
    except Exception as exc:
        result = {"status": "FRESH_FULL_OEHB_PREOUTCOME_FAILURE", "mdp_id": row["mdp_id"],
                  "policy_family": row["policy_family"], "pilot_sha256": row["sha256"],
                  "protocol_sha256": task["protocol_sha256"],
                  "source_script_sha256": task["source_script_sha256"],
                  "backend_sha256": task["backend_sha256"],
                  "memory_addendum_sha256": task["memory_addendum_sha256"],
                  "posterior_chunk_draws": POSTERIOR_CHUNK_DRAWS,
                  "error": repr(exc), "traceback": traceback.format_exc(),
                  "runtime_seconds": time.perf_counter() - started,
                  "test_labels_read": False}
    temporary = output.with_suffix(".json.part")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return {"status": result["status"], "mdp_id": row["mdp_id"],
            "policy_family": row["policy_family"], "seconds": result["runtime_seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise ValueError("At most two GPU workers on this 4 GiB laptop")
    protocol_file, pilot_file = PANEL / "protocol.json", DATA / "manifest.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_file.read_text(encoding="utf-8"))
    addendum = json.loads(MEMORY_ADDENDUM.read_text(encoding="utf-8"))
    script_hash = sha256(Path(__file__))
    if (addendum["status"] != "PRELABEL_POSTERIOR_MEMORY_AMENDMENT" or
            addendum["test_labels_read"] or
            addendum["protocol_sha256"] != sha256(protocol_file) or
            addendum["oehb_script_sha256"] != script_hash or
            addendum["predictive_source_sha256"] != sha256(Path(predictive.__file__)) or
            addendum["single_chunk_test_sha256"] != sha256(ROOT / "tests/test_oeh_final.py") or
            addendum["posterior_chunk_draws"] != POSTERIOR_CHUNK_DRAWS or
            addendum["total_posterior_draws"] != int(protocol["posterior"]["draws"])):
        raise RuntimeError("Posterior memory amendment mismatch")
    if (protocol["status"] != "FRESH_BRIDGE_PROTOCOL_FROZEN_BEFORE_LABELS" or
            pilot["status"] != "SEALED_FRESH_PILOT_ONLY_NO_TEST_LABELS" or
            pilot["protocol_sha256"] != sha256(protocol_file) or
            len(pilot["pilot_groups"]) != 25 or
            set(pilot["tasks"]) != set(protocol["evaluation_tasks"])):
        raise RuntimeError("Fresh preoutcome pilot/protocol gate failed")
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("Fresh OEH-B manifest already sealed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    group_dir = OUTPUT / "groups"
    group_dir.mkdir(exist_ok=True)
    backend_hash = sha256(ROOT / "benchmarks/bridge/fast_backend.py")
    tasks = []
    for row in pilot["pilot_groups"]:
        stem = hashlib.sha256(f"{row['mdp_id']}|{row['policy_family']}".encode()).hexdigest()[:24]
        tasks.append({"row": row, "protocol": protocol,
                      "output": str(group_dir / f"{stem}.json"),
                      "protocol_sha256": sha256(protocol_file),
                      "source_script_sha256": script_hash, "backend_sha256": backend_hash,
                      "memory_addendum_sha256": sha256(MEMORY_ADDENDUM)})
    tasks.sort(key=lambda item: int(item["row"]["num_states"]), reverse=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(run_one, task): task for task in tasks}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            print(f"Fresh OEH-B {len(results)}/25 {result['mdp_id']} / {result['policy_family']}: "
                  f"{result['status']} {result['seconds']:.1f}s", flush=True)
    failures = [r for r in results if r["status"] != "FRESH_FULL_OEHB_PREOUTCOME_ONLY"]
    if failures:
        (OUTPUT / "failure_report.json").write_text(json.dumps({
            "status": "NO_GO_INCOMPLETE_FRESH_FULL_OEHB", "failures": failures},
            indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Fresh OEH-B failed for {len(failures)} groups")
    manifest = {"status": "SEALED_FRESH_FULL_OEHB_PREOUTCOME_NO_TEST_LABELS",
                "pilot_manifest_sha256": sha256(pilot_file),
                "protocol_sha256": sha256(protocol_file),
                "source_script_sha256": script_hash, "backend_sha256": backend_hash,
                "memory_addendum_sha256": sha256(MEMORY_ADDENDUM),
                "posterior_chunk_draws": POSTERIOR_CHUNK_DRAWS,
                "groups": 25, "test_labels_read": False,
                "runtime_seconds_sum": sum(r["seconds"] for r in results),
                "group_sha256": {Path(t["output"]).name: sha256(Path(t["output"])) for t in tasks}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Sealed 25/25 fresh full OEH-B groups before labels", flush=True)


if __name__ == "__main__":
    main()

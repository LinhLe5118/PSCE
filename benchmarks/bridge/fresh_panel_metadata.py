"""Freeze an untouched BRIDGE sticky panel using only published size metadata.

Selects all base Atari/Procgen tasks with at most 1200 sticky states. It never
reads rewards, author method scores, OEH predictions, or offline learner labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "fresh_panel_2026-10-08"
SOURCE = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
OLD_SPLIT = ROOT / "artifacts/week8/bridge_transfer_preflight_freeze_2026-10-07.json"
FIELDS = ("mdp_name", "origin", "base_states", "sticky_states", "actions", "horizon",
          "dense_p_r_mib", "pilot_x10_transitions", "x50_transitions_per_seed")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(name: str) -> list[dict[str, str]]:
    file = HERE / "metadata" / name
    if hashlib.md5(file.read_bytes()).hexdigest() != SOURCE["source_csv_md5"][name]:
        raise RuntimeError(f"Published metadata hash mismatch: {name}")
    with file.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    old = json.loads(OLD_SPLIT.read_text(encoding="utf-8"))
    unavailable = set(old["development_tasks"]) | set(old["candidate_test_tasks"])
    base = {r["mdp_name"]: r for r in read_rows("summary.csv")
            if r["origin"] in {"atari", "procgen"} and r["type"] == "base"}
    sticky = {r["mdp_name"]: r for r in read_rows("summary_sticky.csv")
              if r["origin"] in {"atari", "procgen"}}
    selected = []
    for name in sorted(base.keys() & sticky.keys()):
        old_row, new_row = base[name], sticky[name]
        states, actions, horizon = map(int, (new_row["num_states"],
                                           new_row["num_actions"], new_row["horizon"]))
        if name in unavailable or states > 1200 or horizon > 40:
            continue
        if (actions != int(old_row["num_actions"]) or
                horizon != int(old_row["horizon"])):
            raise RuntimeError(f"Inconsistent source metadata for {name}")
        selected.append({
            "mdp_name": name, "origin": old_row["origin"],
            "base_states": int(old_row["num_states_consolidated"]),
            "sticky_states": states, "actions": actions, "horizon": horizon,
            "dense_p_r_mib": round(2 * states * states * actions * 8 / 2**20, 2),
            "pilot_x10_transitions": 10 * states * actions,
            "x50_transitions_per_seed": 50 * states * actions,
        })
    if len(selected) != 8 or set(r["mdp_name"] for r in selected) & unavailable:
        raise RuntimeError("Expected eight unseen low-resource Atari/Procgen MDPs")
    OUT.mkdir(parents=True)
    csv_path = OUT / "candidate_registry.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    manifest = {
        "status": "FRESH_PANEL_METADATA_FROZEN_NO_LABELS",
        "source": SOURCE["source_record"],
        "source_version": SOURCE["source_dataset_readme_version"],
        "rule": "base Atari/Procgen; sticky S<=1200; H<=40; exclude all old development/test tasks; no outcome columns",
        "task_names": [r["mdp_name"] for r in selected],
        "task_count": len(selected), "old_split_sha256": sha256(OLD_SPLIT),
        "source_metadata_md5": SOURCE["source_csv_md5"],
        "registry_sha256": sha256(csv_path),
        "source_script_sha256": sha256(Path(__file__)),
        "test_labels_read": False,
        "warning": "Metadata candidate panel only; adapter, Markov/horizon equivalence, full OEH runtime and preoutcome seal still required before any labels.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(selected)} untouched BRIDGE candidate tasks: {manifest['task_names']}")


if __name__ == "__main__":
    main()

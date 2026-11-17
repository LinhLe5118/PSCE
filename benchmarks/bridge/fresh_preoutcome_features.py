"""Generate the frozen 30 pilot-only B2/legacy/context features on fresh tasks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from . import preoutcome_features as old
from .fresh_adapter import PANEL
from .fresh_prepare_pilots import OUTPUT as DATA

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/week8/bridge_fresh_pilot_features_preoutcome_2026-10-08"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    pilot_file, protocol_file = DATA / "manifest.json", PANEL / "protocol.json"
    pilot = json.loads(pilot_file.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if (pilot["status"] != "SEALED_FRESH_PILOT_ONLY_NO_TEST_LABELS" or
            pilot["protocol_sha256"] != sha256(protocol_file) or
            len(pilot["pilot_groups"]) != 25 or protocol["test_labels_read"]):
        raise RuntimeError("Fresh pilot feature gate failed")
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("Fresh pilot features already sealed")
    names = json.loads((ROOT / "docs/protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0.json")
                       .read_text(encoding="utf-8"))["feature_sets"]["B2_OEH3_LCOEH4"]
    old.DATA = DATA  # same frozen feature formulas; only the pilot namespace differs
    OUTPUT.mkdir(parents=True, exist_ok=True)
    groups = OUTPUT / "groups"
    groups.mkdir(exist_ok=True)
    rows = []
    for index, record in enumerate(pilot["pilot_groups"], 1):
        stem = hashlib.sha256(f"{record['mdp_id']}|{record['policy_family']}".encode()).hexdigest()[:24]
        path = groups / f"{stem}.json"
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
            if (item.get("pilot_sha256") != record["sha256"] or
                    item.get("source_script_sha256") != sha256(Path(__file__)) or
                    item.get("feature_formula_sha256") != sha256(Path(old.__file__))):
                raise RuntimeError(f"Stale feature checkpoint: {stem}")
        else:
            item = old.calculate(record, names)
            item.update({"source_script_sha256": sha256(Path(__file__)),
                         "feature_formula_sha256": sha256(Path(old.__file__)),
                         "test_labels_read": False})
            temporary = path.with_suffix(".json.part")
            temporary.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        if item["status"] != "PILOT_FEATURES_PREOUTCOME_ONLY":
            raise RuntimeError(f"Pilot feature failure: {stem}")
        rows.extend(item["rows"])
        print(f"Fresh pilot features {index}/25: {record['mdp_id']} / {record['policy_family']} "
              f"{item['total_seconds']:.1f}s", flush=True)
    frame = pd.DataFrame(rows).sort_values(["mdp_id", "policy_family", "algorithm"])
    if len(frame) != 75 or frame[["mdp_id", "policy_family", "algorithm"]].duplicated().any():
        raise RuntimeError("Incomplete fresh pilot features")
    output = OUTPUT / "features.csv"
    frame.to_csv(output, index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "SEALED_FRESH_30_COLUMN_PILOT_FEATURES_NO_TEST_LABELS",
        "pilot_manifest_sha256": sha256(pilot_file), "protocol_sha256": sha256(protocol_file),
        "source_script_sha256": sha256(Path(__file__)),
        "feature_formula_sha256": sha256(Path(old.__file__)),
        "groups": 25, "rows": 75, "feature_columns": len(names),
        "features_sha256": sha256(output), "test_labels_read": False,
        "legacy_depth_note": "The frozen legacy OEH3 diagnostic retains k_max=100 as in the trained feature map; the primary OEH-H block uses task-specific horizon separately."},
        indent=2) + "\n", encoding="utf-8")
    print("Sealed 75 fresh pilot feature rows", flush=True)


if __name__ == "__main__":
    main()

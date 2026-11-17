"""Make an isolated, auditable copy of the sealed five-task BRIDGE pipeline.

The scientific formulas and fixed five-task shape are retained. Only registry,
pilot/output namespaces and imports for the new eligible-MDP loader change.
The old sealed modules and artifacts are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PANEL = HERE / "expanded_panel_2026-10-08"
MODULES = (
    "prepare_pilots", "preoutcome_features", "preoutcome_horizon",
    "preoutcome_oehb", "preoutcome_full_forecast", "preoutcome_b2_forecast",
    "published_survival_forecast", "published_survival_full_forecast",
    "original_scores_preoutcome", "prelabel_seal", "run_test_labels",
    "score_test", "reference_preflight",
)
REWRITES = (
    (".fresh_published_survival_full_forecast", ".expanded_published_survival_full_forecast"),
    (".fresh_published_survival_forecast", ".expanded_published_survival_forecast"),
    (".fresh_prepare_pilots", ".expanded_prepare_pilots"),
    (".fresh_reward_gate", ".expanded_preflight"),
    (".fresh_adapter", ".expanded_adapter"),
    ("load_fresh_sticky_mdp", "load_expanded_sticky_mdp"),
    ("benchmarks/bridge/fresh_run_test_labels.py", "benchmarks/bridge/expanded_run_test_labels.py"),
    ("fresh_panel_2026-10-08", "expanded_panel_2026-10-08"),
    ("offline_pilot_fresh_2026-10-08", "offline_pilot_expanded_2026-10-08"),
    ("bridge_fresh_", "bridge_expanded_"),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    frozen = json.loads((PANEL / "manifest.json").read_text(encoding="utf-8"))
    preflight = json.loads((PANEL / "adapter_reward_preflight.json").read_text(encoding="utf-8"))
    if (frozen["task_count"] != 5 or len(preflight["tasks"]) != 5 or
            any(row["status"] != "ELIGIBLE_NONNEGATIVE" for row in preflight["tasks"])):
        raise ValueError("The expanded panel is not five fully eligible tasks")
    output = PANEL / "generated_pipeline_manifest.json"
    previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
    previous_hashes = ({row["module"]: row["expanded_sha256"] for row in previous["modules"]}
                       if previous is not None else {})
    generated = []
    for name in MODULES:
        source = HERE / f"fresh_{name}.py"
        target = HERE / f"expanded_{name}.py"
        original = source.read_text(encoding="utf-8")
        rewritten = original
        for before, after in REWRITES:
            rewritten = rewritten.replace(before, after)
        if rewritten == original:
            raise ValueError(f"No namespace substitutions in {source}")
        content = ("# Generated from the sealed fresh pipeline by materialize_expanded_pipeline.py.\n"
                   + rewritten).encode("utf-8")
        if target.exists():
            if target.read_bytes() != content:
                if sha256(target.read_bytes()) != previous_hashes.get(name):
                    raise ValueError(f"Refusing to replace modified expanded module: {target}")
                target.write_bytes(content)
        else:
            target.write_bytes(content)
        generated.append({"module": name, "source_sha256": sha256(source.read_bytes()),
                          "expanded_sha256": sha256(content)})
    payload = {"status": "EXPANDED_PIPELINE_PINNED_NO_LABELS",
               "generator_sha256": sha256(Path(__file__).read_bytes()),
               "frozen_manifest_sha256": sha256((PANEL / "manifest.json").read_bytes()),
               "preflight_sha256": sha256((PANEL / "adapter_reward_preflight.json").read_bytes()),
               "modules": generated, "test_labels_read": False}
    content = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    if output.exists():
        if previous["status"] != "EXPANDED_PIPELINE_PINNED_NO_LABELS" or previous["test_labels_read"]:
            raise ValueError("Existing expanded pipeline is no longer prelabel")
        output.write_bytes(content)
    else:
        output.write_bytes(content)
    print(f"Pinned {len(generated)} expanded modules before new labels")


if __name__ == "__main__":
    main()

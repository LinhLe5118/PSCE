"""Step 1: the protocols, pre-outcome seals and labels of the three final tests are the committed bytes."""
from __future__ import annotations

from pathlib import Path

from scripts.reproduce.common import ART, ROOT, TESTS, load_json, sha256, write_json

SPEC = ROOT / "artifacts/week8/oeh_v2_development_2026-10-14/oeh_v2_spec.json"


def check() -> dict:
    report = {"oeh_spec_sha256": sha256(SPEC), "tests": {}}
    for test, t in TESTS.items():
        folder = ART / t["folder"]
        protocol = sha256(folder / "protocol.json")
        rows = {"protocol": protocol == (folder / "protocol.sha256").read_text(encoding="ascii").strip()}
        for panel in t["panels"]:
            target = folder / panel
            seal = load_json(target / "preoutcome_seal.json")
            run = load_json(target / "outcome_run.json")
            files = {name: sha256(target / name) == digest for name, digest in seal["artifact_sha256"].items()}
            rows[panel] = {
                "sealed files match": all(files.values()),
                "seal written before any learner outcome": seal["learner_outcomes_observed"] == 0,
                "seal refers to this protocol": seal["protocol_sha256"] == protocol,
                "labels were opened after this seal": run["preoutcome_seal_sha256"] == sha256(target / "preoutcome_seal.json"),
                "labels are the opened ones": run["labels_sha256"] == sha256(target / "labels.csv"),
            }
            if not all(files.values()):
                rows[panel]["changed files"] = sorted(k for k, ok in files.items() if not ok)
        report["tests"][test] = rows
    report["all_ok"] = all(v if isinstance(v, bool) else all(x for x in v.values() if isinstance(x, bool))
                           for rows in report["tests"].values() for v in rows.values())
    return report


def run(out: Path, log) -> dict:
    target = out / "1_seals.json"
    report = check()
    write_json(target, report)
    for test, rows in report["tests"].items():
        bad = [f"{panel}: {k}" for panel, v in rows.items() if isinstance(v, dict) for k, ok in v.items() if ok is False]
        log(f"  {test}: protocol {'ok' if rows['protocol'] else 'CHANGED'}; " + ("seals ok" if not bad else "FAILED " + "; ".join(bad)))
    if not report["all_ok"]:
        raise SystemExit("STOP: a sealed file differs from the committed bytes (see 1_seals.json)")
    return report

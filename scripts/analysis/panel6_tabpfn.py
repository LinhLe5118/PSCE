"""TabPFN v2 on B2 for panel 6: the conditional descriptive method registered in the frozen protocol.

The protocol (``artifacts/oeh_panel6_2026-10-15/protocol.json``, descriptive.conditional_methods.tabpfn) adds these
forecasts and seals them separately before the label step (checkpoint H3). The forecaster is exactly the T3 spec
(``docs/week8/T3_TABPFN_SPEC_2026-10-15.md``, ``scripts/analysis/baseline_tabpfn_t3.py``). This script refuses to run
when any outcome or label file of the panel exists. Built with PriorLabs-TabPFN (Hollmann et al., 2025).

Usage: ``python -m scripts.analysis.panel6_tabpfn --panel mdpp`` (then ``--panel labmaze``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.analysis import baseline_tabpfn_t3 as T
from scripts.analysis import run_oeh_panel6_20261015 as S6

E = S6.E
KEYS, P = S6.KEYS, S6.P


def seal(panel: str) -> dict:
    doc = S6.protocol()
    if "tabpfn" not in doc["descriptive"]["conditional_methods"]:
        raise RuntimeError("TabPFN is not registered in the protocol")
    target = S6.panel_dir(panel)
    if not (target / "preoutcome_seal.json").exists():
        raise RuntimeError(f"{panel}: the main forecasts are not sealed yet")
    for name in ("outcomes.csv.gz", "labels.csv", "outcome_run.json"):
        if (target / name).exists():
            raise FileExistsError(f"{panel}: {name} exists; TabPFN must be sealed before the label step")
    if (target / "tabpfn_seal.json").exists():
        raise FileExistsError(f"{panel}: TabPFN already sealed")
    from scripts.analysis.compare_difficulty import FEATURES_B2

    query = pd.read_csv(target / "query_preoutcome.csv.gz")[KEYS + list(FEATURES_B2)].assign(source_panel=panel)
    law, meta = T.forecast(T.training_table(), query, T.WORK / f"panel6_{panel}")
    out = law[KEYS].copy()
    out["method"] = "tabpfn_v2"
    out[P] = law[P].to_numpy(float)
    out.to_csv(target / "tabpfn_predictions_preoutcome.csv", index=False)
    record = {"panel": panel, "protocol_sha256": E.sha256(S6.OUTPUT / "protocol.json"),
              "preoutcome_seal_sha256": E.sha256(target / "preoutcome_seal.json"), "learner_outcomes_observed": 0,
              "sealed_at_utc": E.now(), "spec": {"path": E.rel(T.SPEC), "sha256": E.sha256(T.SPEC)}, "tabpfn": meta,
              "attribution": "Built with PriorLabs-TabPFN (Hollmann et al., 2025); weights under the Prior Labs License 1.1",
              "artifact_sha256": {"tabpfn_predictions_preoutcome.csv": E.sha256(target / "tabpfn_predictions_preoutcome.csv")},
              "source_sha256": {E.rel(p): E.sha256(p) for p in (Path(__file__), T.ROOT / "scripts/analysis/baseline_tabpfn_t3.py", T.RUNNER)}}
    E.write_json(target / "tabpfn_seal.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", choices=tuple(S6.PANELS), required=True)
    record = seal(parser.parse_args().panel)
    print(f"sealed TabPFN v2 for {record['panel']}: {record['artifact_sha256']}")


if __name__ == "__main__":
    main()

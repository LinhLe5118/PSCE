"""TabPFN v2 baseline on B2 (task card T3; post hoc on opened panels; spec written before the run).

Spec: ``docs/week8/T3_TABPFN_SPEC_2026-10-15.md``. The classifier runs in the isolated venv through
``scripts/t3/tabpfn_forecast.py``; this script exports its inputs, scores its output with the T1 scorer and writes the
manifest. Built with PriorLabs-TabPFN (Hollmann et al., 2025); the weights are not stored in the repository.

Usage: ``python -m scripts.analysis.baseline_tabpfn_t3``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from scripts.analysis import extrapolation_endpoint as X

ROOT = X.ROOT
OUTPUT = ROOT / "artifacts/week8/baseline_tabpfn_t3_2026-10-15"
SPEC = ROOT / "docs/week8/T3_TABPFN_SPEC_2026-10-15.md"
RUNNER = ROOT / "scripts/t3/tabpfn_forecast.py"
VENV_PYTHON = Path(os.environ.get("TABPFN_PYTHON", "D:/Tools/venvs/tabpfn/Scripts/python.exe"))
CACHE = Path(os.environ.get("TABPFN_MODEL_CACHE_DIR", "D:/Tools/tabpfn_cache"))
LICENSE = Path(os.environ.get("TABPFN_LICENSE", "D:/Tools/tabpfn_license/LICENSE.txt"))
WORK = Path(os.environ.get("TABPFN_WORK", "D:/Tools/tabpfn_work/t3"))
KEYS, P = X.KEYS, X.P
COMPARE = ("b2", "robust", "oeh_f", "oeh")


def training_table() -> pd.DataFrame:
    from scripts.analysis.compare_difficulty import FEATURES_B2
    from scripts.analysis.develop_cohort8_candidates_20261014 import meta_panel

    train = meta_panel(sort=False)
    return train[train.panel == "primary"][["algorithm", "event", *FEATURES_B2]].reset_index(drop=True)


def query_table(panel: str) -> pd.DataFrame:
    from scripts.analysis import baseline_prefix_fqe as B
    from scripts.analysis.compare_difficulty import FEATURES_B2

    rows = X.load(panel)
    rows = rows[rows.canonical == "b2"][KEYS]
    q = rows.merge(B._query_table(panel)[KEYS + list(FEATURES_B2)], on=KEYS, validate="one_to_one")
    if len(q) != len(rows):
        raise RuntimeError(f"{panel}: B2 features missing")
    return q.assign(source_panel=panel)


def forecast(train: pd.DataFrame, query: pd.DataFrame, work: Path) -> tuple[pd.DataFrame, dict]:
    work.mkdir(parents=True, exist_ok=True)
    tpath, qpath, out = work / "train.csv.gz", work / "query.csv.gz", work / "probs.csv.gz"
    train.to_csv(tpath, index=False)
    query.drop(columns=["source_panel"]).to_csv(qpath, index=False)
    env = {**os.environ, "PYTHONUTF8": "1", "TABPFN_MODEL_CACHE_DIR": str(CACHE), "TABPFN_DISABLE_TELEMETRY": "1",
           "TABPFN_ALLOW_CPU_LARGE_DATASET": "1"}
    subprocess.run([str(VENV_PYTHON), "-I", str(RUNNER), "--train", str(tpath), "--query", str(qpath), "--out", str(out)],
                   check=True, cwd=str(work), env=env, stdout=subprocess.DEVNULL)
    probs = pd.read_csv(out)
    meta = json.loads(Path(str(out) + ".meta.json").read_text(encoding="utf-8"))
    merged = query[KEYS + ["source_panel"]].merge(probs, on=KEYS, validate="one_to_one")
    p = merged[P].to_numpy(float)
    if len(merged) != len(query) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-9):
        raise RuntimeError("invalid TabPFN forecast")
    return merged, meta


def main() -> None:
    from scripts.analysis import baseline_prefix_fqe as B

    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = training_table()
    query = pd.concat([query_table(p) for p in X.PANELS], ignore_index=True)
    law, meta = forecast(train, query, WORK)
    law.to_csv(OUTPUT / "predictions.csv.gz", index=False)
    rows = []
    for panel in X.PANELS:
        lab = pd.read_csv(X.PANELS[panel]["labels"])
        if "panel" in lab:
            lab = lab[lab.panel == "primary"]
        m = law[law.source_panel == panel].merge(lab[KEYS + ["event"]], on=KEYS, validate="one_to_one")
        for k, v in X.row_scores(m[P].to_numpy(float), m.event.to_numpy(int)).items():
            m[k] = v
        rows.append(m.assign(canonical="tabpfn_v2"))
    ours = pd.concat(rows, ignore_index=True)
    others = pd.concat([X.load(p) for p in X.PANELS], ignore_index=True)
    prefix = pd.read_csv(B.OUTPUT / "scores_rows.csv.gz")
    prefix = prefix[prefix.canonical == "b2_plus_prefix_fqe"]
    allrows = pd.concat([ours, others, prefix], ignore_index=True)
    metrics = ["ibs", "ll", "ibs_ge", "ibs_gt", "ll_ext"]
    pm = allrows.groupby(["source_panel", "canonical", "mdp_id"])[metrics].mean().reset_index()
    pm.groupby(["source_panel", "canonical"])[["ibs", "ll", "ibs_ge", "ll_ext"]].mean().reset_index().to_csv(OUTPUT / "scores_by_panel.csv", index=False)
    gains = []
    for scope, panels in [(p, (p,)) for p in X.PANELS] + list(X.GROUPS.items()):
        for cand in COMPARE + ("b2_plus_prefix_fqe",):
            g = X.gains(pm, panels, "tabpfn_v2", cand)
            if g:
                gains.append({"scope": scope, "candidate": cand, "baseline": "tabpfn_v2", **g})
    pd.DataFrame(gains).to_csv(OUTPUT / "gains.csv", index=False)
    if LICENSE.exists():
        shutil.copyfile(LICENSE, OUTPUT / "LICENSE_TabPFN_weights.txt")
    files = [OUTPUT / n for n in ("predictions.csv.gz", "scores_by_panel.csv", "gains.csv", "LICENSE_TabPFN_weights.txt") if (OUTPUT / n).exists()]
    manifest = {"status": "posthoc_descriptive", "task": "T3 TabPFN v2 baseline on B2", "created_utc": datetime.now(timezone.utc).isoformat(),
                "attribution": "Built with PriorLabs-TabPFN (Hollmann et al., 2025, Nature); weights under the Prior Labs License 1.1",
                "spec": {"path": SPEC.relative_to(ROOT).as_posix(), "sha256": X.sha(SPEC)}, "tabpfn": meta,
                "outputs_sha256": {p.name: X.sha(p) for p in files},
                "source_sha256": {p.relative_to(ROOT).as_posix(): X.sha(p) for p in (Path(__file__), RUNNER)}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    show = pd.read_csv(OUTPUT / "scores_by_panel.csv")
    print(show[show.canonical.isin(["tabpfn_v2", "b2", "b2_plus_prefix_fqe", "oeh_f", "oeh"])].pivot(index="source_panel", columns="canonical", values="ll").round(3).to_string())


if __name__ == "__main__":
    main()

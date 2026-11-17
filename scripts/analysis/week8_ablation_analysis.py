"""Paired MDP analysis of the opened-cohort matched ablation (exploratory)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/week8/matched_ablation_exploratory_2026-10-07"
OUTPUT = ROOT / "artifacts/week8/matched_ablation_analysis_exploratory_2026-10-07"
P = [f"p_{i}" for i in range(7)]
BASELINES = ("b2_matched", "oeh_full_108")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete_exploratory_opened_cohort":
        raise RuntimeError("Ablation run incomplete")
    for name, expected in manifest["output_sha256"].items():
        if sha256(SOURCE / name) != expected:
            raise RuntimeError(f"Ablation artifact changed: {name}")
    frame = pd.read_csv(SOURCE / "scored.csv")
    methods = set(frame.method)
    if len(methods) != 7 or not set(BASELINES).issubset(methods):
        raise RuntimeError("Missing ablation arms")
    key = ["panel", "algorithm", "mdp_id", "policy_family"]
    if frame.duplicated([*key, "method"]).any() or frame["event"].isna().any():
        raise RuntimeError("Duplicate or missing score rows")
    per_mdp = frame.groupby(["panel", "algorithm", "method", "mdp_id"], observed=True)[
        ["brier", "log_loss"]].mean().reset_index()
    rng = np.random.default_rng(20261007)
    rows = []
    for (panel, learner), group in per_mdp.groupby(["panel", "algorithm"], observed=True):
        for metric in ("brier", "log_loss"):
            pivot = group.pivot(index="mdp_id", columns="method", values=metric)
            if pivot.isna().any().any() or len(pivot) not in (36, 84):
                raise RuntimeError("Unpaired MDP rows in ablation")
            for baseline in BASELINES:
                for method in sorted(methods - {baseline}):
                    delta = (pivot[method] - pivot[baseline]).to_numpy(float)
                    draws = rng.integers(0, len(delta), size=(2000, len(delta)))
                    boot = delta[draws].mean(axis=1)
                    rows.append({"panel": panel, "algorithm": learner, "metric": metric,
                                 "comparison": f"{method} minus {baseline}", "mdps": len(delta),
                                 "mean_delta": delta.mean(), "ci95_low": np.quantile(boot, .025),
                                 "ci95_high": np.quantile(boot, .975),
                                 "better_mdps": int((delta < 0).sum()),
                                 "worse_mdps": int((delta > 0).sum()),
                                 "tied_mdps": int((delta == 0).sum())})
    p = frame[P].to_numpy(float)
    y = frame.event.to_numpy(int)
    cdf = np.cumsum(p, axis=1)[:, :6]
    calibration = []
    for budget in range(6):
        temp = frame[["panel", "algorithm", "method"]].copy()
        temp["budget"] = (1, 2, 5, 10, 20, 50)[budget]
        temp["forecast_cdf"] = cdf[:, budget]
        temp["observed_reach"] = (y <= budget).astype(float)
        calibration.append(temp.groupby(["panel", "algorithm", "method", "budget"], observed=True)[
            ["forecast_cdf", "observed_reach"]].mean().reset_index())
    OUTPUT.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "paired_mdp_delta.csv", index=False)
    pd.concat(calibration, ignore_index=True).to_csv(OUTPUT / "calibration_by_budget.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "exploratory_only_opened_cohort", "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
        "script_sha256": sha256(Path(__file__)), "bootstrap": "2000 MDP-resampled means, percentile 95%",
        "delta_definition": "method minus baseline; negative is better for both losses",
        "output_sha256": {name: sha256(OUTPUT / name) for name in
                          ("paired_mdp_delta.csv", "calibration_by_budget.csv")},
    }, indent=2) + "\n", encoding="utf-8")
    primary = pd.DataFrame(rows)
    print(primary[(primary.panel == "primary") & (primary.metric == "brier")][
        ["algorithm", "comparison", "mean_delta", "ci95_low", "ci95_high"]].to_string(index=False))


if __name__ == "__main__":
    main()

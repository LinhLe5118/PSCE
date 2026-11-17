"""Leave-one-opened-cohort-out development of a pilot-only OEH/B2 early hybrid.

All four query cohorts are already opened. This is model selection, not an
independent result. E1 is always meta-training; the query cohort's labels are
excluded from its fit. B2 and oracle EH use the same tree capacity and inputs
as the cross-capacity OEH model, with their own frozen three-part weights.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import (
    E1_FEATURES, E1_LABELS, E1_ORACLE, EH, FIELDS, KEYS, ROOT, sha256,
)
from scripts.analysis.develop_cross_learner_context import CONTEXT, with_context
from scripts.analysis.develop_eh_oracle_three_way import aft_probabilities, numeric
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.develop_oeh_difficulty import fit_thresholds, score_probabilities
from scripts.analysis.develop_tree_capacity import multiclass
from scripts.analysis.run_oeh_120_cross_capacity import (
    DEV, FIRST_OUTPUT, SECOND_OUTPUT, OUTPUT as THIRD,
)


OUTPUT = ROOT / "artifacts/oeh_early_hybrid_four_opened_development_2026-10-01"
ALPHAS = (.25, .5, .75, 1.)
METHODS_TO_FIT = ("b2_three_way", "b2_plus_eh_oracle_three_way", "oeh_cross_capacity")


def pair(labels: Path, features: Path, oracle: Path) -> pd.DataFrame:
    result = pd.read_csv(labels).merge(pd.read_csv(features), on=KEYS, validate="one_to_one")
    result = result.merge(pd.read_csv(oracle)[["mdp_id", *EH]], on="mdp_id", validate="many_to_one")
    if result[KEYS].duplicated().any() or result.mdp_id.nunique() * 15 != len(result):
        raise RuntimeError(f"Invalid opened cohort: {labels}")
    return result


def load() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[Path]]:
    e1 = pair(E1_LABELS, E1_FEATURES, E1_ORACLE)
    opened = {f"dev_{i + 1}": pair(*paths) for i, paths in enumerate(DEV)}
    for name, folder in (("first_120", FIRST_OUTPUT), ("second_120", SECOND_OUTPUT)):
        opened[name] = pair(folder / "labels.csv", folder / "pilot_features.csv",
                            folder / "oracle_descriptors_preoutcome.csv")
    all_frames = pd.concat((e1, *opened.values()), ignore_index=True)
    if (all_frames.mdp_id.nunique() != 480 or len(all_frames) != 7200
            or all_frames[KEYS].duplicated().any()):
        raise RuntimeError("E1/opened cohort overlap or missing data")
    all_frames = with_context(all_frames)
    e1_rows = all_frames[all_frames.mdp_id.isin(e1.mdp_id)]
    cohort_rows = {name: all_frames[all_frames.mdp_id.isin(frame.mdp_id)]
                   for name, frame in opened.items()}
    sources = [E1_LABELS, E1_FEATURES, E1_ORACLE,
               *(path for triple in DEV for path in triple),
               *(folder / name for folder in (FIRST_OUTPUT, SECOND_OUTPUT)
                 for name in ("labels.csv", "pilot_features.csv", "oracle_descriptors_preoutcome.csv"))]
    return e1_rows, cohort_rows, sources


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Four-cohort development artifact exists")
    protocol_path = THIRD / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    e1, opened, sources = load()
    fields = {"b2_three_way": FEATURES_B2,
              "b2_plus_eh_oracle_three_way": FIELDS,
              "oeh_cross_capacity": (*OEH_FEATURES, *CONTEXT)}
    all_rows = []
    for cohort, query in opened.items():
        source = pd.concat((e1, *(frame for name, frame in opened.items()
                                  if name != cohort)), ignore_index=True)
        if set(source.mdp_id) & set(query.mdp_id):
            raise RuntimeError("Leave-cohort-out training leak")
        for learner in METHODS:
            train = source[(source.panel == "primary") & (source.algorithm == learner)]
            target = query[(query.panel == "primary") & (query.algorithm == learner)]
            y = train.event.to_numpy(dtype=int)
            observed = target.event.to_numpy(dtype=int)
            predicted = {}
            for method in METHODS_TO_FIT:
                names = fields[method]
                x = numeric(train) if method == "b2_plus_eh_oracle_three_way" else np.nan_to_num(
                    train[list(names)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
                q = numeric(target) if method == "b2_plus_eh_oracle_three_way" else np.nan_to_num(
                    target[list(names)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
                parts = np.asarray((aft_probabilities(x, y, q),
                                    fit_thresholds(x, y, q),
                                    multiclass(x, y, q, protocol["multiclass_capacity"])))
                predicted[method] = np.einsum("i,ijk->jk", protocol["weights"][method], parts)
            oeh, b2 = predicted["oeh_cross_capacity"], predicted["b2_three_way"]
            for alpha in ALPHAS:
                hybrid = oeh.copy()
                hybrid[:, :2] *= np.power(np.clip(b2[:, :2], 1e-12, 1.) /
                                               np.clip(oeh[:, :2], 1e-12, 1.), alpha)
                hybrid /= hybrid.sum(axis=1, keepdims=True)
                predicted[f"early_b2_alpha_{alpha:g}"] = hybrid
            for method, p in predicted.items():
                brier, logloss = score_probabilities(p, observed)
                frame = target[KEYS + ["panel", "event"]].copy()
                frame["cohort"] = cohort
                frame["method"] = method
                frame["integrated_brier"] = brier
                frame["seven_class_log_loss"] = logloss
                all_rows.append(frame)
        print(f"Early-hybrid development: {cohort}", flush=True)
    scored = pd.concat(all_rows, ignore_index=True)
    mdp = scored.groupby(["mdp_id", "cohort", "method"], as_index=False)[
        ["integrated_brier", "seven_class_log_loss"]].mean()
    summary = mdp.groupby(["cohort", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("integrated_brier", "mean"),
        seven_class_log_loss=("seven_class_log_loss", "mean"))
    pooled = mdp.groupby("method", as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("integrated_brier", "mean"),
        seven_class_log_loss=("seven_class_log_loss", "mean"))
    if not (pooled.mdps == 252).all():
        raise RuntimeError("Expected 252 opened stochastic development MDPs")
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored.csv", index=False)
    mdp.to_csv(OUTPUT / "mdp_scores.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    pooled.to_csv(OUTPUT / "pooled.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "adaptive_four_opened_cohorts_not_independent",
        "query_mdps": {name: int(frame.mdp_id.nunique()) for name, frame in opened.items()},
        "pilot_only_early_hybrid_alphas": ALPHAS,
        "third_protocol_sha256": sha256(protocol_path),
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                         for path in sources},
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {name: sha256(OUTPUT / name) for name in
                            ("scored.csv", "mdp_scores.csv", "summary.csv", "pooled.csv")},
    }, indent=2) + "\n", encoding="utf-8")
    print(pooled.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

"""Time one unchanged matched B2 forecast on existing meta-training rows.

This is a compute probe only. It neither scores labels nor writes predictions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis import oeh_final_forecast as forecast
from scripts.analysis import run_oeh_final_120 as runner


QUERY = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"


def main() -> None:
    started = time.perf_counter()
    train = runner.meta_training(runner.OPENED, "A2")
    loaded = time.perf_counter()
    labels = pd.read_csv(QUERY / "labels.csv", usecols=["mdp_id", "panel"]).drop_duplicates()
    query = pd.read_csv(QUERY / "pilot_features.csv").merge(
        labels, on="mdp_id", validate="many_to_one"
    )
    if train.mdp_id.isin(query.mdp_id).any() or len(query) != 1800:
        raise ValueError("Unexpected training/query overlap or row count")
    prepared = time.perf_counter()
    forecast.SELECT_CAPACITY = False
    probability, _ = forecast.matched_forecast(train, query, forecast.FEATURES_B2)
    finished = time.perf_counter()
    print(json.dumps({
        "scope": "runtime_probe_only_existing_opened_data_not_a_new_result",
        "meta_training_rows": len(train),
        "query_rows": len(query),
        "arm": "b2_matched",
        "load_seconds": round(loaded - started, 3),
        "query_prepare_seconds": round(prepared - loaded, 3),
        "fit_predict_seconds": round(finished - prepared, 3),
        "total_seconds": round(finished - started, 3),
        "prediction_shape": list(probability.shape),
    }, indent=2))


if __name__ == "__main__":
    main()

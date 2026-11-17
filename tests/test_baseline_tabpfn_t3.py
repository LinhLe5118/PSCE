"""TabPFN v2 baseline (task card T3): class expansion and the recorded run."""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd

from scripts.analysis import baseline_tabpfn_t3 as T

_spec = importlib.util.spec_from_file_location("tabpfn_forecast", T.RUNNER)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def test_unseen_classes_get_zero_probability():
    p = R.expand(np.array([[0.2, 0.8], [0.5, 0.5]]), np.array([1, 6]))
    assert p.shape == (2, 7) and np.allclose(p.sum(1), 1)
    assert np.allclose(p[:, [0, 2, 3, 4, 5]], 0) and np.allclose(p[0, [1, 6]], [0.2, 0.8])


def test_recorded_run_matches_its_manifest_and_spec():
    manifest = json.loads((T.OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "posthoc_descriptive" and manifest["tabpfn"]["model_version"] == "v2"
    assert manifest["tabpfn"]["tabpfn"] == "9.2.0" and "PriorLabs-TabPFN" in manifest["attribution"]
    assert manifest["spec"]["sha256"] == T.X.sha(T.SPEC)
    for name, digest in manifest["outputs_sha256"].items():
        assert T.X.sha(T.OUTPUT / name) == digest
    law = pd.read_csv(T.OUTPUT / "predictions.csv.gz")
    p = law[T.P].to_numpy(float)
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-9) and set(law.source_panel) == set(T.X.PANELS)

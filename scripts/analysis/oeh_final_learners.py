"""Per-dataset adapters of the four registered external learners.

The adapters call exactly the implementations and fixed settings used by the
original E1 external panel (``evaluation/e1_external.py`` in the archived
generator checkout): author Pi-b SPIBB with ``N_wedge = 10``, the VI-LCB port
with ``cb = delta = 0.05``, the authors' discrete BCQ and pinned d3rlpy
discrete CQL with 500 updates. They are used both on simulated posterior
datasets and, through the original callback, on real buffers, so every learner
in the library follows one rule.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from scripts.analysis.run_unseen_validation import archive_imports

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = ROOT / "tmp"
EXTERNAL = ("pib_spibb_n10_author", "vi_lcb_cb005_author_empirical",
            "discrete_bcq_author", "discrete_cql_d3rlpy")
NEURAL_STEPS = 500


def external_learners(methods=EXTERNAL) -> dict[str, Callable[[dict, dict], np.ndarray]]:
    """Return ``{name: learner(arrays, context) -> policy}`` for ``methods``."""
    archive_imports()
    from adaptive_pilot import transitions_from_arrays
    from scripts.external import run_neural_discrete_development as neural
    from scripts.external import run_spibb_development as spibb_adapter
    from vi_lcb import vi_lcb_discounted

    learners: dict[str, Callable[[dict, dict], np.ndarray]] = {}
    if "pib_spibb_n10_author" in methods:
        spibb, model_transitions, _ = spibb_adapter._load_author_modules(EXTERNAL_ROOT / "external_spibb")

        def run_spibb(arrays: dict, context: dict) -> np.ndarray:
            S, A = context["num_states"], context["num_actions"]
            batch = [[int(a), int(s), int(n), float(r)] for a, s, n, r in
                     zip(arrays["action"], arrays["state"], arrays["next_state"], arrays["reward"])]
            with np.errstate(divide="ignore", invalid="ignore"):
                model = model_transitions.ModelTransitions(batch, S, A)
            mask = spibb.compute_mask_N_wedge(S, A, 10, batch)
            learner = spibb.spibb(context["gamma"], S, A, np.asarray(context["behavior"], dtype=float),
                                  mask, model.transitions,
                                  spibb_adapter._reward_model(batch, S, A), "Pi_b_SPIBB")
            learner.fit()
            return np.asarray(learner.pi, dtype=float)

        learners["pib_spibb_n10_author"] = run_spibb
    if "vi_lcb_cb005_author_empirical" in methods:
        def run_vi_lcb(arrays: dict, context: dict) -> np.ndarray:
            full = _natural_fields(arrays)
            return np.asarray(vi_lcb_discounted(
                transitions_from_arrays(full), num_states=context["num_states"],
                num_actions=context["num_actions"], gamma=context["gamma"], cb=.05, delta=.05,
                iterations=100, terminal_states=tuple(context["terminal_states"])).policy, dtype=float)

        learners["vi_lcb_cb005_author_empirical"] = run_vi_lcb
    if {"discrete_bcq_author", "discrete_cql_d3rlpy"} & set(methods):
        import torch

        neural.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("Registered BCQ/CQL require CUDA, as in the E1 panel")
        device = torch.device("cuda:0")
        torch.cuda.get_device_name(0)
        torch.zeros(1, device=device)  # initialize the CUDA context before memory statistics
        torch.set_float32_matmul_precision("high")
        if "discrete_bcq_author" in methods:
            def run_bcq(arrays: dict, context: dict) -> np.ndarray:
                policy, _, _ = neural._fit_bcq(_natural_fields(arrays), context["num_states"],
                                               context["num_actions"], context["gamma"], NEURAL_STEPS,
                                               int(context["seed"]), device, EXTERNAL_ROOT / "external_bcq")
                return policy

            learners["discrete_bcq_author"] = run_bcq
        if "discrete_cql_d3rlpy" in methods:
            def run_cql(arrays: dict, context: dict) -> np.ndarray:
                policy, _, _ = neural._fit_cql(_natural_fields(arrays), context["num_states"],
                                               context["num_actions"], context["gamma"], NEURAL_STEPS,
                                               int(context["seed"]), device)
                return policy

            learners["discrete_cql_d3rlpy"] = run_cql
    return {name: learners[name] for name in methods}


def _natural_fields(arrays: dict) -> dict:
    """Add the trajectory bookkeeping fields expected by archived adapters."""
    terminal = np.asarray(arrays["terminal"], dtype=bool)
    return {"state": np.asarray(arrays["state"], dtype=np.int64),
            "action": np.asarray(arrays["action"], dtype=np.int64),
            "reward": np.asarray(arrays["reward"], dtype=float),
            "next_state": np.asarray(arrays["next_state"], dtype=np.int64),
            "terminal": terminal, "done": terminal,
            "truncated": np.asarray(arrays["truncated"], dtype=bool),
            "trajectory_id": np.asarray(arrays["trajectory_id"], dtype=np.int64),
            "timestep": np.asarray(arrays["timestep"], dtype=np.int64)}

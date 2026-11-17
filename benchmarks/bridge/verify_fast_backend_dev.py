"""Development-only parity and timing checks for the fresh-run kernels."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from oeh import posterior, tabular  # noqa: E402
from .fast_backend import (finite_horizon_policy_returns_cuda,
                           finite_horizon_references_cuda,
                           simulate_streams_searchsorted)

OUT = ROOT / "artifacts/week8/bridge_fast_backend_parity_dev_2026-10-08.json"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    rng = np.random.default_rng(20261008)
    B, S, A = 2, 413, 4
    p = rng.random((B, S, A, S))
    p /= p.sum(-1, keepdims=True)
    rewards = rng.random((B, S, A, S))
    beta = rng.random((B, S, A))
    beta /= beta.sum(-1, keepdims=True)
    terminal = np.zeros(S, dtype=bool)
    terminal[-1] = True
    rho = np.zeros(S)
    rho[0] = 1
    sample = posterior.MDPSample(p, rewards, beta, terminal, rho, .99)
    state1 = np.random.default_rng(314159)
    state2 = np.random.default_rng(314159)
    start = time.perf_counter()
    original = posterior.simulate_streams(sample, state1, streams=3, length=1500, horizon=40)
    cpu_seconds = time.perf_counter() - start
    start = time.perf_counter()
    faster = simulate_streams_searchsorted(sample, state2, streams=3, length=1500, horizon=40)
    fast_seconds = time.perf_counter() - start
    mismatches = {name: int(np.count_nonzero(original[name] != faster[name])) for name in original}
    if any(mismatches.values()) or state1.bit_generator.state != state2.bit_generator.state:
        raise AssertionError(f"Simulated streams differ: {mismatches}")
    r = np.einsum("bsan,bsan->bsa", p, rewards)
    policies = rng.random((B, 6, S, A))
    policies /= policies.sum(-1, keepdims=True)
    random_cpu, optimal_cpu = tabular.finite_horizon_references(p, r, np.broadcast_to(rho, (B, S)), .99, 40)
    return_cpu = tabular.finite_horizon_policy_returns(p, r, np.broadcast_to(rho, (B, S)), policies, .99, 40)
    random_fast, optimal_fast = finite_horizon_references_cuda(p, r, np.broadcast_to(rho, (B, S)), .99, 40)
    return_fast = finite_horizon_policy_returns_cuda(p, r, np.broadcast_to(rho, (B, S)), policies, .99, 40)
    errors = {"random": float(np.max(np.abs(random_cpu - random_fast))),
              "optimal": float(np.max(np.abs(optimal_cpu - optimal_fast))),
              "policy": float(np.max(np.abs(return_cpu - return_fast)))}
    if any(value > 1e-9 for value in errors.values()):
        raise AssertionError(f"CUDA scoring differs: {errors}")
    result = {"status": "PASS_DEVELOPMENT_PARITY_NO_TEST_LABELS", "states": S,
              "draws": B, "streams": 3, "length": 1500, "horizon": 40,
              "stream_mismatches": mismatches, "rng_end_state_identical": True,
              "original_simulation_seconds": cpu_seconds,
              "fast_simulation_seconds": fast_seconds,
              "max_abs_return_errors": errors}
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

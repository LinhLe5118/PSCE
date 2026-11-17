"""Measure finite-horizon policy evaluation on this GPU without changing OEH.

Uses deterministic synthetic dense transition matrices; reports timing and
agreement with the repository's float64 CPU implementation. Development-only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from oeh.tabular import finite_horizon_policy_returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=176)
    parser.add_argument("--policies", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    rng = np.random.default_rng(20261008)
    s, k = args.states, args.policies
    p = rng.random((1, s, 6, s), dtype=np.float64)
    p /= p.sum(axis=-1, keepdims=True)
    r = rng.random((1, s, 6), dtype=np.float64)
    rho = rng.random((1, s), dtype=np.float64)
    rho /= rho.sum()
    selected = rng.integers(0, 6, size=(1, k, s))
    policies = np.eye(6, dtype=np.float64)[selected]
    start = time.perf_counter()
    cpu = finite_horizon_policy_returns(p, r, rho, policies, .99, args.horizon)
    cpu_seconds = time.perf_counter() - start
    result = {"states": s, "policies": k, "horizon": args.horizon,
              "cpu_float64_seconds": cpu_seconds, "torch": torch.__version__,
              "cuda_device": torch.cuda.get_device_name(0)}
    for dtype in (torch.float32, torch.float64):
        torch.cuda.synchronize()
        start = time.perf_counter()
        p_t = torch.as_tensor(p, dtype=dtype, device="cuda")
        r_t = torch.as_tensor(r, dtype=dtype, device="cuda")
        rho_t = torch.as_tensor(rho, dtype=dtype, device="cuda")
        pi_t = torch.as_tensor(policies, dtype=dtype, device="cuda")
        p_pi = torch.einsum("mksa,msan->mksn", pi_t, p_t)
        r_pi = torch.einsum("mksa,msa->mks", pi_t, r_t)
        value = torch.zeros((1, k, s), dtype=dtype, device="cuda")
        for _ in range(args.horizon):
            value = r_pi + .99 * torch.matmul(p_pi, value.unsqueeze(-1)).squeeze(-1)
        gpu = torch.einsum("ms,mks->mk", rho_t, value).cpu().numpy()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        result[str(dtype)] = {"seconds_including_transfer": seconds,
                              "max_abs_error_vs_cpu_float64": float(np.max(np.abs(gpu - cpu))),
                              "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated())}
        torch.cuda.reset_peak_memory_stats()
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

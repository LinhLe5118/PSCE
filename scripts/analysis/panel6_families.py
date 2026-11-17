"""Panel-6 candidate families (task card T8): draft for checkpoint H1; no panel data is generated here.

Two candidate sources not used by any earlier panel, each with its own seeded instance generator:

* ``mdpp``: MDP Playground 1.0.0 (Rajan et al., 2021), ``RLToyEnv`` discrete MDPs: maximally connected successor
  graph on ``diameter`` independent sets of ``A`` states, uniform transition noise eta, rewardable states
  (sequence length 1, no delay), absorbing terminal states;
* ``labmaze``: DeepMind Lab's text-maze generator (``labmaze.RandomMaze`` 1.0.6). Rooms and corridors come from the
  generator; moves are four-way, succeed with probability ``success``, otherwise go in one of the other three
  directions; walls block; reward 1 on entering a goal cell (``G``), start uniform over spawn cells (``P``).

Both libraries run in an isolated venv (``scripts/panel6/export_kernels.py``) and hand over ``npz`` kernels; the
project environment keeps Gymnasium 1.0.0 for the sealed FrozenLake panel. As in panel 5, episodic kernels are made
continuing: a terminal state restarts from the initial distribution and keeps the reward earned on entry.

Distinctness is checked two ways:
* the project's exact fingerprint (``run_oeh_120_unseen.fingerprint``) against ``run_oeh_final_120.seen_kernels``;
* a relabelling-invariant signature against every earlier external kernel kept on this machine.

``python -m scripts.analysis.panel6_families feasibility`` draws development candidates with numpy seed 1 (never the
panel seeds), exports them, applies the size cap and the project's dynamic-range filter, and writes pass rates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/week8/panel6_feasibility_2026-10-15"
EXPORTER = ROOT / "scripts/panel6/export_kernels.py"
VENV_PYTHON = Path(os.environ.get("PANEL6_PYTHON", "D:/Tools/venvs/mdpp/Scripts/python.exe"))
WORK = Path(os.environ.get("PANEL6_WORK", "D:/Tools/panel6_dev"))
SA_CAP = 256
DEV_SEED = 1
FAMILIES = {
    "mdpp": {
        "source": "MDP Playground 1.0.0 (Rajan et al., 2021), RLToyEnv discrete: sequence_length 1, delay 0, "
                  "maximally_connected, reward_noise 0, reward_scale 1",
        "parameters": "A in {2,3,4,5,6,8}; diameter d uniform on the integers with 8 <= d*A <= 64 and d*A*A <= 256; "
                      "transition_noise ~ U(0, 0.3); reward_density ~ U(0.1, 0.5); terminal_state_density in {0, 0.25}",
    },
    "labmaze": {
        "source": "labmaze 1.0.6 RandomMaze (DeepMind Lab text-maze generator), simplify=False, one spawn and one "
                  "object per room; four-way moves",
        "parameters": "height, width in {9, 11}; max_rooms in {1, 2, 3}; room_min_size 3; room_max_size in {3, 4}; "
                      "extra_connection_probability ~ U(0, 0.2); success ~ U(0.7, 0.95)",
    },
}
ACTIONS = (2, 3, 4, 5, 6, 8)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Candidate parameters (seeded); kernels come from the source libraries
# --------------------------------------------------------------------------- #

def diameters(actions: int) -> list[int]:
    return [d for d in range(1, 65) if 8 <= d * actions <= 64 and d * actions * actions <= SA_CAP]


def draw(family: str, rng: np.random.Generator) -> dict:
    if family == "mdpp":
        a = int(rng.choice(ACTIONS))
        return {"actions": a, "diameter": int(rng.choice(diameters(a))),
                "transition_noise": round(float(rng.uniform(0.0, 0.3)), 3),
                "reward_density": round(float(rng.uniform(0.1, 0.5)), 3),
                "terminal_state_density": float(rng.choice([0.0, 0.25])), "source_seed": int(rng.integers(1, 2**31 - 1))}
    return {"height": int(rng.choice([9, 11])), "width": int(rng.choice([9, 11])), "max_rooms": int(rng.choice([1, 2, 3])),
            "room_min_size": 3, "room_max_size": int(rng.choice([3, 4])),
            "extra_connection_probability": round(float(rng.uniform(0.0, 0.2)), 3),
            "success": round(float(rng.uniform(0.7, 0.95)), 3), "source_seed": int(rng.integers(1, 2**31 - 1))}


def candidates(family: str, count: int, seed: int, prefix: str) -> list[dict]:
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < count:
        item = draw(family, rng)
        key = tuple(item.values())
        if key not in seen:
            seen.add(key)
            out.append({"mdp_id": f"{prefix}{family}_{len(out):03d}", "family": family, **item})
    return out


def export(items: list[dict], out: Path, verify: int = 0) -> dict:
    """Run the exporter in the isolated venv; returns its ``export_meta.json``."""
    out.mkdir(parents=True, exist_ok=True)
    spec = out / "candidates.json"
    spec.write_text(json.dumps(items, indent=1), encoding="utf-8")
    env = {**os.environ, "PYTHONUTF8": "1"}
    subprocess.run([str(VENV_PYTHON), "-I", str(EXPORTER), "--candidates", str(spec), "--out", str(out), "--verify", str(verify)],
                   check=True, cwd=str(out), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads((out / "export_meta.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Conversion and fingerprints
# --------------------------------------------------------------------------- #

def continuing(T: np.ndarray, R: np.ndarray, rho: np.ndarray, terminal) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Episodic -> continuing, as in panel 5: a terminal state restarts from the initial distribution."""
    T, R = T.copy(), R.copy()
    for s in terminal:
        T[s] = rho[None, :]
        R[s] = 0.0
    return T, R, rho


def builders():
    """The exporter's numpy-only kernel builders (the exporter itself runs in the isolated venv)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("panel6_export_kernels", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rebuild(family: str, source: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuing kernel from the recorded source tables, without the source libraries."""
    K = builders()
    if family == "mdpp":
        T = K.successor_kernel(np.asarray(source["succ"], int), source["eta"])
        rho = np.asarray(source["rho"], float)
        return continuing(T, T @ np.asarray(source["next_reward"], float), rho / rho.sum(), source["terminal"])
    T, R, rho, terminal = K.grid_kernel(np.array([list(row) for row in source["layout"]]), source["success"])
    return continuing(T, R, rho, terminal)


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return continuing(z["T"], z["R"], z["rho"], z["terminal"])


def fingerprint(T: np.ndarray, R: np.ndarray, rho: np.ndarray) -> str:
    """The project's exact kernel fingerprint (``run_oeh_120_unseen.fingerprint`` of the TabularMDP built from the kernel)."""
    S = R.shape[0]
    digest = hashlib.sha256()
    for array in ((T / T.sum(-1, keepdims=True)).astype(np.float64), np.repeat(R[:, :, None], S, axis=2).astype(np.float64),
                  (rho / rho.sum()).astype(np.float64)):
        digest.update(array.tobytes())
    digest.update(json.dumps([]).encode("ascii"))
    return digest.hexdigest()


def signature(T: np.ndarray, R: np.ndarray, rho: np.ndarray, decimals: int = 9) -> str:
    """Invariant to relabelling states and, per state, actions: equal kernels up to relabelling share it.

    Each state is summarized by the sorted multiset of its actions' (sorted next-state probabilities, reward) and by its
    initial probability; the signature hashes (S, A) and the sorted multiset of state summaries.
    Different signatures prove that two kernels are not relabellings of each other."""
    S, A = R.shape
    states = []
    for s in range(S):
        acts = sorted((tuple(np.round(np.sort(T[s, a])[::-1], decimals)), round(float(R[s, a]), decimals)) for a in range(A))
        states.append((round(float(rho[s]), decimals), tuple(acts)))
    return hashlib.sha256(repr((S, A, sorted(states))).encode()).hexdigest()


def earlier_kernel_files() -> dict[str, list[Path]]:
    """Kernels of earlier external panels kept on this machine (git-ignored)."""
    A = Path(os.environ.get("PANEL6_EARLIER_ROOT", ROOT)) / "artifacts"  # kernels are not tracked; a worktree points at the main checkout
    colosseum = Path(os.environ.get("COLOSSEUM_KERNELS", "D:/Tools/colosseum_work/trial/data"))
    dirs = {"frozenlake": A / "oeh_fifth_panel_2026-10-15/frozenlake/kernels", "gambler": A / "oeh_fifth_panel_2026-10-15/gambler/kernels",
            "spibb": A / "oeh_spibb_panel_2026-10-14/kernels", "spibb2": A / "oeh_v2_sealed_test_2026-10-14/spibb2/kernels",
            "colosseum2029": A / "oeh_v2_sealed_test_2026-10-14/colosseum2029/kernels", "forest": A / "oeh_v2_sealed_test_2026-10-14/forest/kernels",
            **{f"colosseum_{d}": colosseum / d for d in ("colosseum", "confirm", "c8tier", "v2test")}}
    return {name: sorted(d.glob("*.npz")) for name, d in dirs.items() if d.exists()}


def earlier_signatures() -> dict[str, str]:
    out = {}
    for name, files in earlier_kernel_files().items():
        for f in files:
            with np.load(f) as z:
                if "P" in z.files:  # SPIBB format: P (S, A, S), R (S, A, S), terminal
                    T, R, rho = continuing(z["P"], (z["P"] * z["R"]).sum(-1), z["rho"], z["terminal"])
                else:
                    T, R, rho = z["T"], z["R"], z["rho"]
            out[signature(T / T.sum(-1, keepdims=True), R, rho / rho.sum())] = f"{name}/{f.stem}"
    return out


# --------------------------------------------------------------------------- #
# Feasibility on development draws
# --------------------------------------------------------------------------- #

def dynamic_range(kernels: Path, mdp_id: str) -> tuple[float, bool]:
    """The project's dynamic-range value and filter decision (``exact_solver.passes_dynamic_range_filter``)."""
    from scripts.analysis import run_oeh_eighth_cohort_20261014 as E
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    from environments.benchmark_environments import DYNAMIC_RANGE_THRESHOLD
    from environments.tabular_environments import TabularMDP
    import exact_solver
    mdp = E.colosseum_mdp(str(kernels), mdp_id, TabularMDP)
    value = float(exact_solver.dynamic_range(mdp)[0])
    return value, bool(exact_solver.passes_dynamic_range_filter(mdp, threshold=DYNAMIC_RANGE_THRESHOLD))


def feasibility(count: int, verify: int) -> pd.DataFrame:
    from scripts.analysis.run_oeh_final_120 import seen_kernels

    seen = set(seen_kernels())
    earlier = earlier_signatures()
    rows, meta_all = [], {}
    for family in FAMILIES:
        raw = WORK / f"dev_seed{DEV_SEED}" / family
        items = candidates(family, count, DEV_SEED, "dev_")
        meta = export(items, raw / "episodic", verify)
        meta_all[family] = meta
        conv = raw / "continuing"
        conv.mkdir(parents=True, exist_ok=True)
        for c in items:
            T, R, rho = load(raw / "episodic" / f"{c['mdp_id']}.npz")
            S, A = R.shape
            np.savez(conv / f"{c['mdp_id']}.npz", T=T, R=R, rho=rho)
            within = S * A <= SA_CAP
            sig = signature(T, R, rho)
            value, passes = dynamic_range(conv, c["mdp_id"])
            rows.append({**c, "states": S, "n_actions": A, "sa": S * A, "within_cap": within,
                         "dynamic_range": value, "passes_dynamic_range": within and passes,
                         "fingerprint": fingerprint(T, R, rho), "signature": sig,
                         "seen_exact": fingerprint(T, R, rho) in seen, "matches_earlier_panel": earlier.get(sig, ""),
                         **({"verify_bonferroni_p": meta["kernels"][c["mdp_id"]]["verify"]["bonferroni_p"],
                             "verify_max_z": meta["kernels"][c["mdp_id"]]["verify"]["max_z"],
                             "verify_support_ok": meta["kernels"][c["mdp_id"]]["verify"]["support_ok"],
                             "verify_reward_error": meta["kernels"][c["mdp_id"]]["verify"]["max_abs_reward_error"]}
                            if "verify" in meta["kernels"][c["mdp_id"]] else {})})
    table = pd.DataFrame(rows)
    table["duplicate_signature"] = table.duplicated("signature", keep="first")
    table["eligible"] = table.passes_dynamic_range & ~table.seen_exact & (table.matches_earlier_panel == "") & ~table.duplicate_signature
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "feasibility.csv", index=False)
    tables = {mid: entry["source"] for meta in meta_all.values() for mid, entry in meta["kernels"].items()}
    (OUTPUT / "source_tables.json").write_text(json.dumps(tables, separators=(",", ":")), encoding="utf-8")
    summary = table.groupby("family").agg(candidates=("mdp_id", "size"), within_cap=("within_cap", "sum"),
                                          passes_dynamic_range=("passes_dynamic_range", "sum"), eligible=("eligible", "sum"),
                                          distinct_signatures=("signature", "nunique"), median_sa=("sa", "median"),
                                          min_dynamic_range=("dynamic_range", "min"), max_dynamic_range=("dynamic_range", "max"),
                                          min_states=("states", "min"), max_states=("states", "max")).reset_index()
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    versions = {f: {k: v for k, v in m.items() if k != "kernels"} for f, m in meta_all.items()}
    manifest = {"status": "development_draws_only", "task": "T8 panel-6 feasibility (numpy seed 1; never the panel seeds)",
                "created_utc": datetime.now(timezone.utc).isoformat(), "families": FAMILIES, "sa_cap": SA_CAP, "dev_seed": DEV_SEED,
                "versions": versions, "earlier_kernels_compared": {k: len(v) for k, v in earlier_kernel_files().items()},
                "seen_exact_fingerprints": len(seen), "learner_outcomes_observed": 0,
                "outputs_sha256": {p.name: sha(p) for p in (OUTPUT / "feasibility.csv", OUTPUT / "summary.csv", OUTPUT / "source_tables.json")},
                "source_sha256": {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__)), EXPORTER.relative_to(ROOT).as_posix(): sha(EXPORTER)}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("feasibility",))
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--verify", type=int, default=200)
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    print(feasibility(args.count, args.verify).to_string(index=False))


if __name__ == "__main__":
    main()

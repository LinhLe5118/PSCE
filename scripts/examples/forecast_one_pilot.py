"""Forecast the data requirement of BC, CFQI and IQL from ONE pilot with the final OEH method.

Builds a one-group bundle (the pilot streams of one MDP-behavior pair of a dataset in the project's format;
training streams are not copied), runs the single entry point ``scripts.analysis.oeh_pipeline`` on it and prints
the seven-class law. Fitting the heads on the 720-MDP meta panel takes a few minutes.

    python -m scripts.examples.forecast_one_pilot \
        [--dataset datasets/oeh_eighth_cohort_2026-10-14] [--mdp <instance_id>] [--behavior q_star_softmax] \
        [--kernels <dir> --kind colosseum] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def one_group_bundle(dataset: Path, mdp: str | None, behavior: str, target: Path) -> tuple[str, Path]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    index = json.loads((dataset / "data_only_index.json").read_text(encoding="utf-8"))
    mdp = mdp or manifest["instances"][0]["instance_id"]
    rows = [r for r in manifest["datasets"] if r["instance_id"] == mdp and r["policy_family"] == behavior and r.get("stream") == "pilot"]
    if not rows:
        raise SystemExit(f"no pilot of {mdp} / {behavior} in {dataset}")
    keep = {r["data_only_dataset_id"] for r in rows}
    target.mkdir(parents=True, exist_ok=True)
    small = {**manifest, "instances": [i for i in manifest["instances"] if i["instance_id"] == mdp], "datasets": rows,
             "instance_count": 1, "deviation_reason": f"one-group example bundle of {dataset.name}"}
    small_index = {**index, "datasets": [d for d in index["datasets"] if d["dataset_id"] in keep]}
    for item in rows + small_index["datasets"]:
        if "file" in item and not (target / item["file"]).exists():
            (target / item["file"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dataset / item["file"], target / item["file"])
    (target / "manifest.json").write_text(json.dumps(small), encoding="utf-8")
    (target / "data_only_index.json").write_text(json.dumps(small_index), encoding="utf-8")
    return mdp, target


def main() -> None:
    from scripts.analysis import oeh_pipeline as O

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/oeh_eighth_cohort_2026-10-14")
    parser.add_argument("--mdp")
    parser.add_argument("--behavior", default="q_star_softmax")
    parser.add_argument("--kernels", type=Path)
    parser.add_argument("--kind", choices=("internal", "colosseum", "spibb"), default="internal")
    parser.add_argument("--out", type=Path, default=Path("D:/Tools/oeh_example_one_pilot"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    mdp, bundle = one_group_bundle(args.dataset, args.mdp, args.behavior, args.out / "bundle")
    query = O.pilot_inputs(bundle, args.out / "inputs", args.workers, args.kernels, args.kind, "oeh-example-one-pilot")
    law = O.forecast(query, args.workers, args.out / "work")
    law.to_csv(args.out / "oeh_forecast.csv", index=False)
    print(f"OEH forecast for {mdp} / {args.behavior} (P(E = class), classes {', '.join(O.CLASSES)}):")
    for row in law.itertuples(index=False):
        probs = " ".join(f"{getattr(row, p):.3f}" for p in O.P)
        print(f"  {row.algorithm:<26s} {probs}   shift distance d = {row.distance:.2f}")


if __name__ == "__main__":
    main()

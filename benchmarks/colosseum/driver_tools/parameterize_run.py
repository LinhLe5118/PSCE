"""One-off edit: make the driver scripts write each external run to its own paths."""
import pathlib
import shutil

D = pathlib.Path(__file__).resolve().parents[1] / "driver"
backup = D.parent / "driver_backup_run1"
if not backup.exists():
    shutil.copytree(D, backup)


def edit(name, pairs):
    p = D / name
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        if a not in s:
            raise SystemExit(f"{name}: pattern not found: {a[:70]!r}")
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")


edit("common.py", [('COHORTS = {', '''# External run selector: "colosseum" = exploratory run (results/), "confirm" =
# confirmatory run on fresh MDPs (results/confirm/). Paths derive from it.
RUN = os.environ.get("COLOSSEUM_RUN", "colosseum")
RUN_SRC = DATA / RUN
RUN_BUNDLE_NAME = f"{RUN}_bundle"
RUN_BUNDLE = DATA / RUN_BUNDLE_NAME
RES = OUT if RUN == "colosseum" else OUT / RUN

COHORTS = {''')])

edit("colosseum_export.py", [
    ('import json\nimport os', 'import hashlib\nimport json\nimport os'),
    ('SAMPLER_SEED, DRAWS, MAX_PER_FAMILY = 2026, 80, 9\nOUT = TRIAL / "data" / "colosseum"',
     '''RUN = os.environ.get("COLOSSEUM_RUN", "colosseum")
SAMPLER_SEED = int(os.environ.get("SAMPLER_SEED", 2026))
DRAWS = int(os.environ.get("DRAWS", 80))
MAX_PER_FAMILY = int(os.environ.get("MAX_PER_FAMILY", 9))
PREFIX = "colosseum" if RUN == "colosseum" else f"{RUN}_colosseum"
OUT = TRIAL / "data" / RUN


def kernel_hash(T, R):
    return hashlib.sha256(np.asarray(T, np.float64).tobytes() + np.asarray(R, np.float64).tobytes()).hexdigest()


# Kernels used by any earlier run must not reappear (fresh MDPs only).
SEEN = set()
if RUN != "colosseum":
    for path in (TRIAL / "data" / "colosseum").glob("*.npz"):
        with np.load(path) as z:
            SEEN.add(kernel_hash(z["T"], z["R"]))
EXCLUDED_EARLIER = len(SEEN)'''),
    ('        name = f"colosseum_{family}_{draw:03d}"',
     '''        kernel = kernel_hash(T, R)
        if kernel in SEEN:
            index.append({"family": family, "draw": draw, "status": "duplicate_kernel"})
            continue
        SEEN.add(kernel)
        name = f"{PREFIX}_{family}_{draw:03d}"'''),
    ('"status": "candidate", "mdp_id": name, "S": S, "A": A,',
     '"status": "candidate", "mdp_id": name, "S": S, "A": A, "kernel_sha256": kernel,'),
    ('{"sampler_seed": SAMPLER_SEED, "draws_per_family": DRAWS, "max_per_family": MAX_PER_FAMILY,',
     '{"run": RUN, "sampler_seed": SAMPLER_SEED, "draws_per_family": DRAWS, "max_per_family": MAX_PER_FAMILY,\n     "earlier_kernels_excluded": EXCLUDED_EARLIER,'),
])

edit("build_colosseum.py", [
    ('from common import DATA, load_json, sha256, use_workspace, write_json',
     'import os\n\nfrom common import DATA, RUN, RUN_BUNDLE, RUN_SRC, load_json, sha256, use_workspace, write_json'),
    ('SOURCE = DATA / "colosseum"\nTARGET = DATA / "colosseum_bundle"',
     '''SOURCE = RUN_SRC
TARGET = RUN_BUNDLE
MAX_SELECT = int(os.environ.get("MAX_SELECT", 10**6))  # first N passing per family, sampler order
SEED_BASE = int(os.environ.get("SEED_BASE", 9_100_000))'''),
    ('''    selection.append({**{k: record[k] for k in ("mdp_id", "family", "draw", "S", "A")},
                      "dynamic_range": rng, "kept": bool(keep)})
    if not keep:
        continue''', '''    room = counts.get(record["family"], 0) < MAX_SELECT
    selection.append({**{k: record[k] for k in ("mdp_id", "family", "draw", "S", "A")},
                      "dynamic_range": rng, "passes_filter": bool(keep), "kept": bool(keep and room)})
    if not (keep and room):
        continue'''),
    ('seed=9_100_000 + 1_000 * len(specs)', 'seed=SEED_BASE + 1_000 * len(specs)'),
    ('write_json(DATA / "colosseum_selection.json", {', 'write_json(DATA / f"{RUN}_selection.json", {'),
    ('''    "rule": "all exported candidates passing the project's dynamic-range filter",''',
     '''    "rule": "exported candidates passing the project's dynamic-range filter; first MAX_SELECT per family in sampler order",
    "max_select": MAX_SELECT,'''),
])

edit("colosseum_mdp.py", [('from common import DATA\n\nSOURCE = DATA / "colosseum"', 'from common import RUN_SRC\n\nSOURCE = RUN_SRC')])

edit("run_oehb.py", [('"colosseum_bundle": "external-colosseum-trial-20261013",',
                      '"colosseum_bundle": "external-colosseum-trial-20261013",\n              "confirm_bundle": "external-colosseum-confirm-20261014",')])

edit("features_colosseum.py", [
    ('from common import COHORTS, DATA, OUT, load_json, use_workspace\n\nDATASET = DATA / "colosseum_bundle"',
     'from common import COHORTS, RES, RUN_BUNDLE, load_json, use_workspace\n\nDATASET = RUN_BUNDLE'),
    ('    OUT.mkdir(parents=True, exist_ok=True)', '    RES.mkdir(parents=True, exist_ok=True)'),
    ('to_csv(OUT / "colosseum_pilot_features.csv", index=False)', 'to_csv(RES / "colosseum_pilot_features.csv", index=False)'),
    ('to_csv(OUT / "colosseum_horizon.csv", index=False)', 'to_csv(RES / "colosseum_horizon.csv", index=False)'),
])

edit("forecast.py", [
    ('from common import COHORTS, OUT, load_json, sha256, use_workspace, write_json',
     'from common import COHORTS, DATA, OUT, RES, RUN, RUN_BUNDLE_NAME, load_json, sha256, use_workspace, write_json'),
    ('    h = pd.read_csv(OUT / f"hardness_{name}.csv")',
     '    h = pd.read_csv((RES if name == "colosseum" else OUT) / f"hardness_{name}.csv")'),
    ('    q = pd.read_csv(OUT / "colosseum_pilot_features.csv")\n    q = q.merge(F.oeh_feature_rows(OUT / "oehb" / "colosseum_bundle", OUT / "colosseum_horizon.csv"),',
     '    q = pd.read_csv(RES / "colosseum_pilot_features.csv")\n    q = q.merge(F.oeh_feature_rows(OUT / "oehb" / RUN_BUNDLE_NAME, RES / "colosseum_horizon.csv"),'),
    ('        target = OUT / "colosseum_predictions_preoutcome.csv"', '        target = RES / "colosseum_predictions_preoutcome.csv"'),
    ('        if len(query) != 32 * 15 or query[R.KEYS].duplicated().any():',
     '        expected = load_json(DATA / f"{RUN}_selection.json")["selected"] * 15\n        if len(query) != expected or query[R.KEYS].duplicated().any():'),
    ('        write_json(OUT / "colosseum_preoutcome_seal.json", {', '        write_json(RES / "colosseum_preoutcome_seal.json", {'),
    ('"protocol_sha256": sha256(OUT / "protocol_colosseum.json")})', '"protocol_sha256": sha256(RES / "protocol_colosseum.json")})'),
])

edit("labels_colosseum.py", [
    ('from common import DATA, OUT, load_json, sha256, use_workspace, write_json\n\nDATASET = DATA / "colosseum_bundle"',
     'from common import RES as OUT, RUN_BUNDLE, load_json, sha256, use_workspace, write_json\n\nDATASET = RUN_BUNDLE'),
])

edit("score_colosseum.py", [('from common import OUT, load_json, sha256, write_json',
                             'from common import RES as OUT, load_json, sha256, write_json')])
print("all scripts parameterized; backup in driver_backup_run1/")

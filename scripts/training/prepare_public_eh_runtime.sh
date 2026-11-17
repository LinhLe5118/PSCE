#!/usr/bin/env bash
# Isolated WSL setup / author CLI. Run after other timing jobs have finished.
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
repo="$task_dir/tmp/public_effective_horizon"
runtime_project="$task_dir/tmp/julia_eh_project"
# Native Linux artifacts require Unix modes/symlinks for Julia tree hashes.
# NTFS-backed /mnt/d can change those and cause verified artifacts to fail.
export JULIA_DEPOT_PATH=/tmp/daorl_eh_julia_depot
export JULIA_PKG_PRECOMPILE_AUTO=0
export PYTHON=/usr/bin/python3
julia_bin="$task_dir/tmp/julia-1.8.5/bin/julia"

if [[ "$1" == setup ]]; then
    python3 - "$task_dir" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1]) / 'tmp'
name = 'julia-1.8.5-linux-x86_64.tar.gz'
expected = next(line.split()[0] for line in (root/'julia-1.8.5.sha256').read_text().splitlines() if line.split()[-1] == name)
with (root/name).open('rb') as f:
    actual = hashlib.file_digest(f, 'sha256').hexdigest()
if actual != expected:
    raise ValueError('Julia archive checksum mismatch/incomplete download')
print('Official Julia checksum verified:', actual, flush=True)
PY
    if [[ ! -x "$julia_bin" ]]; then
        tar -xzf "$task_dir/tmp/julia-1.8.5-linux-x86_64.tar.gz" -C "$task_dir/tmp"
    fi
    mkdir -p "$runtime_project" "$repo/src/binom/build"
    mkdir -p "$JULIA_DEPOT_PATH"
    for cache_name in packages registries clones; do
        if [[ ! -e "$JULIA_DEPOT_PATH/$cache_name" && -d "$task_dir/tmp/julia_eh_depot/$cache_name" ]]; then
            cp -a "$task_dir/tmp/julia_eh_depot/$cache_name" "$JULIA_DEPOT_PATH/$cache_name"
        fi
    done
    cp -n "$repo/EffectiveHorizon.jl/Project.toml" "$runtime_project/Project.toml"
    cp -n "$repo/EffectiveHorizon.jl/Manifest.toml" "$runtime_project/Manifest.toml"
    if [[ ! -e "$runtime_project/src" ]]; then
        ln -s "$repo/EffectiveHorizon.jl/src" "$runtime_project/src"
    fi
    cp -n "$repo/EffectiveHorizon.jl/src/libbinom.so" "$repo/src/binom/build/libbinom.so"
    "$julia_bin" --project="$runtime_project" -e 'using Pkg; Pkg.instantiate()'
    "$julia_bin" --project="$runtime_project" -e 'using EffectiveHorizon; println("EffectiveHorizon import OK")'
else
    # These budgets match k_results in the three author-provided reference JSONs.
    name="$1"
    horizon="$2"
    max_k="$3"
    data="$repo/data/mdps/$name/consolidated.npz"
    # Resolve Git symlinks represented by text files in the Windows checkout.
    data="$(python3 - "$data" "$repo" <<'PY'
import pathlib, sys
p, root = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve()
for _ in range(8):
    if not p.is_relative_to(root):
        raise ValueError('Data link leaves author clone')
    with p.open('rb') as f:
        if f.read(2) == b'PK':
            print(p)
            break
    p = (p.parent / p.read_text().strip()).resolve()
else:
    raise ValueError('Data link cycle')
PY
)"
    output="$task_dir/artifacts/week5/eh_julia_runtime_verified_2026-09-14/$name.json"
    if [[ -e "$output" ]]; then
        echo "Refusing to overwrite $output" >&2
        exit 1
    fi
    cd "$repo"
    "$julia_bin" --threads=2 --project="$runtime_project" \
        EffectiveHorizon.jl/src/compute_gorp_bounds.jl --mdp "$data" \
        --use_value_dists --horizon "$horizon" --max_k "$max_k" --out "$output"
fi

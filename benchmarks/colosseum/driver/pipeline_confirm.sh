#!/usr/bin/env bash
# Confirmatory Colosseum run on fresh MDPs; protocol frozen beforehand.
set -u
cd "$(dirname "$0")/.."
export TMP="E:/research/oeh_colosseum_trial/.tmp" TEMP="E:/research/oeh_colosseum_trial/.tmp"
export COLOSSEUM_RUN=confirm SAMPLER_SEED=2027 DRAWS=600 MAX_PER_FAMILY=30 MAX_SELECT=18 SEED_BASE=9500000
LOG=logs/confirm_pipeline.log
step() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
run() { local label=$1; shift; local t0=$(date +%s); step "START $label"
  if (cd driver && "$@") >> "logs/confirm_$label.log" 2>&1; then step "DONE  $label ($(( $(date +%s) - t0 ))s)"
  else step "FAIL  $label (see logs/confirm_$label.log)"; exit 1; fi; }
test -f results/confirm/protocol_colosseum.json || { step "FAIL protocol not frozen"; exit 1; }
run export     ../py39/python.exe colosseum_export.py
run hardness   ../py39/python.exe hardness_arrays.py ../data/confirm ../results/confirm/hardness_colosseum.csv
run bundle     ../py312/python.exe build_colosseum.py
run oehb       ../py312/python.exe run_oehb.py confirm_bundle 14
run features   ../py312/python.exe features_colosseum.py 14
run seal       ../py312/python.exe forecast.py colosseum
run labels     ../py312/python.exe labels_colosseum.py 14
run score      ../py312/python.exe score_colosseum.py
step "CONFIRM PIPELINE COMPLETE"

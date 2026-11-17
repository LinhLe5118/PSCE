#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export TMP="E:/research/oeh_colosseum_trial/.tmp" TEMP="E:/research/oeh_colosseum_trial/.tmp"
LOG=logs/pipeline2.log
step() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
run() { local label=$1; shift; local t0=$(date +%s); step "START $label"
  if (cd driver && "$@") >> "logs/$label.log" 2>&1; then step "DONE  $label ($(( $(date +%s) - t0 ))s)"
  else step "FAIL  $label (see logs/$label.log)"; exit 1; fi; }
run features_colosseum ../py312/python.exe features_colosseum.py 14
run seal_colosseum ../py312/python.exe forecast.py colosseum
run labels_colosseum ../py312/python.exe labels_colosseum.py 14
run score_colosseum ../py312/python.exe score_colosseum.py
step "PIPELINE COMPLETE"

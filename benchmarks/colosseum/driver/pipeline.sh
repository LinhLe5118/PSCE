#!/usr/bin/env bash
# Remaining steps of the external Colosseum trial, in order, with timings.
set -u
cd "$(dirname "$0")/.."
export TMP="E:/research/oeh_colosseum_trial/.tmp" TEMP="E:/research/oeh_colosseum_trial/.tmp"
PY=./py312/python.exe
LOG=logs/pipeline.log
step() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
run() {  # run <label> <command...>; stop the pipeline on failure
  local label=$1; shift
  local t0=$(date +%s)
  step "START $label"
  if "$@" >> "logs/$label.log" 2>&1; then
    step "DONE  $label ($(( $(date +%s) - t0 ))s)"
  else
    step "FAIL  $label (see logs/$label.log)"; exit 1
  fi
}

step "waiting for OEH-B chain (c7, c4, colosseum)"
until grep -q "colosseum_bundle exit" logs/oehb_status.log 2>/dev/null; do sleep 30; done
grep -q "colosseum_bundle exit 0" logs/oehb_status.log || { step "FAIL OEH-B chain"; exit 1; }

for c in first60 second60 first120 second120 third120; do
  run "oehb_$c" $PY driver/run_oehb.py "$c" 14
done
for c in first60 second60 first120 second120 third120; do
  run "horizon_$c" $PY driver/run_horizon.py "$c" 14
done
run features_colosseum $PY driver/features_colosseum.py 14

step "waiting for hardness measures of all cohorts"
until [ "$(ls results/hardness_{first60,second60,first120,second120,third120,c4,c7}.csv 2>/dev/null | wc -l)" -eq 7 ] \
      && grep -q diameter_capped results/hardness_second60.csv; do sleep 30; done

run reproduce_c7 $PY driver/forecast.py reproduce
run seal_colosseum $PY driver/forecast.py colosseum
run labels_colosseum $PY driver/labels_colosseum.py 14
run score_colosseum $PY driver/score_colosseum.py
step "PIPELINE COMPLETE"

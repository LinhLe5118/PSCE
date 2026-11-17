# Artifact cleanup -- 20 September 2026

The final active evidence set consists of the three OEH-v3 post-freeze
directories listed in `FINAL_DEVELOPMENT_RESULTS_2026-09-20.md`, the two
LC-OEH-v4 exploratory directories, the historical same-machine all-method
comparison, and the historical OEH saturation, bottleneck, and positive-control
evidence.

## Removed

- 47 obsolete entries (about 123.4 MiB) were removed after the OEH-v2 and
  all-method replacement runs completed. These were failed launches, incomplete
  grids, duplicate summaries, superseded redesign smokes, and temporary
  comparison outputs.
- Completed runs' `progress.json` files are removed because final reports,
  summaries, result tables, and manifests contain the terminal state and
  provenance. This includes the OEH-v3 run and retained historical/final
  comparison runs; their substantive evidence remains unchanged.
- Temporary build and regression directories are not retained.
- Seven superseded research directories were removed after their completed
  `final_*` replacements were verified: the old SPIBB/VI-LCB/BCQ/CQL runs, two
  old external comparison bundles, and the pre-final B0--B3 bundle. Script
  defaults and historical documentation now point to the retained final runs.

## Retained as current evidence

- `artifacts/research/oeh_v3_post_freeze_holdout_2026-09-20/`
- `artifacts/research/oeh_v3_post_freeze_local_methods_2026-09-20/`
- `artifacts/research/oeh_v3_post_freeze_b0_b3_2026-09-20/`
- `artifacts/research/lcoeh_v4_exploratory_2026-09-20/`
- `artifacts/research/lcoeh_v4_exploratory_final_2026-09-20/`
- `artifacts/research/final_all_methods_comparison_2026-09-20/`
- its local, SPIBB, VI-LCB, BCQ, and CQL source result directories.

## Retained as historical evidence

- `artifacts/week2/saturation_audit_2026-08-25/`
- `artifacts/week2/saturation_sweep_2026-08-24/`
- `artifacts/week2/positive_control_2026-09-01/`
- `artifacts/research/oeh_saturation_completion_v2_2026-09-19/`
- `artifacts/research/oeh_redesign_2026-09-18/`
- the final OEH-v2 audit and controlled-factorial artifacts that document the
  transition from the original saturation defect to OEH-v3.

Historical evidence must not be used as the source for current OEH-v3 result
tables. Datasets, source files, and the sealed evaluation split are outside the
artifact cleanup scope.

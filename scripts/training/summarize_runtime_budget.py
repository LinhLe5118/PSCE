"""Collect measured/projection costs without treating missing stages as zero."""
import argparse
import json
from pathlib import Path
import time

from evaluation.io import write_json


def read(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        # Live producers may still be writing; retry at the next watch tick.
        return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,default=Path('artifacts/week5'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--watch',action='store_true',help='Refresh every 30 seconds until the runtime queue finishes (maximum 4 hours).')
    args=parser.parse_args()
    deadline=time.monotonic()+14400
    while True:
        summarize(args)
        queue_paths=[args.artifact_root/'runtime_queue_2026-09-14/status.json']
        eh_status=args.artifact_root/'eh_julia_runtime_verified_2026-09-14/status.json'
        if eh_status.exists():
            queue_paths.append(eh_status)
        eh_retry=args.artifact_root/'eh_julia_linux_depot_retry_2026-09-14/status.json'
        if eh_retry.exists():
            queue_paths.append(eh_retry)
        finished=all((read(path) or {}).get('status') in {'finished','wait_timeout'} for path in queue_paths)
        if not args.watch or finished or time.monotonic()>deadline:
            return
        time.sleep(30)


def summarize(args):
    root=args.artifact_root
    stages=[]
    generation=read(root/'expanded_development_v2_2026-09-13/execution_context.json')
    if generation:
        stages.append(dict(name='expanded_dataset_generation',kind='previous_measured_run',
            seconds=generation['generation_seconds'],scope='Same expanded profile, local CPU; historical generation run, not timed again here'))
    training=read(root/'t2_runtime_expanded_v2_2026-09-13_complete/runtime_summary.json')
    if training:
        stages.append(dict(name='expanded_tabular_training_and_scoring',kind='projection',
            seconds=training['full_grid_estimate_seconds'],scope='48 MDP, 240 groups, 43200 fits; 3 measured seed slices projected to 10'))
    oeh=read(root/'pipeline_runtime_2026-09-13/summary.json')
    if oeh:
        stages.append(dict(name='expanded_raw_adaptive_oeh_and_variants',kind='stratified_projection',
            seconds=sum(oeh['weighted_estimate_seconds'].values()),scope=oeh['note']))
    public=read(root/'public_components_runtime_2026-09-13/summary.json')
    if public:
        stages.append(dict(name='author_return_hll_components',kind='measured_sum_of_medians',
            seconds=public['pilot_component_median_sum_seconds'],scope='240 expanded pilots; excludes normalization/reference data and BC entropy'))
        for row in public['rows']:
            if row['method']=='author_value_iteration':
                stages.append(dict(name='author_vi_'+row['dataset'],kind=row['status'],
                    seconds=row.get('median_seconds'),scope='Public test MDP; NOT full Effective Horizon/GORP/SQIRL'))
    queue=root/'runtime_queue_2026-09-14'
    for name in ['cql_strict','iql_strict','iql_compatibility','cql_compatibility_retry','iql_compatibility_retry']:
        source_dir=(root/'neural_retry_2026-09-14'/name.removesuffix('_retry')) if name.endswith('_retry') else queue/name
        summary=read(source_dir/'summary.json')
        failure=read(source_dir/'failure.json')
        if summary:
            phases=summary['phases']
            if name.startswith('iql'):
                seconds=summary['load_seconds']+summary['setup_seconds']+phases[0]['cold_update_seconds']+1_000_000*phases[0]['seconds_per_update']
            else:
                by_phase={r['phase']:r for r in phases}
                seconds=summary['load_seconds']+summary['setup_seconds']+sum(r['cold_update_seconds'] for r in phases)+40_000*by_phase['early_bc']['seconds_per_update']+2_960_000*by_phase['post_bc']['seconds_per_update']
            stages.append(dict(name=name,kind='update_component_projection',seconds=seconds,
                scope='ONE Hopper-medium-v2 seed, CPU. Excludes evaluation/checkpoints. Not transferable to other datasets or GPUs.',
                compatibility_warning='CQL compatibility retry includes copy-on-write tensor overhead; not the author archived runtime speed.' if name=='cql_compatibility_retry' else 'Current JAX/Flax with recorded API shim; not the archived paper environment.' if name=='iql_compatibility_retry' else None,
                details=summary))
        else:
            stages.append(dict(name=name,kind='failed' if failure else 'pending',seconds=None,details=failure,
                measured_partial_phases=read(source_dir/'partial.json')))
    stages.extend(dict(name=r['name'],kind='historical_proxy' if r['exit_code']==0 else 'failed',
        seconds=r['wall_seconds'] if r['exit_code']==0 else None,scope=r['scope'])
        for r in (read(queue/'measurements.json') or []) if r['name'].startswith('historical_'))
    for eh_name in ['eh_julia_runtime_verified_2026-09-14','eh_julia_linux_depot_retry_2026-09-14']:
      eh_queue=root/eh_name
      for row in read(eh_queue/'measurements.json') or []:
        stages.append(dict(name=eh_name+'_'+row['name'],
            kind=('one_time_setup' if 'setup' in row['name'] else 'measured_author_cli') if row['exit_code']==0 else ('setup_timeout' if row['exit_code']==124 and 'setup' in row['name'] else ('runtime_censored' if row['exit_code']==124 else 'failed_or_skipped')),
            seconds=row['wall_seconds'] if row['exit_code']==0 else None,
            elapsed_before_timeout_seconds=row['wall_seconds'] if row['exit_code']==124 else None,
            scope=row['scope'],platform='WSL Linux on same hardware; not the identical Windows software configuration'))
    core_names={'expanded_dataset_generation','expanded_tabular_training_and_scoring','expanded_raw_adaptive_oeh_and_variants'}
    core=[r for r in stages if r['name'] in core_names]
    write_json(args.output,dict(status='partial_budget',full_pipeline_seconds=None,stages=stages,
        local_core_subtotal_seconds=sum(r['seconds'] for r in core) if len(core)==3 else None,
        local_core_scope='Generation + projected raw OEH/variants + projected learner/scoring. Excludes downstream/export/public work.',
        missing=['Full-grid artifact export/schema validation',
                 'Expanded 48-MDP downstream statistics with its actual full-grid labels',
                 'Exact oracle k-curves and additional registered sensitivity/ablation grids',
                 'Effective Horizon Julia bound / online SQIRL-GORP',
                 'Neural simulator evaluation and checkpoint overhead',
                 'Neural cross-dataset/seed grid and GPU measurements',
                 'Sealed evaluation (not authorized by this runtime probe)'],
        warning='Do not sum differently scoped groups into a claimed complete pipeline time. Setup/download wall time is separate from training.'))


if __name__=='__main__':
    main()

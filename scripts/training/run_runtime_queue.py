"""Run a saved benchmark queue sequentially and persist each exit/time result."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from evaluation.io import write_json, sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    output=Path(plan['output_dir'])
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    write_json(output/'status.json',dict(status='waiting',plan_sha256=sha256_file(args.plan)))
    wait_file=plan.get('wait_for_file')
    wait_status=plan.get('wait_for_finished_queue')
    def ready():
        if wait_file and not Path(wait_file).exists():
            return False
        if wait_status:
            try:
                return json.loads(Path(wait_status).read_text(encoding='utf-8')).get('status')=='finished'
            except (FileNotFoundError,json.JSONDecodeError):
                return False
        return True
    deadline=time.monotonic()+plan.get('wait_timeout_seconds',14400)
    while not ready():
        if time.monotonic()>deadline:
            write_json(output/'status.json',dict(status='wait_timeout',waiting_for=wait_status or wait_file))
            raise TimeoutError(wait_status or wait_file)
        time.sleep(10)
    for stage in plan['stages']:
        write_json(output/'status.json',dict(status='running',current_stage=stage['name'],completed=rows))
        prior={r['name']:r['exit_code'] for r in rows}
        if any(prior.get(name)!=0 for name in stage.get('depends_on',[])):
            rows.append(dict(name=stage['name'],exit_code=125,wall_seconds=0,
                scope=stage['scope'],status='skipped_failed_dependency'))
            write_json(output/'measurements.json',rows)
            continue
        env=os.environ.copy()
        env['PYTHONUTF8']='1'
        # Historical analysis uses its original default profile.
        env.pop('DAORL_PROTOCOL_CONFIG',None)
        env.update(stage.get('environment',{}))
        started=time.perf_counter()
        with (output/(stage['name']+'.log')).open('w',encoding='utf-8') as log:
            try:
                process=subprocess.run(stage['command'],env=env,stdout=log,stderr=subprocess.STDOUT,
                    timeout=stage.get('timeout_seconds'))
                exit_code=process.returncode
            except subprocess.TimeoutExpired:
                exit_code=124
        row=dict(name=stage['name'],exit_code=exit_code,
            wall_seconds=time.perf_counter()-started,scope=stage['scope'])
        rows.append(row)
        write_json(output/'measurements.json',rows)
    write_json(output/'status.json',dict(status='finished',all_stages_passed=all(r['exit_code']==0 for r in rows),completed=rows))


if __name__=='__main__':
    main()

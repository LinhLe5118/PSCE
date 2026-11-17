"""Run the same fixed experiment with equivalent vectorized water filling.

The original scientific sources stay immutable. Both the launch adapter and
kernel are added to plan/report provenance BEFORE estimation starts.
"""
from scripts.analysis import run_oeh_selective as runner
import oeh_certificate
import oeh_selective
from oeh_fast_expectation import expectation


def main():
    original_write=runner.write_json
    extras=['code/oeh_fast_expectation.py','tests/test_oeh_fast_expectation.py',
            'scripts/analysis/run_oeh_selective_fast.py']
    hashes={p:runner.sha256_file(runner.ROOT/p) for p in extras}
    def with_provenance(path,data):
        if path.name in ('plan.json','report.json'):
            data={**data,'source_sha256':{**data['source_sha256'],**hashes},
                  'expectation_kernel':'Equivalent vectorized box-simplex water filling; no tolerance or statistical changes'}
        original_write(path,data)
    runner.write_json=with_provenance
    oeh_certificate._expectation=expectation
    oeh_selective._expectation=expectation
    return runner.main()


if __name__ == '__main__':
    raise SystemExit(main())

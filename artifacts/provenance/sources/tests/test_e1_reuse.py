"""Consumed E1 reuse must preserve input identities, full grids and metrics."""
from copy import deepcopy
import pytest
from evaluation.e1_reuse import ALGORITHMS, GRID, verify_run_grid


def tiny_panel():
    records=[]; rows=[]
    for behavior in range(5):
        for budget in GRID:
            record=dict(instance_id='mdp',policy_family=str(behavior),size_multiplier=budget,
                        mdp_seed=123,stream_seed=20000123,horizon=100,sha256='same-buffer',
                        stream='training',benchmark_split='evaluation')
            records.append(record)
            for method in ALGORITHMS:
                rows.append(dict(mdp_id='mdp',policy_family=str(behavior),training_size_multiplier=str(budget),
                    training_seed='0',algorithm=method,normalized_return='-.25',success='False',
                    run_status='observed',buffer_sha256='same-buffer'))
    return dict(instances=[dict(instance_id='mdp',split='evaluation')],datasets=records),rows


def test_exact_pairing_and_unclipped_negative_returns_are_preserved():
    manifest,rows=tiny_panel()
    indexed,runs,instances=verify_run_grid(manifest,rows,seeds=(0,))
    assert len(indexed)==30 and len(runs)==5 and set(instances)=={'mdp'}
    assert runs[('mdp','0')]['cfqi_tabular'][0]==(-.25,0)


@pytest.mark.parametrize('field,value,reason', [('buffer_sha256','tampered','hash mismatch'),
    ('normalized_return','nan','nonfinite'),('run_status','failed','Failed'),
    ('success','True','threshold'),('algorithm','foreign','unregistered')])
def test_changed_input_or_result_cannot_be_silently_accepted(field,value,reason):
    manifest,rows=tiny_panel(); rows[0][field]=value
    with pytest.raises(ValueError,match=reason):
        verify_run_grid(manifest,rows,seeds=(0,))


def test_missing_or_duplicate_results_fail_before_any_fitting():
    manifest,rows=tiny_panel()
    with pytest.raises(ValueError,match='Incomplete E1 method'):
        verify_run_grid(manifest,rows[:-1],seeds=(0,))
    with pytest.raises(ValueError,match='Duplicate'):
        verify_run_grid(manifest,rows+[deepcopy(rows[0])],seeds=(0,))


def test_incomplete_buffers_or_horizon_change_are_rejected():
    manifest,rows=tiny_panel(); manifest['datasets'].pop()
    with pytest.raises(ValueError,match='Incomplete registered'):
        verify_run_grid(manifest,rows,seeds=(0,))
    manifest,rows=tiny_panel(); manifest['datasets'][0]['horizon']=50
    with pytest.raises(ValueError,match='horizon'):
        verify_run_grid(manifest,rows,seeds=(0,))

"""Important failure cases for nested grouping and rare-event head fitting."""
import numpy as np
import pytest

from scripts.analysis import run_nested_improvement_20261012 as R


def test_shared_mdp_in_different_rows_is_rejected():
    with pytest.raises(ValueError,match='MDP'):
        R.assert_split(np.array(['same','same','other']),np.array([0,2]),np.array([1]))


def test_two_stage_all_censored_remains_a_valid_distribution():
    cfg=R.D.read_json(R.CONFIG)
    p=R.two_stage(np.zeros((8,2)),np.full(8,6),np.zeros((3,2)),cfg)
    assert p.shape==(3,7)
    assert np.all(p>=0)
    assert np.allclose(p.sum(1),1)
    assert np.all(p[:,6]>.9)
    assert np.isfinite(R.scores(p,np.array([0,3,6]))).all()


def test_two_stage_single_finite_class_preserves_six_categories():
    cfg=R.D.read_json(R.CONFIG)
    p=R.two_stage(np.zeros((8,2)),np.full(8,4),np.zeros((3,2)),cfg)
    assert np.allclose(p.sum(1),1)
    assert np.all(p.argmax(1)==4)
    assert np.isfinite(R.scores(p,np.array([0,4,6]))).all()


def test_bootstrap_does_not_make_zero_effects_significant():
    cfg=R.D.read_json(R.CONFIG)
    cfg={**cfg,'contrasts':[['current','temperature']],'conditional_bootstrap_draws':100}
    import pandas as pd
    frame=pd.DataFrame([dict(candidate=c,mdp_id=str(i),brier=.1,log_loss=.3)
                        for c in ('current','temperature') for i in range(8)])
    contrast,_=R.bounds(frame,cfg)
    assert (contrast.conditional_simultaneous_lower_95==0).all()

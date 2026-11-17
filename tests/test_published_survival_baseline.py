"""Checks for the published cumulative-failure adaptation at OEH budgets."""
import numpy as np

from benchmarks.bridge.fresh_published_survival_forecast import expanded_train, predict_law


def test_expanded_training_rows_preserve_event_and_administrative_censoring():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    matrix, label = expanded_train(x, np.array([0, 3, 6]))
    assert matrix.shape == (21, 3)
    assert label.shape == (21,)
    assert label[:6].tolist() == [1] * 6
    assert label[6:12].tolist() == [0, 0, 0, 1, 1, 1]
    assert label[12:18].tolist() == [0] * 6
    assert matrix[-3:, -1].tolist() == [1.0, 10.0, 50.0]
    assert label[-3:].tolist() == [1, 1, 0]


def test_survival_repair_produces_seven_class_law():
    class Model:
        def predict_proba(self, matrix):
            survival = np.array([.9, .8, .85, .4, .3, .2, .99, .8, .7, .6, .5, .4])
            return np.column_stack((survival, 1 - survival))

    law = predict_law(Model(), np.ones((2, 2)))
    assert law.shape == (2, 7)
    assert np.all(law >= 0)
    assert np.allclose(law.sum(axis=1), 1)
    assert law[0, 2] == 0  # .85 cannot reverse the preceding survival drop

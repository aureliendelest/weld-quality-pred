"""Reference predictors every model has to beat."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor

from .evaluation import ModelSpec


class StudyMeanRegressor(RegressorMixin, BaseEstimator):
    """Predict the mean of the same study seen in training; the global mean otherwise.

    Measures how much of a score is only study identity. On a study held out of the
    training set (grouped and leave-one-study-out protocols) it can only fall back to
    the global mean.
    """

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError("StudyMeanRegressor needs `groups` (the study of each row)")
        values = np.asarray(y, dtype=float)
        self.global_mean_ = float(values.mean())
        self.study_means_ = pd.Series(values).groupby(np.asarray(groups)).mean().to_dict()
        return self

    def predict(self, X, groups=None):
        if groups is None:
            raise ValueError("StudyMeanRegressor needs `groups` (the study of each row)")
        return np.array([self.study_means_.get(g, self.global_mean_) for g in groups])


def baseline_models() -> list:
    return [
        ModelSpec("global_mean", "tree", lambda: DummyRegressor(strategy="mean")),
        ModelSpec("study_mean", "tree", StudyMeanRegressor, uses_source=True),
    ]

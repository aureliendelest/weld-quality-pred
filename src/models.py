"""Estimators that need more than a scikit-learn default."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold


class GroupTunedRidge(RegressorMixin, BaseEstimator):
    """Ridge whose penalty is chosen by cross-validation over whole studies of the training data.

    Plain RidgeCV would pick a weak penalty because rows of the same study look alike; holding out
    studies measures what we care about, a study never seen. The study to be scored is never part
    of the choice, since only training rows reach `fit`. `predict` accepts and ignores `groups` so
    the model runs under `ModelSpec(uses_source=True)`.
    """

    def __init__(self, alphas=(1.0, 10.0, 100.0, 1000.0), n_splits=4):
        self.alphas = alphas
        self.n_splits = n_splits

    def fit(self, X, y, groups, sample_weight=None):
        n_splits = min(self.n_splits, len(np.unique(groups)))
        search = GridSearchCV(Ridge(), {"alpha": list(self.alphas)}, cv=GroupKFold(n_splits),
                              scoring="neg_mean_squared_error")
        search.fit(X, y, groups=groups, sample_weight=sample_weight)
        self.model_, self.alpha_ = search.best_estimator_, search.best_params_["alpha"]
        return self

    def predict(self, X, groups=None):
        return self.model_.predict(X)

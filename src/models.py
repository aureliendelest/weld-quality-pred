"""Estimators that need more than a scikit-learn default."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
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


class GroupTunedSearch(RegressorMixin, BaseEstimator):
    """Any regressor tuned by a grid search with cross-validation over whole studies of the training data.

    Same idea as `GroupTunedRidge`, for an arbitrary estimator and grid. The target is standardised
    on the training rows before the search (the penalty of a Lasso and the epsilon of an SVR depend
    on its scale) and the prediction is mapped back. `predict` accepts and ignores `groups` so the
    model runs under `ModelSpec(uses_source=True)`.
    """

    def __init__(self, estimator, grid, n_splits=4):
        self.estimator = estimator
        self.grid = grid
        self.n_splits = n_splits

    def fit(self, X, y, groups):
        values = np.asarray(y, dtype=float)
        self.center_, self.scale_ = values.mean(), values.std() or 1.0
        n_splits = min(self.n_splits, len(np.unique(groups)))
        search = GridSearchCV(clone(self.estimator), self.grid, cv=GroupKFold(n_splits),
                              scoring="neg_mean_squared_error")
        search.fit(X, (values - self.center_) / self.scale_, groups=groups)
        self.model_, self.best_params_ = search.best_estimator_, search.best_params_
        return self

    def predict(self, X, groups=None):
        return self.model_.predict(X) * self.scale_ + self.center_

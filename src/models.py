"""Estimators that need more than a scikit-learn default."""
import numpy as np
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, ParameterGrid


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
        self.chosen_ = {"alpha": self.alpha_}
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
        self.chosen_ = dict(self.best_params_)
        return self

    def predict(self, X, groups=None):
        return self.model_.predict(X) * self.scale_ + self.center_


class GroupTunedCatBoost(RegressorMixin, BaseEstimator):
    """CatBoost whose depth, L2 penalty and number of rounds are chosen by cross-validation over whole studies.

    One fit per (configuration, inner fold) at `max_rounds`; CatBoost logs the validation error after every
    round (`use_best_model=False`: nothing stops or truncates the fit), so the number of rounds is read off the
    error curve pooled over the inner folds instead of being a grid dimension that multiplies the cost. The
    final model is fitted on all the training rows with the chosen configuration and number of rounds. The
    learning rate is fixed. The target is standardised, as in `GroupTunedSearch`.
    """

    def __init__(self, grid, learning_rate=0.05, max_rounds=600, n_splits=4, random_seed=0):
        self.grid = grid
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.n_splits = n_splits
        self.random_seed = random_seed

    def _model(self, params, rounds):
        return CatBoostRegressor(iterations=rounds, learning_rate=self.learning_rate, verbose=0, thread_count=-1,
                                 random_seed=self.random_seed, allow_writing_files=False, **params)

    def fit(self, X, y, groups):
        values = np.asarray(y, dtype=float)
        self.center_, self.scale_ = values.mean(), values.std() or 1.0
        z = (values - self.center_) / self.scale_
        rows = (lambda index: X.iloc[index]) if hasattr(X, "iloc") else (lambda index: X[index])
        folds = list(GroupKFold(min(self.n_splits, len(np.unique(groups)))).split(z, z, groups))
        configs = list(ParameterGrid(self.grid))
        squared_error = np.zeros((len(configs), self.max_rounds))      # summed over the validation rows
        for i, params in enumerate(configs):
            for train, validation in folds:
                model = self._model(params, self.max_rounds).fit(rows(train), z[train], eval_set=(rows(validation), z[validation]),
                                                                 use_best_model=False)
                rmse = np.asarray(model.get_evals_result()["validation"]["RMSE"])
                squared_error[i] += rmse ** 2 * len(validation)
        best_config, best_index = np.unravel_index(np.argmin(squared_error), squared_error.shape)
        self.best_params_, self.best_rounds_ = configs[best_config], int(best_index) + 1
        self.cv_mse_ = squared_error.min() / len(z)
        self.chosen_ = {**self.best_params_, "rounds": self.best_rounds_}
        self.model_ = self._model(self.best_params_, self.best_rounds_).fit(X, z)
        return self

    def predict(self, X, groups=None):
        return self.model_.predict(X) * self.scale_ + self.center_

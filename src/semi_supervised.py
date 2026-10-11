"""Minimal semi-supervised regressor: confidence-ranked pseudo-labeling."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


class PseudoLabelRegressor(RegressorMixin, BaseEstimator):
    """Fit on labeled rows, then repeatedly add the most confident pseudo-labels.

    Confidence is the spread of the individual trees' predictions, so the base
    estimator must be a RandomForest/ExtraTrees regressor. Inputs are DataFrames
    (the output of the preprocessors). `X_unlabeled` rows must come from the
    training fold only.
    """

    def __init__(self, estimator, n_rounds=2, top_fraction=0.5):
        self.estimator = estimator
        self.n_rounds = n_rounds
        self.top_fraction = top_fraction

    @staticmethod
    def _tree_spread(model, X: pd.DataFrame):
        if not isinstance(model, (RandomForestRegressor, ExtraTreesRegressor)):
            raise TypeError("PseudoLabelRegressor needs a RandomForest/ExtraTrees base estimator")
        values = np.asarray(X, dtype=np.float32)
        per_tree = np.stack([tree.predict(values) for tree in model.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0)

    def fit(self, X: pd.DataFrame, y: pd.Series, X_unlabeled: pd.DataFrame | None = None):
        model = clone(self.estimator).fit(X, y)
        X_train, y_train, pool = X, y, X_unlabeled
        if pool is not None:
            for _ in range(self.n_rounds):
                if len(pool) == 0:
                    break
                mean, spread = self._tree_spread(model, pool)
                take = np.argsort(spread)[: max(1, int(len(pool) * self.top_fraction))]
                keep = np.ones(len(pool), dtype=bool)
                keep[take] = False
                X_train = pd.concat([X_train, pool.iloc[take]])
                y_train = pd.concat([y_train, pd.Series(mean[take], index=pool.index[take])])
                pool = pool.iloc[keep]
                model = clone(self.estimator).fit(X_train, y_train)
        self.model_ = model
        self.n_pseudo_labeled_ = len(X_train) - len(X)
        return self

    def predict(self, X):
        return self.model_.predict(X)

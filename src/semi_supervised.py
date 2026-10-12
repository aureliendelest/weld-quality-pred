"""Semi-supervised regressors: confidence-ranked pseudo-labeling and co-training."""
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

    def fit(self, X: pd.DataFrame, y: pd.Series, X_unlabeled: pd.DataFrame | None = None, sample_weight=None):
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        model = clone(self.estimator).fit(X, y, **fit_kwargs)
        X_train, y_train, pool = X, y, X_unlabeled
        if pool is not None and len(pool) > 0:
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


class CoTrainingRegressor(RegressorMixin, BaseEstimator):
    """Dual-model co-training for semi-supervised regression (Zhou & Li, 2005).

    Two diverse tree ensembles (e.g. RandomForestRegressor and ExtraTreesRegressor)
    iteratively teach each other by labeling the unlabeled samples where each has highest
    confidence (lowest inter-tree prediction spread). Predictions are averaged.
    """

    def __init__(self, estimator1=None, estimator2=None, n_rounds=2, top_fraction=0.3):
        self.estimator1 = estimator1
        self.estimator2 = estimator2
        self.n_rounds = n_rounds
        self.top_fraction = top_fraction

    @staticmethod
    def _tree_spread(model, X: pd.DataFrame):
        if not isinstance(model, (RandomForestRegressor, ExtraTreesRegressor)):
            raise TypeError("CoTrainingRegressor needs RandomForest/ExtraTrees base estimators")
        values = np.asarray(X, dtype=np.float32)
        per_tree = np.stack([tree.predict(values) for tree in model.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0)

    def fit(self, X: pd.DataFrame, y: pd.Series, X_unlabeled: pd.DataFrame | None = None, sample_weight=None):
        est1 = self.estimator1 if self.estimator1 is not None else RandomForestRegressor(n_estimators=100, random_state=0)
        est2 = self.estimator2 if self.estimator2 is not None else ExtraTreesRegressor(n_estimators=100, random_state=0)

        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        m1 = clone(est1).fit(X, y, **fit_kwargs)
        m2 = clone(est2).fit(X, y, **fit_kwargs)

        X1_train, y1_train = X.copy(), y.copy()
        X2_train, y2_train = X.copy(), y.copy()
        pool = X_unlabeled.copy() if X_unlabeled is not None else None

        if pool is not None and len(pool) > 0:
            for _ in range(self.n_rounds):
                if len(pool) == 0:
                    break
                mean1, spread1 = self._tree_spread(m1, pool)
                mean2, spread2 = self._tree_spread(m2, pool)

                k = max(1, int(len(pool) * self.top_fraction))
                take1 = np.argsort(spread1)[:k]
                take2 = np.argsort(spread2)[:k]

                # m1 teaches m2
                X2_train = pd.concat([X2_train, pool.iloc[take1]])
                y2_train = pd.concat([y2_train, pd.Series(mean1[take1], index=pool.index[take1])])

                # m2 teaches m1
                X1_train = pd.concat([X1_train, pool.iloc[take2]])
                y1_train = pd.concat([y1_train, pd.Series(mean2[take2], index=pool.index[take2])])

                taken_indices = set(take1).union(set(take2))
                keep = [i not in taken_indices for i in range(len(pool))]
                pool = pool.iloc[keep]

                m1 = clone(est1).fit(X1_train, y1_train)
                m2 = clone(est2).fit(X2_train, y2_train)

        self.model1_ = m1
        self.model2_ = m2
        self.n_pseudo_labeled_ = len(X1_train) - len(X)
        return self

    def predict(self, X):
        pred1 = self.model1_.predict(X)
        pred2 = self.model2_.predict(X)
        return 0.5 * (pred1 + pred2)

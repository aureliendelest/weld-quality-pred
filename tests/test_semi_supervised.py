import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from src.semi_supervised import CoTrainingRegressor, PseudoLabelRegressor


def test_pseudo_label_regressor_requires_tree_ensemble():
    reg = PseudoLabelRegressor(Ridge())
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    y = pd.Series([1.0, 2.0])
    pool = pd.DataFrame({"a": [1.5], "b": [3.5]})
    with pytest.raises(TypeError, match="needs a RandomForest/ExtraTrees base estimator"):
        reg.fit(X, y, X_unlabeled=pool)


def test_pseudo_label_regressor_no_unlabeled():
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    reg = PseudoLabelRegressor(rf)
    X = pd.DataFrame({"a": np.linspace(0, 10, 20)})
    y = pd.Series(np.linspace(0, 10, 20) * 2)
    reg.fit(X, y, X_unlabeled=None)
    assert reg.n_pseudo_labeled_ == 0
    pred = reg.predict(X)
    assert len(pred) == 20


def test_pseudo_label_regressor_adds_pseudo_labels():
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    reg = PseudoLabelRegressor(rf, n_rounds=2, top_fraction=0.5)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 4)))
    y = pd.Series(X.iloc[:, 0] * 3.0 + X.iloc[:, 1] * 2.0)
    pool = pd.DataFrame(rng.normal(size=(20, 4)))

    reg.fit(X, y, X_unlabeled=pool)
    # round 1: takes 50% of 20 = 10
    # round 2: takes 50% of 10 = 5
    # total pseudo-labeled = 15
    assert reg.n_pseudo_labeled_ == 15
    preds = reg.predict(pool)
    assert len(preds) == len(pool)


def test_pseudo_label_regressor_extra_trees():
    et = ExtraTreesRegressor(n_estimators=10, random_state=42)
    reg = PseudoLabelRegressor(et, n_rounds=1, top_fraction=0.4)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(25, 3)))
    y = pd.Series(X.iloc[:, 0] * 2.0)
    pool = pd.DataFrame(rng.normal(size=(10, 3)))

    reg.fit(X, y, X_unlabeled=pool)
    assert reg.n_pseudo_labeled_ == 4


def test_co_training_regressor():
    co = CoTrainingRegressor(n_rounds=1, top_fraction=0.3)
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(30, 4)))
    y = pd.Series(X.iloc[:, 0] * 2.5 - X.iloc[:, 1])
    pool = pd.DataFrame(rng.normal(size=(20, 4)))

    co.fit(X, y, X_unlabeled=pool)
    assert hasattr(co, "model1_")
    assert hasattr(co, "model2_")
    assert co.n_pseudo_labeled_ > 0
    preds = co.predict(pool)
    assert len(preds) == len(pool)


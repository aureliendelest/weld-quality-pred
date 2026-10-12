"""Per-target feature sets and sklearn preprocessors (always fit on the training fold).

Two families:
- "tree":   NaN kept, one-hot categoricals, missing/censored flags.
- "linear": same flags, then shifted log on skewed impurities, train-fold median
            imputation and RobustScaler. Also the input of PCA / semi-supervised models.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

from .cleaning import censored_flag_name
from .config import (
    CATEGORICAL_FEATURES, CENSORED_COLUMNS, EXCLUDED_FROM_MODELING,
    MISSING_FLAG_MIN_RATE, NUMERIC_FEATURES, OUTCOME_COLUMNS, SCALED_CLIP, SKEWED_COLUMNS,
    SOURCE_COLUMN, TARGETS, TEST_CONDITION_FEATURES,
)

FAMILIES = ("tree", "linear")
_FLAG_PATTERN = r"_(?:missing|censored)$"


@dataclass(frozen=True)
class FeatureSpec:
    target: str
    numeric: list
    categorical: list
    censored_flags: list

    @property
    def columns(self) -> list:
        return self.numeric + self.categorical + self.censored_flags


def feature_spec(df: pd.DataFrame, target: str) -> FeatureSpec:
    """Feature columns for one target, with leakage guards."""
    if target not in TARGETS:
        raise ValueError(f"{target!r} is not one of the retained targets {TARGETS}")
    numeric = NUMERIC_FEATURES + TEST_CONDITION_FEATURES.get(target, [])
    flags = [censored_flag_name(c) for c in numeric if c in CENSORED_COLUMNS]
    absent = [c for c in numeric + CATEGORICAL_FEATURES + flags if c not in df.columns]
    if absent:
        raise KeyError(f"columns missing from the cleaned frame: {absent}")
    spec = FeatureSpec(target, list(numeric), list(CATEGORICAL_FEATURES), flags)

    allowed_test_condition = set(TEST_CONDITION_FEATURES.get(target, []))
    forbidden = set(TARGETS) | set(OUTCOME_COLUMNS) | set(EXCLUDED_FROM_MODELING) | {SOURCE_COLUMN}
    forbidden |= {c for cols in TEST_CONDITION_FEATURES.values() for c in cols} - allowed_test_condition
    leaked = forbidden & set(spec.columns)
    if leaked:
        raise ValueError(f"leakage: {sorted(leaked)} in the features of {target}")
    return spec


def labeled_mask(df: pd.DataFrame, target: str) -> pd.Series:
    return df[target].notna()


class MissingnessFlags(TransformerMixin, BaseEstimator):
    """Add `<col>_missing` for columns missing in >= min_rate of the training fold.

    Also casts boolean columns (the `_censored` flags) to float.
    """

    def __init__(self, columns, min_rate=MISSING_FLAG_MIN_RATE):
        self.columns = columns
        self.min_rate = min_rate

    def fit(self, X, y=None):
        rates = X[list(self.columns)].isna().mean()
        self.flagged_ = [c for c in self.columns if rates[c] >= self.min_rate]
        return self

    def transform(self, X):
        out = X.copy()
        for column in out.columns[out.dtypes == bool]:
            out[column] = out[column].astype(float)
        for column in self.flagged_:
            out[f"{column}_missing"] = X[column].isna().astype(float)
        return out


class LogShift(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """log(x + s) with s the smallest positive training value of each column.

    log1p is nearly the identity on wt% values of 0.001-0.14, so it would not
    reduce the skew; a data-scaled shift keeps zeros finite.
    """

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        positive = np.where(values > 0, values, np.nan)
        self.shift_ = np.where(np.isnan(positive).all(axis=0), 1.0, np.nanmin(positive, axis=0))
        self.n_features_in_ = values.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        return np.log(np.asarray(X, dtype=float) + self.shift_)


def _one_hot():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_preprocessor(family: str, spec: FeatureSpec, clip: float | None = SCALED_CLIP) -> Pipeline:
    """Unfitted preprocessor for a family; input is `df[spec.columns]`.

    `clip` bounds the scaled numeric features of the linear family (None disables it).
    """
    flags = MissingnessFlags(columns=spec.numeric)
    if family == "tree":
        columns = ColumnTransformer(
            [("categorical", _one_hot(), spec.categorical)],
            remainder="passthrough", verbose_feature_names_out=False,
        )
    elif family == "linear":
        skewed = [c for c in SKEWED_COLUMNS if c in spec.numeric]
        regular = [c for c in spec.numeric if c not in skewed]

        def scaled(*steps):
            tail = []
            if clip is not None:
                bound = lambda X: np.clip(np.asarray(X, dtype=float), -clip, clip)
                tail = [("clip", FunctionTransformer(bound, feature_names_out="one-to-one"))]
            return Pipeline([*steps, ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                             ("scale", RobustScaler()), *tail])

        columns = ColumnTransformer(
            [
                ("skewed", scaled(("log", LogShift())), skewed),
                ("regular", scaled(), regular),
                ("flags", "passthrough", make_column_selector(pattern=_FLAG_PATTERN)),
                ("categorical", _one_hot(), spec.categorical),
            ],
            verbose_feature_names_out=False,
        )
    else:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    columns.set_output(transform="pandas")
    return Pipeline([("flags", flags), ("columns", columns)])


def build_pipeline(family: str, spec: FeatureSpec, estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(family, spec)), ("model", estimator)])

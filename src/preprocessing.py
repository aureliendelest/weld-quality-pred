"""Per-target feature sets and sklearn preprocessors (always fit on the training fold).

Two families:
- "tree":   NaN kept, one-hot categoricals, missing/censored flags.
- "linear": same flags, then shifted log on skewed impurities, train-fold median
            imputation and RobustScaler. Also the input of PCA / semi-supervised models.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

from .cleaning import censored_flag_name
from .config import (
    CATEGORICAL_FEATURES, CENSORED_COLUMNS, DERIVED_FEATURES, EXCLUDED_FROM_MODELING,
    MISSING_FLAG_MIN_RATE, NUMERIC_FEATURES, OUTCOME_COLUMNS, RANDOM_SEED, SCALED_CLIP,
    SKEWED_COLUMNS, SOURCE_COLUMN, TARGETS, TEST_CONDITION_FEATURES,
)

FAMILIES = ("tree", "linear")
_FLAG_PATTERN = r"_(?:missing|censored)$"


@dataclass(frozen=True)
class FeatureOptions:
    """Departures from the default feature set, for ablations (the defaults are the plan of doc/01)."""
    missing_flags: bool = True        # `<col>_missing` flags
    censored_flags: bool = True       # `<col>_censored` flags
    drop: tuple = ()                  # numeric features removed (their censored flag goes too)
    extra: tuple = ()                 # derived features added (names of DERIVED_FEATURES)
    skewed_extra: tuple = ()          # more columns for the shifted log (linear family)
    pca_components: int = 0           # PCA scores appended, fitted inside each training fold


def add_derived(df: pd.DataFrame, names) -> pd.DataFrame:
    """Copy of df with derived features; each is a row-wise function of inputs (no fitting)."""
    unknown = set(names) - set(DERIVED_FEATURES)
    if unknown:
        raise KeyError(f"unknown derived features {sorted(unknown)}; known: {sorted(DERIVED_FEATURES)}")
    out = df.copy()
    if "ce_iiw" in names:
        out["ce_iiw"] = (df["c_conc_wt"] + df["mn_conc_wt"] / 6
                         + (df["cr_conc_wt"] + df["mo_conc_wt"] + df["v_conc_wt"]) / 5
                         + (df["ni_conc_wt"] + df["cu_conc_wt"]) / 15)
    if "larson_miller" in names:
        temperature, hours = df["pwht_temp_c"], df["pwht_time_h"]
        parameter = (temperature + 273.15) * (20 + np.log10(hours.where(hours > 0)))
        out["larson_miller"] = parameter.where(temperature > 0, 0.0).where(temperature.notna())
    return out


@dataclass(frozen=True)
class FeatureSpec:
    target: str
    numeric: list
    categorical: list
    censored_flags: list

    @property
    def columns(self) -> list:
        return self.numeric + self.categorical + self.censored_flags


def feature_spec(df: pd.DataFrame, target: str, options: FeatureOptions | None = None) -> FeatureSpec:
    """Feature columns for one target, with leakage guards.

    `df` must already hold the derived features of `options.extra` (see `add_derived`).
    """
    if target not in TARGETS:
        raise ValueError(f"{target!r} is not one of the retained targets {TARGETS}")
    options = options or FeatureOptions()
    base = NUMERIC_FEATURES + TEST_CONDITION_FEATURES.get(target, [])
    not_features = set(options.drop) - set(base)
    if not_features:
        raise ValueError(f"cannot drop {sorted(not_features)}: not numeric features of {target}")
    numeric = [c for c in base if c not in options.drop] + list(options.extra)
    flags = ([censored_flag_name(c) for c in numeric if c in CENSORED_COLUMNS]
             if options.censored_flags else [])
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


class WithPCAScores(TransformerMixin, BaseEstimator):
    """Append PCA scores of the scaled numeric features to the output of `base`.

    Everything (the base, the scaler, the PCA) is fitted on the training fold only.
    """

    def __init__(self, base, spec, n_components, clip=SCALED_CLIP):
        self.base = base
        self.spec = spec
        self.n_components = n_components
        self.clip = clip

    def _numeric(self, X):
        return X[list(self.spec.numeric)]

    def fit(self, X, y=None):
        self.base_ = clone(self.base).fit(X)
        numeric_only = FeatureSpec(self.spec.target, list(self.spec.numeric), [], [])
        plain = FeatureOptions(missing_flags=False, censored_flags=False)
        self.scaler_ = build_preprocessor("linear", numeric_only, self.clip, plain).fit(self._numeric(X))
        self.pca_ = PCA(self.n_components, random_state=RANDOM_SEED).fit(self.scaler_.transform(self._numeric(X)))
        return self

    def transform(self, X):
        out = self.base_.transform(X).copy()
        scores = self.pca_.transform(self.scaler_.transform(self._numeric(X)))
        for i in range(self.n_components):
            out[f"pca_{i + 1}"] = scores[:, i]
        return out


def build_preprocessor(family: str, spec: FeatureSpec, clip: float | None = SCALED_CLIP,
                       options: FeatureOptions | None = None):
    """Unfitted preprocessor for a family; input is `df[spec.columns]`.

    `clip` bounds the scaled numeric features of the linear family (None disables it).
    `options` is for ablations; the default reproduces the plan of doc/01.
    """
    options = options or FeatureOptions()
    # A rate above 1 is never reached: no `_missing` flag is created (booleans are still cast).
    flags = MissingnessFlags(columns=spec.numeric,
                             min_rate=MISSING_FLAG_MIN_RATE if options.missing_flags else 2.0)
    if family == "tree":
        columns = ColumnTransformer(
            [("categorical", _one_hot(), spec.categorical)],
            remainder="passthrough", verbose_feature_names_out=False,
        )
    elif family == "linear":
        skewed = [c for c in dict.fromkeys([*SKEWED_COLUMNS, *options.skewed_extra]) if c in spec.numeric]
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
    pipeline = Pipeline([("flags", flags), ("columns", columns)])
    if options.pca_components:
        return WithPCAScores(pipeline, spec, options.pca_components, clip)
    return pipeline


def build_pipeline(family: str, spec: FeatureSpec, estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(family, spec)), ("model", estimator)])

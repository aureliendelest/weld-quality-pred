"""Same folds, same metrics, one score per target: a fair model comparison.

Two protocols are always reported on identical folds:
- "grouped": GroupKFold by source (generalisation to an unseen study)
- "random":  stratified shuffled KFold (in-distribution performance)
A model is fit and scored per target on the rows where that target exists.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .config import N_SPLITS, RANDOM_SEED, SOURCE_COLUMN, TARGETS
from .preprocessing import build_preprocessor, feature_spec

PROTOCOLS = ("grouped", "random")
MIN_TEST_ROWS = 10          # a fold with fewer labeled test rows gives a meaningless R2


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str                        # "tree" or "linear" (see preprocessing.py)
    factory: Callable[[], object]      # returns a fresh estimator
    semi_supervised: bool = False      # estimator.fit(X, y, X_unlabeled=...)


def make_folds(df: pd.DataFrame, protocol: str, n_splits: int = N_SPLITS, seed: int = RANDOM_SEED):
    """List of (train_positions, test_positions) over all rows of df."""
    positions = np.arange(len(df))
    if protocol == "grouped":
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(positions, groups=df[SOURCE_COLUMN]))
    if protocol == "random":
        sizes = df[SOURCE_COLUMN].map(df[SOURCE_COLUMN].value_counts())
        strata = df[SOURCE_COLUMN].where(sizes >= n_splits, "small_sources")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(positions, strata))
    raise ValueError(f"protocol must be one of {PROTOCOLS}")


def unlabeled_pool(df: pd.DataFrame, target: str) -> pd.Index:
    """Rows where this target is missing (usable as unlabeled data for it)."""
    return df.index[df[target].isna()]


def evaluate_model(df: pd.DataFrame, model: ModelSpec, folds: dict, targets=TARGETS) -> pd.DataFrame:
    """One row per (target, protocol, fold) with R2 / RMSE / MAE on the labeled test rows."""
    rows = []
    for target in targets:
        spec = feature_spec(df, target)
        X, y = df[spec.columns], df[target]
        labeled = y.notna().to_numpy()
        for protocol, protocol_folds in folds.items():
            for k, (train, test) in enumerate(protocol_folds):
                train_labeled, test_labeled = train[labeled[train]], test[labeled[test]]
                if len(test_labeled) < MIN_TEST_ROWS:
                    continue
                preprocessor = build_preprocessor(model.family, spec).fit(X.iloc[train_labeled])
                estimator = model.factory()
                X_train = preprocessor.transform(X.iloc[train_labeled])
                unlabeled = train[~labeled[train]]
                if model.semi_supervised:
                    estimator.fit(X_train, y.iloc[train_labeled],
                                  X_unlabeled=preprocessor.transform(X.iloc[unlabeled]))
                else:
                    estimator.fit(X_train, y.iloc[train_labeled])
                predicted = estimator.predict(preprocessor.transform(X.iloc[test_labeled]))
                truth = y.iloc[test_labeled]
                rows.append({
                    "model": model.name, "target": target, "protocol": protocol, "fold": k,
                    "n_train": len(train_labeled), "n_test": len(test_labeled),
                    "n_unlabeled": len(unlabeled) if model.semi_supervised else 0,
                    "r2": r2_score(truth, predicted),
                    "rmse": root_mean_squared_error(truth, predicted),
                    "mae": mean_absolute_error(truth, predicted),
                })
    return pd.DataFrame(rows)


def compare_models(df: pd.DataFrame, models: list, targets=TARGETS, protocols=PROTOCOLS,
                   n_splits: int = N_SPLITS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Fold-level results of every model on the same folds."""
    folds = {p: make_folds(df, p, n_splits, seed) for p in protocols}
    return pd.concat([evaluate_model(df, m, folds, targets) for m in models], ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over folds, per model / target / protocol."""
    grouped = results.groupby(["target", "protocol", "model"])[["r2", "rmse", "mae"]]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_folds"] = results.groupby(["target", "protocol", "model"]).size()
    return summary.reset_index()


def protocol_gap(summary: pd.DataFrame) -> pd.DataFrame:
    """R2(random) - R2(grouped): how much performance depends on study protocol."""
    wide = summary.pivot_table(index=["target", "model"], columns="protocol", values="r2_mean")
    wide["gap"] = wide["random"] - wide["grouped"]
    return wide.reset_index()

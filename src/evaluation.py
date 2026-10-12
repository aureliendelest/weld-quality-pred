"""Same folds, same metrics, one score per target: a fair model comparison.

Protocols, always run on identical folds:
- "grouped": GroupKFold by source (generalisation to an unseen study)
- "random":  stratified shuffled KFold (in-distribution performance)
- "loso":    one fold per study with enough rows (leave one study out), for a score per study
- "grouped_weld": GroupKFold by input fingerprint (the same weld, any Charpy temperature, stays on one
             side). Not for ranking models: random - grouped_weld is the leak through twin rows, and
             grouped_weld - grouped is the cost of a new study.
A model is fit and scored per target on the rows where that target exists.

Besides R2 (which compares with the mean of the *test* fold, a value nobody knows in
advance), every row reports `skill`: 1 - SSE(model) / SSE(training-fold mean). A model
that only predicts the training mean has a skill of exactly 0.
"""
import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .config import (
    CATEGORICAL_FEATURES, LOSO_MIN_ROWS, N_SPLITS, NUMERIC_FEATURES, RANDOM_SEED, SCALED_CLIP,
    SOURCE_COLUMN, TARGETS,
)
from .preprocessing import STUDY_KEY, FeatureOptions, add_derived, build_preprocessor, feature_spec

PROTOCOLS = ("grouped", "random")            # the default pair
ALL_PROTOCOLS = PROTOCOLS + ("loso",)
KNOWN_PROTOCOLS = ALL_PROTOCOLS + ("grouped_weld",)   # grouped_weld is opt-in (a diagnostic, not a ranking protocol)
MIN_TEST_ROWS = 10          # a fold with fewer labeled test rows gives a meaningless R2
METRICS = ["r2", "rmse", "mae", "skill", "bias", "within_rho"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str                        # "tree" or "linear" (see preprocessing.py)
    factory: Callable[[], object]      # returns a fresh estimator
    semi_supervised: bool = False      # estimator.fit(X, y, X_unlabeled=...)
    uses_source: bool = False          # estimator.fit(X, y, groups=...) and predict(X, groups=...)
    clip: float | None = SCALED_CLIP   # bound on the scaled features (linear family only)
    options: FeatureOptions = FeatureOptions()   # feature-set variant (ablations)
    study_weights: bool = False        # estimator.fit(..., sample_weight=1/sqrt(rows of the study))


def weld_fingerprint(df: pd.DataFrame) -> pd.Series:
    """Integer id of the full input vector (numeric + categorical features, no test condition).

    Rows with the same id are the same weld: replicates, or one weld read at several Charpy
    temperatures. NaN is part of the vector (two rows missing the same fields can match).
    """
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return df.groupby(columns, dropna=False, sort=False).ngroup()


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
    if protocol == "loso":
        sizes = df[SOURCE_COLUMN].value_counts()
        labels = df[SOURCE_COLUMN].to_numpy()
        return [(positions[labels != study], positions[labels == study])
                for study in sorted(sizes[sizes >= LOSO_MIN_ROWS].index)]
    if protocol == "grouped_weld":
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(positions, groups=weld_fingerprint(df)))
    raise ValueError(f"protocol must be one of {KNOWN_PROTOCOLS}")


def study_sample_weights(studies) -> np.ndarray:
    """1/sqrt(rows of the study), scaled to a mean of 1: a large study counts less per row."""
    studies = pd.Series(np.asarray(studies))
    weights = 1.0 / np.sqrt(studies.map(studies.value_counts()).to_numpy(dtype=float))
    return weights * len(weights) / weights.sum()


def unlabeled_pool(df: pd.DataFrame, target: str) -> pd.Index:
    """Rows where this target is missing (usable as unlabeled data for it)."""
    return df.index[df[target].isna()]


def _skill(truth: np.ndarray, predicted: np.ndarray, training_mean: float) -> float:
    reference = ((truth - training_mean) ** 2).sum()
    return float(1 - ((truth - predicted) ** 2).sum() / reference) if reference > 0 else np.nan


def _within_study_rho(predicted: np.ndarray, truth: np.ndarray, studies: np.ndarray, min_rows: int = 5) -> float:
    """Spearman between prediction and truth once each study's mean is removed from both.

    In a leave-one-study-out fold this is the plain within-study correlation: does the
    model follow the variations inside the study, whatever its error on the level?
    """
    data = pd.DataFrame({"p": predicted, "t": truth, "s": studies})
    data = data[data.groupby("s")["p"].transform("size") >= min_rows]
    if len(data) < MIN_TEST_ROWS:
        return np.nan
    for column in ("p", "t"):
        data[column] = data[column] - data.groupby("s")[column].transform("mean")
    if data["p"].std() < 1e-9 or data["t"].std() < 1e-9:
        return np.nan
    return float(data["p"].corr(data["t"], method="spearman"))


def evaluate_model(df: pd.DataFrame, model: ModelSpec, folds: dict, targets=TARGETS) -> pd.DataFrame:
    """One row per (target, protocol, fold) with the metrics of METRICS on the labeled test rows."""
    rows = []
    source = df[SOURCE_COLUMN]
    frame = add_derived(df, model.options.extra)
    for target in targets:
        spec = feature_spec(frame, target, model.options)
        X, y = frame[spec.columns], frame[target]
        if model.options.needs_study:
            X = X.assign(**{STUDY_KEY: source})      # only to centre by study, see ReplaceNumeric
        labeled = y.notna().to_numpy()
        for protocol, protocol_folds in folds.items():
            min_test = LOSO_MIN_ROWS if protocol == "loso" else MIN_TEST_ROWS
            for k, (train, test) in enumerate(protocol_folds):
                train_labeled, test_labeled = train[labeled[train]], test[labeled[test]]
                if len(test_labeled) < min_test:
                    continue
                preprocessor = build_preprocessor(model.family, spec, clip=model.clip, options=model.options).fit(X.iloc[train_labeled])
                estimator = model.factory()
                X_train = preprocessor.transform(X.iloc[train_labeled])
                unlabeled = train[~labeled[train]]
                fit_args = {}
                if model.semi_supervised:
                    fit_args["X_unlabeled"] = preprocessor.transform(X.iloc[unlabeled])
                if model.uses_source:
                    fit_args["groups"] = source.iloc[train_labeled].to_numpy()
                if model.study_weights:
                    fit_args["sample_weight"] = study_sample_weights(source.iloc[train_labeled])
                estimator.fit(X_train, y.iloc[train_labeled], **fit_args)
                predict_args = {"groups": source.iloc[test_labeled].to_numpy()} if model.uses_source else {}
                predicted = np.asarray(estimator.predict(preprocessor.transform(X.iloc[test_labeled]), **predict_args))
                truth = y.iloc[test_labeled].to_numpy()
                rows.append({
                    "model": model.name, "target": target, "protocol": protocol, "fold": k,
                    "study": source.iloc[test[0]] if protocol == "loso" else None,
                    "n_train": len(train_labeled), "n_test": len(test_labeled),
                    "n_unlabeled": len(unlabeled) if model.semi_supervised else 0,
                    "chosen": json.dumps(estimator.chosen_, sort_keys=True) if hasattr(estimator, "chosen_") else None,
                    "r2": r2_score(truth, predicted),
                    "rmse": root_mean_squared_error(truth, predicted),
                    "mae": mean_absolute_error(truth, predicted),
                    "skill": _skill(truth, predicted, float(y.iloc[train_labeled].mean())),
                    "bias": float(np.mean(predicted - truth)),
                    "within_rho": _within_study_rho(predicted, truth, source.iloc[test_labeled].to_numpy()),
                })
    return pd.DataFrame(rows)


def compare_models(df: pd.DataFrame, models: list, targets=TARGETS, protocols=PROTOCOLS,
                   n_splits: int = N_SPLITS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Fold-level results of every model on the same folds."""
    folds = {p: make_folds(df, p, n_splits, seed) for p in protocols}
    return pd.concat([evaluate_model(df, m, folds, targets) for m in models], ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over folds, per target / protocol / model."""
    keys = ["target", "protocol", "model"]
    summary = results.groupby(keys)[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_folds"] = results.groupby(keys).size()
    return summary.reset_index()


def protocol_gap(summary: pd.DataFrame) -> pd.DataFrame:
    """R2(random) - R2(grouped): how much performance depends on study protocol."""
    wide = summary.pivot_table(index=["target", "model"], columns="protocol", values="r2_mean")
    wide["gap"] = wide["random"] - wide["grouped"]
    return wide.reset_index()


def per_study_scores(results: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-study-out rows: one score per held-out study, target and model."""
    columns = ["target", "study", "model", "n_train", "n_test", "r2", "rmse", "mae", "skill", "bias", "within_rho"]
    loso = results[results["protocol"] == "loso"]
    return loso[columns].sort_values(["target", "study", "model"]).reset_index(drop=True)

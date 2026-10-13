"""From predictions to a decision: intervals calibrated by held-out study, then conforming / to test / non conforming.

The error of a model on a new study is mostly a level error shared by all the welds of the study (notebook 03), so an
interval calibrated on mixed rows would be too narrow. Here the unit of calibration is the study: every weld of a
held-out study gets the half-width that the *other* studies' leave-one-study-out residuals suggest (split conformal
prediction with the study as the exchangeable unit, each study weighted the same). With the 5-7 studies of a target the
guarantee is weak and the coverage is checked study by study. The decision rule is `Specification.decide`
(doc/06_quality_decision_plan.md).
"""
import numpy as np
import pandas as pd

from .config import LOSO_MIN_ROWS, SOURCE_COLUMN, TARGETS
from .evaluation import ModelSpec, fit_fold, make_folds, predict_rows
from .preprocessing import add_derived, feature_spec
from .specification import (
    CHARPY, CHARPY_TEMPERATURE, CONFORMING, NON_CONFORMING, TO_MEASURE, TO_TEST, Specification,
)

LEVEL = 0.8        # nominal coverage of the intervals
CHARPY_TEMPERATURES = (-45, -40)     # the temperatures of the Charpy requirements of the registry


def loso_predictions(df: pd.DataFrame, model: ModelSpec, targets=TARGETS, charpy_temperatures=(), log=None) -> pd.DataFrame:
    """Prediction of every weld of each held-out study, by a model fitted on the other studies.

    One row per (weld, target); `truth` is NaN where the property was not reported. For the Charpy energy the weld
    is also predicted at each of `charpy_temperatures` (`override_c`), the temperature at which a specification
    requires it; `override_c` is NaN for the prediction at the temperature actually tested.
    """
    source = df[SOURCE_COLUMN]
    frame = add_derived(df, model.options.extra)
    frames = []
    for target in targets:
        spec = feature_spec(frame, target, model.options)
        X, y = frame[spec.columns], frame[target]
        labeled = y.notna().to_numpy()
        for train, test in make_folds(df, "loso"):
            train_labeled = train[labeled[train]]
            if len(test[labeled[test]]) < LOSO_MIN_ROWS:
                continue
            preprocessor, estimator = fit_fold(model, spec, X, y, source, train_labeled)

            def record(rows, override):
                predicted = predict_rows(model, preprocessor, estimator, rows, source.iloc[test])
                frames.append(pd.DataFrame({"row": df.index[test], "target": target, "study": source.iloc[test[0]],
                                            "truth": y.iloc[test].to_numpy(), "prediction": predicted, "override_c": override}))

            record(X.iloc[test], np.nan)
            if target == CHARPY:
                for temperature in charpy_temperatures:
                    record(X.iloc[test].assign(**{CHARPY_TEMPERATURE: float(temperature)}), float(temperature))
        if log:
            log(target)
    return pd.concat(frames, ignore_index=True)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][min(np.searchsorted(cumulative, level - 1e-12), len(values) - 1)])


def study_quantiles(predictions: pd.DataFrame, level: float = LEVEL) -> pd.DataFrame:
    """Half-width of the interval of each (target, held-out study), from the residuals of the other studies.

    Each other study weighs the same, whatever its number of welds. The level is raised by the factor (K + 1) / K of
    split conformal prediction for K calibration studies, and capped at the largest residual.
    """
    measured = predictions[predictions["override_c"].isna() & predictions["truth"].notna()].copy()
    measured["residual"] = (measured["truth"] - measured["prediction"]).abs()
    rows = []
    for target, part in measured.groupby("target"):
        studies = part["study"].unique()
        for study in studies:
            others = part[part["study"] != study]
            weights = 1.0 / others.groupby("study")["residual"].transform("size").to_numpy()
            k = others["study"].nunique()
            rows.append({"target": target, "study": study, "n_calibration_studies": k,
                         "quantile": _weighted_quantile(others["residual"].to_numpy(), weights, min(1.0, level * (k + 1) / k))})
    return pd.DataFrame(rows)


def add_intervals(predictions: pd.DataFrame, quantiles: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(quantiles[["target", "study", "quantile"]], on=["target", "study"], how="left")
    return merged.assign(lower=merged["prediction"] - merged["quantile"], upper=merged["prediction"] + merged["quantile"])


def interval_coverage(intervals: pd.DataFrame) -> pd.DataFrame:
    """Share of the measured welds inside their interval, per target and held-out study, and the mean width."""
    measured = intervals[intervals["override_c"].isna() & intervals["truth"].notna()].copy()
    measured["inside"] = (measured["truth"] >= measured["lower"]) & (measured["truth"] <= measured["upper"])
    return measured.groupby(["target", "study"]).agg(n=("inside", "size"), coverage=("inside", "mean"),
                                                     width=("quantile", lambda q: 2 * q.iloc[0])).reset_index()


def decide_welds(df: pd.DataFrame, intervals: pd.DataFrame, spec: Specification, point: bool = False) -> pd.DataFrame:
    """Decision of every weld of the dataset for one specification (to_measure where it has no prediction).

    With `point=True` the interval is reduced to the prediction itself: the rule of a user who ignores the uncertainty.
    """
    per_requirement = {}
    for requirement in spec.requirements:
        wanted = intervals["target"] == requirement.target
        wanted &= (intervals["override_c"] == requirement.test_temperature_c) if requirement.test_temperature_c is not None \
            else intervals["override_c"].isna()
        part = intervals[wanted].drop_duplicates("row").set_index("row")
        per_requirement[requirement.target] = (part["prediction"], part["prediction"]) if point else (part["lower"], part["upper"])
    rows = []
    for row in df.index:
        bounds = {}
        for target, (lower, upper) in per_requirement.items():
            if row in lower.index:
                bounds[target] = (lower[row], upper[row])
        decision = spec.decide(bounds)
        overall = TO_MEASURE if set(decision.by_target.values()) == {TO_MEASURE} else decision.overall    # no prediction at all
        rows.append({"row": row, "decision": overall, **{f"state_{t}": s for t, s in decision.by_target.items()}})
    return pd.DataFrame(rows).set_index("row")


def decision_report(decisions: pd.DataFrame, labels: pd.Series, studies: pd.Series) -> pd.DataFrame:
    """Calls against the measured conformity, on the welds whose conformity is known and that have a prediction.

    `false_accepts` are the welds called conforming that do not conform: the costly error.
    """
    table = decisions.join(labels.rename("label")).join(studies.rename("study"))
    table = table[table["label"].isin([CONFORMING, NON_CONFORMING]) & (table["decision"] != TO_MEASURE)]    # welds of a study never held out are left out
    rows = []
    for group, part in [("all", table), *table.groupby("study")]:
        called = part[part["decision"] == CONFORMING]
        rows.append({"study": group, "welds": len(part), "true_conforming": int((part["label"] == CONFORMING).sum()),
                     "called_conforming": len(called), "false_accepts": int((called["label"] == NON_CONFORMING).sum()),
                     "to_test": int((part["decision"] == TO_TEST).sum()),
                     "called_non_conforming": int((part["decision"] == NON_CONFORMING).sum()),
                     "false_rejects": int(((part["decision"] == NON_CONFORMING) & (part["label"] == CONFORMING)).sum())})
    return pd.DataFrame(rows)


def run_predictions(df: pd.DataFrame, out_dir, rerun: bool = False, log=None) -> pd.DataFrame:
    """Leave-one-study-out predictions of the retained model (CatBoost, rounds tuned); the saved table is reused."""
    from .zoo import tuned_rounds_model          # zoo imports the ablation module: keep this module importable alone
    file = out_dir / "06_loso_predictions.csv"
    if file.exists() and not rerun:
        return pd.read_csv(file)
    predictions = loso_predictions(df, tuned_rounds_model(), charpy_temperatures=CHARPY_TEMPERATURES, log=log)
    predictions.to_csv(file, index=False)
    return predictions

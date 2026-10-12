"""Turn the raw text frame into typed numeric data, driven by the config registry.

Every parser raises on a value it does not recognise: a new lab notation must
never become a silent NaN.
"""
import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_FILL_UNKNOWN, CENSORED_COLUMNS, CENSORED_VALUE_FACTOR,
    HARDNESS_COLUMN, HARDNESS_SCALE_COLUMN, HARDNESS_UNKNOWN_SCALE, NITROGEN_COLUMN,
    NUMERIC_COLUMNS, RANGE_COLUMNS, UNIT_FIXES, UNKNOWN_CATEGORY, WELD_TYPE_ALIASES,
    WELD_TYPE_OTHER, WELD_TYPE_OTHER_LABEL,
)

_NUMBER = r"(\d*\.?\d+)"
_CENSORED = rf"^<\s*{_NUMBER}$"
_RANGE = rf"^{_NUMBER}-{_NUMBER}$"
_NITROGEN = rf"^{_NUMBER}tot"
_HARDNESS = rf"^{_NUMBER}\(?Hv(\d+)\)?$"


def _fail_on_unparsed(series: pd.Series, parsed: pd.Series, column: str) -> None:
    unparsed = series[series.notna() & parsed.isna()]
    if len(unparsed):
        raise ValueError(f"{column}: unrecognised values {sorted(unparsed.unique())[:10]}")


def censored_flag_name(column: str) -> str:
    return f"{column}_censored"


def parse_censored(series: pd.Series, column: str,
                   factor: float | None = CENSORED_VALUE_FACTOR) -> tuple[pd.Series, pd.Series]:
    """'<x' -> x * factor; returns (values, censored flag).

    `factor=None` keeps censored values as NaN (used for sensitivity analyses).
    """
    s = series.copy()
    for (col, bad), good in UNIT_FIXES.items():
        if col == column:
            s = s.where(s != bad, good)
    flag = s.str.match(_CENSORED, na=False)
    thresholds = pd.to_numeric(s.str.extract(_CENSORED)[0])
    values = pd.to_numeric(s.where(~flag), errors="coerce")
    unparsed = s[s.notna() & ~flag & values.isna()]
    if len(unparsed):
        raise ValueError(f"{column}: unrecognised values {sorted(unparsed.unique())[:10]}")
    values[flag] = np.nan if factor is None else thresholds[flag] * factor
    return values, flag


def parse_range(series: pd.Series, column: str) -> pd.Series:
    """'a-b' -> mean of the bounds; plain numbers are kept."""
    bounds = series.str.extract(_RANGE).apply(pd.to_numeric)
    values = pd.to_numeric(series, errors="coerce").fillna(bounds.mean(axis=1))
    _fail_on_unparsed(series, values, column)
    return values


def parse_nitrogen(series: pd.Series) -> pd.Series:
    """'67tot33res' / '66totndres' -> 67 / 66 (total only)."""
    total = pd.to_numeric(series.str.extract(_NITROGEN)[0])
    values = pd.to_numeric(series, errors="coerce").fillna(total)
    _fail_on_unparsed(series, values, NITROGEN_COLUMN)
    return values


def parse_hardness(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'158(Hv30)' / '459Hv10' / '203' -> (value, scale in {Hv10, Hv30, unknown})."""
    parts = series.str.extract(_HARDNESS)
    plain = pd.to_numeric(series, errors="coerce")
    values = plain.fillna(pd.to_numeric(parts[0]))
    _fail_on_unparsed(series, values, HARDNESS_COLUMN)
    scale = ("Hv" + parts[1]).where(parts[1].notna(), HARDNESS_UNKNOWN_SCALE)
    return values, scale.where(values.notna())


def normalise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    weld = df["weld_type"].replace(WELD_TYPE_ALIASES)
    df["weld_type"] = weld.where(~weld.isin(WELD_TYPE_OTHER), WELD_TYPE_OTHER_LABEL)
    for column in CATEGORICAL_FILL_UNKNOWN:
        df[column] = df[column].fillna(UNKNOWN_CATEGORY)
    return df


def clean_dataframe(raw: pd.DataFrame, censored_factor: float | None = CENSORED_VALUE_FACTOR) -> pd.DataFrame:
    """Typed numeric frame + `<col>_censored` flags + hardness scale.

    `censored_factor`: value given to `<x` is x * factor (None keeps them NaN).
    """
    df = raw.copy()
    done = set()

    for column in CENSORED_COLUMNS:
        df[column], df[censored_flag_name(column)] = parse_censored(raw[column], column, censored_factor)
        done.add(column)
    for column in RANGE_COLUMNS:
        df[column] = parse_range(raw[column], column)
        done.add(column)
    df[NITROGEN_COLUMN] = parse_nitrogen(raw[NITROGEN_COLUMN])
    done.add(NITROGEN_COLUMN)
    df[HARDNESS_COLUMN], df[HARDNESS_SCALE_COLUMN] = parse_hardness(raw[HARDNESS_COLUMN])
    done.add(HARDNESS_COLUMN)

    for column in NUMERIC_COLUMNS:
        if column not in done:
            df[column] = pd.to_numeric(raw[column])   # raises on any leftover text
    return normalise_categoricals(df)

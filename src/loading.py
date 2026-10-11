"""Read welddb.data and derive the study group of each weld."""
import re
from pathlib import Path

import pandas as pd

from .config import (
    COLUMNS, DATA_PATH, SOURCE_ALIASES, SOURCE_COLUMN, SOURCE_PAGE_MARKER_PATTERN,
    SOURCE_PREFIX_RULES,
)


def derive_source(weld_id: str) -> str:
    """Study group of a weld id, following the explicit rules of config.py."""
    tokens = weld_id.split("-")
    label = tokens[0]
    for prefix, group in SOURCE_PREFIX_RULES.items():
        if label.startswith(prefix):
            label = group
    if re.fullmatch(SOURCE_PAGE_MARKER_PATTERN, label) and len(tokens) > 1:
        label = tokens[1]
    return SOURCE_ALIASES.get(label, label)


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Raw frame: every value kept as text, 'N' (not reported) as NaN, plus `source`.

    Values are not interpreted here; cleaning.py parses them.
    """
    df = pd.read_csv(
        path, sep=r"\s+", header=None, names=COLUMNS,
        dtype=str, na_values=["N"], keep_default_na=False,
    )
    df[SOURCE_COLUMN] = df["weld_id"].map(derive_source)
    return df

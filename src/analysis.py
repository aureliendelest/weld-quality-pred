"""Exploration helpers that are reused across notebooks."""
import numpy as np
import pandas as pd


def source_demeaned_correlations(numeric: pd.DataFrame, source: pd.Series, inputs, targets,
                                 min_rows: int = 30, min_source_rows: int = 5,
                                 method: str = "spearman") -> pd.DataFrame:
    """Overall vs within-source correlation of every input/target pair.

    The within-source value subtracts, for each study, the mean of both variables
    (only studies with at least `min_source_rows` rows of the pair are kept), so
    differences *between* studies cannot create or hide a relationship. A large
    drop from `rho_overall` to `rho_within` means the overall coefficient mostly
    reflects who measured what.
    """
    rows = []
    for x in inputs:
        for y in targets:
            pair = pd.DataFrame({"x": numeric[x], "y": numeric[y], "s": source}).dropna()
            if len(pair) < min_rows:
                continue
            rho_overall = pair[["x", "y"]].corr(method=method).iloc[0, 1]
            kept = pair[pair.groupby("s")["x"].transform("count") >= min_source_rows].copy()
            for column in ("x", "y"):
                kept[column] = kept[column] - kept.groupby("s")[column].transform("mean")
            if len(kept) < min_rows or kept["x"].abs().sum() == 0 or kept["y"].abs().sum() == 0:
                continue
            rows.append({
                "input": x, "target": y, "n": len(pair), "n_within": len(kept),
                "n_sources_within": kept["s"].nunique(), "rho_overall": rho_overall,
                "rho_within": kept[["x", "y"]].corr(method=method).iloc[0, 1],
            })
    result = pd.DataFrame(rows)
    if len(result):
        result["ratio"] = result["rho_within"].abs() / result["rho_overall"].abs().replace(0, np.nan)
    return result

"""Exploration helpers that are reused across notebooks."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import SKEWED_COLUMNS
from .preprocessing import LogShift


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


def eta_squared(values, groups, min_group_rows: int = 5, adjusted: bool = True) -> float:
    """Share of the variance of `values` explained by the group means.

    Groups with fewer than `min_group_rows` rows are ignored. The adjusted value
    corrects the upward bias that many small groups create (it can be slightly
    negative when the groups explain nothing).
    """
    data = pd.DataFrame({"v": np.asarray(values, dtype=float), "g": np.asarray(groups)}).dropna()
    data = data[data.groupby("g")["v"].transform("size") >= min_group_rows]
    n, k = len(data), data["g"].nunique()
    if k < 2 or n <= k:
        return np.nan
    grand = data["v"].mean()
    stats = data.groupby("g")["v"].agg(["size", "mean"])
    total = ((data["v"] - grand) ** 2).sum()
    if total == 0:
        return np.nan
    eta = (stats["size"] * (stats["mean"] - grand) ** 2).sum() / total
    return float(1 - (1 - eta) * (n - 1) / (n - k)) if adjusted else float(eta)


def parallel_analysis(X, n_iter: int = 200, percentile: float = 95, seed: int = 0):
    """Horn's parallel analysis: keep the leading components whose eigenvalue beats
    the eigenvalues of the same data with every column shuffled independently.

    Returns (table with eigenvalue / threshold / keep, number of components kept).
    """
    values = np.asarray(X, dtype=float)
    n_rows, n_cols = values.shape
    observed = np.sort(np.linalg.eigvalsh(np.corrcoef(values, rowvar=False)))[::-1]
    rng = np.random.default_rng(seed)
    simulated = np.empty((n_iter, n_cols))
    for i in range(n_iter):
        shuffled = np.column_stack([rng.permutation(values[:, j]) for j in range(n_cols)])
        simulated[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(shuffled, rowvar=False)))[::-1]
    threshold = np.percentile(simulated, percentile, axis=0)
    beats = observed > threshold
    n_keep = n_cols if beats.all() else int(np.argmin(beats))
    table = pd.DataFrame({"component": np.arange(1, n_cols + 1), "eigenvalue": observed,
                          "threshold": threshold, "keep": np.arange(n_cols) < n_keep})
    return table, n_keep


def standardise_for_pca(frame: pd.DataFrame, log_columns=SKEWED_COLUMNS,
                        impute_median: bool = False) -> pd.DataFrame:
    """Shifted log on the skewed impurities, optional median fill, then z-scores.

    Constant columns are dropped (they have no variance to explain). Without
    `impute_median`, the caller must pass rows that are already complete.
    """
    X = frame.astype(float).copy()
    skewed = [c for c in log_columns if c in X.columns]
    if skewed:
        X[skewed] = LogShift().fit_transform(X[skewed])
    if impute_median:
        X = X.fillna(X.median())
    if X.isna().any().any():
        raise ValueError("NaN left: pass complete rows or impute_median=True")
    X = X.loc[:, X.std() > 0]
    return (X - X.mean()) / X.std()


@dataclass(frozen=True)
class PCAReport:
    scores: pd.DataFrame           # rows x components
    correlations: pd.DataFrame     # variables x components: corr(variable, component)
    explained_ratio: pd.Series
    parallel: pd.DataFrame
    n_keep: int

    def components_for(self, share: float) -> int:
        return int(np.searchsorted(self.explained_ratio.cumsum().to_numpy(), share) + 1)


def pca_report(X_std: pd.DataFrame, n_iter: int = 200, seed: int = 0) -> PCAReport:
    """PCA of standardised data with variable-component correlations (circle of correlations)."""
    pca = PCA().fit(X_std)
    names = [f"PC{i + 1}" for i in range(pca.n_components_)]
    scores = pd.DataFrame(pca.transform(X_std), index=X_std.index, columns=names)
    correlations = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_),
                                index=X_std.columns, columns=names)
    parallel, n_keep = parallel_analysis(X_std, n_iter=n_iter, seed=seed)
    return PCAReport(scores, correlations,
                     pd.Series(pca.explained_variance_ratio_, index=names), parallel, n_keep)

"""Classical regressors compared on the same folds, and the paired comparison with a reference.

Every model is tuned by a small grid with cross-validation over studies inside the training fold
(`GroupTunedSearch`), so a held-out study never takes part in a choice. Option 1 of the plan: tune
and judge in leave-one-study-out, then check the best models on the grouped protocol (5 seeds).
Decision: same rule as the ablations (doc/04_ablation_plan.md), paired by held-out study; a tie keeps
the reference.
"""
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from .ablation import VARIANTS, paired_decision
from .baselines import baseline_models
from .config import ABLATION_SEEDS, EXCLUDED_FROM_JUDGMENT, TARGETS, TENSILE_TARGETS
from .evaluation import ModelSpec, evaluate_model, make_folds
from .models import GroupTunedCatBoost, GroupTunedRidge, GroupTunedSearch

REFERENCES = ("random forest", "ridge (study-tuned)")
BOOSTING = ("hist gradient boosting", "xgboost", "catboost")      # always checked on the grouped protocol
DECISION_REFERENCES = (*REFERENCES, "random forest (tuned)")     # the tuned forest shows what tuning alone buys

# name -> (family, estimator, grid). Targets are standardised, so the grids are in standard-deviation units.
_ZOO = {
    "lasso": ("linear", lambda: Lasso(max_iter=10000), {"alpha": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}),
    "elasticnet": ("linear", lambda: ElasticNet(max_iter=10000),
                   {"alpha": [0.01, 0.03, 0.1, 0.3], "l1_ratio": [0.2, 0.5, 0.8]}),
    "knn": ("linear", KNeighborsRegressor,
            {"n_neighbors": [3, 5, 10, 20, 40], "weights": ["uniform", "distance"]}),
    "svr rbf": ("linear", SVR, {"C": [1.0, 10.0, 100.0], "gamma": [0.005, 0.015, 0.05]}),
    # The reference forest is not tuned: this one shows how much of a candidate's gain is only tuning.
    "random forest (tuned)": ("tree", lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0),
                              {"max_features": [0.33, 0.6, 1.0], "min_samples_leaf": [1, 5]}),
    "extra trees": ("tree", lambda: ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=0),
                    {"max_features": [0.33, 0.6, 1.0], "min_samples_leaf": [1, 5]}),
    "hist gradient boosting": ("tree", lambda: HistGradientBoostingRegressor(max_iter=150, random_state=0),
                               {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [6, 15], "min_samples_leaf": [10, 40]}),
    # Same protocol as the histogram boosting (8 configurations, number of rounds fixed beforehand, NaN kept), with each
    # library's own regularisation knobs: XGBoost limits the depth and the hessian mass of a leaf and subsamples rows and
    # columns; CatBoost grows symmetric trees (weaker per round, hence 300 rounds) and is tuned on depth and L2.
    "xgboost": ("tree", lambda: XGBRegressor(n_estimators=150, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=0),
                {"learning_rate": [0.03, 0.1], "max_depth": [3, 5], "min_child_weight": [5, 20]}),
    "catboost": ("tree", lambda: CatBoostRegressor(iterations=300, verbose=0, random_seed=0, thread_count=-1, allow_writing_files=False),
                 {"learning_rate": [0.03, 0.1], "depth": [4, 6], "l2_leaf_reg": [3, 10]}),
}


def reference_models() -> list:
    """The two models every candidate has to beat, then the two trivial predictors."""
    return [
        ModelSpec("random forest", "tree", lambda: RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)),
        ModelSpec("ridge (study-tuned)", "linear", GroupTunedRidge, uses_source=True),
        *baseline_models(),
    ]


def zoo_models(with_drop_arc: bool = True) -> list:
    """The candidates; the linear-family ones also run without current, voltage and heat input."""
    drop_arc = VARIANTS["drop_arc"][0]
    models = []
    for name, (family, estimator, grid) in _ZOO.items():
        def factory(estimator=estimator, grid=grid):
            return GroupTunedSearch(estimator(), grid)
        models.append(ModelSpec(name, family, factory, uses_source=True))
        if with_drop_arc and family == "linear":
            models.append(ModelSpec(f"{name} + drop_arc", family, factory, uses_source=True, options=drop_arc))
    return models


def run_zoo(df: pd.DataFrame, models: list, protocols=("loso",), seeds=(0,), targets=TARGETS, log=None) -> pd.DataFrame:
    """Fold-level results of every model on identical folds, one block per seed (loso folds do not depend on it)."""
    frames = []
    for spec in models:
        for seed in seeds:
            folds = {p: make_folds(df, p, seed=seed) for p in protocols}
            frames.append(evaluate_model(df, spec, folds, targets).assign(seed=seed))
        if log:
            log(spec.name)
    return pd.concat(frames, ignore_index=True)


def zoo_decisions(results: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Every other model against `reference`, paired by (target, study) on the loso rows (rule of the ablations)."""
    loso = results[results["protocol"] == "loso"]
    labelled = loso.assign(algo="zoo", variant=loso["model"].where(loso["model"] != reference, "baseline"))
    candidates = [m for m in loso["model"].unique() if m != reference]
    table = pd.concat([paired_decision(labelled, "zoo", m) for m in candidates], ignore_index=True)
    return table.drop(columns="algo").rename(columns={"variant": "model"}).assign(reference=reference)


def pooled_loso_skill(results: pd.DataFrame, targets=TENSILE_TARGETS) -> pd.Series:
    """Mean over (target, study) of the leave-one-study-out skill of each model, tensile targets pooled.

    The pairs of EXCLUDED_FROM_JUDGMENT are left out, as in the paired decision.
    """
    loso = results[(results["protocol"] == "loso") & results["target"].isin(targets)]
    loso = loso[[(s, t) not in EXCLUDED_FROM_JUDGMENT for s, t in zip(loso["study"], loso["target"])]]
    return loso.groupby("model")["skill"].mean().sort_values(ascending=False)


def retained(results: pd.DataFrame, k: int = 3) -> list:
    """The two references and the k candidates with the best pooled leave-one-study-out skill."""
    ranking = pooled_loso_skill(results)
    candidates = [m for m in ranking.index if m not in REFERENCES and m not in ("global_mean", "study_mean")]
    return [*REFERENCES, *candidates[:k]]


def grouped_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation, over the seeds, of the fold-mean grouped skill, per model and target."""
    grouped = results[results["protocol"] == "grouped"]
    per_seed = grouped.groupby(["model", "target", "seed"])["skill"].mean()
    return per_seed.groupby(["model", "target"]).agg(skill_mean="mean", skill_std="std").reset_index()


def run_option_one(df: pd.DataFrame, out_dir, rerun: bool = False, k: int = 3, also_grouped=BOOSTING, log=None):
    """Option 1 of the plan: leave-one-study-out for every model, then `grouped` (5 seeds) for the k best candidates
    and the references, plus `also_grouped`.

    Tables already saved in `out_dir` are reused (the folds are deterministic), so only the models missing from them
    are computed; `rerun=True` recomputes everything. Returns the loso and the grouped fold-level results.
    """
    models = {m.name: m for m in reference_models() + zoo_models()}
    loso_file, grouped_file = out_dir / "03_zoo_loso_folds.csv", out_dir / "03_zoo_grouped_folds.csv"

    def stage(file, names, **protocol):
        saved = pd.read_csv(file) if file.exists() and not rerun else pd.DataFrame(columns=["model"])
        todo = [n for n in names if n not in set(saved["model"])]
        fresh = [run_zoo(df, [models[n] for n in todo], log=log, **protocol)] if todo else []
        table = pd.concat([saved, *fresh], ignore_index=True)
        if fresh:
            table.to_csv(file, index=False)
        return table

    loso = stage(loso_file, list(models), protocols=("loso",))
    grouped = stage(grouped_file, list(dict.fromkeys([*retained(loso, k), *also_grouped])), protocols=("grouped",), seeds=ABLATION_SEEDS)
    return loso, grouped


TUNED_ROUNDS = "catboost (rounds tuned)"
WIDE_ROUNDS = "catboost (rounds tuned, wide grid)"
# name -> (maximum number of rounds, grid, file tag). The first search put the rounds on the upper edge of its grid
# and the depth and the L2 penalty on the lower one; the second widens it in that direction.
_ROUNDS_SEARCHES = {
    TUNED_ROUNDS: (600, {"depth": [4, 6], "l2_leaf_reg": [3, 10]}, "catboost_rounds"),
    WIDE_ROUNDS: (1500, {"depth": [2, 3, 4], "l2_leaf_reg": [1, 3]}, "catboost_rounds_wide"),
}


def tuned_rounds_model(name: str = TUNED_ROUNDS) -> ModelSpec:
    """CatBoost with the depth, the L2 penalty and the number of rounds chosen by cross-validation over studies.

    The learning rate is fixed at 0.05 and the rounds are read off the validation curve up to the search's maximum.
    """
    max_rounds, grid, _ = _ROUNDS_SEARCHES[name]
    return ModelSpec(name, "tree", lambda: GroupTunedCatBoost(grid, max_rounds=max_rounds), uses_source=True)


def run_tuned_rounds(df: pd.DataFrame, out_dir, name: str = TUNED_ROUNDS, rerun: bool = False, log=None):
    """Leave-one-study-out, then `grouped` (5 seeds), for a CatBoost with tuned rounds; saved tables are reused."""
    spec, tag = tuned_rounds_model(name), _ROUNDS_SEARCHES[name][2]
    tables = []
    for protocol, seeds in (("loso", (0,)), ("grouped", ABLATION_SEEDS)):
        file = out_dir / f"03_{tag}_{protocol}_folds.csv"
        if file.exists() and not rerun:
            tables.append(pd.read_csv(file))
            continue
        table = run_zoo(df, [spec], protocols=(protocol,), seeds=seeds, log=log)
        table.to_csv(file, index=False)
        tables.append(table)
    return tuple(tables)

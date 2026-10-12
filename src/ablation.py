"""Ablations of the feature set on two reference models, and the paired decision rule.

One change at a time against `baseline`, same folds, same rule (doc/04_ablation_plan.md).
Decision: paired by held-out study, a variant "helps" when its mean skill gain is at least
MIN_SKILL_GAIN and it wins on at least MIN_WIN_SHARE of the (target, study) pairs.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import (
    ABLATION_SEEDS, EXCLUDED_FROM_JUDGMENT, MIN_SKILL_GAIN, MIN_WIN_SHARE, TARGETS, TENSILE_TARGETS,
)
from .evaluation import ModelSpec, evaluate_model, make_folds
from .models import GroupTunedRidge
from .preprocessing import FeatureOptions

# name -> (options, study_weights, algorithms it applies to; None = both)
VARIANTS = {
    "baseline": (FeatureOptions(), False, None),
    "no_missing_flags": (FeatureOptions(missing_flags=False), False, None),
    "no_censored_flags": (FeatureOptions(censored_flags=False), False, None),
    "no_flags": (FeatureOptions(missing_flags=False, censored_flags=False), False, None),
    "drop_mo": (FeatureOptions(drop=("mo_conc_wt",)), False, None),
    "drop_arc": (FeatureOptions(drop=("current_a", "voltage_v", "heat_input_kj_mm")), False, None),
    "study_weights": (FeatureOptions(), True, None),
    "pca6": (FeatureOptions(pca_components=6), False, None),
    "pca_replace": (FeatureOptions(numeric_mode="pca"), False, None),
    "pca_replace_within": (FeatureOptions(numeric_mode="pca_within"), False, None),
    "center_within": (FeatureOptions(numeric_mode="within"), False, None),
    "add_ce": (FeatureOptions(extra=("ce_iiw",)), False, None),
    "add_larson_miller": (FeatureOptions(extra=("larson_miller",)), False, None),
    "log_al_nb_v": (FeatureOptions(skewed_extra=("al_conc_ppm", "nb_conc_ppm", "v_conc_wt")), False, ("ridge",)),
}

ALGORITHMS = {
    "random forest": ("tree", lambda: RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1), False),
    "ridge": ("linear", GroupTunedRidge, True),
}


def ablation_models(variants=None, algorithms=None) -> list:
    """(algorithm, variant, ModelSpec) for every applicable combination."""
    chosen = list(VARIANTS) if variants is None else list(variants)
    wanted = list(ALGORITHMS) if algorithms is None else list(algorithms)
    out = []
    for algo in wanted:
        family, factory, uses_source = ALGORITHMS[algo]
        for variant in chosen:
            options, weights, only = VARIANTS[variant]
            if only is not None and algo not in only:
                continue
            out.append((algo, variant, ModelSpec(f"{algo} | {variant}", family, factory,
                                                 uses_source=uses_source, options=options,
                                                 study_weights=weights)))
    return out


def run_ablation(df: pd.DataFrame, variants=None, algorithms=None, targets=TARGETS,
                 seeds=ABLATION_SEEDS, log=None) -> pd.DataFrame:
    """Fold-level results: `grouped` on every seed, `loso` and `random` on the first seed."""
    protocols = [(seed, ("grouped",) + (("loso", "random") if i == 0 else ())) for i, seed in enumerate(seeds)]
    frames = []
    for algo, variant, spec in ablation_models(variants, algorithms):
        for seed, names in protocols:
            folds = {p: make_folds(df, p, seed=seed) for p in names}
            part = evaluate_model(df, spec, folds, targets)
            frames.append(part.assign(algo=algo, variant=variant, seed=seed))
        if log:
            log(f"{algo} | {variant}")
    return pd.concat(frames, ignore_index=True)


def scope_of(target: str) -> str:
    return "tensile" if target in TENSILE_TARGETS else target


def paired_decision(results: pd.DataFrame, algo: str, variant: str,
                    gain: float = MIN_SKILL_GAIN, share: float = MIN_WIN_SHARE) -> pd.DataFrame:
    """Variant against baseline, paired by (target, study) on the loso rows.

    One row per scope: the four tensile targets pooled, then each other target (Charpy).
    Verdict: helps / hurts / tie. Pairs of EXCLUDED_FROM_JUDGMENT are left out.
    """
    loso = results[(results["protocol"] == "loso") & (results["algo"] == algo)]
    keys = ["target", "study"]
    paired = loso[loso["variant"] == "baseline"][keys + ["skill", "within_rho"]].merge(
        loso[loso["variant"] == variant][keys + ["skill", "within_rho"]], on=keys, suffixes=("_base", "_var"))
    keep = [(s, t) not in EXCLUDED_FROM_JUDGMENT for s, t in zip(paired["study"], paired["target"])]
    paired = paired[keep].assign(delta=lambda t: t["skill_var"] - t["skill_base"],
                                 delta_rho=lambda t: t["within_rho_var"] - t["within_rho_base"],
                                 scope=lambda t: t["target"].map(scope_of))
    rows = []
    for scope, part in paired.groupby("scope", sort=False):
        mean_gain, wins, losses = part["delta"].mean(), (part["delta"] > 0).mean(), (part["delta"] < 0).mean()
        verdict = "helps" if mean_gain >= gain and wins >= share else \
                  "hurts" if mean_gain <= -gain and losses >= share else "tie"
        rows.append({"algo": algo, "variant": variant, "scope": scope, "n_pairs": len(part),
                     "mean_skill_gain": mean_gain, "median_skill_gain": part["delta"].median(),
                     "win_share": wins, "loss_share": losses, "mean_rho_gain": part["delta_rho"].mean(),
                     "verdict": verdict})
    return pd.DataFrame(rows)


def protocol_gain(results: pd.DataFrame, protocol: str, algo: str, variant: str) -> pd.DataFrame:
    """Mean skill gain against baseline under `protocol`, per scope (context, never a ranking)."""
    part = results[(results["protocol"] == protocol) & (results["algo"] == algo)].copy()
    part["scope"] = part["target"].map(scope_of)
    mean = part.groupby(["variant", "scope", "target"])["skill"].mean().groupby(["variant", "scope"]).mean()
    gain = (mean.loc[variant] - mean.loc["baseline"]).rename(f"gain_{protocol}")
    return gain.reset_index()


def decision_table(results: pd.DataFrame) -> pd.DataFrame:
    """All variants of all algorithms: the paired decision plus the grouped and random gains."""
    rows = []
    for algo in results["algo"].unique():
        for variant in results[results["algo"] == algo]["variant"].unique():
            if variant == "baseline":
                continue
            table = paired_decision(results, algo, variant)
            for protocol in ("grouped", "random"):
                table = table.merge(protocol_gain(results, protocol, algo, variant), on="scope", how="left")
            rows.append(table)
    return pd.concat(rows, ignore_index=True)


def loso_deltas(results: pd.DataFrame, algo: str, variant: str, metric: str = "skill") -> pd.DataFrame:
    """Variant minus baseline on each held-out study (rows) and target (columns), loso rows."""
    loso = results[(results["protocol"] == "loso") & (results["algo"] == algo)]
    keys = ["study", "target"]
    base = loso[loso["variant"] == "baseline"].set_index(keys)[metric]
    other = loso[loso["variant"] == variant].set_index(keys)[metric]
    return (other - base).unstack("target")[[t for t in TARGETS if t in set(loso["target"])]]


def grouped_seed_spread(results: pd.DataFrame, variant: str = "baseline") -> pd.DataFrame:
    """Mean and standard deviation, over the seeds of the grouped protocol, of the fold-mean skill.

    The spread is the noise floor: a gain smaller than it cannot be told from a different draw of folds.
    """
    part = results[(results["protocol"] == "grouped") & (results["variant"] == variant)]
    per_seed = part.groupby(["algo", "target", "seed"])["skill"].mean()
    return per_seed.groupby(["algo", "target"]).agg(["mean", "std"]).rename(columns={"mean": "skill_mean", "std": "skill_std"})

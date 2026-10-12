import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    clean_dataframe, parse_censored, parse_hardness, parse_nitrogen, parse_range,
)
from src.config import CENSORED_COLUMNS, NUMERIC_COLUMNS, TARGETS
from src.evaluation import make_folds
from src.loading import derive_source, load_raw
from src.preprocessing import feature_spec


def text(values):
    return pd.Series(values, dtype="str")


# ---- parsers ----------------------------------------------------------------
def test_censored_values_become_half_the_threshold_with_a_flag():
    values, flag = parse_censored(text(["<0.01", "0.5", None]), "mo_conc_wt")
    assert values.iloc[0] == pytest.approx(0.005) and values.iloc[1] == 0.5 and np.isnan(values.iloc[2])
    assert flag.tolist() == [True, False, False]


def test_unit_errors_are_converted_before_halving():
    v, _ = parse_censored(text(["<5"]), "v_conc_wt")          # 5 ppm in a wt% column
    ti, _ = parse_censored(text(["<0.01"]), "ti_conc_ppm")    # 0.01 wt% in a ppm column
    assert v.iloc[0] == pytest.approx(0.00025) and ti.iloc[0] == pytest.approx(50)


def test_unknown_notation_raises_instead_of_becoming_nan():
    with pytest.raises(ValueError):
        parse_censored(text(["about 3"]), "s_conc_wt")


def test_range_nitrogen_and_hardness():
    assert parse_range(text(["150-200", "200", None]), "interpass_temp_c").iloc[:2].tolist() == [175, 200]
    assert parse_nitrogen(text(["67tot33res", "66totndres", "83"])).tolist() == [67, 66, 83]
    value, scale = parse_hardness(text(["158(Hv30)", "459Hv10", "203", None]))
    assert value.iloc[:3].tolist() == [158, 459, 203]
    assert scale.iloc[:3].tolist() == ["Hv30", "Hv10", "unknown"] and pd.isna(scale.iloc[3])


@pytest.mark.parametrize("weld_id, group", [
    ("Evans-Al/CMn-1990-490aw", "Evans"),
    ("EvansLetterC+1.5Crch1", "EvansLetter"),
    ("p9-RR82011", "RR82011"),
    ("Ga&K-1975II-4aw", "Gar&K"),
    ("RC81033", "RC81033"),
])
def test_source_grouping_rules(weld_id, group):
    assert derive_source(weld_id) == group


# ---- real data --------------------------------------------------------------
@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def clean(raw):
    return clean_dataframe(raw)


def test_every_numeric_column_is_numeric_and_no_value_is_lost(raw, clean):
    for column in NUMERIC_COLUMNS:
        assert pd.api.types.is_numeric_dtype(clean[column]), column
        assert clean[column].isna().sum() == raw[column].isna().sum(), column
    assert clean["n_conc_ppm"].notna().sum() == 1242
    assert clean["hardness_kgf_mm2"].notna().sum() == 138


def test_censored_flags_match_raw_counts(raw, clean):
    for column in CENSORED_COLUMNS:
        assert clean[f"{column}_censored"].sum() == raw[column].str.startswith("<", na=False).sum(), column
    assert clean["b_conc_ppm_censored"].sum() == 419


def test_source_groups_are_merged(clean):
    groups = clean["source"].unique()
    assert len(groups) == 33
    assert not any(g.startswith("EvansLetter") and g != "EvansLetter" for g in groups)
    assert not any(g.startswith("p") and g[1:].isdigit() for g in groups)


def test_weld_type_levels(clean):
    assert set(clean["weld_type"]) == {"MMA", "SA", "FCA", "TSA", "other"}
    assert (clean["weld_type"] == "MMA").sum() == 1180


# ---- leakage guards -----------------------------------------------------------
def test_charpy_temperature_is_a_feature_of_toughness_only(clean):
    for target in TARGETS:
        columns = feature_spec(clean, target).columns
        assert ("charpy_temp_c" in columns) == (target == "charpy_toughness_j")
        assert not set(TARGETS) & set(columns)
        assert "source" not in columns


def test_grouped_folds_never_share_a_source(clean):
    for train, test in make_folds(clean, "grouped"):
        assert not set(clean["source"].iloc[train]) & set(clean["source"].iloc[test])


def test_folds_are_identical_across_calls(clean):
    first, second = make_folds(clean, "random"), make_folds(clean, "random")
    assert all((a[1] == b[1]).all() for a, b in zip(first, second))


# ---- analysis ---------------------------------------------------------------
def test_within_source_correlation_removes_between_source_effects():
    from src.analysis import source_demeaned_correlations
    rng = np.random.default_rng(0)
    frames = []
    for offset in (0, 10, 20):          # three studies, higher x AND higher y each time
        x = rng.normal(offset, 1, 40)
        frames.append(pd.DataFrame({"x": x, "y": offset * 3 - (x - offset) + rng.normal(0, 0.1, 40),
                                    "s": f"study{offset}"}))
    data = pd.concat(frames, ignore_index=True)
    result = source_demeaned_correlations(data[["x", "y"]], data["s"], ["x"], ["y"]).iloc[0]
    assert result["rho_overall"] > 0.5        # between-study offsets
    assert result["rho_within"] < -0.8        # the real within-study relation


def test_eta_squared_detects_group_structure_and_ignores_noise():
    from src.analysis import eta_squared
    rng = np.random.default_rng(0)
    groups = np.repeat(["a", "b", "c", "d"], 50)
    structured = rng.normal(0, 1, 200) + np.repeat([0, 3, 6, 9], 50)
    noise = rng.normal(0, 1, 200)
    assert eta_squared(structured, groups) > 0.8
    assert abs(eta_squared(noise, groups)) < 0.05


def test_parallel_analysis_recovers_the_number_of_latent_factors():
    from src.analysis import parallel_analysis, pca_report, standardise_for_pca
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(400, 2))
    loadings = rng.normal(size=(2, 8))
    data = pd.DataFrame(latent @ loadings + rng.normal(0, 0.3, (400, 8)),
                        columns=[f"x{i}" for i in range(8)])
    _, n_keep = parallel_analysis(standardise_for_pca(data, log_columns=[]))
    assert n_keep == 2
    report = pca_report(standardise_for_pca(data, log_columns=[]))
    assert report.explained_ratio.iloc[:2].sum() > 0.8
    assert report.components_for(0.9) <= 3


def test_censoring_factor_is_a_parameter():
    values, flag = parse_censored(text(["<0.01", "0.5"]), "mo_conc_wt", factor=1.0)
    assert values.iloc[0] == pytest.approx(0.01)
    values, flag = parse_censored(text(["<0.01", "0.5"]), "mo_conc_wt", factor=None)
    assert np.isnan(values.iloc[0]) and flag.iloc[0] and values.iloc[1] == 0.5


# ---- baselines and per-study evaluation -------------------------------------
@pytest.fixture(scope="module")
def baseline_results(clean):
    from src.baselines import baseline_models
    from src.evaluation import ALL_PROTOCOLS, compare_models
    return compare_models(clean, baseline_models(), targets=["yield_strength_mpa"], protocols=ALL_PROTOCOLS)


def test_study_mean_regressor_uses_the_study_then_falls_back():
    from src.baselines import StudyMeanRegressor
    model = StudyMeanRegressor().fit(None, [1.0, 3.0, 10.0], groups=["a", "a", "b"])
    assert model.predict(None, groups=["a", "b", "unseen"]).tolist() == [2.0, 10.0, pytest.approx(14 / 3)]
    with pytest.raises(ValueError):
        model.predict(None)


def test_global_mean_has_zero_skill_by_construction(baseline_results):
    rows = baseline_results[baseline_results["model"] == "global_mean"]
    assert rows["skill"].abs().max() < 1e-9 and len(rows) > 10


def test_study_mean_only_helps_when_the_study_was_seen(baseline_results):
    study_mean = baseline_results[baseline_results["model"] == "study_mean"]
    by_protocol = study_mean.groupby("protocol")["skill"].mean()
    assert by_protocol["random"] > 0.3                      # same study in training
    assert abs(by_protocol["grouped"]) < 1e-9 and abs(by_protocol["loso"]) < 1e-9


def test_leave_one_study_out_folds_hold_out_one_study_each(clean):
    from src.evaluation import make_folds
    from src.config import LOSO_MIN_ROWS
    folds = make_folds(clean, "loso")
    sizes = clean["source"].value_counts()
    assert len(folds) == int((sizes >= LOSO_MIN_ROWS).sum())
    for train, test in folds:
        assert clean["source"].iloc[test].nunique() == 1
        assert not set(clean["source"].iloc[train]) & set(clean["source"].iloc[test])


def test_weld_folds_keep_a_weld_whole_but_not_a_study(clean):
    from src.evaluation import make_folds, weld_fingerprint
    fingerprint = weld_fingerprint(clean)
    assert fingerprint.nunique() < len(clean)               # replicated inputs exist
    folds = make_folds(clean, "grouped_weld")
    shared_study = False
    for train, test in folds:
        assert not set(fingerprint.iloc[train]) & set(fingerprint.iloc[test])
        shared_study |= bool(set(clean["source"].iloc[train]) & set(clean["source"].iloc[test]))
    assert shared_study                                      # unlike "grouped", studies are split
    again = make_folds(clean, "grouped_weld")
    assert all((a[1] == b[1]).all() for a, b in zip(folds, again))


def test_per_study_scores_report_one_row_per_study_and_model(baseline_results):
    from src.evaluation import per_study_scores
    table = per_study_scores(baseline_results)
    assert {"study", "skill", "bias", "within_rho"} <= set(table.columns)
    assert table.groupby("model")["study"].nunique().nunique() == 1
    assert table["study"].notna().all()


# ---- linear family: bounded features, study-tuned ridge ----------------------
def test_linear_preprocessor_bounds_the_scaled_features(clean):
    from src.config import SCALED_CLIP
    from src.preprocessing import build_preprocessor
    spec = feature_spec(clean, "yield_strength_mpa")
    bounded = build_preprocessor("linear", spec).fit_transform(clean[spec.columns])
    assert not bounded.isna().any().any() and bounded.abs().max().max() <= SCALED_CLIP + 1e-9
    unbounded = build_preprocessor("linear", spec, clip=None).fit_transform(clean[spec.columns])
    assert unbounded.abs().max().max() > 50           # the defect the clip removes


def test_group_tuned_ridge_picks_a_penalty_from_its_grid():
    from src.models import GroupTunedRidge
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    y = X["a"] * 3 + rng.normal(0, 0.5, 120)
    groups = np.repeat(["s1", "s2", "s3", "s4", "s5", "s6"], 20)
    model = GroupTunedRidge().fit(X, y, groups=groups)
    assert model.alpha_ in model.alphas
    assert model.predict(X, groups=groups).shape == (120,) and np.corrcoef(model.predict(X), y)[0, 1] > 0.9


# ---- ablation options and decision rule --------------------------------------
def test_feature_options_change_the_feature_set(clean):
    from src.preprocessing import FeatureOptions, add_derived
    target = "yield_strength_mpa"
    base = feature_spec(clean, target)
    no_mo = feature_spec(clean, target, FeatureOptions(drop=("mo_conc_wt",)))
    assert "mo_conc_wt" not in no_mo.numeric and "mo_conc_wt_censored" not in no_mo.censored_flags
    assert len(no_mo.numeric) == len(base.numeric) - 1
    assert feature_spec(clean, target, FeatureOptions(censored_flags=False)).censored_flags == []
    with pytest.raises(ValueError):
        feature_spec(clean, target, FeatureOptions(drop=("yield_strength_mpa",)))
    frame = add_derived(clean, ("ce_iiw", "larson_miller"))
    assert "ce_iiw" in feature_spec(frame, target, FeatureOptions(extra=("ce_iiw",))).numeric
    assert (frame.loc[clean["pwht_temp_c"] == 0, "larson_miller"] == 0).all()
    assert frame["ce_iiw"].notna().sum() == clean[["c_conc_wt", "mn_conc_wt", "cr_conc_wt", "mo_conc_wt",
                                                   "v_conc_wt", "ni_conc_wt", "cu_conc_wt"]].dropna().shape[0]


def test_option_preprocessors_fit_on_the_training_fold(clean):
    from src.preprocessing import FeatureOptions, build_preprocessor
    target = "yield_strength_mpa"
    spec = feature_spec(clean, target)
    X = clean[spec.columns]
    no_flags = build_preprocessor("tree", spec, options=FeatureOptions(missing_flags=False)).fit(X)
    assert not any(c.endswith("_missing") for c in no_flags.transform(X).columns)
    with_pca = build_preprocessor("tree", spec, options=FeatureOptions(pca_components=3)).fit(X.iloc[:800])
    out = with_pca.transform(X.iloc[800:])
    assert [c for c in out.columns if c.startswith("pca_")] == ["pca_1", "pca_2", "pca_3"]
    assert not out[["pca_1", "pca_2", "pca_3"]].isna().any().any() and len(out) == len(X) - 800


def test_study_weights_have_mean_one_and_favour_small_studies():
    from src.evaluation import study_sample_weights
    weights = study_sample_weights(["big"] * 90 + ["small"] * 10)
    assert weights.mean() == pytest.approx(1.0) and weights[-1] > weights[0]


def test_paired_decision_needs_gain_and_wins():
    from src.ablation import paired_decision
    studies = [f"s{i}" for i in range(6)]

    def loso(variant, skills):
        return pd.DataFrame({"protocol": "loso", "algo": "a", "variant": variant, "target": "uts_mpa",
                             "study": studies, "skill": skills, "within_rho": 0.0})
    base = [0.1] * 6
    helps = pd.concat([loso("baseline", base), loso("v", [0.3] * 6)])
    assert paired_decision(helps, "a", "v")["verdict"].iloc[0] == "helps"
    one_big_win = pd.concat([loso("baseline", base), loso("v", [1.1, 0.1, 0.1, 0.1, 0.1, 0.0])])
    assert paired_decision(one_big_win, "a", "v")["verdict"].iloc[0] == "tie"      # mean gain, but 1 win of 6
    hurts = pd.concat([loso("baseline", base), loso("v", [-0.2] * 6)])
    assert paired_decision(hurts, "a", "v")["verdict"].iloc[0] == "hurts"


def test_excluded_pair_is_left_out_of_the_decision():
    from src.ablation import paired_decision
    rows = []
    for variant, skill in (("baseline", -2.0), ("v", 0.0)):
        rows.append({"protocol": "loso", "algo": "a", "variant": variant, "target": "reduction_of_area_pct",
                     "study": "Cunh", "skill": skill, "within_rho": 0.0})
        rows.append({"protocol": "loso", "algo": "a", "variant": variant, "target": "reduction_of_area_pct",
                     "study": "Evans", "skill": 0.3, "within_rho": 0.0})
    assert paired_decision(pd.DataFrame(rows), "a", "v")["n_pairs"].iloc[0] == 1


def test_loso_deltas_and_seed_spread():
    from src.ablation import grouped_seed_spread, loso_deltas
    rows = []
    for variant, skill in (("baseline", 0.1), ("v", 0.4)):
        for study in ("a", "b"):
            rows.append({"protocol": "loso", "algo": "m", "variant": variant, "target": "uts_mpa",
                         "study": study, "skill": skill})
        for seed in (0, 1):
            rows.append({"protocol": "grouped", "algo": "m", "variant": variant, "target": "uts_mpa",
                         "study": None, "skill": skill + 0.1 * seed, "seed": seed})
    results = pd.DataFrame(rows)
    assert loso_deltas(results, "m", "v")["uts_mpa"].tolist() == pytest.approx([0.3, 0.3])
    assert grouped_seed_spread(results).loc[("m", "uts_mpa"), "skill_std"] == pytest.approx(0.1 / np.sqrt(2))

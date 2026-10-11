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

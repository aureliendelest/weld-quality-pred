"""Single source of truth for every preprocessing rule.

Each rule carries a one-line rationale so the assumptions stay auditable.
The reasoning behind them is in doc/01_preprocessing_plan.md.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "welddb.data"

# --------------------------------------------------------------------------
# Raw columns (order of welddb.data, see data/welddb.info)
# --------------------------------------------------------------------------
CHEMISTRY_WT = [
    "c_conc_wt", "si_conc_wt", "mn_conc_wt", "s_conc_wt", "p_conc_wt", "ni_conc_wt",
    "cr_conc_wt", "mo_conc_wt", "v_conc_wt", "cu_conc_wt", "co_conc_wt", "w_conc_wt",
]
CHEMISTRY_PPM = [
    "o_conc_ppm", "ti_conc_ppm", "n_conc_ppm", "al_conc_ppm", "b_conc_ppm",
    "nb_conc_ppm", "sn_conc_ppm", "as_conc_ppm", "sb_conc_ppm",
]
PROCESS_NUMERIC = [
    "current_a", "voltage_v", "heat_input_kj_mm", "interpass_temp_c",
    "pwht_temp_c", "pwht_time_h",
]
CATEGORICAL = ["current_type", "electrode_polarity", "weld_type"]
MECHANICAL = [
    "yield_strength_mpa", "uts_mpa", "elongation_pct", "reduction_of_area_pct",
    "charpy_temp_c", "charpy_toughness_j", "hardness_kgf_mm2", "fatt50_temp_c",
]
MICROSTRUCTURE = [
    "primary_ferrite_pct", "ferrite_second_phase_pct", "acicular_ferrite_pct",
    "martensite_pct", "ferrite_carbide_aggregate_pct",
]

COLUMNS = (
    CHEMISTRY_WT + CHEMISTRY_PPM
    + ["current_a", "voltage_v", "current_type", "electrode_polarity",
       "heat_input_kj_mm", "interpass_temp_c", "weld_type", "pwht_temp_c", "pwht_time_h"]
    + MECHANICAL + MICROSTRUCTURE + ["weld_id"]
)
assert len(COLUMNS) == 44

# Every column that must end up numeric after cleaning.
NUMERIC_COLUMNS = [c for c in COLUMNS if c not in CATEGORICAL + ["weld_id"]]

# --------------------------------------------------------------------------
# Targets and feature sets
# --------------------------------------------------------------------------
TARGETS = [
    "yield_strength_mpa", "uts_mpa", "elongation_pct",
    "reduction_of_area_pct", "charpy_toughness_j",
]

# 92-98% missing: kept in the DataFrame, never used as features.
EXCLUDED_FROM_MODELING = [
    "fatt50_temp_c", "w_conc_wt", "martensite_pct", "ferrite_carbide_aggregate_pct",
    "ferrite_second_phase_pct", "acicular_ferrite_pct", "primary_ferrite_pct", "co_conc_wt",
]

# Never features: measured outcomes, not weld inputs.
OUTCOME_COLUMNS = ["hardness_kgf_mm2", "fatt50_temp_c"] + MICROSTRUCTURE

# Test condition: an input for the toughness model only. It is present iff
# charpy_toughness_j is present, so for any other target it would only reveal
# which study produced the row.
TEST_CONDITION_FEATURES = {"charpy_toughness_j": ["charpy_temp_c"]}

NUMERIC_FEATURES = [
    c for c in CHEMISTRY_WT + CHEMISTRY_PPM + PROCESS_NUMERIC
    if c not in EXCLUDED_FROM_MODELING
]
CATEGORICAL_FEATURES = list(CATEGORICAL)

# --------------------------------------------------------------------------
# Cleaning rules
# --------------------------------------------------------------------------
# '<x' means "below detection limit x". Replaced by x * factor (LOD convention:
# the true value lies in [0, x]); a '<col>_censored' flag keeps the information
# because the threshold depends on the lab, hence on the source.
CENSORED_COLUMNS = [
    "s_conc_wt", "mo_conc_wt", "v_conc_wt", "cu_conc_wt", "co_conc_wt", "w_conc_wt",
    "ti_conc_ppm", "al_conc_ppm", "b_conc_ppm", "nb_conc_ppm", "sn_conc_ppm",
    "as_conc_ppm", "sb_conc_ppm", "primary_ferrite_pct",
]
CENSORED_VALUE_FACTOR = 0.5

# Thresholds written in the wrong unit. The same column already holds the
# right-unit equivalent (v: '<0.0005' wt%; ti/al: '<100' ppm).
UNIT_FIXES = {
    ("v_conc_wt", "<5"): "<0.0005",      # 5 ppm written in a wt% column
    ("ti_conc_ppm", "<0.01"): "<100",    # 0.01 wt% written in a ppm column
    ("al_conc_ppm", "<0.01"): "<100",    # 0.01 wt% written in a ppm column
}

# 'a-b' -> mean(a, b): a process range is centred on its target value.
RANGE_COLUMNS = ["interpass_temp_c"]

# '67tot33res' -> 67. Undocumented; read as total/residual ('nd' = not
# determined). Only the total is kept.
NITROGEN_COLUMN = "n_conc_ppm"

# Hardness: numeric value + scale column. Not a target, not a feature.
HARDNESS_COLUMN = "hardness_kgf_mm2"
HARDNESS_SCALE_COLUMN = "hardness_scale"
HARDNESS_UNKNOWN_SCALE = "unknown"

# welddb.info: ShMA = MMA (same process).
WELD_TYPE_ALIASES = {"ShMA": "MMA"}
# Levels with fewer than 20 rows cannot be learned (explicit list, not data-driven).
WELD_TYPE_OTHER = ["NGSAW", "NGGMA", "SAA", "GTAA", "GMAA"]
WELD_TYPE_OTHER_LABEL = "other"
# "Not reported" is kept as its own category.
UNKNOWN_CATEGORY = "unknown"
CATEGORICAL_FILL_UNKNOWN = ["current_type", "electrode_polarity"]

# --------------------------------------------------------------------------
# Source groups (validation groups, never a feature)
# --------------------------------------------------------------------------
# A label that starts with the key belongs to the group (EvansLetterC+1.5Cr -> EvansLetter).
SOURCE_PREFIX_RULES = {"EvansLetter": "EvansLetter"}
# 'p9-RR82011': 'pN' is a page marker, the report id follows.
SOURCE_PAGE_MARKER_PATTERN = r"p\d+"
# Same authors and year, label typo (unverified). Merging is the conservative
# direction for a grouped split.
SOURCE_ALIASES = {"Ga&K": "Gar&K", "KocakPEv": "KocakPREv"}

# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
# Rows are not removed. A '<col>_missing' flag is added for features missing in
# at least this share of the training fold.
MISSING_FLAG_MIN_RATE = 0.05

# Right-skewed impurities (exploration 4.A): shifted log, linear family only.
SKEWED_COLUMNS = [
    "s_conc_wt", "p_conc_wt", "cu_conc_wt", "sn_conc_ppm", "b_conc_ppm", "as_conc_ppm",
]

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
RANDOM_SEED = 0
N_SPLITS = 5
SOURCE_COLUMN = "source"

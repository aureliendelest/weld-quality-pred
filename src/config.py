"""Single source of truth for every preprocessing rule.

Each rule carries a one-line rationale so the assumptions stay auditable.
The reasoning behind them is in doc/01_preprocessing_plan.md.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "welddb.data"
FIGURES_DIR = PROJECT_ROOT / "figures"      # one subfolder per notebook
RESULTS_DIR = PROJECT_ROOT / "results"      # one subfolder per notebook: fold-level and summary tables (CSV)

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

# Linear family: scaled numeric features are clipped to [-SCALED_CLIP, SCALED_CLIP]. Without it,
# variables whose middle 50% is a single value keep their raw unit (Nb up to ~1000 ppm) and a
# linear model extrapolates wildly on a new study (notebook 03, 1.4).
SCALED_CLIP = 5.0

# Within-study deviations (ablation variants `center_within`, `pca_replace_within`): a study with fewer
# rows than this in the set being transformed has no usable mean, so its deviation is set to 0.
WITHIN_STUDY_MIN_ROWS = 5
# Permutations of Horn's parallel analysis, which fixes the number of PCA components inside each training fold.
PARALLEL_ANALYSIS_ITER = 100

# Right-skewed impurities (exploration 4.A): shifted log, linear family only.
SKEWED_COLUMNS = [
    "s_conc_wt", "p_conc_wt", "cu_conc_wt", "sn_conc_ppm", "b_conc_ppm", "as_conc_ppm",
]

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
RANDOM_SEED = 0
N_SPLITS = 5
# Leave-one-study-out: only studies with at least this many rows are held out, and a
# held-out study needs this many labeled rows for a target to be scored.
LOSO_MIN_ROWS = 20
SOURCE_COLUMN = "source"

# --------------------------------------------------------------------------
# Ablations and comparison rule (doc/04_ablation_plan.md)
# --------------------------------------------------------------------------
TENSILE_TARGETS = ["yield_strength_mpa", "uts_mpa", "elongation_pct", "reduction_of_area_pct"]

# Derived features, computed row by row from inputs (nothing is fitted, so nothing can leak).
DERIVED_FEATURES = {
    "ce_iiw": "carbon equivalent C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15; NaN when an element is not "
              "reported (a missing element is not zero), so only ~25% of the rows have it",
    "larson_miller": "tempering parameter (T + 273.15) * (20 + log10 t) for the post-weld heat "
                     "treatment; 0 for an as-welded deposit (PWHT temperature 0)",
}

# A model beats another when it gains at least MIN_SKILL_GAIN of skill on average over the
# held-out studies AND wins on at least MIN_WIN_SHARE of them (paired by study); otherwise it is a tie.
MIN_SKILL_GAIN = 0.05
MIN_WIN_SHARE = 2 / 3
ABLATION_SEEDS = (0, 1, 2, 3, 4)       # repeats of the 5-fold grouped protocol

# (study, target) pairs where skill measures nothing: the study's spread is so small that the
# training mean is already within a fraction of it (std 3.0 against 9.1 elsewhere, notebook 03, 2.0).
EXCLUDED_FROM_JUDGMENT = {("Cunh", "reduction_of_area_pct")}

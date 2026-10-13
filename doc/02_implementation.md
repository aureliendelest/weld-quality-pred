# Implementation notes — `src/`

How the preprocessing and evaluation layer is built, how to use it, what was
corrected along the way, and what is still an assumption. The reasoning behind
each rule is in [`01_preprocessing_plan.md`](01_preprocessing_plan.md); this file
describes the code.

## 1. Module map

| Module | Role |
|---|---|
| `src/config.py` | **Single registry of rules**: column lists, targets, excluded columns, censoring / unit-fix / range / nitrogen / hardness rules, weld-type merges, source-group rules, thresholds (`MISSING_FLAG_MIN_RATE`, `N_SPLITS`, `RANDOM_SEED`). Every rule has a one-line rationale. |
| `src/loading.py` | `load_raw()` reads `data/welddb.data` as **text** (`N` becomes NaN, nothing else does) and adds `source`; `derive_source()` applies the grouping rules. |
| `src/cleaning.py` | `clean_dataframe(raw, censored_factor=0.5)` turns text into typed numbers, driven by the registry (`censored_factor=None` keeps censored values as NaN, for sensitivity analyses). Parsers **raise** on any unknown notation instead of producing a silent NaN. |
| `src/preprocessing.py` | `feature_spec()` (per-target features with leakage guards), `MissingnessFlags`, `LogShift`, `build_preprocessor()` / `build_pipeline()` for the `tree` and `linear` families. `FeatureOptions` (flags on/off, dropped features, derived features, extra shifted logs, PCA scores fitted inside the training fold) and `add_derived()` (carbon equivalent, Larson-Miller parameter) serve the ablations; the defaults reproduce the plan. |
| `src/evaluation.py` | `make_folds()` (protocols `grouped`, `random`, `loso` = one held-out study at a time, and the opt-in diagnostic `grouped_weld` = same weld, any Charpy temperature, kept whole through `weld_fingerprint()`), `study_sample_weights()`, `evaluate_model()`, `compare_models()`, `summarize()`, `protocol_gap()`, `per_study_scores()`, `unlabeled_pool()`. Every fold reports R2, RMSE, MAE, `skill` (error reduction against the training-fold mean), `bias` and `within_rho` (Spearman inside a study). |
| `src/baselines.py` | `StudyMeanRegressor` and `baseline_models()`: the global mean and the study mean every model has to beat. |
| `src/models.py` | `GroupTunedCatBoost` (depth, L2 and number of rounds of a CatBoost chosen by cross-validation over studies, the rounds read off the validation curve) and `GroupTunedRidge`: a ridge whose penalty is chosen by cross-validation over whole studies of the training rows (a plain `RidgeCV` would pick a weak penalty because rows of one study look alike). Accepts `sample_weight`. |
| `src/ablation.py` | The feature-set ablations (`VARIANTS`, one change at a time) on two reference models, `run_ablation()`, and the agreed decision rule: `paired_decision()` (paired by held-out study: gain >= 0.05 and wins on >= 2/3 of the pairs, otherwise a tie), `decision_table()`, `loso_deltas()`, `grouped_seed_spread()`. Plan: `04_ablation_plan.md`. |
| `src/zoo.py` | The model zoo (Lasso, ElasticNet, k-NN, SVR, ExtraTrees, tuned forest, histogram boosting, XGBoost, CatBoost) built on `GroupTunedSearch` (`models.py`: grid search with cross-validation over whole studies, target standardised); `run_zoo`, `run_option_one` (leave-one-study-out for all, `grouped` x 5 seeds for the retained ones, saved tables are reused), `zoo_decisions` (paired by held-out study, same rule as the ablations), `retained`. Results: notebook 03, section 3. |
| `src/specification.py` | `Specification` (thresholds of a weld class, with source and verification status), `conformity` (three-valued label of measured welds) and `decide` (conforming / non conforming / to test / to measure from prediction intervals). Registry: ISO 2560-A E42 4, AWS A5.1 E7018, E7018-1, E6010, E6013. Plan: `doc/06_quality_decision_plan.md`. |
| `src/decision.py` | `loso_predictions`, `study_quantiles` / `add_intervals` (split conformal prediction with the study as the unit, each study weighing the same), `interval_coverage`, `decide_welds`, `decision_report`. Results: notebook 06. |
| `src/semi_supervised.py` | `PseudoLabelRegressor`, a minimal confidence-ranked pseudo-labeling wrapper. |
| `src/analysis.py` | `source_demeaned_correlations()` (within-study correlation, 4.E), `eta_squared()` (variance explained by a group, adjusted), `parallel_analysis()` (components above noise), `standardise_for_pca()` and `pca_report()` (notebook 02). |
| `tests/test_cleaning.py` | 34 tests (parsers, censoring factor, real-data counts, leakage guards, folds including `grouped_weld`, within-source correlation, eta-squared, parallel analysis, ridge, feature options, ablation decision rule). |

Data flow:

```
welddb.data --load_raw--> text frame (+ source) --clean_dataframe--> typed frame
   --feature_spec(target)--> columns of one target
   --build_preprocessor(family)--> fit on the TRAINING fold only --> estimator
```

## 2. Using it

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from src.loading import load_raw
from src.cleaning import clean_dataframe
from src.evaluation import ModelSpec, compare_models, summarize, protocol_gap
from src.semi_supervised import PseudoLabelRegressor

df = clean_dataframe(load_raw())          # 1652 rows; numeric columns + `<col>_censored` flags

rf = lambda: RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
models = [
    ModelSpec("rf", "tree", rf),
    ModelSpec("rf_pseudo", "tree", lambda: PseudoLabelRegressor(rf()), semi_supervised=True),
    ModelSpec("ridge", "linear", lambda: Ridge(alpha=1.0)),
]
results = compare_models(df, models)      # all 5 targets, both protocols, identical folds
summary = summarize(results)              # mean / std over folds
gap = protocol_gap(summary)               # R2(random) - R2(grouped)
```

Notebooks add the project root to `sys.path` (the loading cell of
`01_exploration.ipynb` finds it by looking for `src/config.py`). Run the tests with
`python -m pytest` (config in `pytest.ini`).

**Changing a rule** means editing `src/config.py` only (for example the censoring
factor or the weld-type merges), then running the tests: they pin the counts that a
rule change would move.

## 3. What each stage guarantees

- **Loading**: only the literal `N` becomes NaN (`keep_default_na=False`), so no value
  is silently reinterpreted. `source` has 33 groups.
- **Cleaning**: the 17 columns written as text become numeric; every non-null value is
  preserved (NaN counts are identical before and after); `<col>_censored` flags exist
  for the 14 censored columns; `weld_type` is merged / bucketed; missing
  `current_type` / `electrode_polarity` become `unknown`.
- **Feature sets**: 25 numeric features, 3 categorical, 11 censored flags (39 columns;
  40 for toughness, which adds `charpy_temp_c`). `feature_spec` raises if a target,
  an outcome column, an excluded column, `source`, or another target's test condition
  would leak in.
- **Preprocessors** (always `fit` on the training fold):
  - `tree`: missing flags, one-hot, NaN kept.
  - `linear`: missing flags, shifted log on S, P, Cu, Sn, B, As, train-fold median,
    `RobustScaler`, a clip of the scaled numeric features to [-5, 5], one-hot. Output has no
    NaN. `build_preprocessor(..., clip=None)` and `ModelSpec(clip=None)` reproduce the
    unclipped version, kept to document the defect.
  - Flags are created for features missing in at least 5% of the training fold
    (16 of 25 on the full data, 14 on a 1000-row subset), so which flags exist can
    change between folds.
- **Evaluation**: folds are built once over all rows and reused by every model and
  target. Each (target, fold) trains on labeled training rows and scores on labeled
  test rows. A fold with fewer than 10 labeled test rows is skipped (`MIN_TEST_ROWS`).
  The semi-supervised pool is the training-fold rows where that target is missing;
  the preprocessor is still fit on labeled rows only, so baseline and semi-supervised
  models see identical preprocessing.

## 4. Verification

- **Tests**: 34 pass (run them with the project's `.venv`: the default `python3` of the machine has scikit-learn 1.5, which has no `GroupKFold(shuffle=True)`). They pin: all numeric columns numeric, NaN counts unchanged,
  `n_conc_ppm` 1242 non-null, hardness 138, `b_conc_ppm` 419 censored, 33 source
  groups, weld-type levels (`MMA` 1180), `charpy_temp_c` only in the toughness
  features, no target in any feature set, grouped folds never share a source, folds
  are deterministic, and the within-source analysis flips the sign on synthetic data
  whose between-study offsets mask the true relation; eta-squared separates structured
  from random groups; parallel analysis recovers the number of latent factors.
- **Notebooks**: `01_exploration.ipynb` (38 cells) and `02_preprocessing_pca.ipynb`
  (55 cells) run top to bottom without errors from the `notebooks/` folder, on `src`;
  `03_supervised_models.ipynb` (20 cells, section 1 only) too.
- **Reference scores** (notebook 03, section 1; 5 targets, 200 trees): on a study held
  out of training, the training-mean predictor has R2 between -0.37 and -2.9, so R2 alone
  says little; the random forest has a skill against the training mean of 0.28-0.30 (yield
  strength), 0.39-0.42 (UTS), 0.16-0.22 (Charpy), 0.09-0.29 (reduction of area) and
  -0.06-0.28 (elongation), against 0.74-0.86 in the random protocol. The study mean alone
  gives a random-protocol skill of 0.20-0.55. The grouped folds are very uneven (one
  contains `Evans`), which is why the leave-one-study-out scores are reported per study.

## 5. Corrections made along the way

Found while checking the data and `data/welddb.info` before coding:

| Earlier claim | Fact |
|---|---|
| 135 `source` groups (`weld_id.split("-")[0]`) | 33 after explicit rules (`EvansLetter*` merged, `pN` page markers resolved to `RR82011`, two label typos merged) |
| 51 rows without target, 194 with all 5, 456 with 4 | 105, 134, 526: the first counts included hardness and fatt50 |
| Median row has 40% missing features | 25% on the 28 modeling features; max 57%; 32 rows above 50%, all `EvHtIp1979` |
| Source-stratified imputation | Impossible with a source-grouped split (the test source is unseen); replaced by a train-fold median |
| Categorical cardinality assumed low | Measured: `weld_type` has 10 levels (5 under 20 rows); `ShMA` is the same process as `MMA` |

Found while implementing and re-running the exploration:

| Earlier claim | Fact |
|---|---|
| `Wolst` never reports hardness, `EvansLetter` never reports Ni/Cr/Mo (notebook, doc, chat) | **Inverted**: the table reports % missing. `Wolst` always reports hardness; `EvansLetter` always reports Ni/Cr/Mo |
| 21 numeric columns written as text | 17 (21 counted all text columns, including 4 categoricals) |
| `log1p` for skewed impurities | Nearly the identity on wt% values of 0.001-0.14; replaced by `log(x + s)`, `s` = smallest positive training value |
| `n_conc_ppm` has 1183 values | 1242 non-null; 1183 excluded the 59 composite codes |
| Hardness scales Hv10 / Hv30 | Also `Hv5` (4 rows) |
| Process variables are the "solid" correlations (4.C) | Voltage, heat input, current vs reduction of area drop from about -0.6 to -0.1/-0.2 once the study mean is removed (4.E); Cr, V, Nb, Mn hold up |
| Ranking input-target pairs on the overall coefficient | Between-study differences dominate it; feature screening should use the within-source coefficient |
| README: 1,651 rows, `welddb.csv` / `welddb.txt` | 1,652 rows, `welddb.data` / `welddb.info` |
| Boron vs UTS is 0.50 within a study (4.E) | 0.14 once the censored rows are kept (n 45 -> 259): a small-sample artefact (notebook 02, 1.4) |
| Nb-V redundancy 0.56 (4.C) | 0.83: the estimate rested on fewer rows before cleaning |
| R2 close to 0 on a new study means almost nothing is explained (smoke test) | R2 is taken against the *test fold's* mean; the training-mean predictor scores -0.37 on yield strength, so the forest's 0.05 is a skill of 0.28 against it (notebook 03, 1.1) |
| The ridge is worse than the mean on new studies: linear models do not transfer | A defect of the pipeline: `RobustScaler` divides by a tiny interquartile range and scaled values reach 100-1200. Fixed: clip at 5 and a penalty chosen by study cross-validation inside the training data; the skill goes from -1.69 to +0.01 on yield strength (a manual search on the scored studies had suggested +0.09, which was optimistic) (notebook 03, 1.4) |
| `current_a` / `voltage_v` have a real Pearson / Spearman divergence (4.B) | It came mostly through their pairs with `co_conc_wt`; they are not in the top 10 on cleaned data (notebook 02, 1.3) |
| Only 134 rows have all targets, so a joint model is out | 650 rows have all four tensile targets; 134 is the effect of Charpy toughness (notebook 02, 2.8) |
| The variance partition of each input is a direct guide for choosing variables | The study share of an input does not predict whether its target correlations survive within a study (Spearman -0.09); screening stays on the within-study coefficient (notebook 02, 2.8) |
| The missing / censored flags are study fingerprints: they help in the random protocol and not on a new study (notebook 02, 2.4) | For the forest they change nothing (0.00-0.01 under every protocol); for the ridge removing the missing flags costs 0.03 (random), 0.04 (grouped), 0.07 (leave-one-study-out). They are information, not study recognition (notebook 03, 2.1) |
| 100 J in Charpy toughness is a test ceiling (notebook 01, 4.A) | Not a ceiling: 169 values exceed it. 514 of 879 values are 28, 35, 50, 70 or 100 J, probably a temperature reached at a fixed energy (to check in the sources) (notebook 03, `03_project_structure.md`) |
| Reduction of area is capped near 80% | No pile on a value (max 83): a physical saturation, not censoring |
| The random protocol leaks because rows are replicated (53% of rows share an input vector) | Measured on labeled rows: only 10-11% of tensile rows have a labeled twin; for Charpy 70% are the same weld at another temperature, 4% at the same temperature (notebook 02, 2.7) |

## 6. Assumptions that are not verified

- `Ga&K` merged into `Gar&K` and `KocakPEv` into `KocakPREv` (same authors and year,
  judged label typos). The merge is the conservative direction for a grouped split.
- `Evans`, `EvansLetter` and `EvHtIp1979` kept as three groups; they may share
  authors, in which case the grouped protocol is still slightly optimistic.
- `67tot33res` read as total 67 / residual 33 (`nd` = not determined); only the total
  is kept. The notice does not document these codes.
- `x/2` for values below the detection limit, with a `_censored` flag; the unit fixes
  (`<5` in vanadium wt%, `<0.01` in titanium and aluminium ppm) rest on the same
  columns already containing the right-unit equivalent threshold.
- `EvansLetter` ids (`C+1Cr`, `C+1.5Cr`, `C+2Cr`) suggest Cr added in steps to a common
  base weld; inferred from the ids only.

## 7. Known limitations and next steps

- Sections 4.A to 4.E of notebook 01 still run on `num`, a NaN-coerced copy; it is kept
  on purpose as the raw-data exploration. Notebook 02 re-checks 4.A-4.C on the cleaned
  data and gives a verdict per conclusion (its section 1.6); 4.E was not redone.
- `PseudoLabelRegressor` only accepts Random Forest / ExtraTrees (it needs the spread of
  the trees as a confidence measure). Self-training with other estimators, label
  propagation or co-training are not implemented.
- The within-source coefficient assumes a similar slope in every study and rests on few
  studies for some pairs (for example boron vs UTS: 45 rows, 4 studies).
- Not written yet: the model zoo and hyperparameter search (notebook 03, section 3), notebooks 04-05, the 32-row
  `EvHtIp1979` sensitivity run, the Charpy reformulation (waits for the source papers) and the model comparison
  table of the README. Done since the first version: `models.py`, the `grouped_weld` protocol (a diagnostic, opt-in) and
  the feature-set ablations (`04_ablation_plan.md`).
- Rule amendment awaiting agreement: "a tie keeps the baseline" instead of "a tie keeps the simpler setting" (`04_ablation_plan.md`).
- `GroupKFold(shuffle=True)` makes the grouped folds depend on the seed; the skill of the 5-fold grouped protocol moves by
  0.03-0.16 between seeds, so ablations repeat it on 5 seeds and the decision rests on leave-one-study-out.
- PCA findings that constrain the models: every leading component is mostly a study axis
  (adjusted eta-squared of PC1 against `source` 0.65-0.97), and the 25 numeric inputs are
  not compressible (80% of the variance needs 12 components). See `03_project_structure.md`.
- Requirements: `GroupKFold(shuffle=True)` needs scikit-learn 1.6 or later, and the
  type hints need Python 3.10 or later (the environment uses scikit-learn 1.9, Python 3.12).

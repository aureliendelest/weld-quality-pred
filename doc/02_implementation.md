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
| `src/cleaning.py` | `clean_dataframe(raw)` turns text into typed numbers, driven by the registry. Parsers **raise** on any unknown notation instead of producing a silent NaN. |
| `src/preprocessing.py` | `feature_spec()` (per-target features with leakage guards), `MissingnessFlags`, `LogShift`, `build_preprocessor()` / `build_pipeline()` for the `tree` and `linear` families. |
| `src/evaluation.py` | `make_folds()`, `evaluate_model()`, `compare_models()`, `summarize()`, `protocol_gap()`, `unlabeled_pool()`. |
| `src/semi_supervised.py` | `PseudoLabelRegressor`, a minimal confidence-ranked pseudo-labeling wrapper. |
| `src/analysis.py` | `source_demeaned_correlations()`, the within-study correlation used by notebook section 4.E. |
| `tests/test_cleaning.py` | 17 tests (parsers, real-data counts, leakage guards, folds, within-source analysis). |

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
    `RobustScaler`, one-hot. Output has no NaN.
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

- **Tests**: 17 pass. They pin: all numeric columns numeric, NaN counts unchanged,
  `n_conc_ppm` 1242 non-null, hardness 138, `b_conc_ppm` 419 censored, 33 source
  groups, weld-type levels (`MMA` 1180), `charpy_temp_c` only in the toughness
  features, no target in any feature set, grouped folds never share a source, folds
  are deterministic, and the within-source analysis flips the sign on synthetic data
  whose between-study offsets mask the true relation.
- **Notebook**: `01_exploration.ipynb` (38 cells) runs top to bottom without errors
  from its own folder, on `src.loading`.
- **Smoke test** of the evaluation layer (2 targets, 5 folds, default
  hyperparameters, 100 trees; not a result to report):

  | target | model | R2 random | R2 grouped |
  |---|---|---|---|
  | yield strength | random forest | 0.81 | 0.04 |
  | yield strength | ridge | 0.61 | -1.76 |
  | Charpy toughness | random forest | 0.83 | -0.84 |
  | Charpy toughness | ridge | 0.60 | -0.88 |

  Pseudo-labeling matches its baseline (0.81 / 0.83 random). The gap between the two
  protocols is the quantity the double protocol was designed to expose. Grouped folds
  are very uneven (test sizes 168, 413, 127, 174, 770: the 770 contains `Evans`), so
  the grouped numbers are noisy.

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

- Sections 4.A to 4.E of the exploration still run on `num`, a NaN-coerced copy where
  censored values are missing. They should be re-run on `clean_dataframe` output; the
  conclusions mark this as provisional.
- `PseudoLabelRegressor` only accepts Random Forest / ExtraTrees (it needs the spread of
  the trees as a confidence measure). Self-training with other estimators, label
  propagation or co-training are not implemented.
- The within-source coefficient assumes a similar slope in every study and rests on few
  studies for some pairs (for example boron vs UTS: 45 rows, 4 studies).
- Not written yet: PCA notebook, `models.py`, hyperparameter search, the 32-row
  `EvHtIp1979` sensitivity run, and the model comparison table of the README.
- Requirements: `GroupKFold(shuffle=True)` needs scikit-learn 1.6 or later, and the
  type hints need Python 3.10 or later (the environment uses scikit-learn 1.9, Python 3.12).

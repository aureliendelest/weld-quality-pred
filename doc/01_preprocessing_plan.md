# Preprocessing plan — weld-quality-pred

This document tracks the strategic choices made for the preprocessing
pipeline (`src/`), and why. It is the reference so we don't re-litigate the
same questions every iteration, and so that every notebook consumes exactly
the same rules. Every rule below is implemented in `src/config.py` (one
registry, one rationale per rule).

Context: the dataset is [welddb](http://www.phase-trans.msm.cam.ac.uk/map/data/materials/welddb-b.html),
1652 welds, 44 columns, compiled from independent publications. `N` means
"not reported", explicitly **not** zero (`data/welddb.info`). The exploration
(missingness, distributions, correlations) is in `notebooks/01_exploration.ipynb`.

## Revision history

**Revision 2** corrected several claims of the first version, found by checking
the data and `data/welddb.info`:

- `source` was derived as `weld_id.split("-")[0]`, which produced **135**
  pseudo-sources (82 ids have no dash; `EvansLetter*` was spread over 79 labels
  for 156 rows; `p9..p36` are page markers of a single report). After the
  explicit grouping rules (§4) there are **33** groups.
- Target availability had been computed on 7 targets. On the 5 retained
  targets: 0 targets = **105** rows, all 5 = **134**, 4 = **526**.
- "Median row has 40% missing features" was computed on a wider column set. On
  the 28 modeling features: median 25%, max 57%, and only 32 rows (all
  `EvHtIp1979`) exceed 50% (§12).
- Source-stratified imputation is incompatible with a source-grouped split (a
  test source is unseen at fit time). Replaced by a global train-fold median (§8).
- Category cardinalities were assumed, not measured (§7).
- The missingness examples were **inverted** (the table reports % *missing*): it is
  `Wolst` that always reports hardness and `EvansLetter` that always reports Ni/Cr/Mo.
  The conclusion (missingness depends on the study) is unchanged.
- 17 columns (not 21) are numeric columns loaded as text: 21 was the count of all
  text columns, including `current_type`, `electrode_polarity`, `weld_type`, `weld_id`.

## 1. Columns excluded from modeling

**Decision**: 8 columns at 92-98% missing (`fatt50_temp_c`, `w_conc_wt`,
`martensite_pct`, `ferrite_carbide_aggregate_pct`, `ferrite_second_phase_pct`,
`acicular_ferrite_pct`, `primary_ferrite_pct`, `co_conc_wt`) are excluded from
the feature set but kept in the DataFrame. Hardness is also never a feature.
Microstructure, fatt50 and hardness are measured outcomes, not weld inputs.

**Why**: 30-140 non-missing rows out of 1652 cannot support a reliable
feature. The original authors made the same call (`welddb.info`: columns with
little data "have been removed for the analysis"). Keeping them in the
DataFrame preserves a later mechanistic analysis without reloading raw data.
`co_conc_wt` additionally carries a single-source artefact (§4, exploration 4.D).

## 2. Targets

**Decision**: `yield_strength_mpa`, `uts_mpa`, `elongation_pct`,
`reduction_of_area_pct`, `charpy_toughness_j` (700-880 labeled rows each).
Hardness (n=138) and fatt50 (n=31) are out of scope. Targets are never used as
features of each other.

**Why**: only these five have enough rows. They form a known
strength/ductility/toughness trade-off (yield/UTS rho=0.93, both opposed to
ductility), so a single composite score would flatten what we want to model.
Using one target to predict another would not be possible for a new weld.

### `charpy_temp_c`

Present if and only if `charpy_toughness_j` is present (verified). It is a
legitimate **input for the toughness model only** (toughness depends on the
test temperature) and is excluded for the four other targets, where it would
only reveal which study produced the row.

## 3. Validation: double protocol

**Decision**: every model is scored on identical folds under two protocols:
(a) `GroupKFold` by source (generalisation to an unseen study), and
(b) shuffled stratified `KFold` with a fixed seed (in-distribution performance).

**Why**: `Evans` alone is 42% of the rows and 42-50% of the labeled rows of
each target, and only 5-6 groups per target have at least 20 labeled rows.
A grouped protocol alone is therefore very noisy (a fold is dominated by one or
two sources); a random protocol alone is optimistic (the same study on both
sides lets a model learn lab habits). Reporting both gives a stable number and
a pessimistic one, and **the gap between them measures how much performance
depends on study protocol**.

## 4. The `source` group

**Decision**: explicit rules in `config.py`: labels starting with `EvansLetter`
become `EvansLetter`; `pN` page markers resolve to the report id that follows
(`p9-RR82011` -> `RR82011`); two label typos are merged (`Ga&K` -> `Gar&K`,
`KocakPEv` -> `KocakPREv`, same authors/year, unverified). `Evans`,
`EvansLetter` and `EvHtIp1979` stay separate groups. `source` is used for
splitting and analysis only, **never as a feature**.

**Why**: the groups must follow studies, not id formatting, otherwise variants
of one study land on both sides of a grouped split. Merging the two typos is
the conservative direction (a wrong merge only makes the split slightly
harsher). Keeping the three Evans groups apart is an unverified assumption
(they may share authors); merging them would make the Evans family ~54% of the
data. As a feature, `source` would let a model memorise lab habits.

## 5. Missing values (MNAR): indicator flags, NaN kept for trees

**Decision**: for features with at least 5% missing (measured on the training
fold), add a `<col>_missing` flag. Tree models keep the NaNs.

**Why**: missingness depends on the study (`SvenGret` and `EvHtIp1979` never
report Ni/Cr/Mo while `EvansLetter` and `PantK` always do; `Wolst` always reports
hardness while most large sources never do), so it is MNAR. Imputing would
invent measurements a study never made, while the flag preserves "this study
did not measure it", which can itself be informative. MICE / regression
imputation was ruled out for the same reason (false precision).

## 6. Cleaning rules

| Rule | Decision | Why |
|---|---|---|
| Below detection `<x` | `x / 2` **plus** a `<col>_censored` flag | LOD convention. Censoring is massive (b 83%, al 44%, nb 40%, v 33%) with different thresholds per lab, so `x/2` alone would encode the lab's detection limit, i.e. the source. The flag lets a model separate measurements from estimates. |
| Unit errors | V `<5` -> `<0.0005` wt%; Ti and Al `<0.01` -> `<100` ppm | The same columns already contain the right-unit equivalents; unfixed, `x/2` would be wrong by orders of magnitude. |
| Ranges `150-200` | mean of bounds | Process ranges are centred on a target value. |
| Nitrogen `67tot33res` | total only (67); `66totndres` -> 66 | Undocumented; read as total/residual (`nd` = not determined). The residual is dropped. Hypothesis recorded in config. |
| Hardness | one numeric column + a `hardness_scale` column (unknown / Hv10 / Hv30 / Hv5) | 138 values: 80 without a scale, 38 Hv10, 16 Hv30, 4 Hv5. Separate columns would have 38 and 16 rows. Hardness is not a target or a feature, so no finer work is justified. |
| Any other non-numeric value | **raise an error** | A new notation must never become a silent NaN. |

Every rule is declared per column in `config.py` so it is auditable and easy
to change.

## 7. Categorical encoding

Measured cardinalities: `current_type` 2 (+13% NaN), `electrode_polarity` 3
(+9% NaN), `weld_type` 10.

**Decision**: `ShMA` is merged into `MMA` (`welddb.info`: `ShMA = MMA`, 1140 + 40
rows); `weld_type` levels with fewer than 20 rows (`NGSAW`, `NGGMA`, `SAA`,
`GTAA`, `GMAA`) become `other` (explicit list); NaN becomes `unknown` for
`current_type` and `electrode_polarity`; all are one-hot encoded.

**Why**: the same process must not be split across two levels, and levels with
4-18 rows cannot be learned. `unknown` keeps the "not reported" information
consistent with §5.

## 8. Non-tree models (linear, PCA, semi-supervised)

**Decision**: flags (§5), then a **global median fitted on the training fold**,
a log transform with a positive shift on the skewed impurities (S, P, Cu, Sn,
B, As), then `RobustScaler`. Trees receive none of this.

**Why**: these models cannot take NaN, and the median must come from the
training fold only (no leakage). A per-source median is impossible when the
test source is unseen. `RobustScaler` because the extreme S/P/Sn z-scores
(up to 11-12) are likely real rare materials, not errors, and would dominate a
standard scaler. The first version proposed `log1p`; it is nearly the identity
on wt% values of 0.001-0.14, so the transform is `log(x + s)` with `s` the
smallest positive training value of each column.

## 9. Per-target evaluation

**Decision**: one model and one score per target, on the rows where that target
exists. Metrics: R2, RMSE, MAE, as mean and std over folds.

**Why**: only 134 rows have all five targets; evaluating on them would discard
~91% of the usable data and bias toward studies that measure everything. Per-
target scores also show a model that is good on yield strength but poor on
toughness.

## 10. Semi-supervised comparison

**Decision**: for each target the unlabeled pool is every training-fold row where
*that* target is missing (772-952 rows, including the 105 rows with no target
at all). A pseudo-labeling regressor is compared with the supervised baseline
on **identical folds**, with the preprocessor fitted on labeled training rows
only. Test-fold rows are never in the pool.

**Why**: a pool shared by the 105 target-less rows is far smaller, while a study
that reports UTS but not toughness still provides usable features for the
toughness task.

## 11. Traceability

All rules live in `src/config.py` with a rationale; parsers raise on anything
unknown; tests in `tests/` pin the key counts.

## 12. Rows

**Decision**: no global row removal. A sensitivity run without the 32
`EvHtIp1979` rows (>50% missing features) checks that they do not drive results.

**Why**: on the 28 modeling features no row exceeds 57% missing (95th percentile
50%), so there is no "75% empty" noise to filter. The sparsest rows all come from
one study, and a threshold would remove a whole study, which is a bias.

## Summary

| # | Issue | Decision |
|---|---|---|
| 1 | Sparse columns | Out of features, kept in the DataFrame; outcomes never features |
| 2 | Targets | 5 mechanical properties; `charpy_temp_c` input for toughness only |
| 3 | Validation | Grouped CV and random CV on identical folds |
| 4 | Source | Explicit prefix rules (33 groups), never a feature |
| 5 | Missingness | `_missing` flags (>=5%), NaN kept for trees |
| 6 | Cleaning | `x/2` + `_censored` flag, unit fixes, range mean, nitrogen total, hardness value + scale |
| 7 | Categoricals | ShMA=MMA, rare levels -> other, NaN -> unknown, one-hot |
| 8 | Non-tree models | Train-fold median, shifted log, RobustScaler |
| 9 | Evaluation | One score per target on its own rows |
| 10 | Semi-supervised | Pool per target, same folds as the baseline |
| 12 | Rows | No removal; sensitivity run |

## Status

`src/` implements this plan: `config.py`, `loading.py`, `cleaning.py`,
`preprocessing.py`, `evaluation.py`, `semi_supervised.py`; tests in `tests/`.
See [`02_implementation.md`](02_implementation.md) for how the code works, the
corrections made during implementation, and the assumptions still unverified.

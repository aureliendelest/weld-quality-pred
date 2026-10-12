# Project structure and notebook roadmap

How the project is organised now that the preprocessing and evaluation logic
lives in `src/`, what each notebook is for, and what notebook 02 settled for
the next stages. Code details are in [`02_implementation.md`](02_implementation.md),
the reasoning behind the rules in [`01_preprocessing_plan.md`](01_preprocessing_plan.md).

## Principle

Notebooks narrate and visualise; `src/` computes. A notebook contains no cleaning
rule and no fold logic: it imports `src`, saves figures to `figures/<notebook>/`
and tables to `results/`. A later notebook reads earlier tables from `results/`
instead of recomputing them.

## Layout

```
data/                     raw files (welddb.data, welddb.info)
doc/                      one numbered plan per stage
src/                      config, loading, cleaning, preprocessing, evaluation,
                          semi_supervised, analysis        (+ models.py with notebook 03)
tests/                    pinned counts, leakage guards, folds, analysis helpers
notebooks/
  01_exploration.ipynb              raw data (frozen)                     done
  02_preprocessing_pca.ipynb        cleaning effects + PCA                done
  03_supervised_models.ipynb        per-target models, double protocol    started (section 1)
  04_semi_supervised.ipynb          semi-supervised vs baseline           to do
  05_comparison_recommendations.ipynb   reads results/*.csv only          to do
results/                  CSV tables written by the notebooks
figures/<notebook>/       exploration/ (01), preprocessing_pca/ (02), ...
report/                   final report
```

Every notebook starts with the same bootstrap cell: it finds the project root by
looking for `src/config.py` and adds it to `sys.path`, so notebooks run from any folder.

## Notebook contracts

| # | Purpose | Uses | Produces |
|---|---|---|---|
| 01 | Explore the raw data on a NaN-coerced copy. Frozen: it documents what the raw data showed. | `src.loading`, `src.analysis` | `figures/exploration/` |
| 02 | Re-check 4.A-4.C on cleaned data, measure the dependence on the censoring rule, run four PCAs, test a derived alloy-family feature, measure replicated rows. | `src.cleaning`, `src.preprocessing`, `src.analysis` | `figures/preprocessing_pca/`, `results/02_*.csv` |
| 03 | Supervised models per target (Ridge/Lasso, kNN, SVR, Random Forest, Gradient Boosting, MLP) under the validation protocols, hyperparameters tuned inside the training fold, feature importance. | `src.models` (new), `src.evaluation` | `results/03_*.csv` |
| 04 | At least one semi-supervised method justified from the literature, compared with the 03 baseline on identical folds with the per-target unlabeled pool. | `src.semi_supervised`, `src.evaluation` | `results/04_*.csv` |
| 05 | Comparison table (mean and standard deviation over folds, protocol gap), per-target reading, domain recommendations, limits. | `results/*.csv` only | tables and final figures |

## What notebook 02 settled

Verdicts on the conclusions of 01 are in its section 1.6; the decisions that bind
the next stages are:

1. **Cleaning rules stay**: `x/2` plus the `_censored` flag. `x/2`, `x` and `0` give
   within-study correlations with a rank correlation of 0.993-0.997; dropping the
   censored rows (what 01 did) is the biased choice (rank correlation 0.959, changes
   up to 0.36).
2. **Study structure dominates the data**: the leading component of all four PCAs is
   mostly a study axis (adjusted eta-squared against `source` 0.65-0.97), and adding
   the missing/censored flags makes studies separate into tight clusters. The 25
   numeric inputs are not compressible (80% of the variance needs 12 components).
3. **No derived alloy-family feature and no PCA scores by default**: the Cr family
   explains 18-29% of yield strength, UTS and elongation overall but only 4-11% within
   a study; the alloy axis is 97% study.
4. **Feature screening uses the within-study coefficient**: Cr, V, Nb and Mn are
   credible drivers (Cr vs elongation -0.60 overall, -0.43 within); voltage, heat
   input and current are mostly study identifiers (-0.6 overall, -0.1 to -0.2 within).
5. **The targets are mostly within-study and nearly one-dimensional**: the four tensile properties span one axis (76% of their variance, the same inside studies) and only 28% of that axis is between studies, while the arc variables are 89-98% study. 650 rows have all four tensile targets.
6. **Replicated inputs**: 53% of rows share their input vector with another row, but
   only 10-11% of the labeled tensile rows have a labeled twin; for Charpy, 70% of the
   labeled rows are the same weld measured at other temperatures.

## What notebook 03 settled so far

1. **R2 on a new study needs a reference**: the training-mean predictor scores R2 of -0.37 to -2.9, so models are read through `skill`, the error reduction against the training mean.
2. **Strength transfers to unseen studies**: the forest beats the training mean on 6 of 7 studies (yield strength) and 7 of 7 (UTS), with a within-study Spearman of about 0.65-0.75; ductility is shakier (4 of 7 studies), Charpy 4 of 5.
3. **The random protocol is mostly study recognition for some targets**: the study mean alone gives 74% of the forest's random skill for reduction of area, 40-43% for yield strength, UTS and elongation, 25% for Charpy.
4. **The linear family is fixed and modest**: scaled features are clipped to [-5, 5] and the ridge penalty is chosen by study cross-validation; on a new study it is about as good as the mean for yield strength and Charpy (skill +0.01 and -0.02), +0.15 for UTS, below the mean for the ductility targets, and below the forest everywhere.
5. **Scores are reported per held-out study**, since an average is dominated by `Evans` (42% of the rows).

## Open for notebook 03

| Question | Proposal | Needs the user |
|---|---|---|
| A third validation protocol grouped by input fingerprint (same weld, any temperature), to separate "new weld" from "new study" and to deflate the Charpy random score | add `protocol="grouped_weld"` to `make_folds` | yes, it changes the evaluation design validated earlier |
| Missing/censored flags and PCA scores as options | ablations with and without, under both protocols, PCA fitted inside each training fold | no |
| Shifted log on Al, Nb, V (skew 1.7-2.1 once cleaned) | ablation for the linear family | no |
| A joint model of the four tensile properties on their 650 complete rows (multi-output or strength score + residuals) against four separate models; Charpy kept apart | compare under both protocols | no |
| Isolated PCA outliers (`Wats-1981-CWO`, `Kluket-CuMnB-1994-11/12`, `Cunh-1982-19aw/ht`, `PakSven-1990-9`) | inspect before any removal; none is removed | no |

## Conventions

- Seeds, `N_SPLITS` and thresholds come from `src/config.py`; folds come from
  `make_folds`, so results of 03 and 04 are comparable by construction.
- A new stage gets a numbered plan in `doc/` before its notebook.
- Anything computed in a notebook that another notebook needs goes into `src/`,
  with a test.

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
  05_comparison_recommendations.ipynb   reads results/*/*.csv only          to do
results/<notebook>/       preprocessing_pca/ (02), supervised_models/ (03), ...: CSV tables
figures/<notebook>/       exploration/ (01), preprocessing_pca/ (02), ...
report/                   final report
```

Every notebook starts with the same bootstrap cell: it finds the project root by
looking for `src/config.py` and adds it to `sys.path`, so notebooks run from any folder.

## Notebook contracts

| # | Purpose | Uses | Produces |
|---|---|---|---|
| 01 | Explore the raw data on a NaN-coerced copy. Frozen: it documents what the raw data showed. | `src.loading`, `src.analysis` | `figures/exploration/` |
| 02 | Re-check 4.A-4.C on cleaned data, measure the dependence on the censoring rule, run four PCAs, test a derived alloy-family feature, measure replicated rows. | `src.cleaning`, `src.preprocessing`, `src.analysis` | `figures/preprocessing_pca/`, `results/preprocessing_pca/02_*.csv` |
| 03 | Supervised models per target (Ridge/Lasso, kNN, SVR, Random Forest, Gradient Boosting, MLP) under the validation protocols, hyperparameters tuned inside the training fold, feature importance. | `src.models` (new), `src.evaluation` | `results/supervised_models/03_*.csv` |
| 04 | At least one semi-supervised method justified from the literature, compared with the 03 baseline on identical folds with the per-target unlabeled pool. | `src.semi_supervised`, `src.evaluation` | `results/semi_supervised/04_*.csv` |
| 05 | Comparison table (mean and standard deviation over folds, protocol gap), per-target reading, domain recommendations, limits. | `results/*/*.csv` only | tables and final figures |

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

6. **Feature-set ablations (section 2, `doc/04_ablation_plan.md`)**: 36 ties, 1 helps (ridge without the arc variables, +0.06), 1 hurts (ridge with study weights on Charpy). The feature set of the zoo is the baseline. The flags are not study fingerprints for the forest (0.00-0.01 whatever the protocol) and are useful to the ridge (-0.07 without the missing flags on a new study). The grouped skill moves by 0.03-0.16 between seeds, which is the size of the gains looked for.
7. **Why `Wolst`, `Cunh` and `Pat` fail** (diagnosis by a helper agent, scikit-learn 1.5, scripts outside the repository; the Mo effect is reproduced in section 2.2, the rest is **not** reproduced in a notebook yet):
   - molybdenum is the first feature of the forest and carries the failure of `Wolst` (its Mo is out of the training range and its Spearman with yield strength is -0.57 against +0.31 in the other studies), of `Pat` on Charpy and of `Cunh` on reduction of area; removing it is not a fix (it hurts the alloyed studies);
   - `Cunh` reduction of area: the study's standard deviation is 3.0 against 9.1, the skill measures nothing there (pair left out of the comparison rule); `Cunh` elongation is a real failure, unexplained (the Mn sign is opposite to the other studies, n = 31);
   - `Wolst` n = 23: the bootstrap intervals of its skill and within-study Spearman contain 0; its own data are coherent (leave-one-weld-out skill 0.57);
   - `Pat`: weld type `TSA` and 72-75 V do not exist in training, but removing voltage, current and heat input changes nothing.
8. **Charpy toughness: 514 of its 879 values are 28, 35, 50, 70 or 100 J** (`EvansLetter` 104 of 104, `Evans` 344 of 437; checked). 100 J is not a ceiling (169 values exceed it). In `Evans`, a weld has at most one value at 28 J and two at 100 J, and 64 of the 65 welds with both read T(28 J) < T(100 J). Working hypothesis, **not confirmed** (to check in the source papers): these rows give the temperature at which a fixed energy is reached, `charpy_temp_c` being the real measurement. If true, the target has to be reformulated.
9. **Comparison rule (agreed)**: paired by held-out study, gain >= 0.05 and wins on >= 2/3 of the (target, study) pairs, otherwise a tie; the four tensile targets pooled, Charpy apart; (`Cunh`, reduction of area) left out; `grouped_weld` is a diagnostic, never a ranking.

## Open for notebook 03

| Question | Proposal | Needs the user |
|---|---|---|
| ~~A third validation protocol grouped by input fingerprint~~ | **done**: `make_folds(..., "grouped_weld")`, opt-in, diagnostic only | agreed |
| ~~Missing/censored flags and PCA scores as options~~ | **done** (section 2): ties, baseline kept | no |
| ~~Shifted log on Al, Nb, V~~ | **done**: tie (+0.05, 52% of the pairs) | no |
| Charpy reformulation (fixed-energy values) | check the source papers, then predict the temperature for a fixed energy; sensitivity row without these values | the sources |
| `drop_arc` as an option for the linear models of the zoo | test on Lasso, SVR, MLP | no |
| A joint model of the four tensile properties on their 650 complete rows (multi-output or strength score + residuals) against four separate models; Charpy kept apart | compare under both protocols | no |
| Isolated PCA outliers (`Wats-1981-CWO`, `Kluket-CuMnB-1994-11/12`, `Cunh-1982-19aw/ht`, `PakSven-1990-9`) | inspect before any removal; none is removed | no |

## Conventions

- Seeds, `N_SPLITS`, the comparison thresholds (`MIN_SKILL_GAIN`, `MIN_WIN_SHARE`) and the pair left out of the judgment come from `src/config.py`; folds come from
  `make_folds`, so results of 03 and 04 are comparable by construction.
- A new stage gets a numbered plan in `doc/` before its notebook.
- Anything computed in a notebook that another notebook needs goes into `src/`,
  with a test.

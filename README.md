# 🔧 Steel Weld Quality Prediction

> A machine learning project (supervised and semi-supervised) that predicts the mechanical properties of steel weld deposits from their chemical composition and welding parameters.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

---

## 📌 Background

Weld quality is a major concern across many industries (energy, offshore wind, pipelines, shipbuilding…), representing billions of euros. Today, knowledge about what makes a good weld is mostly passed down **from one expert welder to another**, leaving companies highly dependent on a small pool of specialists.

This project aims to:

- **extract and standardize** expert knowledge from data;
- **discover new patterns** that influence weld quality;
- **provide recommendations** for achieving better welds.

## 📊 Dataset

**MAP_DATA_WELD**, a public database from the Phase Transformations Group at the University of Cambridge, compiled by T. Cool and H. K. D. H. Bhadeshia from the literature.

- 🔗 [Database page](https://www.phase-trans.msm.cam.ac.uk/map/data/materials/welddb-b.html)
- **1,652 rows** (weld deposits), **44 columns**
- Files: `data/welddb.data` (data) and `data/welddb.info` (column descriptions)

| Variable group | Examples |
|---|---|
| **Chemical composition** | C, Si, Mn, S, P, Ni, Cr, Mo, V, Cu, Co, W (wt%); O, Ti, N, Al, B, Nb, Sn, As, Sb (ppm) |
| **Welding parameters** | current (A), voltage (V), AC/DC, electrode polarity, heat input (kJ/mm), interpass temperature, weld type (MMA, SAW, FCA, GTAA, GMAA, ES, TSA…) |
| **Heat treatment** | post-weld heat treatment (PWHT) temperature and time |
| **Mechanical properties** | yield strength, ultimate tensile strength (UTS), elongation, reduction of area, Charpy impact toughness and test temperature, hardness, 50% FATT |
| **Microstructure** | % primary ferrite, ferrite with second phase, acicular ferrite, martensite, ferrite with carbide aggregate |
| **Identifier** | Weld ID (literature source) |

⚠️ **Missing values**: an `N` means the value was *not reported* in the source publication. It does **not** mean zero. The dataset is therefore only partially labeled, which motivates the semi-supervised approach.

## 🎯 Objectives

1. **Descriptive analysis** of the dataset and identification of relevant preprocessing steps (units, missing values, scaling, PCA).
2. **Identify target variables** that represent weld quality (e.g. yield strength, UTS, elongation, Charpy toughness) and define a prediction strategy.
3. **Apply several ML approaches** with a rigorous cross-validation protocol.
4. **Apply semi-supervised methods** (at least one) to leverage unlabeled data, with a literature-based justification.
5. **Compare model performance** using appropriate metrics.
6. **Conclude and recommend** how to obtain good weld quality.

## 🧠 Methodology

```
Raw data ──► Cleaning & handling of "N" ──► Exploratory analysis & PCA
                                                    │
                                                    ▼
             Target selection ──► Supervised models (rigorous CV)
                                                    │
                                                    ▼
                          Semi-supervised methods (unlabeled data)
                                                    │
                                                    ▼
                        Model comparison ──► Domain recommendations
```

### Preprocessing
- Convert `N` to missing values (`NaN`)
- Harmonize units (wt% vs ppm)
- Drop / impute columns and rows that are too incomplete
- Encode categorical variables (weld type, AC/DC, polarity)
- Scale / standardize features (very heterogeneous ranges)
- PCA to explore the structure of the data

### Models
- **Supervised**: random forest and a study-tuned ridge are done (reference models); Lasso, k-NN, SVR, histogram gradient boosting and MLP are next, with hyperparameters tuned by cross-validation over studies inside the training fold.
- **Semi-supervised**: confidence-ranked pseudo-labeling (`src/semi_supervised.py`) is written; a transductive variant (unlabeled rows of the held-out study) and a second method for regression (co-training) are planned. They are compared with their supervised base on identical folds.

### Metrics and validation
- RMSE, MAE and R² in physical units, plus the **skill** = 1 - SSE(model) / SSE(training-fold mean): R² compares with the mean of the *test* fold and the training-mean predictor already scores R² of -0.37 to -2.9 on an unseen study.
- Every model runs on identical folds under `grouped` (whole studies held out, repeated on 5 seeds), `loso` (one study at a time, the decision protocol) and `random` (contrast only: it largely measures how well a study is recognised). `grouped_weld` (same weld, any Charpy temperature, kept whole) is a diagnostic.
- Comparison rule, paired by held-out study: a gain of at least 0.05 of skill **and** wins on at least two thirds of the (target, study) pairs, otherwise a tie (`doc/04_ablation_plan.md`).

## 📁 Repository structure

```
.
├── data/
│   ├── welddb.data             # raw data
│   └── welddb.info             # column descriptions
├── doc/
│   ├── 01_preprocessing_plan.md   # strategic choices and why
│   ├── 02_implementation.md       # how src/ works, corrections, open assumptions
│   ├── 03_project_structure.md    # notebook roadmap and decisions for the next stages
│   └── 04_ablation_plan.md        # feature-set ablations and the decision rule
├── figures/                    # one subfolder per notebook (exploration/, preprocessing_pca/)
├── notebooks/
│   ├── 01_exploration.ipynb         # raw data, frozen
│   ├── 02_preprocessing_pca.ipynb   # cleaning effects and PCA
│   ├── 03_supervised_models.ipynb   # baselines, per-study scores, feature-set ablations (started)
│   ├── 04_semi_supervised.ipynb     # planned
│   └── 05_comparison_recommendations.ipynb   # planned
├── results/                    # CSV tables written by the notebooks, one subfolder per notebook
├── src/
│   ├── config.py               # every cleaning / feature rule, with its rationale
│   ├── loading.py              # read the raw file, derive the study group `source`
│   ├── cleaning.py             # text -> typed numeric data
│   ├── preprocessing.py        # per-target features, tree / linear preprocessors
│   ├── evaluation.py           # identical folds, one score per target
│   ├── baselines.py            # global-mean and study-mean reference predictors
│   ├── models.py               # study-tuned ridge, GroupTunedSearch (any regressor tuned by CV over studies)
│   ├── zoo.py                  # classical regressors, small grids, paired comparison against a reference
│   ├── ablation.py             # feature-set ablations and the paired decision rule
│   ├── semi_supervised.py      # pseudo-labeling regressor
│   └── analysis.py             # within-study correlations, PCA helpers
├── tests/
├── report/
├── requirements.txt
└── README.md
```

Design choices (study-grouped validation, missing-value handling, cleaning rules) are
documented in [`doc/01_preprocessing_plan.md`](doc/01_preprocessing_plan.md) and
[`doc/02_implementation.md`](doc/02_implementation.md).

## 🚀 Getting started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the tests
python -m pytest

# Launch the notebooks
jupyter notebook
```

Main dependencies: `pandas`, `numpy`, `scikit-learn` (1.6 or later), `matplotlib`, `seaborn`, `missingno`, `jupyter`, `pytest`.

## 📈 Results

*Work in progress: the model zoo, the semi-supervised comparison and the final table are not done yet. What is established so far (notebook 03, `results/supervised_models/`):*

Skill against the training-fold mean (0 = predicts that mean), 5 targets, by protocol:

| Target | Forest random | Forest grouped | Forest leave-one-study-out | Ridge leave-one-study-out |
|---|---|---|---|---|
| Yield strength | 0.82 | 0.28 | 0.30 | 0.01 |
| UTS | 0.86 | 0.42 | 0.39 | 0.15 |
| Elongation | 0.74 | 0.28 | -0.06 | -0.38 |
| Reduction of area | 0.74 | 0.29 | 0.09 | -0.23 |
| Charpy toughness | 0.83 | 0.16 | 0.22 | -0.02 |

- Strength transfers to a study never seen (the forest beats the mean on 6 of 7 studies for yield strength, 7 of 7 for UTS); ductility and toughness are less reliable.
- The random protocol is far more flattering: knowing only the study already gives 25-74% of the forest's random skill.
- Feature-set ablations (36 ties out of 38 comparisons): the baseline feature set is kept; the skill of the grouped protocol moves by 0.03-0.16 between seeds, the size of the gains looked for.
- Charpy toughness needs reformulating: 514 of 879 values are 28, 35, 50, 70 or 100 J, probably the temperature at which a fixed energy is reached (to be checked in the source papers).

## 💡 Recommendations

*To be completed: main levers identified for improving weld quality (composition, heat input, post-weld heat treatment, etc.).*

## 👥 Team

- **Members**: Aurélien Delest, Antoine Loudier

## 📚 References

1. T. Cool, *Design of Steel Weld Deposits*, PhD Thesis, University of Cambridge, 1996.
2. T. Cool, H. K. D. H. Bhadeshia and D. J. C. MacKay, "The yield and ultimate tensile strength of steel welds", *Materials Science and Engineering A*, 1997.
3. MAP_DATA_WELD database, Phase Transformations Group, University of Cambridge.

## 📄 License

*To be specified (e.g. MIT). The data comes from the public MAP database at the University of Cambridge.*

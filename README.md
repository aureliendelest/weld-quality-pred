# 🔧 Steel Weld Quality Prediction

> A machine learning project (supervised and semi-supervised) that predicts the mechanical properties of steel weld deposits from their chemical composition and welding parameters.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
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
- **1,651 rows** (weld deposits), **44 columns**
- Files: `data/welddb.csv` (data) and `data/welddb.txt` (column descriptions)

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

### Models *(adapt to your actual work)*
- **Supervised**: regularized linear regression (Ridge/Lasso), k-NN, SVR, Random Forest, Gradient Boosting, MLP
- **Semi-supervised**: Self-training, Label Propagation / Label Spreading, co-training *(to be specified)*

### Metrics *(adapt to your actual work)*
- RMSE, MAE, R² for regression
- k-fold cross-validation (watch for data leakage between welds from the same source, see `Weld ID`)

## 📁 Repository structure *(suggested)*

```
.
├── data/
│   ├── welddb.csv
│   └── welddb.txt
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing_pca.ipynb
│   ├── 03_supervised_models.ipynb
│   └── 04_semi_supervised.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── report/
│   └── report.pdf
├── requirements.txt
└── README.md
```

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

# Launch the notebooks
jupyter notebook
```

Main dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`.

## 📈 Results

*To be completed: model comparison table (metrics, cross-validation standard deviations) and key figures (PCA, feature importance, predicted vs. actual values).*

| Model | RMSE | MAE | R² |
|---|---|---|---|
| … | … | … | … |

## 💡 Recommendations

*To be completed: main levers identified for improving weld quality (composition, heat input, post-weld heat treatment, etc.).*

## 👥 Team

- **Team no.**: _to be completed_
- **Members**: _First Last (email)_, _First Last (email)_

## 📚 References

1. T. Cool, *Design of Steel Weld Deposits*, PhD Thesis, University of Cambridge, 1996.
2. T. Cool, H. K. D. H. Bhadeshia and D. J. C. MacKay, "The yield and ultimate tensile strength of steel welds", *Materials Science and Engineering A*, 1997.
3. MAP_DATA_WELD database, Phase Transformations Group, University of Cambridge.

## 📄 License

*To be specified (e.g. MIT). The data comes from the public MAP database at the University of Cambridge.*

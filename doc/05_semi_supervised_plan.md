# Semi-supervised learning plan — weld-quality-pred

This document details the strategy, theoretical foundations, implementation and evaluation
protocol for the semi-supervised learning stage of the project, satisfying the explicit requirements
of `Sujet-soudures.docx`.

## 1. Problem formulation & Motivation

The dataset contains 1,652 welds, but each mechanical target is only partially labeled:
- `yield_strength_mpa`: 879 labeled rows (773 unlabeled)
- `uts_mpa`: 874 labeled rows (778 unlabeled)
- `elongation_pct`: 708 labeled rows (944 unlabeled)
- `reduction_of_area_pct`: 700 labeled rows (952 unlabeled)
- `charpy_toughness_j`: 879 labeled rows (773 unlabeled)
- 105 welds have **no mechanical target** at all.

In standard supervised learning, any weld lacking the target of interest is discarded.
However, their chemical composition ($C, Mn, Si, Cr, Ni, Mo, V, Nb\dots$) and welding process
parameters ($Current, Voltage, Heat\ Input\dots$) are almost fully recorded.

Semi-supervised learning aims to leverage the input marginal distribution $p(X)$ from the
unlabeled pool $\mathcal{U}$ to improve the conditional estimation $p(y \mid X)$, reducing
prediction error and improving generalization to unseen welding studies.

## 2. Bibliographical review: Semi-supervised regression

While semi-supervised classification is widely studied (cluster assumption, margin maximization),
**semi-supervised regression** presents unique challenges:
1. **Absence of natural decision boundaries**: outputs are continuous $\hat{y} \in \mathbb{R}$ without
   clear margins or low-density separation regions.
2. **Error accumulation (confirmation bias)**: if an incorrect continuous pseudo-label is assigned,
   retraining reinforces the bias, leading to model degradation.

Two primary approaches from the literature address these challenges:

### A. Confidence-Ranked Pseudo-Labeling / Self-Training
- **Reference**: Lee (2013), Chapelle, Schölkopf & Zien (2006) *Semi-Supervised Learning*,
  Triguero et al. (2015) *Self-labeled techniques for semi-supervised learning*.
- **Mechanism**: Train a base regressor on labeled data $\mathcal{L}$. For unlabeled points
  $x \in \mathcal{U}$, estimate both the predicted value $\hat{y}(x)$ and the model's confidence /
  epistemic uncertainty $\sigma(x)$. Select the subset of $\mathcal{U}$ with highest confidence
  (lowest uncertainty), assign pseudo-labels, add them to the training set $\mathcal{L} \leftarrow \mathcal{L} \cup \mathcal{U}_{\text{selected}}$,
  and retrain over several rounds.
- **Confidence estimation in regression**: In Random Forests or Extra-Trees ensembles, individual
  trees provide diverse predictions $t_b(x)$. The inter-tree standard deviation:
  $$\sigma(x) = \sqrt{\frac{1}{B}\sum_{b=1}^B (t_b(x) - \bar{t}(x))^2}$$
  provides a well-calibrated measure of predictive uncertainty. Points where trees strongly agree
  ($\sigma(x)$ is minimal) represent high-confidence regions close to training data manifold.

### B. Co-Training for Regression
- **Reference**: Zhou & Li (2005), *Semi-Supervised Regression with Co-Training Style Algorithms*,
  IEEE Transactions on Knowledge and Data Engineering (TKDE).
- **Mechanism**: Uses two distinct learners (or two views). In our domain, a natural two-view
  distinction exists:
  - *View 1 (Metallurgy)*: Chemical composition of base and weld metals.
  - *View 2 (Process)*: Electrical arc parameters, thermal cycle, and heat treatment.
  Alternatively, two models with different inductive biases (e.g. Random Forest and Extra-Trees
  or Ridge) cross-label confident instances for each other.

### C. Inductive vs. Transductive evaluation
- **Inductive setting (Primary Benchmark)**: The unlabeled pool $\mathcal{U}$ contains only the
  unlabeled rows from the **training fold studies**. The held-out test study is completely unseen
  (neither $X_{\text{test}}$ nor $y_{\text{test}}$ are used during training).
- **Transductive setting (Domain adaptation diagnostic)**: In industrial application, a welder
  often knows the input parameters of their new welds ($X_{\text{test}}$) before conducting
  destructive tensile tests. Including $X_{\text{test}}$ in the unlabeled pool evaluates whether
  unlabeled domain adaptation helps cross-study transfer.

## 3. Strict leakage guards & Evaluation protocol

To guarantee scientific rigor:
1. **Identical Folds**: The semi-supervised models run on the exact same folds (`grouped`, `loso`, `random`)
   as the supervised baselines (`src.evaluation.make_folds`).
2. **Preprocessing Isolation**: Preprocessors (`build_preprocessor`) are fitted strictly on the
   **labeled training rows** ($\mathcal{L}_{\text{train}}$). Imputation medians and scaling parameters
   never see unlabeled or test data at fit time.
3. **Identical Estimator Hyperparameters**: The base Random Forest uses the exact same random seed,
   number of trees (200), and max features as the supervised reference.
4. **Metrics**: Evaluated on physical metrics (RMSE, MAE, $R^2$) and the benchmark **`skill`**
   relative to the training-fold mean:
   $$\text{Skill} = 1 - \frac{\sum_{i} (y_i - \hat{y}_i)^2}{\sum_{i} (y_i - \bar{y}_{\text{train}})^2}$$

## 4. Decision rule

Following `doc/04_ablation_plan.md`:
- Paired comparison by held-out study under `loso`.
- Semi-supervised **helps** if $\Delta\text{skill} \ge +0.05$ and it wins on $\ge 2/3$ of the (target, study) pairs.
- Symmetrically **hurts** if $\Delta\text{skill} \le -0.05$ and it loses on $\ge 2/3$ of pairs.
- Otherwise, a **tie**.


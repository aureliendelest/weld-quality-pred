# Ablation plan — before the model zoo

Why: the feature set must be fixed once, on two reference models, before comparing a zoo of
models; otherwise each model gets different inputs and the comparison means nothing. Same
folds, same rule for every variant. Code: `src/ablation.py`; results: `results/supervised_models/03_ablation_*.csv`;
narrative: notebook 03, section 2.

## Models and protocols

- Reference models: random forest (tree family, 200 trees) and the study-tuned ridge (linear family).
- Protocols: `grouped` (5 folds, repeated on 5 seeds), `loso` (one study held out; the decision), `random` (contrast only).
- `grouped_weld` is a diagnostic, not used here.

## Variants (one change at a time against `baseline`)

| Variant | What changes | Why |
|---|---|---|
| `no_missing_flags`, `no_censored_flags`, `no_flags` | remove the `_missing` / `_censored` flags | reporting patterns are study fingerprints (PCA 4): they should help a seen study and not a new one |
| `drop_mo` | remove molybdenum | first feature of the forest and the cause of 3 of the 4 failures diagnosed (Wolst, Pat, Cunh); a mechanism check, not a candidate setting |
| `drop_arc` | remove current, voltage, heat input | 89-98% of their variance is the study (notebook 02, 2.8) |
| `study_weights` | weight each training row by 1/sqrt(rows of its study) | `Evans` is 42% of the rows, the aim is a new study |
| `pca6` | append 6 PCA scores fitted inside each training fold | parallel analysis keeps 6 components; leading ones are study axes |
| `add_ce`, `add_larson_miller` | carbon equivalent, tempering parameter | metallurgical features never tested |
| `log_al_nb_v` | shifted log on Al, Nb, V (ridge only) | skew 1.7-2.1 once cleaned |
| `pca_replace` | the numeric features are replaced by PCA scores (components fixed by parallel analysis inside each training fold) | `pca6` appends scores, this removes the raw features |
| `pca_replace_within` | same, on deviations from the study mean | the leading axes are study axes (notebook 02) |
| `center_within` | the numeric features are replaced by their deviation from the study mean, no PCA | control: separates the effect of centring from that of the PCA |

The study mean is taken over the rows of the set being transformed, so a held-out study is centred on its own inputs
(no label); a study with fewer than 5 rows there gets a deviation of 0. Code: `ReplaceNumeric` in `src/preprocessing.py`.

## Decision rule (agreed)

A variant is compared with `baseline` on identical folds, paired by held-out study (loso).
Unit of comparison: the pairs (target, study) of the four tensile targets pooled, and Charpy
separately (few studies). **Helps** if the mean skill gain is >= 0.05 and it wins on >= 2/3 of
the pairs; **hurts** symmetrically; otherwise **tie**, and **the baseline is kept**: only a verdict
`helps` changes the setting (amended and agreed, see the end of this file). The pair (Cunh, reduction of area) is left out (the skill
there has a tiny denominator). The mean `within_rho` gain is reported alongside; the grouped and
random gains are context, never a ranking.

Limits stated in advance: 5-7 studies per target, so most verdicts will be ties; an ablation
chosen on the scored studies proves a mechanism, it does not give a gain to claim.

## Results (notebook 03, section 2; `results/supervised_models/03_ablation_*.csv`)

38 comparisons: 36 ties, 1 helps, 1 hurts. Nothing helps the forest.

- **Helps**: the ridge without current, voltage and heat input (tensile +0.06, 81% of the pairs; grouped +0.03 on average over 5 seeds, positive on 4; forest -0.03, tie).
- **Hurts**: the ridge with study weights, on Charpy (-0.20).
- **Flags**: no effect on the forest under any protocol (0.00-0.01). For the ridge, removing the missing flags costs 0.03 (random), 0.04 (grouped), 0.07 (loso): they are information, not study recognition. Tie by the rule.
- **Mo**: removing it rescues `Wolst` (UTS +0.57, yield +0.48) and `Pat` (Charpy +0.56) but hurts `Evans`, `EvansLetter`, `Gar&K`, `RR82011`: 21 of 33 pairs get worse. A mechanism check, not a setting.
- **PCA, carbon equivalent, Larson-Miller, shifted log on Al/Nb/V**: ties. The carbon equivalent exists on 25% of the rows only.
- **Noise floor**: the grouped skill moves by 0.03-0.16 between seeds.

## Decision

The feature set of the zoo is the baseline. `drop_arc` is carried as an option for the other linear models.

**Amendment to the rule above (agreed)**: "a tie keeps the simpler setting" was replaced by "**a tie keeps the baseline**; only a verdict `helps` changes the feature set". Applied literally, the first wording would remove the missing flags, Mo and the arc variables, against the evidence.

## Results: replacing the features by PCA scores or by within-study deviations (`results/supervised_models/03_ablation_pca_*.csv`)

12 comparisons, same folds and same rule; the baseline reproduces the first run to 1e-15. Components kept by parallel analysis: 6
(`pca_replace`), 7 (`pca_replace_within`).

| Variant | Forest, tensile | Ridge, tensile | Forest, Charpy | Ridge, Charpy |
|---|---|---|---|---|
| `pca_replace` | -0.54 hurts | +0.03 tie | -0.21 hurts | -0.35 tie |
| `pca_replace_within` | -0.73 hurts | -0.19 hurts | -0.17 tie | +0.08 tie |
| `center_within` | -0.40 hurts | -0.08 tie | +0.06 tie | +0.35 helps |

Nothing is adopted. The feature set stays the baseline. The one `helps` (ridge, Charpy, 5 studies) is not reliable:
the Charpy test temperature is centred with the other numeric features (not separated), and 1 verdict out of 12 is
within chance. The mean within-study rank correlation on the tensile targets barely moves (ridge 0.45 -> 0.47 with `center_within`).

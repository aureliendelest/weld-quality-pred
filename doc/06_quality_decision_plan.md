# Quality decision plan — is a weld qualitative or not?

The project predicts mechanical properties; the initial question is whether a weld is *qualitative*. This document
defines that word, justifies the definition with the standards, and lays out the decision strategy. Code:
`src/specification.py` (tests in `tests/test_cleaning.py`). Sources are listed at the end; each claim says how well
it was checked.

## 1. Definition: conformity to a specification

**Decision**: a weld is qualitative when its all-weld-metal properties **meet a specification**, i.e. a set of
thresholds on the properties of the project (yield strength, UTS, elongation, Charpy energy at a temperature). The
specification depends on the use of the steel; there is no universal threshold.

**Why** (what the sources show):

- A consumable is *classified* by its all-weld-metal properties. ISO 2560 (covered electrodes, non-alloy and
  fine-grain steels) has two systems: A, classified by yield strength with an average impact energy of 47 J, and B,
  classified by tensile strength with 27 J. It applies to steels up to 500 MPa of minimum yield strength or 570 MPa of
  minimum tensile strength [1][2].
- AWS A5.1 classifies the same way: tensile strength in the name (E60xx, E70xx), and minimum yield strength and
  elongation in a tension table. Manufacturers' sheets give for E7018: yield >= 400 MPa (58 ksi), tensile >= 480 MPa
  (70 ksi), elongation >= 22%; for E7018-1 the same plus 27 J at -45 C (-50 F) [3][4]. A secondary copy of the A5.1
  table gives E6010 at 48 / 60 ksi with 22% and E6013 with 17% [5].
- The classification triple is therefore *strength, elongation, impact energy at a temperature*: exactly four of our
  five targets. The reduction of area is not part of it, to the best of our knowledge (not verified in the standards).
- The thresholds of an application come from the **application code**, not from the consumable class: ISO 15614-1
  defers the weld-metal impact values to the application standard (a forum statement, weak source [8]); an offshore
  example quotes 47 J at about -40 C [7], illustrative only.
- Which class to use follows from the **base metal and the service**: filler strength is normally matched to the
  specified minimum strength of the base metal, undermatching is considered for high-strength steels and some
  fillet or partial-penetration joints, overmatching is generally discouraged [9][10][11]. EN 1011 and the DNV rules
  were not reached, so their own limits are unconfirmed.

**Consequence for the project**: the specification is data, chosen by the use of the steel. The prediction models do not
change with it; only the decision step does.

## 2. Is a fixed working specification "classic"?

Yes, as long as it is a real class (the designation of an ISO 2560 electrode, such as E42 2 B32H5, spells out the
strength symbol, the impact symbol and the other fields in that order [13]). A standard class is a point of a small grid: a **strength class** (yield or tensile
level, with the elongation attached) crossed with an **impact grade** (temperature at which 47 J or 27 J is required).
The registry of `src/specification.py` follows that structure and only holds classes taken from standards or
manufacturers, each with its source and a status:

| Name | Requirements | Status |
|---|---|---|
| ISO 2560-A E42 4 | yield >= 420, UTS 500-640, elongation >= 20%, 47 J at -40 C | secondary: the ELGA guide as quoted by a search, the standard itself not read [6] |
| AWS A5.1 E7018 | yield >= 400, UTS >= 480, elongation >= 22% | verified on a manufacturer sheet [3][4], tension table secondary [5] |
| AWS A5.1 E7018-1 | the same, 27 J at -45 C | verified on a manufacturer sheet [3][4] |
| AWS A5.1 E6010 | yield >= 331, UTS >= 414, elongation >= 22% | secondary [5] |
| AWS A5.1 E6013 | yield >= 331, UTS >= 414, elongation >= 17% | secondary [5] |

What is **not** classic would be a threshold invented without a class. The other ISO 2560-A strength classes (35, 38,
46, 50) and the full list of impact grades are not in the registry because their tables could not be read.

**Conditioning by use**: a use is expressed by choosing the specification (`get_specification(name)`); the same
predictions are then judged against its thresholds, and `Specification.conditions()` gives the Charpy test temperature at which
the toughness must be predicted. A function deriving the class from the base metal (matching strength) and the minimum
service temperature needs the ISO 2560:2020 tables and is left for when they are available.

## 3. What the dataset says about these specifications

Measured values only, three-valued (a Charpy result proves conformity only when tested at or below the required
temperature, and non conformity only when tested at or above it, since the energy falls with the temperature):

| Specification | conforming | non conforming | unknown | share conforming among the decided |
|---|---|---|---|---|
| ISO 2560-A E42 4 | 34 | 408 | 1210 | 8% (290 welds fall outside the UTS band of 500-640 MPa) |
| AWS A5.1 E7018 | 465 | 217 | 970 | 68% |
| AWS A5.1 E7018-1 | 30 | 243 | 1379 | 11% |
| AWS A5.1 E6010 | 536 | 144 | 972 | 79% |
| AWS A5.1 E6013 | 642 | 30 | 980 | 96% |

- **A full-specification label exists for very few welds** (30 or 34 of 1652 where Charpy is part of it): most rows
  lack at least one property. A classifier trained directly on "conforming or not" would learn from a few dozen
  examples. The decision is therefore built **per property** from regressions.
- The literature compilation is probably favourable (welds published to show a good result), so these shares are
  orders of magnitude, not rates for a workshop.

## 4. Decision strategy

1. **Predict each required property** with the best model of notebook 03 (tuned forest or CatBoost; Charpy at the
   specification's temperature, `conditions()`).
2. **Attach an interval calibrated by held-out study.** The interval comes from the leave-one-study-out residuals
   (conformal prediction by study), never from mixed rows: the error of a new study is a level error, not a noise
   (mean |bias| 12 against a spread of 4 to 270 depending on the target, notebook 03). Built in `src/decision.py`. The Cambridge group
   reaches the same need with Bayesian neural networks whose error bars flag sparse or noisy regions [12].
3. **Decide in three states per property, then overall** (`Specification.decide`): conforming when the whole interval
   is on the right side of the threshold, non conforming when the whole interval is on the wrong side, **to test** in
   between. One non-conforming property makes the weld non conforming; the weld is conforming only when all are.
4. **A property that no model predicts on a new study has to be measured** (`to_measure`): the elongation, which is
   required by the classes, has a skill near 0 or negative for every model.
5. **Asymmetric cost** (engineering judgement, no source found): accepting a bad weld costs far more than refusing a
   good one, hence the decision on the *lower* bound and a "to test" state. The ratio of costs is a parameter to set
   with the client; it is not known.
6. **Re-level for a new study, process or laboratory** with a few measured welds: the models rank the welds of a
   study well (within-study rank correlation 0.5-0.6) and place its level badly (notebook 03, 3.1).

**Evaluation of the decision** (not the skill): among the welds called conforming, the share that truly conform (the
costly error); the share of welds sent to "to test" (the price of caution); both leave-one-study-out and study by
study, with the worst study reported, not only the mean.

## 5. Results (notebook 06, `results/quality_decision/`)

Model: CatBoost with tuned rounds, predicted by leave-one-study-out; intervals at a nominal coverage of 80%.

- **The intervals are wider than the spread of the data**: +/-109 MPa on yield strength (std 93), +/-90 on UTS (89), +/-7.8
  points on elongation (4.9), +/-13.4 on reduction of area (8.9), +/-105 J on Charpy (50). The coverage is 0.90-0.93 on
  average but 0.52-0.65 on `Wolst`, the study that fails everywhere in notebook 03.
- **The model cannot certify a strict class.** With intervals, `E7018`, `E7018-1` and `ISO 2560-A E42 4` accept no weld (all
  to test): the thresholds are closer to the typical values than the half-widths. Ignoring the uncertainty (point
  rule) accepts almost everything with a precision close to the base rate (0.75 against 0.72 for `E7018`).
- **On a lenient class (`E6013`, 97% conforming) the interval rule accepts 352 of 543 welds with 9 false accepts**: the
  precision of 0.97 equals the base rate, so the model adds nothing over accepting everything.
- **The reliable use is screening for non-conformity**: a weld called non conforming is right 91-100% of the time
  (12% to 45% of the non-conforming welds found, depending on the class).
- **Raising the accepted share costs false accepts** (coverage 0.5: 317 `E7018` accepted, 51 false accepts), so the cost
  ratio between the two errors, unknown, has to fix the level.

The recentring of a new study with a few measured welds (point 6 of the strategy) is the step expected to narrow the
intervals; it is not tested.

## 6. Limits and next steps

- Thresholds quoted from a manufacturer sheet or a secondary copy are marked as such; none was read in the standard
  itself (ISO 2560, AWS A5.1 and the application codes are paywalled). The numbers should be checked against the
  standard of the client before any use outside the project.
- To do, in this order: the recentring of a new study with a few measured welds and its effect on the width of the
  intervals; the class derived from the base metal and the service temperature once the ISO 2560:2020 tables
  are available; the Charpy reformulation (514 of 879 values read as a fixed energy) before relying on the toughness.

## Sources

1. [ISO 2560:2020, Standards Council of Canada](https://scc-ccn.ca/standardsdb/standards/8170836): scope, systems A and B (as read in the search summary).
2. [ISO 2560:2020, AFNOR](https://www.boutique.afnor.org/en-gb/standard/iso-25602020/welding-consumables-covered-electrodes-for-manual-metal-arc-welding-of-nona/xs135675/132382): same.
3. [Lincoln Electric, E7018-1](https://www.lincolnelectric.com/fr-CA/Products/millenniumarc70181_smaw): 400 (58) / 480 (70) / 22, 27 J at -50 F.
4. [Sureweld 7018 specification sheet](https://www.weldingsuppliesfromioc.com/cdn/shop/files/Sureweld_7018_Specs.pdf): E7018 data (the search summary attributed a 7018-1 sheet to ESAB; the owner of this particular file was not checked).
5. [AWS A5.1 tension table, secondary copy (Scribd)](https://www.scribd.com/document/831341815/AWS-Classifications): E6010, E6013, E7018, E7024.
6. [ELGA, classification EN ISO 2560-A](https://elga.evostudio.pl/wp-content/uploads/2021/07/ELGA-Classification_EN-ISO-2560-A-1.pdf): symbol 42 and impact symbol 4 (as summarised by a search; the PDF could not be fetched).
7. [Kobelco, The ABC's of arc welding](https://kobelco-welding.jp/education-center/abc/ABC_2008-04.html): 47 J at about -40 C for an offshore structure (illustrative).
8. [Eng-Tips thread on ISO 15614-1](https://eng-tips.com/goto/post?id=9049776): impact values set by the application standard (forum, weak).
9. [CWB Group, choosing welding materials for strength](https://www.cwbgroup.org/resources/articles/what-are-the-considerations-for-choosing-welding-materials-for-strength-and): matching is the norm, undermatching for high-strength steels.
10. [The Fabricator, welding metals of dissimilar strengths](https://www.thefabricator.com/article/arcwelding/welding-metals-of-dissimilar-strengths): undermatching and overmatching.
11. [Hobart Brothers, matching filler metals](https://www.hobartbrothers.com/?p=3066): example A36 with a 70 ksi filler.
12. [Lalam, Bhadeshia, MacKay, Cambridge](https://brearley.msm.cam.ac.uk/abstracts/harsha.html) and [Metzbower, Graz](https://www.phase-trans.msm.cam.ac.uk/2004/Graz.Metzbower.pdf): Bayesian neural networks with error bars on weld-metal strength, elongation, reduction of area and toughness.
13. [TWI, welding consumables part 3](https://twi-global.com/technical-knowledge/job-knowledge/welding-consumables-part-3-084): the structure of an ISO 2560 designation (E42 2 B32H5). The page could not be fetched; read through a search summary.

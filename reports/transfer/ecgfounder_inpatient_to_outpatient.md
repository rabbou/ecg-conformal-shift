# Transfer report: ECGFounder frozen, with logistic probes, Columbia inpatients to Columbia outpatients

Calibrated on 1903 inpatient ECGs and applied unchanged to 1059 outpatients, the per-label thresholds cover 71.6% of outpatients with structural heart disease (composite) where 90% was asked, cover 97.7% of those without it, and send 32.2% to a human. The composite's prevalence falls from 53.2% to 25.6%.

| | Source | Target |
|---|---|---|
| Cohort | EchoNext validation split, inpatients | EchoNext test split, outpatients |
| ECGs, one per patient | 1903 | 1059 |
| Composite prevalence | 53.2% | 25.6% |
| LVEF ≤45% prevalence | 24.5% | 6.6% |

Model: ECGFounder 12-lead Net1D frozen, one logistic probe per label. Input: the tracing as distributed, 12 leads at 250 Hz, unit z-score. Nothing is refitted on the target except on the ladder at the end of the page.

## Coverage per label at the 90% level

Ill covered is the share of patients with the finding whose decision includes it: the sensitivity for the plain threshold, the coverage of the positive class for the two conformal schemes. A per-label threshold that the calibration positives cannot certify flags everyone, and the row then shows the whole target flagged. The plain threshold and the per-label threshold of the ill are the same calibration quantile, so their columns agree; the per-label scheme adds a threshold for the healthy, and with it the share sent to a human.

| Label | Prevalence, source | Prevalence, target | Ill in target | Ill covered, plain | Ill covered, pooled | Ill covered, per-label | Healthy covered, per-label | Sent to a human, per-label | AUROC, target |
|---|---|---|---|---|---|---|---|---|---|
| LVEF ≤45% | 24.5% | 6.6% | 70 | 84.3% | 55.7% | 84.3% [74.0%, 91.0%] | 98.0% | 12.6% | 0.915 |
| LV wall ≥1.3 cm | 22.1% | 13.8% | 146 | 80.1% | 50.0% | 80.1% [72.9%, 85.8%] | 97.0% | 41.7% | 0.781 |
| Aortic stenosis, moderate+ | 6.8% | 4.8% | 51 | 100.0% | 0.0% | 100.0% [93.0%, 100.0%] | 93.9% | 39.6% | 0.891 |
| Aortic regurgitation, moderate+ | 1.7% | 0.4% | 4 | 100.0% | 0.0% | 100.0% [51.0%, 100.0%] | 95.0% | 83.7% | 0.885 |
| Mitral regurgitation, moderate+ | 7.7% | 2.8% | 30 | 76.7% | 0.0% | 76.7% [59.1%, 88.2%] | 97.2% | 24.2% | 0.764 |
| Tricuspid regurgitation, moderate+ | 9.1% | 2.5% | 26 | 80.8% | 7.7% | 80.8% [62.1%, 91.5%] | 97.7% | 30.4% | 0.804 |
| Pulmonary regurgitation, moderate+ | 0.6% | 0.0% | 0 | n/a | n/a | n/a | 97.1% | 93.6% | n/a |
| RV dysfunction, moderate+ | 11.5% | 1.9% | 20 | 85.0% | 25.0% | 85.0% [64.0%, 94.8%] | 97.9% | 15.3% | 0.867 |
| Pericardial effusion, moderate+ | 1.7% | 0.2% | 2 | 100.0% | 0.0% | 100.0% [34.2%, 100.0%] | 98.5% | 58.5% | 0.951 |
| PASP ≥45 mmHg | 15.4% | 5.9% | 63 | 74.6% | 17.5% | 74.6% [62.7%, 83.7%] | 97.6% | 35.8% | 0.750 |
| TR velocity ≥3.2 m/s | 7.4% | 3.3% | 35 | 82.9% | 0.0% | 82.9% [67.3%, 91.9%] | 97.1% | 34.1% | 0.782 |
| Composite (any of the above) | 53.2% | 25.6% | 271 | 71.6% | 74.5% | 71.6% [65.9%, 76.6%] | 97.7% | 32.2% | 0.791 |

## Coverage of the ill inside groups, composite, per-label thresholds

| Group | | Ill | Ill covered [95% CI] |
|---|---|---|---|
| sex | female | 126 | 62.7% [54.0%, 70.6%] |
| sex | male | 145 | 79.3% [72.0%, 85.1%] |
| age | 18-49 | 30 | 53.3% [36.1%, 69.8%] |
| age | 50-64 | 69 | 59.4% [47.6%, 70.2%] |
| age | 65-79 | 117 | 80.3% [72.2%, 86.5%] |
| age | 80+ | 55 | 78.2% [65.6%, 87.1%] |
| race_ethnicity | asian | 11 | 81.8% [52.3%, 94.9%] |
| race_ethnicity | black | 39 | 74.4% [58.9%, 85.4%] |
| race_ethnicity | hispanic | 58 | 72.4% [59.8%, 82.2%] |
| race_ethnicity | other | 22 | 68.2% [47.3%, 83.6%] |
| race_ethnicity | unknown | 32 | 75.0% [57.9%, 86.7%] |
| race_ethnicity | white | 109 | 68.8% [59.6%, 76.7%] |

## Calibration of the composite probability

Slope 1 and intercept 0 are perfect calibration; the intercept is calibration-in-the-large on the logit scale.

| Test split, context | Prevalence | Mean predicted | Slope | Intercept | Brier |
|---|---|---|---|---|---|
| inpatients | 52.5% | 55.2% | 0.88 | -0.16 | 0.182 |
| emergency | 40.0% | 43.2% | 0.95 | -0.21 | 0.163 |
| outpatients | 25.6% | 31.9% | 0.90 | -0.43 | 0.148 |

Calibration curve on outpatients, ten bins of equal count:

| Predicted range | ECGs | Mean predicted | Observed [95% CI] |
|---|---|---|---|
| 2.2% to 8.1% | 106 | 5.9% | 6.6% [3.2%, 13.0%] |
| 8.2% to 10.9% | 106 | 9.4% | 10.4% [5.9%, 17.6%] |
| 10.9% to 14.3% | 106 | 12.4% | 6.6% [3.2%, 13.0%] |
| 14.3% to 18.3% | 106 | 16.3% | 15.1% [9.5%, 23.1%] |
| 18.4% to 23.3% | 106 | 20.7% | 14.2% [8.8%, 22.0%] |
| 23.3% to 29.8% | 106 | 26.1% | 18.9% [12.6%, 27.4%] |
| 30.0% to 40.3% | 106 | 35.1% | 23.6% [16.5%, 32.5%] |
| 40.4% to 54.5% | 106 | 46.3% | 33.0% [24.8%, 42.4%] |
| 54.7% to 72.4% | 106 | 62.6% | 47.2% [37.9%, 56.6%] |
| 72.4% to 98.6% | 105 | 85.2% | 81.0% [72.4%, 87.3%] |

## Net benefit of sending outpatients to echocardiography, composite

Net benefit is true positives per patient minus false positives per patient weighted by the odds of the threshold.

| Threshold | Model | Echo for all | Echo for none |
|---|---|---|---|
| 5% | 0.216 | 0.217 | 0.000 |
| 10% | 0.177 | 0.173 | 0.000 |
| 20% | 0.127 | 0.070 | 0.000 |
| 30% | 0.092 | -0.063 | 0.000 |
| 40% | 0.068 | -0.240 | 0.000 |
| 50% | 0.052 | -0.488 | 0.000 |

## Positive predictive value at the plain threshold

The source column is what a buyer computes from the sensitivity and specificity measured at the source, at the target's prevalence; the target column is observed.

| Label | Prevalence, target | Sensitivity, source / target | Specificity, source / target | PPV from source | PPV observed | False alerts per ill patient found, from source / observed |
|---|---|---|---|---|---|---|
| Composite (any of the above) | 25.6% | 90.1% / 71.6% | 42.1% / 71.3% | 34.9% | 46.2% | 1.9 / 1.2 |
| LVEF ≤45% | 6.6% | 90.4% / 84.3% | 59.8% / 87.5% | 13.7% | 32.2% | 6.3 / 2.1 |

## Target labels that repair the threshold and the calibration

The target is cut once by patient into a pool and an evaluation half. Rung 0 spends the source thresholds; rung n refits the per-label thresholds and an intercept shift on n ECGs drawn from the pool, over repeated draws, all read on the same evaluation half.

| Label | Target labels | Ill covered, mean [10th, 90th centile] | Healthy covered | Draws that flag everyone | Absolute intercept after shift |
|---|---|---|---|---|---|
| Composite (any of the above) | 0 | 70.8% [70.8%, 70.8%] | 97.2% | 0.0% | 0.44 |
| Composite (any of the above) | 25 | 99.3% [100.0%, 100.0%] | 90.3% | 86.5% | 0.48 |
| Composite (any of the above) | 50 | 96.0% [88.2%, 100.0%] | 90.2% | 10.5% | 0.33 |
| Composite (any of the above) | 100 | 95.3% [88.3%, 100.0%] | 89.5% | 0.0% | 0.20 |
| Composite (any of the above) | 200 | 95.2% [91.2%, 99.3%] | 88.6% | 0.0% | 0.13 |
| Composite (any of the above) | 400 | 94.8% [92.7%, 95.6%] | 88.3% | 0.0% | 0.06 |
| LVEF ≤45% | 0 | 84.4% [84.4%, 84.4%] | 98.0% | 0.0% | 0.65 |
| LVEF ≤45% | 25 | 100.0% [100.0%, 100.0%] | 91.2% | 100.0% | 0.62 |
| LVEF ≤45% | 50 | 100.0% [100.0%, 100.0%] | 91.5% | 100.0% | 0.54 |
| LVEF ≤45% | 100 | 98.3% [93.8%, 100.0%] | 89.9% | 73.0% | 0.36 |
| LVEF ≤45% | 200 | 95.8% [84.4%, 100.0%] | 90.3% | 3.0% | 0.24 |
| LVEF ≤45% | 400 | 96.0% [90.6%, 100.0%] | 90.3% | 0.0% | 0.14 |

## Positive predictive value recomputed by Bayes' rule against observed

At the plain threshold, the recipe carries the source's sensitivity and specificity to the target's true prevalence. The gap is recomputed minus observed, in percentage points, with the 2.5th and 97.5th centiles of 2,000 redraws of both tables; the observed PPV carries its 95% Wilson interval. If only the prevalence had changed, the mean likelihood ratio among the healthy would stay at its source value, close to 1.

| Label | Prevalence, target | Flagged | PPV recomputed | PPV observed | Gap, points | Mean likelihood ratio of the healthy, source / target |
|---|---|---|---|---|---|---|
| LVEF ≤45% | 6.6% | 183 | 13.7% | 32.2% [25.9%, 39.3%] | -18.5 [-23.5, -13.8] | 1.06 / 0.35 |
| LV wall ≥1.3 cm | 13.8% | 493 | 17.5% | 23.7% [20.2%, 27.7%] | -6.2 [-8.5, -4.0] | 1.00 / 0.63 |
| Aortic stenosis, moderate+ | 4.8% | 504 | 8.2% | 10.1% [7.8%, 13.1%] | -1.9 [-3.0, -1.1] | 1.00 / 0.82 |
| Aortic regurgitation, moderate+ | 0.4% | 940 | 0.4% | 0.4% [0.2%, 1.1%] | -0.0 [-0.1, +0.0] | 1.00 / 0.86 |
| Mitral regurgitation, moderate+ | 2.8% | 289 | 5.0% | 8.0% [5.4%, 11.7%] | -3.0 [-4.9, -1.1] | 1.03 / 0.49 |
| Tricuspid regurgitation, moderate+ | 2.5% | 354 | 3.5% | 5.9% [3.9%, 8.9%] | -2.5 [-4.0, -1.1] | 1.04 / 0.51 |
| Pulmonary regurgitation, moderate+ | 0.0% | 1022 | n/a | n/a | n/a | n/a |
| RV dysfunction, moderate+ | 1.9% | 190 | 3.3% | 8.9% [5.7%, 13.9%] | -5.6 [-8.7, -2.9] | 1.03 / 0.33 |
| Pericardial effusion, moderate+ | 0.2% | 636 | 0.2% | 0.3% [0.1%, 1.1%] | -0.1 [-0.2, +0.0] | 1.00 / 0.73 |
| PASP ≥45 mmHg | 5.9% | 412 | 7.6% | 11.4% [8.7%, 14.8%] | -3.8 [-5.8, -2.0] | 1.00 / 0.54 |
| TR velocity ≥3.2 m/s | 3.3% | 395 | 4.3% | 7.3% [5.2%, 10.3%] | -3.0 [-4.6, -1.6] | 1.00 / 0.54 |
| Composite (any of the above) | 25.6% | 420 | 34.9% | 46.2% [41.5%, 51.0%] | -11.3 [-14.9, -7.8] | 0.98 / 0.41 |

## Three repairs judged on net benefit

Read on one half of the outpatients, cut by patient. Net benefit is in true positives per 100 patients. The prevalence correction uses no target label; the recalibration fits an intercept and a slope on 100 labelled ECGs from the other half, averaged over 200 draws (10th to 90th centile in brackets); the per-label sets use no target label and send their abstentions to a human, whose decision is not modelled, so they show two values, abstentions cleared and abstentions referred.

| Label | Threshold | As delivered | Prevalence corrected | Recalibrated on 100 | Per-label sets, cleared / referred | Treat all |
|---|---|---|---|---|---|---|
| LVEF ≤45% | 5% | 4.4 | 0.0 | 4.8 [4.5, 5.0] | 2.7 / 4.7 | 1.5 |
| LVEF ≤45% | 10% | 3.9 | 0.0 | 3.9 [3.8, 4.1] | 2.6 / 4.0 | -4.0 |
| LVEF ≤45% | 20% | 2.8 | 0.0 | 2.9 [2.5, 3.1] | 2.3 / 2.5 | -17.0 |
| Composite (any of the above) | 5% | 22.7 | 0.0 | 21.9 [21.0, 22.7] | 5.6 / 17.0 | 22.9 |
| Composite (any of the above) | 10% | 18.4 | 0.0 | 17.7 [16.9, 18.4] | 5.5 / 15.7 | 18.7 |
| Composite (any of the above) | 20% | 13.2 | 0.0 | 12.8 [12.4, 13.2] | 5.2 / 12.7 | 8.5 |

| Label | Prevalence, evaluation half | Estimated without labels | Sent to a human by the per-label sets |
|---|---|---|---|
| LVEF ≤45% | 6.4% | 0.0% | 11.3% |
| Composite (any of the above) | 26.8% | 0.0% | 32.3% |

## Cells filled by other tasks

- Distance without labels between source and target ECGs: empty, filled by T-068.
- The same pair at a second hospital, Columbia to Beth Israel (MIMIC-IV-Echo): empty, filled by T-065, ambitious version, after PhysioNet credentialing.

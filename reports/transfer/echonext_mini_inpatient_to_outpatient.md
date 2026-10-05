# Transfer report: the published EchoNext mini-model, Columbia inpatients to Columbia outpatients

Calibrated on 1903 inpatient ECGs and applied unchanged to 1059 outpatients, the per-label thresholds cover 72.7% of outpatients with structural heart disease (composite) where 90% was asked, leave 71.1% of those without it unflagged, and send 32.3% to a human. A patient without it whom the thresholds send to a human counts below as covered, which makes the healthy covered (98.2%) more than the healthy left unflagged. The composite's prevalence falls from 53.2% to 25.6%.

| | Source | Target |
|---|---|---|
| Cohort | EchoNext validation split, inpatients | EchoNext test split, outpatients |
| ECGs, one per patient | 1903 | 1059 |
| Composite prevalence | 53.2% | 25.6% |
| LVEF ≤45% prevalence | 24.5% | 6.6% |

Model: the published EchoNext mini-model on its own weights, nothing refitted. Input: the tracing as distributed, 12 leads at 250 Hz, unit z-score. Nothing is refitted on the target except on the ladder at the end of the page.

## Coverage per label at the 90% level

Ill covered is the share of patients with the finding whose decision includes it: the sensitivity for the plain threshold, the coverage of the positive class for the two conformal schemes. A per-label threshold that the calibration positives cannot certify flags everyone, and the row then shows the whole target flagged. The plain threshold and the per-label threshold of the ill are the same calibration quantile, so their columns agree; the per-label scheme adds a threshold for the healthy, and with it the share sent to a human.

| Label | Prevalence, source | Prevalence, target | Ill in target | Ill covered, plain | Ill covered, pooled | Ill covered, per-label | Healthy covered, per-label, sent to a human included | Sent to a human, per-label | AUROC, target |
|---|---|---|---|---|---|---|---|---|---|
| LVEF ≤45% | 24.5% | 6.6% | 70 | 80.0% | 85.7% | 80.0% [69.2%, 87.7%] | 97.8% | 20.8% | 0.873 |
| LV wall ≥1.3 cm | 22.1% | 13.8% | 146 | 85.6% | 71.9% | 85.6% [79.0%, 90.4%] | 97.8% | 47.8% | 0.767 |
| Aortic stenosis, moderate+ | 6.8% | 4.8% | 51 | 90.2% | 96.1% | 90.2% [79.0%, 95.7%] | 95.5% | 24.0% | 0.911 |
| Aortic regurgitation, moderate+ | 1.7% | 0.4% | 4 | 75.0% | 75.0% | 75.0% [30.1%, 95.4%] | 97.5% | 17.8% | 0.796 |
| Mitral regurgitation, moderate+ | 7.7% | 2.8% | 30 | 63.3% | 70.0% | 63.3% [45.5%, 78.1%] | 98.2% | 19.6% | 0.736 |
| Tricuspid regurgitation, moderate+ | 9.1% | 2.5% | 26 | 76.9% | 80.8% | 76.9% [57.9%, 89.0%] | 98.7% | 18.5% | 0.836 |
| Pulmonary regurgitation, moderate+ | 0.6% | 0.0% | 0 | n/a | n/a | n/a | 98.4% | 41.1% | n/a |
| RV dysfunction, moderate+ | 11.5% | 1.9% | 20 | 80.0% | 85.0% | 80.0% [58.4%, 91.9%] | 98.5% | 17.4% | 0.831 |
| Pericardial effusion, moderate+ | 1.7% | 0.2% | 2 | 100.0% | 100.0% | 100.0% [34.2%, 100.0%] | 99.0% | 36.7% | 0.862 |
| PASP ≥45 mmHg | 15.4% | 5.9% | 63 | 68.3% | 71.4% | 68.3% [56.0%, 78.4%] | 98.8% | 29.4% | 0.750 |
| TR velocity ≥3.2 m/s | 7.4% | 3.3% | 35 | 80.0% | 82.9% | 80.0% [64.1%, 90.0%] | 98.8% | 27.4% | 0.818 |
| Composite (any of the above) | 53.2% | 25.6% | 271 | 72.7% | 69.0% | 72.7% [67.1%, 77.7%] | 98.2% | 32.3% | 0.795 |

## Coverage of the ill inside groups, composite, per-label thresholds

| Group | | Ill | Ill covered [95% CI] |
|---|---|---|---|
| sex | female | 126 | 66.7% [58.1%, 74.3%] |
| sex | male | 145 | 77.9% [70.5%, 83.9%] |
| age | 18-49 | 30 | 53.3% [36.1%, 69.8%] |
| age | 50-64 | 69 | 56.5% [44.8%, 67.6%] |
| age | 65-79 | 117 | 76.1% [67.6%, 82.9%] |
| age | 80+ | 55 | 96.4% [87.7%, 99.0%] |
| race_ethnicity | asian | 11 | 72.7% [43.4%, 90.3%] |
| race_ethnicity | black | 39 | 74.4% [58.9%, 85.4%] |
| race_ethnicity | hispanic | 58 | 67.2% [54.4%, 77.9%] |
| race_ethnicity | other | 22 | 72.7% [51.8%, 86.8%] |
| race_ethnicity | unknown | 32 | 78.1% [61.2%, 89.0%] |
| race_ethnicity | white | 109 | 73.4% [64.4%, 80.8%] |

## Calibration of the composite probability

Slope 1 and intercept 0 are perfect calibration; the intercept is calibration-in-the-large on the logit scale.

| Test split, context | Prevalence | Mean predicted | Slope | Intercept | Brier |
|---|---|---|---|---|---|
| inpatients | 52.5% | 49.3% | 0.98 | 0.18 | 0.184 |
| emergency | 40.0% | 39.0% | 1.04 | 0.06 | 0.165 |
| outpatients | 25.6% | 28.0% | 1.04 | -0.16 | 0.141 |

Calibration curve on outpatients, ten bins of equal count:

| Predicted range | ECGs | Mean predicted | Observed [95% CI] |
|---|---|---|---|
| 2.8% to 7.3% | 106 | 6.0% | 7.5% [3.9%, 14.2%] |
| 7.4% to 10.0% | 106 | 8.6% | 7.5% [3.9%, 14.2%] |
| 10.0% to 12.4% | 106 | 11.1% | 11.3% [6.6%, 18.8%] |
| 12.5% to 15.3% | 106 | 13.7% | 9.4% [5.2%, 16.5%] |
| 15.3% to 18.8% | 106 | 17.1% | 17.0% [11.0%, 25.3%] |
| 18.8% to 24.3% | 106 | 21.4% | 17.9% [11.8%, 26.3%] |
| 24.4% to 34.2% | 106 | 28.3% | 18.9% [12.6%, 27.4%] |
| 34.2% to 47.9% | 106 | 41.4% | 31.1% [23.1%, 40.5%] |
| 48.0% to 65.0% | 106 | 56.3% | 54.7% [45.2%, 63.9%] |
| 65.2% to 92.8% | 105 | 76.6% | 81.0% [72.4%, 87.3%] |

## Net benefit of sending outpatients to echocardiography, composite

Net benefit is true positives per patient minus false positives per patient weighted by the odds of the threshold.

| Threshold | Model | Echo for all | Echo for none |
|---|---|---|---|
| 5% | 0.217 | 0.217 | 0.000 |
| 10% | 0.179 | 0.173 | 0.000 |
| 20% | 0.132 | 0.070 | 0.000 |
| 30% | 0.109 | -0.063 | 0.000 |
| 40% | 0.086 | -0.240 | 0.000 |
| 50% | 0.069 | -0.488 | 0.000 |

## Positive predictive value at the plain threshold

The source column is what a buyer computes from the sensitivity and specificity measured at the source, at the target's prevalence; the target column is observed.

| Label | Prevalence, target | Sensitivity, source / target | Specificity, source / target | PPV from source | PPV observed | False alerts per ill patient found, from source / observed |
|---|---|---|---|---|---|---|
| Composite (any of the above) | 25.6% | 90.1% / 72.7% | 38.5% / 71.1% | 33.5% | 46.4% | 2.0 / 1.2 |
| LVEF ≤45% | 6.6% | 90.4% / 80.0% | 52.2% / 78.8% | 11.8% | 21.1% | 7.5 / 3.8 |

## Target labels that repair the threshold and the calibration

The target is cut once by patient into a pool and an evaluation half. Rung 0 spends the source thresholds; rung n refits the per-label thresholds and an intercept shift on n ECGs drawn from the pool, over repeated draws, all read on the same evaluation half. One fixed half makes every draw share its luck, so the centiles below spread the labelled sample only. The study's report draws a new half in every draw and reads the refit on separate outpatients as well (REPORT.md section 3.6, SUPPLEMENT.md Tables S7 to S7c); those figures supersede this ladder's.

| Label | Target labels | Ill covered, mean [10th, 90th centile] | Healthy covered, sent to a human included | Draws that flag everyone | Absolute intercept after shift |
|---|---|---|---|---|---|
| Composite (any of the above) | 0 | 73.0% [73.0%, 73.0%] | 97.7% | 0.0% | 0.16 |
| Composite (any of the above) | 25 | 99.2% [99.3%, 100.0%] | 91.2% | 88.0% | 0.45 |
| Composite (any of the above) | 50 | 96.8% [93.5%, 100.0%] | 91.5% | 21.0% | 0.31 |
| Composite (any of the above) | 100 | 97.6% [94.9%, 99.3%] | 90.1% | 0.5% | 0.20 |
| Composite (any of the above) | 200 | 97.3% [94.9%, 99.3%] | 89.7% | 0.0% | 0.13 |
| Composite (any of the above) | 400 | 97.3% [94.9%, 99.3%] | 89.2% | 0.0% | 0.05 |
| LVEF ≤45% | 0 | 81.2% [81.2%, 81.2%] | 97.8% | 0.0% | 1.95 |
| LVEF ≤45% | 25 | 100.0% [100.0%, 100.0%] | 89.3% | 100.0% | 0.86 |
| LVEF ≤45% | 50 | 100.0% [100.0%, 100.0%] | 89.5% | 100.0% | 0.53 |
| LVEF ≤45% | 100 | 98.7% [100.0%, 100.0%] | 88.3% | 73.0% | 0.39 |
| LVEF ≤45% | 200 | 98.3% [100.0%, 100.0%] | 88.6% | 3.0% | 0.34 |
| LVEF ≤45% | 400 | 99.7% [100.0%, 100.0%] | 88.4% | 0.0% | 0.30 |

## Positive predictive value recomputed by Bayes' rule against observed

At the plain threshold, the recipe carries the source's sensitivity and specificity to the target's true prevalence. The gap is recomputed minus observed, in percentage points, with the 2.5th and 97.5th centiles of 2,000 redraws of both tables; the observed PPV carries its 95% Wilson interval. If only the prevalence had changed, the mean likelihood ratio among the healthy would stay at its source value, close to 1.

| Label | Prevalence, target | Flagged | PPV recomputed | PPV observed | Gap, points | Mean likelihood ratio of the healthy, source / target |
|---|---|---|---|---|---|---|
| LVEF ≤45% | 6.6% | 266 | 11.8% | 21.1% [16.6%, 26.3%] | -9.3 [-12.6, -6.2] | 1.00 / 0.36 |
| LV wall ≥1.3 cm | 13.8% | 541 | 15.3% | 23.1% [19.8%, 26.8%] | -7.8 [-10.0, -5.9] | 1.00 / 0.60 |
| Aortic stenosis, moderate+ | 4.8% | 325 | 8.8% | 14.2% [10.8%, 18.4%] | -5.3 [-7.4, -3.3] | 0.96 / 0.56 |
| Aortic regurgitation, moderate+ | 0.4% | 214 | 0.8% | 1.4% [0.5%, 4.0%] | -0.6 [-1.7, +0.3] | 1.00 / 0.38 |
| Mitral regurgitation, moderate+ | 2.8% | 228 | 5.2% | 8.3% [5.4%, 12.6%] | -3.2 [-5.8, -0.8] | 1.00 / 0.38 |
| Tricuspid regurgitation, moderate+ | 2.5% | 215 | 3.9% | 9.3% [6.1%, 13.9%] | -5.4 [-8.2, -2.7] | 1.01 / 0.35 |
| Pulmonary regurgitation, moderate+ | 0.0% | 452 | n/a | n/a | n/a | n/a |
| RV dysfunction, moderate+ | 1.9% | 206 | 3.0% | 7.8% [4.8%, 12.2%] | -4.7 [-7.5, -2.2] | 0.98 / 0.29 |
| Pericardial effusion, moderate+ | 0.2% | 400 | 0.2% | 0.5% [0.1%, 1.8%] | -0.3 [-0.7, +0.0] | 1.00 / 0.62 |
| PASP ≥45 mmHg | 5.9% | 330 | 8.1% | 13.0% [9.8%, 17.1%] | -5.0 [-7.5, -2.6] | 1.00 / 0.45 |
| TR velocity ≥3.2 m/s | 3.3% | 306 | 4.5% | 9.2% [6.4%, 12.9%] | -4.6 [-6.8, -2.6] | 1.01 / 0.42 |
| Composite (any of the above) | 25.6% | 425 | 33.5% | 46.4% [41.7%, 51.1%] | -12.8 [-16.5, -9.3] | 0.96 / 0.41 |

## Three repairs judged on net benefit

Read on one half of the outpatients, cut by patient. Net benefit is in true positives per 100 patients. The prevalence correction uses no target label; the recalibration fits an intercept and a slope on 100 labelled ECGs from the other half, averaged over 200 draws (10th to 90th centile in brackets); the per-label sets use no target label and send their abstentions to a human, whose decision is not modelled, so they show two values, abstentions cleared and abstentions referred.

| Label | Threshold | As delivered | Prevalence corrected | Recalibrated on 100 | Per-label sets, cleared / referred | Treat all |
|---|---|---|---|---|---|---|
| LVEF ≤45% | 5% | 1.8 | 0.0 | 4.0 [3.9, 4.2] | 1.4 / 4.1 | 1.5 |
| LVEF ≤45% | 10% | -0.4 | 0.0 | 3.1 [2.8, 3.4] | 1.3 / 2.9 | -4.0 |
| LVEF ≤45% | 20% | -1.7 | 0.0 | 1.7 [1.1, 2.1] | 1.1 / 0.2 | -17.0 |
| Composite (any of the above) | 5% | 22.9 | 0.0 | 21.9 [20.5, 22.9] | 5.6 / 17.8 | 22.9 |
| Composite (any of the above) | 10% | 18.5 | 0.0 | 17.8 [17.1, 18.6] | 5.5 / 16.6 | 18.7 |
| Composite (any of the above) | 20% | 13.5 | 0.0 | 13.6 [13.2, 14.0] | 5.4 / 13.7 | 8.5 |

| Label | Prevalence, evaluation half | Estimated without labels | Sent to a human by the per-label sets |
|---|---|---|---|
| LVEF ≤45% | 6.4% | 0.0% | 21.5% |
| Composite (any of the above) | 26.8% | 0.0% | 32.6% |

## Cells filled by other tasks

- Distance without labels between source and target ECGs: empty, filled by T-068.
- The same pair at a second hospital, Columbia to Beth Israel (MIMIC-IV-Echo): empty, filled by T-065, ambitious version, after PhysioNet credentialing.

# Transfer report: the study's ResNet frozen at random initialisation, with logistic probes, Columbia inpatients to Columbia outpatients

Calibrated on 1903 inpatient ECGs and applied unchanged to 1059 outpatients, the per-label thresholds cover 78.6% of outpatients with structural heart disease (composite) where 90% was asked, leave 57.4% of those without it unflagged, and send 44.2% to a human. A patient without it whom the thresholds send to a human counts below as covered, which makes the healthy covered (97.2%) more than the healthy left unflagged. The composite's prevalence falls from 53.2% to 25.6%.

| | Source | Target |
|---|---|---|
| Cohort | EchoNext validation split, inpatients | EchoNext test split, outpatients |
| ECGs, one per patient | 1903 | 1059 |
| Composite prevalence | 53.2% | 25.6% |
| LVEF ≤45% prevalence | 24.5% | 6.6% |

Model: study ResNet1d frozen at its seeded initialisation, one probe per label. Input: the tracing as distributed, 12 leads at 250 Hz, unit z-score. Nothing is refitted on the target except on the ladder at the end of the page.

## Coverage per label at the 90% level

Ill covered is the share of patients with the finding whose decision includes it: the sensitivity for the plain threshold, the coverage of the positive class for the two conformal schemes. A per-label threshold that the calibration positives cannot certify flags everyone, and the row then shows the whole target flagged. The plain threshold and the per-label threshold of the ill are the same calibration quantile, so their columns agree; the per-label scheme adds a threshold for the healthy, and with it the share sent to a human.

| Label | Prevalence, source | Prevalence, target | Ill in target | Ill covered, plain | Ill covered, pooled | Ill covered, per-label | Healthy covered, per-label, sent to a human included | Sent to a human, per-label | AUROC, target |
|---|---|---|---|---|---|---|---|---|---|
| LVEF ≤45% | 24.5% | 6.6% | 70 | 85.7% | 54.3% | 85.7% [75.7%, 92.1%] | 97.2% | 29.3% | 0.841 |
| LV wall ≥1.3 cm | 22.1% | 13.8% | 146 | 86.3% | 52.7% | 86.3% [79.8%, 91.0%] | 95.4% | 51.7% | 0.758 |
| Aortic stenosis, moderate+ | 6.8% | 4.8% | 51 | 94.1% | 0.0% | 94.1% [84.1%, 98.0%] | 92.4% | 55.6% | 0.830 |
| Aortic regurgitation, moderate+ | 1.7% | 0.4% | 4 | 100.0% | 0.0% | 100.0% [51.0%, 100.0%] | 95.7% | 80.7% | 0.855 |
| Mitral regurgitation, moderate+ | 7.7% | 2.8% | 30 | 80.0% | 0.0% | 80.0% [62.7%, 90.5%] | 97.5% | 46.6% | 0.712 |
| Tricuspid regurgitation, moderate+ | 9.1% | 2.5% | 26 | 76.9% | 3.8% | 76.9% [57.9%, 89.0%] | 97.3% | 45.1% | 0.704 |
| Pulmonary regurgitation, moderate+ | 0.6% | 0.0% | 0 | n/a | n/a | n/a | 95.9% | 83.3% | n/a |
| RV dysfunction, moderate+ | 11.5% | 1.9% | 20 | 75.0% | 15.0% | 75.0% [53.1%, 88.8%] | 96.6% | 28.4% | 0.803 |
| Pericardial effusion, moderate+ | 1.7% | 0.2% | 2 | 100.0% | 0.0% | 100.0% [34.2%, 100.0%] | 97.2% | 80.5% | 0.886 |
| PASP ≥45 mmHg | 15.4% | 5.9% | 63 | 81.0% | 17.5% | 81.0% [69.6%, 88.8%] | 96.8% | 63.6% | 0.678 |
| TR velocity ≥3.2 m/s | 7.4% | 3.3% | 35 | 74.3% | 0.0% | 74.3% [57.9%, 85.8%] | 97.3% | 40.9% | 0.704 |
| Composite (any of the above) | 53.2% | 25.6% | 271 | 78.6% | 82.7% | 78.6% [73.3%, 83.1%] | 97.2% | 44.2% | 0.758 |

## Coverage of the ill inside groups, composite, per-label thresholds

| Group | | Ill | Ill covered [95% CI] |
|---|---|---|---|
| sex | female | 126 | 73.8% [65.5%, 80.7%] |
| sex | male | 145 | 82.8% [75.8%, 88.0%] |
| age | 18-49 | 30 | 70.0% [52.1%, 83.3%] |
| age | 50-64 | 69 | 68.1% [56.4%, 77.9%] |
| age | 65-79 | 117 | 82.1% [74.1%, 88.0%] |
| age | 80+ | 55 | 89.1% [78.2%, 94.9%] |
| race_ethnicity | asian | 11 | 81.8% [52.3%, 94.9%] |
| race_ethnicity | black | 39 | 82.1% [67.3%, 91.0%] |
| race_ethnicity | hispanic | 58 | 79.3% [67.2%, 87.7%] |
| race_ethnicity | other | 22 | 72.7% [51.8%, 86.8%] |
| race_ethnicity | unknown | 32 | 78.1% [61.2%, 89.0%] |
| race_ethnicity | white | 109 | 78.0% [69.3%, 84.7%] |

## Calibration of the composite probability

Slope 1 and intercept 0 are perfect calibration; the intercept is calibration-in-the-large on the logit scale.

| Test split, context | Prevalence | Mean predicted | Slope | Intercept | Brier |
|---|---|---|---|---|---|
| inpatients | 52.5% | 54.6% | 0.98 | -0.11 | 0.192 |
| emergency | 40.0% | 46.1% | 0.96 | -0.33 | 0.182 |
| outpatients | 25.6% | 38.9% | 1.08 | -0.74 | 0.173 |

Calibration curve on outpatients, ten bins of equal count:

| Predicted range | ECGs | Mean predicted | Observed [95% CI] |
|---|---|---|---|
| 4.9% to 17.3% | 106 | 13.8% | 10.4% [5.9%, 17.6%] |
| 17.4% to 21.9% | 106 | 19.9% | 8.5% [4.5%, 15.4%] |
| 22.0% to 26.4% | 106 | 24.1% | 14.2% [8.8%, 22.0%] |
| 26.4% to 30.3% | 106 | 28.5% | 12.3% [7.3%, 19.9%] |
| 30.4% to 34.5% | 106 | 32.6% | 13.2% [8.0%, 21.0%] |
| 34.5% to 39.7% | 106 | 36.7% | 14.2% [8.8%, 22.0%] |
| 39.7% to 45.2% | 106 | 42.4% | 30.2% [22.3%, 39.5%] |
| 45.2% to 54.4% | 106 | 49.5% | 41.5% [32.6%, 51.0%] |
| 54.5% to 68.8% | 106 | 61.6% | 42.5% [33.5%, 52.0%] |
| 69.0% to 99.6% | 105 | 80.8% | 69.5% [60.2%, 77.5%] |

## Net benefit of sending outpatients to echocardiography, composite

Net benefit is true positives per patient minus false positives per patient weighted by the odds of the threshold.

| Threshold | Model | Echo for all | Echo for none |
|---|---|---|---|
| 5% | 0.217 | 0.217 | 0.000 |
| 10% | 0.174 | 0.173 | 0.000 |
| 20% | 0.091 | 0.070 | 0.000 |
| 30% | 0.041 | -0.063 | 0.000 |
| 40% | 0.042 | -0.240 | 0.000 |
| 50% | 0.020 | -0.488 | 0.000 |

## Positive predictive value at the plain threshold

The source column is what a buyer computes from the sensitivity and specificity measured at the source, at the target's prevalence; the target column is observed.

| Label | Prevalence, target | Sensitivity, source / target | Specificity, source / target | PPV from source | PPV observed | False alerts per ill patient found, from source / observed |
|---|---|---|---|---|---|---|
| Composite (any of the above) | 25.6% | 90.1% / 78.6% | 32.0% / 57.4% | 31.3% | 38.8% | 2.2 / 1.6 |
| LVEF ≤45% | 6.6% | 90.4% / 85.7% | 47.4% / 68.8% | 10.8% | 16.3% | 8.2 / 5.1 |

## Target labels that repair the threshold and the calibration

The target is cut once by patient into a pool and an evaluation half. Rung 0 spends the source thresholds; rung n refits the per-label thresholds and an intercept shift on n ECGs drawn from the pool, over repeated draws, all read on the same evaluation half. One fixed half makes every draw share its luck, so the centiles below spread the labelled sample only. The study's report draws a new half in every draw and reads the refit on separate outpatients as well (REPORT.md section 3.7, SUPPLEMENT.md Tables S7 to S7c); those figures supersede this ladder's.

| Label | Target labels | Ill covered, mean [10th, 90th centile] | Healthy covered, sent to a human included | Draws that flag everyone | Absolute intercept after shift |
|---|---|---|---|---|---|
| Composite (any of the above) | 0 | 79.6% [79.6%, 79.6%] | 96.9% | 0.0% | 0.76 |
| Composite (any of the above) | 25 | 98.5% [94.2%, 100.0%] | 91.6% | 86.5% | 0.43 |
| Composite (any of the above) | 50 | 90.7% [81.8%, 100.0%] | 90.3% | 10.5% | 0.31 |
| Composite (any of the above) | 100 | 88.8% [83.9%, 94.2%] | 89.8% | 0.0% | 0.18 |
| Composite (any of the above) | 200 | 87.3% [84.7%, 89.1%] | 89.2% | 0.0% | 0.12 |
| Composite (any of the above) | 400 | 86.7% [85.4%, 87.6%] | 88.9% | 0.0% | 0.05 |
| LVEF ≤45% | 0 | 87.5% [87.5%, 87.5%] | 97.4% | 0.0% | 1.11 |
| LVEF ≤45% | 25 | 100.0% [100.0%, 100.0%] | 91.3% | 100.0% | 0.67 |
| LVEF ≤45% | 50 | 100.0% [100.0%, 100.0%] | 90.9% | 100.0% | 0.52 |
| LVEF ≤45% | 100 | 97.5% [93.8%, 100.0%] | 88.8% | 73.0% | 0.36 |
| LVEF ≤45% | 200 | 92.9% [93.8%, 93.8%] | 88.7% | 3.0% | 0.32 |
| LVEF ≤45% | 400 | 93.7% [93.8%, 93.8%] | 88.5% | 0.0% | 0.28 |

## Positive predictive value recomputed by Bayes' rule against observed

At the plain threshold, the recipe carries the source's sensitivity and specificity to the target's true prevalence. The gap is recomputed minus observed, in percentage points, with the 2.5th and 97.5th centiles of 2,000 redraws of both tables; the observed PPV carries its 95% Wilson interval. If only the prevalence had changed, the mean likelihood ratio among the healthy would stay at its source value, close to 1.

| Label | Prevalence, target | Flagged | PPV recomputed | PPV observed | Gap, points | Mean likelihood ratio of the healthy, source / target |
|---|---|---|---|---|---|---|
| LVEF ≤45% | 6.6% | 369 | 10.8% | 16.3% [12.8%, 20.4%] | -5.4 [-7.6, -3.3] | 1.15 / 0.53 |
| LV wall ≥1.3 cm | 13.8% | 615 | 16.4% | 20.5% [17.5%, 23.9%] | -4.1 [-5.7, -2.4] | 1.02 / 0.77 |
| Aortic stenosis, moderate+ | 4.8% | 686 | 6.9% | 7.0% [5.3%, 9.2%] | -0.1 [-0.8, +0.6] | 1.01 / 0.92 |
| Aortic regurgitation, moderate+ | 0.4% | 902 | 0.4% | 0.4% [0.2%, 1.1%] | -0.0 [-0.1, -0.0] | 0.99 / 0.80 |
| Mitral regurgitation, moderate+ | 2.8% | 524 | 4.2% | 4.6% [3.1%, 6.7%] | -0.4 [-1.3, +0.5] | 1.03 / 0.68 |
| Tricuspid regurgitation, moderate+ | 2.5% | 512 | 3.3% | 3.9% [2.5%, 6.0%] | -0.7 [-1.5, +0.3] | 1.02 / 0.66 |
| Pulmonary regurgitation, moderate+ | 0.0% | 925 | n/a | n/a | n/a | n/a |
| RV dysfunction, moderate+ | 1.9% | 342 | 3.0% | 4.4% [2.7%, 7.1%] | -1.4 [-2.7, -0.2] | 1.11 / 0.50 |
| Pericardial effusion, moderate+ | 0.2% | 884 | 0.2% | 0.2% [0.1%, 0.8%] | -0.0 [-0.1, +0.0] | 1.00 / 0.81 |
| PASP ≥45 mmHg | 5.9% | 711 | 6.5% | 7.2% [5.5%, 9.3%] | -0.7 [-1.6, +0.2] | 1.00 / 0.72 |
| TR velocity ≥3.2 m/s | 3.3% | 464 | 4.5% | 5.6% [3.9%, 8.1%] | -1.1 [-2.3, +0.1] | 1.01 / 0.71 |
| Composite (any of the above) | 25.6% | 549 | 31.3% | 38.8% [34.8%, 42.9%] | -7.5 [-10.1, -4.8] | 2.62 / 0.57 |

## Three repairs judged on net benefit

Read on one half of the outpatients, cut by patient. Net benefit is in true positives per 100 patients. The prevalence correction uses no target label; the recalibration fits an intercept and a slope on 100 labelled ECGs from the other half, averaged over 200 draws (10th to 90th centile in brackets); the per-label sets use no target label and send their abstentions to a human, whose decision is not modelled, so they show two values, abstentions cleared and abstentions referred.

| Label | Threshold | As delivered | Prevalence corrected | Recalibrated on 100 | Per-label sets, cleared / referred | Treat all |
|---|---|---|---|---|---|---|
| LVEF ≤45% | 5% | 2.5 | 0.0 | 3.4 [3.0, 3.8] | 3.1 / 3.9 | 1.5 |
| LVEF ≤45% | 10% | 1.0 | 0.0 | 2.4 [2.1, 2.8] | 2.9 / 2.1 | -4.0 |
| LVEF ≤45% | 20% | 0.1 | 0.0 | 2.0 [1.2, 2.5] | 2.5 / -2.0 | -17.0 |
| Composite (any of the above) | 5% | 22.9 | 0.0 | 22.4 [22.0, 22.9] | 5.8 / 18.0 | 22.9 |
| Composite (any of the above) | 10% | 18.8 | 0.0 | 17.7 [16.5, 18.7] | 5.7 / 16.2 | 18.7 |
| Composite (any of the above) | 20% | 10.3 | 0.0 | 12.5 [11.4, 13.2] | 5.4 / 12.0 | 8.5 |

| Label | Prevalence, evaluation half | Estimated without labels | Sent to a human by the per-label sets |
|---|---|---|---|
| LVEF ≤45% | 6.4% | 0.0% | 29.6% |
| Composite (any of the above) | 26.8% | 0.0% | 42.6% |

## Cells filled by other tasks

- Distance without labels between source and target ECGs: empty, filled by T-068.
- The same pair at a second hospital, Columbia to Beth Israel (MIMIC-IV-Echo): empty, filled by T-065, ambitious version, after PhysioNet credentialing.

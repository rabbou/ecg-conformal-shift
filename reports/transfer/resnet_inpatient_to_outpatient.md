# Transfer report: the study's ResNet, trained on EchoNext, Columbia inpatients to Columbia outpatients

Calibrated on 1903 inpatient ECGs and applied unchanged to 1059 outpatients, the per-label thresholds cover 71.6% of outpatients with structural heart disease (composite) where 90% was asked, leave 71.1% of those without it unflagged, and send 29.4% to a human. A patient without it whom the thresholds send to a human counts below as covered, which makes the healthy covered (98.1%) more than the healthy left unflagged. The composite's prevalence falls from 53.2% to 25.6%.

| | Source | Target |
|---|---|---|
| Cohort | EchoNext validation split, inpatients | EchoNext test split, outpatients |
| ECGs, one per patient | 1903 | 1059 |
| Composite prevalence | 53.2% | 25.6% |
| LVEF ≤45% prevalence | 24.5% | 6.6% |

Model: study ResNet1d, 12 sigmoid outputs, trained from scratch on EchoNext train. Input: the tracing as distributed, 12 leads at 250 Hz, unit z-score. Nothing is refitted on the target except on the ladder at the end of the page.

## Coverage per label at the 90% level

Ill covered is the share of patients with the finding whose decision includes it: the sensitivity for the plain threshold, the coverage of the positive class for the two conformal schemes. A per-label threshold that the calibration positives cannot certify flags everyone, and the row then shows the whole target flagged. The plain threshold and the per-label threshold of the ill are the same calibration quantile, so their columns agree; the per-label scheme adds a threshold for the healthy, and with it the share sent to a human.

| Label | Prevalence, source | Prevalence, target | Ill in target | Ill covered, plain | Ill covered, pooled | Ill covered, per-label | Healthy covered, per-label, sent to a human included | Sent to a human, per-label | AUROC, target |
|---|---|---|---|---|---|---|---|---|---|
| LVEF ≤45% | 24.5% | 6.6% | 70 | 87.1% | 60.0% | 87.1% [77.3%, 93.1%] | 97.2% | 12.4% | 0.929 |
| LV wall ≥1.3 cm | 22.1% | 13.8% | 146 | 74.7% | 48.6% | 74.7% [67.0%, 81.0%] | 97.3% | 35.9% | 0.779 |
| Aortic stenosis, moderate+ | 6.8% | 4.8% | 51 | 96.1% | 0.0% | 96.1% [86.8%, 98.9%] | 94.2% | 30.5% | 0.907 |
| Aortic regurgitation, moderate+ | 1.7% | 0.4% | 4 | 100.0% | 0.0% | 100.0% [51.0%, 100.0%] | 97.1% | 19.5% | 0.902 |
| Mitral regurgitation, moderate+ | 7.7% | 2.8% | 30 | 70.0% | 3.3% | 70.0% [52.1%, 83.3%] | 98.3% | 23.3% | 0.759 |
| Tricuspid regurgitation, moderate+ | 9.1% | 2.5% | 26 | 69.2% | 15.4% | 69.2% [50.0%, 83.5%] | 97.3% | 16.4% | 0.842 |
| Pulmonary regurgitation, moderate+ | 0.6% | 0.0% | 0 | n/a | n/a | n/a | 97.8% | 85.7% | n/a |
| RV dysfunction, moderate+ | 11.5% | 1.9% | 20 | 75.0% | 35.0% | 75.0% [53.1%, 88.8%] | 98.0% | 11.1% | 0.872 |
| Pericardial effusion, moderate+ | 1.7% | 0.2% | 2 | 100.0% | 0.0% | 100.0% [34.2%, 100.0%] | 99.1% | 35.0% | 0.985 |
| PASP ≥45 mmHg | 15.4% | 5.9% | 63 | 73.0% | 34.9% | 73.0% [61.0%, 82.4%] | 97.0% | 30.2% | 0.767 |
| TR velocity ≥3.2 m/s | 7.4% | 3.3% | 35 | 74.3% | 11.4% | 74.3% [57.9%, 85.8%] | 97.7% | 19.8% | 0.829 |
| Composite (any of the above) | 53.2% | 25.6% | 271 | 71.6% | 74.5% | 71.6% [65.9%, 76.6%] | 98.1% | 29.4% | 0.805 |

## Coverage of the ill inside groups, composite, per-label thresholds

| Group | | Ill | Ill covered [95% CI] |
|---|---|---|---|
| sex | female | 126 | 61.1% [52.4%, 69.2%] |
| sex | male | 145 | 80.7% [73.5%, 86.3%] |
| age | 18-49 | 30 | 53.3% [36.1%, 69.8%] |
| age | 50-64 | 69 | 59.4% [47.6%, 70.2%] |
| age | 65-79 | 117 | 76.1% [67.6%, 82.9%] |
| age | 80+ | 55 | 87.3% [76.0%, 93.7%] |
| race_ethnicity | asian | 11 | 81.8% [52.3%, 94.9%] |
| race_ethnicity | black | 39 | 71.8% [56.2%, 83.5%] |
| race_ethnicity | hispanic | 58 | 70.7% [58.0%, 80.8%] |
| race_ethnicity | other | 22 | 72.7% [51.8%, 86.8%] |
| race_ethnicity | unknown | 32 | 71.9% [54.6%, 84.4%] |
| race_ethnicity | white | 109 | 70.6% [61.5%, 78.4%] |

## Calibration of the composite probability

Slope 1 and intercept 0 are perfect calibration; the intercept is calibration-in-the-large on the logit scale.

| Test split, context | Prevalence | Mean predicted | Slope | Intercept | Brier |
|---|---|---|---|---|---|
| inpatients | 52.5% | 56.2% | 0.74 | -0.25 | 0.180 |
| emergency | 40.0% | 43.3% | 0.80 | -0.24 | 0.161 |
| outpatients | 25.6% | 30.8% | 0.82 | -0.41 | 0.138 |

Calibration curve on outpatients, ten bins of equal count:

| Predicted range | ECGs | Mean predicted | Observed [95% CI] |
|---|---|---|---|
| 1.9% to 5.6% | 106 | 4.1% | 6.6% [3.2%, 13.0%] |
| 5.6% to 7.9% | 106 | 6.8% | 10.4% [5.9%, 17.6%] |
| 7.9% to 10.9% | 106 | 9.4% | 6.6% [3.2%, 13.0%] |
| 11.0% to 14.5% | 106 | 12.5% | 12.3% [7.3%, 19.9%] |
| 14.5% to 18.5% | 106 | 16.2% | 16.0% [10.3%, 24.2%] |
| 18.5% to 26.5% | 106 | 22.5% | 20.8% [14.1%, 29.4%] |
| 26.5% to 37.3% | 106 | 31.4% | 17.9% [11.8%, 26.3%] |
| 37.4% to 56.6% | 106 | 46.7% | 22.6% [15.7%, 31.5%] |
| 56.6% to 81.7% | 106 | 67.3% | 55.7% [46.2%, 64.8%] |
| 81.8% to 99.8% | 105 | 91.6% | 87.6% [80.0%, 92.6%] |

## Net benefit of sending outpatients to echocardiography, composite

Net benefit is true positives per patient minus false positives per patient weighted by the odds of the threshold.

| Threshold | Model | Echo for all | Echo for none |
|---|---|---|---|
| 5% | 0.216 | 0.217 | 0.000 |
| 10% | 0.181 | 0.173 | 0.000 |
| 20% | 0.129 | 0.070 | 0.000 |
| 30% | 0.100 | -0.063 | 0.000 |
| 40% | 0.080 | -0.240 | 0.000 |
| 50% | 0.074 | -0.488 | 0.000 |

## Positive predictive value at the plain threshold

The source column is what a buyer computes from the sensitivity and specificity measured at the source, at the target's prevalence; the target column is observed.

| Label | Prevalence, target | Sensitivity, source / target | Specificity, source / target | PPV from source | PPV observed | False alerts per ill patient found, from source / observed |
|---|---|---|---|---|---|---|
| Composite (any of the above) | 25.6% | 90.1% / 71.6% | 40.4% / 71.1% | 34.2% | 46.0% | 1.9 / 1.2 |
| LVEF ≤45% | 6.6% | 90.4% / 87.1% | 62.0% / 86.8% | 14.4% | 31.8% | 5.9 / 2.1 |

## Target labels that repair the threshold and the calibration

The target is cut once by patient into a pool and an evaluation half. Rung 0 spends the source thresholds; rung n refits the per-label thresholds and an intercept shift on n ECGs drawn from the pool, over repeated draws, all read on the same evaluation half. One fixed half makes every draw share its luck, so the centiles below spread the labelled sample only. The study's report draws a new half in every draw and reads the refit on separate outpatients as well (REPORT.md section 3.7, SUPPLEMENT.md Tables S7 to S7c); those figures supersede this ladder's.

| Label | Target labels | Ill covered, mean [10th, 90th centile] | Healthy covered, sent to a human included | Draws that flag everyone | Absolute intercept after shift |
|---|---|---|---|---|---|
| Composite (any of the above) | 0 | 69.3% [69.3%, 69.3%] | 97.5% | 0.0% | 0.37 |
| Composite (any of the above) | 25 | 98.9% [99.3%, 100.0%] | 91.6% | 86.5% | 0.53 |
| Composite (any of the above) | 50 | 94.5% [89.1%, 100.0%] | 91.0% | 10.5% | 0.36 |
| Composite (any of the above) | 100 | 93.4% [89.7%, 99.3%] | 90.5% | 0.0% | 0.24 |
| Composite (any of the above) | 200 | 92.7% [90.5%, 95.6%] | 90.1% | 0.0% | 0.16 |
| Composite (any of the above) | 400 | 92.1% [92.0%, 92.0%] | 90.0% | 0.0% | 0.08 |
| LVEF ≤45% | 0 | 81.2% [81.2%, 81.2%] | 97.6% | 0.0% | 0.61 |
| LVEF ≤45% | 25 | 100.0% [100.0%, 100.0%] | 92.7% | 100.0% | 0.66 |
| LVEF ≤45% | 50 | 100.0% [100.0%, 100.0%] | 92.1% | 100.0% | 0.59 |
| LVEF ≤45% | 100 | 96.3% [81.2%, 100.0%] | 91.0% | 73.0% | 0.36 |
| LVEF ≤45% | 200 | 90.5% [68.8%, 100.0%] | 90.9% | 3.0% | 0.24 |
| LVEF ≤45% | 400 | 90.0% [81.2%, 96.9%] | 90.8% | 0.0% | 0.11 |

## Positive predictive value recomputed by Bayes' rule against observed

At the plain threshold, the recipe carries the source's sensitivity and specificity to the target's true prevalence. The gap is recomputed minus observed, in percentage points, with the 2.5th and 97.5th centiles of 2,000 redraws of both tables; the observed PPV carries its 95% Wilson interval. If only the prevalence had changed, the mean likelihood ratio among the healthy would stay at its source value, close to 1.

| Label | Prevalence, target | Flagged | PPV recomputed | PPV observed | Gap, points | Mean likelihood ratio of the healthy, source / target |
|---|---|---|---|---|---|---|
| LVEF ≤45% | 6.6% | 192 | 14.4% | 31.8% [25.6%, 38.7%] | -17.3 [-22.4, -13.1] | 0.93 / 0.30 |
| LV wall ≥1.3 cm | 13.8% | 443 | 17.3% | 24.6% [20.8%, 28.8%] | -7.3 [-10.0, -4.8] | 1.00 / 0.57 |
| Aortic stenosis, moderate+ | 4.8% | 408 | 7.7% | 12.0% [9.2%, 15.5%] | -4.3 [-6.0, -2.9] | 1.01 / 0.66 |
| Aortic regurgitation, moderate+ | 0.4% | 238 | 0.7% | 1.7% [0.7%, 4.2%] | -1.0 [-2.1, -0.2] | 1.01 / 0.41 |
| Mitral regurgitation, moderate+ | 2.8% | 268 | 4.9% | 7.8% [5.2%, 11.7%] | -2.9 [-5.0, -0.8] | 1.01 / 0.43 |
| Tricuspid regurgitation, moderate+ | 2.5% | 210 | 4.3% | 8.6% [5.5%, 13.1%] | -4.2 [-7.1, -1.7] | 1.01 / 0.39 |
| Pulmonary regurgitation, moderate+ | 0.0% | 931 | n/a | n/a | n/a | n/a |
| RV dysfunction, moderate+ | 1.9% | 149 | 3.5% | 10.1% [6.2%, 15.9%] | -6.5 [-10.4, -3.1] | 0.97 / 0.29 |
| Pericardial effusion, moderate+ | 0.2% | 381 | 0.2% | 0.5% [0.1%, 1.9%] | -0.3 [-0.8, +0.0] | 1.00 / 0.46 |
| PASP ≥45 mmHg | 5.9% | 366 | 7.6% | 12.6% [9.6%, 16.4%] | -5.0 [-7.1, -2.8] | 0.99 / 0.45 |
| TR velocity ≥3.2 m/s | 3.3% | 243 | 5.1% | 10.7% [7.4%, 15.2%] | -5.6 [-8.4, -3.0] | 1.00 / 0.39 |
| Composite (any of the above) | 25.6% | 422 | 34.2% | 46.0% [41.3%, 50.7%] | -11.7 [-15.3, -8.3] | 0.95 / 0.37 |

## Three repairs judged on net benefit

Read on one half of the outpatients, cut by patient. Net benefit is in true positives per 100 patients. The prevalence correction uses no target label; the recalibration fits an intercept and a slope on 100 labelled ECGs from the other half, averaged over 200 draws (10th to 90th centile in brackets); the per-label sets use no target label and send their abstentions to a human, whose decision is not modelled, so they show two values, abstentions cleared and abstentions referred.

| Label | Threshold | As delivered | Prevalence corrected | Recalibrated on 100 | Per-label sets, cleared / referred | Treat all |
|---|---|---|---|---|---|---|
| LVEF ≤45% | 5% | 4.3 | 0.0 | 4.5 [4.3, 4.7] | 2.9 / 4.7 | 1.5 |
| LVEF ≤45% | 10% | 3.7 | 0.0 | 3.8 [3.2, 4.3] | 2.7 / 4.0 | -4.0 |
| LVEF ≤45% | 20% | 2.6 | 0.0 | 2.3 [2.1, 2.6] | 2.4 / 2.5 | -17.0 |
| Composite (any of the above) | 5% | 22.8 | 0.0 | 21.8 [20.4, 23.0] | 9.4 / 16.9 | 22.9 |
| Composite (any of the above) | 10% | 18.6 | 0.0 | 17.5 [16.0, 18.5] | 9.4 / 15.7 | 18.7 |
| Composite (any of the above) | 20% | 12.6 | 0.0 | 13.0 [12.7, 13.3] | 9.3 / 12.9 | 8.5 |

| Label | Prevalence, evaluation half | Estimated without labels | Sent to a human by the per-label sets |
|---|---|---|---|
| LVEF ≤45% | 6.4% | 0.0% | 10.9% |
| Composite (any of the above) | 26.8% | 0.1% | 28.1% |

## Cells filled by other tasks

- Distance without labels between source and target ECGs: empty, filled by T-068.
- The same pair at a second hospital, Columbia to Beth Israel (MIMIC-IV-Echo): empty, filled by T-065, ambitious version, after PhysioNet credentialing.

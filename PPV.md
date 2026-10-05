# A model's PPV, recomputed for your hospital, misses by five points and in either direction

A hospital that evaluates an ECG model gets an AUROC, a sensitivity and a specificity measured on someone else's patients. The usual advice, which I have given too, is to recompute the positive predictive value (PPV) at the local prevalence with Bayes' rule. The recipe is exact only if sensitivity and specificity carry over to the new patients unchanged. I tested it on 174 transfers of ECG models from one population to another, where the true PPV could be counted. Given the true local prevalence, the recipe missed the observed PPV by a median of 5.3 percentage points. It landed outside the observed PPV's 95% interval in 75% of the transfers, and it erred upward in some and downward in others. On 72 control pairs, where the target population was the source population, it missed by 0.3 points.

![PPV recomputed against PPV observed, and the gap against the change in specificity](results/figures/ppv_gap.png)

Left: each point is one model, one diagnosis, one pair of populations. Grey points are controls, where the target is drawn from the source population. Right: the same points, with the gap in percentage points against how far specificity moved.

## Bayes' rule holds only if the test performs the same on new patients

AUROC is a property of a model on a sample, and it does not move when the prevalence changes. PPV, the share of flagged patients who have the disease, depends on the prevalence, which is why AUROC alone cannot tell a clinic how many of its alerts will be real. The recipe fixes that by computing PPV = sens × prev / (sens × prev + (1 − spec) × (1 − prev)) with the site's own prevalence.

The hyperkalemia model validated by Harmon and colleagues at Mayo Clinic shows the recipe working. In the emergency department, where 1% of patients had a potassium above 6.0 mEq/L, the model reached 80% sensitivity and 80% specificity and a PPV of 3%. In intensive care, at 3% prevalence, it reached 82% and 82% and a PPV of 14% [1]. Carried from the emergency department to the ICU's prevalence, the recipe gives 12%, two points under what was observed, because sensitivity and specificity barely moved between the two units.

They often move. Ransohoff and Feinstein named the spectrum problem in 1978: a test's sensitivity and specificity depend on which ill and which healthy patients it is measured on [2]. Across 23 meta-analyses, Leeflang and colleagues found that sensitivity or specificity moved by up to 40 points between low- and high-prevalence studies of the same test, and that specificity tended to fall as prevalence rose [3]. In the EchoNext study, at a fixed 70% sensitivity, specificity at the external hospitals was 10 points lower than at Columbia [4]. None of this says by how much a recomputed PPV will miss at one hospital, so I measured it.

## One threshold per source, counted at the target

The scores come from two studies in this repository, on public ECG corpora. In each pair, one threshold is fitted on the source so that 90% of its ill patients are flagged, and it is applied unchanged to the target. The source's sensitivity and specificity at that threshold, with the target's true prevalence, give the recomputed PPV. The observed PPV is the share of flagged target patients who are ill. Giving the recipe the true prevalence is generous, since a buyer would have to estimate it.

| Pairs | Model | Diagnoses | Transfers summarised |
|---|---|---|---|
| Columbia inpatients to Columbia emergency and outpatients (EchoNext) | four: a network trained here, the published EchoNext mini-model, ECGFounder with logistic probes, a random-initialisation floor | eleven echocardiographic findings and their composite | 80 |
| Five corpora from Germany, China and the United States, each to the other four | a network trained on each source | sinus rhythm, atrial fibrillation, left and right bundle-branch block, first-degree AV block | 92 |
| PTB-XL (Germany) to Shandong and Chongqing (China) | a network trained on PTB-XL | myocardial infarction | 2 |

A transfer is summarised when its target holds at least 10 ill patients and its threshold flags at least 20, so that the observed PPV has an interval narrower than the gaps in question. Controls are the same models read on a held-out part of their own source: Columbia inpatients from the validation split to inpatients of the test split, and each corpus to its own test part.

## The recipe missed in both directions

| Pairs | Transfers | Median gap, points | Within two points | Recomputed outside the observed 95% interval | Recipe too high |
|---|---|---|---|---|---|
| Controls, same population | 72 | 0.3 | 86% | 1% | 40% |
| All transfers | 174 | 5.3 | 25% | 75% | 36% |
| Columbia, inpatients to emergency | 44 | 3.2 | 36% | 66% | 9% |
| Columbia, inpatients to outpatients | 36 | 4.8 | 19% | 81% | 0% |
| Five corpora, each to the other four | 92 | 10.9 | 22% | 76% | 62% |

The gap also matters relative to the PPV itself. In 57% of transfers the recomputed PPV was off by a quarter or more of the observed one, against 1% of controls.

At Columbia the recipe was too pessimistic in all 36 outpatient transfers. For an ejection fraction of 45% or less, it predicted that 14% of the outpatients flagged by the study's own network would have one. The observed figure was 32%, and the recipe implied 5.9 false alerts per patient found where there were 2.1. For the composite of eleven findings it predicted 34% and the observed figure was 46%. The cause is in the healthy outpatients. The threshold fitted on inpatients flags 60% of the healthy inpatients but only 29% of the healthy outpatients, so specificity rose from 40% to 71%. Healthy outpatients are healthier than healthy inpatients, which is the spectrum effect in its textbook form.

The gap comes with a diagnostic. If only the prevalence had changed between two populations, a calibrated model's likelihood ratio, averaged over the healthy patients, would be the same in both, close to 1. Among Columbia inpatients of the test split the median across all labels and models is 0.99. Among emergency patients it is 0.75, and among outpatients 0.49. The healthy ECGs themselves changed, and the three contexts rank the same way on this ratio as on the PPV gap.

At Chongqing the error ran the other way. The recipe predicted a PPV of 45% for infarction, and 26% was observed, because specificity fell from 81% to 58%. Chongqing's label is an acute infarction read on a coronary angiogram, while PTB-XL's positives are mostly the ECG pattern of an old one, so part of that fall is a different disease definition. At Shandong, at 1% prevalence, the recipe missed by under a point, 4.5% against 5.4%. That point is a sixth of the observed PPV: 21 false alerts per infarction found, where there were 17. Across the five-corpus rotation the worst miss was a left bundle-branch block at Shandong, predicted at 23% and observed at 88%.

Three predictions that need no local label did worse than the recipe given the true prevalence. The recipe at a prevalence estimated from the site's unlabelled ECGs missed by a median of 13.7 points. The model's own mean probability among the flagged missed by 8.2, and the same probability corrected to the estimated prevalence by 12.6.

## Net benefit decides whether to act on an alert

A low PPV does not make a model useless. Whether to act on an alert depends on what a false alert costs and what a miss costs, and decision curve analysis puts that trade-off in one number [5]. A clinician who would send a patient to echocardiography above a 10% probability of structural heart disease accepts nine normal studies for one abnormal one. Net benefit counts the true positives per patient minus the false positives weighted by those odds, and compares the model with sending everyone and sending no one.

![Net benefit of the model and of three repairs at three sites](results/figures/decision_curves.png)

At Columbia, for outpatients at a 10% threshold, the model as delivered gives 18.6 net true positives per 100 outpatients, against 18.7 for sending everyone. At this prevalence and threshold, an echocardiogram for every outpatient does as well as the model. From about 15% upward, the model beats both strategies. At Shandong, the model as delivered does worse than sending no one at every threshold from 2% upward, because its probabilities, fitted where a quarter of the patients had an infarction, are far too high where one in a hundred does.

## Of three repairs, only recalibration on local labels gained on average

A site can repair a transferred model in three ways. I applied all three to the same scores, read each on one half of the target cut by patient, and kept the other half as the pool the labelled repair could draw from.

The first uses no label: it estimates the site's prevalence from its unlabelled probabilities, by the maximum-likelihood method of Saerens and colleagues [6], after recalibrating the model on the source as Alexandari and colleagues advise [7], and moves every probability to that prevalence. The second recalibrates the model with a logistic intercept and slope fitted on 100 labelled site ECGs, or the intercept alone when the draw holds fewer than 10 of the rarer class [8]. The third is per-label conformal prediction from the study: one threshold for the ill and one for the healthy, fitted on the source, with patients between them sent to a human reader.

| At a 10% threshold, over 146 transfers | Mean change in net benefit against the model as delivered, per 1,000 patients | Transfers where it is the best rule |
|---|---|---|
| Prevalence corrected, no label | −15.6 | 15 |
| Recalibrated on 100 local labels | +11.0 | 35 |
| Per-label sets, abstentions cleared | −13.5 | 27 |
| Per-label sets, abstentions referred | −4.2 | 22 |
| Model as delivered | 0 | 39 |

The table counts a tie for every rule that reaches it. Recalibrating on 100 local labels was the only repair that gained on average, at each of the three thresholds of 5%, 10% and 20%: 6.9, 11.0 and 18.1 net true positives per 1,000 patients. It fails where 100 labels hold almost no ill patient. At Shandong a draw of 100 ECGs holds about one infarction, and the repair stays below the label-free correction there.

The label-free correction works when only the prevalence changed and fails when the healthy patients changed too. At Shandong it lifted the model from below the no-treatment line to the highest curve on the figure from a 6% threshold upward. At Columbia it estimated the outpatients' prevalence of structural heart disease at 0.1% where it is 27%, because healthy outpatients look healthier than any healthy inpatient. It then sent no one to echocardiography. The likelihood-ratio check above says which case a site is in, but it needs labels to compute.

Abstention does not repair a threshold. The per-label sets protect the share of ill patients covered when the ECGs of the ill and the healthy are the same as at the source. They hand the uncertain middle to a human, and their net benefit depends on what that human decides, which this study does not model.

## What to ask for before switching a model on

- The PPV and the false alerts per patient found, counted on your own patients at the threshold you will run, from a local sample with at least ten ill patients. A PPV recomputed from a vendor's sensitivity and specificity can miss in either direction by more than the margin of most decisions.
- A decision curve at a named threshold for the action the alert triggers, against treating everyone and treating no one. At high prevalence a model can do no better than treating everyone; at low prevalence an uncalibrated model can do worse than doing nothing.
- A recalibration on local labels before use. In these 146 transfers, 100 labelled ECGs bought more net benefit on average than any label-free correction.
- When a vendor offers a correction that needs no local label, evidence that your healthy patients look like the vendor's healthy patients. At Columbia's outpatient clinic they did not.

## Limits

This is a retrospective measurement on public, de-identified data. The threshold rule is the study's, 90% sensitivity at the source; vendors pick theirs differently, and the gap depends on where the threshold sits. Chongqing's label names a different clinical event from PTB-XL's. EchoNext's tracings carry no physical unit and each target was standardised as the source was. The five rotation models were trained on 3,500 ECGs each, smaller than a commercial model. The threshold of 10% for an echocardiogram is a stated assumption, not a measured preference.

The code, the per-cell results and the commands are in this repository: `scripts/ppv_gap.py` writes `results/ppv_gap.json` and, where EchoNext is on disk, `results/echonext_ppv_gap.json`; `scripts/repairs.py` writes `results/repairs.json`; `scripts/ppv_figures.py` draws the two figures. Each model's transfer page in [reports/transfer/](reports/transfer/) carries the gap for every label, and the repairs for the composite and for an ejection fraction of 45% or less.

*Disclosure: I worked at Anumana, which commercialises Mayo Clinic's AI-ECG work, including the lineage behind the hyperkalemia study above, and before that at Idoven.*

## Sources

1. Harmon DM et al. Validation of noninvasive detection of hyperkalemia by artificial intelligence-enhanced electrocardiography in high acuity settings. *CJASN* 2024;19(8):952-958. doi:10.2215/CJN.0000000000000483. [verified 2026-10-05 — full text, PMC11321728, Results: 351 of 40,128 and 87 of 2,636 patients, the two sets of figures quoted]
2. Ransohoff DF, Feinstein AR. Problems of spectrum and bias in evaluating the efficacy of diagnostic tests. *N Engl J Med* 1978;299(17):926-930. doi:10.1056/NEJM197810262991705. [verified 2026-10-05 — bibliographic record and abstract; full text not read]
3. Leeflang MMG, Rutjes AWS, Reitsma JB, Hooft L, Bossuyt PMM. Variation of a test's sensitivity and specificity with disease prevalence. *CMAJ* 2013;185(11):E537-E544. [verified 2026-10-05 — abstract: 23 meta-analyses, changes of 0 to 40 points, specificity lower at higher prevalence]
4. Poterucha TJ et al. Detecting structural heart disease from electrocardiograms using AI. *Nature* 2025;644:221-230. [verified 2026-10-05 — full text, PMC12328201, External validation: "At a fixed sensitivity of 70%, the external cohorts showed comparable positive predictive value, but a 10% drop in specificity"]
5. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making* 2006;26(6):565-574. [verified 2026-10-05 — bibliographic record and abstract]
6. Saerens M, Latinne P, Decaestecker C. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Comput* 2002;14(1):21-41. [verified 2026-10-05 — bibliographic record and abstract]
7. Alexandari A, Kundaje A, Shrikumar A. Maximum likelihood with bias-corrected calibration is hard-to-beat at label shift adaptation. ICML 2020, arXiv:1901.06852. [verified 2026-10-05 — abstract: calibration before maximum likelihood, likelihood concave]
8. Steyerberg EW, Borsboom GJ, van Houwelingen HC, Eijkemans MJ, Habbema JD. Validation and updating of predictive logistic regression models: a study on sample size and shrinkage. *Stat Med* 2004;23(16):2567-2586. [verified 2026-10-05 — abstract: parsimonious updating preferred in small samples]

# Questions and answers

Ten questions about this study, from three kinds of reader: clinician, technical peer, commercial. Every number can be checked against a file under [`results/`](results/). The findings are in [REPORT.md](REPORT.md).

## 1. Does "90%" apply to my patient? (clinician)

No. The guarantee is marginal: an average over patients and over draws of the calibration data. It is compatible with systematic failure inside a subgroup, and the measurement shows this at the source hospital: 90.0% of all sets contain the truth while 73.5% (sd 3.0) of infarction sets do (`results/shift.json`). Per-class calibration narrows the promise to 90% within each class, which is still conditional on the patient's true class, unknown at the point of care. A per-patient guarantee does not exist in this framework, and a tool that implies one is overclaiming.

## 2. What does a clinician see when the model is unsure? (clinician)

A set with both labels, or with none. With the smallest-set score at 90% confidence, about 95 of 100 tracings get a single label at the source hospital; most of the rest get both labels. The both-label share grows with the requested confidence: 18% at 95% at the source, 22% in Chongqing (figure 2, `results/shift.json`). At lower confidence the model returns empty sets instead, for tracings unlike anything in the calibration data. Both outcomes mean the same thing in practice: no machine answer, a human reads the ECG. Abstention rates per corpus and per confidence level are in `results/abstention.json`.

## 3. Why believe the underlying model is any good? (clinician)

Its first task was to reproduce a published number, and it did: AUROC 0.932, 95% CI 0.921 to 0.943, on the PTB-XL benchmark split where the published figure for the same architecture is 0.930 (figure 4, `results/baseline.json`; the published figure averages five diagnostic superclasses, and the file records why the infarction-only column is unavailable). The baseline is deliberately ordinary and well documented. The study is about what calibration promises survive a change of hospital, and that question is only worth asking on a model of known quality.

## 4. Would this hold at my hospital? (clinician, commercial)

Unknown. The two hospitals measured here moved in opposite directions: coverage rose in Shandong and fell in Chongqing. Neither direction nor magnitude can be extrapolated to a third site. What transfers is the harness. The measurement needs a few hundred labelled tracings from the target site and answers the question directly, per class, with its spread. The repository offers a way to measure the promise at a given site, not an assurance that it holds there.

## 5. Is the good Shandong number just prevalence? (technical peer)

Yes. 99% of Shandong's tracings have no infarction, the healthy class is the easy one, and the marginal average is dominated by it; the uncorrected threshold over-covers there (infarctions at 93.6%, sd 0.3, against 90% requested). The same mechanism hides the 73.5% infarction coverage at the source. This is why every coverage figure in the repository is reported per class (criterion C-11 in [PLAN.md](PLAN.md)), and why a single-number coverage claim on an imbalanced cohort deserves suspicion in general.

## 6. Why did the estimated correction fail where the exact one worked? (technical peer)

The weighted correction needs the target's class mix, which is unknown and must be estimated, here by BBSE from the model's own unlabelled predictions. The estimate was poor: 40.9% infarction estimated for Chongqing against 14.9% observed, 2.4% for Shandong against 1.0% (`results/shift.json`). Reweighting also spends data: the effective calibration size drops from 1,099 points to 864 in Shandong and 967 in Chongqing (`results/shift.json`, criterion C-9). Its guarantee is asymptotic. Mondrian estimates nothing: one threshold per class, exact in finite samples under any change of class mix. Paired on the same draws, Mondrian gains +16.7 points of infarction coverage at the source where weighting gains +0.1, and in Shandong weighting loses 2.3 points.

## 7. Why does the exact correction not fix Chongqing? (technical peer)

More than the class mix moved. Both corrections assume that the appearance of each class is stable across hospitals and only the proportions change. Chongqing's infarction label refers to a different clinical event, acute and angiography-confirmed, where PTB-XL's is an ECG diagnosis, mostly of older infarcts. When the appearance of the positive class changes, reweighting source calibration data cannot follow it. Mondrian lifts Chongqing's infarction coverage from 72.5% to 84.0% (sd 1.0) and stops short of the requested 90%. The residual gap is the part of the shift that calibration cannot absorb, and it is reported rather than tuned away.

## 8. Two encoders saw the test data during pre-training. Did that flatter them? (technical peer)

No home advantage is visible in the comparison. ECG-FM and HuBERT-ECG both had PTB-XL, the source corpus, in their pre-training. The arm with no public corpus at all, ECGFounder, still has the best source AUROC: 0.919 against 0.891 and 0.838 (figure 3, `results/arms.json`, every comparison paired). HuBERT-ECG also saw Shandong, and on Shandong its coverage gap sits between the two uncontaminated arms. Its small Chongqing gap (+0.047 against +0.650 for the random-init arm) is best read as transfer from the broadest pre-training mix, nine corpora including several Chinese ones, since Chongqing is in nobody's pre-training list. The grid cannot prove the absence of contamination effects; it shows that on this break, having seen the test distributions did not by itself produce better results.

## 9. Can this go into a product or a trial? (commercial)

No. It is a measurement study on public retrospective data: no device claim, no outcome claim, no prospective patient contact, no regulatory status. Two licensing constraints matter commercially: the HuBERT-ECG weights are CC BY-NC 4.0, so that arm is limited to research use; everything else, the code (MIT), the corpora (CC BY 4.0 and CC0) and the other weights, is permissively licensed. What the study contributes to a product conversation is the checklist a deployment would owe: per-class coverage at the deployment site, measured before use, with the correction chosen for the shift actually present.

## 10. Why conformal prediction rather than calibrated probabilities? (technical peer)

Temperature or Platt scaling adjusts the probability a model reports. It carries no finite-sample guarantee, and a calibrated probability still needs a hand-picked threshold to become a decision. Split conformal gives a distribution-free finite-sample guarantee on an object a workflow can act on, a set that either decides or abstains, with abstention arising from the same mechanism as the guarantee. The limitation: the guarantee is marginal and requires exchangeability, which is what this repository measures. The two families also compose. A better-calibrated score gives conformal smaller sets at the same coverage, so the choice is a layering rather than an alternative.

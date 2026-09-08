# Questions and answers

Ten questions about this study, from three kinds of reader: clinician, technical peer, commercial. Every number can be checked against a file under [`results/`](results/). The findings are in [REPORT.md](REPORT.md).

## 1. Does "90%" apply to my patient? (clinician)

No. The guarantee is marginal: an average over patients and over draws of the calibration data. It is compatible with systematic failure inside a subgroup, and the measurement shows this at the source site: 89.9% of all sets contain the truth while 73.2% of infarction sets do (`results/outcomes.json`). Per-class calibration narrows the promise to 90% within each class, which is still conditional on the patient's true class, unknown at the point of care. A per-patient guarantee does not exist in this framework, and a tool that implies one is overclaiming.

## 2. What does a clinician see when the model is unsure? (clinician)

A set with both labels. With the smallest-set score at 90% confidence, about 95 of 100 tracings get a single label at the source site, and every one of the rest gets both labels: the calibration quantile there is 0.566, and because two label probabilities sum to one, a set can come back empty only when that quantile is below 0.5. The both-label share grows with the requested confidence: 18% at 95% at the source, 22% in Chongqing (`results/figures/fig2_set_sizes.png`, `results/shift.json`). Empty sets do appear once the thresholds admitting each label sum below one, for tracings unlike anything in the calibration data, but not at the 90% this study reports. Either way it means the same thing in practice: no machine answer, a human reads the ECG. Abstention rates per corpus and per confidence level are in `results/abstention.json`.

## 3. Why believe the underlying model is any good? (clinician)

Its MI AUROC on the PTB-XL benchmark split is 0.932, 95% CI 0.921 to 0.943, which sits beside the 0.930 published for the same architecture and split (`results/figures/fig4_discrimination.png`, `results/baseline.json`; the published figure averages five diagnostic superclasses, and the file records why the infarction-only column is unavailable). The baseline is deliberately ordinary and well documented. The study is about what calibration promises survive a change of site, and that question is only worth asking on a model of known quality.

## 4. Would this hold at my site? (clinician, commercial)

Unknown. The two sites measured here moved in opposite directions: coverage rose in Shandong and fell in Chongqing. Neither direction nor magnitude can be extrapolated to a third site. What transfers is the harness. The measurement is per class, so what it needs is cases of the rarer class, not tracings: about 864 infarctions to pin per-class coverage to within two points, which is roughly 5,800 tracings at Chongqing's prevalence and 86,000 at Shandong's. The repository offers a way to measure the promise at a given site, not an assurance that it holds there.

## 5. Is the good Shandong number just prevalence? (technical peer)

Partly, and not in the way the overall figure suggests. Prevalence is what makes Shandong's *marginal* coverage look healthy: 99% of its tracings have no infarction, so the average is dominated by the easy class. But the 93.6% (sd 0.3) for infarctions is not that effect. At a fixed threshold, coverage within a class depends only on that class's scores, and Shandong's infarctions simply score further from the threshold than PTB-XL's do. Both classes over-cover there, the healthy one at 94.8%. What prevalence does hide is the 73.2% infarction coverage at the source, where the marginal figure still reads 90.0%. This is why every coverage figure in the repository is reported per class (criterion C-11 in [PLAN.md](PLAN.md)), and why a single-number coverage claim on an imbalanced cohort deserves suspicion in general.

## 6. Why did the estimated correction fail where the exact one worked? (technical peer)

The weighted correction needs the target's class mix, which is unknown and must be estimated, here by BBSE from the model's own unlabelled predictions. The estimate was poor: 40.9% infarction estimated for Chongqing against 14.9% observed, 2.4% for Shandong against 1.0% (`results/shift.json`). Reweighting also spends data: the effective calibration size drops from 1,099 points to 864 in Shandong and 967 in Chongqing (`results/shift.json`, criterion C-9). Its guarantee is asymptotic. Mondrian estimates nothing: one threshold per class, exact in finite samples under any change of class mix. Paired on the same draws, Mondrian gains +16.7 points of infarction coverage at the source where weighting gains +0.1, and in Shandong weighting loses 2.3 points.

## 7. Why does the exact correction not fix Chongqing? (technical peer)

More than the class mix moved. Both corrections assume that the appearance of each class is stable across sites and only the proportions change. Chongqing's infarction label refers to a different clinical event: its `AMI` column is set from the discharge diagnosis, in a cohort every patient of which underwent coronary angiography, where PTB-XL's label is an ECG diagnosis, mostly of older infarcts. When the appearance of the positive class changes, reweighting source calibration data cannot follow it. Mondrian lifts Chongqing's infarction coverage from 72.5% to 84.0% (sd 1.0) and stops short of the requested 90%. The residual gap is the part of the shift that calibration cannot absorb, and it is reported rather than tuned away.

## 8. Two encoders saw the test data during pre-training. Did that flatter them? (technical peer)

The comparison cannot be read yet. ECG-FM and HuBERT-ECG both had PTB-XL, the source corpus, in their pre-training, and HuBERT-ECG also saw Shandong, so the grid was built to ask whether that showed. It cannot answer while `results/arms.json` stands: its random-initialisation control was built without a fixed seed, and the arm is constructed once per corpus, so each corpus went through a different random network and the probe fitted on PTB-XL was spent in another corpus's feature space. That is why the control scores an AUROC of 0.360 at Shandong, below chance with an interval excluding 0.5, which is not a result about pre-training but an artefact of the control. The seed is fixed in the code now; the file and its figure have to be regenerated before any comparison between arms is quoted, and nothing in the report depends on them.

## 9. Can this go into a product or a trial? (commercial)

No. It is a measurement study on public retrospective data: no device claim, no outcome claim, no prospective patient contact, no regulatory status. Two licensing constraints matter commercially: the HuBERT-ECG weights are CC BY-NC 4.0, so that arm is limited to research use; everything else, the code (MIT), the corpora (CC BY 4.0 and CC0) and the other weights, is permissively licensed. What the study contributes to a product conversation is the checklist a deployment would owe: per-class coverage at the deployment site, measured before use, with the correction chosen for the shift actually present.

## 10. Why conformal prediction rather than calibrated probabilities? (technical peer)

Temperature or Platt scaling adjusts the probability a model reports. It carries no finite-sample guarantee, and a calibrated probability still needs a hand-picked threshold to become a decision. Split conformal gives a distribution-free finite-sample guarantee on an object a workflow can act on, a set that either decides or abstains, with abstention arising from the same mechanism as the guarantee. The limitation: the guarantee is marginal and requires exchangeability, which is what this repository measures. The two families also compose. A better-calibrated score gives conformal smaller sets at the same coverage, so the choice is a layering rather than an alternative.

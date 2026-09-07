# Per-label conformal calibration of an ECG classifier under prevalence shift

PTB-XL · SPH · ACS-ECG · 45,923 tracings scored · working draft, 3 September 2026

## Abstract

**Background.** A binary diagnostic classifier returns one label per case by comparing a continuous score against a threshold fixed during validation, so the sensitivity and specificity that accompany that threshold describe the validation population rather than the model. Deployed where disease prevalence differs, the classifier will operate at different error rates, and a single-label output gives no indication that it has done so. Conformal prediction (CP) offers an alternative in which the model returns a set of admissible labels together with a distribution-free guarantee on the proportion of cases whose set contains the true label. That proportion is called coverage, and a 90% coverage target means that 90 cases in 100 should receive a set holding the correct answer; it is not a false-positive rate and not a sensitivity. The guarantee costs an exchangeability assumption between the calibration and deployment populations.

**Objective.** To measure what three ways of converting the same frozen scores into a clinical output deliver in per-label error terms, and how each behaves once its calibration threshold is transferred without adjustment to sites of markedly different disease prevalence. The proof of concept is detection of myocardial infarction (MI) from the 12-lead electrocardiogram (ECG).

**Methods.** A supervised residual network was trained on PTB-XL, a German research dataset, after which its scores were held fixed. Three schemes were compared: a single threshold tuned to 90% sensitivity with no deferral option, split CP with one threshold pair calibrated on the pooled calibration sample, and split CP with one threshold pair calibrated within each label, both conformal schemes at a 90% coverage target. The single threshold's 90% and the conformal 90% are different quantities, matched deliberately so that the resulting miss rates are comparable. Thresholds fitted on held-out PTB-XL patients were then applied without re-calibration to two Chinese hospital datasets at 1.0% and 14.9% MI prevalence, with coverage reported per label as a mean over 200 patient-level calibration draws. Throughout, an error rate is a share of all cases carrying that label, with deferred cases counted as neither correct nor wrong.

**Results.** At the source site, pooled CP met its 90% coverage target overall while covering only 73.5% of MI cases, and gave the non-MI label alone to 26.8% of MI patients where the single tuned threshold gave it to 10.4%. Class-conditional calibration restored coverage to 90.1% within each label and, at an unchanged MI miss rate, reduced the false-positive rate from 19.0% to 10.0% while deferring approximately one case in ten. Under transfer, coverage of MI cases rose to 93.6% at the low-prevalence site and fell to 72.5% at the acute-presentation site, where class-conditional calibration recovered it to 84.0% without reaching the 90% requested.

**Conclusion.** Pooled conformal calibration satisfies its stated guarantee while systematically under-serving the minority label, to the point of performing worse for MI patients than an ordinary tuned threshold on this cohort. Calibrating within each label removes that failure at the source site and converts an otherwise unobservable degradation at a new site into a quantity measurable from a few hundred locally labelled tracings.

[README.md](README.md) documents the datasets and the reproduction commands. [QUESTIONS.md](QUESTIONS.md) answers the questions this study raises.

## 1. Introduction

Binary classifiers intended for clinical use report one label per case, obtained by comparing a continuous score against a threshold selected during validation, commonly to meet a sensitivity target or to maximise the sum of sensitivity and specificity. Because that threshold is chosen on a particular population, the operating characteristics it delivers are conditional on that population; a site whose patients differ in disease prevalence, presentation, acquisition hardware or annotation practice will obtain different error rates, and nothing in a single-label output reveals when this has happened.

Conformal prediction addresses the second half of that problem. Given any scoring model and a labelled calibration sample the model has not been trained on, split CP returns for each new case the set of labels that cannot be excluded at a requested confidence level, guaranteeing without distributional assumptions that the set contains the true label in a stated proportion of cases [1]. That proportion is the coverage, and it is the only quantity the guarantee constrains. In a two-label problem the output takes one of four forms: either label alone, both labels, or neither. The last two are deferrals, also called abstentions, and route the case to a specialist rather than to an automated label.

Two properties of the guarantee shape everything that follows. It holds only while the calibration and deployment populations remain exchangeable, meaning drawn from the same distribution, which is precisely the condition a change of site breaks. And it is marginal, meaning that it averages over cases rather than holding within any subgroup of them. Ninety per cent coverage therefore implies neither ninety per cent for a given patient nor ninety per cent among the patients who are ill.

On imbalanced data the marginal property has a known consequence: since the guarantee averages over the whole population, it can be satisfied while coverage within the minority label falls well below the requested level. Both variants below are split CP and differ only in the population each threshold is fitted on. Class-conditional CP avoids the failure by fitting one threshold within each label, which Vovk proves is valid conditionally on the label whatever the class proportions turn out to be, the threshold being fitted inside a taxonomy that partitions the calibration set by label [2]; the same construction is widely called Mondrian conformal prediction. A weighted alternative instead reweights the calibration cases toward the target population's estimated prevalence, with an asymptotic guarantee that depends on that estimate [3,4].

The empirical consequence has been demonstrated recently outside cardiology: two 2026 benchmarks report marginal CP under-covering the minority label, and class-conditional calibration restoring it, in virtual drug screening and across fifteen imbalanced tabular datasets [5,6]. Within cardiology, the two nearest pieces of work are narrower than their titles suggest. A conference abstract by Dzikowicz and colleagues applies split, class-conditional and learn-then-test CP to acute MI on PTB-XL, reporting an AUROC of 0.923 without CP against 0.932 for split and 0.943 for class-conditional, with overall coverage falling to 84.8% and 86.7% respectively; it uses no dataset besides PTB-XL, reports coverage over all cases rather than within each label, and sets no tuned single threshold beside the conformal ones [7]. The second is a reader study in which 62 cardiologists recruited through the Portuguese Society of Cardiology interpreted 20 ECG Wave-Maven cases under single-valued against set-valued AI support at 95% confidence, which measures how clinicians respond to prediction sets rather than how the sets themselves behave across sites [8].

Neither of those therefore answers the questions this report takes up, and a search of the ECG literature conducted on 2 and 4 September 2026 found nothing else that does. The first question is how a conformal threshold calibrated at one site behaves when it is transferred without re-calibration to sites at substantially different MI prevalence, measured within each label rather than over all cases. The second is how conformal schemes compare against an ordinary tuned threshold in the error terms a clinician uses, namely the miss rate, the false-alarm rate and the share of cases deferred. This report addresses both, and contributes a measurement rather than a method, since every technique applied is established.

## 2. Methods

### 2.1 Datasets

Three public datasets were used, with PTB-XL serving as the training and calibration source while the two remaining datasets served as transfer targets. Each target was scored once, and no label from either entered any threshold.

| Dataset | Country, years | Tracings scored | MI prevalence | Role |
|---|---|---|---|---|
| PTB-XL [9] | Germany, 1989–96 | 2,198 | 25.0% | calibration, in-distribution test |
| SPH, Shandong [10] | China, 2019–20 | 25,770 | 1.0% | transfer target |
| ACS-ECG, Chongqing [11] | China, 2015–24 | 17,955 | 14.9% | transfer target |

Two properties differ between source and targets. MI prevalence falls from 25.0% to 1.0% in the Shandong dataset and to 14.9% in the Chongqing dataset, and the label definition differs as well: Chongqing annotates acute MI confirmed by angiography, whereas PTB-XL annotates ECG diagnoses, predominantly of older infarct patterns. Because the Shandong dataset annotates chronic infarct patterns comparable to those of PTB-XL, its shift approximates a pure change in prevalence, while the Chongqing shift involves the definition of the positive class as well.

### 2.2 Model and calibration

A supervised residual network was trained on PTB-XL folds 1 to 8 and scored on fold 10, where its discrimination reproduces the published benchmark value for this architecture and split: area under the receiver operating characteristic curve (AUROC) 0.932, with a 95% confidence interval of 0.921 to 0.943, against a published 0.930. AUROC, the probability that a randomly selected MI tracing receives a higher score than a randomly selected non-MI tracing, is reported here to certify that the baseline is of ordinary and documented quality, and is not itself a study endpoint. The model was frozen thereafter, so that all three schemes operate on identical scores and any difference between them is a difference in threshold placement alone.

The three schemes were as follows. The **single tuned threshold** places one threshold on the calibration half to achieve 90% sensitivity, with every case receiving a label; this represents current practice. Its 90% is a sensitivity, whereas the 90% of the two conformal schemes is a coverage; the two were matched so that the schemes sit at comparable MI miss rates and the remaining differences are interpretable. **Pooled conformal calibration** fits one threshold pair on all calibration cases, such that 90% of them receive a set containing the true label. **Class-conditional conformal calibration** fits one threshold within the MI cases of the calibration half and one within the remainder, so that the 90% holds separately within each label. A fourth scheme, label-shift weighting with the target prior estimated by black-box shift estimation, was also evaluated; because its effect was small and its prior estimate poor at the Chongqing site, it is reported in the results files rather than discussed here.

### 2.3 Evaluation

Every reported figure is a mean over 200 draws in which fold 10 was halved by patient, so that no patient contributed to both calibration and test, with standard deviations across draws recorded in the results files. Coverage is reported separately for each label as well as overall, and outcomes are decomposed into three exhaustive categories per label: the case received the correct label alone, the case was deferred, or the case received the incorrect label alone. Each is expressed as a share of all cases carrying that label, so that a miss rate and a false-alarm rate remain comparable across schemes with and without deferral; deferred cases are counted as neither correct nor wrong rather than excluded from the denominator. Each methodological commitment is enforced by a named test in the repository, including patient-level splitting, the absence of any hard-coded threshold, paired comparisons between encoders, and a test that fails if any target-dataset label reaches a threshold.

## 3. Results

### 3.1 Threshold placement

Figure 1 shows where each scheme places its thresholds on the score distribution of the source dataset. Pooled calibration places both thresholds substantially to the right of the single tuned threshold, because three quarters of the calibration population are non-MI cases and the requirement is stated over that population as a whole, so that serving the majority well is the cheapest way to satisfy it. Class-conditional calibration cannot make that trade, since each of its thresholds is answerable only to the label it governs.

![Figure 1](results/figures/fig1_thresholds.png)

Figure 1. Threshold placement on the model score for MI, PTB-XL fold 10. Grey: 1,648 non-MI cases. Coral: 550 MI cases. Dashed lines mark the fitted decision boundaries; the labelled bands give the resulting decision regions.

### 3.2 Per-label outcomes at the source site

Table 1 and figure 2 give the decomposition at the source site. Pooled conformal calibration satisfied its stated coverage, covering 90.0% of all cases, while covering only 73.5% of MI cases; it gave the non-MI label alone to 26.8% of MI patients, against 10.4% under the single tuned threshold. The guarantee was at no point violated, since it concerns the average case, and the average case in this cohort is healthy.

![Figure 2](results/figures/fig2_outcomes.png)

Figure 2. Outcome decomposition by label at the source site, averaged over 200 patient-level calibration draws. The two conformal schemes run at a 90% coverage target and the single threshold at 90% sensitivity, which places all three at a comparable MI miss rate. Each share is of all cases carrying that label.

Table 1. Source-site outcomes per 100 patients of each label, from `results/outcomes.json`.

| Outcome | Single threshold | Pooled CP | Class-conditional CP |
|---|---|---|---|
| MI case, labelled non-MI | 10 | 27 | **10** |
| MI case, deferred | 0 | 10 | 13 |
| Non-MI case, labelled MI | 19 | **5** | 10 |
| Non-MI case, deferred | 0 | 4 | 10 |

Class-conditional calibration returned coverage to 90.1% within each label. Measured against the single tuned threshold at a matched MI miss rate of approximately 10%, it reduced the false-positive rate from 19.0% to 10.0% while deferring 12.7% of MI cases and 9.7% of non-MI cases; measured against pooled calibration, it raised the false-positive rate from 4.5% to 10.0% while reducing missed infarctions from 26.8% to 10.0%. Which of the two trades is preferable depends on the cost of a missed infarction relative to an unnecessary workup, a judgement this design does not attempt.

### 3.3 Transfer to the two target sites

Thresholds fitted on PTB-XL were applied unchanged to both target datasets, whereupon coverage moved in opposite directions. At the Shandong site, where 99 tracings in 100 are negative, coverage of MI cases under pooled calibration reached 93.6%, above the 90% requested, because the easy majority carries the average upward, such that a report of the overall figure alone would have read as a success. At the Chongqing site coverage of MI cases fell to 72.5%, 17.5 points below the 90% requested, and class-conditional calibration lifted it to 84.0% without reaching the requested level. For comparison, the single tuned threshold fell from 89.6% sensitivity at the source to 83.7% at Chongqing, with specificity falling from 81.0% to 57.9%.

![Figure 3](results/figures/fig1_coverage.png)

Figure 3. Coverage by site (rows) and calibration scheme (columns). Grey: all cases. Red: MI cases. Dashed line: requested coverage. Bars are means over 200 calibration draws, with whiskers at one standard deviation.

Class-conditional calibration did not restore the Chongqing site to the requested coverage because both corrections assume that only the label proportions change between sites. At Chongqing the positive label denotes a different clinical event, so the score distribution within the positive class has moved as well, and no rearrangement of thresholds fitted on German data can follow it.

## 4. Discussion

A coverage guarantee stated over a whole population is compatible with systematic failure inside a minority label, and on a cohort at 25% prevalence that gap proved wide enough to make pooled conformal calibration worse for MI patients than an ordinary tuned threshold. A single-number coverage claim on an imbalanced diagnosis is therefore close to uninformative about the label that matters clinically, and per-label reporting should be treated as the minimum standard for such a tool. On ECG this reproduces an effect already reported in other domains [5,6].

Calibrating within each label removes the failure at no computational cost and with nothing to estimate. Against current practice it offers a lower false-positive rate at an unchanged miss rate, purchased by deferring roughly one case in ten to a specialist, so that whether the trade is favourable turns on the relative costs of a missed infarction, an unnecessary workup and a specialist read, none of which this design evaluates.

Most relevant to deployment, neither conformal scheme prevented degradation at the harder target site, where the single tuned threshold degraded as well, losing six points of sensitivity and twenty-three of specificity. What distinguishes a conformal scheme is that it commits to an explicit quantity a receiving site can test, namely the share of cases of each label whose set contains the true label, which is estimable from a few hundred locally collected labelled tracings before the classifier is placed in service. That coverage guarantees fail under domain and label shift, and that recalibration is the remedy, is already established for medical imaging [12], and the value of estimating a model's performance at a new hospital before clinical deployment has been argued there too, albeit through label-free accuracy estimation rather than by re-measuring coverage [13]. What this study supplies is the per-label magnitude on ECG, together with the observation that the direction of the effect is not predictable in advance, given that the two sites moved opposite ways.

## 5. Limitations

- **No outcome endpoint.** Deferral is reported as a rate rather than as a clinical benefit, since whether routing one case in ten to a specialist improves care depends on reader availability and turnaround, neither of which was measured.
- **Limited generality.** Two target sites, one diagnosis and one source dataset, with the two targets moving in opposite directions, so that neither the direction nor the magnitude of the effect extrapolates to a third site.
- **Confounded shift.** Acquisition hardware, population, era and label definition differ simultaneously between source and targets, so the design measures the resulting change without being able to attribute it to any single factor.
- **Marginal guarantee.** Coverage is an average over the cases of a label and supports no statement about an individual tracing, while the class-conditional guarantee is conditional on the true label, which is unknown at the point of care.
- **Not a device.** This is a retrospective measurement study on public data, carrying no device claim, no outcome claim, no prospective patient contact and no regulatory status.

## References

1. Angelopoulos AN, Bates S. *A gentle introduction to conformal prediction and distribution-free uncertainty quantification.* arXiv:2107.07511.
2. Vovk V. *Conditional validity of inductive conformal predictors.* ACML 2012, PMLR 25:475–490. Proposition 3 gives label conditional validity for label conditional inductive conformal predictors.
3. Tibshirani RJ, Barber RF, Candès E, Ramdas A. *Conformal prediction under covariate shift.* NeurIPS 2019. The weighted conformal quantile the reweighting arm is built on.
4. Podkopaev A, Ramdas A. *Distribution-free uncertainty quantification for classification under label shift.* UAI 2021, PMLR 161:844–853. arXiv:2103.03323. Reweighting conformal prediction and calibration by importance weights estimated from unlabelled target data.
5. Tursunbadalov M, Tursunbadalov M. *A quiet failure in calibrated virtual screening: marginal conformal prediction under-covers the minority class, and a class-conditional fix recovers it.* arXiv:2607.06605, July 2026. Four molecular datasets; at a 90% global target, minority coverage falls to 64.8% on blood-brain-barrier penetration, 38.9% on a Tox21 endpoint and 4.2% on clinical-trial toxicity.
6. Singh M, Srikantha A, Lakhanpal S. *Cost-sensitive conformal prediction and human-in-the-loop abstention for imbalanced high-stakes decision support: a multi-domain benchmark.* arXiv:2607.27143, July 2026. Fifteen real-world imbalanced tabular datasets, seven models; marginal CP averages 30.5% minority coverage and Mondrian CP recovers 61.7 points of it.
7. Dzikowicz D, Garcia JJ, Zègre-Hemsey J, Kitzmiller R, Rogers D, Betts J, Bouvier M, Goyal A, Hu X, Xiao R. *Conformal prediction improves acute myocardial infarction identification from 12-lead ECGs: a practical deep learning application.* Conference abstract in *Abstracts: 50th International Congress on Electrocardiology*, Annals of Noninvasive Electrocardiology 2025;30(Suppl 1). doi:10.1111/anec.70099, PMCID PMC12234157. Read in full 2026-09-04 via PubMed Central; a one-paragraph meeting abstract, not a full paper.
8. Folgado D, Famiglini L, Campagner A, Dores H, Barandas M, Gamboa H, Cabitza F. *Conformal prediction for ECG interpretation: a study on human-AI collaboration in clinical decision support.* AIME 2025, LNCS 15734. doi:10.1007/978-3-031-95838-0_14. Supplementary material doi:10.5281/zenodo.15322935, read 2026-09-04: design, cohort of 62 cardiologists and the 20-case Wave-Maven stimulus set are taken from it; the chapter text sits behind Springer.
9. Wagner P, Strodthoff N, Bousseljot R-D, Kreiseler D, Lunze FI, Samek W, Schaeffter T. *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data 2020. doi:10.1038/s41597-020-0495-6. Recorded by Schiller AG devices between October 1989 and June 1996, curated at the Physikalisch-Technische Bundesanstalt.
10. Liu H, Chen Dan, Chen Da, Zhang X, Li H, Bian L, Shu M, Wang Y. *A large-scale multi-label 12-lead electrocardiogram database with standardized diagnostic statements.* Scientific Data 2022. doi:10.1038/s41597-022-01403-5. 25,770 records from 24,666 patients, acquired at Shandong Provincial Hospital between 2019/08 and 2020/08.
11. Du X, Liu Y, Wang L, He W, Yang J, Bin G, Deng G. *A large-scale 12-lead electrocardiogram dataset for acute coronary syndrome prediction containing 19,955 ECGs.* Scientific Data 2026, doi:10.1038/s41597-026-07278-0; data at doi:10.6084/m9.figshare.29925314. Preoperative ECGs from 18,909 patients undergoing coronary angiography at the First Affiliated Hospital of Chongqing Medical University, December 2015 to May 2024.
12. Mehrtens H, Bucher T-C, Brinker TJ. *Pitfalls of conformal predictions for medical image classification.* arXiv:2506.18162, 2025. German Cancer Research Center. Coverage under domain shift and label shift on CAMELYON17 and HAM10000, and the marginal against conditional coverage distinction.
13. Lu C, Ahmed SR, Singh P, Kalpathy-Cramer J. *Estimating test performance for AI medical devices under distribution shift with conformal prediction.* arXiv:2207.05796, 2022. Estimates a black-box model's accuracy on an unlabelled target domain; argues for knowing performance "at new hospitals, patient populations, medical scanner equipment, etc. before actual clinical deployment".

Every reference above was read in full on 4 September 2026, except the AIME 2025 chapter [8], which is behind Springer and is characterised here from its open supplementary material.

## Data and code

Numbers: `results/outcomes.json` (per-label outcomes and decision boundaries), `results/shift.json` (coverage per site and correction), `results/arms.json` (encoders), `results/abstention.json`, `results/baseline.json`. `scripts/outcomes.py` builds the outcome table and `scripts/figures.py` regenerates every figure from a results file committed before it. Reproduction: `uv sync`, `uv run pytest`, `uv run python scripts/outcomes.py`, `uv run python scripts/figures.py`.

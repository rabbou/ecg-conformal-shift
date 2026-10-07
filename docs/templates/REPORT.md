# Predicting an ECG model's positive predictive value from a clinic's prevalence: a retrospective measurement across hospitals and care settings

Ruben Abbou · October 2026

## Abstract

**Background.** A clinic adopting an ECG model receives a sensitivity and specificity measured elsewhere and estimates its positive predictive value (PPV) by Bayes' theorem at its own prevalence. This prevalence adjustment assumes that sensitivity and specificity are transportable, unchanged between populations. A spectrum shift, a change in disease severity or presentation, violates it.

**Methods.** A threshold set for 90% sensitivity in a source population was applied unchanged to a target in {{ppv_gap_cells}} transfers: four structural heart disease (SHD) models from Columbia inpatients to emergency patients and outpatients (EchoNext), an infarction network from Germany to two Chinese hospitals, and five rhythm and conduction diagnoses rotated across five corpora. The PPV recomputed at the target's true prevalence was compared with the observed PPV; {{ppv_control_cells}} controls used held-out source patients.

**Results.** Values in parentheses are 95% confidence intervals. The recomputed PPV differed from the observed PPV by a median of {{ppv_gap_median_ci}} percentage points ({{ppv_control_median_ci}} in controls) and fell outside the observed PPV's interval in {{ppv_outside}} of transfers, in both directions. Among Columbia outpatients, of whom {{col_prev_out_ci}} had SHD, the trained network's observed PPV was {{col_ppv_obs_ci}} and the recomputed PPV {{col_ppv_rec}}. With an AUROC of {{auroc_resnet_in}} among inpatients and {{auroc_resnet_out}} among outpatients, sensitivity at the inpatient threshold fell from {{sens_resnet_in}} to {{sens_resnet_out}} and specificity rose from {{spec_resnet_in}} to {{spec_resnet_out}}. The proportion of outpatients flagged, {{lf_flag_out_ci}}, was below what any prevalence could produce at the inpatients' sensitivity and specificity. Recalibrated on {{vo_n}} separate outpatients, the threshold reached {{vo_sens_resnet}} sensitivity ({{vo_sens_resnet_ci}}). At Chongqing, the infarction network's PPV was {{acs_ppv_rec}} recomputed and {{acs_ppv_obs_ci}} observed.

**Conclusions.** An ECG model's PPV in a new clinical setting was not reliably predicted from the local prevalence: the ranking of patients was preserved, but the operating point of a fixed threshold was not. The PPV in a new setting is obtained by measuring it there, in patients with known diagnoses. The proportion flagged, which requires no reference standard, can indicate that the operating point has shifted.

## 1. Introduction

A hospital evaluating an ECG model receives an AUROC, and a sensitivity and a specificity at a threshold on the model's score, measured in another institution's patients. None of these gives the proportion of a clinic's alerts that are true positives. That proportion, the positive predictive value (PPV), and its counterpart for test-negative patients, the negative predictive value (NPV), depend on the prevalence of the condition among the clinic's patients. Standard practice, which we have also recommended, is prevalence adjustment: the PPV is recomputed at the local prevalence by Bayes' theorem, PPV = sens × prev / (sens × prev + (1 − spec) × (1 − prev)).

Prevalence adjustment is exact when sensitivity and specificity are transportable between the two populations, that is, when only the prevalence differs and cases and non-cases are otherwise alike. In the machine-learning literature this prevalence shift is termed label shift [5]. Under covariate shift the distribution of ECGs changes, and under spectrum shift the cases or non-cases differ in severity or presentation; under either form of dataset shift the adjusted PPV can be biased. Clinical epidemiology calls this the spectrum effect [2,3]: across 23 meta-analyses, the sensitivity or specificity of a single test differed by up to 40 percentage points between low- and high-prevalence studies [4].

Prevalence adjustment sometimes performs well: a hyperkalemia model validated at Mayo Clinic had a sensitivity and a specificity of 80% in the emergency department, at a prevalence of 1%, and of 82% in intensive care, at 3% [1], so Bayes' theorem carries the PPV from one unit to the other. For structural heart disease (SHD) the evidence is less favourable. In the EchoNext study, at a fixed sensitivity of 70%, specificity at external hospitals was 10 percentage points lower than at Columbia; on a multicentre test set the full model had an AUROC of 0.843 among outpatients and 0.841 among inpatients, with no sensitivity or specificity by care setting [6]. An external validation of the published mini-model reports its AUROC by care setting without a sensitivity at a fixed cut-off [7]. In a community cohort the AUROC fell to 0.71, from 0.83 in hospital, which the authors attribute to milder disease [8]. A preserved AUROC does not imply a preserved operating point at a fixed threshold.

The Mayo Clinic model for an ejection fraction of 35% or less kept an AUROC of 0.82 in a Russian population sample, yet the cut-off of its original study detected 26.9% of cases [9]. Others transported: at a fixed cut-off, a model for an ejection fraction of 40% or less had a sensitivity of 84.5% and a specificity of 83.6% across four American sites [10], and PRESENT-SHD, whose threshold was set for 90% sensitivity at one hospital, detected 92.5% to 96.0% of cases at four others [11]. Cut-offs have been recalibrated on local samples in pathology and CT [12,13] and in screening mammography [14]. Searches of PubMed and Europe PMC on 5 October 2026 (supplement S1.13) identified no study that measured, for SHD, the operating point of a fixed threshold when one hospital's model is transferred from its inpatients to its outpatients. Supplement S1.14 gives these studies in full.

We examined whether a model's PPV in a new setting can be predicted from the local prevalence, by measuring the difference between the recomputed and the observed PPV in {{ppv_gap_cells}} threshold transfers, and examined the transfer from inpatients to outpatients at Columbia to characterise the source of the error.

## 2. Methods

This is a retrospective study of the transportability of ECG classification models across hospitals and care settings, reported following the TRIPOD+AI guideline [26]. The unit of analysis is a transfer: one model, one diagnosis and one threshold, set in a source population and applied unchanged in a target population.

### 2.1 Choice of datasets

A transfer can be studied only where the target's diagnoses are recorded, since the observed PPV is computed from them; the datasets are public, de-identified ECG collections that allow this after a shift. EchoNext is, to our knowledge, the only public ECG dataset distributed with both the care setting of each ECG (inpatient, emergency or outpatient) and a target condition confirmed by echocardiography [6], so a transfer between care settings can be observed within one institution, with the same reference standard on both sides. PTB-XL [15], with the Shandong [16] and Chongqing [17] datasets, gives an infarction model transferred across countries. The five rotated corpora share the rhythm and conduction labels of the PhysioNet/Computing in Cardiology Challenge 2021 [18] and come from three countries, Germany, China and the United States, so each is both a source and a target for the same diagnoses. MIMIC-IV-ECG, in which Otabor and colleagues validated the EchoNext mini-model by care setting [7], is not yet in the analysis.

### 2.2 Participants and outcome at Columbia

EchoNext v1.1.1 holds 100,000 ECGs from two NewYork-Presbyterian sites, referred to below as Columbia, each paired with a transthoracic echocardiogram [6]. The target condition is moderate or worse SHD, which EchoNext defines as any of eleven echocardiographic findings, among them a left ventricular ejection fraction of 45% or less. An ECG is positive when recorded within one year before an echocardiogram showing any finding, and negative when recorded at any time before the patient's most recent echocardiogram showing none. The models were trained on the {{train_n}} ECGs of EchoNext's training group. The thresholds were set on the {{cal_n}} inpatient ECGs of the validation group, {{cal_prev}} of them positive, referred to below as the calibration inpatients. All measurements were made on the test group: {{out_n}} outpatient ECGs ({{out_prev}} positive), {{in_n}} other inpatient ECGs ({{in_prev}} positive) and {{em_n}} emergency ECGs ({{em_prev}} positive). No patient is in two groups. Supplement S1.15 gives the methods at Columbia in full, with the eleven findings and the accounting of every ECG.

### 2.3 Participants and outcome at the other hospitals and corpora

PTB-XL is a German research database of ECGs recorded between 1989 and 1996 [15]. An infarction network trained on it was transferred to Shandong Provincial Hospital [16] and to the First Affiliated Hospital of Chongqing Medical University [17]. In PTB-XL an infarction is an infarct pattern on the tracing, most often without a stated stage; at Shandong it is also read from the tracing, and most are coded as old; at Chongqing it is an acute infarction named in the discharge diagnosis, in a cohort in which every patient underwent coronary angiography.

Five corpora were rotated through the source role: PTB-XL, Shandong, and three from the Challenge 2021 collection, Chapman-Shaoxing with Ningbo (China), Georgia (United States) and CPSC 2018 with its extension (China). The diagnoses are sinus rhythm, atrial fibrillation, left bundle-branch block, right bundle-branch block and first-degree atrioventricular block. Each was split by patient after merging duplicate tracings, with its training and calibration parts capped at the size of the smallest corpus (supplement S2.4).

### 2.4 Models

At Columbia four models score every ECG: a residual neural network trained on EchoNext for this study; the published EchoNext mini-model, run unchanged on its authors' weights, with an AUROC of {{auroc_mini_all}} on the EchoNext test patients, the figure its authors report [6]; ECGFounder, a network pre-trained on more than ten million ECGs from another hospital [19], kept frozen, with a logistic regression fitted on EchoNext to its embeddings; and the first network with randomly initialised weights, read through the same kind of regression. This last model, the untrained floor, is a negative control. For infarction a residual network was trained on PTB-XL, and for the rotation one on each corpus. The [supplement](SUPPLEMENT.md) gives the architectures, how the published weights were loaded, and the statistical details of every method below.

### 2.5 Decision threshold

For each model and diagnosis, the threshold was placed among the source cases so that 90% of them scored at or above it, a target sensitivity of 90%. A patient at or above it is test-positive (flagged). When cases are few, the threshold is placed slightly below the empirical 90th percentile (supplement S1.1).

At Columbia the source population is the calibration inpatients, as for a model validated in hospital and deployed in outpatient clinics. The same rule was also applied to all {{va_n}} validation patients regardless of care setting, as a vendor might do, and to the {{vo_n}} validation outpatients, {{vo_ill}} of them positive, distinct from the test outpatients. For infarction, the threshold was set on one half of PTB-XL's held-out patients and applied to the other half and to the two Chinese sites; in the rotation, each corpus's calibration part set the threshold for the other four. Both repeated the draw 200 times.

### 2.6 Primary performance measure: recomputed and observed PPV

In each transfer, the source sensitivity and specificity at the threshold, entered into Bayes' theorem with the target's true prevalence, give the recomputed (prevalence-adjusted) PPV. The observed PPV is the proportion of test-positive target patients who are cases. Supplying the true prevalence favours the recomputation. The gap is the recomputed PPV minus the observed PPV, in percentage points, with an interval from 2,000 redraws of both the source and the target counts. A transfer enters the summaries when its target holds at least 10 cases and its threshold flags at least 20 patients; {{ppv_not_summarised}} transfers did not.

Controls evaluated the same models on held-out source patients. Transfers that share a model and a pair of populations are not independent, so each proportion across transfers carries a 95% cluster bootstrap interval that resamples whole groups of them, {{ppv_clusters}} groups for the transfers and {{ppv_control_clusters}} for the controls (supplement S3.1). As a check of the label-shift assumption, the median likelihood ratio the model assigns to the non-cases of each target was computed; under label shift it is close to 1. Three PPV predictions that require no local reference standard were also compared with the observed PPV (supplement S3.5).

### 2.7 Secondary measures and local updating

At Columbia, outcomes are counted per 100 patients with and without SHD, and discrimination is compared by the AUROC. A logistic regression fitted on the cases of both settings standardised the outpatients' sensitivity to the inpatients' case mix (supplement S1.6).

Three updating methods were compared by net benefit, the measure of decision curve analysis [22], at decision thresholds of 5%, 10% and 20%. The first, a label-free correction, shifts every probability to a prevalence estimated from the site's unlabelled scores [5,20]; the second is logistic recalibration on 100 local ECGs with known diagnoses [21]; the third adds a second threshold and refers patients between the two to a human reader (supplement S3.5).

In threshold recalibration on local labels only the cut-off is reset. At Columbia, 100 outpatients with known diagnoses were drawn from a random half of the test outpatients, the threshold was set on them by the rule of section 2.5, and the result was evaluated on the other half, {{lad_draws}} times. Because a clinic applies its threshold to new patients, the threshold was also set on all {{vo_n}} validation outpatients, and on 100 drawn from them, and evaluated on every test outpatient.

### 2.8 Statistical analysis

Proportions among patients carry Wilson 95% confidence intervals [27], and the AUROC a percentile bootstrap interval. Where the threshold itself is uncertain, the calibration patients were resampled as well, and the text says so. Subgroup comparisons by sex and age used Fisher's exact test or a chi-square test with Holm's correction for six tests [28] and are exploratory. No sample size was calculated in advance. Every ECG of the validation and test groups had a score, an age, a sex and the label, and the study was not registered.

## 3. Results

The PPV recomputed by Bayes' theorem differed from the observed PPV in most transfers, in both directions. At Columbia the model's ranking of patients was preserved while the outpatients' scores were lower, so the same threshold corresponded to a different operating point, with a different sensitivity and specificity.

### 3.1 The recomputed PPV was biased in both directions

Across {{ppv_gap_cells}} transfers, the recomputed PPV differed from the observed PPV by a median of {{ppv_gap_median_ci}} percentage points. It fell within two points in {{ppv_within2}} of transfers and outside the observed PPV's 95% confidence interval in {{ppv_outside}}. In the {{ppv_control_cells}} controls it differed by {{ppv_control_median_ci}} points and fell outside the interval in {{ppv_control_outside}}. Relative to the PPV itself, the recomputed PPV was in error by a quarter or more of the observed value in {{ppv_quarter}} of transfers, against {{ppv_control_quarter}} of controls.

Table 1. The recomputed PPV against the observed PPV, from `results/ppv_intervals.json`, `results/ppv_gap.json` and `results/echonext_ppv_gap.json`. Groups are the model-and-population pairs the intervals resample. Too high: the recomputed PPV above the observed PPV. The two infarction transfers are reported in section 3.4.

| Pairs | Transfers | Groups | Median gap, points | Within two points | Outside the observed 95% interval | Off by a quarter or more | Too high |
|---|---|---|---|---|---|---|---|
{{rows:ppv_blocks}}

The direction of the error depended on the pair of populations. At Columbia the recomputation underestimated the PPV in every outpatient transfer. Across the rotation it more often overestimated it. The largest error was for left bundle-branch block transferred from PTB-XL to Shandong, recomputed at {{worst_rec}} and observed at {{worst_obs_ci}}.

![Figure 1](results/figures/ppv_gap.png)

Figure 1. Left: each point is one transfer (one model, one diagnosis and one pair of populations), with the recomputed PPV against the observed PPV; grey points are controls, in which the target is drawn from the source population. Right: the same points, the gap in percentage points against the change in specificity from source to target.

Where specificity rose in the target, the recomputation underestimated the PPV; where it fell, the recomputation overestimated it (Figure 1, right). The median likelihood ratio among non-cases was {{lr_healthy_in}} among Columbia's test inpatients, the control, {{lr_healthy_em}} among emergency patients and {{lr_healthy_out}} among outpatients. The distribution of ECGs among non-cases therefore changed, which label shift excludes, and the three settings ranked in the same order on this ratio as on the PPV gap.

### 3.2 Columbia: preserved discrimination, shifted operating point

At Columbia, {{col_prev_out_ci}} of the test outpatients had moderate or worse SHD. Of the outpatients flagged by the trained network at its inpatient threshold, {{col_ppv_obs_ci}} had SHD. Among the calibration inpatients the threshold had a sensitivity of {{col_sens_src}} and a specificity of {{col_spec_src}}; Bayes' theorem with those two values at the outpatients' prevalence gives {{col_ppv_rec}}, an underestimate of {{col_gap}} points ({{col_gap_ci}}). The recomputation implied {{col_fa_rec}} false positives per true positive where there were {{col_fa_obs}}.

The trained network's threshold had a sensitivity of {{sens_resnet_in}} among the other inpatients ({{sens_resnet_in_ci}}) and {{sens_resnet_out}} among outpatients ({{sens_resnet_out_ci}}). Resampling the calibration inpatients as well, so that the threshold varies too, widens the outpatient interval to {{ts_ci_resnet_out}}. Emergency patients were intermediate ({{sens_resnet_em}}). The threshold also flagged fewer patients without SHD: {{flagged_resnet_out}} per 100 among outpatients, against {{flagged_resnet_in}} per 100 among the other inpatients; its specificity was {{spec_resnet_out}} ({{spec_resnet_out_ci}}) among outpatients and {{spec_resnet_in}} ({{spec_resnet_in_ci}}) among inpatients. The sensitivity among outpatients was {{sens_mini_out}} for the published mini-model and {{sens_ecgf_out}} for ECGFounder.

Table 2. The trained network's inpatient threshold, per 100 patients with SHD and per 100 patients without SHD in each group: {{n_ill_resnet_in}} with and {{n_healthy_resnet_in}} without SHD among the other inpatients, {{n_ill_resnet_out}} with and {{n_healthy_resnet_out}} without SHD among outpatients, from `results/echonext_clinical.json`.

| | Other inpatients | Outpatients |
|---|---|---|
| With SHD, detected (true positives) | {{caught_resnet_in}} | {{caught_resnet_out}} |
| With SHD, missed (false negatives) | {{missed_resnet_in}} | {{missed_resnet_out}} |
| Without SHD, flagged (false positives) | {{flagged_resnet_in}} | {{flagged_resnet_out}} |

In a clinic seeing 1,000 outpatients at Columbia's outpatient prevalence, the inpatient threshold refers {{k_flagged_out}} for echocardiography, detects {{k_found_out}} of the {{k_ill_out}} patients with SHD and misses {{k_missed_out}}. Of the outpatients below the threshold, {{col_npv_out_ci}} do not have SHD.

Discrimination was preserved. The AUROC of the trained network was {{auroc_resnet_in}} among inpatients ({{auroc_resnet_in_ci}}) and {{auroc_resnet_out}} among outpatients ({{auroc_resnet_out_ci}}), and changed as little for the other two models. Outpatients with and without SHD both scored lower than their inpatient counterparts, so the same threshold fell at a different point on a nearly unchanged receiver operating characteristic curve (Figure 2): it detected fewer cases and flagged fewer non-cases. A score built on age and sex alone kept its sensitivity among outpatients, and the three ECG models detected similar proportions of the same outpatients with SHD (supplement S1.16).

![Figure 2](results/figures/fig_curve.png)

Figure 2. Receiver operating characteristic curves of the trained network: patients with SHD detected per 100 against patients without SHD flagged per 100, at every possible threshold, among the {{in_n}} test inpatients ({{n_ill_resnet_in}} with SHD) and the {{out_n}} test outpatients ({{n_ill_resnet_out}} with SHD). The dots mark the threshold set on the {{cal_n}} calibration inpatients.

The magnitude of the shift depended on the population in which the threshold was set. Set on all {{va_n}} validation patients, regardless of setting, the threshold had a sensitivity of {{va_sens_resnet_out}} among outpatients ({{va_sens_resnet_out_ci}}): the decrease was smaller but persisted.

![Figure 3](results/figures/fig_patients.png)

Figure 3. The inpatient threshold applied to the test outpatients ({{out_ill}} with SHD, {{out_healthy}} without), per 100 of each, for the three models and the untrained floor. Rows marked one threshold use the threshold alone; rows marked two thresholds add the second threshold of supplement S1.19, and patients between the two are referred to a reader (grey). Blue: a true positive or a true negative, with no reader. Orange: a false negative, or a false positive with no reader. Patients with SHD detected are blue plus grey. Each row's counts are rounded to sum to 100.

### 3.3 Factors associated with the lower outpatient scores

The loss of sensitivity was concentrated in some findings: among patients with a wall thickness of 1.3 cm or more the threshold detected {{bf_lvwt_out}} of outpatients against {{bf_lvwt_in}} of inpatients, and among those with an ejection fraction of 45% or less {{bf_lvef_out}} against {{bf_lvef_in}}. Standardised to the inpatients' findings, ejection fraction, age and sex, the outpatients' sensitivity would have been {{mix_cov_resnet}} instead of {{sens_resnet_out}}, a share of {{mix_share_resnet}} ({{mix_share_resnet_ci}}). Case mix thus accounted for about a quarter of the fall, and the remainder occurred among patients with the same findings, ejection fraction, age and sex. Women and younger outpatients scored lower, with and without SHD (supplement S1.17).

### 3.4 Across hospitals, discrimination also changed

The infarction threshold, set for 90% sensitivity in PTB-XL, had a sensitivity of {{mi_sens_ptbxl}} there ({{mi_sens_ptbxl_ci}}), {{mi_sens_acs}} at Chongqing ({{mi_sens_acs_ci}}) and {{mi_sens_sph}} at Shandong ({{mi_sens_sph_ci}}). At Chongqing the recomputation overestimated the PPV, {{acs_ppv_rec}} against {{acs_ppv_obs_ci}} observed, because specificity fell from {{acs_spec_src}} to {{acs_spec_tgt}}; the AUROC fell from {{mi_auroc_ptbxl}} to {{mi_auroc_acs}}. At Shandong the AUROC rose to {{mi_auroc_sph}}, and the recomputed PPV differed from the observed PPV by less than one point, {{sph_ppv_rec}} against {{sph_ppv_obs_ci}}. In both transfers the definition of infarction changed with the population (section 2.3). Supplement S2.15 gives the three sites in full; Table 1 summarises the rotation's {{ppv_rotation_cells}} transfers, and Part 2 of the supplement reports them by diagnosis.

### 3.5 The proportion flagged indicated the shift without a reference standard

Without diagnoses, a clinic observes the proportion of patients the model flags. From inpatients to outpatients, the proportion flagged by the trained network fell from {{lf_flag_in_ci}} to {{lf_flag_out_ci}}. A lower prevalence alone cannot produce that fall. Had the inpatients' sensitivity and specificity been preserved, the proportion flagged would have been {{lf_expect_resnet}} at the outpatients' prevalence, and no prevalence could reduce it below {{lf_floor_resnet}}, the false-positive rate among inpatients. The mini-model flagged {{lf_flag_mini_out}} of outpatients against a lower bound of {{lf_floor_mini}}, and ECGFounder {{lf_flag_ecgf_out}} against {{lf_floor_ecgf}}. The proportion flagged does not give the number of false negatives, {{missed_resnet_out}} per 100 outpatients with SHD here, which only a reference standard can count. Predictions of the PPV that require no reference standard had larger errors than the recomputation at the true prevalence (supplement S3.5).

### 3.6 Three updating methods compared by net benefit

Logistic recalibration on 100 local labels was the only updating method with a positive mean gain in net benefit across transfers, {{rep_gain10}} net true positives per 1,000 patients at a decision threshold of 10%. For Columbia's outpatients it lowered net benefit at decision thresholds of 5% and 10% for all three models. The label-free correction gave the highest net benefit of the four rules at Shandong, at decision thresholds of 10% and 20%, and performed poorly at Columbia, where it estimated the outpatients' prevalence at {{prior_prev_estimated}} and referred no one to echocardiography (supplement S3.6).

### 3.7 Threshold recalibration on local labels

Recalibrated on 100 outpatients with known diagnoses, the threshold had a mean sensitivity of {{lad100_sens_resnet}} over {{lad_draws}} repetitions, against {{lad0_sens_resnet}} for the inpatient threshold (Figure 4), and flagged a mean of {{lad100_hflag_resnet}} per 100 outpatients without SHD, against {{flagged_resnet_out}}. In {{lad100_below_resnet}} of the repetitions it still had a sensitivity below 90% among the outpatients it had not been set on; for the mini-model and ECGFounder, {{lad100_below_mini}} and {{lad100_below_ecgf}}, about one repetition in three. With 200 outpatients the spread across repetitions narrowed but the proportion below 90% did not decrease ({{lad200_below_resnet}}): the rule targets 90% sensitivity on average (supplement S1.18).

These 100 were drawn from the outpatients on whom the threshold was evaluated, whereas a clinic sets its threshold on past patients and applies it to new ones. Set on all {{vo_n}} validation outpatients, {{vo_ill}} of them cases, the threshold had a sensitivity among the test outpatients of {{vo_sens_resnet}} ({{vo_sens_resnet_ci}}), {{vo_sens_mini}} ({{vo_sens_mini_ci}}) for the mini-model and {{vo_sens_ecgf}} ({{vo_sens_ecgf_ci}}) for ECGFounder: short of 90% for all three, and for the trained network the interval excludes 90%. It flagged {{vo_hflag_resnet}}, {{vo_hflag_mini}} and {{vo_hflag_ecgf}} per 100 outpatients without SHD.

![Figure 4](results/figures/fig_repair.png)

Figure 4. The trained network's threshold recalibrated on n outpatients with known diagnoses (x axis, with the mean number of cases among them) and evaluated on the other half of the outpatients, {{lad_draws}} repetitions with a new split each. Solid line: true positives per 100 patients with SHD; dashed line: false positives per 100 patients without SHD. Lines are means and bands the middle 80% of the repetitions; x = 0 is the inpatient threshold, and the dotted line marks 90.

In the clinic of 1,000 outpatients, the recalibrated threshold would refer {{k_flagged_lad100}} for echocardiography and miss {{k_missed_lad100}} of the {{k_ill_out}} patients with SHD, against {{k_flagged_out}} and {{k_missed_out}} with the inpatient threshold: about {{echo_per_extra_ill}} additional echocardiograms per additional case detected. At a decision threshold of 10%, its net benefit per 1,000 outpatients was {{nb10_refit}}, against {{nb10_all}} for echocardiography for every outpatient and {{nb10_in}} for the inpatient threshold; at 5% echocardiography for every outpatient had the highest net benefit, and at 20% the inpatient threshold (supplement, Table S7d). Class-conditional conformal prediction [23,24], with which this study began, gives the threshold of section 2.5; its guarantee requires new patients exchangeable with the calibration sample, which outpatients were not with inpatients (supplement S1.19).

## 4. Discussion

### 4.1 Principal findings

A PPV recomputed by Bayes' theorem at a clinic's true prevalence fell outside the observed PPV's 95% confidence interval in {{ppv_outside_share}} of transfers, with a median difference of {{ppv_gap_median}} points, overestimating the PPV in some transfers and underestimating it in others; in controls the median difference was {{ppv_control_median}} points. These results suggest that, between hospitals and care settings, more than the prevalence changed. At Columbia, discrimination among outpatients was preserved, but outpatients with and without SHD both scored lower than inpatients. A fixed threshold therefore detected fewer cases and flagged fewer non-cases, and the observed PPV exceeded the value predicted from the inpatients' specificity. Case mix accounted for about a quarter of the decrease in sensitivity. At Chongqing the error was in the opposite direction, so its direction could not be inferred from the type of setting. The proportion of patients flagged, which requires no reference standard, fell at Columbia below any value that a lower prevalence could produce.

### 4.2 Comparison with prior work

These findings are consistent with the spectrum effect [2,3,4], here measured on the PPV, the quantity on which a clinic acts. The low ejection fraction model of Attia and colleagues kept its discrimination in a Russian population sample while its original cut-off detected a minority of cases [9]. Carter and colleagues [10] and PRESENT-SHD [11] found performance preserved, in transfers between hospitals and to a community cohort; the transfer studied here, from inpatient wards to outpatient clinics of one hospital, where cases have fewer findings and non-cases lower scores, differs from both, and we did not test that difference on their sites. A validation that reports the AUROC by care setting, as EchoNext's did [6], therefore does not establish the sensitivity, specificity or PPV that a clinic will obtain at a fixed threshold. Methods for label shift [5,20] assume that the distribution of ECGs within each diagnosis is unchanged; at Columbia this assumption did not hold, and the correction estimated a prevalence near zero. This study did not vary the number of labelled cases independently, so it does not measure how that number limits a threshold's precision [25]. Supplement S3.7 gives this comparison in full.

### 4.3 Strengths and limitations

The {{ppv_gap_cells}} transfers span shifts between care settings, between hospitals in different countries and between corpora, with controls drawn from the source population. At Columbia, three models of different provenance were evaluated on the same patients with an untrained negative control, and threshold recalibration was evaluated on outpatients distinct from those on whom it was set.

The datasets were selected by availability: they are public collections in which an observed PPV can be computed after a shift, and they do not represent the settings in which ECG models are deployed, so the proportion of transfers in which prevalence adjustment failed does not estimate how often it fails in practice. In the two infarction transfers the definition of infarction changed with the population, from an infarct pattern on the tracing in PTB-XL to mostly old infarcts at Shandong and acute infarction in a discharge diagnosis after angiography at Chongqing. Their PPV gaps therefore combine a change of label definition with a change of population, equipment and period of recording, and describe what a clinic would see on adopting a model whose label is defined differently from its own, not the effect of a change of population alone.

The SHD result comes from one hospital system and two care settings, and its magnitude in a second system is unknown. The three Columbia models were fitted on the same EchoNext patients, so they are not independent tests of the mechanism. Outpatients and inpatients also differ in recording period, although the decrease persisted within each band of years. The size of the decrease depends on the population in which the threshold is set: set on every validation patient, the threshold detected {{va_caught_resnet_out}} per 100 outpatients with SHD instead of {{caught_resnet_out}}. The {{vo_n}} separate outpatients form one group, which cannot separate sampling variation from systematic error in its shortfall from 90%. Negatives are defined with no time limit; excluding those recorded more than a year before their echocardiogram, the trained network flagged {{unmeas_flag_resnet_out}} per 100 outpatients without SHD instead of {{flagged_resnet_out}}, so specificity and the predictive values depend on this definition.

With four groups in each Columbia block, the intervals across transfers are imprecise. The decision threshold of 10% is an assumption, net benefit counts true and false positives, not clinical outcomes or costs, and the reader's decisions on referred patients were not measured. Supplement S3.7 gives these limitations in full.

### 4.4 Implications for practice and research

For a site adopting a model validated elsewhere, a PPV recomputed from a vendor's sensitivity and specificity can be in error, in either direction, by more than a margin relevant to clinical decisions. The PPV can instead be measured in the site's own patients with known diagnoses at the threshold to be deployed, in a sample large enough that the confidence interval is narrower than that margin. Before any diagnosis is available, a proportion flagged below the lowest value any prevalence allows indicates that the score distribution has shifted. A correction that requires no local diagnosis assumes that the site's non-cases resemble those of the source population; among Columbia's outpatients they did not.

A site can retain the delivered threshold and measure the resulting PPV, or recalibrate the threshold on its own patients to obtain a sensitivity near 90%, at the cost of echocardiography for most of its outpatients without SHD, and then evaluate the recalibrated threshold on patients it was not set on. The appropriate choice depends on the harm of a missed valve lesion or cardiomyopathy relative to that of an additional echocardiogram, which net benefit summarises at a stated decision threshold.

The SHD result requires replication in a second hospital system. Validation studies of ECG models could report sensitivity, specificity and PPV at a fixed threshold by care setting alongside the AUROC, and the proportion flagged at validation as a reference for monitoring after deployment.

This is a retrospective measurement study on public, de-identified data. Nothing here is intended to guide the care of any patient.

## Declarations

**Competing interests.** The author has worked at Idoven, which develops ECG analysis software, and at Anumana, which commercialises Mayo Clinic's AI-ECG work, including the lineage behind the hyperkalemia study [1]; the studies cited as [9] and [10] also come from Mayo Clinic. Idoven had no part in the design, conduct, funding or reporting of this study. None of either company's data, models or software were used.

**Funding.** None.

**Data.** EchoNext is distributed by PhysioNet under its Restricted Health Data License; its tracings and the per-record scores computed from them stay outside this repository, which holds counts and aggregate figures only. EchoNext v1.1.1's files match their published SHA-256 sums (`results/echonext_provenance.json`); the date they were downloaded is not recorded. PTB-XL and the Shandong and Chongqing datasets are public and were downloaded on 23 August 2026; Chongqing's MD5 sums are in `scripts/fetch_open_corpora.sh`, and the PTB-XL and Shandong files were not checksummed. The Challenge 2021 corpora and their sources are listed in `docs/data.md`.

**Ethics.** This is a secondary analysis of publicly released, de-identified ECG data, and it involved no contact with patients. EchoNext: the DISCOVERY trial and its analyses were approved by the Institutional Review Board at NewYork-Presbyterian Hospital/Columbia University Irving Medical Center, as its PhysioNet page states; no approval number is given there. EchoNext is released under PhysioNet's Restricted Health Data License, whose terms forbid any attempt at re-identification. PTB-XL: the Institutional Ethics Committee approved publication of the anonymous data in an open-access database, reference PTB-2020-1. Shandong: approved by the Institutional Review Board of Shandong Provincial Hospital, with the requirement for individual patient consent waived and public sharing permitted after de-identification; no approval number is given. Chongqing: approved by the ethics committee of the First Affiliated Hospital of Chongqing Medical University, approval number 2024-256-01, with written informed consent waived. No attempt was made to re-identify any patient.

**Code and numbers.** Every number in this report is read from a file under `results/` by `tests/paper_values.py`, and `tests/test_paper_numbers.py` fails if the text and the files part. Figures quoted from cited studies are the exception: that test lists them by source, and they were checked against the sources by hand. The EchoNext figures rest on scores that may not be committed; `results/echonext_clinical.json` records their SHA-256 sums, and `tests/test_echonext_data.py`, run where EchoNext and the scores are installed, rebuilds the file from them. The supplement gives the commands.

## References

1. Harmon DM, Liu K, Dugan J, et al. Validation of noninvasive detection of hyperkalemia by artificial intelligence-enhanced electrocardiography in high acuity settings. *Clin J Am Soc Nephrol* 2024;19(8):952-958. doi:10.2215/CJN.0000000000000483.
2. Usher-Smith JA, Sharp SJ, Griffin SJ. The spectrum effect in tests for risk prediction, screening, and diagnosis. *BMJ* 2016;353:i3139. doi:10.1136/bmj.i3139.
3. Ransohoff DF, Feinstein AR. Problems of spectrum and bias in evaluating the efficacy of diagnostic tests. *N Engl J Med* 1978;299(17):926-930. doi:10.1056/NEJM197810262991705.
4. Leeflang MMG, Rutjes AWS, Reitsma JB, et al. Variation of a test's sensitivity and specificity with disease prevalence. *CMAJ* 2013;185(11):E537-E544. doi:10.1503/cmaj.121286.
5. Saerens M, Latinne P, Decaestecker C. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Comput* 2002;14(1):21-41. doi:10.1162/089976602753284446.
6. Poterucha TJ, Jing L, Ricart RP, et al. Detecting structural heart disease from electrocardiograms using AI. *Nature* 2025;644(8075):221-230. doi:10.1038/s41586-025-09227-0. Table 2 gives the AUROC by care setting on the multicentre test set; Supplementary Table 15 gives the mini-model's AUROC on the public set.
7. Otabor E, Hassan A, Okunlola A, et al. Transportability of an artificial intelligence electrocardiography model for structural heart disease: external validation and recalibration of EchoNext-Mini in MIMIC-IV. *Eur Heart J Digit Health* 2026;7(8):ztag147. doi:10.1093/ehjdh/ztag147.
8. Poterucha TJ, Hughes JW, Brener MI, et al. AI-ECG detection of structural heart disease in the community setting: transportability and spectrum effects in the PREVUE-VALVE study. *J Am Coll Cardiol* 2026;88(7):752-764. doi:10.1016/j.jacc.2026.06.013.
9. Attia IZ, Tseng AS, Benavente ED, et al. External validation of a deep learning electrocardiogram algorithm to detect ventricular dysfunction. *Int J Cardiol* 2021;329:130-135. doi:10.1016/j.ijcard.2020.12.065.
10. Carter RE, Johnson PW, Strom JB, et al. Multisite, external validation of an AI-enabled ECG algorithm for detection of low ejection fraction. *JACC Adv* 2026;5(2):102537. doi:10.1016/j.jacadv.2025.102537.
11. Dhingra LS, Aminorroaya A, Sangha V, et al. Ensemble deep learning algorithm for structural heart disease screening using electrocardiographic images. *J Am Coll Cardiol* 2025;85(12):1302-1313. doi:10.1016/j.jacc.2025.01.030.
12. Pignet A, Klein J, Robin G, et al. Robust sensitivity control in digital pathology via tile score distribution matching. In: *Medical Image Computing and Computer Assisted Intervention, MICCAI 2025*. Lecture Notes in Computer Science. Cham: Springer; 2025:565-574. doi:10.1007/978-3-032-04978-0_54.
13. Adhikary S, Chabi N, Mastmeyer A. Bound-aware per-organ recall risk control for multi-organ CT segmentation under clinical domain shift. arXiv:2608.18193 [preprint]. 2026.
14. de Vries CF, Colosimo SJ, Staff RT, et al. Impact of different mammography systems on artificial intelligence performance in breast cancer screening. *Radiol Artif Intell* 2023;5(3):e220146. doi:10.1148/ryai.220146.
15. Wagner P, Strodthoff N, Bousseljot R-D, et al. PTB-XL, a large publicly available electrocardiography dataset. *Sci Data* 2020;7:154. doi:10.1038/s41597-020-0495-6.
16. Liu H, Chen D, Chen D, et al. A large-scale multi-label 12-lead electrocardiogram database with standardized diagnostic statements. *Sci Data* 2022;9:272. doi:10.1038/s41597-022-01403-5.
17. Du X, Liu Y, Wang L, et al. A large-scale 12-lead electrocardiogram dataset for acute coronary syndrome prediction containing 19,955 ECGs. *Sci Data* 2026;13:1009. doi:10.1038/s41597-026-07278-0.
18. Reyna MA, Sadr N, Perez Alday EA, et al. Will two do? Varying dimensions in electrocardiography: the PhysioNet/Computing in Cardiology Challenge 2021. In: *2021 Computing in Cardiology (CinC)*. IEEE; 2021:1-4. doi:10.23919/CinC53138.2021.9662687.
19. Li J, Aguirre AD, Moura Junior V, et al. An electrocardiogram foundation model built on over 10 million recordings. *NEJM AI* 2025;2(7). doi:10.1056/AIoa2401033.
20. Alexandari A, Kundaje A, Shrikumar A. Maximum likelihood with bias-corrected calibration is hard-to-beat at label shift adaptation. In: *Proceedings of the 37th International Conference on Machine Learning*. PMLR 2020;119:222-232.
21. Steyerberg EW, Borsboom GJ, van Houwelingen HC, et al. Validation and updating of predictive logistic regression models: a study on sample size and shrinkage. *Stat Med* 2004;23(16):2567-2586. doi:10.1002/sim.1844.
22. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making* 2006;26(6):565-574. doi:10.1177/0272989X06295361.
23. Angelopoulos AN, Bates S. Conformal prediction: a gentle introduction. *Found Trends Mach Learn* 2023;16(4):494-591. doi:10.1561/2200000101.
24. Vovk V. Conditional validity of inductive conformal predictors. In: *Proceedings of the Asian Conference on Machine Learning*. PMLR 2012;25:475-490.
25. Han W, Qu L. The label complexity of class-conditional coverage under distribution shift. arXiv:2607.18088 [preprint]. 2026.
26. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378. doi:10.1136/bmj-2023-078378.
27. Wilson EB. Probable inference, the law of succession, and statistical inference. *J Am Stat Assoc* 1927;22(158):209-212. doi:10.1080/01621459.1927.10502953.
28. Holm S. A simple sequentially rejective multiple test procedure. *Scand J Stat* 1979;6:65-70. doi:10.2307/4615733.

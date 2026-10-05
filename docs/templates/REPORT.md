# Carrying an ECG-AI threshold from inpatients to outpatients: a retrospective measurement of three structural heart disease models at one hospital

Ruben Abbou · October 2026

## Abstract

**Background.** An ECG model for structural heart disease gives each patient a score, and a cut-off on that score, the threshold, decides who is flagged for an echocardiogram. The threshold is set on patients who already had an echocardiogram, most of them inpatients, and the model is then offered for outpatients. Published validations report the AUROC by care setting, the chance that a randomly chosen ill patient scores higher than a randomly chosen healthy one, and find it stable. The AUROC does not say how many ill outpatients the inpatient threshold catches.

**Methods.** EchoNext holds 100,000 ECGs from Columbia, each paired with an echocardiogram. Three models scored them: a network trained here, the published EchoNext mini-model and the ECGFounder foundation model. For each, the threshold was set on {{cal_n}} inpatients to catch 90% of those with moderate or worse structural heart disease, and applied unchanged to {{out_n}} outpatients, {{out_prev}} of them ill. It was then set again on 100 outpatients with known diagnoses. An infarction model carried from a German ECG database to two Chinese hospitals is reported beside it.

**Results.** Of 100 ill outpatients, the inpatient threshold caught {{caught_resnet_out}} and missed {{missed_resnet_out}} for the trained network; the mini-model caught {{caught_mini_out}} and ECGFounder {{caught_ecgf_out}}. It also flagged fewer healthy patients: {{flagged_resnet_out}} of 100 healthy outpatients, against {{flagged_resnet_in}} of 100 healthy inpatients. The AUROC moved only from {{auroc_resnet_in}} to {{auroc_resnet_out}}: the model told ill from healthy outpatients almost as well, and the threshold, unchanged, flagged fewer healthy patients at the cost of missing more ill ones. Set again on 100 outpatients, about {{lad100_ill_resnet}} of them ill, the threshold caught {{lad100_sens_resnet}} of the ill on average and flagged {{lad100_hflag_resnet}} of 100 healthy outpatients.

**Conclusion.** A threshold validated on inpatients does not keep its sensitivity in outpatients, and the AUROC does not show it. A clinic adopting such a model has to set the threshold on its own outpatients with known diagnoses, and catching 90 of 100 ill outpatients means sending most of the healthy ones for an echocardiogram.

## 1. Introduction

A threshold validated on inpatients catches far fewer ill outpatients, and nothing in an AUROC table warns of it. Earlier work shows thresholds failing between populations, not between the care settings of one hospital.

An ECG model returns a score for each patient, and a threshold on that score decides who is flagged. The threshold is usually set on validation patients to catch a stated share of the ill, often 90%; that share is the sensitivity, and the share of the healthy left unflagged is the specificity. Both belong to the threshold and to the patients it was set on. A test meets different ill and different healthy patients in another population, so its sensitivity and specificity change there; this is called the spectrum effect [1,2]. For an ECG model of structural heart disease, the patients on whom the threshold is set are those who had an echocardiogram, and in a hospital most of them are inpatients.

EchoNext is a public set of 100,000 ECGs from Columbia, each paired with an echocardiogram [3]. Its authors report an AUROC of 84.3 for outpatients and 84.1 for inpatients, and describe consistent performance across care settings. Their table by care setting reports the AUROC and other measures of how well the model separates ill from healthy patients, but no sensitivity or specificity. At another hospital, the published mini-model separated almost as well, with an AUROC of 0.790 against 0.820 on the EchoNext test patients [4]. In a community cohort of people aged 65 to 85 the AUROC fell to 71%, from 83% in hospital, which the authors attribute to milder disease [5].

Thresholds that do not travel are not new. The Mayo Clinic model for an ejection fraction of 35% or less still separated ill from healthy well in a Russian population sample, with an AUROC of 0.82, yet the cut-off chosen in its original study caught only 26.9% of those with the condition; its authors concluded that each population may need its own cut-off [6]. The opposite has been seen too: across four American sites and 13,960 patients, a low-ejection-fraction model at a cut-off fixed in advance was 84.5% sensitive and 83.6% specific, with no difference between sites [7]. In pathology, CT and screening mammography, groups have reset cut-offs on a few local cases [8,9,10]. For structural heart disease, no study has measured what happens to a fixed threshold when one hospital's model moves from its inpatients to its outpatients. This study measures it in patients caught and missed, on three models, together with what it costs to set the threshold again.

An infarction model carried from a German database to two Chinese hospitals is reported beside it, as a case where the model's own separation of ill from healthy changes too.

## 2. Methods

Thresholds were set on Columbia inpatients and applied, without change, to Columbia outpatients; then they were set again on outpatients with known diagnoses. Every outcome is counted per 100 ill and per 100 healthy patients.

### 2.1 Patients and ECGs

EchoNext v1.1.1 holds 100,000 ECGs from Columbia University Irving Medical Center, each recorded within a year before a transthoracic echocardiogram [3]. The condition studied is moderate or worse structural heart disease on echocardiography, which EchoNext defines as any of eleven findings: a left ventricular ejection fraction of 45% or less; a left ventricular wall thickness of 1.3 cm or more; moderate or worse aortic stenosis, aortic regurgitation, mitral regurgitation, tricuspid regurgitation or pulmonary regurgitation; moderate or worse right ventricular systolic dysfunction; a moderate or large pericardial effusion; a pulmonary artery systolic pressure of 45 mmHg or more; and a tricuspid regurgitation peak velocity of 3.2 m/s or more. A patient with any of the eleven is called ill here, and one with none is called healthy.

EchoNext divides its patients into three groups. The models learned from the {{train_n}} ECGs of the first. The thresholds were set on the {{cal_n}} inpatient ECGs of the second, {{cal_prev}} of them from ill patients; these are called the calibration inpatients below. Everything was then measured on the third: {{out_n}} outpatient ECGs ({{out_prev}} ill), and beside them {{in_n}} other inpatient ECGs ({{in_prev}} ill) and {{em_n}} emergency ECGs ({{em_prev}} ill). Each patient contributes one ECG and belongs to one group only. EchoNext records the care setting of each ECG; nothing else separates the cohorts.

### 2.2 Models

Four models score every ECG. The first is a deep neural network (a residual network) trained on EchoNext for this study. The second is the published EchoNext mini-model, run on its authors' weights with nothing changed; on the EchoNext test patients it reaches an AUROC of {{auroc_mini_all}}, the figure reported for it there [4]. The third is ECGFounder, a network pre-trained on more than ten million ECGs from another hospital [11]. Its network was kept as published; it turns each ECG into a list of numbers, and a simple statistical formula (a logistic regression) fitted on EchoNext turns that list into a score for each finding. The fourth is the first network with its weights left at random, read through the same kind of formula: a floor that any useful model must beat. The results compare the first three. The [supplement](SUPPLEMENT.md) gives the architectures, how the published weights were read, and the statistics behind every method below.

### 2.3 The threshold and the second threshold

For each model, the threshold was placed among the calibration inpatients so that 90% of the ill scored at or above it. A patient above it is flagged for an echocardiogram, and a patient below it is cleared. With few ill patients the threshold is placed slightly lower than the exact 90% point, so that it errs toward catching more (supplement S1.1).

A second, higher threshold was placed among the healthy calibration inpatients, so that 90% of them scored at or below it. Patients above the second threshold are flagged with no reader. Patients between the two thresholds are flagged too, but a person reads their ECG before anything else happens. Patients below the first threshold are cleared, whatever the second threshold does. So the second threshold does not change who is caught or missed: it decides which flagged patients a person reads first. The supplement gives its statistical name and construction, and a third design the main text does not use.

### 2.4 What was counted

For 100 ill patients: caught, meaning flagged by the first threshold, and missed. For 100 healthy patients: cleared and flagged. With the second threshold, the flagged split into those flagged with no reader and those sent to a reader. The positive predictive value is the share of flagged patients who are ill, and the negative predictive value the share of cleared patients who are healthy. Both depend on the prevalence, the share of patients who are ill, and are given at the outpatients' own prevalence and at 10% and 5%. The figures at 10% and 5% assume that sensitivity and specificity stay the same at those prevalences, an assumption a companion study measured and found off by several points [14]. A range in brackets after a percentage is its 95% confidence interval; a range across repetitions is named as such.

To see how well each model separates ill from healthy in each setting, the AUROC was computed, and Figure 2 draws every possible threshold as a curve of ill caught against healthy flagged. The three models were compared on the same outpatients.

Milder disease among outpatients could explain part of the loss. To measure how much, a formula fitted on the ill patients of both settings estimated each patient's chance of being caught from their findings, ejection fraction, age, sex and setting. Applied to the ill inpatients as if they were outpatients, it gives the outpatients' sensitivity had they carried the inpatients' case mix (supplement S1.6). Differences between women and men and between age groups were tested with a correction for running six tests at once (supplement S1.8); they are exploratory.

To measure what setting the threshold again costs, the outpatients were split at random into two halves. 100 patients with known diagnoses were picked from the first half, both thresholds were set again on them, and the result was read on the second half. This was repeated 200 times with a new split each time, and each repetition noted how many ill patients its 100 held.

### 2.5 Infarction across three hospitals

A deep neural network was trained on PTB-XL, a German research database of ECGs recorded between 1989 and 1996 [15]. Its threshold was set to catch 90% of infarctions in half of PTB-XL's held-out patients and applied unchanged; this was repeated with 200 different random halves, and each figure is the average, to Shandong Provincial Hospital [16] and to the First Affiliated Hospital of Chongqing Medical University [17]. At PTB-XL and Shandong an infarction is read on the tracing, mostly as an old one; at Chongqing it is an acute infarction named in the discharge diagnosis of patients who all had coronary angiography. The supplement repeats the design across five databases, each in turn setting the threshold that is then applied to the other four, on five rhythm and conduction diagnoses: sinus rhythm, atrial fibrillation, left bundle-branch block, right bundle-branch block and first-degree atrioventricular block.

## 3. Results

The inpatient threshold caught {{caught_range}} of 100 ill outpatients for each of the three models, while the models separated ill from healthy outpatients about as well as inpatients. Catching 90 again means flagging most healthy outpatients.

### 3.1 The threshold catches {{caught_resnet_out}} of 100 ill outpatients

Set to catch 90% of the ill calibration inpatients, the trained network's threshold caught {{sens_resnet_in}} of the ill among the other inpatients (95% confidence interval {{sens_resnet_in_ci}}) and {{sens_resnet_out}} of ill outpatients ({{sens_resnet_out_ci}}). The published mini-model caught {{sens_mini_out}} and ECGFounder {{sens_ecgf_out}} of the ill outpatients. Emergency patients sat between, at {{sens_resnet_em}}. Figure 1 counts the outcomes per 100 outpatients.

![Figure 1](results/figures/fig_patients.png)

Figure 1. What the inpatient threshold does to 100 ill and 100 healthy outpatients, for each model, with the threshold alone and with the second threshold. Blue: the right answer with no reader (an ill patient flagged, a healthy patient cleared). Grey: sent to a reader. Orange: the wrong answer (an ill patient missed, a healthy patient flagged). {{out_ill}} ill and {{out_healthy}} healthy outpatients.

Table 1. The trained network's inpatient threshold, per 100 patients of each group, from `results/echonext_clinical.json`.

| | Other inpatients | Outpatients |
|---|---|---|
| Ill, caught | {{caught_resnet_in}} | {{caught_resnet_out}} |
| Ill, missed | {{missed_resnet_in}} | {{missed_resnet_out}} |
| Healthy, flagged | {{flagged_resnet_in}} | {{flagged_resnet_out}} |
| With the second threshold: ill flagged with no reader | {{pl_alone_resnet_in}} | {{pl_alone_resnet_out}} |
| With the second threshold: ill sent to a reader | {{pl_def_resnet_in}} | {{pl_def_resnet_out}} |
| With the second threshold: healthy sent to a reader | {{pl_hdef_resnet_in}} | {{pl_hdef_resnet_out}} |
| With the second threshold: healthy flagged with no reader | {{pl_hflag_resnet_in}} | {{pl_hflag_resnet_out}} |

In a clinic seeing 1,000 outpatients at Columbia's outpatient prevalence of {{out_prev}}, the inpatient threshold sends {{k_flagged_out}} for an echocardiogram, finds {{k_found_out}} of the {{k_ill_out}} ill and misses {{k_missed_out}}. Of the outpatients it flags, {{ppv_resnet_out}} are ill; of those it clears, {{npv_resnet_out}} are healthy. At a screening prevalence of 10% those figures would be {{ppv_resnet_10}} and {{npv_resnet_10}}, and at 5%, {{ppv_resnet_5}} and {{npv_resnet_5}}, if sensitivity and specificity held there.

The second threshold sent {{pl_def_resnet_out}} of 100 ill outpatients and {{pl_hdef_resnet_out}} of 100 healthy outpatients to a reader, and left {{pl_hflag_resnet_out}} healthy outpatients in 100 flagged with no reader. In all, {{def_resnet_out}} of outpatients went to a reader, against {{def_resnet_in}} of inpatients.

The loss was not spread evenly over the eleven findings. A threshold set the same way for each finding on its own kept its sensitivity in outpatients for an ejection fraction of 45% or less ({{sens_lvef_out}}, against {{sens_lvef_in}} among inpatients) and for aortic stenosis ({{sens_as_out}}, against {{sens_as_in}}). For a wall thickness of 1.3 cm or more it fell from {{sens_lvwt_in}} to {{sens_lvwt_out}}, and for mitral regurgitation, tricuspid regurgitation and right ventricular dysfunction it fell by {{drop_other_range}} points (supplement, Table S3).

### 3.2 The model separates as well; the threshold sits elsewhere

The AUROC of the trained network was {{auroc_resnet_in}} among inpatients and {{auroc_resnet_out}} among outpatients. For the mini-model it moved from {{auroc_mini_in}} to {{auroc_mini_out}}, and for ECGFounder from {{auroc_ecgf_in}} to {{auroc_ecgf_out}}. Figure 2 draws, for every possible threshold, how many ill it catches against how many healthy it flags. The inpatient and outpatient curves nearly coincide, so the model separates the two groups about as well in both settings. The same threshold lands at a different point on that curve: among inpatients it caught {{caught_resnet_in}} ill in 100 and flagged {{flagged_resnet_in}} healthy in 100; among outpatients it caught {{caught_resnet_out}} and flagged {{flagged_resnet_out}}. Ill and healthy outpatients both scored lower than their inpatient counterparts.

![Figure 2](results/figures/fig_curve.png)

Figure 2. Ill patients caught against healthy patients flagged, per 100, for the trained network, among inpatients ({{in_n}}) and outpatients ({{out_n}}). Each curve runs through every possible threshold. The dots mark the threshold set on the calibration inpatients.

The curves also give the price of catching 90 of 100 ill outpatients when every outpatient's diagnosis is known: {{oracle_flag_resnet_out}} of 100 healthy outpatients flagged, against {{oracle_flag_resnet_in}} of 100 healthy inpatients; for the mini-model, {{oracle_flag_mini_out}} and {{oracle_flag_mini_in}}, and for ECGFounder, {{oracle_flag_ecgf_out}} and {{oracle_flag_ecgf_in}}. The threshold carried from the inpatients therefore misses {{extra_missed}} more ill outpatients in 100 than one set for 90% among outpatients, and flags {{extra_cleared}} fewer healthy ones.

### 3.3 Three models lose the same share

On the same ill outpatients, the trained network caught exactly as many as ECGFounder and {{diff_resnet_mini}} fewer than the mini-model. With 95% confidence, no pair of the three models differs by more than {{diff_bound}}. The fourth, untrained model caught more ill outpatients, {{sens_floor_out}}, only because it flagged far more healthy ones: {{flagged_floor_out}} in 100, against {{flagged_resnet_out}} for the trained network. Three models built three ways fall to the same level, so the loss does not belong to one model. All three learned from the same EchoNext patients, so they are not three independent tests of its cause.

### 3.4 Milder disease explains about a quarter of the fall

The ill outpatients were ill more mildly than the ill inpatients. Among the ill, the median number of the eleven findings was {{sev_findings_in}} for the calibration inpatients and {{sev_findings_out}} for outpatients, and the median ejection fraction {{sev_lvef_in}} against {{sev_lvef_out}}. The trained network's sensitivity fell by {{mix_gap_pts_resnet}} points from inpatients to outpatients. Had the ill outpatients carried the inpatients' findings, ejection fraction, age and sex, it would have been {{mix_cov_resnet}} instead of {{sens_resnet_out}}: {{mix_pts_resnet}} of the {{mix_gap_pts_resnet}} points, {{mix_share_resnet}} of the fall ({{mix_share_resnet_ci}}). For the other two models the share was {{mix_share_mini}} and {{mix_share_ecgf}}. Most of the fall happens among patients with the same findings, ejection fraction, age and sex.

Among the ill outpatients, the trained network caught {{sub_women}} of women ({{sub_women_n}}) and {{sub_men}} of men ({{sub_men_n}}), and {{sub_young}} of patients aged 18 to 49 ({{sub_young_n}}) against {{sub_old}} of those aged 80 and over. Both differences were larger than chance would explain after correcting for six tests (p = {{p_sex_resnet}} for sex and {{p_age_resnet}} for age), and the other two models showed the same pattern (corrected p no higher than {{p_max_others}}). These are exploratory readings at one hospital.

### 3.5 Nothing visible without diagnoses warns of the loss

Without diagnoses, a clinic sees only how many patients the model flags and how many it sends to a reader. From inpatients to outpatients, the share flagged by the trained network fell from {{lf_flag_in}} to {{lf_flag_out}}, and the share sent to a reader from {{def_resnet_in}} to {{def_resnet_out}}. A clinic with fewer ill patients would see the same fall, so it looks like reassurance; it does not show the {{missed_resnet_out}} ill outpatients in 100 who were missed.

### 3.6 Setting the threshold again on outpatients restores sensitivity and flags most healthy ones

Set again on 100 outpatients with known diagnoses, the threshold caught {{lad100_sens_resnet}} of the ill on average over 200 repetitions, against {{lad0_sens_resnet}} with the inpatient threshold on the same patients (Figure 3). The 100 held {{lad100_ill_resnet}} ill patients on average, between {{lad100_ill_min}} and {{lad100_ill_max}} across the repetitions. In {{lad100_below_resnet}} of the repetitions the new threshold still caught fewer than 90% of the ill outpatients it had not seen. The healthy paid: it flagged {{lad100_hflag_resnet}} of 100 healthy outpatients on average, against {{flagged_resnet_out}} with the inpatient threshold. The mini-model and ECGFounder behaved alike, catching {{lad100_sens_mini}} and {{lad100_sens_ecgf}} of the ill and flagging {{lad100_hflag_mini}} and {{lad100_hflag_ecgf}} of 100 healthy. With 200 outpatients the trained network caught {{lad200_sens_resnet}} and flagged {{lad200_hflag_resnet}} of 100 healthy.

![Figure 3](results/figures/fig_repair.png)

Figure 3. The trained network's threshold set again on outpatients with known diagnoses, against how many there were, with the average number of ill patients among them. The result is read on other outpatients. Lines: the average over 200 repetitions. Bands: the middle 80% of the repetitions.

In the same clinic of 1,000 outpatients, the new threshold would send {{k_flagged_lad100}} for an echocardiogram and miss {{k_missed_lad100}} of the {{k_ill_out}} ill. Part of that cost comes from setting it on only about {{lad100_ill_resnet}} ill patients: with every outpatient's diagnosis known, catching 90 of 100 ill flags {{oracle_flag_resnet_out}} of 100 healthy, while a threshold set on a small sample is placed with a safety margin, slightly lower than the sample's own 90% point, so that it still catches 90% on average when the sample happens to be unrepresentative, and so it flags more.

### 3.7 Infarction: across hospitals, the model's separation changes too

The infarction threshold, set to catch 90% of infarctions at PTB-XL, caught {{mi_sens_ptbxl}} there (95% confidence interval {{mi_sens_ptbxl_ci}}), {{mi_sens_acs}} at Chongqing ({{mi_sens_acs_ci}}) and {{mi_sens_sph}} at Shandong ({{mi_sens_sph_ci}}). Its specificity was {{mi_spec_ptbxl}} at PTB-XL, {{mi_spec_acs}} at Chongqing and {{mi_spec_sph}} at Shandong.

Table 2. The PTB-XL threshold at three hospitals, from `results/infarction_sites.json` and `results/outcomes.json`.

| | PTB-XL, Germany | Chongqing, China | Shandong, China |
|---|---|---|---|
| Tracings | {{mi_n_ptbxl}} | {{mi_n_acs}} | {{mi_n_sph}} |
| Infarctions | {{mi_pos_ptbxl}} | {{mi_pos_acs}} | {{mi_pos_sph}} |
| Infarctions caught, per 100 | {{mi_caught_ptbxl}} | {{mi_caught_acs}} | {{mi_caught_sph}} |
| Others flagged, per 100 | {{mi_flagged_ptbxl}} | {{mi_flagged_acs}} | {{mi_flagged_sph}} |
| AUROC | {{mi_auroc_ptbxl}} | {{mi_auroc_acs}} | {{mi_auroc_sph}} |

At Chongqing the threshold caught fewer infarctions and flagged more other patients at once, and the AUROC fell from {{mi_auroc_ptbxl}} to {{mi_auroc_acs}}: the model separated acute infarctions from other angiography patients worse than it separated PTB-XL's infarct patterns, a different definition of the disease. At Shandong, where most infarctions are old ones read on the tracing as at PTB-XL, the AUROC rose to {{mi_auroc_sph}} and the threshold caught more infarctions than it was set for; even the low end of its confidence interval lies above 90%. The same threshold erred in one direction at one hospital and in the other at the next.

## 4. Discussion

A threshold set on inpatients caught {{caught_range}} of 100 ill outpatients in one hospital, for three models built three ways, while the AUROC stayed where it was. A clinic adopting such a model gets a different trade from the one validated, and only its own patients with known diagnoses show which.

The model did not get worse at telling ill from healthy outpatients; the threshold landed at a different point of the same curve. Ill and healthy outpatients both scored lower than their inpatient counterparts, so a fixed threshold caught fewer ill and flagged fewer healthy at once. The AUROC compares ill with healthy patients within one setting, and because both groups moved together it barely changed. A validation that reports the AUROC by care setting, as EchoNext's did [3], therefore sees nothing. A clinic that receives a model validated on inpatients is receiving {{caught_resnet_out}} ill caught and {{flagged_resnet_out}} healthy flagged per 100, not the trade advertised.

This is the spectrum effect [1,2], measured inside one hospital on three models. It agrees with Attia and colleagues' low ejection fraction model, whose original cut-off caught 26.9% of ill patients in a Russian population sample where its AUROC was still 0.82 [6]. Carter and colleagues found sensitivity holding at four sites with comparable clinical populations [7]. One difference is the move here from hospital beds to clinics, where disease is milder and the healthy are healthier; this study does not test it against Carter's sites. Milder disease explained about a quarter of the fall, and most of it happens among patients with the same findings, ejection fraction, age and sex. What a clinic sees without diagnoses, the share flagged and the share sent to a reader, fell as it would with fewer ill patients.

The remedy is to set the threshold again on the clinic's own patients, and it is not free. With every outpatient's diagnosis known, catching 90 of 100 ill outpatients flags {{oracle_flag_resnet_out}} of 100 healthy ones. With 100 outpatients of known diagnosis, about {{lad100_ill_resnet}} of them ill, the new threshold reaches 90% on average, still falls short in about one repetition in four, and flags {{lad100_hflag_resnet}} of 100 healthy. The number of ill patients sets the precision, not the number of ECGs: Han and Qu show that the ill patients needed grow fast as the precision asked for tightens [18], and about {{lad100_ill_resnet}} ill patients leave the result uncertain by several points. A clinic therefore faces a choice the inpatient validation hid: keep the delivered trade, or buy 90% sensitivity with echocardiograms for most of its healthy outpatients. Which is right depends on the cost of a missed valve lesion or cardiomyopathy against that of an echocardiogram, which this study does not measure.

The second threshold spends a reader's time; it catches no one the first threshold misses. Among the outpatients the first threshold flags, it sends to a reader half or more of the ill and nearly all of the healthy, those whose scores sit closest to the healthy patients'. Whether that helps depends on what the reader decides, which was not measured.

The infarction model gives the other case. Between hospitals the model's own separation changed, from an AUROC of {{mi_auroc_ptbxl}} at PTB-XL to {{mi_auroc_acs}} at Chongqing, where the disease is defined differently, and its threshold lost specificity as well as sensitivity. Within Columbia the model separated as well as before and the threshold failed anyway. Either way, the sensitivity a clinic gets is known only once it is measured on its own patients.

## 5. Limitations

- One hospital and two care settings. The EchoNext result has not been measured at a second hospital, so its size there is not known.
- The three models learned from the same EchoNext patients, and the network trained here was trained once. The mini-model was trained by its authors and ECGFounder's network elsewhere, with only its final regression fitted here, so a quirk of one training run cannot explain the agreement of all three.
- Outpatients were told apart from inpatients only by the care setting EchoNext records. Why the ill outpatients score lower is explained only in part, the quarter that milder disease accounts for.
- The threshold rule is the one stated, 90% of the ill in the group where the threshold is set. A vendor may set its threshold otherwise, and the size of the fall depends on where the threshold sits.
- What a reader does with a patient sent to them, and what a missed patient or an extra echocardiogram costs, were not measured, so no net clinical benefit is claimed.
- Predictive values at 10% and 5% assume that sensitivity and specificity hold in a screening population, which the main result shows they need not.
- At Chongqing the definition of infarction, the population, the recording equipment and the era change together, and the design does not separate them.

This is a retrospective measurement study on public, de-identified data. It is not a medical device and has no regulatory status, it involved no contact with patients, and nothing here is meant to guide the care of any patient.

## Declarations

**Competing interests.** The author has worked at Idoven, which develops ECG analysis software, and at Anumana, which commercialises Mayo Clinic's AI-ECG work; the studies cited as [6] and [7] come from Mayo Clinic. Idoven had no part in the design, conduct, funding or reporting of this study. None of either company's data, models or software were used.

**Funding.** None.

**Data.** EchoNext is distributed by PhysioNet under its Restricted Health Data License; its tracings and the per-record scores computed from them stay outside this repository, which holds counts and aggregate figures only. PTB-XL and the Shandong and Chongqing datasets are public, downloaded on 23 August 2026; access dates and checksums are in `scripts/fetch_open_corpora.sh` and `results/ingest_report.json`.

**Ethics.** This is a secondary analysis of publicly released, de-identified ECG data, and it involved no contact with patients. EchoNext is released under PhysioNet's Restricted Health Data License, whose terms forbid any attempt at re-identification. PTB-XL: the Institutional Ethics Committee approved publication of the anonymous data in an open-access database, reference PTB-2020-1. Shandong: approved by the Institutional Review Board of Shandong Provincial Hospital, with the requirement for individual patient consent waived and public sharing permitted after de-identification; no approval number is given. Chongqing: approved by the ethics committee of the First Affiliated Hospital of Chongqing Medical University, approval number 2024-256-01, with written informed consent waived. No attempt was made to re-identify any patient.

**Code and numbers.** Every number in this report is read from a file under `results/` by `tests/paper_values.py`, and `tests/test_paper_numbers.py` fails if the text and the files part. The supplement gives the commands.

## References

1. Usher-Smith JA, Sharp SJ, Griffin SJ. The spectrum effect in tests for risk prediction, screening, and diagnosis. *BMJ* 2016;353:i3139. doi:10.1136/bmj.i3139.
2. Ransohoff DF, Feinstein AR. Problems of spectrum and bias in evaluating the efficacy of diagnostic tests. *N Engl J Med* 1978;299(17):926-930. doi:10.1056/NEJM197810262991705.
3. Poterucha TJ, Jing L, Ricart RP, et al. Detecting structural heart disease from electrocardiograms using AI. *Nature* 2025;644:221-230. doi:10.1038/s41586-025-09227-0. Table 2 gives the AUROC by care setting.
4. Otabor E, Hassan A, Okunlola A, et al. Transportability of an artificial intelligence electrocardiography model for structural heart disease: external validation and recalibration of EchoNext-Mini in MIMIC-IV. *Eur Heart J Digit Health* 2026;7(8):ztag147. doi:10.1093/ehjdh/ztag147.
5. Poterucha TJ, Hughes JW, Brener MI, et al. AI-ECG detection of structural heart disease in the community setting: transportability and spectrum effects in the PREVUE-VALVE study. *J Am Coll Cardiol* 2026;88(7):752-764. doi:10.1016/j.jacc.2026.06.013.
6. Attia IZ, Tseng AS, Benavente ED, et al. External validation of a deep learning electrocardiogram algorithm to detect ventricular dysfunction. *Int J Cardiol* 2021;329:130-135. doi:10.1016/j.ijcard.2020.12.065.
7. Carter RE, Johnson PW, Strom JB, et al. Multisite, external validation of an AI-enabled ECG algorithm for detection of low ejection fraction. *JACC Adv* 2026;5(2):102537. doi:10.1016/j.jacadv.2025.102537.
8. Pignet A, Klein J, Robin G, et al. Robust sensitivity control in digital pathology via tile score distribution matching. arXiv:2502.20144, 2025.
9. Adhikary S, Chabi N, Mastmeyer A, et al. Bound-aware per-organ recall risk control for multi-organ CT segmentation under clinical domain shift. arXiv:2608.18193, 2026.
10. de Vries CF, Colosimo SJ, Staff RT, et al. Impact of different mammography systems on artificial intelligence performance in breast cancer screening. *Radiol Artif Intell* 2023;5(3):e220146. doi:10.1148/ryai.220146.
11. Li J, Aguirre AD, Moura V, et al. An electrocardiogram foundation model built on over 10 million recordings. *NEJM AI* 2025;2(7). doi:10.1056/AIoa2401033.
12. Vovk V. Conditional validity of inductive conformal predictors. ACML 2012, PMLR 25:475-490.
13. Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511.
14. Abbou R. A model's PPV, recomputed for your hospital, misses by five points and in either direction. [PPV.md](PPV.md), this repository, 2026.
15. Wagner P, Strodthoff N, Bousseljot R-D, et al. PTB-XL, a large publicly available electrocardiography dataset. *Sci Data* 2020;7:154. doi:10.1038/s41597-020-0495-6.
16. Liu H, Chen D, Chen D, et al. A large-scale multi-label 12-lead electrocardiogram database with standardized diagnostic statements. *Sci Data* 2022;9:272. doi:10.1038/s41597-022-01403-5.
17. Du X, Liu Y, Wang L, et al. A large-scale 12-lead electrocardiogram dataset for acute coronary syndrome prediction containing 19,955 ECGs. *Sci Data* 2026. doi:10.1038/s41597-026-07278-0.
18. Han W, Qu L. The label complexity of class-conditional coverage under distribution shift. arXiv:2607.18088, 2026.

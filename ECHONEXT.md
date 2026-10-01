# Structural heart disease from the ECG, calibrated on inpatients and read on outpatients

Asked for 90% coverage, a per-label conformal threshold calibrated on Columbia
inpatients covers 71.6% of Columbia outpatients who have moderate or worse
structural heart disease on echocardiography, and 98.1% of those who do not.
One hundred labelled outpatient ECGs bring the coverage of the ill back to
93.4%.

## The shift inside one hospital

A model that reads structural heart disease from the ECG learns from the
patients who had an echocardiogram, and in a hospital those are mostly
inpatients. A screening programme would run it on outpatients. In EchoNext's
validation and test splits, one ECG per patient, the composite of eleven
echocardiographic findings is present in 52.8% of inpatient ECGs and 26.7% of
outpatient ECGs; an ejection fraction of 45% or less, in 24.5% and 7.7%.

Calibration per label (Mondrian) is built for this case: each class gets its
own threshold, so a change in prevalence alone cannot move the coverage of
either class. A coverage of the ill that still falls on outpatients points to
a change in the ECGs of the ill themselves [belief: milder disease among outpatients is the likely reason; the
echocardiographic severity values that would show it are in the metadata and
have not been compared].

## What was measured

EchoNext v1.1.1 holds 100,000 ECGs from Columbia, each within a year before a
transthoracic echocardiogram, with eleven binary findings and their composite.
The thresholds are fitted on the 1,903 inpatient ECGs of the validation split
and applied unchanged to the test split: 1,059 outpatient ECGs, and the
emergency and inpatient ECGs beside them. No patient appears in two of the
roles (training, calibration, test); `transfer_cohorts` raises if one does.

Three ways of deciding are compared at the 90% level: one threshold set at 90%
sensitivity on the inpatients (plain), split conformal over all calibration
ECGs (pooled), and one conformal threshold per class (per-label).

Two arms ran. The study's ResNet, trained from scratch on the train split,
reaches an AUROC of 0.834 for the composite on the whole test split and 0.805
on outpatients. The same network frozen at random initialisation with one
logistic probe per label, the floor a pre-trained encoder has to clear,
reaches 0.792 and 0.758. Neither is the published EchoNext mini-model, whose
AUROC of 82.0% on this test split is the reference; its arm did not run (see
below).

## Findings, for the trained ResNet

- Per-label thresholds cover 71.6% of the ill outpatients and 98.1% of the
  healthy ones, and send 29.4% of outpatients to a human. On emergency ECGs the
  coverage of the ill is 86.2%.
- An ejection fraction of 45% or less holds up better: 87.1% of those
  outpatients are covered.
- Inside the outpatients, the coverage of the ill is 61.1% for women and 80.7%
  for men.
- The pooled threshold covers 74.5% of the ill outpatients for the composite
  and none of them for three rarer findings (aortic stenosis, aortic
  regurgitation, pericardial effusion), whose pooled quantile the healthy
  majority sets.
- On the half of the outpatients kept for evaluation, the inpatient thresholds
  cover 69.3% of the ill; refitted on 100 outpatient ECGs drawn from the other
  half, 93.4%, with 90.5% of the healthy covered.

The floor arm loses less coverage on outpatients (78.6%) because it sends more
of them to a human (44.2%). Every figure for both arms, per label and per care
context, is in `results/echonext_transfer.json` and
`results/echonext_coverage.csv`, and the one-page reports are in
`reports/transfer/`.

## The unit of the tracings

EchoNext's tracings have no physical unit. Each lead was median-filtered,
clipped at its 0.1st and 99.9th percentiles and standardised with a mean and a
standard deviation computed over the training set, so every tracing here is in
z-score at 250 Hz, as `results/echonext_provenance.json` records for all
100,000 of them. The authors' code package publishes those per-lead means and
standard deviations, in the integer counts of the GE MUSE export; it does not
state how many microvolts a count is, so the millivolt scale cannot be
recovered from the files. Moving a model trained here to another hospital
means replaying the same preprocessing on that hospital's ECGs, with that
hospital's own standardisation, which absorbs part of any difference between
electrocardiographs.

## What did not run

The protocol names two pre-trained arms that are absent: the published EchoNext
mini-model (weights and architecture from the authors' IntroECG repository) and
ECGFounder frozen with logistic probes. Both load third-party code or pickled
weights, which this run was not cleared to execute. Their cells are empty and
named in each report.

EchoNext is under PhysioNet's restricted licence. Its tracings, and the
per-record scores computed from them, stay outside this repository; the results
hold counts and aggregate figures only.

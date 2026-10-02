# Per-label conformal calibration of an ECG classifier for structural heart disease, fitted on inpatients and tested on outpatients

Asked for 90% coverage, a per-label conformal threshold calibrated on Columbia
inpatients covers 71.6% of Columbia outpatients who have moderate or worse
structural heart disease on echocardiography, and 98.1% of those who do not.
One hundred labelled outpatient ECGs bring the coverage of the ill back to
93.4%.

## The shift inside one hospital

A model that reads structural heart disease from the ECG learns from the
patients who had an echocardiogram, and in a hospital those are mostly
inpatients. A screening programme would run it on outpatients. In the cohorts
used here, one ECG per patient, the composite of eleven echocardiographic
findings is present in 53.2% of the 1,903 inpatient ECGs that calibrate the
thresholds and in 25.6% of the 1,059 outpatient ECGs that test them; an
ejection fraction of 45% or less, in 24.5% and 6.6%.

Calibration per label (Mondrian) is built for this case: each class gets its
own threshold, so a change in prevalence alone cannot move the coverage of
either class. A coverage of the ill that still falls on outpatients points to
a change in the ECGs of the ill themselves. Milder disease among outpatients is a
likely reason; the echocardiographic severity values that would show it are in
the metadata and have not been compared.

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

Four arms ran. The study's ResNet, trained from scratch on the train split,
reaches an AUROC of 0.834 for the composite on the whole test split and 0.805
on outpatients. The published EchoNext mini-model, run on its own weights with
nothing refitted, reaches 0.820, against the 82.0% its authors report on this
test split; the gap is under a tenth of a point. ECGFounder, a foundation model
pre-trained on more than ten million ECGs from another hospital, frozen with
one logistic probe per label fitted on the train split, reaches 0.824. The
study's ResNet frozen at random initialisation with the same probes, the floor
a pre-trained encoder has to clear, reaches 0.792.

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

## The four arms side by side

The three arms that discriminate best lose the same coverage on outpatients:
71.6% of the ill for the ResNet and for ECGFounder, 72.7% for the published
mini-model. A network trained here, one trained by the EchoNext authors and
one pre-trained elsewhere all fall to the same level, which points at the
outpatients' ECGs rather than at any one model.

| Arm | AUROC, whole test split | AUROC, outpatients | Ill outpatients covered | Healthy outpatients covered | Outpatients sent to a human | Ill emergency patients covered | Ill outpatients covered after 100 local labels |
|---|---|---|---|---|---|---|---|
| Study ResNet, trained | 0.834 | 0.805 | 71.6% | 98.1% | 29.4% | 86.2% | 93.4% |
| EchoNext mini-model, published | 0.820 | 0.795 | 72.7% | 98.2% | 32.3% | 85.5% | 97.6% |
| ECGFounder, frozen, probes | 0.824 | 0.791 | 71.6% | 97.7% | 32.2% | 82.9% | 95.3% |
| Random initialisation, frozen, probes | 0.792 | 0.758 | 78.6% | 97.2% | 44.2% | 84.4% | 88.8% |

Coverage is for the composite, with per-label thresholds calibrated on the
inpatients at the 90% level. The floor arm loses less coverage on outpatients
because it sends more of them to a human. Every figure for the four arms, per
label and per care context, is in `results/echonext_transfer.json` and
`results/echonext_coverage.csv`, and each arm has its one-page report in
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

## Reproduce

EchoNext's tracings stay outside the repository. Point `ECS_ECHONEXT_DIR` at
the distribution (default `~/data/echonext`); `ECS_ECHONEXT_DERIVED` (default
`~/data/echonext-derived`) and `ECS_EMBEDDING_STORE` (default
`~/data/ecg-embeddings`) receive what the scripts derive from it.

```bash
uv run python scripts/echonext_provenance.py   # results/echonext_provenance.json
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python scripts/echonext_transfer.py     # the coverage table and the reports
uv run pytest -m data tests/test_echonext_data.py
```

`echonext_transfer.py` scores an arm first when its scores are missing, which
for the trained ResNet is a training run on the Apple GPU or the CPU. The two
published arms read their weights from `data/weights/echonext_mini/weights.pt`
and `data/weights/ecgfounder/12_lead_ECGFounder.pth`, each checked against the
SHA-256 it had when it was fetched.

## How the published weights were run

The mini-model's weights come from the authors' IntroECG repository (commit
15233e93) and hold tensors only, so `torch.load` reads them with
`weights_only=True`. Its architecture is `EchoNextMini` in
`src/ecs/echonext_mini.py`, written from the shapes of the checkpoint rather
than copied, since the repository states no licence; every tensor of the
checkpoint loads into it, with none missing. It reads the tracing as
distributed and the seven tabular features the distribution ships already
standardised (sex, age, rates and intervals), which the other three arms do not
see. Scoring the validation and test splits took 2.9 seconds on the Apple GPU.

ECGFounder's checkpoint holds one numpy scalar beside its tensors, which
`weights_only=True` refuses on torch 2.2.2. It is read through an unpickler
that resolves five names (the tensor rebuild, an ordered dict, the numpy
scalar, its dtype and a byte codec) and refuses any other, so the file is never
unpickled in full. Its network is the authors' Net1D, vendored in
`third_party/ecgfounder`. The model card asks for 500 Hz, so each tracing is
resampled from 250 Hz by polyphase filtering; the amplitude stays in z-score,
since no millivolt scale exists to restore. The 1,024 features before the
dense head feed one logistic probe per label. Embedding the 82,543 training,
validation and test ECGs and fitting the probes took 332 seconds.

EchoNext is under PhysioNet's restricted licence. Its tracings, and the
per-record scores computed from them, stay outside this repository; the results
hold counts and aggregate figures only.

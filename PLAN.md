# Execution plan

Six working days: Mon 24 → Fri 28 Aug, plus Mon 31. Written the night of 22–23
August, when the scaffold and the corpora landed.

## Acceptance criteria

Each maps to a named test. Unchecked means unverified, not done.

| # | Criterion | Test |
|---|---|---|
| C-1 | THE PTB-XL loader SHALL return 21,799 records, 18,869 patients, 5,469 MI positives — and 5,288 with subendocardial-injury statements excluded. | `test_labels.py::TestPTBXL` |
| C-2 | THE SPH loader SHALL return 25,770 records, 24,666 patients, 260 MI positives, 233 of them chronic. | `test_labels.py::TestSPH` |
| C-3 | THE ACS loader SHALL return 17,960 labelled records, 17,018 patients, 1,151 OMI / 1,442 STEMI / 2,679 AMI. | `test_labels.py::TestACS` |
| C-4 | WHERE a calibration/test boundary is drawn, THE splitter SHALL place every record of a given patient on one side only. | pending — `test_splits.py` |
| C-5 | WHEN the calibrator is fitted at alpha on an exchangeable sample, THE empirical coverage SHALL fall within [1-alpha-0.012, 1-alpha+0.025]. | `test_conformal.py::TestCoverageGuarantee` |
| C-6 | WHEN a calibration sample is smaller than ceil(1/alpha)-1, THE quantile SHALL be +inf rather than a finite threshold. | `test_conformal.py::TestQuantile` |
| C-7 | WHEN the class prior changes between calibration and test, THE Mondrian thresholds SHALL hold class-conditional coverage at 1-alpha. | `test_label_shift.py::TestMondrianUnderPrevalenceShift` |
| C-8 | WHEN weights are uniform, THE weighted quantile SHALL equal the unweighted quantile exactly. | `test_conformal.py::TestWeightedQuantile` |
| C-9 | IF a weighted correction is reported, THEN the harness SHALL report its effective sample size alongside. | pending — `test_report.py` |
| C-10 | THE harness SHALL report every coverage figure as a mean over >= 100 calibration/test draws, with its standard deviation. | pending — `test_report.py` |
| C-11 | THE harness SHALL report coverage separately for MI and non-MI cases. | `metrics.class_conditional_coverage`, wired at C-10 |
| C-12 | THE results file SHALL record the encoder's pre-training corpora. | pending — `test_report.py` |
| C-13 | THE ingestion layer SHALL return, for every corpus, a float32 array of shape (N, 12, 5000): 10 seconds, 500 Hz, millivolts, leads ordered I, II, III, aVR, aVL, aVF, V1-V6. | pending — `test_ingest.py` |
| C-14 | THE ingestion layer SHALL apply an identical filter and scaling chain to every corpus, and SHALL emit, per corpus, the list of steps that could not be made identical. | pending — `test_ingest.py` |
| C-15 | WHEN a record contains a NaN or Inf sample, THE loader SHALL exclude it and report the excluded count per corpus. | pending — `test_ingest.py` |
| C-16 | WHEN PTB-XL is round-tripped through 250 Hz and back to 500 Hz, THE resulting coverage SHALL move by less than one third of the coverage gap attributed to dataset shift. | pending — `test_resample_control.py` |
| C-17 | THE encoder comparison SHALL include a frozen randomly-initialised encoder, reported alongside the pre-trained arms. | pending — `test_report.py` |
| C-18 | THE harness SHALL report per-task AUROC and AUPRC with bootstrapped confidence intervals, and SHALL use paired comparisons for any claim that one arm beats another. | pending — `test_report.py` |

## The ingestion contract

Every corpus is reduced to one canonical form before anything else touches it:
**float32, shape (N, 12, 5000) — ten seconds, 500 Hz, millivolts, leads in the
order I, II, III, aVR, aVL, aVF, V1-V6.** Records carrying a NaN or Inf sample
are dropped and counted. This is the form prescribed in arXiv:2602.17531, which
the ptbxl5d report already cites, so it is a convention with a citation behind
it rather than a house style.

**The rule that matters more than the values: the chain is identical for every
corpus.** Tuning preprocessing per dataset would raise each one's signal quality
and destroy the only quantity this study measures, because a difference produced
by my own pipeline is indistinguishable from one produced by the hospital. Where
a corpus cannot be made to match, the deviation is named in the results file
rather than absorbed silently.

Four deviations are known in advance and none can be fully removed:

- **Resampling.** PTB-XL and Shandong are natively 500 Hz; EchoNext is 250 Hz,
  INCART 257 Hz, PTB 1000 Hz. Upsampling invents detail and downsampling discards
  it, so resampling is itself a source of apparent shift. Polyphase resampling
  with anti-aliasing (`scipy.signal.resample_poly`) is the defensible choice, and
  C-16 measures what it costs instead of assuming it is free.
- **EchoNext cannot join the contract.** Its waveforms ship already median-
  filtered, percentile-clipped and normalised with a dataset-wide mean and
  standard deviation, so they are not in millivolts and cannot be returned to
  that scale. Worse for this purpose, that normalisation was computed across all
  splits including test. EchoNext is usable as a benchmark target, not as a
  cohort in a like-for-like shift comparison.
- **Lead order is not documented consistently.** The ACS-ECG paper states it two
  different ways in consecutive paragraphs, with aVR and aVL swapped. Order is
  read from each record's own header, never assumed.
- **Amplitude scaling differs at source** (PTB-XL in mV, Chapman at 4.88 uV per
  least-significant bit). Conversion happens once, in the loader, and the factor
  for each corpus is asserted by a test against a known record.

On filtering: the cited protocol prescribes none, and neither do we beyond what a
corpus already carried at source. Every filter added is a chance to help one
cohort more than another, so the honest default is the lightest chain all
corpora can share.


## Days

1. **Mon 24 — the ingestion contract, then the timing probe.** Build the canonical
   loader (C-13 to C-15) and patient-level splits (C-4). Baseline
   stopwatch out: that number sizes every experiment this week. Do not launch
   anything long before it.
2. **Tue 25 — abstention table on PTB-XL.** Conformal wired to the model, C-5
   verified in distribution, silence rate against set size. Reproduces known
   ground.
3. **Wed 26 — the break.** Calibrate on PTB-XL, measure on SPH then ACS-ECG.
   Publishable either way: a clear break is the result, a negligible one is an
   honest null.
4. **Thu 27 — three encoders on the same break.** ECG-FM (saw PTB-XL),
   ECGFounder (saw no public corpus), supervised-from-scratch. The gap between
   the first two is the contamination measurement.
5. **Fri 28 — exact correction vs estimated.** Mondrian against label-shift
   weighting, effective sample size on both (C-9), and a reasoned answer to
   which one this shift needed.
6. **Mon 31 — the page a cardiologist can read.** Plus the ten questions the
   object will trigger, answered in writing.

**Sacrifice order if it overruns:** fourth encoder, then ACS-ECG as second
target, then the third encoder branch. The day-6 page is never cut.

## Hardware

Laptop is a 2016 Intel Mac — 16 GB, ~32 GB free, and torch dropped macOS x86_64
wheels after 2.2.2, which is why that version is pinned. The Linux box (i7-8700,
12 threads, 391 GB free, no GPU) carries the heavy runs. Neither has CUDA, so
encoder inference cost is the schedule's real risk and day 1 measures it first.

## Rollback

Standalone repo; nothing outside it is modified. PTB-XL is read in place and
never written. `data/` and `results/*.json` are gitignored. Any unit reverts with
`git reset --hard` without touching the corpora.

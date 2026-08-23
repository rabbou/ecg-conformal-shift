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

## Days

1. **Mon 24 — harness + the timing probe.** Patient-level splits (C-4). Baseline
   1D-ResNet trained through on PTB-XL. Then 100 records through each encoder,
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

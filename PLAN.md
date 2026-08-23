# Execution plan

Six working days: Mon 24 → Fri 28 Aug, plus Mon 31. Written the night of 22–23
August, when the scaffold and the corpora landed.

## Why this object exists

The venture's bar is a clinician committed in writing by 31 October. The
institutional messages that open those conversations leave on 1 September, and
each one points at this repository as the proof that the sender can do the
work: a model that knows when it is wrong, measured across hospitals, with the
code open. Without it the consulting offer rests on a CV; with it, a
cardiologist or a hospital data-science lead can inspect the method instead of
taking it on trust. The object is therefore the gate on the whole September
sequence, which is why it outranks every other venture task until 31 August.
It is tracked as task T-001 in `~/Developer/lab/program/`.

## Corpus status on 23 August

| Corpus | Labels | Waveforms | Where |
|---|---|---|---|
| PTB-XL v1.0.3 | on disk | on disk, 500 Hz (2.6 GB) | `~/Developer/ptbxl5d/data/records500` |
| SPH / Shandong | on disk | on disk, 25,770 `.h5` (4.3 GB) | `data/sph/records` |
| ACS-ECG / Chongqing | on disk (`data/acs/CSV`) | **not on disk** | figshare 10.6084/m9.figshare.29925314, CC0, `ECG_row_data.zip` 1.28 GB, no account |

The Chongqing waveforms were never fetched: `CSV.zip` holds the label tables
only, and `fetch_open_corpora.sh` has no ACS entry. Day 1 starts the download
in the background (1.28 GB; a few minutes) before anything else, so that
Wednesday's second target exists. [verified 2026-08-23 — api.figshare.com/v2/articles/29925314: files CSV.zip, ECG_median_data.zip 0.14 GB, ECG_row_data.zip 1.28 GB; licence CC0; published 2026-07-09]

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
| C-14b | THE corpora SHALL be stored exactly as distributed; no transformed waveform array is persisted, and every transform is applied at read time. | pending — `test_ingest.py` |
| C-15 | WHEN a record contains a NaN or Inf sample, THE loader SHALL exclude it and report the excluded count per corpus. | pending — `test_ingest.py` |
| C-16 | WHEN PTB-XL is round-tripped through 250 Hz and back to 500 Hz, THE resulting coverage SHALL move by less than one third of the coverage gap attributed to dataset shift. | pending — `test_resample_control.py` |
| C-17 | THE encoder comparison SHALL include a frozen randomly-initialised encoder, reported alongside the pre-trained arms. | pending — `test_report.py` |
| C-18 | THE harness SHALL report per-task AUROC and AUPRC with bootstrapped confidence intervals, and SHALL use paired comparisons for any claim that one arm beats another. | pending — `test_report.py` |
| C-19 | WHEN the supervised baseline is trained on PTB-XL folds 1–8 and scored on fold 10, ITS MI AUROC SHALL lie within 0.03 of the value the PTB-XL benchmark reports for that split (Strodthoff et al. 2020, IEEE JBHI) — the reference value is read from the paper or its repository before the run, never from memory. Read 2026-08-23 from the repository README (github.com/helme/ecg_ptbxl_benchmarking, "PTB-XL: Diagnostic superclasses"): the benchmark reports the **macro** AUROC over the five superclasses (NORM, MI, STTC, CD, HYP) on fold 10, folds 1–8 train and 9 validation — resnet1d_wang 0.930(05), xresnet1d101 0.928(05), inception1d 0.921(06). It does not report a per-class MI AUROC; that number has to be computed from the archived predictions the README links (`preds_x.npy` / `targs_x.npy`, datacloud.hhi.fraunhofer.de/s/gLkjQL94d7FXBbS) before the baseline run, or the criterion compares the macro figure. | pending — `test_baseline.py` |
| C-20 | THE report SHALL contain exactly the figures listed under "Figures, fixed before any result", each regenerated by a script from a `results/` file committed before the figure, and THE external corpora SHALL each be scored once with the frozen PTB-XL calibration, never re-calibrated on themselves. | pending — `test_report.py` |

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

**Storage stays raw.** Each corpus sits on disk exactly as distributed and every
transform happens at read time, so the transform remains a parameter that can be
varied and tested rather than a fact baked into a file. EchoNext is the
cautionary example, and it is sitting in our own data directory: its published
arrays have median-filtering, percentile-clipping and dataset-wide normalisation
already applied, so millivolts cannot be recovered and the test-set leakage in
its normalisation cannot be undone. Persisting a preprocessed array is how a
corpus becomes unusable to everyone downstream.

The caveat that has to be said out loud: **there is no truly raw ECG here.**
Every recording has already passed through its device's anti-aliasing, mains
notch and baseline correction — PTB-XL through Schiller hardware of 1989-96,
Chongqing through a single Mecg-300, EchoNext through GE MUSE. "Raw" can only
mean "as the corpus ships it", and that residual device filtering is itself part
of the shift being measured rather than something the pipeline removes.

On filtering: the cited protocol prescribes none, and neither do we beyond what a
corpus already carried at source. Every filter added is a chance to help one
cohort more than another, so the honest default is the lightest chain all
corpora can share.


## Figures, fixed before any result

Named now so that day 4 onwards produces them instead of inventing them, and so
that a surprising or negative coverage result stays in the report rather than
being re-cut. Each is drawn by `scripts/figures.py` from a results file.

1. **Coverage against target.** Empirical coverage of the 90 % set on PTB-XL
   (in distribution), Shandong and Chongqing, per class (MI / non-MI), with the
   spread over ≥100 calibration draws. One panel per correction: none, Mondrian,
   label-shift weighted.
2. **Set-size distribution.** Histogram of set sizes per corpus; the share of
   empty, singleton and full sets. Smaller is not better — the spread is what
   shows the model separating easy from hard tracings.
3. **Encoder arms on the same break.** Coverage gap (source minus target) per
   arm, with bootstrapped intervals and paired differences: random-init frozen,
   ECGFounder, ECG-FM, HuBERT-ECG if its contamination claim verifies.
4. **Baseline discrimination.** AUROC and AUPRC per corpus with intervals; the
   reproduction of known ground that licenses everything else.

The numbers behind every figure live in `results/` as JSON or CSV and are
committed before the figure. Each external corpus is scored once with the
PTB-XL calibration; nothing is tuned on Shandong or Chongqing.

## Days

1. **Mon 24 — the ingestion contract, then the timing probe.** In this order:
   start the Chongqing waveform download in the background; build the
   canonical loader (C-13 to C-15) and patient-level splits (C-4); then put one
   hundred tracings through every encoder arm, stopwatch out, and write the
   seconds per hundred records to `results/timing.json`. That number sizes
   every experiment this week. **No training run starts before it exists.**
   Evidence the day is done: the loader tests pass on all three corpora, the
   split test passes, and `timing.json` has one row per arm (or a named reason
   an arm could not be timed).
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

**What the day-1 probe decided (23 Aug, `results/timing.json`).** Seconds per
hundred PTB-XL records on the laptop under a load average near 400, so
pessimistic: random-init ResNet1d 12.65 · ECGFounder 38.95 · HuBERT-ECG 101.46 ·
ECG-FM 173.81. Every arm loaded under the pinned torch 2.2.2. A full pass over
the ~67,000 records therefore costs, at that speed, about 2 h, 7 h, 19 h and
33 h respectively. The consequence: each encoder's representations (one vector
per tracing) are computed **once per arm on `esprimo`, as overnight jobs in
cost order, and cached**; every conformal experiment then runs on the cache in
seconds. Caching a representation does not breach C-14b, which is about
waveforms: a representation is a computed result, not a rewritten tracing.
The random arm starts Monday morning to measure the box's real speed.

**Sacrifice order if it overruns:** ACS-ECG as a second target, then encoder
arms from the least informative upward. The day-6 page is never cut.

**Encoder count is a compute question, not a plan question.** Adding a frozen
arm costs almost no developer time — same code path, different weights — and
costs one more forward pass over ~65,000 records on a CPU with no GPU. Monday's
timing probe produces that number, and it decides how many arms fit. Do not
trade an arm away before the probe has run.

The grid the arms are meant to fill:

| Arm | Saw PTB-XL (source) | Saw Shandong (target) |
|---|---|---|
| Random init, frozen | no | no |
| ECGFounder | no | no |
| ECG-FM | yes | no |
| HuBERT-ECG | yes | yes |

The bottom row is where pre-training contamination should look most flattering,
so it is the most informative arm, not the most expendable. Two conditions on
it: its licence is CC BY-NC 4.0, which permits a public research demonstration
but excludes anything the venture ships (D-059 excluded it on venture grounds;
a demonstration is not a product, and that distinction is Ruben's call). And the
claim that its pre-training included SPH is second-hand — the audit could not
fetch medRxiv directly and relied on a search-retrieved quote. **Verify that at
source before Thursday, because the bottom row rests entirely on it.**

## Hardware

Laptop is a 2016 Intel Mac — 8 threads, 16 GB, ~28 GB free, and torch dropped
macOS x86_64 wheels after 2.2.2, which is why that version is pinned. The Linux
box `esprimo` (i7-8700, 12 threads, 15 GB, 389 GB free, no GPU, Python 3.12,
reachable with `ssh esprimo`) carries the heavy runs: install `uv` there, clone
this repository, rsync `data/` and the PTB-XL directory, and run with the same
pinned environment so both machines produce the same numbers. Neither has CUDA,
so encoder inference cost is the schedule's real risk and day 1 measures it
first.

## Rollback

Standalone repo; nothing outside it is modified. PTB-XL is read in place and
never written. `data/` and `results/*.json` are gitignored. Any unit reverts with
`git reset --hard` without touching the corpora.

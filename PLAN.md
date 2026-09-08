# Execution plan

Six working days: Mon 24 → Fri 28 Aug, plus Mon 31. Written the night of 22–23
August, when the scaffold and the corpora landed.

## Why this object exists

A model that knows when it is wrong, measured across hospitals, with the code
open. The point is that a cardiologist or a hospital data-science lead can
inspect the method rather than take it on trust, which is a thing a description
of the work cannot do and a repository can.

## Corpora

| Corpus | Labels | Waveforms on disk | Records kept | Where |
|---|---|---|---|---|
| PTB-XL v1.0.3 | `ptbxl_database.csv` | 500 Hz WFDB, 2.6 GB | 21,799 / 21,799 | `~/Developer/ptbxl5d/data/records500` (read in place) |
| SPH / Shandong | `metadata.csv` | 25,770 HDF5 files, 4.3 GB | 25,770 / 25,770 | `data/sph/records` |
| ACS-ECG / Chongqing | `data/acs/CSV` | 19,955 WFDB records, 1.3 GB | 19,950 / 19,955 | `data/acs/row_data` |

Sources: PTB-XL and SPH from PhysioNet and Springer Nature figshare respectively (licences CC BY 4.0 and CC0); ACS-ECG from figshare 10.6084/m9.figshare.29925314 (CC0, `ECG_row_data.zip`, 1.28 GB, no account) [verified 2026-08-23 — api.figshare.com/v2/articles/29925314]. `scripts/fetch_open_corpora.sh` fetches ACS-ECG and the four auxiliary PhysioNet corpora, verifying each against the checksums their publishers give; PTB-XL and SPH are not fetched by it and were downloaded by hand. The five Chongqing records excluded carry a non-finite sample (3) or are shorter than their header says (2); `results/ingest_report.json` lists them by id.

## Acceptance criteria

Each maps to a named test. Unchecked means unverified, not done.

| # | Criterion | Test |
|---|---|---|
| C-1 | THE PTB-XL loader SHALL return 21,799 records, 18,869 patients, 5,469 MI positives — and 5,288 with subendocardial-injury statements excluded. | `test_labels.py::TestPTBXL` |
| C-2 | THE SPH loader SHALL return 25,770 records, 24,666 patients, 260 MI positives, 233 of them chronic. | `test_labels.py::TestSPH` |
| C-3 | THE ACS loader SHALL return 17,960 labelled records, 17,018 patients, 1,151 OMI / 1,442 STEMI / 2,679 AMI. | `test_labels.py::TestACS` |
| C-4 | WHERE a calibration/test boundary is drawn, THE splitter SHALL place every record of a given patient on one side only. | met — `test_splits.py::TestPatientSplit` |
| C-5 | WHEN the calibrator is fitted at alpha on an exchangeable sample, THE empirical coverage SHALL fall within [1-alpha-0.012, 1-alpha+0.025]. | `test_conformal.py::TestCoverageGuarantee`, `test_report.py::TestTheGuaranteeOnDataWhoseAnswerIsKnown`, `test_report.py::TestTheCommittedAbstentionTable` |
| C-6 | WHEN a calibration sample is smaller than ceil(1/alpha)-1, THE quantile SHALL be +inf rather than a finite threshold. | `test_conformal.py::TestQuantile` |
| C-7 | WHEN the class prior changes between calibration and test, THE Mondrian thresholds SHALL hold class-conditional coverage at 1-alpha. | `test_label_shift.py::TestMondrianUnderPrevalenceShift` |
| C-8 | WHEN weights are uniform, THE weighted quantile SHALL equal the unweighted quantile exactly. | `test_conformal.py::TestWeightedQuantile` |
| C-9 | IF a weighted correction is reported, THEN the harness SHALL report its effective sample size alongside. | `test_label_shift.py::TestWhatTheWeightingCosts` on the statistic, where `test_the_effective_sample_size_is_the_one_worked_out_by_hand` pins it to a reference value computed by hand (765.0 of 1,000 for a 25% source reweighted to a 1% target). On the files: `test_report.py::TestWhatTheBreakTableCarries::test_every_cell_reports_the_effective_size_of_what_calibrated_it` and `TestTheCommittedBreakTable::test_every_weighted_cell_reports_what_the_weighting_cost_it` — every (level, score, correction, corpus) cell carries the effective size of the calibration sample behind its threshold. Under `none` and `mondrian` nothing is reweighted and the effective size is the count; under `weighted` the two part company, and on the committed table the effective size falls to 864 (Shandong) and 967 (Chongqing) of 1,099. The per-class calibration counts the Mondrian split leaves are on the same row — `test_the_split_reports_what_the_minority_class_is_calibrated_on`, 275 infarctions of 1,099 |
| C-10 | THE harness SHALL report every coverage figure as a mean over >= 100 calibration/test draws, with its standard deviation. | `test_report.py::TestTheCommittedAbstentionTable`, `test_report.py::TestTheCommittedBreakTable::test_every_figure_is_a_mean_over_at_least_a_hundred_draws_with_its_spread` |
| C-11 | THE harness SHALL report coverage separately for MI and non-MI cases. | `test_report.py::TestTheCommittedAbstentionTable::test_coverage_is_reported_for_infarction_and_for_not`, `test_report.py::TestTheCommittedBreakTable::test_coverage_is_reported_for_infarction_and_for_not_on_every_corpus` |
| C-12 | THE results file SHALL record the encoder's pre-training corpora. | `test_report.py::TestTheEncoderArmsOnRecord` on every cached representation; `test_report.py::TestTheCommittedArmGrid::test_every_arm_names_the_corpora_it_was_pre_trained_on` on the grid the arms are compared in, where `test_the_arms_that_saw_the_calibration_corpus_say_so` pins which two saw PTB-XL and which one also saw Shandong — the sentence for HuBERT-ECG carries its address (medRxiv 10.1101/2024.11.14.24317328v3, Methods) |
| C-13 | THE ingestion layer SHALL return, for every corpus, a float32 array of shape (N, 12, 5000): 10 seconds, 500 Hz, millivolts, leads ordered I, II, III, aVR, aVL, aVF, V1-V6. | met — `test_ingest.py::test_shape_dtype_and_identity_on_a_canonical_record` |
| C-14 | THE ingestion layer SHALL apply an identical filter and scaling chain to every corpus, and SHALL emit, per corpus, the list of steps that could not be made identical. | `test_ingest.py::TestAssembly::test_deviations_name_what_the_chain_did_differently`; on the results file, `test_report.py::TestTheCommittedBreakTable::test_each_corpus_names_what_could_not_be_made_identical` |
| C-14b | THE corpora SHALL be stored exactly as distributed; no transformed waveform array is persisted, and every transform is applied at read time. | pending — `test_ingest.py` |
| C-15 | WHEN a record contains a NaN or Inf sample, THE loader SHALL exclude it and report the excluded count per corpus. | met — `test_ingest.py::test_nan_and_inf_records_are_excluded_and_counted` |
| C-16 | WHEN PTB-XL is round-tripped through 250 Hz and back to 500 Hz, THE resulting coverage SHALL move by less than one third of the coverage gap attributed to dataset shift. | pending — `test_resample_control.py` |
| C-17 | THE encoder comparison SHALL include a frozen randomly-initialised encoder, reported alongside the pre-trained arms. | `test_report.py::TestTheEncoderArmsOnRecord::test_the_frozen_random_arm_is_among_them` among the representations; `test_report.py::TestTheCommittedArmGrid::test_the_frozen_random_arm_is_reported_beside_the_others` in the grid, which carries it in every block rather than mentioning it in passing |
| C-18 | THE harness SHALL report per-task AUROC and AUPRC with bootstrapped confidence intervals, and SHALL use paired comparisons for any claim that one arm beats another. | intervals: `test_metrics.py::TestBootstrapInterval` and `test_baseline.py::TestTheReferenceValue` on the statistic, `test_report.py::TestTheCommittedArmGrid::test_every_arm_and_corpus_carries_auroc_and_auprc_with_an_interval` on every cell of the arm grid. Paired comparisons: `test_arms.py::TestPairedDifference` on the statistic itself, where `test_pairing_is_tighter_than_two_separate_intervals` is the reason the comparison is paired — it holds a pair of arms whose own intervals overlap while the paired difference excludes zero — and `test_report.py::TestTheCommittedArmGrid::test_every_pair_of_arms_is_compared_paired_on_every_corpus` on the committed grid, with `test_a_paired_difference_is_only_called_separated_when_it_excludes_zero` holding the word "separated" to the interval. Arm-versus-arm differences use a paired bootstrap over records rather than DeLong, which tests AUROC only and has no AUPRC counterpart |
| C-19 | WHEN the supervised baseline is trained on PTB-XL folds 1–8 and scored on fold 10, ITS MI AUROC SHALL lie within 0.03 of the reference value recorded in `results/baseline.json`. That reference is **0.930**, the macro AUROC over the five diagnostic superclasses (NORM, MI, STTC, CD, HYP) that the benchmark repository (github.com/helme/ecg_ptbxl_benchmarking, table “PTB-XL: Diagnostic superclasses”) reports for `resnet1d_wang` on fold 10, folds 1–8 train and 9 validation — read from the repository on 2026-08-24, never from memory. It is a macro figure and not the MI column alone: the archived predictions the README links (datacloud.hhi.fraunhofer.de/s/gLkjQL94d7FXBbS) now serve a different study's `output.zip`, which holds no `preds_x.npy` or `targs_x.npy`; the repository commits only the 71-statement `exp0` outputs; and the paper text carries no per-class MI table. `results/baseline.json` records all three routes. Because MI is averaged in with four other superclasses, a gap wider than 0.03 is investigated before it is read as a broken baseline. | `test_baseline.py::TestTheReferenceValue` |
| C-20 | THE report SHALL contain exactly the figures listed under "Figures", each redrawn pixel for pixel by a script from a committed `results/` file, and THE external corpora SHALL each be scored once with the frozen PTB-XL calibration, never re-calibrated on themselves. | all six figures, redrawn and compared pixel by pixel: `test_report.py::TestTheFigures::test_each_figure_redraws_pixel_for_pixel`; the report's own figure list: `test_outcomes.py::TestTheFiguresDrawnFromIt::test_the_report_shows_exactly_the_figures_it_names`; the one-shot half: `test_report.py::TestTheFrozenCalibrationOnDataWhoseAnswerIsKnown::test_no_target_label_ever_reaches_a_threshold` and `TestTheCommittedBreakTable::test_every_threshold_was_fitted_on_ptbxl_and_nowhere_else`. Figure 3 draws from `results/arms.json`, committed before it, and `TestTheFigures::test_figure_three_carries_every_arm_and_every_target` holds it to every arm; `test_figure_three_says_what_it_is_waiting_for_rather_than_drawing_empty` holds the behaviour when that file is absent. The weighted correction is the one row family whose threshold differs by corpus: it reads that corpus's unlabelled predicted-label marginal to estimate its class mix, and nothing else of it — the label-inversion test above is what holds that, under all three corrections. `test_an_unweighted_threshold_does_not_depend_on_which_corpus_it_is_spent_on` holds the other two to one identical threshold in every corpus block. Figure 1 carries one panel per correction and one panel row per corpus: `TestTheFigures::test_figure_one_carries_a_panel_per_corpus_and_per_correction` |
| C-21 | THE outcome table SHALL split every case of a label into exactly three shares — the correct label alone, deferred, the wrong label alone — summing to one, so that a scheme with deferrals is comparable with one without. | `test_outcomes.py::TestTheSplitItself`, `TestTheCommittedOutcomeTable::test_every_label_and_scheme_partitions_into_the_three_outcomes` |
| C-22 | WHERE a single tuned threshold is compared against a conformal scheme, THE results file SHALL record that its target is a sensitivity and the conformal target a coverage, and SHALL record the decision boundaries on the probability axis rather than the raw score quantiles. | `test_outcomes.py::TestTheCommittedOutcomeTable::test_the_table_says_the_two_targets_are_different_quantities`, `test_the_denominator_is_named_in_the_file` |


| C-23 | THE label table SHALL map each corpus's own code system onto the five rotation diagnoses, and ITS per-corpus per-class counts SHALL equal the counts the Challenge publishes for its four partitions once the records carrying both halves of a fused pair are subtracted. | `test_label_map.py::TestTheCountsCloseAgainstThePublishedTable`, on `results/label_map.json`; the mapping itself against the three code tables in `mappings/` by `test_small_set.py` |
| C-24 | THE label table SHALL name every join it could not make cleanly, and WHERE a class cannot be read on a corpus THE table SHALL refuse that cell with the ambiguity that refused it rather than fill it. | `test_label_map.py::TestEveryCorpusAndClassHasACell::test_a_refused_cell_is_empty_and_says_which_ambiguity_refused_it`; `test_small_set.py::TestEveryAmbiguityIsWritten` |
| C-25 | WHERE a Challenge-2021 partition is ingested, THE loader SHALL return the canonical (N, 12, 5000) form and SHALL report that the bundle ships no patient identifier; WHEN a record carries fewer samples than the ten-second window, THE loader SHALL exclude it and count it. | `test_rotation.py::TestTheIngestionContract`, `TestTheSplit::test_the_corpora_that_ship_short_records_say_how_many_they_dropped`; `test_ingest.py::TestAssembly::test_a_record_too_short_for_the_window_is_excluded_and_counted` |
| C-26 | WHEN a source model's thresholds are fitted, THEY SHALL be a function of that source's calibration records alone; permuting every label of a target corpus SHALL leave every threshold unchanged, on each of the twenty ordered pairs. | `test_rotation.py::TestNoThresholdReadsATargetLabel` on synthetic scores, `TestNoTargetLabelReachesAThresholdOnTheRealPairs` parametrised over the twenty pairs, and `TestTheCommittedRotation::test_an_unweighted_threshold_does_not_depend_on_which_corpus_it_is_spent_on` on the committed table |
| C-27 | THE rotation file SHALL report, for every (source, target, diagnosis, correction, level), the coverage mean and spread over at least 200 calibration draws, with the effective size of the calibration sample behind the threshold. | `test_rotation.py::TestTheCommittedRotation::test_every_figure_is_a_mean_over_two_hundred_draws_with_its_spread`, `test_every_cell_reports_the_effective_size_of_what_calibrated_it` |
| C-28 | THE rotation file SHALL report the coverage bias per diagnosis with its spread across the five sources, not only across the twenty pairs. | `test_rotation.py::TestTheCommittedRotation::test_the_bias_carries_its_spread_across_sources` |
| C-29 | THE target-scale file SHALL report coverage on Chongqing after recalibration on 0, 100, 500 and 2,000 labelled target records, measured on one held-out half that no rung calibrates on. | `test_target_scale.py` |
| C-30 | THE encoder arms SHALL include ECG-JEPA, and THE README SHALL name which arms saw PTB-XL at pre-training and HuBERT-ECG's non-commercial licence. | `test_encoder_arms.py`; the README's "Encoder arms" table |

| C-31 | THE rotation SHALL report, beside the spread over calibration draws, a 95% interval obtained by resampling each target cohort by patient, so that a coverage read on a thin class carries the uncertainty of the class being thin. | `test_rotation_uncertainty.py::TestTheCommittedUncertainty`, on `results/rotation_uncertainty.csv`; the resampling unit itself by `TestTheBootstrapItself` |
| C-32 | THE rotation SHALL report, for every pair, the coverage a per-class rejection rule with one plain empirical quantile per class reaches on the same scores, and the difference from the conformal figure. | `test_rotation_uncertainty.py::TestChowIsMondrianWithoutTheCorrection` on the rule, `TestTheCommittedUncertainty::test_the_two_rules_agree_except_where_the_class_is_starved` on the committed file |

| C-33 | THE rotation SHALL repeat every coverage cell by sex and by age band wherever the corpus records them, with the number of positives behind each cell and its patient bootstrap interval, and SHALL flag a cell resting on fewer than 25 positives rather than let it read as a result. | `test_rotation_uncertainty.py::TestTheSubgroups`, on `results/rotation_uncertainty.csv` |

| C-34 | WHERE a corpus files one tracing under more than one record identifier, THE splitter SHALL place every record of that group on one side of every boundary, and no group SHALL sit in two parts a figure is read across. | `test_duplicates.py::TestTheSplitNoLongerLeaks`, on `results/split_leak.json`; the key itself by `TestTheWidenedKey` |
| C-35 | THE results SHALL record, per corpus, how many groups of identical tracings straddled a boundary before the splitter was widened and how many straddle it after. | `test_duplicates.py::TestTheSplitNoLongerLeaks::test_the_leak_it_closed_is_on_the_record` |

## The ingestion contract

Every corpus is reduced to one canonical form before anything else touches it:
float32, shape (N, 12, 5000) — ten seconds, 500 Hz, millivolts, leads in the
order I, II, III, aVR, aVL, aVF, V1-V6. Records carrying a NaN or Inf sample
are dropped and counted. This is the form prescribed in arXiv:2602.17531, which
the ptbxl5d report already cites, so it is a convention with a citation behind
it rather than a house style.

The binding rule is that the chain is identical for every corpus, which
matters more than the values. Tuning preprocessing per dataset would raise each one's signal quality
and destroy the only quantity this study measures, because a difference produced
by my own pipeline is indistinguishable from one produced by the hospital. Where
a corpus cannot be made to match, the deviation is named in the results file
rather than absorbed silently.

Four deviations are known in advance and none can be fully removed:

- Resampling. PTB-XL and Shandong are natively 500 Hz; EchoNext is 250 Hz,
  INCART 257 Hz, PTB 1000 Hz. Upsampling invents detail and downsampling discards
  it, so resampling is itself a source of apparent shift. Polyphase resampling
  with anti-aliasing (`scipy.signal.resample_poly`) is the defensible choice, and
  C-16 measures what it costs instead of assuming it is free.
- EchoNext cannot meet the contract. Its waveforms ship already median-
  filtered, percentile-clipped and normalised with a dataset-wide mean and
  standard deviation, so they are not in millivolts and cannot be returned to
  that scale. Worse for this purpose, that normalisation was computed across all
  splits including test. EchoNext is usable as a benchmark target, not as a
  cohort in a like-for-like shift comparison.
- Lead order is not documented consistently. The ACS-ECG paper states it two
  different ways in consecutive paragraphs, with aVR and aVL swapped. Order is
  read from each record's own header, never assumed.
- Amplitude scaling differs at source (PTB-XL in mV, Chapman at 4.88 uV per
  least-significant bit). Conversion happens once, in the loader, and the factor
  for each corpus is asserted by a test against a known record.

Storage stays raw. Each corpus is stored exactly as distributed and every
transform happens at read time, so the transform remains a parameter that can be
varied and tested rather than a fact baked into a file. EchoNext is the
cautionary example, and it is sitting in our own data directory: its published
arrays have median-filtering, percentile-clipping and dataset-wide normalisation
already applied, so millivolts cannot be recovered and the test-set leakage in
its normalisation cannot be undone. Persisting a preprocessed array is how a
corpus becomes unusable to everyone downstream.

One caveat: there is no truly raw ECG here.
Every recording has already passed through its device's anti-aliasing, mains
notch and baseline correction — PTB-XL through Schiller hardware of 1989-96,
Chongqing through a single Mecg-300, EchoNext through GE MUSE. "Raw" can only
mean "as the corpus ships it", and that residual device filtering is itself part
of the shift being measured rather than something the pipeline removes.

On filtering: the cited protocol prescribes none, and neither do we beyond what a
corpus already carried at source. Every filter added is a chance to help one
cohort more than another, so the default is the lightest chain all corpora can
share.


## Figures

Named now so that day 4 onwards produces them instead of inventing them, and so
that a surprising or negative coverage result stays in the report rather than
being re-cut. Each is drawn by `scripts/figures.py` from a results file.

1. Coverage against target. Empirical coverage of the 90 % set on PTB-XL
   (in distribution), Shandong and Chongqing, per class (MI / non-MI), with the
   spread over ≥100 calibration draws. One panel per correction: none, Mondrian,
   label-shift weighted.
2. Set-size distribution. Histogram of set sizes per corpus; the share of
   empty, singleton and full sets. Smaller is not better — the spread is what
   shows the model separating easy from hard tracings.
3. Encoder arms on the same break. Coverage gap (source minus target) per
   arm, with bootstrapped intervals and paired differences: random-init frozen,
   ECGFounder, ECG-FM, HuBERT-ECG if its contamination claim verifies.
4. Baseline discrimination. AUROC and AUPRC per corpus with intervals, set
   beside the published macro figure as the plausibility check C-19 defines --
   a different quantity, so not a reproduction.
5. Threshold placement. Where each of the three schemes puts its decision
   boundaries on the model's probability axis, over the source score
   distribution of each class, with the resulting miss rate, false-alarm rate
   and deferral share beside each panel.
6. Per-label outcomes. What a case of each label receives under each scheme:
   the correct label alone, a deferral, or the wrong label alone.

7. The source rotation. Coverage of each of the five diagnoses when the
   calibration source is each of five corpora in turn and its thresholds are
   spent on the other four: one panel per correction, one point per ordered
   pair coloured by its source, the source's own held-out reading beside it,
   and the mean over the away pairs with its spread across sources.
8. The target scale. Coverage on Chongqing against how many labelled Chongqing
   tracings the threshold saw — 0, 100, 500, 2,000 — recalibrated on those
   records against pooled with the source calibration half.

Figures 5 and 6 are the report's figures 1 and 2; the coverage grid is its
figure 3. Figures 2 and 4 above are kept in `results/figures/` and cited from
`README.md` and `QUESTIONS.md` rather than carried in the report. Figures 7 and
8 belong to the rotation and are not in the report yet.

The numbers behind every figure live in `results/` as JSON or CSV and are
committed before the figure. Each external corpus is scored once with the
PTB-XL calibration; nothing is tuned on Shandong or Chongqing.

## The source rotation

The break table measures one calibration source against two targets. That is a
pair, and a pair cannot say whether the break is a property of conformal
prediction under a change of hospital or a property of PTB-XL and Chongqing.
The rotation makes it an estimate: five corpora take turns as the source, each
one's thresholds are spent once on each of the other four, and the coverage bias
is reported per diagnosis with its spread across sources. This is Leinonen et
al.'s rotation protocol (Comput Biol Med 2024, PMID 39427424) asked of coverage
rather than of discrimination.

Five corpora — PTB-XL, Shandong, Chapman-Shaoxing with Ningbo, Georgia, CPSC
2018 with its extension — and five diagnoses a cardiologist reads at a glance:
sinus rhythm, atrial fibrillation, left and right bundle-branch block,
first-degree atrioventricular block. Infarction is not among them because it is
not a scored Challenge class and exists with usable counts on PTB-XL, Shandong
and Chongqing alone; the infarction axis stays as it is, and the target-scale
ladder is measured on it.

Every corpus is cut once by patient into train, validation, calibration and
test. The test part is the same records whether the corpus is the source or a
target, so a difference between home and away is the threshold rather than the
sample. Train and calibration are capped at the size the smallest corpus
reaches, so "which source" is not read together with "how much data the source
had".

`results/label_map.json` is the piece this rests on: the mapping from three
annotation schemes onto five classes, its counts checked against the
Challenge's own published table, and the five joins that could not be made
cleanly written down rather than decided in passing.

### What the coverage table cannot say on its own

Two objections apply to the rotation as much as to the infarction axis, and
both are answered by `results/rotation_uncertainty.csv` rather than left to the
reader.

*The spread in the table is the wrong uncertainty.* Every figure in
`results/rotation.csv` is a mean over 200 calibration draws on a target cohort
that never moves, so it carries the variability of the threshold and none of
the variability of the population. A coverage read on Shandong's 23 left
bundle-branch blocks is uncertain because there are 23 of them. The uncertainty
file resamples each target cohort by patient and reports a percentile interval
beside the draw spread.

*A split by patient does not hold a repeated tracing.* Three of the five corpora
file one tracing under several record identifiers, and the Challenge bundle
names no patient, so each record was its own patient and the copies went
wherever the shuffle sent them: 485 groups straddled a boundary a figure is read
across, 421 of them in CPSC, and some of those boundaries were train against
test. The screen is the delivery corpus's own (`ecg-data-chain` 4bff859): the
first ten seconds of the twelve leads quantised to ten microvolts, SHA-256. On
the 49,199 records both repositories digest, the two agree on every one. A group
is now one splitting unit and the count after is zero.

*A figure that holds over a cohort can fail over half of it.* Every row is
repeated by sex and by age band. Coverage of sinus rhythm away from home falls
from 0.841 under 50 to 0.619 at 75 and over, against the 0.90 promised; right
bundle-branch block runs the other way, 0.714 to 0.906. The widest gap between
the sexes on a single pair is left bundle-branch block from PTB-XL to
Chapman-Shaoxing with Ningbo, 0.505 for men against 0.821 for women. A cell
resting on fewer than 25 positives is flagged, because a 95% interval on twenty
cases is wider than any difference the rotation looks for.

*A rejection rule may do the same work.* One plain empirical quantile per class
— Chow, *IEEE Trans Inf Theory* 1970 — is Mondrian minus the finite-sample
`(n+1)` correction. If the two land in the same place, the conformal formalism
is a rename of a per-class rejection rule, and the object has to say so. The
uncertainty file reports both coverages and their difference, per pair, so what
conformal prediction adds here is a number rather than a claim.

### The nearest prior work, and what is not known about it

El Allam and Hamlich, "Quantization-aware Mondrian conformal prediction for
embedded ECG classification", *Biomed Signal Process Control*, November 2026,
10.1016/j.bspc.2026.111217. Title, authors, journal and date were read from
Crossref on 2026-09-08; the DOI resolves.

**The paper itself has not been read here.** It is closed access, and neither
Crossref nor Semantic Scholar carries its abstract. So what corpora it uses,
what diagnosis, and whether it reports coverage on a target site are unknown to
this repository. That matters directly: it applies Mondrian conformal prediction
to ECG, which is the correction this rotation reports, so if it already carries
external per-label coverage then the novelty claim falls and this section is
rewritten. Obtaining the full text, by library or by writing to the authors, is
a dependency on the reading and not on the code.

## Days

1. Mon 24 — the ingestion contract, then the timing probe. In this order:
   start the Chongqing waveform download in the background; build the
   canonical loader (C-13 to C-15) and patient-level splits (C-4); then put one
   hundred tracings through every encoder arm, stopwatch out, and write the
   seconds per hundred records to `results/timing.json`. That number sizes
   every experiment this week. No training run starts before it exists.
   Evidence the day is done: the loader tests pass on all three corpora, the
   split test passes, and `timing.json` has one row per arm (or a named reason
   an arm could not be timed).
2. Tue 25 — abstention table on PTB-XL. Conformal wired to the model, C-5
   verified in distribution, silence rate against set size. Reproduces known
   ground.
3. Wed 26 — the break. Calibrate on PTB-XL, measure on SPH then ACS-ECG.
   Publishable either way: a clear break is a result and a negligible one is a
   valid null result.
4. Thu 27 — three encoders on the same break. ECG-FM (saw PTB-XL),
   ECGFounder (saw no public corpus), supervised-from-scratch. The gap between
   the first two is the contamination measurement.
5. Fri 28 — exact correction vs estimated. Mondrian against label-shift
   weighting, effective sample size on both (C-9), and a reasoned answer to
   which one this shift needed.
6. Mon 31 — the page a cardiologist can read. Plus the ten questions the
   object will trigger, answered in writing.

Compute budget (`results/timing.json`): seconds per hundred PTB-XL records
on the laptop, measured while other jobs loaded it, so upper bounds: random-init
ResNet1d 12.65 · ECGFounder 38.95 · HuBERT-ECG 101.46 · ECG-FM 173.81. Every
arm loads under the pinned torch 2.2.2 (ECG-FM in its own environment,
`scripts/setup_ecgfm_env.sh`). A full pass over the ~67,000 records costs, at
that speed, about 2 h, 7 h, 19 h and 33 h respectively. Each encoder's
representations (one vector per tracing) are therefore computed once per arm
on `esprimo`, as overnight jobs in cost order, and cached; every conformal
experiment runs on the cache in seconds. Caching a representation is within
C-14b, which is about waveforms: a representation is a computed result, not a
rewritten tracing.

Sacrifice order if it overruns: ACS-ECG as a second target, then encoder
arms from the least informative upward. The day-6 page is never cut.

Encoder count is decided by compute. Adding a frozen arm costs almost no
developer time — same code path, different weights — and costs one more forward
pass over ~65,000 records on a CPU with no GPU. Monday's timing probe produces
that number, which sets how many arms fit. Do not trade an arm away before the
probe has run.

The grid the arms are meant to fill:

| Arm | Saw PTB-XL (source) | Saw Shandong (target) |
|---|---|---|
| Random init, frozen | no | no |
| ECGFounder | no | no |
| ECG-FM | yes | no |
| HuBERT-ECG | yes | yes |

The bottom row is where pre-training contamination should look most flattering,
so it is the most informative arm and is dropped last. Two conditions on
it: its licence is CC BY-NC 4.0, which permits a public research demonstration
but excludes commercial use, and a demonstration is not a product. And the
claim that its pre-training included SPH is second-hand — the audit could not
fetch medRxiv directly and relied on a search-retrieved quote. Verify that at
source before Thursday, because the bottom row rests entirely on it.

## Hardware

Laptop is a 2016 Intel Mac — 8 threads, 16 GB, ~28 GB free, and torch dropped
macOS x86_64 wheels after 2.2.2, which is why that version is pinned. The Linux
box (i7-8700, 12 threads, 15 GB, no GPU, Python 3.12) carries the heavy runs:
install `uv` there, clone this repository, copy `data/` and the PTB-XL
directory, and run with the same pinned environment so both machines produce
the same numbers. Neither has CUDA,
so encoder inference cost is the schedule's real risk and day 1 measures it
first.

## Rollback

Standalone repo; nothing outside it is modified. PTB-XL is read in place and
never written. `data/` and `results/*.json` are gitignored. Any unit reverts with
`git reset --hard` without touching the corpora.

# Predicting an ECG model's positive predictive value from a clinic's prevalence: a retrospective measurement across hospitals and care settings

The article: [REPORT.md](REPORT.md) · Methods in statistical terms, the infarction study and the five-corpus rotation: [SUPPLEMENT.md](SUPPLEMENT.md) · Ruben Abbou · 2026

A clinic that adopts an ECG model receives a sensitivity and a specificity measured on another hospital's patients, and the usual advice is to get the positive predictive value (PPV), the share of flagged patients who are truly ill, by applying Bayes' theorem at the clinic's own prevalence. Across 174 transfers of a threshold between hospitals, care settings and corpora, the recomputed PPV missed the observed one by a median of 5.3 (4.2 to 7.8) percentage points, against 0.3 (0.2 to 1.0) on controls, and fell outside the observed PPV's 95% confidence interval in 75% (67% to 82%) of transfers, too high in some and too low in others. At Columbia, 25.6% (23.1% to 28.3%) of outpatients were ill and 46.0% (41.3% to 50.7%) of those the trained network flagged were ill; the recomputation gave 34.2%. The network separated ill from healthy outpatients almost as well as inpatients (AUROC 0.815 and 0.805), but ill and healthy outpatients both scored lower, so the inpatient threshold caught 71.6% (65.9% to 76.6%) of ill outpatients and flagged 29 of 100 healthy ones, against 90.4% and 58 among inpatients. The share of outpatients flagged, 39.8% (36.9% to 42.8%), was lower than any prevalence could produce with the inpatients' sensitivity and specificity. Set again on 858 separate outpatients, the threshold caught 86.3% (81.7% to 89.9%) of ill outpatients and flagged 52 of 100 healthy. At Chongqing the recomputation erred the other way: 45% predicted, 26% (25% to 27%) observed.

A model's PPV at a new clinic is not reliably predicted from the clinic's prevalence: the model's ordering of patients carries over, but the point where a fixed threshold lands does not. A clinic learns its PPV by counting it on its own patients with known diagnoses. The share of patients it flags, which needs no diagnosis, shows early that the threshold has moved.

![PPV recomputed against PPV observed, and the gap against the change in specificity](results/figures/ppv_gap.png)

Each point is one transfer: one model, one diagnosis, one pair of populations. Grey points are controls, where the target is drawn from the source population. Right: the gap in percentage points against how far specificity moved from source to target.

## What else the article measures

Case mix, the outpatients' fewer findings and higher ejection fraction among the ill, explains about a quarter of the fall in sensitivity at Columbia. Two other models gave the same answer there: the published EchoNext mini-model caught 72.7% of ill outpatients and the ECGFounder foundation model 71.6%. Of three repairs judged on net benefit over 146 transfers, logistic recalibration on 100 local labels was the one that gained on average; for Columbia's outpatients it lowered net benefit at decision thresholds of 5% and 10% for all three models. Setting the threshold again on 100 outpatients with known diagnoses, about 26 of them ill, caught 91.6% of the other outpatients' ill on average and flagged 70 of 100 healthy, about 7 more echocardiograms for each extra ill outpatient found. Conformal prediction, the method the study set out with, gave a threshold equal to the plain 90th percentile, and its guarantee did not hold for outpatients, who are not exchangeable with inpatients.

This is a retrospective measurement study on public, de-identified data. It is not a medical device and has no regulatory status, it involved no contact with patients, and nothing here is meant to guide the care of any patient. The ethics approvals and the author's competing interests are in [REPORT.md](REPORT.md).

## Design

EchoNext holds 100,000 ECGs from Columbia, each paired with an echocardiogram. The label studied is moderate or worse structural heart disease on echocardiography, a composite of eleven findings that EchoNext records for each ECG. Thresholds are set on the 1,903 inpatient ECGs of EchoNext's validation split and applied unchanged to the 1,059 outpatient ECGs of its test split, one ECG per patient and no patient in both. Structural heart disease is present in 53.2% of the calibration inpatients and in 25.6% of the test outpatients.

Four models score every ECG: a residual network trained on EchoNext for this study; the published EchoNext mini-model, run on its authors' weights; ECGFounder, a foundation model pre-trained at another hospital and frozen under one logistic regression per label; and the study's residual network frozen at random initialisation under the same regressions, the floor a pre-trained model has to clear.

| Model | AUROC, outpatients | Ill outpatients caught | Healthy outpatients flagged, per 100 | Ill caught, threshold set again on 100 outpatients | Repetitions below 90% | Healthy flagged per 100, threshold set again | Ill caught, threshold set on 858 other outpatients |
|---|---|---|---|---|---|---|---|
| Residual network, trained on EchoNext | 0.805 | 71.6% | 29 | 91.6% | 33.1% | 70 | 86.3% |
| EchoNext mini-model, published weights | 0.795 | 72.7% | 29 | 91.9% | 31.1% | 72 | 87.8% |
| ECGFounder, frozen, logistic regressions | 0.791 | 71.6% | 29 | 91.8% | 29.9% | 70 | 89.7% |
| Random initialisation, frozen, logistic regressions | 0.758 | 78.6% | 43 | | | | 91.9% |

An infarction network trained on PTB-XL, a German research corpus, had its threshold set to catch 90% of infarctions there and caught 83.7% at a hospital in Chongqing and 97.3% at one in Shandong. Between hospitals the model's own separation moved with the threshold: its AUROC went from 0.932 at PTB-XL to 0.793 at Chongqing, where an infarction is an acute event in the discharge diagnosis rather than a pattern on the tracing, and the population, equipment and era differ as well; the design does not say which of these lowered it. A rotation of five corpora through the source role covers five diagnoses: sinus rhythm, atrial fibrillation, left and right bundle-branch block, and first-degree atrioventricular block. [SUPPLEMENT.md](SUPPLEMENT.md) gives both in full.

EchoNext is under PhysioNet's restricted licence, so no tracing and no per-record score is in this repository. Each EchoNext model's one-page transfer report is in [reports/transfer/](reports/transfer/), with the PPV gap for every finding; its threshold ladder reads one fixed half of the outpatients, and the article's section 3.7 supersedes it.

## Reproduce

Every number and figure the article prints is redrawn from the files committed here, so no raw tracing is needed for the infarction study. `outcomes.py` and `subgroups.py` need PTB-XL's `ptbxl_database.csv` (6.6 MB from PhysioNet), which holds each record's patient, sex and age; point `ECS_PTBXL_DIR` at the directory holding it. The EchoNext results need the EchoNext distribution and the stored per-record scores; [SUPPLEMENT.md](SUPPLEMENT.md) gives those commands.

```bash
uv sync                                  # 1 min
uv run pytest -m "not data"              # unit tests, no corpora needed, 2 min
uv run python tests/paper_render.py      # checks REPORT.md, SUPPLEMENT.md, README.md and CITATION.cff against results/
uv run python scripts/figures.py         # redraws the infarction and rotation figures, 5 s
uv run python scripts/paper_figures.py   # redraws Figures 2, 3 and 5 from results/echonext_clinical.json
uv run python scripts/infarction_sites.py  # rebuilds results/infarction_sites.json, 2 s
export ECS_PTBXL_DIR=/path/to/ptbxl      # the directory with ptbxl_database.csv
uv run python scripts/outcomes.py        # rebuilds results/outcomes.json, 3 s
uv run python scripts/subgroups.py       # rebuilds results/subgroups.json, 23 s
uv run python scripts/ppv_gap.py         # rebuilds results/ppv_gap.json, 2 s
uv run python scripts/ppv_intervals.py   # rebuilds results/ppv_intervals.json, the intervals across transfers
uv run python scripts/ppv_figures.py     # redraws Figures 1 and 4, from the committed results
```

Timings are wall clock on a six-core i7-8700, CPU only; the full suite on a cold clone took 28 minutes. Re-scoring from the raw tracings, the corpus downloads and the rotation chain are in [docs/data.md](docs/data.md).

## Code

| Module | Role |
|---|---|
| `src/ecs/clinical.py` | outcomes per 100 ill and healthy patients, predictive values, paired differences, the re-drawn refitting ladder, the case-mix standardisation and Holm's adjustment |
| `src/ecs/conformal.py` | split conformal (LAC and APS scores), per-class quantiles, covariate- and label-shift weighting, BBSE |
| `src/ecs/metrics.py` | coverage, Wilson intervals, class-conditional coverage, set size, abstention, effective sample size |
| `src/ecs/labels.py` | one comparable infarction label across three annotation schemes (SCP-ECG, AHA, discharge diagnosis) |
| `src/ecs/config.py` | corpus paths and the label vocabulary |
| `src/ecs/small_set.py` | the five rotation diagnoses, the codes each corpus names them by, and the joins that stayed ambiguous |
| `src/ecs/challenge.py` | the Challenge-2021 partitions: headers, SNOMED labels, completeness against the bundle manifest |
| `src/ecs/rotation.py` | the five corpora split the same way, capped to a common size |
| `src/ecs/encoders.py` | the five encoder arms and the chain each one demands |
| `src/ecs/echonext.py` | EchoNext's files, one provenance row per tracing, and its inpatient, emergency and outpatient cohorts |
| `src/ecs/severity.py` | the severity of the ill by care setting, and coverage reweighted to another setting's severity |
| `src/ecs/transfer.py` | thresholds fitted on a source cohort and spent unchanged on a target, and refitted on target labels |
| `src/ecs/calibration.py` | calibration curve, slope and intercept, Brier score, net benefit and positive predictive value of a probability at one site |
| `scripts/ppv_intervals.py` | the 95% intervals across transfers, resampling groups that share a model and a pair of populations, and the Wilson intervals of the Columbia shares |
| `src/ecs/ppv_gap.py` | the PPV recomputed by Bayes' rule against the PPV observed, with Wilson and bootstrap intervals and a likelihood-ratio check of label shift |
| `src/ecs/repairs.py` | label-free prevalence correction, recalibration on local labels and per-label abstention, judged on net benefit |
| `src/ecs/transfer_report.py` | the one-page transfer report, rendered from `results/echonext_transfer.json` alone |
| `src/ecs/embedding_store.py` | encoder vectors filed by encoder, version and tracing digest, outside the repository |
| `src/ecs/echonext_mini.py` | the published EchoNext mini-model, rebuilt to run its own checkpoint |
| `mappings/` | the three published code tables the label mapping joins on, with provenance and digests |

`pre-commit` runs `ruff`, `mypy --disallow-untyped-defs` and `pytest` on every commit, and the unit tests run again before a push.

## Licence and citation

The code is MIT ([LICENSE](LICENSE)), with three exceptions. `third_party/ecg_jepa/` is vendored from Sehun Kim's ECG-JEPA release under MIT, with its own LICENSE. `third_party/ecgfounder/` is not covered by the root licence; its MIT and Apache 2.0 notices are in [PROVENANCE.md](third_party/ecgfounder/PROVENANCE.md). The HuBERT-ECG weights are CC BY-NC 4.0, and so is every number computed from them: `results/embeddings/hubert_ecg/` and the HuBERT-ECG rows of `results/arms.json` are for research use only.

The committed score files hold one row per record (identifier, label, score), derived data under each corpus's licence; [docs/data.md](docs/data.md) lists them. PTB-XL is CC BY 4.0, which requires attribution: cite Wagner et al. 2020 (doi:10.1038/s41597-020-0495-6), the PhysioNet resource (doi:10.13026/kfzx-aw45) and PhysioNet itself (Goldberger et al., Circulation 2000;101(23):e215–e220). To cite the study, use [CITATION.cff](CITATION.cff).

## Sources

- EchoNext: Poterucha et al., *Nature* 2025, 10.1038/s41586-025-09227-0
- ECGFounder: Li et al., *NEJM AI* 2025, 10.1056/AIoa2401033
- PTB-XL: Wagner et al., *Sci Data* 2020, 10.1038/s41597-020-0495-6
- SPH: Liu et al., *Sci Data* 2022, 10.1038/s41597-022-01403-5
- ACS-ECG: Du et al., *Sci Data* 2026, 10.1038/s41597-026-07278-0
- PhysioNet/CinC Challenge 2021: Reyna et al., Computing in Cardiology 2021
- Split conformal: Angelopoulos & Bates, Foundations and Trends in Machine Learning 2023, 10.1561/2200000101
- Label conditional validity: Vovk, ACML 2012, PMLR 25:475-490

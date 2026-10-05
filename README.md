# Carrying an ECG-AI threshold from inpatients to outpatients: a retrospective measurement of three structural heart disease models at one hospital

The study: [REPORT.md](REPORT.md) · Methods in statistical terms, the infarction study and the five-corpus rotation: [SUPPLEMENT.md](SUPPLEMENT.md) · The PPV recomputed for a new site: [PPV.md](PPV.md) ([français](PPV.fr.md)) · Ruben Abbou · 2026

At Columbia, a threshold set to catch 90% of inpatients with moderate or worse structural heart disease caught 71.6% of outpatients with it. Two other models gave the same answer: the published EchoNext mini-model caught 72.7% and the ECGFounder foundation model 71.6%. The AUROC hardly moved, from 0.815 to 0.805. The model separated ill from healthy outpatients almost as well as inpatients, and the threshold landed at another point of the same trade-off: per 100 healthy outpatients it flagged 29, against 58 per 100 healthy inpatients. Milder disease among outpatients explains about a quarter of the fall. Set again on 100 outpatients with known diagnoses, about 26 of them ill, the threshold caught 91.6% of the other outpatients' ill on average and flagged 70 of 100 healthy. Set on 858 separate outpatients of the same hospital, it caught 86.3% (81.7% to 89.9%), 87.8% and 89.7% for the three models, short of 90% for all three, and flagged 52 to 60 of 100 healthy. A clinic adopting such a model has to measure sensitivity on its own patients, and decide what it will pay for it.

![Ill patients caught against healthy patients flagged, inpatients and outpatients](results/figures/fig_curve.png)

Ill patients caught against healthy patients flagged, per 100, for the network trained here, among inpatients and outpatients of EchoNext's test split. The dots mark the threshold set on inpatients of the validation split.

## What a clinic should measure

The sensitivity and specificity published with a model belong to the patients its threshold was set on. At Columbia those were inpatients, and on outpatients the same threshold missed 28 ill patients in 100 instead of 10. An AUROC reported by care setting does not show this, because it compares ill with healthy patients inside one setting, and both groups scored lower among outpatients.

A second threshold, set on the healthy inpatients, sends patients whose scores sit between the two to a human reader. It misses the same patients the first threshold misses. Among outpatients it sent 36 ill and 27 healthy in 100 to a reader, and left 2 healthy in 100 flagged with no reader.

Setting the threshold again on 100 outpatients, about 26 of them ill, drawn from the group it was read on, brought sensitivity to 91.6% on average, at a price in echocardiograms, and still fell short of 90% in 29.9% to 33.1% of the repetitions for the three models. With 200 outpatients the share was 36.4% to 37.0%: more labels narrow the spread around 90% without lowering the share that falls short. Read on outpatients it never saw, a threshold set on the 858 outpatients of EchoNext's validation group fell short of 90% for all three models. Set on every validation patient whatever the setting, it caught 77.1% of ill outpatients: the size of the fall depends on where the threshold is set.

This is a retrospective measurement study on public, de-identified data. It is not a medical device and has no regulatory status, it involved no contact with patients, and nothing here is meant to guide the care of any patient. The ethics approvals and the author's competing interests are in [REPORT.md](REPORT.md).

## Design

EchoNext holds 100,000 ECGs from Columbia, each paired with an echocardiogram. The label studied is moderate or worse structural heart disease on echocardiography, a composite of eleven findings that EchoNext records for each ECG. Thresholds are fitted on the 1,903 inpatient ECGs of EchoNext's validation split and applied unchanged to the 1,059 outpatient ECGs of its test split, one ECG per patient and no patient in both. Structural heart disease is present in 53.2% of the calibration inpatients and in 25.6% of the test outpatients.

Four models score every ECG: a residual network trained on EchoNext for this study; the published EchoNext mini-model, run on its authors' weights; ECGFounder, a foundation model pre-trained at another hospital and frozen under one logistic regression per label; and the study's residual network frozen at random initialisation under the same regressions, the floor a pre-trained model has to clear.

| Model | AUROC, outpatients | Ill outpatients caught | Healthy outpatients flagged, per 100 | Ill caught, threshold set again on 100 outpatients | Repetitions below 90% | Healthy flagged per 100, threshold set again | Ill caught, threshold set on 858 other outpatients |
|---|---|---|---|---|---|---|---|
| Residual network, trained on EchoNext | 0.805 | 71.6% | 29 | 91.6% | 33.1% | 70 | 86.3% |
| EchoNext mini-model, published weights | 0.795 | 72.7% | 29 | 91.9% | 31.1% | 72 | 87.8% |
| ECGFounder, frozen, logistic regressions | 0.791 | 71.6% | 29 | 91.8% | 29.9% | 70 | 89.7% |
| Random initialisation, frozen, logistic regressions | 0.758 | 78.6% | 43 | | | | 91.9% |

An infarction model gives a second case. A residual network trained on PTB-XL, a German research corpus, had its threshold set to catch 90% of infarctions there and caught 83.7% at a hospital in Chongqing and 97.3% at one in Shandong. Between hospitals the model's own separation moved with the threshold: its AUROC went from 0.932 at PTB-XL to 0.793 at Chongqing, where an infarction is an acute event in the discharge diagnosis rather than a pattern on the tracing. [SUPPLEMENT.md](SUPPLEMENT.md) gives the infarction study in full and a rotation of five corpora through the calibration role on five diagnoses: sinus rhythm, atrial fibrillation, left and right bundle-branch block, and first-degree atrioventricular block.

EchoNext is under PhysioNet's restricted licence, so no tracing and no per-record score is in this repository. Each EchoNext model's one-page report is in [reports/transfer/](reports/transfer/); its refitting ladder reads one fixed half of the outpatients and counts a healthy patient sent to a human as covered, and the report's section 3.6 supersedes it.

## The positive predictive value a buyer recomputes

A positive predictive value recomputed by Bayes' rule from a source's sensitivity and specificity, at a target's true prevalence, misses the observed one by a median of 5.3 percentage points over 174 transfers between populations, in either direction, against 0.3 points on 72 controls. The transfers are the EchoNext pairs above, the infarction pairs and the five-corpus rotation. Recalibrating on 100 labelled local ECGs is the one repair of three that raises net benefit on average over those transfers; for EchoNext's outpatients it lowered net benefit at decision thresholds of 5% and 10% for all three models. [PPV.md](PPV.md) gives the measurement and the repairs.

## Reproduce

Every number and figure the report prints is redrawn from the files committed here, so no raw tracing is needed for the infarction study. `outcomes.py` and `subgroups.py` need PTB-XL's `ptbxl_database.csv` (6.6 MB from PhysioNet), which holds each record's patient, sex and age; point `ECS_PTBXL_DIR` at the directory holding it. The EchoNext results need the EchoNext distribution and the stored per-record scores; [SUPPLEMENT.md](SUPPLEMENT.md) gives those commands.

```bash
uv sync                                  # 1 min
uv run pytest -m "not data"              # unit tests, no corpora needed, 2 min
uv run python tests/paper_render.py      # checks REPORT.md, SUPPLEMENT.md, README.md and CITATION.cff against results/
uv run python scripts/figures.py         # redraws the infarction and rotation figures, 5 s
uv run python scripts/paper_figures.py   # redraws the report's three figures from results/echonext_clinical.json
uv run python scripts/infarction_sites.py  # rebuilds results/infarction_sites.json, 2 s
export ECS_PTBXL_DIR=/path/to/ptbxl      # the directory with ptbxl_database.csv
uv run python scripts/outcomes.py        # rebuilds results/outcomes.json, 3 s
uv run python scripts/subgroups.py       # rebuilds results/subgroups.json, 23 s
uv run python scripts/ppv_gap.py         # rebuilds results/ppv_gap.json, 2 s
uv run python scripts/ppv_figures.py     # the two PPV figures, from the committed results
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
- Split conformal: Angelopoulos & Bates, arXiv:2107.07511
- Label conditional validity: Vovk, ACML 2012, PMLR 25:475-490

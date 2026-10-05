# Threshold generalisation to new sites and care settings for ECG-AI diagnostic support

Infarction across hospitals: [REPORT.md](REPORT.md) · Structural heart disease, inpatients to outpatients: [ECHONEXT.md](ECHONEXT.md) · The PPV recomputed for a new site: [PPV.md](PPV.md) ([français](PPV.fr.md)) · A sixth corpus, Beth Israel Deaconess on machine labels: [MIMIC.md](MIMIC.md) · Ruben Abbou · 2026

An ECG classifier's decision threshold is set on one population and used on others. Conformal calibration sets it so that 90% of tracings receive a set of possible diagnoses holding the right one. On PTB-XL, a German research corpus, the target holds over all tracings (89.9%) and fails within infarction (73.2%). Carried unchanged to a hospital in Chongqing, the threshold covers 72.5% of infarctions; recalibrating on 100 tracings labelled there brings it to 88.9% on the held-out tracings. At Columbia, thresholds fitted on 1,903 inpatients cover about 72% of outpatients with structural heart disease for each of the three strongest models, so the cause is the outpatients' ECGs rather than one model. Refitting them on 100 labelled outpatient ECGs brings that to between 93.4% and 97.6%, averaged over 200 draws. A site adopting such a tool has to measure coverage within each diagnosis and care setting, on its own patients.

![Coverage by site and calibration scheme](results/figures/fig3_coverage.png)

Coverage by site (rows) and calibration scheme (columns) as the confidence asked for rises. Red: infarction cases. Grey: all cases. Dashed: the coverage asked for.

## What a receiving site should measure

The 90% is an average over patients and over draws of the calibration data, and an average can be met while one group of patients is failed: at PTB-XL, 89.9% of all sets hold the right diagnosis and 73.2% of infarction sets do. Fitting one threshold within each diagnosis makes the 90% hold inside each, which still depends on the patient's true diagnosis, unknown at the point of care. Nothing in this framework promises 90% to a single patient.

When the model cannot rule a diagnosis out, the tracing receives a set holding both labels. That is a deferral: there is no machine answer, and a human reads the ECG. At the 90% target, PTB-XL defers about 5 tracings in 100 with one threshold for all tracings and about 10 in 100 with one threshold per diagnosis. `results/abstention.json` gives the rate per scheme and per confidence level.

What happens at a third site cannot be predicted from these two. Infarction coverage rose at Shandong, to 93.6%, and fell at Chongqing. A site can reuse the measurement code, and what the measurement needs is cases of the rarer diagnosis: 865 infarctions pin coverage within that diagnosis to two points, about 5,800 tracings at Chongqing's prevalence and 86,000 at Shandong's.

This is a retrospective measurement study on public, de-identified data. It is not a medical device and has no regulatory status, it involved no contact with patients, and nothing here is meant to guide the care of any patient. The ethics approvals and the author's competing interests open [REPORT.md](REPORT.md).

## Design

A residual network trained on PTB-XL supplies scores that are then held fixed (AUROC 0.932 for infarction on the benchmark's test fold). Three schemes turn those scores into an output. A single threshold tuned to 90% sensitivity labels every tracing. Split conformal prediction, which fits its threshold on tracings the model never trained on, either pools all of them, so the 90% holds on average, or fits one threshold within each diagnosis (Mondrian calibration), so it holds inside each. A fourth scheme, label-shift weighting, reweights the calibration tracings toward the target's estimated share of each diagnosis. Each figure is a mean over 200 draws that split PTB-XL's test patients in half.

The infarction transfer calibrates on PTB-XL and carries the threshold, with no target label reaching it, to Shandong and Chongqing. The rotation gives five corpora the calibration role in turn, each threshold spent unchanged on the other four, on five diagnoses all of them annotate: sinus rhythm, atrial fibrillation, left and right bundle-branch block, and first-degree atrioventricular block. [docs/data.md](docs/data.md) gives the label mapping.

| Corpus | Country | Records in the release | Role |
|---|---|---|---|
| PTB-XL | Germany | 21,799 | infarction source; rotation |
| SPH (Shandong) | China | 25,770 | infarction target; rotation |
| ACS-ECG (Chongqing) | China | 19,955 | infarction target |
| Chapman-Shaoxing and Ningbo | China | 45,152 | rotation, as one source |
| Georgia | United States | 10,344 | rotation |
| CPSC 2018 and its extension | China | 10,330 | rotation |

## Structural heart disease, from inpatients to outpatients

EchoNext holds 100,000 ECGs from Columbia, each paired with an echocardiogram. The label studied is moderate or worse structural heart disease on echocardiography, a composite of eleven findings that EchoNext records for each ECG. Thresholds are fitted on the 1,903 inpatient ECGs of EchoNext's validation split and applied unchanged to the 1,059 outpatient ECGs of its test split, one ECG per patient and no patient in both. Structural heart disease is present in 53.2% of the 1,903 calibration inpatients and in 25.6% of the 1,059 test outpatients.

Per-label conformal calibration fits one threshold on the calibration patients with the disease and another on those without it, each placed so that 90% of its group receive a set of labels holding the right one. A patient whose set holds both labels, disease and no disease, gets no machine answer and goes to a human reader. With a separate threshold for each class, a fall in prevalence from inpatients to outpatients cannot by itself lower the coverage of patients with the disease.

Four models score every ECG: a residual network trained on EchoNext for this study; the published EchoNext mini-model, run on its authors' weights; ECGFounder, a foundation model pre-trained at another hospital and frozen under one logistic regression per label; and the study's residual network frozen at random initialisation under the same regressions, the floor a pre-trained model has to clear.

| Model | AUROC for the composite, whole test split | Outpatients with the disease covered | Outpatients sent to a human |
|---|---|---|---|
| Residual network, trained on EchoNext | 0.834 | 71.6% | 29.4% |
| EchoNext mini-model, published weights | 0.820 | 72.7% | 32.3% |
| ECGFounder, frozen, logistic regressions | 0.824 | 71.6% | 32.2% |
| Random initialisation, frozen, logistic regressions | 0.792 | 78.6% | 44.2% |

Among outpatients with structural heart disease, the three models with the highest AUROC (the trained network, the mini-model and ECGFounder) each cover between 71.6% and 72.7%, against the 90% asked for. When a network trained here, a network trained by the EchoNext authors and a foundation model pre-trained elsewhere lose the same coverage, the cause lies in the outpatients' ECGs rather than in one model. Milder disease among outpatients explains about a third of that loss: reweighted to the inpatients' severity, coverage of the ill reaches 76.9% to 78.7%, still short of 90%. The randomly initialised floor model covers 78.6% of outpatients with the disease because it sends 44.2% of all outpatients to a human reader. Refitting the per-label thresholds on 100 labelled outpatient ECGs, drawn 200 times from one half of the outpatients, covers between 93.4% and 97.6% of the other half's patients with the disease for the trained network, the mini-model and ECGFounder, on average over the draws.

EchoNext is under PhysioNet's restricted licence, so no tracing and no per-record score is in this repository. The per-label results, the commands and how the published weights were run are in [ECHONEXT.md](ECHONEXT.md), and each model's one-page report is in [reports/transfer/](reports/transfer/).

## The positive predictive value a buyer recomputes

A positive predictive value recomputed by Bayes' rule from a source's sensitivity and specificity, at a target's true prevalence, misses the observed one by a median of 5.3 percentage points over 174 transfers between populations, in either direction, against 0.3 points on 72 controls. The transfers are the EchoNext pairs above, the infarction pairs and the five-corpus rotation. Recalibrating on 100 labelled local ECGs is the one repair of three that raises net benefit on average. [PPV.md](PPV.md) gives the measurement and the repairs.

## Reproduce

Every number and figure the report prints is redrawn from the scores committed here, so no raw tracing is needed. `outcomes.py` and `subgroups.py` need PTB-XL's `ptbxl_database.csv` (6.6 MB from PhysioNet), which holds each record's patient, sex and age; point `ECS_PTBXL_DIR` at the directory holding it.

```bash
uv sync                                  # 1 min
uv run pytest -m "not data"              # unit tests, no corpora needed, 2 min
uv run python scripts/figures.py         # redraws every figure, 5 s
uv run python scripts/figures.py --lang fr  # the report's five figures in French, *_fr.png
export ECS_PTBXL_DIR=/path/to/ptbxl      # the directory with ptbxl_database.csv
uv run python scripts/outcomes.py        # rebuilds results/outcomes.json, 3 s
uv run python scripts/subgroups.py       # rebuilds results/subgroups.json, 23 s
uv run python scripts/ppv_gap.py         # rebuilds results/ppv_gap.json, 2 s
uv run python scripts/ppv_figures.py     # the two PPV figures, from the committed results
```

`scripts/repairs.py` (52 s on an Apple M5) needs the patient tables of PTB-XL, Shandong and Chongqing beside the corpora, and EchoNext's stored scores for its EchoNext rows.

Timings are wall clock on a six-core i7-8700, CPU only; the full suite on a cold clone took 28 minutes. Re-scoring from the raw tracings, the corpus downloads and the rotation chain are in [docs/data.md](docs/data.md).

## Code

| Module | Role |
|---|---|
| `src/ecs/conformal.py` | split conformal (LAC + APS scores), Mondrian quantiles, covariate- and label-shift weighting, BBSE |
| `src/ecs/metrics.py` | coverage, Wilson intervals, class-conditional coverage, set size, abstention, effective sample size |
| `src/ecs/labels.py` | one comparable MI label across three annotation schemes (SCP-ECG, AHA, discharge diagnosis) |
| `src/ecs/config.py` | corpus paths and the label vocabulary |
| `src/ecs/small_set.py` | the five rotation diagnoses, the codes each corpus names them by, and the joins that stayed ambiguous |
| `src/ecs/challenge.py` | the Challenge-2021 partitions: headers, SNOMED labels, completeness against the bundle manifest |
| `src/ecs/rotation.py` | the five corpora split the same way, capped to a common size |
| `src/ecs/encoders.py` | the five encoder arms and the chain each one demands |
| `src/ecs/echonext.py` | EchoNext's files, one provenance row per tracing, and its inpatient, emergency and outpatient cohorts |
| `src/ecs/transfer.py` | thresholds fitted on a source cohort and spent unchanged on a target, and refitted on 25 to 400 target labels |
| `src/ecs/calibration.py` | calibration curve, slope and intercept, Brier score, net benefit and positive predictive value of a probability at one site |
| `src/ecs/ppv_gap.py` | the PPV recomputed by Bayes' rule against the PPV observed, with Wilson and bootstrap intervals and a likelihood-ratio check of label shift |
| `src/ecs/repairs.py` | label-free prevalence correction, recalibration on local labels and per-label abstention, judged on net benefit |
| `src/ecs/transfer_report.py` | the one-page transfer report, rendered from `results/echonext_transfer.json` alone |
| `src/ecs/embedding_store.py` | encoder vectors filed by encoder, version and tracing digest, outside the repository |
| `src/ecs/echonext_mini.py` | the published EchoNext mini-model, rebuilt to run its own checkpoint |
| `src/ecs/mimic.py` | MIMIC-IV-ECG: cart statements to the five diagnoses, one ECG per patient, the carts ordered in time |
| `mappings/` | the three published code tables the label mapping joins on, with provenance and digests |

`pre-commit` runs `ruff`, `mypy --disallow-untyped-defs` and `pytest` on every commit, and the unit tests run again before a push.

## Licence and citation

The code is MIT ([LICENSE](LICENSE)), with three exceptions. `third_party/ecg_jepa/` is vendored from Sehun Kim's ECG-JEPA release under MIT, with its own LICENSE. `third_party/ecgfounder/` is not covered by the root licence; its MIT and Apache 2.0 notices are in [PROVENANCE.md](third_party/ecgfounder/PROVENANCE.md). The HuBERT-ECG weights are CC BY-NC 4.0, and so is every number computed from them: `results/embeddings/hubert_ecg/` and the HuBERT-ECG rows of `results/arms.json` are for research use only.

The committed score files hold one row per record (identifier, label, score), derived data under each corpus's licence; [docs/data.md](docs/data.md) lists them. PTB-XL is CC BY 4.0, which requires attribution: cite Wagner et al. 2020 (doi:10.1038/s41597-020-0495-6), the PhysioNet resource (doi:10.13026/kfzx-aw45) and PhysioNet itself (Goldberger et al., Circulation 2000;101(23):e215–e220). To cite the study, use [CITATION.cff](CITATION.cff).

## Sources

- PTB-XL: Wagner et al., *Sci Data* 2020, 10.1038/s41597-020-0495-6
- SPH: Liu et al., *Sci Data* 2022, 10.1038/s41597-022-01403-5
- ACS-ECG: Du et al., *Sci Data* 2026, 10.1038/s41597-026-07278-0
- PhysioNet/CinC Challenge 2021: Reyna et al., Computing in Cardiology 2021
- Split conformal: Angelopoulos & Bates, arXiv:2107.07511
- APS: Romano, Sesia & Candès, NeurIPS 2020
- Covariate shift: Tibshirani, Barber, Candès & Ramdas, NeurIPS 2019
- Label conditional validity: Vovk, ACML 2012, PMLR 25:475-490
- Conformal prediction under label shift: Podkopaev & Ramdas, UAI 2021, arXiv:2103.03323
- BBSE: Lipton, Wang & Smola, ICML 2018

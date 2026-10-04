# Per-label conformal calibration of ECG classifiers across hospitals and care settings

Infarction across hospitals, and structural heart disease from inpatients to outpatients: [REPORT.md](REPORT.md) · Ruben Abbou · 2026

An ECG classifier's decision threshold is set on one population and used on others. Conformal calibration sets it so that 90% of tracings receive a set of possible diagnoses holding the right one. On PTB-XL, a German research corpus, the target holds over all tracings ({{s.none.ptbxl.all}}) and fails within infarction ({{s.none.ptbxl.mi}}). Carried unchanged to a hospital in Chongqing, the threshold covers {{s.none.acs.mi}} of infarctions; recalibrating on 100 tracings labelled there brings it to {{ts.none.recalibrated.100}} on the held-out tracings. At Columbia, thresholds fitted on {{calibration_ecgs}} inpatients cover about {{strongest_about}} of outpatients with structural heart disease for each of the three strongest models, so the cause is the outpatients' ECGs rather than one model. Refitting them on 100 labelled outpatient ECGs brings that to between {{ladder_low}} and {{ladder_high}}, averaged over {{ladder_draws}} draws. A site adopting such a tool has to measure coverage within each diagnosis and care setting, on its own patients.

![Coverage by site and calibration scheme](results/figures/fig3_coverage.png)

Coverage by site (rows) and calibration scheme (columns) as the confidence asked for rises. Red: infarction cases. Grey: all cases. Dashed: the coverage asked for.

## What a receiving site should measure

The 90% is an average over patients and over draws of the calibration data, and an average can be met while one group of patients is failed: at PTB-XL, {{s.none.ptbxl.all}} of all sets hold the right diagnosis and {{s.none.ptbxl.mi}} of infarction sets do. Fitting one threshold within each diagnosis makes the 90% hold inside each, which still depends on the patient's true diagnosis, unknown at the point of care. Nothing in this framework promises 90% to a single patient.

When the model cannot rule a diagnosis out, the tracing receives a set holding both labels. That is a deferral: there is no machine answer, and a human reads the ECG. At the 90% target, PTB-XL defers about {{d.pooled_deferred_per_100}} tracings in 100 with one threshold for all tracings and about {{d.perlabel_deferred_per_100}} in 100 with one threshold per diagnosis. `results/abstention.json` gives the rate per scheme and per confidence level.

What happens at a third site cannot be predicted from these two. Infarction coverage rose at Shandong, to {{s.none.sph.mi}}, and fell at Chongqing. A site can reuse the measurement code, and what the measurement needs is cases of the rarer diagnosis: {{aux.mi_cases_needed}} infarctions pin coverage within that diagnosis to two points, about {{d.acs_tracings_rounded}} tracings at Chongqing's prevalence and {{d.sph_tracings_rounded}} at Shandong's.

This is a retrospective measurement study on public, de-identified data. It is not a medical device and has no regulatory status, it involved no contact with patients, and nothing here is meant to guide the care of any patient. The ethics approvals and the author's competing interests open [REPORT.md](REPORT.md).

## Design

A residual network trained on PTB-XL supplies scores that are then held fixed (AUROC {{inf.baseline_auroc}} for infarction on the benchmark's test fold). Three schemes turn those scores into an output. A single threshold tuned to 90% sensitivity labels every tracing. Split conformal prediction, which fits its threshold on tracings the model never trained on, either pools all of them, so the 90% holds on average, or fits one threshold within each diagnosis (Mondrian calibration), so it holds inside each. A fourth scheme, label-shift weighting, reweights the calibration tracings toward the target's estimated share of each diagnosis. Each figure is a mean over 200 draws that split PTB-XL's test patients in half.

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

EchoNext holds {{records}} ECGs from Columbia, each paired with an echocardiogram. The label studied is moderate or worse structural heart disease on echocardiography, a composite of eleven findings that EchoNext records for each ECG. Thresholds are fitted on the {{calibration_ecgs}} inpatient ECGs of EchoNext's validation split and applied unchanged to the {{outpatient_ecgs}} outpatient ECGs of its test split, one ECG per patient and no patient in both. Structural heart disease is present in {{prevalence_in}} of the {{calibration_ecgs}} calibration inpatients and in {{prevalence_out}} of the {{outpatient_ecgs}} test outpatients.

Per-label conformal calibration fits one threshold on the calibration patients with the disease and another on those without it, each placed so that 90% of its group receive a set of labels holding the right one. A patient whose set holds both labels, disease and no disease, gets no machine answer and goes to a human reader. With a separate threshold for each class, a fall in prevalence from inpatients to outpatients cannot by itself lower the coverage of patients with the disease.

Four models score every ECG: a residual network trained on EchoNext for this study; the published EchoNext mini-model, run on its authors' weights; ECGFounder, a foundation model pre-trained at another hospital and frozen under one logistic regression per label; and the study's residual network frozen at random initialisation under the same regressions, the floor a pre-trained model has to clear.

| Model | AUROC for the composite, whole test split | Outpatients with the disease covered | Outpatients sent to a human |
|---|---|---|---|
| Residual network, trained on EchoNext | {{ea.resnet.test}} | {{e.resnet.outpatient.perlabel.pos}} | {{e.resnet.outpatient.perlabel.abst}} |
| EchoNext mini-model, published weights | {{ea.echonext_mini.test}} | {{e.echonext_mini.outpatient.perlabel.pos}} | {{e.echonext_mini.outpatient.perlabel.abst}} |
| ECGFounder, frozen, logistic regressions | {{ea.ecgfounder.test}} | {{e.ecgfounder.outpatient.perlabel.pos}} | {{e.ecgfounder.outpatient.perlabel.abst}} |
| Random initialisation, frozen, logistic regressions | {{ea.random_init.test}} | {{e.random_init.outpatient.perlabel.pos}} | {{e.random_init.outpatient.perlabel.abst}} |

Among outpatients with structural heart disease, the three models with the highest AUROC (the trained network, the mini-model and ECGFounder) each cover between {{strongest_low}} and {{strongest_high}}, against the 90% asked for. When a network trained here, a network trained by the EchoNext authors and a foundation model pre-trained elsewhere lose the same coverage, the cause lies in the outpatients' ECGs rather than in one model. Milder disease among outpatients explains about a third of that loss: reweighted to the inpatients' severity, coverage of the ill reaches {{reweighted_low}} to {{reweighted_high}}, still short of 90%. The randomly initialised floor model covers {{floor_sens_out}} of outpatients with the disease because it sends {{floor_referred_out}} of all outpatients to a human reader. Refitting the per-label thresholds on 100 labelled outpatient ECGs, drawn {{ladder_draws}} times from one half of the outpatients, covers between {{ladder_low}} and {{ladder_high}} of the other half's patients with the disease for the trained network, the mini-model and ECGFounder, on average over the draws.

EchoNext is under PhysioNet's restricted licence, so no tracing and no per-record score is in this repository. The per-label results, the commands and how the published weights were run are in [REPORT.md](REPORT.md), sections 2.2, 3.7 and 3.8.

## Reproduce

Every number and figure the report prints is redrawn from the scores committed here, so no raw tracing is needed. `outcomes.py` and `subgroups.py` need PTB-XL's `ptbxl_database.csv` (6.6 MB from PhysioNet), which holds each record's patient, sex and age; point `ECS_PTBXL_DIR` at the directory holding it.

```bash
uv sync                                  # 1 min
uv run pytest -m "not data"              # unit tests, no corpora needed, 2 min
uv run python scripts/figures.py         # redraws every figure, 5 s
uv run python scripts/figures.py --lang fr  # the five translated figures in French, *_fr.png
export ECS_PTBXL_DIR=/path/to/ptbxl      # the directory with ptbxl_database.csv
uv run python scripts/outcomes.py        # rebuilds results/outcomes.json, 3 s
uv run python scripts/subgroups.py       # rebuilds results/subgroups.json, 23 s
```

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
| `src/ecs/embedding_store.py` | encoder vectors filed by encoder, version and tracing digest, outside the repository |
| `src/ecs/echonext_mini.py` | the published EchoNext mini-model, rebuilt to run its own checkpoint |
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

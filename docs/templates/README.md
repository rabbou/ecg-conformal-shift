# A 90% ECG threshold does not hold at another hospital or in another care setting

Infarction across hospitals, and structural heart disease from inpatients to outpatients: [REPORT.md](REPORT.md) · Ruben Abbou · 2026

An ECG model gives each recording a score, and a cut-off on that score, the threshold, decides who is called ill. The cut-off is set on one group of patients, usually to catch 90% of the ill, and then used on others. Carried to new patients, a cut-off set for 90% catches a different share, while the AUROC a receiving site is shown, the usual single accuracy score, can stay where it was: the AUROC does not say how many ill patients a given cut-off catches.

Conformal prediction, a standard statistical method, adds a grey zone around the cut-off: a patient scoring inside it gets no machine answer and is referred to a doctor. It is built to guarantee a share of patients given no wrong answer (the right answer, or a referral), which statisticians call the coverage; among the ill, it is the share caught, flagged or referred. On PTB-XL, a German research collection, with the grey zone set for all patients together, 90% is met over all tracings ({{s.none.ptbxl.all}}) and missed among infarctions ({{s.none.ptbxl.mi}}). Carried unchanged to a hospital in Chongqing, the cut-offs catch {{ts.none.recalibrated.0}} of infarctions among tracings kept for evaluation; reset on 100 tracings with known diagnoses from Chongqing, {{ts.none.recalibrated.100}}. At Columbia, cut-offs set on {{calibration_ecgs}} inpatients catch about {{strongest_about}} of outpatients with structural heart disease for each of the three strongest models, while the trained network's AUROC moves only from {{resnet_auroc_in}} among inpatients to {{resnet_auroc_out}} among outpatients. That points at the outpatients' ECGs rather than at one model, though all three were fitted to the same EchoNext training split. Reset on 100 outpatient ECGs with known diagnoses, the cut-offs catch between {{ladder_low}} and {{ladder_high}} of the ill, on average over {{ladder_draws}} draws. A site adopting such a tool has to count, on its own patients, the share of the ill each cut-off catches, for each diagnosis and care setting.

![Share given no wrong answer, by hospital and way of setting the cut-offs](results/figures/fig3_coverage.png)

Share of patients given no wrong answer, by hospital (rows) and by way of setting the cut-offs (columns), as the share asked for rises. Red: patients with infarction. Grey: all patients. Dashed: the share asked for.

## What a receiving site should measure

The 90% is an average over all patients, and an average can be met while one group of patients is failed: at PTB-XL, {{s.none.ptbxl.all}} of all patients get no wrong answer and {{s.none.ptbxl.mi}} of infarctions are caught. Setting one cut-off for the ill and another for the healthy (per-diagnosis calibration) makes the 90% hold within each group, which still depends on the patient's true diagnosis, unknown when the ECG is read. Nothing in this framework promises 90% to a single patient.

When the model cannot rule a diagnosis out, the patient is referred: there is no machine answer, and a doctor reads the ECG. Asked for 90%, PTB-XL refers about {{d.pooled_deferred_per_100}} tracings in 100 with the cut-offs set for all patients together and about {{d.perlabel_deferred_per_100}} in 100 with the cut-offs set per diagnosis. `results/abstention.json` gives the rate per option and per share asked for.

What happens at a third site cannot be predicted from these two. Infarctions caught rose at Shandong, to {{s.none.sph.mi}}, and fell at Chongqing. A site can reuse the measurement code, and what the measurement needs is patients with the rarer diagnosis: {{aux.mi_cases_needed}} infarctions pin the share caught to two points, about {{d.acs_tracings_rounded}} tracings at Chongqing's share of infarction and {{d.sph_tracings_rounded}} at Shandong's.

This is a retrospective measurement study on public, de-identified data. It is not a medical device and has no regulatory status, it involved no contact with patients, and nothing here is meant to guide the care of any patient. The ethics approvals and the author's competing interests are in the declarations of [REPORT.md](REPORT.md).

## Design

A residual network trained on PTB-XL supplies scores that are then held fixed (AUROC {{inf.baseline_auroc}} for infarction on the benchmark's test fold). Three options turn those scores into an answer. A single cut-off set for 90% sensitivity answers yes or no for every tracing. Conformal prediction, which sets its cut-offs on tracings the model never trained on (the calibration patients), adds the grey zone, its edges set either for all tracings together, so the 90% holds on average, or separately for the ill and the healthy (per-diagnosis, or Mondrian, calibration), so it holds within each. A fourth option reweights the calibration tracings toward the share of each diagnosis estimated at the receiving site. Each figure is an average over 200 random splits of PTB-XL's test patients in half.

The infarction study sets the cut-offs on PTB-XL and carries them, with no diagnosis from the receiving hospital reaching them, to Shandong and Chongqing. The rotation gives five collections the role of setting the cut-offs in turn, each cut-off used unchanged on the other four, on five diagnoses all of them record: sinus rhythm, atrial fibrillation, left and right bundle-branch block, and first-degree atrioventricular block. [docs/data.md](docs/data.md) gives how the diagnoses were matched.

| Collection | Country | Records in the release | Role |
|---|---|---|---|
| PTB-XL | Germany | 21,799 | sets the infarction cut-offs; rotation |
| SPH (Shandong) | China | 25,770 | receives the infarction cut-offs; rotation |
| ACS-ECG (Chongqing) | China | 19,955 | receives the infarction cut-offs |
| Chapman-Shaoxing and Ningbo | China | 45,152 | rotation, as one collection |
| Georgia | United States | 10,344 | rotation |
| CPSC 2018 and its extension | China | 10,330 | rotation |

## Structural heart disease, from inpatients to outpatients

EchoNext holds {{records}} ECGs from Columbia, each paired with an echocardiogram. The diagnosis studied is moderate or worse structural heart disease on echocardiography, a composite of eleven findings that EchoNext records for each ECG. The cut-offs are set on the {{calibration_ecgs}} inpatient ECGs of EchoNext's validation part and used unchanged on the {{outpatient_ecgs}} outpatient ECGs of its test part, one ECG per patient and no patient in both. Structural heart disease is present in {{prevalence_in}} of the {{calibration_ecgs}} calibration inpatients and in {{prevalence_out}} of the {{outpatient_ecgs}} test outpatients.

The per-diagnosis option sets one cut-off on the calibration patients with the disease and another on those without it, each placed so that 90% of its group get no wrong answer. A patient scoring between the two gets no machine answer and is referred to a doctor. With a separate cut-off for each group, a fall in the share of ill patients from inpatients to outpatients cannot by itself lower the share of the ill caught.

Four models score every ECG: a residual network trained on EchoNext for this study; the published EchoNext mini-model, run on its authors' weights; ECGFounder, a model pre-trained at another hospital and left unchanged, read through one logistic regression per finding; and the study's residual network left untrained, with random weights, read through the same regressions, the floor a pre-trained model has to clear.

| Model | AUROC for the composite, whole test part | Outpatients with the disease caught | Outpatients referred to a doctor |
|---|---|---|---|
| Residual network, trained on EchoNext | {{ea.resnet.test}} | {{e.resnet.outpatient.perlabel.pos}} | {{e.resnet.outpatient.perlabel.abst}} |
| EchoNext mini-model, published weights | {{ea.echonext_mini.test}} | {{e.echonext_mini.outpatient.perlabel.pos}} | {{e.echonext_mini.outpatient.perlabel.abst}} |
| ECGFounder, frozen, logistic regressions | {{ea.ecgfounder.test}} | {{e.ecgfounder.outpatient.perlabel.pos}} | {{e.ecgfounder.outpatient.perlabel.abst}} |
| Random initialisation, frozen, logistic regressions | {{ea.random_init.test}} | {{e.random_init.outpatient.perlabel.pos}} | {{e.random_init.outpatient.perlabel.abst}} |

Among outpatients with structural heart disease, the three models with the highest AUROC (the trained network, the mini-model and ECGFounder) each catch between {{strongest_low}} and {{strongest_high}}, against the 90% asked for. When a network trained here, a network trained by the EchoNext authors and a model pre-trained elsewhere lose the same share, that points at the outpatients' ECGs rather than at one model, though all three were fitted to the same EchoNext training split. Milder disease among outpatients explains about a third of that loss: reweighted to look like the ill inpatients, the share caught reaches {{reweighted_low}} to {{reweighted_high}}, still short of 90%. The untrained floor model catches {{floor_sens_out}} of outpatients with the disease because it refers {{floor_referred_out}} of all outpatients to a doctor. Resetting the per-diagnosis cut-offs on 100 outpatient ECGs with known diagnoses, drawn {{ladder_draws}} times from one half of the outpatients, catches between {{ladder_low}} and {{ladder_high}} of the other half's patients with the disease for the trained network, the mini-model and ECGFounder, on average over the draws.

EchoNext is under PhysioNet's restricted licence, so no tracing and no per-record score is in this repository. The per-diagnosis results are in [REPORT.md](REPORT.md), sections 3.3 to 3.6, and how the published weights were run is in its Appendix C.

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
| `src/ecs/conformal.py` | conformal prediction (LAC and APS scores), per-diagnosis (Mondrian) quantiles, covariate- and label-shift weighting, BBSE |
| `src/ecs/metrics.py` | coverage, Wilson intervals, coverage per diagnosis, set size, referral, effective sample size |
| `src/ecs/labels.py` | one comparable infarction diagnosis across three annotation schemes (SCP-ECG, AHA, discharge diagnosis) |
| `src/ecs/config.py` | corpus paths and the label vocabulary |
| `src/ecs/small_set.py` | the five rotation diagnoses, the codes each corpus names them by, and the joins that stayed ambiguous |
| `src/ecs/challenge.py` | the Challenge-2021 partitions: headers, SNOMED labels, completeness against the bundle manifest |
| `src/ecs/rotation.py` | the five corpora split the same way, capped to a common size |
| `src/ecs/encoders.py` | the five encoder arms and the chain each one demands |
| `src/ecs/echonext.py` | EchoNext's files, one provenance row per tracing, and its inpatient, emergency and outpatient cohorts |
| `src/ecs/transfer.py` | cut-offs set on one cohort and used unchanged on another, and reset on 25 to 400 ECGs of the receiving cohort |
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

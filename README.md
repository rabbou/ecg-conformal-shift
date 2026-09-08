# ecg-conformal-shift

Split conformal prediction on 12-lead ECG, calibrated on one hospital corpus
and evaluated without re-calibration on others. The question measured is
whether the 90% coverage guarantee survives the change of hospital; the two
standard corrections, Mondrian per-class calibration and label-shift
weighting, are compared on the same break.

Two measurements sit on that question. The first is infarction, calibrated on
PTB-XL and spent on Shandong and Chongqing. The second rotates the calibration
source over five corpora and five diagnoses a cardiologist reads at a glance,
so that the break has a spread across sources instead of a single pair.

The findings, the three figures and the limitations are in
[REPORT.md](REPORT.md). Common questions are answered in
[QUESTIONS.md](QUESTIONS.md). This file documents the corpora, the layout and
the reproduction commands.

## Cohorts

| Cohort | Country, years | Records | Patients | MI prevalence | Role |
|---|---|---|---|---|---|
| PTB-XL v1.0.3 | Germany, 1989–96 | 21,799 | 18,869 | 25.09% | calibration |
| SPH (Shandong) | China, 2019–20 | 25,770 | 24,666 | 1.01% | shifted test |
| ACS-ECG (Chongqing) | China, 2015–24 | 17,960 | 17,018 | 14.92% acute MI | shifted test |

All three are public and permissively licensed (CC BY 4.0, CC0, CC0). The
counts are computed from each corpus's description file, not quoted from a
paper, and are pinned by tests in `tests/test_labels.py`.

The source rotation adds three corpora from the PhysioNet/CinC Challenge-2021
bundle, and reuses PTB-XL and Shandong as sources in their turn.

| Corpus | Country | Records | Role in the rotation |
|---|---|---|---|
| PTB-XL | Germany | 21,799 | source and target |
| SPH (Shandong) | China | 25,770 | source and target |
| Chapman-Shaoxing and Ningbo | China | 45,152 | source and target |
| Georgia | United States | 10,344 | source and target |
| CPSC 2018 and its extension | China | 10,330 | source and target |

Five diagnoses: sinus rhythm, atrial fibrillation, left bundle-branch block,
right bundle-branch block, first-degree atrioventricular block. The mapping
that puts three annotation schemes (SNOMED CT, AHA, SCP-ECG) on those five
classes is in `results/label_map.json`; it reproduces the Challenge's own
published per-partition counts, and it names the five joins that could not be
made cleanly instead of choosing quietly. Sinus rhythm is refused on Shandong,
whose "Normal ECG" code is a narrower statement than the Challenge's "sinus
rhythm".

## Corpora on disk

Contents recorded from the distributed files on 2026-08-23.

| | PTB-XL | SPH (Shandong) | ACS-ECG (Chongqing) |
|---|---|---|---|
| Where | `~/Developer/ptbxl5d/data/records500/` | `data/sph/records/*.h5` | `data/acs/row_data/*.{dat,hea}` |
| Format | WFDB, format 16 (int16), `.hea` header | HDF5, one dataset `ecg`, float16, no attributes | WFDB, format 16 (int16), `.hea` header |
| Rate, length | 500 Hz, 5000 samples | 500 Hz, 5000–28,000 samples in steps of 500 (18,842 of 25,770 exactly 5000) | 500 Hz, 5000 samples |
| Amplitude | 1000 ADC units per mV, baseline 0 (header) | mV, stored as float16 (paper: 24-bit ADC, "16-bit precision") | 1000 ADC units per mV, baseline 0 (header) |
| Lead order | in each header (`AVR`/`AVL`/`AVF` upper-case) | not in the file; paper, Data Records: I, II, III, aVR, aVL, aVF, V1–V6 | in each header (`aVR` lower-case) |
| Filtering at source | Schiller device | MedEx MECG-200: mains, baseline wander and muscle noise removed by the machine, nothing added by the authors | Mecg-300 (Medex); the paper describes none |
| Records | 21,799 | 25,770 | 19,955 = 17,960 `train.csv` + 1,995 `test.csv` (no labels) |

Two properties of the Chongqing files are absent from the paper. The header
columns WFDB reserves for a lead's initial value and checksum hold the lead's
maximum and minimum, so `wfdb` reads the samples but the checksum cannot
validate them. Every sample is an even number of ADC units, so the effective
resolution is 2 µV against PTB-XL's 1 µV. The CSV column `ecg_row_record`
identifies the file (`04904.dat`); the CSV entries and the files on disk match
one-to-one, and no patient appears in both CSVs.

A full pass through the ingestion chain (`scripts/scan_corpora.py`, counts in
`results/ingest_report.json`) keeps every PTB-XL and Shandong record and drops
five of Chongqing's 19,955: two (`03228`, `14262`) whose `.dat` holds 3,500
samples under a header stating 5,000, and three (`02008`, `03054`, `16558`)
with WFDB's missing-sample code, which reads back as NaN. 6,928 Shandong
records are longer than ten seconds and are cropped to the first ten.

Reading cost: `wfdb.rdrecord` spends ~23 ms per record parsing the header
(wfdb 4.3.1 does it through pandas) against 0.8 ms reading the samples, so a
full pass over a WFDB corpus takes about eight minutes on a laptop; the HDF5
corpus reads at ~7 ms per record.

`scripts/fetch_open_corpora.sh` contains the figshare file ids and MD5 sums
for the three Chongqing archives.

## Prevalence gap

Infarction is 25 times rarer in Shandong than in PTB-XL. The gap is not a
labelling artefact: dropping PTB-XL's five subendocardial-injury statements
moves prevalence only from 25.09% to 24.26%. PTB-XL is a research corpus
enriched for pathology and Shandong is an unselected hospital series. 89.6% of
Shandong's infarctions carry the AHA modifier for an old infarct, which is the
target PTB-XL's label mostly describes; the remaining 10.4% are marked acute,
recent, or carry no modifier at all, so the two label sets are close but not
the same thing.

The dominant shift is therefore in the class prior P(Y) rather than in the
feature distribution P(X), and covariate-shift weighting does not apply. Two
corrections are implemented:

- Mondrian (class-conditional) conformal: one threshold per class. Exactly
  valid in finite samples under any change of class proportions, with nothing
  to estimate. Vovk (ACML 2012) calls this label conditional validity and
  proves it in his Proposition 3.
- Label-shift weighting: reweight by `w(y) = q(y)/p(y)` in the Tibshirani
  form, with the target prior estimated by BBSE. The guarantee is asymptotic
  and depends on that estimate, so effective sample size is reported next to
  every result.

Chongqing is the harder target: its positive is the dataset's `AMI` column, set
from the discharge diagnosis, in a cohort every patient of which underwent
coronary angiography. That is a different clinical event from an ECG diagnosis
of an old infarct, so the label definition changes as well as the prevalence.

## Layout

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
| `mappings/` | the three published code tables the label mapping joins on, with provenance and digests |

## Encoder arms

Five ways of turning a tracing into a vector, compared on the same records.
Four are published encoders; the fifth is this project's own ResNet1d frozen at
its random initialisation, the floor the others have to clear.

| Arm | Pre-trained on | Saw PTB-XL | Licence |
|---|---|---|---|
| `random_init` | nothing (frozen random initialisation) | no | MIT |
| `ecgfounder` | Harvard-Emory ECG Database, >10M recordings, no public corpus named | no | MIT |
| `ecgfm` | MIMIC-IV-ECG and PhysioNet/CinC 2021 | **yes** | MIT |
| `hubert_ecg` | CODE, CPSC and CPSC-Extra, PTB and PTB-XL, Georgia, Chapman-Shaoxing, Ningbo, Tianchi, Shandong, MIMIC-IV-ECG | **yes** | **CC BY-NC 4.0** |
| `ecg_jepa` | Chapman-Shaoxing with Ningbo, and CODE-15 | no | MIT |

**Two arms saw PTB-XL at pre-training.** ECG-FM's own README lists
PhysioNet/CinC 2021 among its pre-training corpora, and that bundle contains
PTB-XL; HuBERT-ECG's paper lists PTB-XL directly, and Shandong as well. Their
figures on PTB-XL, and HuBERT-ECG's on Shandong, are therefore partly
memorisation and not a measurement of transfer, and they are read as an upper
bound rather than as a result. ECGFounder and ECG-JEPA are the two arms whose
PTB-XL figure is external; ECG-JEPA in turn saw Chapman-Shaoxing and Ningbo, so
its figure there carries the same caveat.

What each arm was pre-trained on is written into every result file that uses
it (C-12), sourced to the authors' own description, so the caveat travels with
the number.

ECG-JEPA is wired, loads its published checkpoint whole and embeds a batch, and
is covered by `tests/test_encoder_arms.py`. Its representations are not yet
cached, so `results/arms.json` still compares four arms: the encoder runs at
296 ms per record on this machine's four threads, which is 5.6 hours for the
67,519 records the grid covers, and that job has not been run.

## Reproduce

Every number and every figure the report prints is redrawn from the scores
committed here, so none of the raw tracings are needed. Two of the three
scripts do need one file that is not committed: PTB-XL's `ptbxl_database.csv`,
6.6 MB from PhysioNet, which holds the patient each record belongs to and the
sex and age the subgroup table reports. Point `ECS_PTBXL_DIR` at the directory
holding it. The figures redraw without it.

```bash
uv sync                                  # 1 min
uv run pytest -m "not data"              # unit tests, no corpora needed, 2 min
uv run python scripts/figures.py         # redraws all six figures, 5 s
export ECS_PTBXL_DIR=/path/to/ptbxl      # the directory with ptbxl_database.csv
uv run python scripts/outcomes.py        # rebuilds results/outcomes.json, 3 s
uv run python scripts/subgroups.py       # rebuilds results/subgroups.json, 23 s
```

Timings are wall clock on a six-core i7-8700, CPU only. The full suite on a
cold clone, where every test module is imported for the first time, took 28
minutes.

The corpora themselves are only needed to re-score from the raw tracings,
which the committed `.npz` files make unnecessary for reproducing the report. The three
of them take 7.3 GB on disk, and the four auxiliary PhysioNet corpora the
script also fetches take roughly 21 GB more:

```bash
./scripts/fetch_open_corpora.sh          # ACS-ECG and four PhysioNet corpora
uv run pytest                            # adds the corpus-backed tests
```

`fetch_open_corpora.sh` does not fetch PTB-XL or SPH. PTB-XL is read in place
from `ECS_PTBXL_DIR` (default `~/Developer/ptbxl5d/data`) and is downloaded
from PhysioNet; SPH is downloaded from its figshare record and unpacked under
`data/sph`. Without them the corpus-backed tests skip rather than fail.

The rotation, in order; each step writes the file the next one reads.

```bash
scripts/fetch_mappings.sh                              # check the three code tables
uv run python scripts/label_table.py                   # results/label_map.json
uv run python scripts/duplicate_scan.py                # results/duplicate_groups.json
uv run python scripts/split_leak.py                    # results/split_leak.json
uv run python scripts/train_source.py --source ptbxl   # once per source
uv run python scripts/score_rotation.py                # every corpus by every model
uv run python scripts/rotation_table.py --draws 200    # results/rotation.json and .csv
uv run python scripts/rotation_uncertainty.py          # intervals, subgroups, comparator
uv run python scripts/figures.py                       # the figures, from those files
```

The scan reads every record of the five corpora and takes about twenty minutes;
everything after it reads the files the step before wrote. Training one source
held 612 MB on a six-core i7-8700, and the whole chain after the scan took an
hour.

## Gates

`ruff` (lint + format), `mypy --disallow-untyped-defs`, `pytest` — on every
commit via `pre-commit`, tests at pre-push.

## What this is not

A retrospective measurement study on public, de-identified data. No device
claim, no outcome claim, no prospective patient contact, no regulatory status,
and nothing here is intended to guide the care of any patient. The ethics
approvals under which each dataset was released, and the author's competing
interests, are stated at the top of [REPORT.md](REPORT.md).

## Licence

The code is MIT ([LICENSE](LICENSE)), with three exceptions.

`third_party/ecg_jepa/` is vendored from Sehun Kim's ECG-JEPA release under
MIT, with its own LICENSE beside it.

`third_party/ecgfounder/net1d.py` is vendored from PKUDigitalHealth's
ECGFounder release under MIT, and is itself derived from `hsd1503/resnet1d` by
Shenda Hong under the Apache License 2.0. Both notices and the chain between
them are in
[`third_party/ecgfounder/PROVENANCE.md`](third_party/ecgfounder/PROVENANCE.md).
The root MIT licence does not cover that directory.

The HuBERT-ECG weights are CC BY-NC 4.0. That restriction reaches further than
the arm itself: any number in this repository computed from those weights is a
derivative of them, so `results/embeddings/hubert_ecg/` and the HuBERT-ECG rows
of `results/arms.json` are limited to research use, whatever the root licence
says about the code that produced them.

The three code tables in `mappings/` are redistributed under their own
licences, with their digests and provenance in
[`mappings/NOTICE.md`](mappings/NOTICE.md): BSD 2-Clause for the Challenge's
scored-diagnosis table, MIT for Leinonen et al.'s AHA-to-SNOMED table, CC BY 4.0
for PTB-XL+.

No raw tracing is redistributed here, but the repository is not free of corpus
data either. `results/baseline/scores.npz` and `results/external/*.npz` carry
one row per record — the record identifier, its label and its model score, for
45,923 records across the three corpora — which is derived data under each
corpus's licence and joinable against the public patient tables. PTB-XL is
CC BY 4.0, which requires attribution: cite Wagner et al. 2020
(doi:10.1038/s41597-020-0495-6), the PhysioNet resource
(doi:10.13026/kfzx-aw45) and PhysioNet itself (Goldberger et al., Circulation
2000;101(23):e215–e220). The Shandong and Chongqing datasets are CC0. The
PhysioNet/CinC Challenge 2021 collection (CC BY 4.0), read only to count how
much infarction its non-PTB-XL partitions carry for
`results/seen_target.json`, is Reyna et al., Computing in Cardiology 2021.

The rotation adds derived data of the same kind over two more corpora:
`results/rotation/*/scores/*.npz` carry one row per record per model, 171,190
rows over 42,238 distinct records of the five rotation corpora, and
`results/duplicate_groups.json` names the records a corpus holds twice. Georgia,
Chapman-Shaoxing, Ningbo and CPSC reach this study through that Challenge
collection and are covered by its CC BY 4.0 attribution above; Shandong is
CC0.

## Sources

PTB-XL: Wagner et al., *Sci Data* 2020, 10.1038/s41597-020-0495-6 · SPH: Liu et
al., *Sci Data* 2022, 10.1038/s41597-022-01403-5 · ACS-ECG: *Sci Data* 2026,
10.1038/s41597-026-07278-0 · Split conformal: Angelopoulos & Bates,
arXiv:2107.07511 · APS: Romano, Sesia & Candès, NeurIPS 2020 · Covariate shift:
Tibshirani, Barber, Candès & Ramdas, NeurIPS 2019 · Label conditional
validity: Vovk, ACML 2012, PMLR 25:475-490 · Conformal prediction under label
shift: Podkopaev & Ramdas, UAI 2021, arXiv:2103.03323 · BBSE: Lipton, Wang &
Smola, ICML 2018.

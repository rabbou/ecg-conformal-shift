# ecg-conformal-shift

Split conformal prediction on 12-lead ECG, calibrated on PTB-XL and evaluated
without re-calibration on two external hospital corpora (SPH Shandong, ACS-ECG
Chongqing). The question measured is whether the 90% coverage guarantee
survives the change of hospital; the two standard corrections, Mondrian
per-class calibration and label-shift weighting, are compared on the same
break.

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

## Reproduce

Everything the report prints is redrawn from files committed here, so the
figures and the outcome table need no corpus at all:

```bash
uv sync                                  # 1 min, 86 packages
uv run pytest -m "not data"              # unit tests, no corpora needed, 4 min
uv run python scripts/outcomes.py        # rebuilds results/outcomes.json, 2 s
uv run python scripts/figures.py         # redraws all six figures, 4 s
```

Timings are wall clock on a six-core i7-8700, CPU only. The full suite on a
cold clone, where every test module is imported for the first time, took 28
minutes.

The corpora are only needed to re-score from the raw tracings, which the
committed `.npz` files make unnecessary for reproducing the report. They take
9.5 GB on disk:

```bash
./scripts/fetch_open_corpora.sh          # ACS-ECG and four PhysioNet corpora
uv run pytest                            # adds the corpus-backed tests
```

`fetch_open_corpora.sh` does not fetch PTB-XL or SPH. PTB-XL is read in place
from `ECS_PTBXL_DIR` (default `~/Developer/ptbxl5d/data`) and is downloaded
from PhysioNet; SPH is downloaded from its figshare record and unpacked under
`data/sph`. Without them the corpus-backed tests skip rather than fail.

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

## Sources

PTB-XL: Wagner et al., *Sci Data* 2020, 10.1038/s41597-020-0495-6 · SPH: Liu et
al., *Sci Data* 2022, 10.1038/s41597-022-01403-5 · ACS-ECG: *Sci Data* 2026,
10.1038/s41597-026-07278-0 · Split conformal: Angelopoulos & Bates,
arXiv:2107.07511 · APS: Romano, Sesia & Candès, NeurIPS 2020 · Covariate shift:
Tibshirani, Barber, Candès & Ramdas, NeurIPS 2019 · Label conditional
validity: Vovk, ACML 2012, PMLR 25:475-490 · Conformal prediction under label
shift: Podkopaev & Ramdas, UAI 2021, arXiv:2103.03323 · BBSE: Lipton, Wang &
Smola, ICML 2018.

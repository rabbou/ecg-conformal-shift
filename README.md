# ecg-conformal-shift

Split conformal prediction on 12-lead ECG, calibrated on PTB-XL and evaluated
without re-calibration on two external hospital corpora (SPH Shandong, ACS-ECG
Chongqing). The question measured is whether the 90% coverage guarantee
survives the change of hospital; the two standard corrections, Mondrian
per-class calibration and label-shift weighting, are compared on the same
break.

The findings, the four figures and the limitations are in
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
Shandong's infarctions are annotated as old, the same chronic-infarct target
that PTB-XL uses, so the labels are comparable.

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

Chongqing is the harder target: its label is angiographically confirmed acute
infarction, a different clinical event, so the label definition changes as
well as the prevalence.

## Layout

| Module | Role |
|---|---|
| `src/ecs/conformal.py` | split conformal (LAC + APS scores), Mondrian quantiles, covariate- and label-shift weighting, BBSE |
| `src/ecs/metrics.py` | coverage, Wilson intervals, class-conditional coverage, set size, abstention, effective sample size |
| `src/ecs/labels.py` | one comparable MI label across three annotation schemes (SCP-ECG, AHA, angiographic) |
| `src/ecs/config.py` | corpus paths and the label vocabulary |

## Reproduce

```bash
uv sync
uv run pytest -m "not data"   # unit tests, no corpora needed
uv run pytest                 # adds the reference-value tests against the corpora
```

PTB-XL is read in place from `ECS_PTBXL_DIR`; SPH and ACS-ECG are stored under
`data/`.

## Gates

`ruff` (lint + format), `mypy --disallow-untyped-defs`, `pytest` — on every
commit via `pre-commit`, tests at pre-push.

## Licence

MIT ([LICENSE](LICENSE)). The vendored ECGFounder architecture in
`third_party/ecgfounder/` is MIT, PKUDigitalHealth. The corpora keep their own
licences (CC BY 4.0 and CC0) and are fetched by script, not redistributed
here. The HuBERT-ECG weights are CC BY-NC 4.0, so that encoder arm is limited
to research use and is excluded from any commercial product.

## Sources

PTB-XL: Wagner et al., *Sci Data* 2020, 10.1038/s41597-020-0495-6 · SPH: Liu et
al., *Sci Data* 2022, 10.1038/s41597-022-01403-5 · ACS-ECG: *Sci Data* 2026,
10.1038/s41597-026-07278-0 · Split conformal: Angelopoulos & Bates,
arXiv:2107.07511 · APS: Romano, Sesia & Candès, NeurIPS 2020 · Covariate shift:
Tibshirani, Barber, Candès & Ramdas, NeurIPS 2019 · Label conditional
validity: Vovk, ACML 2012, PMLR 25:475-490 · Conformal prediction under label
shift: Podkopaev & Ramdas, UAI 2021, arXiv:2103.03323 · BBSE: Lipton, Wang &
Smola, ICML 2018.

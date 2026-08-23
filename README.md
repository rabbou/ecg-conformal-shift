# ecg-conformal-shift

**Does a conformal coverage guarantee survive a change of hospital?**

A 12-lead ECG classifier that is allowed to abstain, calibrated on one cohort and
measured on two others. Split conformal prediction promises that its prediction
sets contain the true label 90% of the time. That promise is a theorem, and the
theorem assumes calibration and test patients are exchangeable. Hospitals are
not exchangeable. This measures what the promise is actually worth when they are
not, and which of the two standard corrections repairs it.

## The cohorts

| Cohort | Country, years | Records | Patients | MI prevalence | Role |
|---|---|---|---|---|---|
| PTB-XL v1.0.3 | Germany, 1989–96 | 21,799 | 18,869 | 25.09% | calibration |
| SPH (Shandong) | China, 2019–20 | 25,770 | 24,666 | 1.01% | shifted test |
| ACS-ECG (Chongqing) | China, 2015–24 | 17,960 | 17,018 | 14.92% acute MI | shifted test |

All three are public and permissively licensed (CC BY 4.0, CC0, CC0). None of
the counts above is quoted from a paper: each is computed from the corpus's own
description file and pinned by a test in `tests/test_labels.py`.

## Corpora on disk

What each corpus is, as read off its own files on 2026-08-23.

| | PTB-XL | SPH (Shandong) | ACS-ECG (Chongqing) |
|---|---|---|---|
| Where | `~/Developer/ptbxl5d/data/records500/` | `data/sph/records/*.h5` | `data/acs/row_data/*.{dat,hea}` |
| Format | WFDB, format 16 (int16), `.hea` header | HDF5, one dataset `ecg`, float16, no attributes | WFDB, format 16 (int16), `.hea` header |
| Rate, length | 500 Hz, 5000 samples | 500 Hz, 5000–28,000 samples in steps of 500 (18,842 of 25,770 exactly 5000) | 500 Hz, 5000 samples |
| Amplitude | 1000 ADC units per mV, baseline 0 (header) | mV, stored as float16 (paper: 24-bit ADC, "16-bit precision") | 1000 ADC units per mV, baseline 0 (header) |
| Lead order | in each header (`AVR`/`AVL`/`AVF` upper-case) | not in the file; paper, Data Records: I, II, III, aVR, aVL, aVF, V1–V6 | in each header (`aVR` lower-case) |
| Filtering at source | Schiller device | MedEx MECG-200: mains, baseline wander and muscle noise removed by the machine, nothing added by the authors | Mecg-300 (Medex); the paper describes none |
| Records | 21,799 | 25,770 | 19,955 = 17,960 `train.csv` + 1,995 `test.csv` (no labels) |

Two things about the Chongqing files that are not in its paper: the header
columns WFDB reserves for a lead's initial value and checksum hold that lead's
maximum and minimum instead, so `wfdb` reads the samples correctly but the
checksum cannot validate them; and every sample is an even number of ADC units,
so the effective resolution is 2 µV against PTB-XL's 1 µV. The CSV column
`ecg_row_record` names the file (`04904.dat`); every name in the CSVs is on
disk and every file on disk is named, and no patient appears in both CSVs.

A full pass through the ingestion chain (`scripts/scan_corpora.py`, counts in
`results/ingest_report.json`) keeps every PTB-XL and Shandong record and drops
five of Chongqing's 19,955: two (`03228`, `14262`) whose `.dat` holds 3,500
samples under a header claiming 5,000, and three (`02008`, `03054`, `16558`)
carrying WFDB's missing-sample code, which reads back as NaN. 6,928 Shandong
records are longer than ten seconds and are cropped to the first ten.

Reading cost: `wfdb.rdrecord` spends ~23 ms per record parsing the header (wfdb
4.3.1 does it through pandas) against 0.8 ms reading the samples, so a full
pass over a WFDB corpus takes about eight minutes on the laptop; the HDF5 corpus
reads at ~7 ms per record.

The fetch is reproducible: `scripts/fetch_open_corpora.sh` carries the figshare
file ids and MD5 sums for the three Chongqing archives.

## What the prevalence gap means

MI is 25 times rarer in Shandong than in PTB-XL. That is not a labelling
artefact — dropping PTB-XL's five subendocardial-injury statements, the usual
suspect, moves prevalence only from 25.09% to 24.26%. PTB-XL is a research
corpus enriched for pathology; Shandong is an unselected hospital series. And
89.6% of Shandong's infarctions are annotated *old*, the same chronic-infarct
target PTB-XL carries, so the two are comparable and the gap is real.

The consequence is methodological. The dominant shift here is in **P(Y)**, not
**P(X)**. Covariate-shift weighting assumes the opposite and is the wrong
instrument. Two corrections are implemented instead:

- **Mondrian (class-conditional) conformal** — calibrate inside each class.
  Exactly valid under any change of class proportions, in finite samples, with
  nothing to estimate.
- **Label-shift weighting** — reweight by `w(y) = q(y)/p(y)` in the Tibshirani
  form, with the target prior estimated by BBSE. Asymptotic, and only as good as
  that estimate — so effective sample size is reported next to every result.

Chongqing is the harder shift: its label is angiographically confirmed *acute*
MI, a different clinical target, so label semantics move too.

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

PTB-XL is read in place from `ECS_PTBXL_DIR`; SPH and ACS-ECG land in `data/`.

## Gates

`ruff` (lint + format), `mypy --disallow-untyped-defs`, `pytest` — on every
commit via `pre-commit`, tests at pre-push.

## Sources

PTB-XL: Wagner et al., *Sci Data* 2020, 10.1038/s41597-020-0495-6 · SPH: Liu et
al., *Sci Data* 2022, 10.1038/s41597-022-01403-5 · ACS-ECG: *Sci Data* 2026,
10.1038/s41597-026-07278-0 · Split conformal: Angelopoulos & Bates,
arXiv:2107.07511 · APS: Romano, Sesia & Candès, NeurIPS 2020 · Covariate shift:
Tibshirani, Barber, Candès & Ramdas, NeurIPS 2019 · Label shift: Podkopaev &
Ramdas, UAI 2021, arXiv:2103.03323 · BBSE: Lipton, Wang & Smola, ICML 2018.

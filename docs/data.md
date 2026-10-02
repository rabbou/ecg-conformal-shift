# Corpus reference

Seven public corpora feed the study, counting Chapman-Shaoxing and Ningbo apart. Each count below is of the full release unless it says otherwise; [REPORT.md](../REPORT.md) section 2.1 gives the smaller counts that were scored.

## Corpora, licences and downloads

| Cohort | Country, years | Records | Patients | MI prevalence | Role |
|---|---|---|---|---|---|
| PTB-XL v1.0.3 | Germany, 1989–96 | 21,799 | 18,869 | 25.09% | calibration |
| SPH (Shandong) | China, 2019–20 | 25,770 | 24,666 | 1.01% | shifted test |
| ACS-ECG (Chongqing) | China, 2015–24 | 17,960 labelled | 17,018 | 14.92% acute MI | shifted test |

All three are public and permissively licensed: PTB-XL is CC BY 4.0, Shandong and Chongqing are CC0. The counts are computed from each corpus's description file, not quoted from a paper, and `tests/test_labels.py` pins them. Chongqing releases 19,955 tracings, of which the 17,960 in `train.csv` carry a label and the 1,995 in `test.csv` do not. Dropping PTB-XL's five subendocardial-injury statements from the infarction label moves its prevalence only from 25.09% to 24.26%, so the gap to Shandong is not an artefact of those statements.

The source rotation adds four corpora from the PhysioNet/CinC Challenge-2021 collection (CC BY 4.0): Chapman-Shaoxing and Ningbo, which rotate as one source, Georgia and CPSC. PTB-XL and Shandong take the source role in their turn.

| Corpus | Country | Records | Role in the rotation |
|---|---|---|---|
| PTB-XL | Germany | 21,799 | source and target |
| SPH (Shandong) | China | 25,770 | source and target |
| Chapman-Shaoxing and Ningbo | China | 45,152 | source and target |
| Georgia | United States | 10,344 | source and target |
| CPSC 2018 and its extension | China | 10,330 | source and target |

`scripts/fetch_open_corpora.sh` downloads ACS-ECG and the four PhysioNet corpora, and holds the figshare file ids and MD5 sums of the three Chongqing archives. Shandong, Chongqing and PTB-XL take 7.3 GB on disk, and the four PhysioNet corpora roughly 21 GB more. The script does not fetch PTB-XL or Shandong. PTB-XL is downloaded from PhysioNet and read in place from `$ECS_PTBXL_DIR`; Shandong is downloaded from its figshare record and unpacked under `data/sph`.

## File formats and ingestion

Contents recorded from the distributed files on 2026-08-23.

| | PTB-XL | SPH (Shandong) | ACS-ECG (Chongqing) |
|---|---|---|---|
| Where | `$ECS_PTBXL_DIR/records500/` | `data/sph/records/*.h5` | `data/acs/row_data/*.{dat,hea}` |
| Format | WFDB, format 16 (int16), `.hea` header | HDF5, one dataset `ecg`, float16, no attributes | WFDB, format 16 (int16), `.hea` header |
| Rate, length | 500 Hz, 5000 samples | 500 Hz, 5000–28,000 samples in steps of 500 (18,842 of 25,770 exactly 5000) | 500 Hz, 5000 samples |
| Amplitude | 1000 ADC units per mV, baseline 0 (header) | mV, stored as float16 (paper: 24-bit ADC, "16-bit precision") | 1000 ADC units per mV, baseline 0 (header) |
| Lead order | in each header (`AVR`/`AVL`/`AVF` upper-case) | not in the file; paper, Data Records: I, II, III, aVR, aVL, aVF, V1–V6 | in each header (`aVR` lower-case) |
| Filtering at source | Schiller device | MedEx MECG-200: mains, baseline wander and muscle noise removed by the machine, nothing added by the authors | Mecg-300 (Medex); the paper describes none |
| Records | 21,799 | 25,770 | 19,955 = 17,960 `train.csv` + 1,995 `test.csv` (no labels) |

Two properties of the Chongqing files are absent from the paper. The header columns WFDB reserves for a lead's initial value and checksum hold the lead's maximum and minimum, so `wfdb` reads the samples but the checksum cannot validate them. Every sample is an even number of ADC units, so the effective resolution is 2 µV against PTB-XL's 1 µV. The CSV column `ecg_row_record` identifies the file (`04904.dat`); the CSV entries and the files on disk match one-to-one, and no patient appears in both CSVs.

A full pass through the ingestion chain (`scripts/scan_corpora.py`, counts in `results/ingest_report.json`) keeps every PTB-XL and Shandong record and drops five of Chongqing's 19,955: two (`03228`, `14262`) whose `.dat` holds 3,500 samples under a header stating 5,000, and three (`02008`, `03054`, `16558`) with WFDB's missing-sample code, which reads back as NaN. 6,928 Shandong records are longer than ten seconds and are cropped to the first ten.

`wfdb.rdrecord` spends about 23 ms per record parsing the header (wfdb 4.3.1 does it through pandas) against 0.8 ms reading the samples, so a full pass over a WFDB corpus takes about eight minutes on a laptop. The HDF5 corpus reads at about 7 ms per record.

## Identical tracings and the splitting rule

Three of the rotation corpora file the same tracing more than once under different record identifiers, and the Challenge collection ships no patient key, so a split by patient alone would put copies of one tracing on both sides: 421 groups of identical tracings in CPSC, 56 in Georgia, 8 in Chapman-Shaoxing with Ningbo. Each group of identical tracings is therefore one splitting unit, and `results/split_leak.json` carries the count of groups split across two parts with and without that rule; with it, the count is zero. Shandong repeats tracings too but files them under one patient, so its split already held them. PTB-XL's distribution repeats none.

## Label mapping

The rotation runs on five diagnoses: sinus rhythm, atrial fibrillation, left bundle-branch block, right bundle-branch block and first-degree atrioventricular block. The mapping that puts three annotation schemes (SNOMED CT, AHA, SCP-ECG) on those five classes is in `results/label_map.json`. It reproduces the Challenge's own published per-partition counts, and it names the five joins that could not be made cleanly. Sinus rhythm is refused on Shandong, whose "Normal ECG" code is a narrower statement than the Challenge's "sinus rhythm".

## Encoder arms

Five ways of turning a tracing into a vector are compared on the same records. Four are published encoders. The fifth is this project's own ResNet1d frozen at its random initialisation, the floor the others have to clear.

| Arm | Pre-trained on | Saw PTB-XL | Licence |
|---|---|---|---|
| `random_init` | nothing (frozen random initialisation) | no | MIT |
| `ecgfounder` | Harvard-Emory ECG Database, >10M recordings, no public corpus named | no | MIT |
| `ecgfm` | MIMIC-IV-ECG and PhysioNet/CinC 2021 | **yes** | MIT |
| `hubert_ecg` | CODE, CPSC and CPSC-Extra, PTB and PTB-XL, Georgia, Chapman-Shaoxing, Ningbo, Tianchi, Shandong, MIMIC-IV-ECG | **yes** | **CC BY-NC 4.0** |
| `ecg_jepa` | Chapman-Shaoxing with Ningbo, and CODE-15 | no | MIT |

Two arms saw PTB-XL at pre-training. ECG-FM's own README lists PhysioNet/CinC 2021 among its pre-training corpora, and that collection contains PTB-XL. HuBERT-ECG's paper lists PTB-XL directly, and Shandong as well. Their figures on PTB-XL, and HuBERT-ECG's on Shandong, are therefore partly memorisation and not a measurement of transfer, and they are read as an upper bound. ECGFounder and ECG-JEPA are the two arms whose PTB-XL figure is external; ECG-JEPA in turn saw Chapman-Shaoxing and Ningbo, so its figure there carries the same caveat.

What each arm was pre-trained on is written into every result file that uses it, sourced to the authors' own description, both as a sentence and as a list of the corpora it saw. A script should read the list: ECG-JEPA's sentence reads "not PTB-XL, not Shandong, not Chongqing", and matching on the name of a corpus would count it as having seen all three. Extracting ECG-JEPA over the three infarction corpora took 6 h 53 on four threads.

## Re-scoring and the rotation chain

The corpora are needed only to re-score from the raw tracings, which the committed `.npz` files make unnecessary for reproducing the report.

```bash
./scripts/fetch_open_corpora.sh          # ACS-ECG and four PhysioNet corpora
uv run pytest                            # adds the corpus-backed tests
```

On a clone that holds only the committed files, none of the corpora and no optional `timm` extra, `uv run pytest -o addopts=""` ends with `540 passed, 59 skipped, 5 warnings` and exit code 0. The 59 skipped tests are the ones that read a corpus, the Challenge-2021 collection or `timm`. The project's `addopts` already carries `-q`, so a bare `uv run pytest -q` prints the dots and the warnings without that summary line.

The rotation runs in this order, each step writing the file the next one reads.

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

The scan reads every record of the five corpora and takes about twenty minutes; everything after it reads the files the step before wrote. Training one source held 612 MB on a six-core i7-8700, and the whole chain after the scan took an hour.

## Derived data and licences

The three code tables in `mappings/` are redistributed under their own licences, with their digests and provenance in [`mappings/NOTICE.md`](../mappings/NOTICE.md): BSD 2-Clause for the Challenge's scored-diagnosis table, MIT for Leinonen et al.'s AHA-to-SNOMED table, CC BY 4.0 for PTB-XL+.

No raw tracing is redistributed, but the repository holds corpus data in derived form. `results/baseline/scores.npz` and `results/external/*.npz` carry one row per record, holding the record identifier, its label and its model score, for 45,923 records across the three infarction corpora. That is derived data under each corpus's licence, and it can be joined against the public patient tables. PTB-XL's CC BY 4.0 attribution is in the [README](../README.md#licence-and-citation); Shandong and Chongqing are CC0. The PhysioNet/CinC Challenge 2021 collection (CC BY 4.0), read to count how much infarction its non-PTB-XL partitions carry for `results/seen_target.json`, is Reyna et al., Computing in Cardiology 2021.

The rotation adds derived data of the same kind over two more corpora: `results/rotation/*/scores/*.npz` carry one row per record per model, 171,190 rows over 42,238 distinct records of the five rotation corpora, and `results/duplicate_groups.json` names the records a corpus holds twice. Georgia, Chapman-Shaoxing, Ningbo and CPSC reach this study through the Challenge collection and are covered by its CC BY 4.0 attribution; Shandong is CC0.

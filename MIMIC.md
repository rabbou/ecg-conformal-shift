# Conformal coverage of five ECG diagnoses at Beth Israel Deaconess, on machine labels

Thresholds calibrated at one of five other hospitals and carried to Beth Israel Deaconess in Boston fall short of 90% coverage of the ill in 13 of 24 hospital and diagnosis pairs, with the whole 95% interval below 90%. The lowest is right bundle-branch block from PTB-XL, at 54.3%. The same holds the other way: thresholds calibrated at Beth Israel miss 90% in 8 of 24 pairs elsewhere. Run in shadow mode inside Beth Israel, from its earliest tracings to its latest, the model's own thresholds keep the coverage of the ill within a few points but send more of the patients without a diagnosis to a human reader: 22.9% of tracings for left bundle-branch block, against 10.3% within the early years. Every Beth Israel label here is the ECG cart's own statement, not a cardiologist's reading.

## A sixth corpus, of routine care in the United States

The rotation in the [README](README.md) measures coverage on five corpora, none of them from routine care in the United States, and the largest holds 45,152 records. MIMIC-IV-ECG v1.0 holds 800,035 ten-second ECGs from 161,352 patients seen at Beth Israel Deaconess between 2008 and 2019, under an open licence (ODbL). It carries no cardiologist label. What it carries is the interpretation the cart prints, up to eighteen statements per tracing, written by machines from three manufacturers.

So this extension asks two questions. Does the coverage measured on five corpora survive a sixth, an order of magnitude larger? And what is a coverage measured on machine labels worth to the hospital that buys the model? The second question needs the discharge diagnoses of MIMIC-IV-ECG-Ext-ICD and a human-read corpus, both under credentialed access, and is not answered here.

## From machine statements to five diagnoses

`src/ecs/mimic.py` maps every statement onto the rotation's five diagnoses (sinus rhythm, atrial fibrillation, left and right bundle-branch block, first-degree atrioventricular block) and onto old infarction, kept for the later comparison with discharge codes. Each rule gives a statement one status per class: definite, borderline, undated infarct, hedged ("possible", "probable", "cannot rule out") or excluded, for a neighbouring finding. The 800,035 tracings carry 3,127 distinct statements once case and punctuation are set aside, and 992 of them touch a class. `results/mimic_label_map.json` lists every one with its count and status.

| Diagnosis | Tracings | Share | Definite statements only |
|---|---|---|---|
| Sinus rhythm | 461,068 | 57.6% | 461,068 |
| Atrial fibrillation | 78,069 | 9.8% | 78,069 |
| Left bundle-branch block | 23,451 | 2.9% | 23,451 |
| Right bundle-branch block | 57,360 | 7.2% | 57,360 |
| First-degree AV block | 66,689 | 8.3% | 34,719 |
| Old infarction | 70,388 | 8.8% | 10,222 |

Eleven joins could go the other way, and the table gives each one's count so a reader can rebuild the class. Two of them double a class. The cart grades 31,970 tracings "borderline 1st degree A-V block". Its own PR measurement on those tracings has a median of 208 ms and is above 200 ms on 97.5% of them, which meets the definition of first-degree block, so they count as positive. The cart's commonest infarct statement is undated ("Inferior infarct - age undetermined", 60,430 tracings); with acute statements excluded, an undated infarct pattern is what old infarction means on a resting ECG, so it counts too. Atrial fibrillation stated as "probable" (3,026 tracings) and infarcts stated as "possible" (114,776) stay out. Sinus bradycardia, tachycardia and arrhythmia (186,513 tracings) are not sinus rhythm, because the Challenge codes they would be compared against keep them apart. A conduction defect the cart calls "of LBBB type" is not a block, although its QRS is 120 ms or more on 96% of those tracings: the cart declined to call it one.

The table was the test that could have stopped the work: if the statements had not mapped onto the five diagnoses, the rotation could not take MIMIC. They map, with 23,451 tracings in the smallest class.

## The rotation with six corpora

MIMIC enters the rotation as source and as target, cut by patient into the same four parts as the other five corpora, with the same caps (3,500 training, 2,000 calibration, 8,000 test). A MIMIC patient has five ECGs on average and up to 260, so one ECG per patient is drawn with a fixed seed. Age and sex are not in the open release. Ninety-four of the 10,000 calibration and test tracings hold missing samples and are dropped.

Each source's per-label thresholds are fitted once, on its whole calibration part, and spent unchanged on every test part. `results/mimic_rotation.csv` gives every pair, diagnosis and decision rule with 95% Wilson intervals; a Wilson interval treats the test tracings as independent and the threshold as fixed, so it is the precision of one deployed threshold on this test part.

| Calibrated at | Sinus rhythm | Atrial fibrillation | LBBB | RBBB | First-degree block |
|---|---|---|---|---|---|
| PTB-XL | 95.1% | 82.9% | 90.3% | 54.3% | 87.2% |
| Shandong | refused | 80.0% | 95.2% | 90.3% | 82.8% |
| Chapman-Shaoxing and Ningbo | 82.7% | 83.7% | 100% | 84.0% | 87.4% |
| Georgia | 71.4% | 97.6% | 99.3% | 89.8% | 75.7% |
| CPSC | 63.7% | 83.5% | 99.3% | 89.0% | 87.2% |
| Beth Israel (home) | 89.0% | 92.7% | 93.1% | 86.1% | 84.0% |

Coverage of the ill at Beth Israel, 90% asked for, per-label calibration. Sinus rhythm is refused on Shandong, whose "Normal ECG" code is narrower than the class.

Atrial fibrillation falls short from four of the five foreign sources, between 80.0% and 83.7%, and first-degree block from all five, between 75.7% and 87.4%. Left bundle-branch block holds from every source. Sinus rhythm swings from 95.1% to 63.7% with the source, which says more about how each corpus labels sinus rhythm than about Beth Israel.

Carried the other way, Beth Israel's thresholds cover 74.0% of sinus rhythm at PTB-XL, 70.3% and 70.5% of left bundle-branch block at Chapman-Shaoxing and Ningbo and at CPSC, and 68.6% of right bundle-branch block at CPSC.

At home, three of the five figures sit below 90% with their whole interval: sinus rhythm 89.0%, right bundle-branch block 86.1%, first-degree block 84.0%. Over 200 half-size draws of the calibration part, the home means are 89.1%, 88.2% and 86.8%, with standard deviations of 0.8, 5.3 and 4.9 points; Georgia's left bundle-branch block at home averages 86.8% in the same way. One calibration draw of about a hundred positives moves a home figure by several points, and the guarantee holds on average over draws, not for one draw.

Two encoders compared elsewhere in the study saw MIMIC-IV-ECG at pre-training, ECG-FM and HuBERT-ECG, and `ecs.encoders.SAW` marks both. Any of their figures on MIMIC is an upper bound, not a transfer. The rotation itself uses the study's ResNet trained from scratch, which saw nothing outside its source.

## The same recipe, trained twice

The published rotation's checkpoints were not kept, so the five original sources were retrained on another machine with the same seed, caps and schedule. The same estimator run on the published scores and on the retrained ones gives the size of that change on the 116 cells the two runs share: the coverage of the ill moves by 1.0 point at the median, 4.5 points at the 90th percentile, and 17.3 points at most, for the Shandong model's atrial fibrillation at PTB-XL, where its AUROC went from 0.846 to 0.948. The spread the published rotation reports is over calibration draws only; a second training of the same model can move an away figure by more than that spread.

## Shadow mode inside Beth Israel

A hospital that runs a model in shadow mode fits its thresholds on what it has and lets them run on what comes next. MIMIC shifts every patient's dates into the twenty-second century by an offset of its own, and the only key back to real years is the credentialed MIMIC-IV patients table. The open run therefore places tracings in time from open data. A patient's tracings keep their real intervals, so the differences between one patient's tracings on two carts order the 156 carts in time by least squares, and each tracing is then placed by its patient's carts. On the 92 patients of the open MIMIC-IV demo, whose real year group is published, the estimate's earliest third falls on a mean real year of 2012.5, the middle third on 2014.1 and the latest third on 2016.4 (Spearman 0.71 over 644 tracings; the reference itself is a three-year group).

The Beth Israel model's thresholds are fitted on 2,000 tracings of the earliest third and spent on 8,000 of the latest, beside the same thresholds on 4,000 other early patients and the rotation's patient split. No patient in these cohorts trained or stopped the model.

| | Patient split | Early to early | Early to late |
|---|---|---|---|
| Sinus rhythm | 89.0% | 90.5% | 88.7% |
| Atrial fibrillation | 92.7% | 94.4% | 93.1% |
| LBBB | 93.1% | 82.8% | 81.7% |
| RBBB | 86.1% | 91.6% | 96.2% |
| First-degree block | 84.0% | 92.5% | 85.9% |
| LBBB, tracings sent to a human | 7.6% | 10.3% | 22.9% |

Coverage of the ill, 90% asked for, per-label calibration, Beth Israel model.

Time moves first-degree block 6.6 points down and right bundle-branch block 4.6 points up, and for both the early and late intervals overlap. The visible change is in the patients without the diagnosis: for left bundle-branch block, the share of late tracings with no machine answer more than doubles. The carts change underneath: 87.4% of the early calibration tracings come from the cart family that filters at 0.005 to 150 Hz, and 54.4% of the late tracings from a family that filters at 0.0005 to 150 Hz and hardly appears early. This run cannot separate the passing of time from the change of machines.

The real-year run, 2008 to 2011 against 2016 to 2019, is one flag away (`scripts/mimic_shadow.py --eras anchor`) once the credentialed patients table is on disk.

## Reproduce

The label table needs `record_list.csv` and `machine_measurements.csv` from MIMIC-IV-ECG v1.0 (253 MB). The rotation and shadow scripts need the 27,216 records they read, which `scripts/mimic_extract.py` takes from the release zip and checks against the release's own digests, plus the five other corpora as in [docs/data.md](docs/data.md).

```bash
export ECS_MIMIC_ECG_DIR=~/data/mimic-iv-ecg
uv run python scripts/mimic_label_table.py                       # 2 min
for s in ptbxl sph chapman_ningbo georgia cpsc mimic; do
  uv run python scripts/train_source.py --source $s --device mps --rotation-dir results/rotation_six
done                                                              # 4 min each
uv run python scripts/score_rotation.py --sources ptbxl,sph,chapman_ningbo,georgia,cpsc,mimic \
  --corpora ptbxl,sph,chapman_ningbo,georgia,cpsc,mimic --device mps --rotation-dir results/rotation_six
uv run python scripts/mimic_rotation.py                           # 7 min
uv run python scripts/mimic_shadow.py --device mps                # 10 min
```

Timings are on an Apple M5 with the `mps` backend; torch 2.2's CPU kernels for one-dimensional convolution train about fifty times slower on that machine. `tests/test_mimic.py` and `tests/test_mimic_results.py` recompute every committed MIMIC figure from the committed scores.

MIMIC-IV-ECG is Gow et al., PhysioNet 2023, doi:10.13026/4nqg-sb35, under the Open Database License 1.0. The committed score files hold one row per MIMIC record (study identifier, five labels, five probabilities), derived data under the same licence.

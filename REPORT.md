# The report

**The question, in two sentences.** A heart-attack detector that is allowed to abstain promises that its output contains the correct diagnosis 90% of the time. This page reports whether that promise survives the move from the hospital whose data tuned it to two hospitals it has never seen, and which of the two standard corrections repairs what breaks.

Written for a reader with five minutes; every term is defined where it first appears. The technical front door, with file-level detail, is [README.md](README.md). The ten questions this page tends to raise are answered in [QUESTIONS.md](QUESTIONS.md).

## The object

A neural network reads a resting 12-lead electrocardiogram and scores it for **myocardial infarction** — the electrical signature of a heart attack, present or past. On top of that score sits a layer that is allowed to hedge. Instead of always answering "infarction" or "no infarction", it returns a **prediction set**: the list of labels it cannot rule out at the confidence asked for. Most tracings get a single label. A tracing can get both labels, meaning the model declines to choose, or no label at all, meaning the tracing resembles nothing it was calibrated on. In a clinic those two outcomes mean the same thing: the tracing goes to a human.

The dial that decides how much to hedge is set from data, never by hand. **Split conformal prediction** works as follows: put aside labelled tracings the model was not trained on (the **calibration set**) and pick the threshold so that the sets contain the true label at the rate asked for, say 90%. That rate, as observed on new patients, is called **coverage**. The method comes with a theorem: ask for 90%, get 90%. The theorem has one premise — the calibration patients and the future patients are statistically interchangeable. Two hospitals are not. This measurement is what the theorem is worth once its premise fails.

One honest word about the promise before it is tested: it is **marginal**, an average over patients and over draws of the calibration set. "90% coverage" means 90% of tracings on average, never "90% for this patient", and never by itself "90% within each class of patient". That distinction decides everything below.

## The measurement

One classifier was trained on PTB-XL, a German research corpus. One threshold was fitted on PTB-XL calibration patients and then spent, unchanged, on three test sets: held-out PTB-XL patients (home ground), all of Shandong, all of Chongqing. The two Chinese hospitals were scored once each; nothing was tuned on them and no threshold ever saw their labels. Every coverage figure is a mean over 200 draws of the calibration set, reported with its standard deviation (sd).

| Cohort | Country, years | Tracings scored | Infarction share | Role |
|---|---|---|---|---|
| PTB-XL | Germany, 1989–96 | 2,198 (fold 10) | 25.0% | calibration + home test |
| SPH (Shandong) | China, 2019–20 | 25,770 | 1.0%, mostly old infarcts | shifted test |
| ACS-ECG (Chongqing) | China, 2015–24 | 17,955 | 14.9%, acute, angiography-confirmed | shifted test |

The disease mix moves by a factor of 25 in one direction and the meaning of the label itself moves in the other: Chongqing's infarctions are acute events confirmed by angiography, where PTB-XL's are ECG diagnoses, largely older ones. Three versions of the threshold meet this: **no correction** (one threshold for everyone); **Mondrian** (one threshold per class, fitted inside each class on PTB-XL — exact mathematics, nothing estimated, built for a change in disease mix); **label-shift weighting** (calibration points reweighted toward the target hospital's class mix, which is not known and must be estimated from the model's own unlabelled predictions there).

## The four figures

![Figure 1](results/figures/fig1_coverage.png)

**Figure 1 — where the promise holds.** Share of tracings whose set contains the true label, per hospital (rows) and correction (columns); the grey bar is everyone, the red bar is the infarctions, the dashed line is the level asked for.

![Figure 2](results/figures/fig2_set_sizes.png)

**Figure 2 — what the clinician would see.** As the confidence asked for rises, the share of tracings answered with one label, with both, or with none; the last two are the abstentions that go to a human.

![Figure 3](results/figures/fig3_arms.png)

**Figure 3 — four encoders on the same break.** How far coverage falls away from home for each of four network backbones, two of which saw the calibration corpus during their pre-training; the arm that saw no public corpus is the strongest at home.

![Figure 4](results/figures/fig4_discrimination.png)

**Figure 4 — the reproduction that licenses the rest.** The supervised baseline reaches an AUROC of 0.932 [0.921, 0.943] on the benchmark split where the published figure is 0.930, so every measurement above sits on a model of known, ordinary quality. (AUROC: the probability that a random infarction tracing is ranked above a random non-infarction one; 0.5 is a coin flip.)

## The numbers to remember

**The promise was already hollow at home.** On PTB-XL itself, with 90% asked for, 90.0% of all sets contain the truth — and only 73.5% (sd 3.0) of infarction sets do. The average is carried by the healthy majority. A hospital switch does not create this gap; it only changes how well the majority hides it.

**Away from home the same threshold breaks in opposite directions.** In Shandong, where 99% of tracings are healthy, infarction coverage lands at 93.6% (sd 0.3) — above the level asked, for the unflattering reason that easy healthy tracings dominate. In Chongqing it falls to 72.5% (sd 0.8): nearly three infarctions in ten receive a set that does not contain the truth.

**The exact correction repairs what it can; the estimated one does not.** Calibrating per class (Mondrian) puts home infarction coverage at 90%, a paired gain of +16.7 points (sd 2.6), and lifts Chongqing by +11.5 points (sd 1.0) to 84.0%. Estimating the target's class mix instead moves almost nothing — +0.1 at home, +3.6 in Chongqing — and in Shandong it is a small loss, −2.3 points, because the estimate itself is poor: it reads Chongqing as 40.9% infarction against a truth of 14.9%. The exact repair has a price: wider sets (mean set size 1.05 → 1.11 at home) and a minority class calibrated on 275 of 1,099 points.

**Neither correction restores Chongqing to 90%.** Both assume only the disease mix moves between hospitals. In Chongqing the label means a different clinical event, so what an infarction looks like moves too, and no reweighting of German calibration data can repair that.

**The backbone changes how far the promise falls.** Across four frozen encoders, the coverage gap to Chongqing runs from +0.047 to +0.650 (Figure 3, 80% setting). The encoder that saw no public corpus in pre-training (ECGFounder) is the strongest at home (AUROC 0.919 [0.907, 0.932]); the two that saw PTB-XL do not beat it there, so pre-training contamination bought no visible home advantage in this grid.

## What this does not claim

- **Generality.** Two target hospitals, one disease, one source corpus. Nothing here says how a fourth hospital would behave, and the two measured here broke in opposite directions.
- **Causes.** The break is measured, never explained. Device, population, era and label semantics all differ at once; this design cannot attribute the break among them.
- **Bedside validity.** Coverage is an average over patients, never a per-patient statement. The Mondrian guarantee is conditional on the patient's true class — the thing nobody at the bedside knows. And no part of this pipeline is a medical device or was tested in care.
- **A fair trial for weighting.** The weighted correction received its class-mix estimate from this particular model's own predictions; a better estimator would give it a better run. What is claimed is what happened here, with the estimator named (BBSE), its effective sample size reported, and the estimate printed next to the truth.

## How it was built

The working habits this repository formalises, each held by a named test rather than by intention: a published value is reproduced first and gates everything downstream (Figure 4); no threshold is hard-coded — every operating point is calibrated from data; discrimination is reported without choosing a threshold, with bootstrap confidence intervals, and every arm-versus-arm claim is a paired comparison on the same tracings; coverage is reported per class, never only on average; every calibration/test boundary is drawn between patients, so no patient sits on both sides; and no target-hospital label ever reaches a threshold — a test fails if one does. The acceptance criteria in [PLAN.md](PLAN.md) map one-to-one to tests in `tests/`.

## Code, data, and how to check

Every number above lives in a JSON file under [`results/`](results/) — the break in `shift.json`, the encoder grid in `arms.json`, the abstention rates in `abstention.json`, the baseline in `baseline.json` — and every figure is redrawn from those files by `scripts/figures.py`. The corpora are public: [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) (PhysioNet, CC BY 4.0), [SPH](https://doi.org/10.1038/s41597-022-01403-5) (*Scientific Data*, CC0), [ACS-ECG](https://doi.org/10.6084/m9.figshare.29925314) (figshare, CC0). Method sources: split conformal, Angelopoulos & Bates, arXiv:2107.07511; Mondrian under label shift, Podkopaev & Ramdas, arXiv:2103.03323; weighted conformal, Tibshirani et al., NeurIPS 2019; the prior estimator, Lipton et al., ICML 2018 (BBSE). To reproduce: `uv sync`, then `uv run pytest` for the gates and `uv run python scripts/figures.py` for the figures.

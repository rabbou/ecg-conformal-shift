# The ten questions this repository triggers

Each question is one a real reader asks — a clinician, a technical peer, or a commercial decision-maker. The answers below are written, sourced, and short; every number traces to a file in [`results/`](results/). The page they follow from is [REPORT.md](REPORT.md).

## 1. Does "90%" apply to my patient? *(clinician)*

No. The guarantee is marginal: an average over patients and over draws of the calibration data. It is compatible with systematic failure inside a subgroup, and that is what the measurement shows at home — 90.0% of all sets contain the truth while only 73.5% (sd 3.0) of infarction sets do (`results/shift.json`). Per-class calibration narrows the promise to "90% within each class", which is better and still conditional on the patient's true class — the thing nobody at the bedside knows. A per-patient guarantee does not exist in this framework, and any tool that implies one is overclaiming.

## 2. What would I actually see when the model is unsure? *(clinician)*

A set with both labels, or with none. With the smallest-set score at 90% confidence, about 95 tracings in 100 get a single label at home; most of the rest get both labels, and the both-label share grows as the confidence asked for rises — 18% at 95% at home, 22% in Chongqing (Figure 2, `results/shift.json`). At lower confidence the model starts returning empty sets instead — a tracing that resembles nothing it was calibrated on. Operationally both mean the same thing: no machine answer, a human reads the ECG. The abstention rate, per corpus and per confidence level, is the committed table in `results/abstention.json`.

## 3. Why should I believe the underlying model is any good? *(clinician)*

Because its first job was to reproduce a published number, and it did: AUROC 0.932 [0.921, 0.943] on the PTB-XL benchmark split where the published figure for the same architecture is 0.930 (Figure 4, `results/baseline.json` — the published figure averages five diagnostic superclasses, and the file records why the infarction-only column is unavailable). This is a deliberately ordinary, well-documented baseline. The point of the study is what calibration promises survive a change of hospital, and that question is only worth asking on a model of known, believable quality.

## 4. Would this hold at my hospital? *(clinician, commercial)*

Unknown, and the repository says so. The two hospitals measured here broke in opposite directions — coverage rose in Shandong and fell in Chongqing — so no direction, let alone magnitude, can be extrapolated to a third site. What transfers is the harness: the measurement needs a few hundred labelled tracings from your site and answers the question directly, per class, with its spread. That is the honest offer this object makes: a way to measure the promise where you are, in place of an assurance that it holds.

## 5. Isn't the good Shandong number just prevalence? *(technical peer)*

Yes, and the report says exactly that. 99% of Shandong's tracings carry no infarction, healthy tracings are the easy class, and the marginal average is dominated by them; the uncorrected threshold over-covers (infarctions at 93.6%, sd 0.3, against 90% asked). That flattering number is the same mechanism that hides the 73.5% infarction coverage at home. It is why every coverage figure in the repository is reported per class (criterion C-11 in [PLAN.md](PLAN.md)) and why a single-number coverage claim on an imbalanced cohort should raise suspicion anywhere.

## 6. Why did the estimated correction fail where the exact one worked? *(technical peer)*

The weighted correction needs the target hospital's class mix, which is unknown and must be estimated — here by BBSE, from the model's own unlabelled predictions. The estimate was poor: 40.9% infarction estimated for Chongqing against 14.9% true, 2.4% for Shandong against 1.0% (`results/shift.json`, Figure 1 header). Reweighting also spends data — the effective calibration size drops from 1,099 points to 864 (Shandong) and 967 (Chongqing) (`results/shift.json`, criterion C-9) — and its guarantee is asymptotic. Mondrian estimates nothing: one threshold per class, exact in finite samples under any change of class mix. Paired on the same draws, Mondrian gains +16.7 points of infarction coverage at home where weighting gains +0.1, and in Shandong weighting is a 2.3-point loss.

## 7. Then why doesn't the exact correction fix Chongqing? *(technical peer)*

Because more than the class mix moved. Both corrections assume the appearance of each class is stable across hospitals and only the proportions change. Chongqing's infarction label names a different clinical event — acute, angiography-confirmed — where PTB-XL's is an ECG diagnosis, largely of older infarcts. When what an infarction looks like changes, no reweighting of source calibration data can follow it; Mondrian lifts Chongqing's infarction coverage from 72.5% to 84.0% (sd 1.0) and stops short of the 90% asked. The residual gap is the part of the shift that calibration cannot absorb, and the repository leaves it visible rather than tuning it away.

## 8. Two of the encoders saw the test data during pre-training. Did that flatter them? *(technical peer)*

Not where it was looked for. ECG-FM and HuBERT-ECG both had PTB-XL, the source corpus, in their pre-training; the arm with no public corpus at all (ECGFounder) still ranks best at home — AUROC 0.919 [0.907, 0.932] against 0.891 and 0.838 (Figure 3, `results/arms.json`, every comparison paired). HuBERT-ECG also saw Shandong, and on Shandong its behaviour sits between the two uncontaminated arms. Its distinctly small Chongqing gap (+0.047 against +0.650 for the random-init arm) reads as transfer from the broadest pre-training mix — nine corpora, several Chinese — since Chongqing itself is in nobody's pre-training list. The grid cannot prove absence of contamination effects; it shows that on this break, having seen the test distributions was neither necessary nor sufficient to look good.

## 9. Can this go into a product or a trial? *(commercial)*

No, and it does not ask to. It is a measurement study on public retrospective data: no device claim, no outcome claim, no prospective patient contact, no regulatory status. Two specific constraints for anyone thinking commercially: the HuBERT-ECG weights are CC BY-NC 4.0, so that arm is for research demonstration only and is excluded from anything shipped; everything else — the code (MIT), the corpora (CC BY 4.0 and CC0), the other weights — is permissively licensed. What the study contributes to a product conversation is the checklist a deployment would owe: per-class coverage at the deployment site, measured before use, with the correction chosen for the shift actually present.

## 10. Why conformal prediction rather than just calibrating the probabilities? *(technical interviewer)*

Temperature or Platt scaling adjusts the probability a model reports; it carries no finite-sample guarantee, and a calibrated probability still forces someone to pick the threshold that turns it into a decision. Split conformal gives a distribution-free, finite-sample guarantee on an object a workflow can act on — a set that either decides or abstains — with abstention arising from the same mechanism as the guarantee rather than from a hand-picked cut-off. The catch is that the guarantee is marginal and dies with exchangeability, which is precisely what this repository measures. The two families also compose: a better-calibrated score gives conformal smaller sets at the same coverage, so the choice is a layering, never either-or.

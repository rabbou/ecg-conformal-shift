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

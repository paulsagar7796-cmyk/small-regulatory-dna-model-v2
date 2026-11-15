# Model 2: locked test report

**Date of the test:** 16 November 2026 (commit `0c09f1c`). **Run once. Result: PASS on all four marks.**

Source of every test number below: `results/locked/RESULT.json`. Practice numbers come from `results/practice/`, `results/PRACTICE_RUN_1.md` and `runs/*/log.json`, and are labelled as practice. All numbers are rounded to 3 decimals.

## The question

Model 1 is a 1.32M-parameter network that predicts how strongly a 200-letter piece of DNA switches on a reporter gene. It was trained on the lentiMPRA tables only. MPRA (massively parallel reporter assay) is a lab method that measures the activity of many thousands of DNA pieces at once.

Model 2 is the same network trained on the lentiMPRA tables **plus** the natural-variant MPRA table of Siraj and others (Nature 2026). In that table each variant is a single-letter difference found between people, and both versions of the piece were measured. After training, model 2 was taught by three trained copies of itself (the same second step that model 1 had).

The question, fixed in advance in `predeclarations/PREDECLARATION_MORE_DATA.md`: **is model 2 a better judge of single-letter changes than model 1, on variants that no model and no choice of ours has seen?**

The measure is the Pearson correlation between the measured effect of a variant (version 2 minus version 1) and the predicted effect, in each cell type (HepG2, a liver cell line, and K562, a blood cell line). Only variants where at least one version is active in that cell type count. The two cell types are averaged.

## What was fixed before the test, in order

| When | What | Commit |
|---|---|---|
| 14 Nov 2026 | Split by chromosome: training = all others, practice = 6, 13, 20, locked = 5, 14, 22. Leak guards that remove the same DNA across parts. Locked measurements not written to disk | `0bf7b52` |
| 14 to 15 Nov 2026 | Practice runs: model 2 and a control without the new data; model 2 on three seeds | `7153f82`, `fab454d` |
| 15 Nov 2026 | Read-out chosen on practice data: model 2's new outputs (the HepG2 and K562 outputs trained on the new table). Self-teaching step added | `1cc26cc` |
| 15 Nov 2026 | All four pass marks confirmed by the maintainer, who raised the gain mark from the proposed 0.03 to 0.06 | `0f6435e`, `2f1e8c1` |
| 15 Nov 2026 | Student fixed (the taught model 2, seeds 20261113 and 2). Test script checked on the practice chromosomes, where it reproduces 0.584. Pre-declaration marked CONFIRMED | `d4ad416` |
| 16 Nov 2026 | Locked chromosomes written and scored, once | `0c09f1c` |

Fixed students:
- main: `more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed20261113`
- seed 2: `more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed2`

## The locked test

29,397 variants on chromosomes 5, 14 and 22, in 2,580 genome blocks of 100,000 letters. Scored: 5,800 variants in HepG2 and 6,081 in K562 (those with at least one active version). Intervals: 2,000 bootstrap resamples of genome blocks, the same resample for every model.

### Result table

| Model | Mean effect Pearson | HepG2 | K562 | Mean Spearman | 95% interval of the mean |
|---|---|---|---|---|---|
| **Model 2 (main seed)** | **0.594** | 0.632 | 0.555 | 0.512 | 0.566 to 0.619 |
| Model 2 (seed 2) | 0.592 | 0.630 | 0.555 | 0.517 | 0.565 to 0.618 |
| Model 1 (main seed) | 0.508 | 0.543 | 0.472 | 0.406 | 0.479 to 0.535 |
| Model 1 (seed 2) | 0.508 | 0.543 | 0.473 | 0.406 | 0.480 to 0.535 |
| Published MPRA-LegNet | 0.466 | 0.487 | 0.444 | 0.363 | 0.437 to 0.493 |
| 6-letter-word ridge baseline | 0.319 | 0.349 | 0.290 | 0.283 | 0.293 to 0.344 |

**Gain of model 2 over model 1: 0.086, 95% interval 0.074 to 0.099.** (Model 2 seed 2 over model 1: 0.072 to 0.098.) Model 2 over the published model: 0.114 to 0.143. Model 1 over the published model: 0.032 to 0.052.

### The four marks

| # | Mark | Value | Result |
|---|---|---|---|
| 1 | The 95% interval of the gain over model 1 is above zero | 0.074 to 0.099 | PASS |
| 2 | The gain is at least 0.06 | 0.086 | PASS |
| 3 | In neither cell type is model 2 more than 0.01 below model 1 | model 2 is above model 1 in both; the smaller margin is 0.083 (K562) | PASS |
| 4 | The two seeds' means differ by at most 0.02 | 0.001 | PASS |

## Reported beside the test (not pass or fail)

**Variants whose pieces share no 24-letter word with any training piece** (a stricter view of leaks through repeated DNA; 3,576 variants in HepG2, 3,587 in K562):

| Model | Mean effect Pearson |
|---|---|
| Model 2 (main / seed 2) | 0.565 / 0.562 |
| Model 1 (main / seed 2) | 0.485 / 0.488 |
| Published MPRA-LegNet | 0.443 |
| 6-letter-word baseline | 0.319 |

The order is the same as on all variants.

**Activity correlation** (how well each model predicts the activity of the pieces themselves, mean of HepG2 and K562): model 2 0.796 (seed 2: 0.798), model 1 0.619 (seed 2: 0.617), published 0.590, baseline 0.414.

**SK-N-SH** (a nerve cell line; only model 2 has an output for it): effect Pearson 0.568 (seed 2: 0.565), over 5,935 variants.

**Whole pieces on lentiMPRA fold 9 (practice fold, not a test result).** Fold 10 was read twice for model 1 and is closed, so model 2 can only be shown on fold 9. Mean Pearson over HepG2, K562 and WTC11, best epoch: model 2 0.844 (seed 2: 0.840), against model 1's 0.842. So the new data did not cost whole-piece accuracy on the old task.

## What helped (practice numbers only)

These numbers were used to make choices. They are not test results. "Natural variants" = practice chromosomes 6, 13, 20, same measure as the test. "Scan" = the saturation scan's practice half (every possible single-letter change in a set of pieces), mean of HepG2 and K562.

| Run | Seeds | Natural variants | Scan | Fold 9 whole pieces | Training time (min) |
|---|---|---|---|---|---|
| Model 1 (reference) | main / 2 | 0.491 / 0.484 | 0.627 / 0.623 | 0.842 | |
| Control: lentiMPRA only, same removals, no new data | 20261113 | 0.473 | 0.595 | 0.824 | 48 |
| Model 2, untaught, new outputs | 20261113 / 2 / 3 | 0.573 / 0.573 / 0.573 | 0.687 / 0.679 / 0.683 | 0.829 / 0.831 / 0.833 | 110 / 121 / 120 |
| Model 2, taught by three copies of itself, new outputs | 20261113 / 2 | 0.584 / 0.584 | 0.699 / 0.696 | 0.844 / 0.840 | 427 / 503 |

Reading:
- **The new data does the main work.** The control, without the new data, is slightly below model 1 (0.473 against 0.491). Adding the new data lifts the natural-variant score to 0.573.
- **The new outputs matter.** On the same untaught runs, the lentiMPRA outputs score 0.511 to 0.522 on natural variants; the new outputs score 0.573.
- **Self-teaching adds a little.** About 0.011 on natural variants and about 0.015 on the scan, and it brings whole-piece accuracy back level with model 1.

## Limits

- **Same-assay advantage.** Model 2 saw this lab method and this laboratory's measurements in training. Model 1 and the published model did not. Part of the gain may come from learning the quirks of this assay, not biology in general. The honest claim is **"a better small model with more data"**, not a same-data comparison.
- **Other models on this data family were not run.** Other groups have trained models on this family of data (Malinois, MPAC). We did not run them and make no claim against them.
- **One table, one split.** The result rests on one study's table and one choice of locked chromosomes.
- **Natural variants mostly have small effects.** Most measured variants change activity little, so the correlation leans on a minority of larger effects. The saturation scan is the complementary measure, and it was used only as practice here.
- **The locked chromosomes are now closed.** Chromosomes 5, 14 and 22 of this table will not be scored again.
- **Not peer reviewed.** This is a report from a small independent lab.
- **Not clinical evidence.** The model predicts a lab measurement in cell lines. It says nothing validated about disease risk in a person.

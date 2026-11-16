# Extra tests for model 2 (2026-11-17)

Rules recorded first: `predeclarations/EXTRA_TESTS_MODEL2.md` (commit 73a9d00). Model 2 was fixed on 2026-11-15.
Nothing was chosen or changed from these results. The original tests of record are unchanged.
Tests 1 to 3 re-read data already used, so they are **not blind**. Scripts: `src/extra_tests_model2.py`, `src/traitgym_zero_shot.py`.

## Checks (the scripts reproduce known numbers)
| Check | Got | Known |
|---|---|---|
| Fold 9: model 2 (lentiMPRA outputs) / model 1 / published | 0.844 / 0.842 / 0.816 | 0.844 / 0.842 / 0.816 |
| Scan practice half: model 2 (new outputs) / model 1 | 0.699 / 0.627 | 0.699 / 0.627 |
| Fold 10 (same run as test): model 1 / published | 0.842 / 0.818 | 0.842 / 0.818 |
| Scan locked half (same run): model 1 / published | 0.625 / 0.588 | 0.625 / 0.588 |
| Kircher (same run): model 1 / published | 0.639 / 0.594 | 0.639 / 0.594 |
| TraitGym CADD, pooled / chromosome-weighted | 0.694 / 0.712 | 0.694 / 0.712 |
| TraitGym centre letter = reference letter | 3,380 of 3,380 | |

## Overlap with model 2's new training table
Near-copy = 27 or more shared 24-letter words (either strand) with one training piece.
- Fold 10: 132 of 18,167 elements (HepG2 27, K562 93, WTC11 12).
- Scan locked half: 0 of 112 backgrounds (10 share at least one word).
- Kircher: F9, LDLR, PKLR none. **SORT1: 362 of 600 windows have a near-copy**; 4 training pieces from the SORT1 region share up to all 177 words.
- Found here: the training table's `pos` column is hg19, not hg38. So the Kircher check is by words, not coordinates.

## Test 1: lentiMPRA fold 10, whole pieces (third read; not blind)
| | HepG2 | K562 | WTC11 | Mean |
|---|---|---|---|---|
| Model 2 | 0.842 | 0.856 | 0.813 | **0.837** |
| Model 2 seed 2 | 0.842 | 0.857 | 0.814 | 0.838 |
| Model 1 | 0.841 | 0.868 | 0.816 | 0.842 |
| Published MPRA-LegNet | 0.818 | 0.857 | 0.779 | 0.818 |
| 6-mer baseline | 0.618 | 0.656 | 0.600 | 0.625 |

Model 2 minus model 1: CI -0.008 to -0.001 (model 2 slightly **worse**; K562 -0.016 to -0.009). Minus published: 0.013 to 0.025.
Without near-copy elements: model 2 0.837, model 1 0.841.

## Test 2: saturation scan, locked half (second read; not blind)
56 pieces; 89 HepG2 and 92 K562 backgrounds.
| | HepG2 | K562 | Mean | Spearman | minus model 1 (CI) | minus published (CI) |
|---|---|---|---|---|---|---|
| Model 2 (new outputs) | 0.693 | 0.742 | **0.718** | 0.603 | 0.071 to 0.116 | 0.109 to 0.156 |
| Model 2 seed 2 (new) | 0.694 | 0.741 | 0.717 | 0.601 | 0.071 to 0.115 | 0.108 to 0.154 |
| Model 2 lentiMPRA outputs (secondary) | 0.647 | 0.685 | 0.666 | 0.554 | 0.026 to 0.056 | 0.064 to 0.096 |
| Model 2 both averaged (secondary) | 0.693 | 0.738 | 0.715 | 0.601 | 0.070 to 0.111 | 0.108 to 0.152 |
| Model 1 | 0.604 | 0.646 | 0.625 | 0.515 | | 0.025 to 0.051 |
| Published | 0.552 | 0.623 | 0.588 | 0.472 | | |
| Baseline | 0.380 | 0.391 | 0.386 | 0.338 | | |
No locked background has a near-copy. Caveat: model 2 was trained on this lab method (same plasmid MPRA family); model 1 and published were not.

## Test 3: four Kircher 2019 elements (third use; not blind)
| | F9 | LDLR | SORT1 | PKLR-48h | Mean |
|---|---|---|---|---|---|
| Model 2 (new) | 0.724 | 0.564 | 0.565 | 0.770 | **0.656** |
| Model 2 seed 2 (new) | 0.716 | 0.530 | 0.571 | 0.776 | 0.648 |
| Model 2 lentiMPRA outputs (secondary) | 0.639 | 0.629 | 0.641 | 0.736 | 0.661 |
| Model 1 | 0.582 | 0.637 | 0.645 | 0.693 | 0.639 |
| Published | 0.488 | 0.634 | 0.550 | 0.706 | 0.594 |
| Baseline | 0.417 | 0.237 | 0.367 | 0.367 | 0.347 |
Mean, model 2 minus model 1: CI 0.000 to 0.033; minus published 0.042 to 0.082.
Per element vs model 1: F9 +0.104 to +0.182, PKLR +0.051 to +0.107, LDLR **-0.106 to -0.044**, SORT1 **-0.117 to -0.045**.
SORT1 by near-copy (extra check, not pre-declared): without near-copy (694 changes) model 2 0.557 vs model 1 0.700; with (1,016) 0.576 vs 0.554.

## Test 4: TraitGym Mendelian (zero-shot, first use)
3,380 variants, 338 positive, random about 0.10. Primary: mean |alt - ref| of the HepG2 and K562 outputs, variant at index 100 of the 200-letter window (1-based letters pos-100 .. pos+99).
| | Pooled AUPRC (95% CI) | Chromosome-weighted |
|---|---|---|
| Model 2 | **0.259** (0.218 to 0.308) | 0.299 |
| Model 2 seed 2 | 0.262 (0.222 to 0.311) | 0.301 |
| Model 1 | 0.267 (0.227 to 0.318) | 0.298 |
| Model 1 seed 2 | 0.268 (0.226 to 0.318) | 0.303 |
| CADD (check) | 0.694 (0.653 to 0.740) | 0.712 |
Model 2 minus model 1: CI -0.037 to 0.017. Secondary (pooled, model 2 / model 1): all outputs 0.249 / 0.262; max over outputs 0.277 / 0.296; five offsets 0.268 / 0.276.
For context only: PhyloGPN 0.296 and GPN-Star (M) 0.764 in earlier runs.

## Honest reading
- Model 2 is much better than model 1 on single-letter changes measured with its own lab method (scan locked half +0.093).
- On whole lentiMPRA pieces it is slightly worse than model 1 (0.837 vs 0.842), still ahead of published.
- On Kircher it is about level with model 1 on average, with large swings by element.
- On TraitGym both are far above random and far below CADD; model 2 adds nothing there.

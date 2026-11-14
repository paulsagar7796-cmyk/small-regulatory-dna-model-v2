# Model 2, first practice runs

**Date:** 14 November 2026.

**These are PRACTICE numbers from ONE seed. They are not test results.** The practice data is used to make choices. The locked chromosomes (5, 14, 22) have not been touched.

## The runs

Both runs: Kaggle T4 GPU, seed 20261113, 25 epochs, the recipe of model 1's teachers, no self-distillation.

| Run | Folder | Parameters | Training rows | Outputs | Time |
|---|---|---|---|---|---|
| Model 2 | `more_HEPG2-K562-SKNSH-HCT116-A549_e25_seed20261113` | 1,324,264 | 277,376 lentiMPRA + 417,566 new | 8 | 110 min |
| Control | `more_control_e25_seed20261113` | 1,322,979 | 277,376 lentiMPRA | 3 | 48 min |

The control uses the same recipe and the same leak-guard removals as model 2, but no new data.

Result files: `results/practice/<run folder>.json` for each run, and `results/practice/references.json` for model 1, the published model and the baseline.

## Measure A: saturation scan, practice half

Odd chromosomes and X. Pearson correlation between the measured and predicted effect of every single-letter change, mean of HepG2 and K562. The interval is the 95% interval of the difference from model 1 (bootstrap over pieces).

| Model | Mean | HepG2 | K562 | Difference from model 1 |
|---|---|---|---|---|
| Model 1 (released) | 0.627 | 0.629 | 0.626 | reference (seed 2: 0.623) |
| Published MPRA-LegNet | 0.579 | 0.567 | 0.591 | |
| 6-letter-word baseline | 0.382 | 0.383 | 0.381 | |
| Control (lentiMPRA outputs) | 0.595 | 0.596 | 0.594 | -0.051 to -0.016 |
| Model 2, lentiMPRA outputs | 0.634 | 0.633 | 0.634 | -0.010 to 0.021 |
| Model 2, new outputs | **0.687** | 0.680 | 0.694 | **0.043 to 0.075** |
| Model 2, both averaged | 0.686 | 0.683 | 0.688 | 0.043 to 0.072 |

## Measure B: natural variants, practice chromosomes 6, 13, 20

Pearson correlation between the measured and predicted effect, over variants where at least one version is active (5,400 variants in HepG2, 5,788 in K562), mean of the two cell types. The interval is the 95% interval of the effect difference from model 1 (bootstrap over 100,000-letter genome blocks).

| Model | Effect | Activity of the pieces | Effect difference from model 1 |
|---|---|---|---|
| Model 1 (released) | 0.491 | 0.610 | reference (seed 2: 0.484) |
| Published MPRA-LegNet | 0.455 | 0.584 | -0.046 to -0.025 |
| 6-letter-word baseline | 0.313 | 0.407 | |
| Control | 0.473 | 0.590 | -0.028 to -0.008 |
| Model 2, lentiMPRA outputs | 0.511 | 0.618 | 0.009 to 0.032 |
| Model 2, new outputs | **0.573** | 0.778 | **0.068 to 0.098** |
| Model 2, both averaged | 0.570 | 0.754 | 0.066 to 0.092 |

## Whole pieces: lentiMPRA fold 9 (practice fold)

Mean Pearson over HepG2, K562 and WTC11.

| Model | Mean | HepG2 | K562 | WTC11 |
|---|---|---|---|---|
| Model 1 (with self-distillation) | 0.842 | 0.833 | 0.866 | 0.827 |
| Model 1's teachers (same recipe, before the new leak-guard removals) | 0.833 | 0.822 | 0.858 | 0.819 |
| Control | 0.824 | 0.812 | 0.842 | 0.819 |
| Model 2 | 0.829 | 0.815 | 0.849 | 0.823 |

## Reading (the lab's interpretation)

1. **The new data clearly helps single-letter accuracy.** The new outputs gain +0.059 on the scan practice half and +0.082 on the natural variants, and both intervals are above zero. This is the largest gain of any change tried so far. Design changes and further self-teaching gave about zero.
2. **The scan result matters most.** The scan pieces were kept out of training, and the scan is a different kind of experiment (every letter changed) from the natural variants. But it comes from the same laboratory and assay as the new training data, so part of the gain may be assay-specific. Model 2's lentiMPRA outputs were not trained on that assay's labels. They also move in the right direction, but not clearly (0.634 against 0.627; the interval includes zero).
3. **The new outputs are the better read-out.** Averaging them with the lentiMPRA outputs adds nothing. The read-out is not fixed yet; it will be fixed after the second seed.
4. **Whole-piece accuracy on lentiMPRA** is below model 1 (0.829 against 0.842) but above the control (0.824). Model 2 has not had the self-distillation step that took model 1 from 0.833 to 0.842, so this is not like for like. The control is 0.009 below the earlier 0.833. The only difference is the 5,950 lentiMPRA training pieces removed by the new leak guard, plus run-to-run noise, which was not measured here.

## Limits

- One seed.
- Practice data, which is used to make choices.
- Same-assay advantage: the new training data and the scan come from the same laboratory and assay.
- The proposed pass mark (a gain of 0.03) looks reachable on this evidence, but that is a practice reading only.

## Next

1. A second seed of model 2 (running).
2. Self-distillation for model 2.
3. The maintainer is asked to confirm the pass marks.

The locked chromosomes (5, 14, 22) stay untouched until then.

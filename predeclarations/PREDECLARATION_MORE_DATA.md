# Model 2 pre-declaration: does more training data make a better judge of single-letter changes?

**Status: DRAFT** (written by Claude on 2026-11-14. The split and the leak guards below are fixed. The pass marks are proposals: the maintainer must confirm or change them before the locked chromosomes are scored. Nothing in this file is confirmed yet.)

## The question
Model 1 (the released student, repository `small-regulatory-dna-model`) was trained on the lentiMPRA tables only. Design changes and further self-teaching gave no more gain, and averaging our best models added almost nothing, so the remaining route is new training data. **Model 2** is the same network trained on the lentiMPRA tables plus the natural-variant MPRA table of Siraj and others (Nature 2026, Table S3). Is model 2 a better judge of single-letter changes than model 1, on variants that neither model, nor any choice we made, has seen?

## Data and split (fixed on 2026-11-14, before any model was trained or scored on this table)
- **Table:** 304,278 variants, each with both versions of a 200-letter piece, measured by plasmid MPRA in HepG2, K562, SK-N-SH and partly HCT116 and A549. Used: the 281,325 single-letter variants whose two versions are both 200 letters of A, C, G, T.
- **Split by chromosome:** locked = **5, 14, 22**; practice = **6, 13, 20**; training = all others. The three locked and three practice chromosomes were picked by hand to hold about 10% of the variants each; no measurement was looked at to pick them.
- **Leak guards** (full text in `src/prepare_more_data.py`): variants next to a saturation-scan piece are removed everywhere; training pieces that are the same DNA (about 50 letters or more in common with one piece) as a practice or locked piece, or as a lentiMPRA piece of fold 9 or 10, are removed; lentiMPRA training pieces that are the same DNA as a practice or locked variant piece are removed.
- **Sizes after the guards** (`results/split_counts.json`): training 208,783 variants, practice 28,104, locked 29,397. The guards removed 475 variants next to scan pieces, 11,509 + 6,041 training variants, and 5,950 of 276,476 lentiMPRA training pieces.
- **The locked table is not written to disk** until this file says CONFIRMED (`src/prepare_more_data.py --locked` refuses before that).

## History of this table (declared)
- 2026-11-14, before the split: Claude computed counts and measurement-quality statistics over **all** chromosomes (how many variants are measured and active per cell type, the size of the standard errors, an estimate of label reliability). No model prediction was compared with any measurement.
- The **sequences** of the locked chromosomes are read by the leak guards. Their measurements are not.
- A first version of the leak guard was too strict (it matched common repeat families) and was replaced before any model was trained.
- The saturation scan of the same study (its practice half) is model 1's and model 2's practice tool. Its locked half was used once for model 1 and is closed.

## The test (run once)
- **Measure:** for each cell type (HepG2, K562), the Pearson correlation between the measured effect (log2Skew, version 2 minus version 1) and the predicted effect, over the locked variants where at least one version is active (log2FC at least 1) in that cell type. The two cell types are averaged.
- **Recipe:** as in model 1's tests: each 200-letter piece between the training adapters, both orientations averaged, effect = version 2 minus version 1.
- **Interval:** 2,000 bootstrap resamples of 100,000-letter genome blocks (neighbouring variants share DNA, so variants are not resampled one by one), the same resample for every model.
- **Compared with:** model 1 (main seed), run on exactly the same variants. Also reported: published MPRA-LegNet, the 6-letter-word baseline, model 2's second seed.
- **Which read-out of model 2** (its lentiMPRA outputs, its new outputs, or both averaged) is chosen on practice data and written here before the test.

<!-- students: {"fixed": false} -->

## Pass marks (PROPOSED by Claude, NOT confirmed)
| # | Test | Pass if |
|---|---|---|
| 1 | Better than model 1? | Model 2's mean effect correlation is higher, and the 95% interval of the difference is above zero |
| 2 | Clearly better, not just measurably? | The difference is at least **0.03** |
| 3 | No cell type given up? | In neither cell type is model 2 more than **0.01** below model 1 |
| 4 | Stable? | The two seeds' means differ by at most **0.02** |

<!-- marks: {"min_gain": 0.03, "max_cell_deficit": 0.01, "max_seed_gap": 0.02} -->

**Are the marks reachable?** Not known yet. What is known from the practice chromosomes (6, 13, 20; `results/practice/references.json`): model 1 scores 0.491 on this measure (seed 2: 0.484), the published model 0.455, the 6-letter-word baseline 0.313. The table's own standard errors put the ceiling for a perfect model near 0.91 (an estimate). Model 2's practice numbers will be added here before the maintainer is asked to confirm.

## Reported beside the test, not pass or fail
- The same measure on the locked variants whose pieces share **no** 24-letter word with any training piece (a stricter view of leakage through repeated DNA).
- Activity correlation (both versions), SK-N-SH, and Spearman correlations.
- **Whole pieces on lentiMPRA:** fold 10 has been read twice and is closed, so model 2's whole-piece accuracy can only be shown on the practice fold (fold 9). It is not a test result.

## What a result would mean
- **Pass:** more lab data makes a better small judge of single-letter changes. The honest wording is "a better small model with more data". It is not a same-data comparison: model 2 has seen the test's assay and laboratory during training, and model 1 and the published model have not.
- **Fail:** more data of this kind does not help at this model size. It is reported as it is, and the locked chromosomes are not scored again.

## Known limitations (declared in advance)
- **Same assay advantage** (above). Models trained on this family of data by other groups (Malinois, MPAC) exist and were not run by us; this test makes no claim against them.
- The test variants are natural variants, most of which have small or no effect. The saturation scan's practice half is the complementary practice measure.
- One table, one split.

# D4c pre-declaration: single-letter changes on fresh data (the Siraj scan)

**Status: CONFIRMED** (marks confirmed by the maintainer on 2026-11-14, who raised mark 2 from the proposed 0.55 to 0.60 and confirmed the rest. The locked half had not been scored by anything. The test itself waits until the student is fixed below.)

## The data (real lab labels, new to every model here)
Siraj and others, "Functional dissection of complex trait variants at single-nucleotide resolution" (Nature 2026; preprint bioRxiv 10.1101/2024.05.05.592437, CC BY-NC). Supplementary Table 20: a "saturation mutagenesis" scan. For 114 trait-linked DNA pieces of 200 letters, on both versions (alleles) of each piece, every letter was changed to every other letter, and the effect on gene activity was measured in HepG2 and K562 cells. Tables downloaded 2026-11-14 from Europe PMC (PMC11100724) into `data/siraj/`, converted by `src/prepare_siraj.py`. The same measurements are on ENCODE (ENCSR694VUM, ENCSR909JVP).

- Used: the 228 single-variant backgrounds (114 pieces x 2 alleles), 600 single-letter changes each, in 2 cell types: 273,532 measured changes. The 56 two-variant ("haplotype") backgrounds are left out.
- Neither our student nor the published MPRA-LegNet model was trained on this data. Both were trained on the same other dataset (lentiMPRA), so the comparison is on equal terms.
- The lab method differs from the training data (a plasmid test with a minimal promoter, not a virus-based one), so some mismatch is expected for every model.

**What was looked at before recording this (measurements only, no model):**
- The change labels map exactly onto the piece sequences (100% of original letters match).
- **How reliable the labels are:** the same change measured on the two alleles of a piece agrees at a median Pearson of 0.74. It depends on how active the piece is: 0.82 for pieces with baseline activity (log2 fold change) of 2 or more, 0.62 between 1 and 2, 0.40 between 0.5 and 1, 0.21 below 0.5.
- **Overlap with training data:** 61 of the 114 pieces share at least one 31-letter word with a lentiMPRA training piece (32 of 58 in the practice half, 29 of 56 in the locked half). That shows a model the unchanged piece, not the effect of any change, and it holds equally for the published model.

## The split (fixed now)
- **Practice half:** pieces on odd-numbered chromosomes and X (58 pieces). Open for choosing designs and for proposing marks.
- **Locked half:** pieces on even-numbered chromosomes (56 pieces). Scored once, after the marks are confirmed and the student is fixed.

## How a model scores a change (fixed now)
1. The 200-letter piece goes between the two 15-letter adapters the students were trained with.
2. Predicted activity = average over the piece, its reverse complement, and the piece inserted the other way round.
3. Predicted effect = prediction for the changed piece minus prediction for the original piece. No sliding window is needed: the piece is exactly what the lab tested.
4. Measured effect = the table's `log2Skew` (changed minus original, log2 scale).

**Measure:** Pearson correlation between predicted and measured effects over the 600 changes of one background in one cell type. Only backgrounds whose own baseline activity in that cell type is at least 1 (log2 fold change) are scored, because below that the labels are mostly noise (see above). The score of a model is the mean over backgrounds within each cell type, then the mean of the two cell types. Intervals come from resampling pieces (both alleles and both cell types of a piece stay together).

**Comparison model:** the published MPRA-LegNet weights `best_model_test10_val9`, one per cell type, with the same recipe. Also reported: our v1 student, the 6-letter-word baseline, Spearman correlation, and results split by "shares DNA with training data" or not.

## Practice half, scored 2026-11-14 (`src/evaluate_scan.py --practice`, numbers in `results/scan/practice.json`)
58 pieces; 103 backgrounds scored in HepG2 and 81 in K562 (baseline activity at least 1).

| Model | Mean Pearson (95% interval) | HepG2 | K562 | Spearman | Pieces sharing DNA with training / not sharing |
|---|---|---|---|---|---|
| 6-letter-word baseline | 0.382 (0.343 to 0.416) | 0.383 | 0.381 | 0.323 | 0.366 / 0.400 |
| Our v1 student | 0.539 (0.487 to 0.590) | 0.516 | 0.562 | 0.398 | 0.509 / 0.569 |
| Published MPRA-LegNet | 0.579 (0.524 to 0.631) | 0.567 | 0.591 | 0.453 | 0.552 / 0.608 |
| **v2 student** | **0.627** (0.577 to 0.676) | 0.629 | 0.626 | 0.494 | 0.591 / 0.665 |
| v2 student, seed 2 | 0.623 | 0.627 | 0.620 | 0.495 | 0.592 / 0.655 |

v2 minus published: 95% interval 0.033 to 0.067. Sharing DNA with the training data does not raise the scores (they are lower there for every model). The labels' own agreement between alleles is about 0.74, so there is room above 0.63.

## Pass marks (confirmed by the maintainer on 2026-11-14, BEFORE the locked half was scored; mark 2 raised from 0.55 to 0.60)
| # | Test | Pass if |
|---|---|---|
| 1 | Better than the published model? | Mean Pearson on the locked half is higher than the published model's, and the 95% interval of the difference is above zero |
| 2 | Good in absolute terms? | Mean Pearson is at least **0.60** |
| 3 | No cell type given up? | In neither cell type is the student more than **0.01** below the published model |
| 4 | Stable? | The two seeds differ by at most **0.02** |

<!-- marks: {"min_mean_pearson": 0.60, "max_cell_deficit": 0.01, "max_seed_gap": 0.02} -->

**Are the marks reachable?** On the practice half the v2 student would pass all four (0.627; lead 0.033 to 0.067; ahead in both cell types; seeds 0.004 apart). The locked half has 56 different pieces, so the lead can shrink. Mark 2 (0.60, the maintainer's choice) sits about 0.03 under the practice score, so it can fail if the locked pieces are harder.

## Which student
Fixed from practice-half and fold-9 results only, and written here before the locked half is scored.
**Student fixed on 2026-11-14 07:35, before the locked half is scored:** the v2 student (published design + shared body + learning from three trained students, 1,322,979 parameters), seed 20261113 for the main numbers and seed 2 for the stability mark. Four design variants were screened first (`results/SCREEN_V3.md`); none beat it on fold 9 or on the practice half, so nothing changes.
<!-- students: {"fixed": true, "main": {"arch": "legnet", "width": 1.0, "run": "distill_legnet_t3_lam1.0_mut0.05_e25_seed20261113"}, "other_seed": {"arch": "legnet", "width": 1.0, "run": "distill_legnet_t3_lam1.0_mut0.05_e25_seed2"}} -->

## Known limitations (declared in advance)
- About half of the pieces overlap training pieces (see above).
- Choices made on the practice half can overfit it; the locked half is the check.
- One dataset from one lab, two cell types, 200-letter pieces only.

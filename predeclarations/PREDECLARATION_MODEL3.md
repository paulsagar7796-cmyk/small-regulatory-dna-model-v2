# Model 3 pre-declaration: a small student taught mainly by AlphaGenome

**Status: DRAFT (2026-11-18).** The split and the teachers below are fixed now, before any model 3 table is built or any teacher label is computed. The pass marks are proposals; the maintainer must confirm them before the locked chromosome is written to disk.

## The question
Can a small network (1.3M or about 4M parameters), taught mainly by the MPRA-fine-tuned AlphaGenome encoder plus MPAC, judge single-letter changes **clearly better than MPAC**, on DNA that no teacher and no student has seen?

## Permission (declared)
The maintainer reports written permission from Google DeepMind to train a student on AlphaGenome outputs. Publishing the weights or results is subject to DeepMind's review of the final report. The AlphaGenome-FT weights and any student trained on their outputs fall under the AlphaGenome Model Terms of Use (non-commercial). The release text will name the teachers.

## Split (fixed now)
- **Locked: chromosome 7.** It is in the test set of the AlphaGenome-FT Gosai checkpoints (test = chr7, chr13; validation = chr19, chr21, chrX; `alphagenome_ft_mpra/episomal_utils.py`, commit 7319c1c) and in MPAC's held-out pair (7, 16). Our models 1 and 2 trained on chromosome 7, so they are reported beside the test but are not fair comparators there. The locked table is not written to disk until this file says CONFIRMED.
- **Practice: chromosome 13** (clean for AlphaGenome-FT and MPAC's (10, 13) fold). All pilot choices are made here only.
- **Training: every other chromosome** of the natural-variant table (Siraj and others 2026, single-letter variants with two 200-letter versions), plus the lentiMPRA tables as in model 2. Chromosomes 5, 14, 22 (model 2's closed locked set) and 6, 20 (model 2's practice) are ordinary training data for model 3.
- **Leak guard:** the exact guard (`exact_best_match`, `src/prepare_gosai_extra.py`): training pieces that share 27 or more of their 177 24-letter words with any chromosome-7 or chromosome-13 piece are removed; variants next to a saturation-scan piece are removed everywhere.

## Teachers (fixed now)
- **AG:** AlphaGenome-FT `Gosai-HepG2-optimal`, `Gosai-K562-optimal` (and `Gosai-SKNSH-optimal` for SK-N-SH), stage 2, 600-letter input with the Gosai vector flanks, both strands averaged (format settled in the teacher check, commit cadb30a).
- **MPAC:** the 10 networks of MPAC's (7, 16) fold only, one centred window, both strands; they never saw chromosome 7.
- **Mix:** AG 3 : MPAC 1 on z-scored outputs, chosen on practice chromosome 13 (0.697 against 0.694 for AG alone and 0.676 for an equal three-way average with our model 2; `weights_chr13` check, 2026-11-18). Our own models are not teachers.
- Labels are computed once, on every training piece and on one fixed copy of each piece with 5% of letters changed (seed fixed in the labelling script).

## Pilots (practice chromosome 13 only; one seed each)
- **P0:** lab labels + teacher mix, plain average (model 2's recipe with precomputed teacher labels).
- **P1:** P0 + (1) per-piece blend of lab label and teacher by reliability (lab standard error against the teacher's error measured on training chromosomes), (3) mutated copies taught by the teacher mix only, (5) an extra loss on the measured effect (version 2 minus version 1).
- **P2:** P1 + (2) the teacher term dropped where it disagrees with a precise measurement by more than a fixed multiple of the error bar, (4) the teacher term down-weighted where AG and MPAC disagree.
- The best pilot on chromosome 13 (effect Pearson, mean of HepG2 and K562, active variants) is then trained at about 4M parameters; the better size gets a second seed. The student is fixed in this file before chromosome 7 is written.

## The test (run once, on chromosome 7)
Effect Pearson (log2Skew against predicted version 2 minus version 1), active variants (max allele log2FC at least 1), per cell (HepG2, K562), averaged; 2,000 resamples of 100,000-letter genome blocks.
Compared with: **MPAC** ((7, 16) fold, same recipe as our reference run), and reported beside: AlphaGenome-FT itself, models 1 and 2 (not fair: trained on chromosome 7), published MPRA-LegNet, the 6-letter-word baseline.
Also reported (fixed now): the student against its teacher mix; on the 20% of variants where the teacher mix is furthest from the measurement, the student's and the teacher's error; the correlation of student and teacher errors.

## Proposed pass marks (NOT confirmed)
| # | Test | Pass if |
|---|---|---|
| 1 | Better than MPAC? | 95% interval of (student minus MPAC) above zero |
| 2 | Clearly better? | difference at least **0.03** (to be checked for reachability on chromosome 13 before confirming) |
| 3 | No cell type given up? | in neither HepG2 nor K562 more than 0.01 below MPAC |
| 4 | Stable? | the two seeds' means differ by at most 0.02 |

## Known limitations (declared)
One chromosome, one table. Same-assay advantage: the teachers and the student are trained on the same lab method as the test labels, as MPAC is. AlphaGenome-FT's own training split was chosen by its authors; its validation chromosomes (19, 21, X) are training data here. Practice chromosome 13 has been used for many earlier choices.

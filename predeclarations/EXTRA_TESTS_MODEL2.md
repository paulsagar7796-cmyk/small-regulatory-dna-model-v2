# Extra tests for model 2 (recorded before any scoring)

Date: 2026-11-17. **Status: RECORDED before scoring.**

## Why
On 2026-11-17 the maintainer asked for model 2 to be scored on tests that were closed after model 1,
and for models 1 and 2 to be scored on the TraitGym disease-variant benchmark.

## Fixed facts
- Model 2 was fixed on 2026-11-15 (runs `more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed20261113`
  = main, and `..._seed2`). Nothing is trained, chosen or changed from these results.
- Model 1 = `distill_legnet_t3_lam1.0_mut0.05_e25_seed20261113` (main) and `..._seed2`.
- The original tests of record are unchanged. Results on re-read data are NOT blind tests.
- Each scoring below is run once. Scripts are checked on practice data first (lentiMPRA fold 9; the scan's practice half).
- Known leak caveat: model 2's training table was filtered with a guard later found to miss repeat-family
  near-copies. For tests 1 to 3 we report how many test pieces have a near-copy (27 or more shared 24-letter words,
  either strand, with ONE piece) among model 2's new-table training pieces, and the score without them where feasible.

## The four evaluations

| # | Data | Measure | Model 2 read-out (fixed here) | Status |
|---|---|---|---|---|
| 1 | lentiMPRA fold 10, whole pieces | element-level Pearson (both orientations averaged), mean of HepG2, K562, WTC11; as model 1's test | its lentiMPRA outputs (how fold 9 was always read) | THIRD read of fold 10 (first v0 student, second model 1); not blind |
| 2 | Saturation scan, locked half (even chromosomes, 56 pieces) | same recipe and measure as model 1's test (Pearson per background, mean over backgrounds per cell type, mean of HepG2 and K562) | its new outputs (HEPG2, K562); lentiMPRA outputs and both averaged as secondary | SECOND read; not blind |
| 3 | Four Kircher 2019 elements (F9, LDLR, SORT1 in HepG2; PKLR-48h in K562) | Pearson of single-letter effects, as model 1's Part B | its new outputs (HEPG2 for F9, LDLR, SORT1; K562 for PKLR-48h); lentiMPRA outputs as secondary | third use of this data; not blind |
| 4 | TraitGym Mendelian (338 positives, 3,042 matched controls) | pooled AUPRC with the D1 script's stratified bootstrap CI, and the chromosome-weighted AUPRC; random about 0.10 | see below | new, zero-shot use; expected to be low |

For test 1 we also re-score model 1 and published MPRA-LegNet with the same script, as a check that it
reproduces 0.842 and 0.818. Tests 1 to 3 report 95% intervals of model 2 minus model 1 and minus published
(same resamples for all models) where the existing code provides the bootstrap.

### Test 4 details (fixed here)
- PRIMARY: for each variant, the 200-letter hg38 window with the variant at index 100 (0-based, letters
  pos-101 .. pos+98 in 1-based coordinates), placed between the training adapters; reference and alternative
  predicted in both orientations, averaged; score = mean over the HepG2 and K562 outputs of |alt - ref|.
  Model 1: its lentiMPRA HepG2 and K562 outputs. Model 2: its new HEPG2 and K562 outputs.
  Higher score = more likely disease-causing. Metric and bootstrap exactly as `d1/evaluate.py`.
- SECONDARY (labelled as such): mean |change| over all outputs of the model; maximum |change| over outputs;
  the primary score averaged over 5 window offsets (variant at index 60, 80, 100, 120, 140).
- Check: CADD re-computed with the same code must give 0.694. Unit check: the window's centre letter must equal
  the variant's reference letter (if many mismatch, stop).

# Reference run: MPAC on model 2's test variants (rule recorded before any MPAC score exists)

**Recorded 2026-11-16, before MPAC was downloaded or run.** This is a reference run of an outside model. It is not a new test of model 2: model 2's locked result (`results/locked/RESULT.json`) is final and is not changed by anything here.

## Why MPAC and not single-model Malinois
MPAC (Butts and others, bioRxiv 2025, doi 10.1101/2025.04.16.648420) is 110 networks of the Malinois design (about 4.1 million parameters each), trained on the same family of experiments as our new training table, in 11 chromosome-pair folds. For any chromosome, 10 of its networks never trained on that chromosome. So it has a clean ensemble for each of our locked chromosomes (5, 14, 22) and practice chromosomes (6, 13, 20). The single released Malinois model trained on five of those six chromosomes (it held out only 7, 13, 19, 21, X), so a score from it on our variants would not be a fair one. (Facts from an automated lookup of the papers and model cards on 2026-11-16, not a full read.)

## What is run
- **Model:** for each variant, the 10 MPAC networks that did not train on its chromosome (averaged), from the safetensors files only. Outputs used: HepG2 and K562.
- **Input:** our two 200-letter versions of each variant, placed between MPAC's own flanks by MPAC's own prediction code, with its own strand handling. Effect = version 2 minus version 1.
- **Declared deviation:** the MPAC preprint scores a variant by averaging 18 shifted windows of genome context. Our table holds only the centred 200-letter piece (the piece the laboratory measured), so one centred window is used. If MPAC's precomputed predictions for these variants can be matched, they are reported beside as "MPAC, published scheme".
- **Measure, variants and resampling:** exactly those of the locked test (`src/evaluate_locked.py`): effect Pearson over variants with at least one active version, mean of HepG2 and K562, the same 2,000 block resamples. Model 2's numbers must reproduce `RESULT.json` exactly in the same run, or the run is void.
- **Order:** practice chromosomes first, as a check of the script. Sanity line: MPAC's activity correlation there must be at least 0.80 (its paper reports about 0.89 on held-out chromosomes); below that, the script is treated as faulty and the locked chromosomes are not scored until it is fixed. Then the locked chromosomes, once.

## How the result will be read (fixed now)
On the locked chromosomes, with the 95% interval of (model 2 minus MPAC):
- interval entirely below zero: **MPAC is ahead**;
- interval contains zero: **level**;
- interval entirely above zero: **model 2 is ahead**.
Whatever the outcome, the size difference is stated beside it: model 2 is one network of 1.32 million parameters; the MPAC ensemble for one chromosome is 10 networks, about 41 million parameters in total.

## Expectation (Claude's, recorded so it cannot be adjusted later)
MPAC is expected to be ahead: it is about 30 times larger as used, is a 10-network average, and trained on a larger table of the same experiments.

## Limits
- MPAC and model 2 both trained on other chromosomes of the same experiments, so neither has the same-assay handicap here.
- The weights come from a third-party Hugging Face mirror (`saarantras1/MPAC`, card licence CC BY 4.0); the primary deposit is on Zenodo (10.5281/zenodo.15178434). Ownership of the mirror was not verified.

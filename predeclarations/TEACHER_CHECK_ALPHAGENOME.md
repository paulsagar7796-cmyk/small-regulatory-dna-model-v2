# Teacher check: does the MPRA-fine-tuned AlphaGenome encoder qualify as an extra teacher?

**Status: RECORDED before any AlphaGenome prediction was made (2026-11-17).** Practice data only. Nothing locked is read.

## Why
Model 3's teacher would be an average of our models, MPAC and possibly AlphaGenome. A teacher only helps the student if it adds information the other teachers lack. This check asks exactly that, on variants that are clean for all three.

## Model checked
The Gosai-trained AlphaGenome encoder heads of Murphy, Duran and Koo (2026), Hugging Face `Al-Murphy/alphagenome-encoder-ft`, checkpoints `Gosai-HepG2-optimal` and `Gosai-K562-optimal`, stage 2 (fine-tuned). Their code (`alphagenome_ft_mpra/episomal_utils.py`, commit 7319c1c) fixes the split: test = chr7, chr13; validation = chr19, chr21, chrX; training = the rest. Weights are under the AlphaGenome Model Terms of Use (non-commercial). **This check is inference only; no model is trained on AlphaGenome outputs.** Training a student on them waits for the maintainer's written permission from Google DeepMind.

## Data
Natural variants on **practice chromosome 13 only** (7,685 variants; active in HepG2 1,366, in K562 1,504). Chromosome 13 is in AlphaGenome-FT's test set, in MPAC's clean fold, and in our practice set. Chromosomes 6 and 20 are excluded because AlphaGenome-FT trained on them.

## Input format (decided on activity, before any effect is computed)
The checkpoint's own `config.json` is read and reported. If it does not settle the input format, two formats are tried on the first 300 chromosome-13 pieces: (a) 600 letters, the 200-letter piece between the last 200 letters of MPAC's `MPRA_UPSTREAM` and the first 200 of `MPRA_DOWNSTREAM` (the Gosai vector flanks); (b) the 200-letter piece alone, as in the authors' `scripts/test_episomal_mpra.py`. The format with the higher **activity** Pearson (measured log2FC of version 1) is used. Both strands of the input are averaged. **Sanity line:** activity Pearson on all chromosome-13 pieces at least 0.80 in both cells, else the run is stopped and reported, not tuned further.

## Measure
Effect Pearson (log2Skew against predicted version 2 minus version 1) over active variants (max allele log2FC at least 1), per cell (HepG2, K562), averaged. 1,000 block-bootstrap resamples of 100,000-letter genome blocks, seed 20261113, the same resamples for every comparison.

Teacher groups, effects z-scored per cell before averaging:
- OURS = mean of the two taught model 2 students (seeds 20261113 and 2), new outputs.
- MPAC = mean of the 10 clean networks of chromosome 13's fold (one centred window, as in our reference run).
- AG = the AlphaGenome-FT checkpoint of that cell.

## Rule (fixed now)
AlphaGenome-FT **qualifies as a teacher** if both hold:
1. (OURS + MPAC + AG) / 3 minus (OURS + MPAC) / 2: the 95% interval is above zero.
2. That gain is at least **0.005**.

Also reported, not part of the rule: AG alone, AG minus MPAC (interval), OURS + MPAC + AG with other weights is NOT searched.

## Limits
One chromosome, about 1,400 active variants per cell, so intervals are wide. AlphaGenome-FT and MPAC were trained on the Gosai table, which is the same measurement as our test labels (same-assay advantage, as for model 2). Our models' practice chromosomes have been used for many earlier choices.

# Experimental Log

## 1. Experimental Setup

* **Datasets:**
  * Lentiviral MPRA (lentiMPRA) activity tables in three cell types, 230-letter rows (200-letter piece plus adapters): HepG2 61,463 elements, K562 98,332 elements, WTC11 23,123 elements; dataset folds 1 to 8 used for training, fold 9 for validation (checkpoint choice), fold 10 as a held-out test. Exact reverse-complement duplicate rows were dropped.
  * Plasmid MPRA natural-variant table: 304,278 variants, each with two 200-letter versions (reference and alternative allele) measured in HepG2, K562 and SK-N-SH (about 302,000 each), HCT116 (185,108) and A549 (125,796). Only single-letter variants with two 200-letter versions were used (281,325). Coordinates in the table are hg19.
  * Split of the natural-variant table, fixed before any training: locked test chromosomes 5, 14, 22 (29,397 variants), practice chromosomes 6, 13, 20 (28,104 variants), training on all other chromosomes (208,783 variants, two pieces each). Variants within 200 letters of a saturation-mutagenesis test piece were removed from training.
  * Saturation-mutagenesis scan from the same plasmid MPRA study: 114 single-variant pieces, each in both allele backgrounds, with all 600 single-letter changes measured in HepG2 and K562. Split before any scoring into a practice half (odd chromosomes and X, 58 pieces) and a locked half (even chromosomes, 56 pieces). Only backgrounds with baseline log2FC at least 1 were scored.
  * Saturation mutagenesis of four regulatory elements from an earlier study (F9, LDLR, SORT1, PKLR at 48 hours).
  * TraitGym Mendelian benchmark (338 disease variants, 3,042 matched controls), used zero-shot as an out-of-scope check.
* **Evaluation Metrics:**
  * Primary: effect Pearson, the Pearson correlation between predicted effect (predicted activity of the alternative version minus the reference version) and measured log2 allelic skew, on active variants (maximum of the two allele log2FC values at least 1), computed separately for HepG2 and K562 and averaged. 95% intervals from 2,000 bootstrap resamples of 100,000-letter genome blocks (2,580 blocks on the locked chromosomes). Locked test: 5,800 active variants scored in HepG2 and 6,081 in K562.
  * Secondary: effect Spearman; activity Pearson; element-level Pearson on lentiMPRA folds; per-background Pearson on saturation scans; AUPRC on TraitGym.
* **Baselines Compared:**
  * Model 1: same design trained on lentiMPRA only (with the shared body and self-distillation), our previous model.
  * Published MPRA-LegNet: the released single-cell-type networks (1.32 million parameters, trained on lentiMPRA).
  * 6-mer ridge regression baseline.
  * MPAC: the released ensemble of Malinois networks (4.1 million parameters each); for each test chromosome only the 10 networks of the fold that never saw it were averaged (about 41 million parameters). Run with one centred window (the 200-letter piece padded to 600 letters with the published vector flanks), both strands, instead of the 18 shifted windows used by its authors; its activity Pearson here was 0.825.
  * For reference on disease variants only: CADD and PhyloGPN on TraitGym.
* **Implementation Details:**
  * Architecture: MPRA-LegNet convolutional design, one shared body, 8 outputs (lentiMPRA HepG2, K562, WTC11; plasmid MPRA HepG2, K562, SK-N-SH, HCT116, A549); 1,324,264 parameters; 230-letter one-hot input.
  * Loss: masked MSE on standardised labels, averaged per output over measured rows, then over outputs.
  * Recipe: 25 epochs, batch 1024, one-cycle schedule with peak learning rate 0.01, AdamW weight decay 0.1, random shift up to 21 letters filled with construct letters, reverse-complement augmentation; checkpoint by mean Pearson on lentiMPRA fold 9.
  * Self-distillation: three untaught models trained with different seeds; the student trains on label MSE plus MSE (weight 1.0) to the teachers' average, also on copies of training pieces with 5% of letters randomly changed.
  * Prediction: both orientations of the insert and both strands averaged.
  * Two students with different seeds (20261113 and 2) were trained; the first was fixed as the reported model before the locked test. Training time per student on one NVIDIA T4 GPU: 427 and 503 minutes.
  * Leak guards: lentiMPRA pieces overlapping held-out variant pieces removed; near-copy guard on 24-letter words (a training piece counts as a near-copy if it shares at least 27 of its 177 24-letter words with any one held-out piece).
  * Pre-declaration: pass marks were written and confirmed before the locked chromosomes were written to disk: (1) 95% interval of model 2 minus model 1 above zero, (2) gain at least 0.06, (3) neither HepG2 nor K562 more than 0.01 below model 1, (4) seed means differ by at most 0.02. The locked test was run once.

## 2. Raw Numeric Data

### Locked test, chromosomes 5, 14, 22 (29,397 variants), effect Pearson on active variants

| Model | Parameters | HepG2 | K562 | Mean | 95% interval of mean | Spearman (mean) | Activity Pearson (mean) |
|---|---|---|---|---|---|---|---|
| Model 2 (seed 20261113) | 1.32M | 0.632 | 0.555 | 0.594 | 0.566 to 0.619 | 0.512 | 0.796 |
| Model 2 (seed 2) | 1.32M | 0.630 | 0.555 | 0.592 | 0.565 to 0.618 | 0.517 | 0.798 |
| MPAC (10 networks for each chromosome) | about 41M | 0.633 | 0.575 | 0.604 | 0.578 to 0.631 | 0.542 | 0.825 |
| Model 1 | 1.32M | 0.543 | 0.472 | 0.508 | 0.479 to 0.535 | 0.406 | 0.619 |
| Published MPRA-LegNet | 1.32M | 0.487 | 0.444 | 0.466 | 0.437 to 0.493 | 0.363 | 0.590 |
| 6-mer ridge baseline | - | 0.349 | 0.290 | 0.319 | 0.293 to 0.344 | 0.283 | 0.414 |

### Locked test, paired differences (95% block-bootstrap intervals)

| Comparison | Difference in mean effect Pearson | 95% interval |
|---|---|---|
| Model 2 minus model 1 | 0.086 | 0.074 to 0.099 |
| Model 2 minus published MPRA-LegNet | 0.128 | 0.114 to 0.143 |
| Model 2 minus 6-mer baseline | 0.274 | 0.253 to 0.297 |
| Model 2 minus MPAC | -0.010 | -0.019 to -0.001 |
| Model 2 seed 20261113 minus seed 2 | 0.001 | -0.002 to 0.005 |

All four pre-declared marks passed: gain over model 1 0.086 (mark 0.06), interval above zero, HepG2 +0.089 and K562 +0.083 over model 1, seed gap 0.001.

### Locked test, other subsets

| Subset | Model 2 | Model 1 | Published MPRA-LegNet | MPAC |
|---|---|---|---|---|
| Variants whose pieces share no 24-letter word with any training piece | 0.565 | 0.485 | 0.443 | 0.574 |
| SK-N-SH output, 5,935 active variants | 0.568 | - | - | - |

### Leak-guard correction (found after the locked test, disclosed)

The first near-copy guard looked at only the first 8 owners of each 24-letter word, so some repeat-family near-copies stayed in training (about 5.3% of training pieces had a near-copy among practice pieces and about 5.6% among locked pieces). 3,277 of 29,397 locked variants had a near-copy in training. Re-scoring without them (rule approved before the re-score, run once):

| Subset | Model 2 | Model 2 seed 2 | Model 1 | Published MPRA-LegNet | 6-mer baseline |
|---|---|---|---|---|---|
| All locked variants | 0.594 | 0.592 | 0.508 | 0.466 | 0.319 |
| Without near-copies (5,259 HepG2, 5,389 K562 active) | 0.590 | 0.588 | 0.509 | 0.468 | 0.319 |
| Near-copies only | 0.627 | - | 0.494 | 0.464 | 0.323 |

Model 2 minus model 1 without near-copies: 0.080 (95% interval 0.068 to 0.094); all four marks still pass. On the practice chromosomes the same correction moved model 2 from 0.584 to 0.577. MPAC was not re-scored on this subset.

### Additional tests (not blind: sets that had been read before; no choice was made from them)

| Test | Model 2 | Model 1 | Published MPRA-LegNet | 6-mer baseline |
|---|---|---|---|---|
| Saturation scan, locked half (56 pieces), per-background Pearson, mean of HepG2 and K562 | 0.718 (HepG2 0.693, K562 0.742; Spearman 0.603) | 0.625 | 0.588 | 0.386 |
| Four-element saturation mutagenesis, mean Pearson | 0.656 (F9 0.724, LDLR 0.564, SORT1 0.565, PKLR 0.770) | 0.639 (F9 0.582, LDLR 0.637, SORT1 0.645, PKLR 0.693) | 0.594 | - |
| LentiMPRA fold 10, element Pearson, mean of three cell types | 0.837 (HepG2 0.842, K562 0.856, WTC11 0.813) | 0.842 (HepG2 0.841, K562 0.868, WTC11 0.816) | 0.818 (HepG2 0.818, K562 0.857, WTC11 0.779) | - |

Model 2 minus model 1 on lentiMPRA fold 10: 95% interval -0.008 to -0.001. Model 2 minus model 1 on the saturation scan locked half: 95% interval 0.071 to 0.116. Model 2 minus model 1 on the four elements: 95% interval 0.000 to 0.033. The saturation-scan locked half had been read once before (for model 1); this was its second read.

### Out-of-scope check: TraitGym Mendelian disease variants, zero-shot (pooled AUPRC; random about 0.10)

| Model | Pooled AUPRC | 95% interval |
|---|---|---|
| CADD | 0.694 | - |
| PhyloGPN | 0.296 | 0.255 to 0.346 |
| Model 1 | 0.267 | 0.227 to 0.318 |
| Model 2 | 0.259 | 0.218 to 0.308 |

### Practice chromosomes 6, 13, 20: ablation of the training method (effect Pearson, mean of HepG2 and K562)

| Model | Practice chromosomes | Saturation scan practice half | LentiMPRA fold 9 |
|---|---|---|---|
| Model 1 (lentiMPRA only) | 0.491 | 0.627 | 0.842 |
| Published MPRA-LegNet | 0.455 | 0.579 | 0.816 (per-cell-type models) |
| 6-mer baseline | 0.313 | 0.382 | - |
| Model 2 without self-distillation (3 seeds) | 0.573 (all three seeds identical to 3 decimals) | 0.679 to 0.687 | 0.829 to 0.833 |
| Model 2 with self-distillation (2 seeds) | 0.584 / 0.584 | 0.699 / 0.696 | 0.844 / 0.840 |

### Practice chromosomes: changes that did not help (one change each, against model 2 without self-distillation: 0.573 / 0.679 to 0.687 / 0.829 to 0.833)

| Change | Practice chromosomes | Scan practice half | LentiMPRA fold 9 |
|---|---|---|---|
| Extra loss on the difference between the two versions of a variant (2 seeds) | 0.560 / 0.554 | 0.666 / 0.670 | 0.808 |
| Same paired batches with negligible loss weight (control) | 0.578 | 0.684 | - |
| About 142,000 extra pieces from the larger table of the same assay (off-centre windows, insertion/deletion pairs, multi-variant pieces) (2 seeds) | 0.575 / 0.575 | 0.688 / 0.697 | - |
| Wider body, 4.0M parameters (1 seed) | 0.582 | 0.694 | 0.838 |
| Wider body plus extra pieces (1 seed) | 0.583 | 0.704 | 0.840 |
| EvoAug-style sequence augmentation, 2 per piece, then 5 clean epochs (2 seeds) | 0.549 / 0.545 | 0.653 / 0.652 | 0.808 / 0.804 |
| EvoAug-style, 0 to 2 per piece | 0.558 | 0.666 | 0.821 |
| Malinois design (5.0M parameters) with our recipe (2 seeds) | 0.472 / 0.484 | 0.580 / 0.580 | 0.755 / 0.758 |
| Malinois design, peak learning rate 0.003 | 0.521 | 0.598 | 0.790 |
| Drop pieces with large measurement error (max standard error 1.0) | 0.575 | 0.683 | 0.831 |
| Duplicate active pieces (log2FC at least 0.5) | 0.572 | 0.679 | 0.832 |

Earlier design screen on the lentiMPRA-only model (lentiMPRA fold 9 mean / scan practice half), against 0.833 / 0.614: per-cell-type heads 0.835 / 0.609; no pooling after the first block 0.834 / 0.598; strand-symmetric network 0.821 / 0.569; weight averaging (EMA) 0.829 / -.

### Practice chromosomes: where the gap to MPAC comes from (no training)

| Predictor | Effect Pearson, practice chromosomes |
|---|---|
| Single MPAC networks (10 networks) | 0.578 to 0.588, mean 0.583 |
| Average of 2 / 3 / 5 / 10 MPAC networks | 0.601 / 0.604 / 0.608 / 0.609 |
| Model 2 (single network) | 0.584 |
| Average of 3 of our untaught models | 0.581 |
| Average of 6 of our 1.3M models | 0.585 |
| Average of 2 of our 4.0M models | 0.588 |
| Average of all 10 of our models | 0.589 |
| Model 2 plus MPAC average | 0.608 |
| Model 2 with test-time shift averaging | +0.002 over model 2 |

Mean pairwise agreement (Pearson) of predicted effects between ensemble members: MPAC networks 0.908; our models with different seeds 0.958.

### Data reliability (estimated from the table's standard errors)

| Quantity | Estimated reliability |
|---|---|
| Activity, HepG2 / K562 / SK-N-SH | 0.93 / 0.93 / 0.88 |
| Single-letter effect among active pieces | 0.83 to 0.87 |
| Agreement of the two allele backgrounds in the saturation scan | about 0.74 |

## 3. Qualitative Observations

* Adding the same-assay plasmid MPRA data to the shared body raised locked-test effect Pearson from 0.508 (model 1) to 0.594; this gain is in part a same-assay advantage, because model 1 and the published MPRA-LegNet never saw this assay.
* Self-distillation added about 0.011 on practice chromosomes (0.573 to 0.584), a small part of the total gain.
* Model 2 was slightly but significantly behind MPAC on the locked chromosomes (difference -0.010, interval -0.019 to -0.001), with the larger deficit in K562 (0.555 against 0.575) and in Spearman (0.512 against 0.542). MPAC was run with one window instead of 18, which may understate it.
* A single MPAC network was about as accurate as model 2 on practice chromosomes (0.583 against 0.584), so the size advantage of model 2 over a comparable single network is about 3 times (1.32M against 4.1M parameters); the ensemble used per chromosome is about 31 times larger. The ensemble's lead came from averaging members whose predictions agree less with each other (0.908) than our seeds do (0.958); averaging our own models reached only 0.589.
* Of eight training-method and design changes screened on practice data, only a wider body helped (+0.009 with 3 times the parameters); the paired-effect loss, extra same-assay pieces, EvoAug-style augmentation, the Malinois design with our recipe, noise filtering and oversampling did not help or hurt. The Malinois-design runs were still improving at the last epoch, so they were undertrained under our 25-epoch recipe.
* Model 2 kept lentiMPRA accuracy close to model 1 (fold 10: 0.837 against 0.842, slightly but significantly lower, mostly in K562) while adding five plasmid-MPRA outputs.
* The model is not a disease-variant scorer: on TraitGym Mendelian it scored 0.259 AUPRC against CADD's 0.694, about the level of PhyloGPN (0.296).
* A leak-guard flaw (near-copies from repeat families) was found after the locked test; re-scoring without near-copies changed the main result by 0.004 (0.594 to 0.590) and did not change any verdict.
* The model runs on a laptop CPU or a consumer GPU; inference speed was not measured.
* Training and evaluation data come from plasmid and lentiviral MPRAs in cancer and stem-cell lines; results may not transfer to endogenous genomic context.

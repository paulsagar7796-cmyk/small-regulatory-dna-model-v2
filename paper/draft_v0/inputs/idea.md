## Problem Statement

Massively parallel reporter assays (MPRAs) measure the regulatory activity of hundreds of thousands of short DNA pieces at once, and the allelic version of the assay measures how a single-letter change (a single-nucleotide variant) alters that activity ("allelic skew", log2 of the activity ratio between the two alleles). Sequence-to-activity models trained on these data are used to score variants that were never measured. The strongest published predictor of allelic skew on the plasmid-MPRA natural-variant data of Siraj and colleagues (Nature 2026) is MPAC, an ensemble of 110 Malinois networks (4.1 million parameters each; 10 networks, about 41 million parameters, are averaged for any one chromosome). Smaller single models exist (MPRA-LegNet, 1.3 million parameters, trained on lentiviral MPRA), but they were trained on a different assay, and it is not known how close a single small network can get to the ensemble, what actually closes the gap, and how such models behave under a strictly blind evaluation.

The technical problem: predict, from 200 DNA letters alone, (a) the activity of a regulatory piece in several cell types and two MPRA assay types, and (b) the change in activity caused by a single-letter change, with a single network of about 1.3 million parameters; and measure honestly how far such a model is from the ensemble state of the art.

## Core Hypothesis

1. A single small network can learn from two MPRA assay types at once (lentiviral MPRA in HepG2, K562, WTC11 and plasmid MPRA in HepG2, K562, SK-N-SH, HCT116, A549) through one shared body with one output per assay and cell type, and the added same-assay plasmid data is what most improves single-letter effect prediction.
2. Self-distillation (training a student on the lab labels plus the averaged predictions of three trained copies of itself, including on randomly mutated copies of the training pieces) gives a further, smaller gain.
3. The remaining gap to the ensemble is explained by ensembling diverse members, not by the design of a single network: one MPAC network is about as accurate as our single network.

## Proposed Methodology (Detailed Technical Approach)

- **Architecture:** the published MPRA-LegNet convolutional design (EfficientNet-style blocks), unchanged except for the output head: one shared body and 8 outputs [lentiMPRA HepG2, K562, WTC11; plasmid MPRA HepG2, K562, SK-N-SH, HCT116, A549]; 1,324,264 parameters. Input: the 200-letter piece between two 15-letter adapters (230 letters), one-hot.
- **Loss:** masked mean squared error on standardised labels; each output is averaged over the rows where it is measured, then over outputs (a row measured in one assay contributes only to that assay's outputs).
- **Training recipe (published MPRA-LegNet recipe):** 25 epochs, batch 1024, one-cycle learning rate with peak 0.01, AdamW with weight decay 0.1, random shifts of up to 21 letters filled with construct letters, reverse-complement augmentation; checkpoint chosen by mean Pearson on lentiMPRA validation fold 9.
- **Self-distillation:** three untaught models (seeds differ) are trained first; the student's loss is label MSE plus (weight 1.0) MSE to the average of the three teachers, on the training pieces and on copies of them with 5% of letters randomly changed.
- **Variant effect:** predicted activity of the alternative version minus that of the reference version, both orientations and both strands averaged.
- **Data split and leak guards:** natural-variant table split by chromosome before any training: locked test chromosomes 5, 14, 22 (never loaded until the pass marks were confirmed), practice chromosomes 6, 13, 20 (all choices made here), training on the rest. Variants near the separate saturation-mutagenesis test pieces were removed. LentiMPRA pieces sharing 24-letter words with held-out pieces were removed. A near-copy guard removes training pieces that share at least 27 of their 177 24-letter words with any one held-out piece.
- **Pre-declared evaluation:** effect Pearson between predicted and measured log2 skew on active variants (max allele log2FC at least 1), per cell type (HepG2, K562), averaged; 95% intervals from 2,000 resamples of 100,000-letter genome blocks. Pass marks written and confirmed before the one-shot locked test.
- **Comparators:** model 1 (same design, lentiMPRA only, self-distilled), the published MPRA-LegNet weights, a 6-letter-word ridge baseline, and MPAC (published weights for the fold that never saw each test chromosome; one centred 600-letter window with the published vector flanks, both strands).

## Expected Contribution

1. A compact (1.3M parameter), letters-only, laptop-runnable model that covers two MPRA assay types and 8 cell-type outputs and predicts single-letter effects within about 0.01 Pearson of a 41-million-parameter ensemble on blind chromosomes (slightly but significantly behind it), and far ahead of the published small model.
2. A strict blind-evaluation protocol for MPRA variant-effect models (pre-declared marks, one-shot locked chromosomes, block bootstrap, near-copy leak guards), including a disclosed leak-guard correction.
3. A measured decomposition of the gap to the ensemble (single ensemble members match our single model; the lead comes from averaging more diverse members) and a set of negative results: eight training and design changes that did not help on practice data.
4. Released weights and a prediction script.

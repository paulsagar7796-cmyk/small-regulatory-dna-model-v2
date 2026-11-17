# Cislume Dibang (model 2)

**A 1.3-million-parameter model that reads a 200-letter piece of human DNA and predicts how strongly it switches on a reporter gene in a reporter assay (MPRA), in several kinds of human cell. It also predicts the effect of changing any single letter.**

Cislume Dibang is one network of 1,324,264 parameters. It was trained on two public sets of lab measurements: the lentiMPRA tables of Agarwal and others (Nature 2025), and the natural-variant plasmid MPRA of Siraj and others (Nature 2026), in which both versions of about 280,000 single-letter variants were measured. It was then taught by three trained copies of itself (self-distillation). On a pre-declared, one-shot, blind test it predicts the effect of single-letter variants clearly better than its predecessor, Cislume Lohit (model 1), and the published MPRA-LegNet model.

From [PurvoLabs](https://github.com/PurvoLabs), an independent research lab in Assam, India.

> **Status (November 2026):** research release, not peer reviewed. The outputs are predictions of a lab reporter measurement in cell lines. They are not clinical evidence.

**The name.** *Cislume* = "cis" (cis-regulatory DNA: the switches that sit next to genes) + "lume" (light: a reporter assay reads out how strongly a piece of DNA switches on a reporter gene). Lohit and Dibang are two rivers of Assam that join the Siang near Sadiya to form the Brahmaputra. The Dibang carries more water, as model 2 learned from more data.

## Quick start

```bash
pip install -r requirements.txt   # versions used for the reported results (Python 3.13)

python predict.py --seq <200 letters>                        # activity of one piece
python predict.py --fasta pieces.fa --out activity.tsv        # activity of many pieces
python predict.py --fasta pieces.fa --scan --out effects.tsv  # effect of every single-letter change
```

Run from the repository root. Each piece must be exactly 200 letters of A, C, G, T.

| Output column | What it estimates |
|---|---|
| `plasmid_HepG2`, `plasmid_K562`, `plasmid_SKNSH`, `plasmid_HCT116`, `plasmid_A549` | log2FC in the plasmid MPRA of Siraj and others (Nature 2026). These outputs are behind the single-letter results below |
| `lenti_HepG2`, `lenti_K562`, `lenti_WTC11` | log2(RNA / DNA) in the lentiMPRA of Agarwal and others (Nature 2025), as for Cislume Lohit |

With `--scan`, the effect of a change is the activity of the changed piece minus the activity of the original. Each piece is placed between the two 15-letter adapters used in training and scored in both orientations, which are averaged: the recipe of every test in this repository.

The weights are in `weights/` (two students, plus `normalisation.json` for the output scales). By default `predict.py` uses the main student, the model behind the reported numbers; `--both` averages the two students. Scoring the practice chromosomes with the released main student reproduces its recorded practice score (0.5838) exactly.

## Result of record: the locked test

Run once on 16 November 2026, after the pass marks were confirmed by the maintainer and the model was fixed ([predeclarations/PREDECLARATION_MORE_DATA.md](predeclarations/PREDECLARATION_MORE_DATA.md)). Blind: no model and no choice of ours had seen these chromosomes.

**Data:** chromosomes 5, 14 and 22 of the natural-variant table, 29,397 variants. **Measure:** Pearson correlation between the measured effect of a variant (version 2 minus version 1) and the predicted effect, over variants with at least one active version (5,800 in HepG2, 6,081 in K562), mean of the two cell types. **Intervals:** 2,000 bootstrap resamples of 100,000-letter genome blocks.

| Model | Mean | HepG2 | K562 |
|---|---|---|---|
| **Cislume Dibang (model 2)** | **0.594** (seed 2: 0.592) | 0.632 | 0.555 |
| Cislume Lohit (model 1) | 0.508 | 0.543 | 0.472 |
| Published MPRA-LegNet | 0.466 | 0.487 | 0.444 |
| 6-letter-word ridge baseline | 0.319 | 0.349 | 0.290 |

Gain over Cislume Lohit: **0.086 (95% interval 0.074 to 0.099)**. All four pre-declared marks were met: the interval of the gain is above zero; the gain is at least 0.06; neither cell type is more than 0.01 below Cislume Lohit (both are well above it); the two seeds differ by 0.001 (at most 0.02 allowed). Also reported beside the test: SK-N-SH effect 0.568, and 0.565 against 0.485 on variants whose pieces share no 24-letter word with any training piece. Full report: [results/locked/REPORT.md](results/locked/REPORT.md). The locked chromosomes are now closed.

### Correction: near-copies in training

After the test we found a flaw in the leak guard: it checked only the first 8 pieces that contain each 24-letter word, so it missed some near-copies from repeated DNA. 3,277 of the 29,397 locked variants had a near-copy among the training pieces. The maintainer approved one re-score of the fixed models on the variants without a near-copy (5,259 HepG2 / 5,389 K562); no model or setting was chosen from it.

| Variants | Cislume Dibang | Seed 2 | Cislume Lohit | Published MPRA-LegNet | Baseline | Gain over Cislume Lohit (95% interval) |
|---|---|---|---|---|---|---|
| All (the test of record) | 0.594 | 0.592 | 0.508 | 0.466 | 0.319 | 0.086 (0.074 to 0.099) |
| Without a near-copy in training | 0.590 | 0.588 | 0.510 | 0.468 | 0.319 | 0.080 (0.068 to 0.094) |

All four marks still hold on the corrected subset. The original number stays the result of record. Numbers: [results/locked/NEARCOPY_RESCORE.json](results/locked/NEARCOPY_RESCORE.json).

## Comparison with MPAC

MPAC (Butts and others, bioRxiv 2025, doi [10.1101/2025.04.16.648420](https://doi.org/10.1101/2025.04.16.648420)) was trained on the same family of experiments. For any chromosome it uses the 10 Malinois-design networks that never trained on it, each about 4.1 million parameters, so about 41 million parameters in total. We ran it once on the locked chromosomes under a rule recorded first ([predeclarations/REFERENCE_MPAC.md](predeclarations/REFERENCE_MPAC.md)).

| Locked chromosomes 5, 14, 22 | Mean | HepG2 | K562 |
|---|---|---|---|
| MPAC (10 networks, about 41M parameters) | 0.604 | 0.633 | 0.575 |
| Cislume Dibang (1 network, 1.3M parameters) | 0.594 | 0.632 | 0.555 |

Cislume Dibang minus MPAC: 95% interval −0.019 to −0.001. **MPAC is narrowly ahead.** Declared deviation: we scored one centred 200-letter window per version, not the 18 shifted genome windows of the MPAC preprint, so our run may understate MPAC. MPAC was not re-scored on the near-copy-free subset. Numbers: [results/locked/MPAC_REFERENCE.json](results/locked/MPAC_REFERENCE.json).

**Practice data only, not a test** (practice chromosomes 6, 13, 20; [results/SCREEN_M3.md](results/SCREEN_M3.md), section "Practice checks without training"). Scored one at a time, a single MPAC network reaches 0.578 to 0.588 (mean 0.583), the same as Cislume Dibang's 0.584 at about a third of the size. MPAC's lead comes from averaging ten networks: its members' predicted effects agree less with each other (mean pairwise Pearson 0.908) than our seeds do (0.958), so averaging them gains more (average of 10: 0.609).

## Further one-time tests

Each was run once on 17 November 2026 under rules recorded first ([predeclarations/EXTRA_TESTS_MODEL2.md](predeclarations/EXTRA_TESTS_MODEL2.md)). Nothing was chosen or changed from them. **Only TraitGym is blind**; the other three re-read data used before. Report: [results/extra_tests/REPORT.md](results/extra_tests/REPORT.md).

| Test | Cislume Dibang | Cislume Lohit | Published MPRA-LegNet | Reading |
|---|---|---|---|---|
| lentiMPRA fold 10, whole pieces (lentiMPRA outputs; mean Pearson over HepG2, K562, WTC11) | 0.837 | 0.842 | 0.818 | Slightly below Cislume Lohit (interval of the difference −0.008 to −0.001), still ahead of the published model |
| Saturation scan of Siraj and others, locked half (every single-letter change in 56 pieces) | 0.718 | 0.625 | 0.588 | Gain over Cislume Lohit 0.071 to 0.116; same lab method as the training data |
| Four Kircher 2019 saturation-mutagenesis elements (F9, LDLR, SORT1, PKLR) | 0.656 | 0.639 | 0.594 | About level on average (interval 0.000 to 0.033); better on F9 and PKLR, worse on LDLR and SORT1 |
| TraitGym Mendelian, zero-shot (pooled AUPRC; random about 0.10) | 0.259 | 0.267 | | Far below CADD (0.694). Cislume Dibang is not a disease-variant scorer |

## How it was trained

- **Design:** the published MPRA-LegNet network with one shared body and eight outputs (three lentiMPRA cell types, five plasmid-MPRA cell types). Same design and size as Cislume Lohit, apart from the extra outputs.
- **Data:** the lentiMPRA training folds, plus both versions of 208,783 natural variants (417,566 pieces) from the training chromosomes of the Siraj table.
- **Split, fixed before any training on the new table:**

| Part | Chromosomes | Variants | Use |
|---|---|---|---|
| Training | all others | 208,783 | Training |
| Practice | 6, 13, 20 | 28,104 | Every choice |
| Locked | 5, 14, 22 | 29,397 | The one test above; now closed |

- **Leak guards** remove training pieces that are the same DNA as a practice or locked piece, and variants next to the saturation-scan pieces (`src/prepare_more_data.py`, counts in `results/split_counts.json`; see the correction above).
- **Self-distillation:** three copies trained from different seeds (20261113, 2, 3) teach one student of the same size, which learns from the lab labels plus the teachers' average, also on mutated copies of the pieces. The released students are seeds 20261113 (main) and 2.
- On practice data the new table did the main work: a control trained without it scored 0.473 on the practice chromosomes, the untaught model 0.573, the taught model 0.584 ([results/locked/REPORT.md](results/locked/REPORT.md), section "What helped").

## Limits

- **Same-assay advantage.** Cislume Dibang was trained on the same lab method and laboratory as the test labels. Cislume Lohit and MPRA-LegNet were not. Part of the gain may come from learning this assay's quirks. The honest claim is **"a better small model with more data"**, not "same data, better training".
- **One table, one split.** The result rests on one study's table and one choice of locked chromosomes.
- **Most natural variants have small effects**, so the correlation leans on a minority of larger effects.
- **Coordinates.** The `pos` column of the natural-variant table (and the scan identifiers) is hg19, not hg38. Matching against hg38 coordinates needs a liftover.
- **Research use only.** The model predicts a lab measurement in cell lines. It says nothing validated about disease risk in a person, and it is not for clinical use.

## Data and licences

No data is stored in this repository. Everything is public and is downloaded into `data/` (ignored by git). Full list: [DATA.md](DATA.md).

| Data | Source | Licence, as stated by the source |
|---|---|---|
| lentiMPRA activity tables (Agarwal and others, Nature 2025) | `datasets/original/` in [github.com/autosome-ru/human_legnet](https://github.com/autosome-ru/human_legnet) | MIT (repository) |
| Published MPRA-LegNet weights (comparison model) | [zenodo.org/records/10558183](https://zenodo.org/records/10558183) | CC BY 4.0 |
| Supplementary tables of Siraj and others (natural variants, saturation scan) | Europe PMC, article PMC11100724 | CC BY-NC (preprint) |
| Saturation mutagenesis of regulatory elements (Kircher and others 2019) | The study's Open Science Framework project | See the source |
| Human reference genome hg38 (only for the Kircher test) | UCSC | UCSC terms |

Check each source before reusing its data. `src/third_party/legnet_model.py` is copied from the MPRA-LegNet repository and stays under its MIT licence. `src/third_party/modeling_mpac.py` is third-party code (MIT, as stated in its header), used only for the MPAC reference run.

**Licence of this repository and the weights:** to be decided.

## Repository layout

| Path | What |
|---|---|
| `predict.py` | Predictions from the released weights |
| `weights/` | The two released students and the output scaling |
| `src/prepare_more_data.py` | The split and the leak guards |
| `src/train_more_data.py` | Training (and the control with `--new_cells none`) |
| `src/evaluate_practice.py`, `src/evaluate_locked.py` | Practice scoring and the locked test (run once) |
| `predeclarations/` | Rules recorded before each test |
| `results/` | Result files and reports |
| other files in `src/` | Cislume Lohit's code, which this model builds on |

## Related

- **Predecessor:** [Cislume Lohit (model 1)](https://github.com/paulsagar7796-cmyk/small-regulatory-dna-model), the same network trained on the lentiMPRA tables only ("same data, better training").

## Safety and maintainer

This model scores DNA. It does not generate DNA. **Maintainer:** [ORCID 0009-0007-7020-8949](https://orcid.org/0009-0007-7020-8949).

# Indel pre-declaration: can model 2 judge insertions and deletions it has never seen?

**Status: CONFIRMED** (drafted 2026-11-21; **confirmed by the maintainer on 2026-11-21, before any locked chromosome was read**. All four marks stand exactly as proposed, with no change to any number. The maintainer was told before confirming that mark 4's seed gap of 0.03 is looser than the 0.02 used in the earlier tests, because the practice gap of 0.021 would already fail 0.02, and accepted 0.03; that mark 1 is close to a coin flip on the practice interval; and that MPAC stays reported and is not a pass mark. The primary subset stands as drafted: indel size 1 to 21 letters, near-copies excluded. Nothing on the locked chromosomes had been scored when this line was changed.)

## Why this test
Published score tables already cover human single-letter variants (MPAC alone published about 575 million such predictions), so a lookup is cheaper than running any model on them. A small model earns its place only where a table cannot reach: DNA that nobody has scored. Insertions and deletions ("indels") are effectively unlimited in number, and the project's pitch (recorded 2026-11-09) names them explicitly. The council of 2026-11-09 left one change open (its change 4): how would the model score an insertion or deletion? This test is the answer, run once, on data the model has not seen.

## Terms, in plain language
- **Indel:** a change of the DNA that adds letters (insertion) or removes letters (deletion), instead of swapping one letter for another. After the change, the piece of DNA has a different length.
- **Allele:** one version of a DNA position. Each variant in the table has two: allele 1 (the reference) and allele 2 (the alternative), each a short DNA piece around the position. For a deletion, allele 1 is the 200-letter piece and allele 2 is shorter; for an insertion, allele 2 is the 200-letter piece and allele 1 is shorter (the shorter one is 200 minus the indel size).
- **Reporter assay (MPRA):** the lab cuts the DNA piece into a small genetic circuit and measures how strongly it switches a gene on in a cell line. `log2FC` is that activity on a log2 scale.
- **Allelic skew (`log2Skew`):** the measured activity of allele 2 minus allele 1, on the log2 scale. It is the lab's label for "how much did this change alter the activity". The model's job is to predict it: predicted effect = predicted activity of allele 2 minus predicted activity of allele 1. (ML analogy: the model is a regressor of activity, and the test checks whether differences of its outputs match measured differences.)

## The question
Model 2 (Cislume Dibang: model 1's network trained on the lentiMPRA tables plus the Siraj natural-variant table, then taught by three copies of itself) was trained on single-letter variants only. Every piece it saw had exactly 200 letters. Does its predicted effect still track the measured effect for natural insertions and deletions on chromosomes it never trained on, and is it better than the earlier model and the published model, which also never saw this lab's assay or any indel?

## Data and how it is built (`src/prepare_indels.py --locked`)
- **Source:** the natural-variant table of Siraj and others (Nature 2026, Table S3; `data/siraj/variants.parquet` and `oligos.parquet`), the same table as the single-letter test. Locked chromosomes: **5, 14, 22** (fixed on 2026-11-14 with the single-letter test; closed for every other purpose). Practice chromosomes 6, 13, 20.
- **Filters, in this order (all in the script):**
  1. reference and alternative differ in length (indels only);
  2. both oligos contain only A, C, G, T;
  3. one version is exactly 200 letters (a deletion's allele 1; an insertion's allele 2) and the other is exactly 200 minus the indel size;
  4. indel size at most 54 letters.
  On the practice chromosomes none of the 2,376 indels was dropped by filters 2 to 4 (`practice_counts.json`).
- **Scan guard:** variants within 200 letters of a saturation-scan variant on the same chromosome, or whose 200-letter version shares at least 27 of its 24-letter words with one scan piece, are dropped (the same guard 1 as the single-letter test). On the practice chromosomes it removed 5 variants.
- **Near-copy measure:** for each indel the script computes `best_shared_words`: the exact largest number of distinct 24-letter words (both strands counted as one word) that its 200-letter version shares with any ONE training piece of `data/more_data/train.parquet` (both versions of every training variant). At least **27** means near-copy ("about 50 letters in common"). This is the exact count (as in `src/rescore_nearcopy.py`), not the capped index that missed repeat-family copies in 2026-11-16.
- **Labels:** only the labels of the locked chromosomes are attached, after the filters. The script asserts that no other chromosome is present.
- **Expected size (an estimate, by analogy):** the table holds 2,443 indel ids on chromosomes 5, 14 and 22 (counted from the id column only; 2,376 on the practice chromosomes). The practice chromosomes kept 2,371 after the filters and guard, of which 2,141 are in the primary subset. If the locked set behaves alike, expect about 2,400 variants and about 2,150 in the primary subset, with about 370 to 410 scored per cell type. The real numbers will be known only when the script runs.

## How a model scores a shorter piece (fixed now, from `src/evaluate_indels.py`)
A piece of length 200 minus k (k from 0 to 54; the code asserts this):
- **Our LegNet-family models (model 2, model 1, the published MPRA-LegNet; 230-letter input).** `frame_legnet` builds, for orientation 0, LEFT adapter (15) + piece + RIGHT adapter (15) + the first k letters of the construct's own continuation (`RIGHT_FLANK`, from the MPRA-LegNet code). Orientation 1 is the same with the piece reverse-complemented between the same adapters. The input is always 15 + (200 - k) + 15 + k = 230 letters. The prediction is the average of the two orientations. The models were trained with a slide of up to 21 letters filled with this same continuation, so k up to 21 stays inside the shifts seen in training. **Not verified:** how the models behave for k above 21 (the secondary 22 to 54 subset); that is why the primary subset stops at 21.
- **MPAC (600-letter input).** `mpac_predict` builds forward = left + piece + right, and reverse = left + reverse_complement(piece) + right, where left = the last (200 + k // 2) letters of MPAC's upstream vector sequence and right = the first (200 + k - k // 2) letters of its downstream vector sequence. The input is always 600 letters (asserted). The prediction is the mean of the 10-network ensemble over the two strands. The ensemble for a variant is the 10 networks whose held-out chromosome pair contains the variant's chromosome (`load_ensemble`), so no MPAC network used in the test has trained on it. This is the same one-window deviation as the earlier MPAC reference run (the preprint averages 18 shifted windows; our table holds only the 200-letter piece).
- **Two asserts, run before every scoring (`framing_asserts`):** (1) `frame_legnet` must equal the existing single-letter framing (`evaluate_scan.pieces`) exactly, on 1,000 practice pieces of 200 letters; (2) `mpac_predict` must match MPAC's own `ensemble.predict` to within 1e-5 on 300 practice pieces of one chromosome. In the practice check both held (the MPAC difference was 0.0). The pipeline check (`--check-pipeline`) also reproduced earlier single-letter practice numbers: model 2 0.58382030 against 0.58382030 and MPAC 0.60889044 against 0.60889044 on 28,104 variants (`results/practice/indels_pipeline_check.json`).

## The measure (as in the single-letter test, `src/evaluate_locked.py`)
- For each cell type (HepG2, K562): the Pearson correlation between the measured `log2Skew` and the predicted effect (allele 2 minus allele 1), over the variants where both versions are measured, `log2Skew` is present, and at least one version is active (`log2FC` at least 1) in that cell type. The two cell types are then averaged. This is the mean effect Pearson.
- **Interval:** 2,000 resamples of 100,000-letter genome blocks (variants on the same block stay together, since neighbours share DNA). The same resamples are used for every model and every subset. Differences between models use the same resamples (`model2_minus_this_ci95`).
- **Model 2's read-out:** its new outputs (the HepG2 and K562 outputs trained on the natural-variant table), as fixed in the single-letter pre-declaration. Model 1 is the "v2 student" of the code (`student_v2`, main seed). The published model is the published MPRA-LegNet weights, one per cell type.

## Subsets
**PRIMARY (the only subset that decides the verdict):** indel size 1 to 21 letters, and no near-copy of training DNA (`best_shared_words` below 27). On the practice chromosomes: 2,141 variants (1,187 deletions, 954 insertions), 373 scored in HepG2 and 414 in K562.

**Secondary (reported, never used to decide the verdict):** size 1 to 21 including near-copies; all sizes up to 54 including near-copies; deletions only (primary filter); insertions only (primary filter); size 1 only; size 2 to 21; size 22 to 54 (39 practice variants, 16 and 15 scored: too few to read); near-copies only (size 1 to 21). Also SK-N-SH for model 2, and Spearman and activity correlations.

The subsets are defined in `subsets_for` of `src/evaluate_indels.py`, committed with the practice tables (f142770). The repository does not record whether the primary subset was fixed before the practice numbers were first seen; the practice file contains all nine subsets.

## Models, and which ones are excluded
**Scored:** model 2 (both seeds), model 1 (both seeds), published MPRA-LegNet, the 6-letter-word baseline (ridge on word counts), and MPAC (10 clean networks).

**Not scored, with reasons:**
- **Malinois (the single released model):** it trained on chromosomes 5 and 14, so it is not a fair comparator here.
- **AlphaGenome-FT and the model 3 pilots:** they trained on these chromosomes (model 3 held out only chromosome 7), and the permission to use AlphaGenome outputs is conditional and still pending. They stay out of this test.

## The student tested (the same two fixed on 2026-11-15; nothing new is chosen here)
The two released Cislume Dibang students, exactly as fixed in `PREDECLARATION_MORE_DATA.md` (commit d4ad416): main seed 20261113 and second seed 2, both read through the new outputs. No training, setting or read-out is changed for this test.

<!-- students: {"fixed": true, "main": "more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed20261113", "seed2": "more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed2"} -->

## History of this data (declared)
- **2026-11-21, practice chromosomes only:** the indel tables for chromosomes 6, 13, 20 were built, and every model was scored on them (`results/practice/indels_practice.json`). The numbers below come from there. The practice half has therefore been seen, and the marks are informed by it. They are written down here before the locked half is read.
- **Locked chromosomes before this file:** the single-letter variants of chromosomes 5, 14, 22 were scored once (the model 2 locked test, closed). That test used single-letter variants only. To the best of the record, no indel row of these chromosomes was scored by any model; this was read from the scripts and not otherwise verified. The 2,443 indel ids on these chromosomes were counted from the id column, with no measurement and no sequence read.
- **Seeds and models were not tuned on indels.** Nothing about model 2 changed after its single-letter pre-declaration.

## Practice results (primary subset, 2,141 variants; chromosomes 6, 13, 20; `results/practice/indels_practice.json`)
| Model | Mean effect Pearson (95% interval) | HepG2 | K562 | Model 2 minus this (95% interval) |
|---|---|---|---|---|
| 6-letter-word baseline | 0.161 (0.062 to 0.249) | 0.144 | 0.178 | 0.222 to 0.396 |
| Published MPRA-LegNet | 0.305 (0.209 to 0.395) | 0.302 | 0.307 | 0.078 to 0.255 |
| Model 1 (main seed) | 0.398 (0.305 to 0.484) | 0.430 | 0.367 | 0.006 to 0.141 |
| Model 1, seed 2 | 0.386 (0.288 to 0.479) | 0.389 | 0.383 | 0.020 to 0.153 |
| **Model 2 (main seed)** | **0.473** (0.344 to 0.584) | 0.499 | 0.446 | |
| Model 2, seed 2 | 0.493 (0.379 to 0.596) | 0.512 | 0.474 | -0.056 to 0.012 (against main seed) |
| MPAC (10 clean networks) | 0.514 (0.401 to 0.618) | 0.533 | 0.495 | -0.093 to 0.011 |

Activity correlation (the two versions' own activity): model 2 0.750, MPAC 0.781, model 1 0.572, published 0.542.

Secondary subsets, practice (model 2 main seed / model 1 main seed / MPAC), mean effect Pearson:

| Subset | Variants | Model 2 | Model 1 | MPAC | Model 2 minus model 1 (95% interval) |
|---|---|---|---|---|---|
| size 1 to 21, near-copies included | 2,332 | 0.506 | 0.446 | 0.542 | 0.003 to 0.125 |
| all sizes, near-copies included | 2,371 | 0.533 | 0.473 | 0.550 | 0.007 to 0.123 |
| deletions (primary filter) | 1,187 | 0.547 | 0.441 | 0.561 | 0.033 to 0.166 |
| insertions (primary filter) | 954 | 0.371 | 0.345 | 0.446 | -0.094 to 0.156 |
| size 1 only | 1,104 | 0.397 | 0.360 | 0.424 | -0.071 to 0.132 |
| size 2 to 21 | 1,037 | 0.521 | 0.428 | 0.572 | 0.005 to 0.180 |
| size 22 to 54 | 39 | 0.716 | 0.698 | 0.569 | -0.224 to 0.316 |
| near-copies only | 191 | 0.722 | 0.723 | 0.735 | -0.137 to 0.141 |

For context only: on the single-letter practice chromosomes the same models scored 0.584 (model 2) against 0.491 (model 1), a gain of 0.093; on the indel primary subset the gain is 0.074. These are different variants, so the comparison is loose.

## Pass marks (PROPOSED by Claude on 2026-11-21 from the practice numbers; the maintainer must confirm or change each)
The test is a pass only if all four hold on the PRIMARY subset.

| # | Test | Pass if (proposed) | Practice number it comes from |
|---|---|---|---|
| 1 | Better than model 1? | Model 2's mean effect Pearson (main seed) is higher than model 1's (main seed), and the 95% interval of the difference is above zero | Practice difference 0.074, interval 0.006 to 0.141: the lower end is only just above zero |
| 2 | Good in absolute terms? | Model 2's mean effect Pearson (main seed) is at least **0.42** | Model 2 scored 0.473 (seed 2: 0.493); 0.42 lies above model 1's 0.398, so a model no better than model 1 cannot pass it, and about 0.05 below model 2's own score, which the interval (0.344 to 0.584) can reach |
| 3 | Better than the published model and the baseline? | The 95% interval of (model 2 minus published MPRA-LegNet) is above zero, and so is the interval of (model 2 minus the 6-letter-word baseline) | Practice intervals 0.078 to 0.255 and 0.222 to 0.396: both clear zero, the published one by less |
| 4 | Stable? | The two seeds' mean effect Pearson differ by at most **0.03** | Practice gap 0.021 (0.473 against 0.493); above the 0.02 used in the earlier tests, and seeds disagree more on a set this small |

<!-- marks: {"model1_gain_ci_above_zero": true, "min_mean_pearson": 0.42, "published_and_baseline_ci_above_zero": true, "max_seed_gap": 0.03} -->

**MPAC is reported and is NOT a pass mark.** MPAC's training data contains about 44,000 indel pieces (the figure comes from the instructions for this draft and was not independently counted); model 2 has seen none. A mark of "beat MPAC" would compare a model that trained on indels with one that did not, so it would not test what this file asks. MPAC's practice score (0.514) is above model 2's (0.473), but the interval of the difference (-0.093 to 0.011) includes zero. Its result will be reported beside the verdict and read by the rule of `REFERENCE_MPAC.md`: interval entirely below zero, MPAC ahead; containing zero, level; entirely above zero, model 2 ahead. This is not a mark.

**The marks code.** `src/evaluate_indels.py --locked` does not read the marks block. It checks the status line, the students block (key `fixed` must be true, keys `main` and `seed2` must name the runs), that `results/locked/INDELS.json` does not exist, and that the pipeline check passed. The marks above are applied by hand, using the written result file, when the result is recorded. The key names in the marks block are a record and are not read by any script.

**Are the marks reachable?** On the practice numbers all four would pass (0.074 with lower end 0.006; 0.473; both intervals above zero; gap 0.021 against 0.03). The locked set has about as many variants, so the intervals will be about as wide (about 0.07 to 0.09 on each side for model 2 alone). My rough guess, from the practice intervals and not checked by simulation: mark 1 is close to a coin flip, slightly better (about 55%); mark 2 passes about four times in five; mark 3 passes about nine times in ten; mark 4 about three times in four; all four together about one chance in three. These numbers are Claude's opinion.

## Reported beside the test, not pass or fail
- Every secondary subset above, for every model, with intervals.
- SK-N-SH for model 2 (no other model has that output).
- Activity and Spearman correlations.
- MPAC, as above.
- Model 2 on indels of size 22 to 54, flagged as outside the sizes the slide covered in training.
- Nothing is chosen from these.

## What a result would mean
- **Pass:** model 2's scores track measured effects of insertions and deletions it never saw, better than the earlier and the published models. The honest wording is "a small model scores indels with accuracy near single-letter accuracy". It does not show that model 2 handles indels better because of anything done for indels: nothing was. A pass would also say nothing about MPAC.
- **Pass on mark 1 alone is weak evidence about indels.** Model 2 has trained on the same lab's assay on other chromosomes, and model 1 and the published model have not. A gain over model 1 may be the same gain that single-letter variants showed, not something specific to indels. The single-letter practice gain (0.093) against the indel gain (0.074) is the only hint in the other direction; it is loose.
- **Fail:** model 2 does not clearly extend to indels at this size. It is reported as it is, and the locked chromosomes are not scored again.

## Known limitations (declared in advance)
- **One dataset and one split.** One table from one laboratory method (plasmid MPRA), two cell types scored, chromosomes 5, 14 and 22 only.
- **The locked chromosomes are not new.** Their single-letter variants were scored before (the model 2 locked test and its near-copy re-score). These indels are new to every model, but the chromosomes are not.
- **The practice half has been seen**, so the marks are informed by it. Marks chosen after seeing practice numbers can fit the practice set; the locked set is the check, and the marks are recorded before it.
- **Small primary subset.** About 370 to 410 variants per cell type are scored, so the intervals are wide (practice interval for model 2 alone: 0.344 to 0.584). Small differences cannot be resolved; the insertions-only and size-1 subsets are smaller still.
- **Indel effects may be measured less reliably than single-letter effects.** This is unverified: the data checked for this draft (the practice counts and scores) contain no statement of the labels' noise for indels. No label-reliability estimate was made for indels.
- **The ceiling is unknown.** With noisier labels, no model can reach a high correlation. The earlier estimate of about 0.91 for single-letter variants does not transfer.
- **Framing choices.** A shorter piece is padded with the construct's own continuation (our models) or with longer flanks (MPAC). That treatment is natural but not the only one, and other choices could move the scores. It was fixed here, not tuned.
- **Models beyond the six are not run** (Malinois, AlphaGenome-FT, model 3 pilots; see above).
- **MPAC's indel training data** (about 44,000 pieces) is not matched by model 2, so MPAC ahead of or level with model 2 has a simple explanation that is not a flaw of model 2.
- **Same-assay advantage** over model 1 and the published model, as in the single-letter test.

## Before the test can run (the gates, from the code)
1. This file's status line must read `Status: CONFIRMED` in bold, as in the earlier pre-declarations. `src/prepare_indels.py --locked` checks that exact bold string and stops if it is missing; `src/evaluate_indels.py --locked` checks it too.
2. The students block above must say `"fixed": true` (it does, copied from the single-letter test) and name `main` and `seed2`.
3. `data/indels/locked.parquet` must exist: run `src/prepare_indels.py --locked` first. It writes `data/indels/locked.parquet` and `locked_counts.json`.
4. `results/practice/indels_pipeline_check.json` must say pass (it does: 2026-11-21).
5. `results/locked/INDELS.json` must not exist. The test is run once.
6. Run `src/evaluate_indels.py --locked`. It repeats the two framing asserts first and stops if either fails.

## What the user must decide
1. **Mark 1 (model 2 against model 1):** keep "interval above zero" (practice lower end 0.006, close to a coin flip), or relax it (for example, point estimate above zero), or add a minimum gain such as 0.03 on top (practice 0.074, but it would make the mark harder).
2. **Mark 2 (minimum mean effect Pearson):** proposed **0.42**. Raise it to 0.45 or lower it to 0.40 if you want a different bar (model 1's practice score was 0.398, model 2's 0.473).
3. **Mark 3 (published and baseline):** proposed both intervals above zero. Confirm, or drop the baseline part (it is the weaker bar).
4. **Mark 4 (seed gap):** proposed **0.03** (the earlier tests used 0.02; practice here was 0.021). Keep 0.03 or hold 0.02, which the practice numbers say would already fail.
5. **MPAC:** confirm it stays reported and is not a pass mark.
6. **Primary subset:** confirm size 1 to 21 without near-copies, or change it now (the size 22 to 54 subset is too small to be primary; the all-sizes subset includes sizes the slide did not cover in training).
7. **Then** change the status line to CONFIRMED and tell Claude, who runs `prepare_indels.py --locked` and then `evaluate_indels.py --locked` once, and records the result through an Opus agent.

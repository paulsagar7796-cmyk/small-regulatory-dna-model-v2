# Indel locked test: can model 2 judge insertions and deletions? Report

**Date of the test:** 21 November 2026 (result commit `209e20d`). **Run once, blind. Result: FAIL.**

Mark 1 failed; marks 2, 3 and 4 passed. The pre-declaration required all four.

Source of every test number below: `results/locked/INDELS.json` and `results/locked/INDELS_VERDICT.json`. Practice numbers come from `results/practice/indels_practice.json` and are labelled as practice. Counts come from `data/indels/locked_counts.json` and `practice_counts.json`. Numbers are rounded to 3 decimals, except the mark-1 comparison, which is given to 4 as in the verdict file. Nothing here was recomputed: every figure is quoted from those files.

## Short answer

**Model 2's advantage over model 1 does not extend to insertions and deletions.** On the primary subset model 2 scored a mean effect Pearson of **0.4873** against model 1's **0.4908**, a difference of **-0.0034** with a 95% interval of **-0.075 to 0.081**. On the practice chromosomes the same comparison had given **+0.074** with an interval of **0.006 to 0.141**. The practice gain did not replicate.

## The question, and why it was asked

Published score tables already cover human single-letter variants; MPAC alone has published about 575 million such predictions (figure quoted from the pre-declaration). For those, a lookup is cheaper than running any model. A small model earns its place only where no table reaches. Insertions and deletions ("indels") are effectively unlimited in number, so nobody can tabulate them. The project's pitch, recorded on 9 November 2026, names them explicitly, and the council of 9 November 2026 left one change open — its change 4: how would the model score an insertion or deletion? This test is the answer.

Model 2 (Cislume Dibang, 1.32M parameters) is model 1's network trained on the lentiMPRA tables plus the Siraj natural-variant table, then taught by three trained copies of itself. **Every DNA piece it has ever seen had exactly 200 letters, and every variant it trained on was a single-letter change.** An indel makes the piece shorter or longer, so this is a question about a kind of input the model was never shown.

The measure, unchanged from the single-letter locked test: for each cell type (HepG2, a liver line; K562, a blood line), the Pearson correlation between the measured allelic skew (`log2Skew`, the lab's activity of version 2 minus version 1) and the predicted effect (predicted activity of version 2 minus version 1), over variants where both versions are measured and at least one is active. The two cell types are then averaged. Intervals are 2,000 bootstrap resamples of 100,000-letter genome blocks, the same resamples for every model and every subset.

## What was fixed before the test, in order

| When | What | Commit |
|---|---|---|
| 14 Nov 2026 | Locked chromosomes 5, 14, 22 fixed (with the single-letter test), and the leak guards that remove shared DNA | `0bf7b52` |
| 15 Nov 2026 | The two students fixed (taught model 2, seeds 20261113 and 2). Nothing about them was changed for this test | `d4ad416` |
| 21 Nov 2026 | Indel tables for the practice chromosomes 6, 13, 20 built and every model scored on them; the locked branch of the code gated on CONFIRMED | `f142770` |
| 21 Nov 2026 | Pre-declaration drafted with proposed marks (DRAFT, awaiting the maintainer) | `c532a01` |
| 21 Nov 2026 | **All four marks CONFIRMED by the maintainer, as proposed, with no number changed, before any locked chromosome was read** | `d148544` |
| 21 Nov 2026 | Locked indel table written and scored, once | `209e20d` |

The pre-declaration is `predeclarations/PREDECLARATION_INDELS.md`. The maintainer was told before confirming that mark 4's seed gap of 0.03 was looser than the 0.02 used in earlier tests, that mark 1 was close to a coin flip on the practice interval, and that MPAC would be reported and would not be a pass mark. The confirmation is recorded in the file's status line.

Fixed students:
- main: `more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed20261113`
- seed 2: `more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed2`

Before scoring, the code repeated the two framing checks: that the short-piece framing equals the existing single-letter framing on 1,000 pieces of 200 letters, and that our MPAC wrapper matches MPAC's own predict function on 300 pieces. Both held; the MPAC maximum absolute difference was 0.0.

## The data

The source is the Siraj and others natural-variant table (Nature 2026), the same table as the single-letter test. Of 22,951 indels in the table, 2,443 lie on chromosomes 5, 14 and 22. One was dropped for exceeding 54 letters; none was dropped for non-ACGT letters or wrong oligo lengths; the scan guard removed none (on the practice chromosomes it removed 5). That leaves **2,442 variants**: 1,383 deletions and 1,059 insertions; 1,181 on chromosome 5, 721 on 14, 540 on 22. By size: 1,243 of size 1, 1,155 of size 2 to 21, 44 of size 22 to 54. 217 of the size 1 to 21 variants are near-copies of training DNA (27 or more shared 24-letter words with one training piece).

The **primary subset** — the only one that decides the verdict — is size 1 to 21 with near-copies excluded: **2,181 variants** (1,234 deletions, 947 insertions). Of these, **430 were scored in HepG2 and 439 in K562** (those with at least one active version), and 2,174 / 2,170 entered the activity correlation.

## Result: the primary subset

All seven scored models, mean effect Pearson over HepG2 and K562, 2,181 variants, 430 scored in HepG2 and 439 in K562.

| Model | Mean effect Pearson (95% interval) | HepG2 | K562 | Mean Spearman | Activity correlation | Model 2 minus this (95% interval) |
|---|---|---|---|---|---|---|
| **Model 2 (main seed)** | **0.487** (0.399 to 0.565) | 0.436 | 0.539 | 0.334 | 0.781 | — |
| Model 2 (seed 2) | 0.496 (0.412 to 0.572) | 0.446 | 0.546 | 0.343 | 0.793 | -0.032 to 0.016 |
| Model 1 (main seed) | 0.491 (0.376 to 0.593) | 0.459 | 0.522 | 0.299 | 0.589 | -0.075 to 0.081 |
| Model 1 (seed 2) | 0.472 (0.355 to 0.571) | 0.456 | 0.489 | 0.267 | 0.584 | -0.056 to 0.099 |
| Published MPRA-LegNet | 0.420 (0.314 to 0.523) | 0.380 | 0.459 | 0.252 | 0.571 | 0.003 to 0.138 |
| 6-letter-word ridge baseline | 0.204 (0.118 to 0.294) | 0.160 | 0.248 | 0.175 | 0.374 | 0.213 to 0.345 |
| MPAC (10 clean networks) | 0.479 (0.383 to 0.567) | 0.427 | 0.531 | 0.330 | 0.823 | -0.041 to 0.058 |

The four values for our two models interleave: 0.472 (model 1, seed 2), 0.487 (model 2, main), 0.491 (model 1, main), 0.496 (model 2, seed 2). The verdict file states it plainly: "the two models overlap".

## The four marks

| # | Mark | Needed | Got | Result |
|---|---|---|---|---|
| 1 | Better than model 1 | Model 2 above model 1, and the 95% interval of the difference above zero | 0.4873 against 0.4908; difference **-0.0034**, interval **-0.075 to 0.081** | **FAIL** (both clauses fail) |
| 2 | Good in absolute terms | Mean effect Pearson at least 0.42 | 0.4873 | PASS |
| 3 | Better than the published model and the baseline | Both 95% intervals of the difference above zero | vs published **0.003 to 0.138**; vs baseline **0.213 to 0.345** | PASS |
| 4 | Stable across seeds | The two seeds within 0.03 | 0.0086 | PASS |

**Verdict: FAIL.** A pass needed all four.

Mark 3 passed by a very small margin: the lower end of the interval against the published model is 0.003. On the practice chromosomes that same interval had been 0.078 to 0.255.

## Mark 1: practice against locked

| Half | Model 2 (main) | Model 1 (main) | Difference | 95% interval of the difference |
|---|---|---|---|---|
| Practice (chromosomes 6, 13, 20; 2,141 variants) | 0.473 | 0.398 | +0.074 | 0.006 to 0.141 |
| **Locked (chromosomes 5, 14, 22; 2,181 variants)** | **0.487** | **0.491** | **-0.0034** | **-0.075 to 0.081** |

**This is a failure to replicate.** The practice interval's lower end was 0.006 — the pre-declaration said so at the time and called mark 1 "close to a coin flip, slightly better (about 55%)". The locked interval straddles zero and is about twice as wide on each side.

The mechanism, on the numbers: model 2 did not get worse between the halves (0.473 to 0.487). **Model 1 got much better** (0.398 to 0.491), and so did the published model (0.305 to 0.420) and the baseline (0.161 to 0.204), while MPAC went slightly down (0.514 to 0.479). My own inference, not a verified finding: the two halves differ in which indels they happen to contain, and the halves are small enough that a 0.09 swing in one model's score is possible; the two models that had seen this assay in training (model 2, MPAC) were flat or down, while the two that had not (model 1, published) rose. I have no explanation for that pattern and did not test one. It is also consistent with the practice gain having been noise in the first place.

## What passed, and what it means

- **Model 2 beats the published MPRA-LegNet** on these indels: 0.487 against 0.420, interval of the difference 0.003 to 0.138. The margin is small and the interval only just clears zero.
- **Model 2 beats the 6-letter-word baseline** comfortably: 0.487 against 0.204, interval 0.213 to 0.345. So the predictions carry real information about indel effects, well beyond word counts.
- **The two seeds agree to 0.009**, inside the 0.03 mark. Model 1's two seeds differ by more (0.491 and 0.472, a spread of about 0.019) — larger than the whole model-2-minus-model-1 difference of 0.0034. The quantity mark 1 was testing is smaller than the seed noise of one of the two models.
- **Absolute accuracy clears the 0.42 bar** at 0.487. An untrained reader should not read that as "accurate": no reliability estimate exists for these labels (see Limitations), so there is no ceiling to compare it with.
- **MPAC scored 0.479 and the interval of the difference (-0.041 to 0.058) contains zero**, which `predeclarations/REFERENCE_MPAC.md`'s rule reads as **level**. As declared in advance, this is reported and is **not** a pass mark, and it is not a win: MPAC's training data holds indel pieces (about 44,000, a figure quoted from the pre-declaration and never independently counted) and model 2's holds none, so "level" here is a comparison that favours MPAC's side on data and ours on size. **Model 1 is also level with MPAC on this subset** (model 1 0.491 against MPAC 0.479; the model-2-minus-MPAC interval and model 2's overlap with model 1 make the three indistinguishable here), so nothing about model 2 in particular follows from it.
- **Model 2 is far better at the activity of indel-bearing pieces** than model 1 or the published model: 0.781 against 0.589 and 0.571, with MPAC at 0.823. So the extra training data did transfer to indel pieces — for predicting how active a piece is. It did not transfer to the difference between the two versions, which is what this test measures. My inference, not verified: a model can place both versions of a variant correctly on the activity scale and still get the small gap between them wrong, and the gap is the quantity of interest.

## All nine subsets

Mean effect Pearson. The primary subset is the first row; the other eight were pre-declared as reported-only and decide nothing.

| Subset | Variants | Model 2 | Model 2 s2 | Model 1 | Model 1 s2 | Published | Baseline | MPAC | Model 2 minus model 1 (95% interval) |
|---|---|---|---|---|---|---|---|---|---|
| **Primary: size 1-21, no near-copy** | 2,181 | **0.487** | 0.496 | 0.491 | 0.472 | 0.420 | 0.204 | 0.479 | **-0.075 to 0.081** |
| Size 1-21, near-copies included | 2,398 | 0.487 | 0.489 | 0.470 | 0.449 | 0.400 | 0.191 | 0.488 | -0.052 to 0.095 |
| All sizes (1-54), near-copies included | 2,442 | 0.428 | 0.423 | 0.386 | 0.383 | 0.362 | 0.181 | 0.499 | -0.048 to 0.142 |
| Deletions (primary filter) | 1,234 | 0.471 | 0.489 | 0.454 | 0.430 | 0.371 | 0.180 | 0.461 | -0.078 to 0.119 |
| Insertions (primary filter) | 947 | 0.511 | 0.509 | 0.521 | 0.508 | 0.478 | 0.258 | 0.506 | -0.100 to 0.116 |
| Size 1 only, no near-copy | 1,117 | 0.462 | 0.457 | 0.454 | 0.424 | 0.416 | 0.228 | 0.405 | -0.100 to 0.138 |
| Size 2-21, no near-copy | 1,064 | 0.505 | 0.525 | 0.508 | 0.503 | 0.415 | 0.186 | 0.538 | -0.092 to 0.105 |
| Size 22-54, no near-copy | 35 | 0.341 | 0.267 | 0.098 | 0.052 | 0.330 | 0.417 | 0.671 | -0.234 to 0.786 |
| Near-copies only, size 1-21 | 217 | 0.553 | 0.481 | 0.292 | 0.241 | 0.212 | 0.093 | 0.559 | 0.108 to 0.437 |

Notes on this table, all from the same file:

- **No subset shows model 2 clearly ahead of model 1 except the near-copies one** (interval 0.108 to 0.437). That is the subset whose DNA model 2's training set contains near-copies of. On the practice chromosomes the same subset showed no gain at all (model 2 0.722 against model 1 0.723). Two subsets of 191 and 217 variants, with 41 to 56 scored per cell type, disagreeing completely, is what small samples look like; I would not draw a conclusion from either, and the primary subset excludes them for exactly this reason.
- **Insertions are the one direction where model 2 sits below model 1** (0.511 against 0.521), and deletions the one where it sits above (0.471 against 0.454). Both intervals span zero. On practice the deletions subset had been model 2's strongest (0.547 against 0.441, interval 0.033 to 0.166); that lead is almost gone here.
- **Large indels are where our design breaks down.** On size 22 to 54, model 2 scores 0.341 and model 1 0.098, while MPAC scores 0.671 and even the word-count baseline scores 0.417. This was expected and declared: our models were trained with a slide of up to 21 letters, so k above 21 is outside anything they saw, and MPAC takes a 600-letter input where the shorter piece costs it proportionally less. The subset has 35 variants with only 14 and 11 scored, and 2 of the 2,000 bootstrap resamples were degenerate, so **it cannot be read**. Its intervals (for model 2, -0.140 to 0.788) say so.
- **On all sizes together MPAC is effectively ahead** (0.499 against 0.428; the interval of the difference is -0.143 to 0.0001, which only just contains zero). That is driven by the 22-to-54 tail.
- **SK-N-SH** (a nerve line; only model 2 has an output for it) scored 0.351 on the primary subset (seed 2: 0.361), over 421 variants — well below its HepG2 and K562 numbers, and below its own practice value of 0.457. No other model has that output, so there is nothing to compare it with.

## Honest interpretation

**What this test shows.** Model 2's extra training data — about 209,000 measured single-letter variants — bought a large, replicated gain on single-letter effects (locked single-letter test: 0.594 against model 1's 0.508, gain 0.086, interval 0.074 to 0.099). **On insertions and deletions that gain is absent.** The two models are indistinguishable here, and so is MPAC. Both of our models beat the published MPRA-LegNet and both beat the word-count baseline, so a small letters-only network does carry usable signal about indel effects — but model 2's distinctive advantage is not a general advantage over model 1; it is specific to the kind of change it was trained on.

**What this test does NOT show.**
- It does not show that model 2 is *bad* at indels. 0.487 is well above the baseline and above the published model. It shows that model 2 is **not better than model 1** at indels.
- It does not show that model 1 is better than model 2 at indels. The point difference is -0.0034 on an interval of -0.075 to 0.081, smaller than model 1's own seed-to-seed spread. The honest reading is "no difference measured", not "model 1 wins".
- It does not show that the single-letter gain was wrong or inflated. That was a separate test on 29,397 variants with a tight interval.
- It does not show anything about MPAC. MPAC trained on indel pieces and we did not, so "level with MPAC" is not a result about model 2's quality, and **model 1 is level with MPAC on this subset too**. Reading the MPAC line as a consolation win would be wrong on two counts: the comparison was declared not to be a mark, and it does not separate our two models.
- It does not tell us why. We did not diagnose the failure. The activity numbers (model 2 0.781 against model 1 0.589) say model 2 does understand these pieces better in absolute terms; something about the *difference* between a 200-letter piece and a shorter or longer one is where the advantage disappears. That is an open question, not a finding.
- The practice-to-locked pattern (model 1 and the published model both jumping about 0.09 to 0.12, model 2 and MPAC flat or down) is unexplained. It could be the halves' composition, it could be noise, and it weakens any reading of either half on its own.

**On the mark that failed.** Mark 1 was the only mark that asked the question the test was built for — does the extra data help where lookup tables cannot reach? The three marks that passed are the floor marks: beat the published model, beat the baseline, be stable, be above an absolute bar. The test failed on the one mark that mattered, and the pre-declaration had set the rule that all four were needed. That rule is what turns an attractive practice number (+0.074, interval just clear of zero) into a recorded failure rather than a claim.

## Limitations

From the pre-declaration's own list, plus what the numbers add:

- **Small subsets and wide intervals.** The primary subset has 2,181 variants, but only 430 and 439 are scored, because most pieces are not active in a given cell type. Model 2's interval alone spans 0.399 to 0.565. Differences of a few hundredths cannot be resolved, which is exactly the size of the difference mark 1 asked about.
- **One dataset, one split, one lab method.** One table from one laboratory (plasmid MPRA), two cell types scored, chromosomes 5, 14 and 22 only.
- **The practice half had been seen**, so the marks were informed by it. That is declared, and the locked half was the check. The check did not confirm the practice number.
- **The size 22 to 54 subset cannot be read**: 35 variants, 14 and 11 scored, 2 degenerate bootstrap draws, intervals hundreds of points wide. Nothing in it — including MPAC's 0.671 and the baseline's 0.417 — should be treated as a measurement.
- **The near-copies subset behaves differently and inconsistently.** It is the only subset where model 2 clearly beats model 1 (interval 0.108 to 0.437), and on the practice half the same subset showed no gain whatsoever. Its 217 variants and 41 to 56 scored points are too few to tell leak-driven advantage from noise. It is excluded from the primary subset.
- **Indel label reliability was never estimated.** No noise estimate for indel `log2Skew` exists in anything read for this test, so the ceiling is unknown. The earlier figure of about 0.91 for single-letter variants does not transfer, and 0.487 therefore cannot be placed on a scale.
- **Framing choices were fixed, not tuned.** A shorter piece is padded with the construct's own continuation (our models) or absorbed by longer vector flanks (MPAC). That is a natural treatment but not the only one. We ran the MPAC wrapper with one centred window, not the preprint's 18 shifted windows, as in the earlier MPAC reference run; so our MPAC numbers may understate it.
- **The locked chromosomes are not new ground.** Their single-letter variants were scored before (the model 2 locked test and its near-copy re-score). The indel rows are new to every model; the chromosomes are not.
- **Same-assay advantage** over model 1 and the published model, as in the single-letter test: model 2 saw this lab method and this laboratory's measurements. That advantage did not produce a measurable gain here.
- **Models beyond the seven were not run.** Malinois trained on chromosomes 5 and 14, so it is not a fair comparator. AlphaGenome-FT and the model 3 pilots trained on these chromosomes and their permission terms are conditional, so they were excluded in advance.
- **Not peer reviewed**, and **not clinical evidence**: this is a lab measurement in cell lines, from a small independent lab.

## Status

**The indels of chromosomes 5, 14 and 22 are now closed.** The test was run once, blind, and will not be run again. The result of record is FAIL on mark 1.

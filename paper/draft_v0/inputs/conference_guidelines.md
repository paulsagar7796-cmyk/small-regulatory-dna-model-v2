# Guidelines: bioRxiv-style preprint (first internal draft), sized to fit a later ML-for-genomics workshop

## Submission deadline

November 18, 2026. Use `cutoff_date = 2026-11-18`: cite only work that is public by this date.

## Page limit

Main text at most **9 pages** single-column (excluding references and appendix). Appendix allowed for the full screen tables.

## Mandatory sections, in this order

1. Abstract (single paragraph, 150-250 words)
2. Introduction
3. Related Work
4. Methods
5. Results
6. Discussion and Limitations
7. Conclusion
8. Data and Code Availability
9. References
10. Appendix (optional)

## Formatting rules

- Single-column, 11pt, 1in margins, `\documentclass[11pt]{article}` (use the bundled template)
- natbib numeric citations via `\cite{...}`
- Tables with booktabs; figures at 300 DPI
- Leave the author block as "[Authors to be added]"; no names, affiliations, emails or acknowledgements

## Audience

Computational biologists and ML-for-genomics researchers. Define assay terms (MPRA, lentiMPRA, allelic skew, log2FC) briefly on first use.

## Claims policy (strict; reviewers will check every number against the experimental log)

- Every number must come from the experimental log, unchanged. No new numbers, no rounding beyond 3 decimals, no speed or cost figures (inference speed was not measured).
- Do NOT claim state of the art. The model is slightly but significantly behind MPAC on the locked test; say so in the abstract.
- Size comparison: about 3 times smaller than one comparable single network (1.32M vs 4.1M); about 31 times smaller than the ensemble used per chromosome. Do not say "30x smaller" without that qualification.
- Clearly separate the one-shot blind locked test from the additional tests that are NOT blind (saturation-scan locked half second read, four-element test, lentiMPRA fold 10).
- State the same-assay advantage: the gain over model 1 and the published MPRA-LegNet is partly because model 2 was trained on the same assay as the test; MPAC was also trained on this assay family.
- Disclose the leak-guard correction and the one-window MPAC deviation.
- The architecture is the published MPRA-LegNet design; contributions are the multi-assay shared body, self-distillation (small gain), the blind-evaluation protocol, the ensemble decomposition and the negative results.
- Do not mention any other model of this lab beyond model 1 and model 2. Call them "model 1" and "model 2" (the released names are Cislume Lohit and Cislume Dibang; mention once in Data and Code Availability).

## Review criteria

Scientific soundness, honesty of claims, clarity for biologists, reproducibility (seeds, hyperparameters, splits, leak guards).

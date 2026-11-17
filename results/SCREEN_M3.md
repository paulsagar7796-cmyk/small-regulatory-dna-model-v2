# Model 3 screen (practice data only)

One change at a time against untaught model 2. Read-out: the new outputs. Practice numbers, not test results.

| Run | Seed | Practice chromosomes 6, 13, 20 (effect) | Scan practice half | lentiMPRA fold 9 |
|---|---|---|---|---|
| Untaught model 2 (reference) | 20261113 / 2 / 3 | 0.573 / 0.573 / 0.573 | 0.687 / 0.679 / 0.683 | 0.829 / 0.831 / 0.833 |
| (a) Paired-effect loss, weight 1.0 | 20261113 | 0.560 | 0.666 | 0.808 |
| (a) Paired-effect loss, weight 1.0 | 2 | 0.554 | 0.670 | see log.json |
| Control: paired rows, loss weight 0.000001 | 20261113 | 0.578 | 0.684 | see log.json |
| (b) Extra Gosai pieces (142,485, exact leak guard) | 20261113 | 0.575 | 0.688 | see log.json |
| (b) Extra Gosai pieces | 2 | 0.575 | 0.697 | see log.json |
| (c) Wider body, 4.0M parameters (`--width 1.75`) | 20261113 | 0.582 | 0.694 | 0.838 |
| (c) + (b) Wider body with the extra Gosai pieces | 20261113 | 0.583 | 0.704 | 0.840 |
| (d) EvoAug-style augmentation, exactly 2 per piece, 5 clean epochs | 20261113 | 0.549 | 0.653 | 0.808 |
| (d) same | 2 | 0.545 | 0.652 | 0.804 |
| (d) 0 to 2 per piece, 5 clean epochs | 20261113 | 0.558 | 0.666 | 0.821 |
| (e) Malinois design (MPAC's body, 5.0M, random start), our recipe | 20261113 | 0.472 | 0.580 | 0.755 |
| (e) same | 2 | 0.484 | 0.580 | 0.758 |
| (e) same, peak learning rate 0.003 | 20261113 | 0.521 | 0.598 | 0.790 |

**Reading (2026-11-16):** the paired-effect loss lowers every measure in both seeds: closed, no gain. The control shows the extra paired rows alone do no harm (0.578 against 0.573, one seed, within what one seed can show), so the loss term is what hurts.

(b) The extra pieces of the same measurement add almost nothing on natural variants (0.575 / 0.575 against 0.573) and at most a little on the scan (0.688 / 0.697 against 0.679 to 0.687). More data of this kind is not the limit at this model size.

(c) A three times larger body gains about 0.009 on natural variants (one seed). With the extra pieces it gains the same there and a little more on the scan. For scale: the taught model 2 scores 0.584 / 0.699 and MPAC 0.609 on the practice chromosomes, so no single change here closes the gap to MPAC.

(d) EvoAug-style augmentation (deletions, insertions, inversions, rolls, letter swaps, then clean fine-tuning) lowers every measure, more so in the heavier setting: closed, no gain. The model looks limited by capacity, not by overfitting.

(e) The Malinois design trained with our data and recipe is far below our own design. A lower learning rate helps (0.472 to 0.521), and fold 9 was still rising at the last epoch in all three runs, so these runs are undertrained; but even the best is 0.05 below a network a quarter of its size. Closed for now: MPAC's strength does not transfer by swapping in its body.

Note: the reference runs' training table has the cap-8 leak-guard flaw (near-copies of practice pieces in training, worth about 0.007 on the taught model's practice score); the extra pieces were filtered with the exact guard.

Result files: `results/practice/more_HEPG2-K562-SKNSH-HCT116-A549_pair*.json`.

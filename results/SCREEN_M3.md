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

**Reading (2026-11-16):** the paired-effect loss lowers every measure in both seeds: closed, no gain. The control shows the extra paired rows alone do no harm (0.578 against 0.573, one seed, within what one seed can show), so the loss term is what hurts.

(b) The extra pieces of the same measurement add almost nothing on natural variants (0.575 / 0.575 against 0.573) and at most a little on the scan (0.688 / 0.697 against 0.679 to 0.687). More data of this kind is not the limit at this model size.

Note: the reference runs' training table has the cap-8 leak-guard flaw (near-copies of practice pieces in training, worth about 0.007 on the taught model's practice score); the extra pieces were filtered with the exact guard.

Result files: `results/practice/more_HEPG2-K562-SKNSH-HCT116-A549_pair*.json`.

# small-regulatory-dna-model-v2

**Work in progress: model 2, the "more data" route.** This repository continues [small-regulatory-dna-model](https://github.com/paulsagar7796-cmyk/small-regulatory-dna-model) (model 1), the lab's released 1.3M-parameter model of regulatory DNA activity.

From [PurvoLabs](https://github.com/PurvoLabs). There are practice numbers from one seed, but no test result yet.

## Why a second model

Model 1 was trained on one public dataset (lentiMPRA, three cell types). After it was released we tried to push it further without new data, and every route was used up:

| What we tried on model 1 | Outcome |
|---|---|
| Changes to the network design (five of them) | No gain |
| A 1.5 times wider network | Small gain on whole pieces only |
| A second round of learning from our own trained models | No gain |
| Averaging our four best models | +0.007 on single-letter changes, not clearly above zero |

All those models learned from the same data, so they share the same mistakes. The remaining route is **new training data**.

## The plan

```mermaid
flowchart LR
    A["lentiMPRA tables<br/>model 1's data<br/>3 cell types"] --> M["One network<br/>same design and size as model 1<br/>8 outputs"]
    B["Natural-variant MPRA table<br/>Siraj and others 2026<br/>about 209,000 variants, 2 versions each<br/>up to 5 cell types"] --> M
    M --> P["Practice measures<br/>scan practice half<br/>chromosomes 6, 13, 20<br/>lentiMPRA fold 9"]
    P --> L["Locked test<br/>chromosomes 5, 14, 22<br/>only after the pass marks are confirmed"]
```

- **Same network, more data.** The design and size of model 1, with five extra outputs for the new table's cell types (HepG2, K562, SK-N-SH, HCT116, A549).
- **Why this table.** It holds about 280,000 single-letter variants, each measured as a pair: the piece with one letter, and the piece with the other. Model 1 never trained on a measured single-letter pair.
- **A control run** trains on the lentiMPRA data alone with the same pieces removed, so the effect of the new data can be read cleanly.

## The split (fixed before any model was trained on the new table)

| Part | Chromosomes | Variants | Use |
|---|---|---|---|
| Training | all others | 208,783 | Training |
| Practice | 6, 13, 20 | 28,104 | Every choice |
| Locked | 5, 14, 22 | 29,397 | One test, after the maintainer confirms the pass marks. Not written to disk before that |

Leak guards remove pieces that are the same DNA across parts, and everything next to model 1's single-letter test pieces. Details and counts: `src/prepare_more_data.py`, `results/split_counts.json`.

## What model 2 has to beat (practice chromosomes 6, 13, 20; practice numbers, not test results)

Pearson correlation between measured and predicted effect of a single-letter variant, over variants where at least one version is active, mean of HepG2 and K562:

| Model | Effect of the variant | Activity of the pieces |
|---|---|---|
| Model 1 (released) | 0.491 (seed 2: 0.484) | 0.610 |
| Published MPRA-LegNet | 0.455 | 0.584 |
| 6-letter-word baseline | 0.313 | 0.407 |

On model 1's other practice measure (the saturation scan's practice half) model 1 scores 0.627.

## First practice result (one seed)

Pearson correlation, practice data only, one seed (20261113). Not test results.

| Model | Scan practice half | Natural variants (chr 6, 13, 20) | lentiMPRA fold 9 |
|---|---|---|---|
| Model 1 (released) | 0.627 | 0.491 | 0.842 |
| Control (lentiMPRA only, same removals) | 0.595 | 0.473 | 0.824 |
| Model 2, new outputs | 0.687 | 0.573 | 0.829 |

The new data clearly improves single-letter accuracy on both practice measures, with intervals of the difference from model 1 above zero. Whole-piece accuracy on lentiMPRA is slightly below model 1, but model 2 has not yet had model 1's self-distillation step. Part of the gain may come from training on the same laboratory's assay. Full record: [results/PRACTICE_RUN_1.md](results/PRACTICE_RUN_1.md).

## Status

| Step | Status |
|---|---|
| Split and leak guards | Done, 14 Nov 2026 |
| Draft pre-declaration ([predeclarations/PREDECLARATION_MORE_DATA.md](predeclarations/PREDECLARATION_MORE_DATA.md)) | Written. **Pass marks not confirmed** |
| Reference numbers for model 1 on the practice chromosomes | Done (table above) |
| First practice runs: model 2 and the control | Done, 14 Nov 2026 ([results/PRACTICE_RUN_1.md](results/PRACTICE_RUN_1.md)) |
| Second seed of model 2 | Running |
| Choose the read-out and the training recipe on practice data | To do |
| Locked test | Only after the maintainer confirms the pass marks |

## Honest framing, decided in advance

If model 2 wins, the claim is **"a better small model with more data"**. It is not a same-data comparison: model 2 has seen the test's assay and laboratory during training, and model 1 and the published model have not. Other groups have trained models on this family of data (Malinois, MPAC); we have not run them and make no claim against them.

## Layout

| Path | What |
|---|---|
| `src/prepare_more_data.py` | The split and the leak guards |
| `src/train_more_data.py` | Trains model 2 (and the control with `--new_cells none`) |
| `src/evaluate_practice.py` | Scores a run on practice data beside model 1 and the published model |
| `predeclarations/` | The draft rules for the locked test |
| `results/` | Result files as they are produced |
| other files in `src/` | Model 1's code, which model 2 builds on |

Model 1's weights are read through a local link (`released_weights`) to the first repository's `weights/` folder. Data sources: [DATA.md](DATA.md).

## Safety, maintainer, licence

This model scores DNA. It does not generate DNA. **Maintainer:** [ORCID 0009-0007-7020-8949](https://orcid.org/0009-0007-7020-8949). **Licence:** not chosen yet; `src/third_party/legnet_model.py` stays under its MIT licence. The code and text were produced with an AI coding assistant (Claude), directed and checked by the maintainer.

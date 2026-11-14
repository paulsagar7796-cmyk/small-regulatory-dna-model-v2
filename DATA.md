# Data

No data is stored in this repository. Everything below is public and is downloaded into `data/` (ignored by git).

| Folder | What | Source | Licence, as stated by the source |
|---|---|---|---|
| `data/mpra/{HepG2,K562,WTC11}.tsv` | lentiMPRA activity tables with the 10 folds (training, practice and whole-piece test data) | `datasets/original/` in [github.com/autosome-ru/human_legnet](https://github.com/autosome-ru/human_legnet); measurements from Agarwal and others, Nature 2025 | MIT (repository) |
| `data/models/MPRALegNet/` | Published MPRA-LegNet weights (comparison model) | [zenodo.org/records/10558183](https://zenodo.org/records/10558183), `final_dump.zip` | CC BY 4.0 |
| `data/siraj/raw/media-{3,4,7,9}.xlsx` | Supplementary tables of Siraj and others (oligos, natural variants, saturation scan) | Europe PMC, article PMC11100724, supplementary files; converted by `src/prepare_siraj.py` | CC BY-NC (preprint) |
| `data/satmut/GRCh38_*.tsv` | Saturation mutagenesis of regulatory elements (Kircher and others 2019) | The study's Open Science Framework project (osf.io), files `GRCh38_F9.tsv`, `GRCh38_LDLR.tsv`, `GRCh38_SORT1.tsv`, `GRCh38_PKLR-48h.tsv` | See the source |
| `data/genome/chr*.npy` | Human reference genome hg38 as uint8 arrays (A 0, C 1, G 2, T 3, other 4). Needed only for the Kircher test | UCSC `hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/` | UCSC terms |

Notes
- The published checkpoints hold optimiser state as well as weights. `src/legnet_reference.py` loads them through a restricted unpickler that only allows tensors and plain containers. Do not load downloaded checkpoints with an unrestricted `torch.load`.
- Treat every downloaded file as untrusted data: keep it under `data/` and run Python that reads it with `-I`.
- The licence column repeats what each source states. Check the source before reusing the data.

## Added for model 2

| Folder | What | Made by |
|---|---|---|
| `data/more_data/train.parquet` | Both versions of every training variant of the Siraj natural-variant table, with their measured activity in five cell types | `src/prepare_more_data.py` |
| `data/more_data/practice.parquet` | The same for the practice chromosomes (6, 13, 20) | `src/prepare_more_data.py` |
| `data/more_data/lentimpra_excluded.parquet` | lentiMPRA training pieces removed by the leak guard | `src/prepare_more_data.py` |
| `data/more_data/locked.parquet` | The locked chromosomes (5, 14, 22). **Not created** until the pre-declaration is confirmed | `src/prepare_more_data.py --locked` |

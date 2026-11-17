"""Model 3: build the training and practice tables of the new split (locked = chromosome 7, practice = chromosome 13).
Contract: predeclarations/PREDECLARATION_MODEL3.md. Same source, row filter and columns as src/prepare_more_data.py
(natural-variant table of Siraj and others: data/siraj/variants.parquet + oligos.parquet; single-letter variants whose two
versions are both 200 letters of ACGT).

    training  = every chromosome except 7 and 13  (so chromosomes 5, 14, 22, 6, 20 are ordinary training data here)
    practice  = chromosome 13
    locked    = chromosome 7   (its MEASUREMENTS are never read here; with --locked the table is written, which is refused
                                unless the pre-declaration's status line says CONFIRMED)

Leak guards (the EXACT count, `exact_best_match` of src/prepare_gosai_extra.py: "same DNA" = one piece shares at least 27
of its 177 24-letter words with ONE other piece, either strand):
  1. Variants within 200 letters of a saturation-scan variant on the same chromosome, or the same DNA as a scan piece, are
     removed from ALL parts (as in model 2).
  2. Training variant pieces that are the same DNA as a chromosome-7 or chromosome-13 variant piece (either version) are removed.
     The chromosome-7 pieces are read as SEQUENCES only, to build the guard's index.
  3. Training variant pieces that are the same DNA as a lentiMPRA piece of fold 9 or 10 (any cell type) are removed (as model 2).
  4. lentiMPRA training pieces (folds 1 to 8, 230-letter rows) that are the same DNA as a chromosome-7 or chromosome-13 variant
     piece are listed in data/model3/lentimpra_excluded.parquet (column `seq`, the 230-letter row, the format of model 2's
     lentimpra_excluded.parquet); the trainer removes them, together with fold 9/10 sequences, as src/train_more_data.py does.
Queries use both versions of each variant (the larger count counts).

Run from the repository root:
    ~/PurvoLabs/small-dna-model/.venv/bin/python -I src/prepare_model3.py            (training, practice, lentiMPRA exclusions)
    ~/PurvoLabs/small-dna-model/.venv/bin/python -I src/prepare_model3.py --locked   (also the chromosome-7 table; refuses unless CONFIRMED)
Writes data/model3/{train,practice13,lentimpra_excluded}.parquet and results/model3_split_counts.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from mpra_data import CELLS, VALIDATION_FOLD, encode, load  # noqa: E402
from prepare_gosai_extra import exact_best_match  # noqa: E402
from prepare_more_data import NEW_CELLS, SAME_DNA, SIRAJ, Index  # noqa: E402

OUT = Path("data/model3")
COUNTS = Path("results/model3_split_counts.json")
PREDECLARATION = Path("predeclarations/PREDECLARATION_MODEL3.md")
LOCKED_CHROM, PRACTICE_CHROM = "7", "13"


def main() -> None:
    locked_mode = "--locked" in sys.argv
    if locked_mode and "**Status: CONFIRMED" not in PREDECLARATION.read_text():
        sys.exit(f"Stopped: {PREDECLARATION} is not CONFIRMED yet.")
    OUT.mkdir(parents=True, exist_ok=True)
    ids = pd.read_parquet(SIRAJ / "variants.parquet", columns=["variant"])  # the variant names only
    o = pd.read_parquet(SIRAJ / "oligos.parquet").set_index("Variant").loc[ids.variant]
    p = ids.variant.str.split(":", expand=True)
    t = pd.DataFrame({"variant": ids.variant, "chr": p[0].str[3:], "pos": p[1].astype(int),
                      "seq1": o["Allele 1 Oligo"].to_numpy(), "seq2": o["Allele 2 Oligo"].to_numpy()})
    ok = (t.seq1.str.len() == 200) & (t.seq2.str.len() == 200) & t.seq1.str.fullmatch("[ACGT]+") & t.seq2.str.fullmatch("[ACGT]+")
    t = t[ok].reset_index(drop=True)
    t["part"] = np.where(t.chr == LOCKED_CHROM, "locked", np.where(t.chr == PRACTICE_CHROM, "practice", "train"))
    c1, c2 = encode(t.seq1), encode(t.seq2)
    counts = {"usable_single_letter_variants": int(len(t)), "by_part_before_guards": t.part.value_counts().to_dict()}

    held = (t.part != "train").to_numpy()
    held_seqs = np.concatenate([c1[held], c2[held]])
    held_index = Index(held_seqs, np.arange(len(held_seqs)))  # pieces of chromosomes 7 and 13, both versions: sequences only

    def best(index: Index, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum(exact_best_match(index, a), exact_best_match(index, b))

    # guard 1: saturation scan
    s = pd.read_parquet(SIRAJ / "satmut.parquet", columns=["var1", "var2"])
    scan_ids = sorted(set(pd.unique(pd.concat([s.var1, s.var2]).dropna())) - {"NA"})
    sp = pd.Series(scan_ids).str.split(":", expand=True)
    scan_pos = pd.DataFrame({"chr": sp[0].str[3:], "pos": sp[1].astype(int)})
    near = np.zeros(len(t), bool)
    for ch, g in scan_pos.groupby("chr"):
        m = (t.chr == ch).to_numpy()
        near[m] = (np.abs(t.pos.to_numpy()[m][:, None] - g.pos.to_numpy()[None, :]) < 200).any(axis=1)
    so = pd.read_parquet(SIRAJ / "oligos.parquet")
    so = so[so.Variant.isin(scan_ids)]
    scan_seqs = pd.concat([so["Allele 1 Oligo"], so["Allele 2 Oligo"]])
    scan_seqs = scan_seqs[(scan_seqs.str.len() == 200) & scan_seqs.str.fullmatch("[ACGT]+")]
    scan_index = Index(encode(scan_seqs), np.arange(len(scan_seqs)))
    g_scan = near | (exact_best_match(scan_index, c1) >= SAME_DNA) | (exact_best_match(scan_index, c2) >= SAME_DNA)

    # guard 3: lentiMPRA fold 9 and 10
    lenti = {c: load(c, list(range(1, 11))) for c in CELLS}
    lenti_held = pd.concat([d.seq[d.fold >= VALIDATION_FOLD] for d in lenti.values()]).drop_duplicates()
    lenti_held_index = Index(encode(lenti_held)[:, 15:215], np.arange(len(lenti_held)))

    train = (t.part == "train").to_numpy()
    g2 = train & (best(held_index, c1, c2) >= SAME_DNA)
    g3 = train & (best(lenti_held_index, c1, c2) >= SAME_DNA)
    t["drop"] = g_scan | g2 | g3
    counts.update({"removed_scan_neighbours_all_parts": int(g_scan.sum()), "of_which_within_200_letters": int(near.sum()),
                   "removed_scan_neighbours_by_part": t.part[g_scan].value_counts().to_dict(),
                   "removed_train_same_dna_as_chr7_or_chr13_variants": int(g2.sum()),
                   "removed_train_same_dna_as_lentimpra_fold_9_10": int(g3.sum()),
                   "train_removed_by_any_guard": int((t["drop"] & train).sum())})
    t = t[~t["drop"]].drop(columns="drop").reset_index(drop=True)
    counts["by_part_after_guards"] = t.part.value_counts().to_dict()

    # guard 4: lentiMPRA training pieces
    lenti_train = pd.concat([d.seq[d.fold < VALIDATION_FOLD] for d in lenti.values()]).drop_duplicates().reset_index(drop=True)
    g4 = exact_best_match(held_index, encode(lenti_train)[:, 15:215]) >= SAME_DNA
    counts["lentimpra_training_pieces_removed"] = int(g4.sum())
    counts["lentimpra_training_pieces_total"] = int(len(lenti_train))
    pd.DataFrame({"seq": lenti_train[g4]}).to_parquet(OUT / "lentimpra_excluded.parquet", index=False)

    want = ["variant"] + [f"{a}.log2FC{se}_{c}" for a in ("allele1", "allele2") for se in ("", ".SE") for c in NEW_CELLS] \
           + [f"log2Skew{se}_{c}" for se in ("", ".SE") for c in NEW_CELLS]
    if locked_mode:
        m = pd.read_parquet(SIRAJ / "variants.parquet", columns=want, filters=[("variant", "in", t.variant[t.part == "locked"].tolist())])
        t[t.part == "locked"].merge(m, on="variant").to_parquet(OUT / "locked7.parquet", index=False)
        print("wrote the locked table:", int((t.part == "locked").sum()), "variants")
        return
    keep = t[t.part != "locked"]  # chromosome 7's measurements are never read below
    m = pd.read_parquet(SIRAJ / "variants.parquet", columns=want, filters=[("variant", "in", keep.variant.tolist())])
    full = keep.merge(m, on="variant")
    assert set(full.chr) .isdisjoint({LOCKED_CHROM})
    full[full.part == "train"].to_parquet(OUT / "train.parquet", index=False)
    full[full.part == "practice"].to_parquet(OUT / "practice13.parquet", index=False)
    counts["written"] = {"train": int((full.part == "train").sum()), "practice13": int((full.part == "practice").sum()),
                         "train_chromosomes": sorted(full.chr[full.part == "train"].unique().tolist())}
    COUNTS.parent.mkdir(parents=True, exist_ok=True)
    COUNTS.write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()

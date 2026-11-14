"""Model 2 (more data): build the extra training table and the practice table from the natural-variant table of
Siraj and others (Table S3; data/siraj/variants.parquet and oligos.parquet, made by src/prepare_siraj.py).

The split is by chromosome and is fixed HERE, before any model is trained or scored on this table:
    locked   = chromosomes 5, 14, 22   (never written to disk by this script without --locked, which needs CONFIRMED)
    practice = chromosomes 6, 13, 20
    training = every other chromosome
Only single-letter variants whose two versions are both 200 letters long are used (281,327 of 304,278).

Leak guards. Two pieces count as "the same DNA" when ONE piece shares at least 27 of its 177 24-letter words with
ONE other piece (either strand), which means they have about 50 letters or more in common, exactly. A first
version of this script counted any shared word with ANY piece of the other set; that removed a third of the
training variants because common repeat families (the same words occur thousands of times in the genome)
matched everywhere, so it was replaced before any model was trained.
  1. Variants within 200 letters of a saturation-scan variant on the same chromosome, or the same DNA as a scan
     piece, are removed from ALL parts (the scan is model 1's single-letter practice and test data).
  2. Training variant pieces that are the same DNA as a practice or locked variant piece are removed
     (copies of a region on another chromosome).
  3. Training variant pieces that are the same DNA as a lentiMPRA piece in fold 9 or 10 (any cell type) are removed.
  4. lentiMPRA training pieces that are the same DNA as a practice or locked variant piece are listed in
     data/more_data/lentimpra_excluded.parquet and are removed from training by src/train_more_data.py.
Guards 2 and 4 read the SEQUENCES of the locked chromosomes (never their measurements).
The lentiMPRA tables carry no genome coordinates, which is why guards 3 and 4 compare sequences.

Run from the repository root:
    .venv/bin/python -I src/prepare_more_data.py            (training and practice tables)
    .venv/bin/python -I src/prepare_more_data.py --locked   (the locked table; refuses unless CONFIRMED)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from mpra_data import CELLS, VALIDATION_FOLD, encode, load  # noqa: E402

SIRAJ, OUT = Path("data/siraj"), Path("data/more_data")
PREDECLARATION = Path("predeclarations/PREDECLARATION_MORE_DATA.md")
LOCKED_CHROMS, PRACTICE_CHROMS = ["5", "14", "22"], ["6", "13", "20"]
NEW_CELLS = ["HEPG2", "K562", "SKNSH", "HCT116", "A549"]  # the table's names
K = 24


def words(codes: np.ndarray) -> np.ndarray:
    """(n, L) codes -> (n, L-K+1) uint64, one number per K-letter word."""
    c = codes.astype(np.uint64)
    out = np.zeros((c.shape[0], c.shape[1] - K + 1), np.uint64)
    for j in range(K):
        out = (out << np.uint64(2)) | c[:, j : c.shape[1] - K + 1 + j]
    return out


class Index:
    """Every 24-letter word of a set of pieces (both strands), with the piece it came from."""

    def __init__(self, codes: np.ndarray, piece: np.ndarray):
        w = np.concatenate([words(codes).ravel(), words((3 - codes)[:, ::-1]).ravel()])
        ids = np.tile(np.repeat(piece.astype(np.int32), codes.shape[1] - K + 1), 2)
        order = np.argsort(w, kind="stable")
        self.w, self.ids = w[order], ids[order]

    def best_match(self, codes: np.ndarray, own: np.ndarray | None = None, cap: int = 8, chunk: int = 10000) -> np.ndarray:
        """For each query piece: the largest number of its words found in a single indexed piece.
        `own` (optional) is the query's own id in the index, which is ignored. Each word looks at up to `cap` owners."""
        out = np.zeros(len(codes), np.int32)
        for i in range(0, len(codes), chunk):
            q = words(codes[i : i + chunk])
            lo, hi = np.searchsorted(self.w, q, "left"), np.searchsorted(self.w, q, "right")
            got = np.full(q.shape + (cap,), -1, np.int32)
            for j in range(cap):
                has = lo + j < hi
                got[..., j][has] = self.ids[(lo + j)[has]]
            if own is not None:
                got[got == own[i : i + chunk, None, None]] = -1
            # a word found twice in the same piece counts once
            got = np.sort(got, axis=2)
            got[..., 1:][got[..., 1:] == got[..., :-1]] = -1
            s = np.sort(got.reshape(len(q), -1), axis=1)
            same = (s[:, 1:] == s[:, :-1]) & (s[:, 1:] >= 0)
            pos = np.arange(same.shape[1])[None, :]
            last_break = np.maximum.accumulate(np.where(~same, pos, -1), axis=1)
            run = np.where(same, pos - last_break, 0).max(axis=1) + 1
            out[i : i + chunk] = np.where((s >= 0).any(axis=1), run, 0)
        return out


SAME_DNA = 27  # words shared with one piece: about 50 letters in common


def main() -> None:
    locked_mode = "--locked" in sys.argv
    if locked_mode and "**Status: CONFIRMED**" not in PREDECLARATION.read_text():
        sys.exit(f"Stopped: {PREDECLARATION} is not CONFIRMED yet.")
    OUT.mkdir(parents=True, exist_ok=True)
    ids = pd.read_parquet(SIRAJ / "variants.parquet", columns=["variant"])
    o = pd.read_parquet(SIRAJ / "oligos.parquet").set_index("Variant").loc[ids.variant]
    p = ids.variant.str.split(":", expand=True)
    t = pd.DataFrame({"variant": ids.variant, "chr": p[0].str[3:], "pos": p[1].astype(int),
                      "seq1": o["Allele 1 Oligo"].to_numpy(), "seq2": o["Allele 2 Oligo"].to_numpy()})
    ok = (t.seq1.str.len() == 200) & (t.seq2.str.len() == 200) & t.seq1.str.fullmatch("[ACGT]+") & t.seq2.str.fullmatch("[ACGT]+")
    t = t[ok].reset_index(drop=True)
    t["part"] = np.where(t.chr.isin(LOCKED_CHROMS), "locked", np.where(t.chr.isin(PRACTICE_CHROMS), "practice", "train"))
    c1 = encode(t.seq1)  # the two versions differ in one letter, so version 1's words plus the centre cover both closely
    c2 = encode(t.seq2)
    counts = {"usable_single_letter_variants": int(len(t)), "by_part_before_guards": t.part.value_counts().to_dict()}

    piece = np.arange(len(t))
    held = (t.part != "train").to_numpy()
    held_index = Index(np.concatenate([c1[held], c2[held]]), np.concatenate([piece[held], piece[held]]))
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
    lenti = {c: load(c, list(range(1, 11))) for c in CELLS}
    lenti_held = pd.concat([d.seq[d.fold >= VALIDATION_FOLD] for d in lenti.values()]).drop_duplicates()
    lenti_held_index = Index(encode(lenti_held)[:, 15:215], np.arange(len(lenti_held)))

    g_scan = near | (scan_index.best_match(c1) >= SAME_DNA)
    train = (t.part == "train").to_numpy()
    g2 = train & (held_index.best_match(c1) >= SAME_DNA)
    g3 = train & (lenti_held_index.best_match(c1) >= SAME_DNA)
    t["drop"] = g_scan | g2 | g3
    counts.update({"removed_scan_neighbours_all_parts": int(g_scan.sum()), "of_which_within_200_letters": int(near.sum()),
                   "removed_train_same_dna_as_held_out_variants": int(g2.sum()),
                   "removed_train_same_dna_as_lentimpra_fold_9_10": int(g3.sum())})
    t = t[~t["drop"]].drop(columns="drop").reset_index(drop=True)
    counts["by_part_after_guards"] = t.part.value_counts().to_dict()

    lenti_train = pd.concat([d.seq[d.fold < VALIDATION_FOLD] for d in lenti.values()]).drop_duplicates().reset_index(drop=True)
    g4 = held_index.best_match(encode(lenti_train)[:, 15:215]) >= SAME_DNA
    counts["lentimpra_training_pieces_removed"] = int(g4.sum())
    counts["lentimpra_training_pieces_total"] = int(len(lenti_train))

    want = ["variant"] + [f"{a}.log2FC{se}_{c}" for a in ("allele1", "allele2") for se in ("", ".SE") for c in NEW_CELLS] \
           + [f"log2Skew{se}_{c}" for se in ("", ".SE") for c in NEW_CELLS]
    if locked_mode:
        m = pd.read_parquet(SIRAJ / "variants.parquet", columns=want)
        t[t.part == "locked"].merge(m, on="variant").to_parquet(OUT / "locked.parquet", index=False)
        print("wrote the locked table:", int((t.part == "locked").sum()), "variants")
        return
    keep = t[t.part != "locked"]  # the locked chromosomes' measurements are never read below
    m = pd.read_parquet(SIRAJ / "variants.parquet", columns=want, filters=[("variant", "in", keep.variant.tolist())])
    full = keep.merge(m, on="variant")
    full[full.part == "train"].to_parquet(OUT / "train.parquet", index=False)
    full[full.part == "practice"].to_parquet(OUT / "practice.parquet", index=False)
    pd.DataFrame({"seq": lenti_train[g4]}).to_parquet(OUT / "lentimpra_excluded.parquet", index=False)
    (OUT / "counts.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()

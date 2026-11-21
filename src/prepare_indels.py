"""Indel (insertion / deletion) variant-effect test for model 2 and its comparison models: build the tables.

Source: the natural-variant table of Siraj and others (data/siraj/variants.parquet and oligos.parquet). Model 2 was trained
on single-letter variants only, so it has never seen an indel.

Which variants: ref and alt differ in length; both oligos use only A, C, G, T; ONE version is exactly 200 letters (the
deletion's allele 1 / the insertion's allele 2) and the other is exactly 200 - size letters; size <= 54.
Splits, fixed with the single-letter test (src/prepare_more_data.py): practice chromosomes 6, 13, 20; locked 5, 14, 22.
Leak guard (the same as guard 1 of prepare_more_data.py): variants within 200 letters of a saturation-scan variant on the
same chromosome, or sharing >= 27 24-letter words with one scan piece, are dropped.
`best_shared_words`: for each indel, the EXACT largest number of distinct canonical 24-letter words that its 200-letter
version shares with any ONE training piece of data/more_data/train.parquet (seq1 and seq2), counted as in
src/rescore_nearcopy.py (not the capped Index.best_match, which misses repeat-family near-copies). Near-copy = >= 27.

    .venv/bin/python src/prepare_indels.py             practice chromosomes -> data/indels/practice.parquet (+ practice_counts.json)
    .venv/bin/python src/prepare_indels.py --locked    refuses unless predeclarations/PREDECLARATION_INDELS.md contains
                                                       **Status: CONFIRMED**; writes data/indels/locked.parquet (NOT run during practice work)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "src/third_party")
from evaluate_locked import words  # noqa: E402  (canonical 24-letter words, both strands = one word)
from mpra_data import encode  # noqa: E402
from prepare_more_data import LOCKED_CHROMS, NEW_CELLS, PRACTICE_CHROMS, SAME_DNA, SIRAJ, Index  # noqa: E402

PRE = Path("predeclarations/PREDECLARATION_INDELS.md")
OUT = Path("data/indels")
MORE = Path("data/more_data")
MAX_SIZE = 54
LABELS = ["variant"] + [f"{a}.log2FC{se}_{c}" for a in ("allele1", "allele2") for se in ("", ".SE") for c in NEW_CELLS] \
         + [f"log2Skew{se}_{c}" for se in ("", ".SE") for c in NEW_CELLS]  # exactly `want` of prepare_more_data.py


def best_shared_words(query_codes: np.ndarray, train: pd.DataFrame) -> np.ndarray:
    """For each query piece (n, 200): the largest number of distinct canonical 24-letter words shared with ONE training
    piece (columns seq1 and seq2 of `train`), exact. Same method as src/rescore_nearcopy.py."""
    n = len(query_codes)
    hw = words(query_codes)
    key = np.unique(np.stack([hw.ravel(), np.repeat(np.arange(n, dtype=np.uint64), hw.shape[1])], 1), axis=0)  # a word counts once per piece
    W, P = key[:, 0], key[:, 1].astype(np.int64)
    best = np.zeros(n, np.int64)
    for col in ("seq1", "seq2"):
        codes = encode(train[col])
        for a in range(0, len(codes), 2000):
            q = np.sort(words(codes[a:a + 2000]), axis=1)
            first = np.concatenate([np.ones((len(q), 1), bool), q[:, 1:] != q[:, :-1]], 1).ravel()
            rows, w = np.repeat(np.arange(len(q)), q.shape[1])[first], q.ravel()[first]
            lo, hi = np.searchsorted(W, w, "left"), np.searchsorted(W, w, "right")
            c = hi - lo
            use = c > 0
            r, l, c = rows[use], lo[use], c[use]
            if not len(r):
                continue
            idx = np.repeat(l, c) + (np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c))
            u, k = np.unique(np.repeat(r, c).astype(np.int64) * n + P[idx], return_counts=True)
            np.maximum.at(best, u % n, k)
    return best


def main() -> None:
    locked_mode = "--locked" in sys.argv
    if locked_mode and (not PRE.exists() or "**Status: CONFIRMED**" not in PRE.read_text()):
        sys.exit(f"Stopped: {PRE} does not exist or is not CONFIRMED.")
    chroms = LOCKED_CHROMS if locked_mode else PRACTICE_CHROMS
    part = "locked" if locked_mode else "practice"
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. ids only (no measurements), every chromosome
    ids = pd.read_parquet(SIRAJ / "variants.parquet", columns=["variant"])
    p = ids.variant.str.split(":", expand=True)
    p.columns = ["c", "pos", "ref", "alt"]
    t = pd.DataFrame({"variant": ids.variant, "chr": p.c.str[3:], "pos": p.pos.astype(int), "ref": p.ref, "alt": p.alt})
    t = t[t.ref.str.len() != t.alt.str.len()].reset_index(drop=True)
    counts = {"indels_all_chromosomes": int(len(t))}
    t = t[t.chr.isin(chroms)].reset_index(drop=True)
    counts[f"indels_on_{part}_chromosomes"] = int(len(t))
    o = pd.read_parquet(SIRAJ / "oligos.parquet").set_index("Variant").loc[t.variant]
    t["seq1"], t["seq2"] = o["Allele 1 Oligo"].to_numpy(), o["Allele 2 Oligo"].to_numpy()
    t["kind"] = np.where(t.ref.str.len() > t.alt.str.len(), "deletion", "insertion")
    t["size"] = (t.ref.str.len() - t.alt.str.len()).abs()
    acgt = t.seq1.str.fullmatch("[ACGT]+") & t.seq2.str.fullmatch("[ACGT]+")
    l1, l2 = t.seq1.str.len(), t.seq2.str.len()
    shape = np.where(t.kind == "deletion", (l1 == 200) & (l2 == 200 - t["size"]), (l2 == 200) & (l1 == 200 - t["size"]))
    counts["dropped_non_acgt"] = int((~acgt).sum())
    counts["dropped_wrong_oligo_lengths"] = int((acgt & ~shape).sum())
    counts["dropped_size_over_54"] = int((acgt & shape & (t["size"] > MAX_SIZE)).sum())
    t = t[acgt & shape & (t["size"] <= MAX_SIZE)].reset_index(drop=True)
    assert (np.where(t.kind == "deletion", t.seq1.str.len() == 200, t.seq2.str.len() == 200)).all()
    assert (np.where(t.kind == "deletion", t.seq2.str.len() == 200 - t["size"], t.seq1.str.len() == 200 - t["size"])).all()
    counts["after_filters"] = int(len(t))
    long_codes = encode(pd.Series(np.where(t.kind == "deletion", t.seq1, t.seq2)))  # the 200-letter version

    # 2. guard 1 (saturation scan), as in prepare_more_data.py
    s = pd.read_parquet(SIRAJ / "satmut.parquet", columns=["var1", "var2"])
    scan_ids = sorted(set(pd.unique(pd.concat([s.var1, s.var2]).dropna())) - {"NA"})
    sp = pd.Series(scan_ids).str.split(":", expand=True)
    scan_pos = pd.DataFrame({"chr": sp[0].str[3:], "pos": sp[1].astype(int)})
    near = np.zeros(len(t), bool)
    for ch, g in scan_pos.groupby("chr"):
        m = (t.chr == ch).to_numpy()
        if m.any():
            near[m] = (np.abs(t.pos.to_numpy()[m][:, None] - g.pos.to_numpy()[None, :]) < 200).any(axis=1)
    so = pd.read_parquet(SIRAJ / "oligos.parquet")
    so = so[so.Variant.isin(scan_ids)]
    scan_seqs = pd.concat([so["Allele 1 Oligo"], so["Allele 2 Oligo"]])
    scan_seqs = scan_seqs[(scan_seqs.str.len() == 200) & scan_seqs.str.fullmatch("[ACGT]+")]
    scan_index = Index(encode(scan_seqs), np.arange(len(scan_seqs)))
    same = scan_index.best_match(long_codes) >= SAME_DNA
    drop = near | same
    counts["removed_by_guard_within_200_letters_of_scan_variant"] = int(near.sum())
    counts["removed_by_guard_same_dna_as_scan_piece"] = int(same.sum())
    counts["removed_by_guard_total"] = int(drop.sum())
    t, long_codes = t[~drop].reset_index(drop=True), long_codes[~drop]

    # 3. exact best_shared_words against the training pieces
    tr = pd.read_parquet(MORE / "train.parquet", columns=["seq1", "seq2"])
    t["best_shared_words"] = best_shared_words(long_codes, tr).astype(int)

    # 4. labels: only of the chromosomes of this part (never of any other chromosome)
    assert set(t.chr) <= set(chroms)
    m = pd.read_parquet(SIRAJ / "variants.parquet", columns=LABELS, filters=[("variant", "in", t.variant.tolist())])
    t = t.merge(m, on="variant", how="left")
    assert len(t) == counts["after_filters"] - counts["removed_by_guard_total"]
    assert set(t.chr) <= set(chroms)
    cols = ["variant", "chr", "pos", "ref", "alt", "kind", "size", "seq1", "seq2", "best_shared_words"] + LABELS[1:]
    t[cols].to_parquet(OUT / f"{part}.parquet", index=False)

    bins = pd.cut(t["size"], [0, 1, 21, MAX_SIZE], labels=["1", "2-21", "22-54"])
    counts.update({"final_variants": int(len(t)), "by_kind": t.kind.value_counts().to_dict(),
                   "by_size_bin": bins.value_counts().reindex(["1", "2-21", "22-54"]).astype(int).to_dict(),
                   "by_kind_and_size_bin": {f"{k}|{b}": int(((t.kind == k) & (bins == b)).sum()) for k in ("deletion", "insertion") for b in ("1", "2-21", "22-54")},
                   "near_copies_best_shared_words_ge_27": int((t.best_shared_words >= SAME_DNA).sum()),
                   "near_copies_in_size_1_to_21": int(((t.best_shared_words >= SAME_DNA) & (t["size"] <= 21)).sum()),
                   "share_any_word_with_training": int((t.best_shared_words > 0).sum()),
                   "by_chromosome": t.chr.value_counts().to_dict()})
    (OUT / f"{part}_counts.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()

"""Model 3 (more data): the extra training pieces from Table S2 of Gosai and others (data/gosai/Table_S2__MPRA_dataset.tsv,
798,064 pieces, labels in K562, HepG2 and SK-N-SH).

Table S2 is the same measurement family as the natural-variant table that model 2 trains on (99.98% of our pieces are in
it and the shared labels agree at about 0.997). What it adds are PIECES model 2 never saw: both versions of variants that
our table lacks (insertions and deletions, other windows), off-centre windows, pieces with a second variant, and pieces
that are not variants at all (regulatory elements). This script keeps those that are new and safe to train on.

The file is third-party data: it is read only as a table (never executed). Run it with the -I flag.

What is kept (every step is counted in results/gosai_extra_counts.json):
  1. exactly 200 letters of A, C, G, T;
  2. the chromosome is a TRAINING chromosome (not 5, 14, 22 = locked, not 6, 13, 20 = practice). The chromosome is read
     from the piece's ID (its first field) and from the `chr` column; they must agree. All IDs give one;
  3. the sequence is not already in data/more_data/train.parquet (either version);
  4. exact duplicate sequences inside the new set: one is kept;
  5. the same leak guards as src/prepare_more_data.py (same definition: "same DNA" = one piece shares at least 27 of
     its 177 24-letter words with ONE other piece, either strand; counted EXACTLY here, see exact_best_match):
       - saturation-scan pieces (data/siraj), or any variant position of the piece within 200 letters of a scan variant;
       - practice and locked variant pieces (both versions of every variant of chromosomes 6, 13, 20 and 5, 14, 22;
         SEQUENCES only, rebuilt from the oligo table exactly as prepare_more_data.py does, then checked to contain every
         sequence of practice.parquet and locked.parquet);
       - lentiMPRA pieces of fold 9 and fold 10 (any cell type).
Labels: activity (log2FC) in HEPG2, K562, SKNSH, in the same units as train.parquet; HCT116 and A549 are NaN.
The table has one row per piece: columns log2FC_<CELL> and log2FC.SE_<CELL> (the train table has allele1./allele2. prefixes).

Run from the repository root:
    ~/PurvoLabs/small-dna-model/.venv/bin/python -I src/prepare_gosai_extra.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from mpra_data import CELLS, VALIDATION_FOLD, encode, load  # noqa: E402
from prepare_more_data import K, LOCKED_CHROMS, NEW_CELLS, PRACTICE_CHROMS, SAME_DNA, SIRAJ, Index, words  # noqa: E402

GOSAI = Path("data/gosai/Table_S2__MPRA_dataset.tsv")
OUT, COUNTS = Path("data/more_data/train_gosai_extra.parquet"), Path("results/gosai_extra_counts.json")
LABELS = {"HEPG2": "HepG2", "K562": "K562", "SKNSH": "SKNSH"}  # our name -> the Gosai column prefix
CATEGORIES = ["single-letter pair", "insertion/deletion pair", "off-centre window", "non-variant piece", "multi-variant piece"]
HELD = set(LOCKED_CHROMS) | set(PRACTICE_CHROMS)


def parse_ids(ids: pd.Series) -> pd.DataFrame:
    """ID shapes: 'chr:pos:NA:NA' (no variant); 'chr:pos:ref:alt:R|A:wC|wL|wR' (one variant, R = reference allele, A = alternative);
    the same + ':Alt_chr-pos-R|A' (a second variant's state is noted); several of those joined by commas (the same oligo listed
    under several names, or several variants).
    Category rules: no variant -> non-variant piece; two or more different variants, or a second variant in its A state ->
    multi-variant piece; otherwise one variant: window wL or wR -> off-centre window; window wC with one-letter ref and alt ->
    single-letter pair, else insertion/deletion pair. (Alt notes ending in -R do not make a piece multi-variant.)"""
    rows = []
    for s in ids:
        toks = s.split(",")
        f = toks[0].split(":")
        positions = [int(f[1])]
        for t in toks:
            g = t.split(":")
            positions.append(int(g[1]))
            if len(g) == 7:
                positions.append(int(g[6].split("-")[1]))
        chroms = {t.split(":")[0] for t in toks}
        has_alt = any(len(t.split(":")) == 7 for t in toks)
        variants = {tuple(t.split(":")[:4]) for t in toks}  # distinct (chr, pos, ref, alt) among the main fields
        second_alt_allele = any(len(t.split(":")) == 7 and t.split(":")[6].endswith("-A") for t in toks)  # a second variant in its alt state
        if len(f) == 4:
            cat = "non-variant piece"
        elif len(variants) > 1 or second_alt_allele:  # an 'Alt_..-R' note only says that a second variant is in its reference state
            cat = "multi-variant piece"
        elif f[5] != "wC":
            cat = "off-centre window"
        elif len(f[2]) == 1 and len(f[3]) == 1 and f[2] in "ACGT" and f[3] in "ACGT":
            cat = "single-letter pair"
        else:
            cat = "insertion/deletion pair"
        key = ",".join(":".join(t.split(":")[:4] + ["*"] + t.split(":")[5:]) if len(t.split(":")) > 4 else t for t in toks)
        rows.append((f[0] if len(chroms) == 1 else None, int(f[1]), tuple(positions), f[4] if len(f) > 4 else None, key, cat, has_alt, len(toks)))
    return pd.DataFrame(rows, columns=["id_chr", "pos", "all_positions", "version", "pair_key", "category", "has_alt_note", "n_tokens"])


def exact_best_match(index: Index, codes: np.ndarray, chunk: int = 400) -> np.ndarray:
    """For each query piece: the largest number of its distinct 24-letter words found in ONE indexed piece (either strand).
    Same quantity as Index.best_match, but exact: Index.best_match looks at only the first `cap` = 8 pieces that contain a word,
    so for words that occur in many pieces (repeat families) it can miss a piece that shares 27 or more words. A leak check of
    the first version of this table found about 7,000 such pieces with 27 or more words, hence this function."""
    out = np.zeros(len(codes), np.int32)
    n_owner = int(index.ids.max()) + 1
    for i in range(0, len(codes), chunk):
        q = words(codes[i : i + chunk])  # (n, 177)
        lo, hi = np.searchsorted(index.w, q.ravel(), "left"), np.searchsorted(index.w, q.ravel(), "right")
        cnt = hi - lo
        total = int(cnt.sum())
        if total == 0:
            continue
        word_id = np.repeat(np.arange(q.size, dtype=np.int64), cnt)  # which query word (row * 177 + column)
        pos = np.repeat(lo, cnt) + (np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        owner = index.ids[pos].astype(np.int64)
        pairs = np.unique(word_id * n_owner + owner)  # a word counts once per owning piece
        row_owner = np.unique((pairs // n_owner // q.shape[1]) * n_owner + pairs % n_owner, return_counts=True)
        rows = row_owner[0] // n_owner
        np.maximum.at(out, i + rows, row_owner[1].astype(np.int32))
    return out


def main() -> None:
    steps, counts = [], {}

    def snap(name: str, d: pd.DataFrame, **extra) -> None:
        steps.append({"step": name, "pieces": int(len(d)), "by_category": {c: int((d.category == c).sum()) for c in CATEGORIES}, **extra})
        print(f"{name}: {len(d):,}  " + ", ".join(f"{c} {n:,}" for c, n in steps[-1]["by_category"].items()), flush=True)

    s = pd.read_csv(GOSAI, sep="\t", dtype={"chr": str, "OL": str, "IDs": str})
    s["seq"] = s.sequence.str.upper()
    s = pd.concat([s.drop(columns=["sequence"]), parse_ids(s.IDs)], axis=1)
    s["id_shape"] = s.IDs.str.count(":").astype(str) + " colons, " + s.n_tokens.astype(str) + " tokens"
    snap("0 start (all rows of Table S2)", s)
    complete_keys = set(s.pair_key[s.version == "R"]) & set(s.pair_key[s.version == "A"])  # both versions exist somewhere in Table S2
    counts["rows_with_duplicate_ids"] = int(s.IDs.duplicated(keep=False).sum())

    # ---- chromosome from the ID ----
    no_chr = s.id_chr.isna() | ~s.id_chr.isin([str(i) for i in range(1, 23)] + ["X", "Y"])
    counts["chromosome_question"] = {
        "ids_without_a_chromosome_or_chromosome_not_1_22_X_Y": int(no_chr.sum()),
        "id_chromosome_differs_from_chr_column": int((s.id_chr != s.chr).sum()),
        "id_shapes_all_rows": s.id_shape.value_counts().to_dict(),
        "examples_of_ids_per_shape": {k: g.IDs.head(2).tolist() for k, g in s.groupby("id_shape")},
        "example_ids_without_chromosome": s.IDs[no_chr].head(10).tolist(),
        "chromosome_Y_pieces": int((s.chr == "Y").sum()),
    }
    s = s[~no_chr & (s.id_chr == s.chr)]
    snap("1 chromosome established from the ID (agrees with the chr column)", s)

    ok = (s.seq.str.len() == 200) & s.seq.str.fullmatch("[ACGT]+")
    counts["not_200_letters_acgt_by_category"] = {c: int(((s.category == c) & ~ok).sum()) for c in CATEGORIES}
    s = s[ok]
    snap("2 exactly 200 letters of A, C, G, T", s)

    counts["on_held_chromosomes_by_part"] = {"locked_5_14_22": int(s.chr.isin(LOCKED_CHROMS).sum()), "practice_6_13_20": int(s.chr.isin(PRACTICE_CHROMS).sum())}
    s = s[~s.chr.isin(HELD)]
    snap("3 training chromosomes only", s)

    # ---- already in train.parquet? and the label agreement on those ----
    tr = pd.read_parquet("data/more_data/train.parquet")
    lab = pd.concat([pd.DataFrame({"seq": tr[col].str.upper(), **{c: tr[f"{a}.log2FC_{c}"] for c in LABELS}})
                     for a, col in (("allele1", "seq1"), ("allele2", "seq2"))])
    have = set(lab.seq)
    inn = s.seq.isin(have)
    agree = {}
    j = s[inn].groupby("seq")[[f"{p}_log2FC" for p in LABELS.values()]].mean().join(lab.groupby("seq")[list(LABELS)].mean())
    for c, p in LABELS.items():
        m = j[c].notna() & j[f"{p}_log2FC"].notna()
        a, b = j.loc[m, c].to_numpy(), j.loc[m, f"{p}_log2FC"].to_numpy()
        agree[c] = {"pieces": int(m.sum()), "pearson": float(np.corrcoef(a, b)[0, 1]), "mean_abs_difference": float(np.abs(a - b).mean()),
                    "mean_difference_gosai_minus_train": float((b - a).mean())}
    counts["label_agreement_on_pieces_in_both_tables"] = agree
    counts["already_in_train_table_by_category"] = {c: int(((s.category == c) & inn).sum()) for c in CATEGORIES}
    s = s[~inn]
    snap("4 sequence not already in train.parquet (either version)", s)

    # ---- exact duplicates inside the new set ----
    order = {c: i for i, c in enumerate(CATEGORIES)}
    s = s.assign(_o=s.category.map(order)).sort_values(["_o", "IDs"], kind="stable")
    dup = s.seq.duplicated(keep="first")
    counts["exact_duplicate_sequences_removed"] = int(dup.sum())
    s = s[~dup].drop(columns="_o").sort_values("IDs", kind="stable").reset_index(drop=True)
    snap("5 exact duplicate sequences removed (one kept)", s)

    # ---- leak guards (the code and the definition of prepare_more_data.py) ----
    codes = encode(s.seq)
    ids = pd.read_parquet(SIRAJ / "variants.parquet", columns=["variant"])
    o = pd.read_parquet(SIRAJ / "oligos.parquet").set_index("Variant").loc[ids.variant]
    p = ids.variant.str.split(":", expand=True)
    t = pd.DataFrame({"chr": p[0].str[3:], "seq1": o["Allele 1 Oligo"].to_numpy(), "seq2": o["Allele 2 Oligo"].to_numpy()})
    ok = (t.seq1.str.len() == 200) & (t.seq2.str.len() == 200) & t.seq1.str.fullmatch("[ACGT]+") & t.seq2.str.fullmatch("[ACGT]+")
    t = t[ok & t.chr.isin(HELD)].reset_index(drop=True)
    held_seqs = pd.concat([t.seq1, t.seq2]).reset_index(drop=True)
    for name in ("practice", "locked"):  # sequences only: the measurement columns are never read
        d = pd.read_parquet(f"data/more_data/{name}.parquet", columns=["seq1", "seq2"])
        missing = set(pd.concat([d.seq1, d.seq2])) - set(held_seqs)
        assert not missing, f"{len(missing)} {name} sequences are not in the rebuilt held-out set"
    held_index = Index(encode(held_seqs), np.arange(len(held_seqs)))
    counts["held_out_pieces_in_guard"] = int(len(held_seqs))

    sc = pd.read_parquet(SIRAJ / "satmut.parquet", columns=["var1", "var2"])
    scan_ids = sorted(set(pd.unique(pd.concat([sc.var1, sc.var2]).dropna())) - {"NA"})
    sp = pd.Series(scan_ids).str.split(":", expand=True)
    scan_pos = pd.DataFrame({"chr": sp[0].str[3:], "pos": sp[1].astype(int)})
    near = np.zeros(len(s), bool)
    for ch, g in scan_pos.groupby("chr"):
        sorted_pos = np.sort(g.pos.to_numpy())
        for i in np.flatnonzero((s.chr == ch).to_numpy()):
            q = np.array(s.all_positions.iat[i])
            k = np.searchsorted(sorted_pos, q)
            lo, hi = sorted_pos[np.clip(k - 1, 0, None)], sorted_pos[np.clip(k, None, len(sorted_pos) - 1)]
            near[i] = (np.minimum(np.abs(q - lo), np.abs(q - hi)) < 200).any()
    so = pd.read_parquet(SIRAJ / "oligos.parquet")
    so = so[so.Variant.isin(scan_ids)]
    scan_seqs = pd.concat([so["Allele 1 Oligo"], so["Allele 2 Oligo"]])
    scan_seqs = scan_seqs[(scan_seqs.str.len() == 200) & scan_seqs.str.fullmatch("[ACGT]+")]
    scan_index = Index(encode(scan_seqs), np.arange(len(scan_seqs)))
    lenti = {c: load(c, list(range(1, 11))) for c in CELLS}
    lenti_held = pd.concat([d.seq[d.fold >= VALIDATION_FOLD] for d in lenti.values()]).drop_duplicates()
    lenti_held_index = Index(encode(lenti_held)[:, 15:215], np.arange(len(lenti_held)))

    g_pos = near
    old_scan, old_held, old_lenti = (ix.best_match(codes) >= SAME_DNA for ix in (scan_index, held_index, lenti_held_index))  # the cap-8 version, for the record
    g_scan_seq, g_held, g_lenti = (exact_best_match(ix, codes) >= SAME_DNA for ix in (scan_index, held_index, lenti_held_index))
    counts["guard_hits_with_the_original_cap_8_function"] = {
        "same_dna_as_scan_piece": int(old_scan.sum()), "same_dna_as_practice_or_locked_variant_piece": int(old_held.sum()),
        "same_dna_as_lentimpra_fold_9_10": int(old_lenti.sum())}
    counts["guard_hits_missed_by_cap_8_but_found_by_exact_count"] = {
        "scan": int((g_scan_seq & ~old_scan).sum()), "practice_or_locked": int((g_held & ~old_held).sum()), "lentimpra": int((g_lenti & ~old_lenti).sum()),
        "pieces_hit_by_cap_8_but_not_by_exact_count (should be 0)": int(((old_scan & ~g_scan_seq) | (old_held & ~g_held) | (old_lenti & ~g_lenti)).sum())}
    counts["guard_hits_each_alone_before_any_guard_removal"] = {
        "within_200_letters_of_a_scan_variant": int(g_pos.sum()), "same_dna_as_scan_piece": int(g_scan_seq.sum()),
        "same_dna_as_practice_or_locked_variant_piece": int(g_held.sum()), "same_dna_as_lentimpra_fold_9_10": int(g_lenti.sum()),
        "any_guard": int((g_pos | g_scan_seq | g_held | g_lenti).sum())}
    counts["guard_hits_by_category"] = {c: int((((g_pos | g_scan_seq | g_held | g_lenti)) & (s.category == c)).sum()) for c in CATEGORIES}
    keep = np.ones(len(s), bool)
    for name, g in (("6 guard: scan (within 200 letters, or the same DNA as a scan piece)", g_pos | g_scan_seq),
                    ("7 guard: same DNA as a practice or locked variant piece", g_held),
                    ("8 guard: same DNA as a lentiMPRA piece of fold 9 or 10", g_lenti)):
        keep &= ~g
        snap(name + " (cumulative)", s[keep], removed_by_this_guard_alone=int(g.sum()))
    s = s[keep].reset_index(drop=True)

    # ---- output ----
    out = pd.DataFrame({"gosai_id": s.IDs, "chr": s.chr, "pos": s.pos, "pair_key": s.pair_key, "version": s.version, "category": s.category,
                        "has_alt_note": s.has_alt_note, "data_project": s.data_project, "seq": s.seq})
    for c, pfx in LABELS.items():
        out[f"log2FC_{c}"] = s[f"{pfx}_log2FC"].to_numpy()
    for c in NEW_CELLS:
        if c not in LABELS:
            out[f"log2FC_{c}"] = np.nan
    for c, pfx in LABELS.items():
        out[f"log2FC.SE_{c}"] = s[f"{pfx}_lfcSE"].to_numpy()
    for c in NEW_CELLS:
        if c not in LABELS:
            out[f"log2FC.SE_{c}"] = np.nan
    # a piece's partner (the other version of the same variant and window) may be missing from the kept set or from Table S2
    out["pair_complete_in_gosai"] = out.pair_key.isin(complete_keys)
    out["pair_complete_in_kept_set"] = out.pair_key.isin(set(out.pair_key[out.version == "R"]) & set(out.pair_key[out.version == "A"]))
    out.to_parquet(OUT, index=False)
    counts.update({"final_pieces": int(len(out)), "final_by_category": out.category.value_counts().to_dict(),
                   "final_by_data_project": out.data_project.value_counts().to_dict(),
                   "final_pairs_complete_in_kept_set": int(out.pair_complete_in_kept_set.sum()),
                   "final_with_a_missing_partner": int((~out.pair_complete_in_kept_set).sum()),
                   "labels_missing_in_final": {c: int(out[f"log2FC_{c}"].isna().sum()) for c in NEW_CELLS},
                   "steps": steps})
    COUNTS.parent.mkdir(parents=True, exist_ok=True)
    COUNTS.write_text(json.dumps(counts, indent=1))
    print(json.dumps({k: v for k, v in counts.items() if k != "steps"}, indent=1))


if __name__ == "__main__":
    main()

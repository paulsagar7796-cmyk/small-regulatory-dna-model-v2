"""Use the released model 2 (Cislume Dibang): predict the regulatory activity of 200-letter DNA pieces, or the effect
of every single-letter change in them.

Run from the repository root:

    python predict.py --seq <200 letters>                        activity of one piece
    python predict.py --fasta pieces.fa --out activity.tsv        activity of many pieces
    python predict.py --fasta pieces.fa --scan --out effects.tsv  effect of every single-letter change

Outputs (columns)
- plasmid_HepG2, plasmid_K562, plasmid_SKNSH, plasmid_HCT116, plasmid_A549: the model's estimate of log2FC in the
  plasmid MPRA of Siraj and others (Nature 2026). These are the outputs behind model 2's reported single-letter results.
- lenti_HepG2, lenti_K562, lenti_WTC11: the model's estimate of log2(RNA / DNA) in the lentiMPRA of Agarwal and others
  (Nature 2025), as for model 1.
- Effect of a change: activity of the changed piece minus activity of the original piece.

Each piece is placed between the two 15-letter adapters used in training and scored in both orientations (and both
strands), which are averaged: the recipe of every test in this repository. These are predictions of a lab reporter
measurement. They are research outputs, not clinical evidence.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "src/third_party")
from malinois_body import build_arch  # noqa: E402
from mpra_train_multi import predict_multi  # noqa: E402

WEIGHTS = Path("weights")
STUDENTS = ["more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed20261113",
            "more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed2"]
LEFT, RIGHT = "AGGACCGGATCAACT", "CATTGCGTGAACCGA"
CODE = {b: i for i, b in enumerate("ACGT")}
NAMES = {"HepG2": "lenti_HepG2", "K562": "lenti_K562", "WTC11": "lenti_WTC11", "new_HEPG2": "plasmid_HepG2",
         "new_K562": "plasmid_K562", "new_SKNSH": "plasmid_SKNSH", "new_HCT116": "plasmid_HCT116", "new_A549": "plasmid_A549"}
ORDER = ["new_HEPG2", "new_K562", "new_SKNSH", "new_HCT116", "new_A549", "HepG2", "K562", "WTC11"]  # columns as printed


def encode(seq: str) -> np.ndarray:
    return np.array([CODE.get(b, 4) for b in seq.upper()], dtype=np.uint8)


def load_students(both: bool, device):
    norm = json.loads((WEIGHTS / "normalisation.json").read_text())
    outs = norm["outputs"]
    models = []
    for run in STUDENTS[: 2 if both else 1]:
        m = build_arch("legnet", len(outs)).to(device)
        m.load_state_dict(torch.load(WEIGHTS / run / "best.pt", map_location=device, weights_only=True))
        models.append(m)
    return models, outs, norm["scale"]


def activity(pieces: np.ndarray, students, device) -> np.ndarray:
    """(n, 200) codes -> (n, 8) predicted activities, columns in the order of ORDER, in the data's own units."""
    models, outs, scale = students
    mean = np.array([scale[o]["mean"] for o in outs])
    std = np.array([scale[o]["std"] for o in outs])
    l, r = (np.tile(encode(a), (len(pieces), 1)) for a in (LEFT, RIGHT))
    flipped = np.where(pieces < 4, 3 - pieces, 4)[:, ::-1]  # the piece inserted the other way round
    total = 0
    for insert in (pieces, flipped):
        x = torch.from_numpy(np.ascontiguousarray(np.concatenate([l, insert, r], axis=1)))
        total = total + sum(predict_multi(m, x, device) for m in models) / len(models)
    pred = (total / 2) * std + mean
    return pred[:, [outs.index(o) for o in ORDER]]


def read_fasta(path: str) -> list[tuple[str, str]]:
    out, name, parts = [], None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(parts)))
            name, parts = line[1:].split()[0], []
        elif line:
            parts.append(line)
    if name is not None:
        out.append((name, "".join(parts)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", action="append", default=[], help="a 200-letter piece (can be given several times)")
    ap.add_argument("--fasta", help="a FASTA file of 200-letter pieces")
    ap.add_argument("--scan", action="store_true", help="score every single-letter change of each piece")
    ap.add_argument("--both", action="store_true", help="average the two released students (default: the main one, "
                    "which is the model behind the reported numbers)")
    ap.add_argument("--out", help="write a tab-separated table here (default: print)")
    args = ap.parse_args()

    named = [(f"seq{i + 1}", s) for i, s in enumerate(args.seq)] + (read_fasta(args.fasta) if args.fasta else [])
    if not named:
        ap.error("give --seq or --fasta")
    for name, s in named:
        if len(s) != 200 or set(s.upper()) - set("ACGT"):
            sys.exit(f"{name}: needs exactly 200 letters of A, C, G, T (got {len(s)}). The model was trained on 200-letter pieces.")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    students = load_students(args.both, device)
    cols = [NAMES[o] for o in ORDER]
    lines = []
    if not args.scan:
        pred = activity(np.stack([encode(s) for _, s in named]), students, device)
        lines.append("\t".join(["name"] + [f"activity_{c}" for c in cols]))
        lines += ["\t".join([name] + [f"{v:.4f}" for v in row]) for (name, _), row in zip(named, pred)]
    else:
        lines.append("\t".join(["name", "position", "original", "changed_to"] + [f"effect_{c}" for c in cols]))
        for name, s in named:
            base = encode(s)
            changed, labels = [], []
            for pos in range(200):
                for alt in range(4):
                    if alt != base[pos]:
                        c = base.copy()
                        c[pos] = alt
                        changed.append(c)
                        labels.append((pos + 1, "ACGT"[base[pos]], "ACGT"[alt]))
            pred = activity(np.stack([base] + changed), students, device)
            effect = pred[1:] - pred[0]
            lines += ["\t".join([name, str(p), a, b] + [f"{v:.4f}" for v in row]) for (p, a, b), row in zip(labels, effect)]
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {len(lines) - 1} rows to {args.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

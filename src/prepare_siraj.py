"""Convert the supplementary tables of Siraj and others (bioRxiv 2024, doi 10.1101/2024.05.05.592437; published as
"Functional dissection of complex trait variants at single-nucleotide resolution", Nature 2026) into parquet files.

Source: the preprint's supplementary archive from Europe PMC,
    https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11100724/supplementaryFiles
(media-3.xlsx = Table S2 oligo sequences, media-4.xlsx = Table S3 results for 221,412 variants and controls,
media-7.xlsx = Table S20 saturation mutagenesis, media-9.xlsx = small tables incl. S26 ENCODE accessions),
unpacked into data/siraj/raw/. The preprint is CC BY-NC; the same measurements are on ENCODE (accessions in S26).

This script only converts the tables. It compares nothing with any model.
    .venv/bin/python -I src/prepare_siraj.py
"""

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

RAW, OUT = Path("data/siraj/raw"), Path("data/siraj")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
TABLES = {"media-3.xlsx": "oligos", "media-4.xlsx": "variants", "media-7.xlsx": "satmut"}


def col_index(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_sheet(path: Path) -> pd.DataFrame:
    """First sheet of an .xlsx, streamed. Row 1 is the table's title, row 2 the header."""
    z = zipfile.ZipFile(path)
    strings = []
    with z.open("xl/sharedStrings.xml") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag == NS + "si":
                strings.append("".join(x.text or "" for x in el.iter(NS + "t")))
                el.clear()
    header, cols, n = None, None, 0
    with z.open("xl/worksheets/sheet1.xml") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag != NS + "row":
                continue
            n += 1
            cells = {}
            for c in el.findall(NS + "c"):
                v = c.find(NS + "v")
                if v is not None:
                    cells[col_index(c.get("r"))] = strings[int(v.text)] if c.get("t") == "s" else v.text
            el.clear()
            if n == 2:
                header = [cells.get(i, f"col{i}") for i in range(max(cells) + 1)]
                cols = [[] for _ in header]
            elif n > 2 and cells:
                for i in range(len(header)):
                    cols[i].append(cells.get(i))
    d = pd.DataFrame(dict(zip(header, cols)))
    for c in d.columns:  # numbers where the whole column is numeric (NA allowed)
        as_num = pd.to_numeric(d[c].replace("NA", None), errors="coerce")
        if as_num.notna().sum() == (d[c].notna() & (d[c] != "NA")).sum():
            d[c] = as_num
    return d


def main() -> None:
    for name, out in TABLES.items():
        d = read_sheet(RAW / name)
        d.to_parquet(OUT / f"{out}.parquet", index=False)
        print(f"{name} -> {out}.parquet: {len(d):,} rows, {len(d.columns)} columns", flush=True)


if __name__ == "__main__":
    main()

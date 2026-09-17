"""Recover readable text from the five McDonald's PDFs that pypdf extracts as mojibake.

The problem
-----------
Five of the thirteen annual reports embed subset fonts with no usable ToUnicode CMap.
`pypdf` (and PyMuPDF) return the raw glyph codes, which come out as control characters:

    *#\x1f)\x1b\x1a\x01()\x17)\x1b(  ...

It is not a single shift cipher, so it cannot be undone with one offset: each font
subset orders its glyphs arbitrarily. Some fonts in the same documents (the Speedee
family used for the shareholder-letter prose) are encoded normally and read fine.

What this does
--------------
Walks every text span with PyMuPDF, keeps the spans that are already readable English,
and discards the ones that are still glyph soup. The result is the narrative sections
of each report -- the CEO/shareholder letters -- while the 10-K financial tables set in
the broken Arial subsets are lost. Recovering those would need OCR (Tesseract), which is
not installed here.

This is lossy and the README says so. It is not a general PDF repair tool.

Run:  python scripts/recover_pdf_text.py
Out:  corpus_sets/mcdonalds_all/<name>.recovered.txt
"""
import re
from pathlib import Path

import pymupdf

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "corpus_sets" / "mcdonalds"
OUT = REPO / "corpus_sets" / "mcdonalds_all"

BROKEN = [
    "2023 Annual Report_vf.pdf",
    "MCD 2021 Annual Report.pdf",
    "MCD_2022_Annual_Report.pdf",
    "McD - 2024 Annual Report to Shareholders.pdf",
    "MCD 2025 Annual Report.pdf",
]

COMMON = set("""the and of to in for is on that a as with by are was were be from or an at it its
company financial year our we this operating results total net income restaurants business
customers growth market brand global system sales people team world new more than have has
will their they there been about into over all can what who when which""".split())

PRINTABLE = re.compile(r"[ -~‘’“”–—]+$")


def english_ratio(text):
    words = re.findall(r"[A-Za-z]{2,}", text)
    if not words:
        return 0.0
    return sum(1 for w in words if w.lower() in COMMON) / len(words)


def recover(path):
    doc = pymupdf.open(path)
    kept, dropped = [], 0
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"]
                    if len(text.strip()) < 2:
                        continue
                    readable = english_ratio(text) >= 0.15 or (
                        PRINTABLE.match(text) and english_ratio(text) > 0
                    )
                    if readable:
                        kept.append(text.strip())
                    else:
                        dropped += 1
    return " ".join(kept), len(kept), dropped


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in BROKEN:
        text, kept, dropped = recover(SRC / name)
        # normalise whitespace; the corpus loader splits into <=47-token passages itself
        text = re.sub(r"\s+", " ", text).strip()
        dest = OUT / (Path(name).stem + ".recovered.txt")
        dest.write_text(text + "\n", encoding="utf-8")
        words = len(text.split())
        total += words
        print(f"{name[:44]:44s} kept={kept:5d} dropped={dropped:5d} words={words:6d} "
              f"eng={english_ratio(text):.2f} -> {dest.name}")
    print(f"\nrecovered {total:,} words from {len(BROKEN)} broken PDFs")
    print("NOTE: financial tables in the broken Arial subsets are NOT recovered.")


if __name__ == "__main__":
    main()

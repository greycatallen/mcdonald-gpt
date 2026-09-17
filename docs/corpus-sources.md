# Corpus sources

## Experiment 2 — McDonald's annual reports and CEO letters

Supplied by the student from McDonald's public investor-relations materials: annual
reports (SEC Form 10-K) and CEO letters to shareholders, freely distributed via SEC EDGAR
and McDonald's investor-relations site.

**These source PDFs are published in this repository** at
[`corpus_sets/mcdonalds/`](../corpus_sets/mcdonalds/) (~45 MB), together with the full
extracted text at `evidence/experiment2-mcdonalds/corpus.txt`, so the run is reproducible
without hunting for the exact editions. SHA-256 for every file is listed below. Note this
is third-party copyrighted material redistributed as public company filings; it contains no
personal, confidential, or private data.

Ingest mode: `classroom` · max passage tokens: 47 · split unit: deduplicated passage, not source file

| File | Pages | Bytes | Characters | Passages | Unique | Warnings | SHA-256 |
|---|---:|---:|---:|---:|---:|---|---|
| `2016 Annual Report1.pdf` | 68 | 1,088,810 | 571,339 | 14,123 | 5,334 | none | `b325386d426dda79…` |
| `2019 Annual Report.pdf` | 79 | 3,744,516 | 299,302 | 7,043 | 4,096 | none | `2f5714c588d9ea11…` |
| `2020 Annual Report.pdf` | 98 | 2,606,175 | 375,078 | 8,243 | 5,156 | Page 98: no text extracted (blank or scanned); OCR may be needed. | `992c8c5ca2d5d860…` |
| `2023 Annual Report_vf.pdf` | 78 | 2,317,115 | 300,930 | 7,678 | 5,290 | none | `ad60a1c5fb79b62b…` |
| `2023 CEO Letter_vf.pdf` | 4 | 846,476 | 12,242 | 320 | 194 | none | `b82a68de989bb9c5…` |
| `MCD 2021 Annual Report.pdf` | 76 | 3,888,012 | 292,389 | 6,498 | 5,819 | none | `8858a34fc3a83c8c…` |
| `MCD 2025 Annual Report.pdf` | 86 | 7,570,722 | 299,631 | 6,161 | 5,700 | Page 2: no text extracted (blank or scanned); OCR may be needed., Page 85: no text extracted (blank or scanned); OCR may be needed. | `34e1d8fde35efb48…` |
| `MCD 2025 CEO Letter.pdf` | 6 | 2,985,474 | 11,373 | 321 | 315 | none | `deb3e53d136d02bc…` |
| `MCD_2022_Annual_Report.pdf` | 73 | 7,714,231 | 288,626 | 6,329 | 5,724 | none | `4c8e30e749ad6363…` |
| `McD - 2024 Annual Report to Shareholders.pdf` | 82 | 9,381,444 | 305,101 | 6,353 | 5,752 | none | `a2bb6771da2338aa…` |
| `McD - 2024 CEO Letter.pdf` | 4 | 2,909,825 | 9,069 | 236 | 233 | none | `8395b35323aaf083…` |
| `McDonald's 2017 Annual Report1.pdf` | 72 | 1,018,607 | 573,875 | 14,321 | 5,401 | none | `f93d65b516c054b0…` |
| `McDonalds_2018_Annual_Report_unlocked.pdf` | 94 | 882,503 | 345,397 | 4,906 | 4,261 | Page 25: no text extracted (blank or scanned); OCR may be needed. | `1c1ae2ca3d178586…` |
| **Total (13 files)** | | | **3,684,352** | **82,532** | **53,275** | | |

After de-duplication across all sources: **40,834 unique passages** (36,242 new beyond the classroom base of 6,200; 47,898 duplicates removed).

### Extraction check

All 13 PDFs contain selectable text, so no OCR was required for the bulk of the content.
Extraction was verified by reading the per-file previews in `corpus_manifest.json` and
confirming each begins with the expected SEC Form 10-K cover page rather than garbled or
out-of-order text.

**Four page-level warnings were raised across three files**, and they are resolved as
follows rather than ignored:

| File | Warning | Resolution |
|---|---|---|
| `2020 Annual Report.pdf` | Page 98: no text extracted | Verified: **0 characters but 12 embedded images** — an image-only back cover, not a blank page. Any text inside those graphics would need OCR to recover. Accepted as a bounded gap: one page of cover art out of ~800 ingested pages. |
| `MCD 2025 Annual Report.pdf` | Page 2: no text extracted | Verified: 0 characters, 0 images — genuinely blank page facing the cover. Accepted. |
| `MCD 2025 Annual Report.pdf` | Page 85: no text extracted | Verified: 0 characters, 0 images — genuinely blank, second-to-last page. Accepted. |
| `McDonalds_2018_Annual_Report_unlocked.pdf` | Page 25: no text extracted | Verified directly: the page contains **0 characters and 0 embedded images** — a genuinely blank separator page between the revenues table (p. 24) and the following section. Nothing to recover. Accepted. |

All four pages were opened individually with `pypdf` and their character and image counts
recorded. **Three are genuinely blank** (0 characters, 0 images). **One — page 98 of the 2020
report — is image-only** (0 characters, 12 images): a graphical back cover whose text, if any,
sits inside the images and would require OCR. That is the single acknowledged gap, and it is
one page of cover art out of roughly 800 ingested pages, immaterial to a word-frequency
vocabulary. No file was dropped, and `ignored` in the manifest is empty.

*Two corrections were made to this section rather than left standing: an initial draft claimed
no warnings were raised at all, and a second draft described page 25 of the 2018 report as a
financial chart needing OCR. Both were assumptions. The counts above are measured.*

### The one file that needed intervention

`McDonalds_2018_Annual_Report.pdf` carries an owner-password flag. The notebook refuses
encrypted files outright:

```
Could not import McDonalds_2018_Annual_Report.pdf: encrypted PDF;
export an unlocked copy you are allowed to use
```

The file has **no user password** — `pypdf` opens it with an empty string, so nothing was
bypassed or guessed; the flag only restricts printing/editing in viewers. Following the
error message's own instruction, the pages were re-saved to
`McDonalds_2018_Annual_Report_unlocked.pdf` and the original was removed from the corpus
folder. Verified afterward: `is_encrypted: False`, 94 pages, 3,696 characters on page 1 —
identical to the original's extraction.

## Experiment 3 — targeted teaching material

Synthetic text written for this assignment by
[`scripts/build_targeted_corpus.py`](../scripts/build_targeted_corpus.py) and committed in
full at [`corpus_sets/targeted/`](../corpus_sets/targeted/). No third-party material, no
personal or confidential data, so it is published in full and is safe to share.

## Experiment 1 — starter corpus

The notebook's own generated classroom sentences. No external files (`0 ingestible file(s)`).

# Corpus sources

What each experiment trained on, where it came from, and how the extracted text was checked.
The analysis of these results is in the [README](../README.md).

## Experiment 1 — classroom sentences only

No files. The notebook generates 6,360 sentences in `classroom_corpus()` ([`custom_llm.py`](../custom_llm.py),
line 228); 160 are withheld because they contain an eval prompt, and after removing duplicates
4,592 unique passages remain. Every experiment includes these sentences.

## Experiment 2 — McDonald's annual reports and CEO letters

McDonald's own public investor-relations documents: 10 annual reports (SEC Form 10-K, 2016–2025)
and 3 CEO letters to shareholders (2023–2025), about 820 pages in total, available from SEC EDGAR
and the company's investor-relations site. They contain no personal or confidential data. They
are third-party copyrighted documents, republished here so the run can be reproduced.

- Original PDFs: [`corpus_sets/mcdonalds/`](../corpus_sets/mcdonalds/)
- What Experiment 2 actually trained on: [`corpus_sets/mcdonalds_all/`](../corpus_sets/mcdonalds_all/)
  — 8 PDFs used as-is, plus 5 text files recovered from the reports that do not extract cleanly

### Files used in Experiment 2

From [`evidence/experiment2-mcdonalds/corpus_manifest.json`](../evidence/experiment2-mcdonalds/corpus_manifest.json).
Passages are chunks of at most 47 tokens; *unique* counts them after removing repeats within the file.

| File | Pages | Characters | Passages | Unique | SHA-256 |
|---|---:|---:|---:|---:|---|
| `2016 Annual Report1.pdf` | 68 | 571,339 | 14,123 | 5,334 | `b325386d426dda79…` |
| `2019 Annual Report.pdf` | 79 | 299,302 | 7,043 | 4,096 | `2f5714c588d9ea11…` |
| `2020 Annual Report.pdf` | 98 | 375,078 | 8,243 | 5,156 | `992c8c5ca2d5d860…` |
| `2023 Annual Report_vf.recovered.txt` | — | 109,066 | 672 | 656 | `da4f93d62672cf6d…` |
| `2023 CEO Letter_vf.pdf` | 4 | 12,242 | 320 | 194 | `b82a68de989bb9c5…` |
| `MCD 2021 Annual Report.recovered.txt` | — | 10,812 | 81 | 81 | `9acf5feea4a44fb6…` |
| `MCD 2025 Annual Report.recovered.txt` | — | 11,169 | 80 | 80 | `be0330a694a1ff5b…` |
| `MCD 2025 CEO Letter.pdf` | 6 | 11,373 | 321 | 315 | `deb3e53d136d02bc…` |
| `MCD_2022_Annual_Report.recovered.txt` | — | 8,843 | 58 | 58 | `8fb14e5cd38a0c2f…` |
| `McD - 2024 Annual Report to Shareholders.recovered.txt` | — | 9,309 | 70 | 70 | `d593912f63144291…` |
| `McD - 2024 CEO Letter.pdf` | 4 | 9,069 | 236 | 233 | `8395b35323aaf083…` |
| `McDonald's 2017 Annual Report1.pdf` | 72 | 573,875 | 14,321 | 5,401 | `f93d65b516c054b0…` |
| `McDonalds_2018_Annual_Report_unlocked.pdf` | 94 | 345,397 | 4,906 | 4,261 | `1c1ae2ca3d178586…` |
| **Total (13 files)** | | **2,346,874** | **50,474** | **25,935** | |

Across all files, and together with the classroom sentences, this gives **24,199 unique passages**
(19,607 new beyond the classroom base; 32,475 repeats removed). Much of
the repetition is boilerplate that appears in every annual report.

### Checking the extraction

**Five reports do not extract as text.** The 2021, 2022, 2023, 2024 and 2025 annual reports set
their financial sections in fonts with no mapping back to letters. `pypdf` returns control
characters, PyMuPDF returns the same, and `pdfplumber` returns raw glyph codes such as
`(cid:43)(cid:36)`. The notebook gives no warning, and the manifest previews look fine because
each report's first page extracts cleanly — the problem only shows when you read further into
the extracted text.

| Original PDF | Pages | Garbled pages | SHA-256 of original |
|---|---:|---:|---|
| `2023 Annual Report_vf.pdf` | 78 | 51 | `ad60a1c5fb79b62b…` |
| `MCD 2021 Annual Report.pdf` | 76 | 69 | `8858a34fc3a83c8c…` |
| `MCD_2022_Annual_Report.pdf` | 73 | 66 | `4c8e30e749ad6363…` |
| `McD - 2024 Annual Report to Shareholders.pdf` | 82 | 73 | `a2bb6771da2338aa…` |
| `MCD 2025 Annual Report.pdf` | 86 | 73 | `34e1d8fde35efb48…` |
| **Total** | | **332** | |

The narrative sections of these reports (the shareholder letters) use a normally encoded font.
[`scripts/recover_pdf_text.py`](../scripts/recover_pdf_text.py) keeps every readable span and
drops the rest, recovering **22,158 words**; the financial tables in these five reports are lost.
Recovering them would need OCR. The effect of this problem on the model is analysed in README §9.

**One report was encrypted.** `McDonalds_2018_Annual_Report.pdf` has an owner-password flag, and
the notebook rejects encrypted files. It has no user password — `pypdf` opens it with an empty
string, so nothing was bypassed. Following the notebook's error message ("export an unlocked copy
you are allowed to use"), its pages were re-saved as `McDonalds_2018_Annual_Report_unlocked.pdf`
and checked: 94 pages, same text on page 1.

**Four pages produced no text.** Each was opened and its characters and images counted:

| File | Page | Finding |
|---|---:|---|
| `2020 Annual Report.pdf` | 98 | 0 characters, 12 images — graphical back cover |
| `MCD 2025 Annual Report.pdf` | 2 | 0 characters, 0 images — blank |
| `MCD 2025 Annual Report.pdf` | 85 | 0 characters, 0 images — blank |
| `McDonalds_2018_Annual_Report_unlocked.pdf` | 25 | 0 characters, 0 images — blank |

Nothing of value is lost on these pages. (The 2025 report is used through its recovered text
file, so only the 2020 and 2018 warnings appear in Experiment 2's manifest.)

### Reproducing the Experiment 2 corpus

```bash
./.venv/bin/python scripts/recover_pdf_text.py   # writes the 5 .recovered.txt files
./.venv/bin/python scripts/check_leakage.py corpus_sets/mcdonalds_all
```

The 8 cleanly extracting PDFs are already in `corpus_sets/mcdonalds_all/`.
`corpus_sets/mcdonalds_clean/` holds just those 8 and was used only for the diagnostic run that
isolated the extraction problem.

## Experiment 3 — targeted teaching material

Synthetic sentences written for this assignment by
[`scripts/build_targeted_corpus.py`](../scripts/build_targeted_corpus.py) and published in full
at [`corpus_sets/targeted/`](../corpus_sets/targeted/): 2,442 sentences in five files. No
third-party, personal or confidential material.

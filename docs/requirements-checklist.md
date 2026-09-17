# Requirements traceability

Every requirement from the assignment brief, mapped to where it is satisfied and its current
state. Source: [`assignment-brief.txt`](assignment-brief.txt) (§ "What You Are Submitting",
"README Requirements", "Evidence Required in the README", "Submission Checklist",
"Definition of Done").

Legend: ✅ done · 🟡 partly done · ⬜ not started

**Outstanding before submission:** section E (conceptual write-ups, must be in the student's
own words), README §12 (drafted, needs review), and F3/F4/F7 (verify rendering signed out,
then submit the URL).

---

## A. Deliverable quality (4 points)

| # | Requirement | Where | State |
|---|---|---|---|
| A1 | Executed `custom_llm.ipynb` for the **starter** experiment, outputs not cleared | `notebooks/experiment1-starter.executed.ipynb` | ✅ `notebooks/experiment1-starter.executed.ipynb` |
| A2 | Executed `custom_llm.ipynb` for the **extension** experiment, outputs not cleared | `notebooks/experiment2-extended.executed.ipynb` | ✅ Exp 2, 3 and 3b notebooks |
| A3 | Readable source code in repo | `custom_llm.py`, `run_evals.py`, `chat.py`, `nanogpt_model.py`, `scripts/` | ✅ |
| A4 | Corpus sources, permissions, and choices explained | README §5 | ✅ README §5 + `docs/corpus-sources.md` |
| A5 | Clear README as grading entry point | `README.md` | ✅ results filled in |
| A6 | Three choices + reasons (corpus, steps, learning rate) | README §3 | ✅ README §3 |
| A7 | Why ≥2 extension categories were chosen and how the new material addresses their gaps | README §9 | ✅ README §9 — 4 taught + 4 control |
| A8 | Prediction written **before** training | README §4 | ✅ committed `16faf8e` pre-training |
| A9 | Learning process explained with actual token/embedding/gradient/loss evidence | README §7, §11 | 🟡 evidence in §7; §11 prose outstanding |

## B. Testing & evaluation (3 points)

| # | Requirement | Where | State |
|---|---|---|---|
| B1 | All 48 cases, unchanged, run **before** training — starter | `evidence/experiment1-starter/language_evals/untrained/` | ✅ |
| B2 | All 48 cases, unchanged, run **after** training — starter | `evidence/experiment1-starter/language_evals/final/` | ✅ |
| B3 | All 48 cases, unchanged, run **before** training — extended | `evidence/experiment2-extended/language_evals/untrained/` | ✅ (Exp 2, 3 and 3b) |
| B4 | All 48 cases, unchanged, run **after** training — extended | `evidence/experiment2-extended/language_evals/final/` | ✅ (Exp 2, 3 and 3b) |
| B5 | Suite provably unmodified | SHA-256 `e8affcd7…3e17f7`, verified in README §2 | ✅ |
| B6 | Four-row comparison table in README | README §8 | ✅ 8 rows (4 experiments × 2 stages) |
| B7 | All-case success **and** scorable accuracy **and** coverage reported | README §8 | ✅ README §8 |
| B8 | Group/category breakdowns | README §8 | ✅ README §8, §9 |
| B9 | Actual free continuations shown, distinct from MC score | README §8 | ✅ README §8 table |
| B10 | Every case saved (CSV + JSON + summary) | `evidence/*/language_evals/` | ✅ `evidence/*/language_evals/` |
| B11 | Separation checks saved | `evidence/*/eval_separation.json` | ✅ all 4 runs |
| B12 | Limits of exact-match leakage checking stated | README §8 | ✅ written |
| B13 | Described as a public development benchmark, not an unseen final test | README §8 | ✅ written |
| B14 | Failures explained, not hidden; changes attributed to coverage vs. learned patterns | README §8, §9 | ✅ README §9 coverage-vs-pattern table |

## C. Working result (3 points)

| # | Requirement | Where | State |
|---|---|---|---|
| C1 | Trained nanoGPT demonstrated | `evidence/experiment*/model.pt` | ✅ `model.pt` committed per experiment |
| C2 | Evals re-runnable against the saved model | README §13 commands | ✅ documented |
| C3 | Working chat interface producing real replies from the trained model | `chat.py` + notebook §10 | ✅ 6 real turns |
| C4 | Launch instructions, tested | README §10 | ✅ tested |
| C5 | Model / run identity recorded | README §10 | ✅ run + SHA-256 in §10 |
| C6 | ≥3 real chat interactions | `evidence/*/chat_transcript.json` | ✅ 6 interactions |
| C7 | Screenshot or recording of the interface | `evidence/chat-screenshot.*` | ✅ typescript recording + SVG |

## D. Specific evidence artifacts

| # | Artifact | State |
|---|---|---|
| D1 | Untrained / halfway / final samples, same generation settings, full files linked | ✅ |
| D2 | ≥1 visible change (or lack of change) between samples, explained | ✅ §7 |
| D3 | `training_curves.svg` embedded in README | ✅ §7 |
| D4 | Full loss table from `history.json` | ✅ §7 |
| D5 | Stated: fixed panels, ≤20 docs each | ✅ README §7 |
| D6 | `tokenization.json` + `inspection.json` linked | ✅ |
| D7 | One word → ID → 64-number vector, before **and** after | ✅ |
| D8 | First parameter's value / gradient / post-update value | ✅ |
| D9 | Next-token probability comparison, same prefix | ✅ |
| D10 | `config.json`, `training.csv`, `training_summary.json` linked | ✅ |
| D11 | Three-temperature comparison + `temperature_comparison.json` | ✅ §7 |
| D12 | Stated: temperature changes sampling only, no weight updates | ✅ README §7 |
| D13 | `corpus_manifest.json` + `vocabulary_report.json` linked | ✅ |
| D14 | Unique passages, vocab size, both unknown-token rates, split sizes | ✅ §5 |
| D15 | Completed steps, elapsed time, hardware, parameter count | ✅ §6 |
| D16 | Interruptions/failures identified | ✅ none so far; smoke test documented |

## E. Conceptual explanations — **must be written by the student, in their own words**

| # | Question | State |
|---|---|---|
| E1 | Corpus: what it teaches, what's missing, why hold data out | ⬜ |
| E2 | Token vs. token ID vs. vector vs. embedding | ⬜ |
| E3 | What makes this a neural network; loss → gradients → optimizer → weights | ⬜ |
| E4 | What attention combines; why it cannot see future tokens | ⬜ |
| E5 | Probabilities → generated text; temperature; whether weights changed | ⬜ |
| E6 | Did samples + both loss curves support the prediction; honest conclusion | ⬜ |
| E7 | One observed limitation + one proposed next experiment with predicted effect | ⬜ |

## F. Submission mechanics

| # | Requirement | State |
|---|---|---|
| F1 | Public GitHub repository | ✅ https://github.com/greycatallen/mcdonald-gpt |
| F2 | Notebook outputs **not** cleared | ✅ outputs preserved |
| F3 | Notebook renders on GitHub with plot/samples/losses visible | ⬜ verify before submit |
| F4 | Repository opens correctly **signed out** | ⬜ verify before submit |
| F5 | Both results ZIPs kept | ⬜ |
| F6 | No eval prompts / answer keys / eval outputs in training corpus | ✅ guard fired once, fixed; `check_leakage.py` CLEAN on all sets |
| F7 | Submit repo URL through course portal | ⬜ student action |

---

## Hard constraints

- **Do not** copy eval prompts, answer choices, the answer key, scoring rules, eval outputs,
  or chat logs into training data. The brief states this may incur a penalty beyond the
  3-point evaluation category.
- **Do not** substitute another model or an API. Replies must come from the nanoGPT trained
  in this repo.
- **Do not** invent, estimate, or borrow results from the upstream reference run.
- Training material must live only in `corpus/`; the exam lives only in `evals/`.
- **Do not** place any `.md`/`.txt`/`.pdf` documentation inside `corpus/` subfolders.
  `custom_llm.py:175` exempts only the top-level `corpus/README.md`; anything else under
  `corpus/` becomes training text. Notes about the corpus belong in `docs/`.

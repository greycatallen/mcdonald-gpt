# Requirements traceability

Every requirement from the assignment brief, mapped to where it is satisfied and its current
state. Source: [`assignment-brief.txt`](assignment-brief.txt) (§ "What You Are Submitting",
"README Requirements", "Evidence Required in the README", "Submission Checklist",
"Definition of Done").

Legend: ✅ done · 🟡 partly done · ⬜ not started

---

## A. Deliverable quality (4 points)

| # | Requirement | Where | State |
|---|---|---|---|
| A1 | Executed `custom_llm.ipynb` for the **starter** experiment, outputs not cleared | `notebooks/experiment1-starter.executed.ipynb` | ⬜ |
| A2 | Executed `custom_llm.ipynb` for the **extension** experiment, outputs not cleared | `notebooks/experiment2-extended.executed.ipynb` | ⬜ |
| A3 | Readable source code in repo | `custom_llm.py`, `run_evals.py`, `chat.py`, `nanogpt_model.py`, `scripts/` | ✅ |
| A4 | Corpus sources, permissions, and choices explained | README §5 | ⬜ |
| A5 | Clear README as grading entry point | `README.md` | 🟡 skeleton written, results pending |
| A6 | Three choices + reasons (corpus, steps, learning rate) | README §3 | 🟡 LR rationale written; values pending |
| A7 | Why ≥2 extension categories were chosen and how the new material addresses their gaps | README §9 | ⬜ |
| A8 | Prediction written **before** training | README §4 | ⬜ **blocking Experiment 1** |
| A9 | Learning process explained with actual token/embedding/gradient/loss evidence | README §7, §11 | ⬜ |

## B. Testing & evaluation (3 points)

| # | Requirement | Where | State |
|---|---|---|---|
| B1 | All 48 cases, unchanged, run **before** training — starter | `evidence/experiment1-starter/language_evals/untrained/` | ⬜ |
| B2 | All 48 cases, unchanged, run **after** training — starter | `evidence/experiment1-starter/language_evals/final/` | ⬜ |
| B3 | All 48 cases, unchanged, run **before** training — extended | `evidence/experiment2-extended/language_evals/untrained/` | ⬜ |
| B4 | All 48 cases, unchanged, run **after** training — extended | `evidence/experiment2-extended/language_evals/final/` | ⬜ |
| B5 | Suite provably unmodified | SHA-256 `e8affcd7…3e17f7`, verified in README §2 | ✅ |
| B6 | Four-row comparison table in README | README §8 | ⬜ |
| B7 | All-case success **and** scorable accuracy **and** coverage reported | README §8 | ⬜ |
| B8 | Group/category breakdowns | README §8 | ⬜ |
| B9 | Actual free continuations shown, distinct from MC score | README §8 | ⬜ |
| B10 | Every case saved (CSV + JSON + summary) | `evidence/*/language_evals/` | ⬜ |
| B11 | Separation checks saved | `evidence/*/eval_separation.json` | ⬜ |
| B12 | Limits of exact-match leakage checking stated | README §8 | ✅ written |
| B13 | Described as a public development benchmark, not an unseen final test | README §8 | ✅ written |
| B14 | Failures explained, not hidden; changes attributed to coverage vs. learned patterns | README §8, §9 | ⬜ |

## C. Working result (3 points)

| # | Requirement | Where | State |
|---|---|---|---|
| C1 | Trained nanoGPT demonstrated | `evidence/experiment*/model.pt` | ⬜ |
| C2 | Evals re-runnable against the saved model | README §13 commands | ✅ documented |
| C3 | Working chat interface producing real replies from the trained model | `chat.py` + notebook §10 | ⬜ |
| C4 | Launch instructions, tested | README §10 | 🟡 written, untested against final model |
| C5 | Model / run identity recorded | README §10 | ⬜ |
| C6 | ≥3 real chat interactions | `evidence/*/chat_transcript.json` | ⬜ |
| C7 | Screenshot or recording of the interface | `evidence/chat-screenshot.*` | ⬜ |

## D. Specific evidence artifacts

| # | Artifact | State |
|---|---|---|
| D1 | Untrained / halfway / final samples, same generation settings, full files linked | ⬜ |
| D2 | ≥1 visible change (or lack of change) between samples, explained | ⬜ |
| D3 | `training_curves.svg` embedded in README | ⬜ |
| D4 | Full loss table from `history.json` | ⬜ |
| D5 | Stated: fixed panels, ≤20 docs each | ✅ README §7 |
| D6 | `tokenization.json` + `inspection.json` linked | ⬜ |
| D7 | One word → ID → 64-number vector, before **and** after | ⬜ |
| D8 | First parameter's value / gradient / post-update value | ⬜ |
| D9 | Next-token probability comparison, same prefix | ⬜ |
| D10 | `config.json`, `training.csv`, `training_summary.json` linked | ⬜ |
| D11 | Three-temperature comparison + `temperature_comparison.json` | ⬜ |
| D12 | Stated: temperature changes sampling only, no weight updates | ✅ README §7 |
| D13 | `corpus_manifest.json` + `vocabulary_report.json` linked | ⬜ |
| D14 | Unique passages, vocab size, both unknown-token rates, split sizes | ⬜ |
| D15 | Completed steps, elapsed time, hardware, parameter count | ⬜ |
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
| F2 | Notebook outputs **not** cleared | ⬜ verify before submit |
| F3 | Notebook renders on GitHub with plot/samples/losses visible | ⬜ verify before submit |
| F4 | Repository opens correctly **signed out** | ⬜ verify before submit |
| F5 | Both results ZIPs kept | ⬜ |
| F6 | No eval prompts / answer keys / eval outputs in training corpus | ⬜ verify via `eval_separation.json` + manual read |
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
